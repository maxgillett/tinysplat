import os, logging
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
from torch import nn, Tensor
from torch import dtype as dtypes
from torch import device
from torch.nn.parameter import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import SSIM
from sklearn.neighbors import NearestNeighbors
from gsplat.sh import num_sh_bases, deg_from_sh
from plyfile import PlyData, PlyElement
from pytorch3d.ops import knn_points, ball_query

from ..scene import PointCloud
from ..utils import random_quat_tensor, quat_to_rot_tensor, RGB2SH, SH2RGB

class GaussianModel(nn.Module):
    def __init__(self,
        num_points: Optional[int] = None,
        device = device("cuda:0"),
        **kwargs
    ):
        super(GaussianModel, self).__init__()
        self.device = device

        if not kwargs.get('train', False): return
        assert num_points is not None, "Number of points must be specified for training"

        # Configuration
        self.max_sh_degree = kwargs['sh_degree']
        self.active_sh_degree = 1
        self.epsilon_alpha = kwargs['epsilon_alpha']
        self.tau_means = kwargs['tau_means']
        self.phi = kwargs['phi']

        # Learning rates
        self.lr_means = kwargs['lr_means']
        self.lr_colors_dc = kwargs['lr_colors_dc']
        self.lr_colors_rest = kwargs['lr_colors_rest']
        self.lr_scales = kwargs['lr_scales']
        self.lr_quats = kwargs['lr_quats']
        self.lr_opacities = kwargs['lr_opacities']

        # Densification parameters
        self.warmup_densify = kwargs['warmup_densify']
        self.warmup_grad = kwargs['warmup_grad']
        self.interval_densify = kwargs['interval_densify']
        self.densify_end = kwargs['densify_end']
        self.interval_opacity_reset = kwargs['interval_opacity_reset']
        self.densify_scale_thresh = kwargs['densify_scale_thresh']

        # Quality metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

        self.background = torch.zeros(3, device=device)

        # Accumulate the 'means' gradients for use in cloning/splitting/pruning
        self.means_grad_accum = torch.zeros(num_points, device=device).float()

    @classmethod
    def from_pcd(cls, pcd: PointCloud, **kwargs):
        num_points = pcd.xyz.shape[0]
        model = cls(num_points, **kwargs)

        # Initialize colors (using spherical harmonics)
        dim_sh = num_sh_bases(kwargs['sh_degree'])
        colors = torch.zeros((num_points, dim_sh, 3), dtype=torch.float32)
        colors[:, 0, :] = RGB2SH(pcd.colors / 255)

        # Find the average of the three nearest neighbors for each point and 
        # use that as the scale
        nn = NearestNeighbors(n_neighbors=4, algorithm='auto', metric='euclidean')
        nn = nn.fit(pcd.xyz.cpu().numpy())
        distances, indices = nn.kneighbors(pcd.xyz.cpu().numpy())
        log_mean_dist = np.log(np.mean(distances[:, 1:], axis=1).astype(np.float32))
        scales = np.repeat(log_mean_dist[:, np.newaxis], 3, axis=1)

        # Differentiable model parameters
        model.means = Parameter(pcd.xyz.float())
        model.colors_dc = Parameter(colors[:, 0, :])
        model.colors_rest = Parameter(colors[:, 1:, :])
        model.scales = Parameter(torch.as_tensor(scales, device=model.device))
        model.quats = Parameter(random_quat_tensor(num_points))
        model.opacities = Parameter(torch.logit(0.1 * torch.ones(num_points, 1, device=model.device)))
        return model

    @classmethod
    def from_state_checkpoint(cls, state_dict, **kwargs):
        num_points = state_dict['means'].shape[0]
        dim_sh = state_dict['colors_rest'].shape[1]
        model = cls(num_points, **kwargs)
        device = model.device

        model.means = Parameter(torch.zeros(num_points, 3, device=device))
        model.colors_dc = Parameter(torch.zeros(num_points, 3, device=device))
        model.colors_rest = Parameter(torch.zeros(num_points, dim_sh, 3, device=device))
        model.scales = Parameter(torch.zeros(num_points, 3, device=device))
        model.quats = Parameter(torch.zeros(num_points, 4, device=device))
        model.opacities = Parameter(torch.zeros(num_points, 1, device=device))

        model.max_sh_degree = deg_from_sh(dim_sh + 1)
        model.active_sh_degree = model.max_sh_degree

        model.load_state_dict(state_dict)
        return model

    def parameters(self):
        return [
            {'params': [self.means], 'lr': self.lr_means, "name": "means"},
            {'params': [self.colors_dc], 'lr': self.lr_colors_dc, "name": "colors_dc"},
            {'params': [self.colors_rest], 'lr': self.lr_colors_rest, "name": "colors_rest"},
            {'params': [self.scales], 'lr': self.lr_scales, "name": "scales"},
            {'params': [self.quats], 'lr': self.lr_quats, "name": "quats"},
            {'params': [self.opacities], 'lr': self.lr_opacities, "name": "opacities"}
        ]

    def update_learning_rate(self, step, optim):
        # TODO
        pass

    def increment_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def update_grad_accum(self, step, extras):
        if step < self.warmup_grad: return
        self.means_grad_accum += extras['xys'].grad.norm(dim=-1)

    def reset_opacities(self, step):
        if step % self.interval_opacity_reset != 0: return
        self.opacities.data[:] = self.epsilon_alpha / 2

    def densify_and_prune(self, step, optim, extras):
        if step < self.warmup_densify or step % self.interval_densify != 0:
            return

        if step > self.densify_end:
            return

        # FIXME: Temporarily limit number of Gaussians
        if self.means.shape[0] > 1000000:
            return

        width = extras['camera']['width']
        height = extras['camera']['height']
        grad_norm_avg = self.means_grad_accum / self.interval_densify / 2 * max(width, height)
        grad_mask = grad_norm_avg >= self.tau_means

        # Clone Gaussians (TODO: move them in the direction of the gradient)
        mask = self.scales.exp().max(dim=-1).values < self.densify_scale_thresh
        mask &= grad_mask
        cloned_means = self.means[mask]
        cloned_means = cloned_means
        cloned_colors_dc = self.colors_dc[mask]
        cloned_colors_rest = self.colors_rest[mask]
        cloned_scales = self.scales[mask]
        cloned_quats = self.quats[mask]
        cloned_opacities = self.opacities[mask]
        logging.debug("Cloned {} Gaussians".format(mask.sum()))

        # Split Gaussians by sampling from a distribution
        split_mask = self.scales.exp().max(dim=-1).values > self.densify_scale_thresh
        split_mask &= grad_mask
        dist = GaussianDistribution(self, split_mask)
        samples = dist.sample(n_samples=2)
        split_means = samples['means']
        split_colors_dc = samples['colors_dc']
        split_colors_rest = samples['colors_rest']
        split_scales = samples['scales']
        split_quats = samples['quats']
        split_opacities = samples['opacities']
        logging.debug("Split {} Gaussians".format(split_mask.sum()))

        # Prune low opacity and large scale Gaussians marked for splitting
        # (we also prune the original "split" Gaussians, as we sampled twice above)
        prune_mask = (torch.sigmoid(self.opacities) < 0.1).squeeze()
        prune_mask &= torch.exp(self.scales).max(dim=-1).values > 0.5
        prune_mask |= split_mask
        logging.debug("Pruned {} Gaussians".format(prune_mask.sum()))

        # Concatenate all newly created tensors
        tensors = dict()
        tensors['means'] = torch.cat((cloned_means, split_means))
        tensors['colors_dc'] = torch.cat((cloned_colors_dc, split_colors_dc))
        tensors['colors_rest'] = torch.cat((cloned_colors_rest, split_colors_rest))
        tensors['scales'] = torch.cat((cloned_scales, split_scales))
        tensors['quats'] = torch.cat((cloned_quats, split_quats))
        tensors['opacities'] = torch.cat((cloned_opacities, split_opacities))
        self.update_state(optim, prune_mask, tensors)

        # Reset accumulated gradients
        self.means_grad_accum = torch.zeros(self.means.shape[0], device=self.device)

    def update_state(self, optim, mask, _tensors=dict()):
        # If no tensors are provided (only a mask), we need to construct empty tensors
        # of the appropriate shape and data type
        def empty_tensor(param_name):
            dtype = getattr(self, param_name).dtype
            shape = getattr(self, param_name).shape
            return (param_name, torch.empty((0, *shape[1:]), dtype=dtype).to(self.device))
        tensors = dict([empty_tensor(name) for name in 
            ['means', 'colors_dc', 'colors_rest', 'scales', 'quats', 'opacities']])
        tensors.update(_tensors)

        # Replace optimizer state and tensor parameters.
        # We do this instead of creating a new optimizer because we want to
        # retain momentum information.
        params = {
            "means": (self.means[~mask, ...], tensors['means']),
            "colors_dc": (self.colors_dc[~mask, ...], tensors['colors_dc']),
            "colors_rest": (self.colors_rest[~mask, ...], tensors['colors_rest']),
            "scales": (self.scales[~mask, ...], tensors['scales']),
            "quats": (self.quats[~mask, ...], tensors['quats']),
            "opacities": (self.opacities[~mask, ...], tensors['opacities']),
        }
        for group in optim.param_groups:
            name = group["name"]
            param = torch.cat(params[name])
            new_param = params[name][1]

            # Mask internal state variables
            state = optim.state[group["params"][0]]
            state["exp_avg"] = state["exp_avg"][~mask, ...]
            state["exp_avg_sq"] = state["exp_avg_sq"][~mask, ...]

            # Concatenate interval state variables
            new_exp_avg = torch.zeros_like(new_param)
            new_exp_avg_sq = torch.zeros_like(new_param)
            state["exp_avg"] = torch.cat((state["exp_avg"], new_exp_avg))
            state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], new_exp_avg_sq))

            # Replace parameter in optimizer state and model 
            del optim.state[group["params"][0]]
            group["params"][0] = Parameter(param)
            optim.state[group["params"][0]] = state
            setattr(self, name, group["params"][0])
        self.means_grad_accum = self.means_grad_accum[~mask]

    ###-------- Surface regularization ---------------------------------------
    ### - Implementation of eqs. (1) and (5) from https://arxiv.org/abs/2311.12775

    def covariance(self):
        R = quat_to_rot_tensor(self.quats)
        S_2 = torch.exp(self.scales).pow(2).unsqueeze(2)
        sigma = R @ (R.transpose(-2, -1) * S_2)
        try:
            sigma_inv = sigma.inverse()
        except:
            sigma_inv = sigma.pinverse()
        return sigma_inv

    def density_function(self, points, update_neighbors=True):
        # Get the closest Gaussians to each point
        if update_neighbors or not hasattr(self, 'knn_idxs'):
            self.knn_idxs = knn_points(points[None], self.means[None], K=16).idx[0]
        neighbor_idxs = self.knn_idxs

        # Compute the density function d as a sum of the opacity-weighted sampled point values
        mu_g = self.means[neighbor_idxs]
        mu = points[:, None] - mu_g
        mu = mu[:,:,None,:]
        sigma_inv = self.covariance()[neighbor_idxs]
        out = torch.matmul(mu, sigma_inv)
        out = (out * mu).sum(-1)
        out = out.clamp(min=0, max=1e8)
        d = torch.exp(-0.5 * out).squeeze()
        d = torch.sum(d * torch.sigmoid(self.opacities[neighbor_idxs].squeeze()), dim=-1)
        d[d > 1] = 1 + 1e-12
        return d, neighbor_idxs

    def approximate_density_function(self, points, depth, camera, beta, znear=0.001, return_sdf=False):
        # Transform points to camera space
        view_mat = camera.view_matrix.to(self.device)
        ones = torch.ones(points.shape[0], 1).to(self.device)
        points = torch.cat((points, ones), dim=1)
        points = torch.matmul(view_mat, points.t()).t()
        z = points[:, 2]
        mask = z > znear

        # Project points to image space
        depth = depth.unsqueeze(0).unsqueeze(0)
        proj_mat = camera.proj_matrix.to(self.device)
        points_proj = torch.matmul(proj_mat, points.t()).t()[:,:2]
        points_proj[..., 0] = -camera.width * points_proj[..., 0]
        points_proj[..., 1] = -camera.height * points_proj[..., 1]

        # Mask the visible region
        x = points_proj[:, 0]
        y = points_proj[:, 1]
        mask &= -camera.width < x
        mask &= x <= 0
        mask &= -camera.height < y
        mask &= y <= 0

        # Find the corresponding rendered depth values
        points_proj = points_proj.unsqueeze(0).unsqueeze(2)
        z_map = torch.nn.functional.grid_sample(
            input=depth,
            grid=points_proj,
            mode='bilinear',
            padding_mode='border'
        )[0, 0, :, 0]
        
        sdf_estimate = z_map[mask] - z[mask]
        if return_sdf:
            # Compute the estimated SDF
            return sdf_estimate
        else:
            # Compute the estimated density from the estimated SDF
            d_estimate = torch.exp(-0.5 * sdf_estimate.pow(2) / beta[mask].pow(2))
            return d_estimate, mask

    def sample_points(self, num_samples):
        scales = torch.exp(self.scales)
        areas = torch.prod(scales, dim=-1).abs()
        probs = areas.cumsum(dim=-1) / torch.sum(areas, dim=-1, keepdim=True)
        idxs = torch.multinomial(probs, num_samples, replacement=True)
        xi = torch.randn_like(self.means[idxs]) * scales[idxs]
        xi = torch.bmm(quat_to_rot_tensor(self.quats[idxs]), xi[..., None]).squeeze()
        xi = self.means[idxs] + xi
        return xi, idxs

    ###-------- Model export ----------------------------------

    def export_ply(self, filepath):
        # Build data type
        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(np.prod(self.colors_dc.shape[1:])):
            attrs.append('f_dc_{}'.format(i))
        for i in range(np.prod(self.colors_rest.shape[1:])):
            attrs.append('f_rest_{}'.format(i))
        attrs.append('opacity')
        for i in range(self.scales.shape[1]):
            attrs.append('scale_{}'.format(i))
        for i in range(self.quats.shape[1]):
            attrs.append('rot_{}'.format(i))
        obj_dtype = [(attr, 'f4') for attr in attrs]

        # Prepare data
        means = self.means.detach().cpu().numpy()
        normals = np.zeros_like(means)
        colors_dc = self.colors_dc.detach()[:,None,:].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        colors_rest = self.colors_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacities.detach().cpu().numpy()
        scales = self.scales.detach().cpu().numpy()
        quats = self.quats.detach().cpu().numpy()

        # Concatenate data and create PLY object
        elements = np.empty(means.shape[0], dtype=obj_dtype)
        attributes = np.concatenate((means, normals, colors_dc, colors_rest, opacities, scales, quats), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        # Write binary PLY file
        with open(filepath, 'wb') as f:
            PlyData([el]).write(f)

    def export_splat(self, filepath):
        raise NotImplementedError

    def export_mesh(self, filepath, algorithm, **kwargs):
        import open3d as o3d

        if algorithm == 'marching_cubes':
            mesh = self.extract_mesh_marching_cubes(filepath)
        elif algorithm == 'poisson':
            mesh = self.extract_mesh_poisson(kwargs['scene'], kwargs['cameras'])
        else:
            raise ValueError('Unknown mesh extraction algorithm: {}'.format(algorithm))

        # Decimate mesh
        logging.debug("Decimating mesh")
        decimation_target = 250_000
        decimated_mesh = mesh.simplify_quadric_decimation(decimation_target)

        # Clean mesh
        logging.debug("Cleaning mesh")
        decimated_mesh.remove_degenerate_triangles()
        decimated_mesh.remove_duplicated_triangles()
        decimated_mesh.remove_duplicated_vertices()
        decimated_mesh.remove_non_manifold_edges()

        # Save mesh
        o3d.io.write_triangle_mesh(
            filepath,
            decimated_mesh,
            write_triangle_uvs=True,
            write_vertex_colors=False,
            write_vertex_normals=True)

    ###-------- Mesh extraction ---------------------------------------

    def extract_mesh_poisson(self, scene, cameras, **kwargs):
        import open3d as o3d

        # Compute level surface points for each camera
        surface_level = 0.3
        num_total_points = 2_000_000
        num_points_per_camera = num_total_points // len(cameras)

        self.background = torch.zeros(3, device=self.device)

        p_intersects = []
        for cam in tqdm(cameras):
            with torch.no_grad():
                # Render depth
                _, extras = scene.render(cam)
                depth = extras['depth']

                # Backproject depth to 3D points
                idxs = torch.randperm(depth.reshape(-1).shape[0])[:num_points_per_camera]
                y, x = torch.meshgrid(torch.arange(cam.width), torch.arange(cam.height))
                x = x.reshape(-1)[idxs].to(self.device)
                y = y.reshape(-1)[idxs].to(self.device)
                depth = depth.reshape(-1)[idxs]
                p_3d = torch.stack([x, y, depth], dim=-1)
                p_world = cam.backproject_points(p_3d)

                # Find closest Gaussians to each point
                neighbor_idxs = knn_points(p_world[None], self.means[None], K=16).idx[0]
                
                # Prepare samples
                p_std = torch.exp(self.scales)[neighbor_idxs[:,0]].norm(dim=-1)
                p_range = torch.linspace(-3, 3, 21).to(self.device)[None, :, None]
                p_range = p_range * p_std[..., None, None].expand(-1, 21, 1)
                p_norm = torch.nn.functional.normalize(p_world - torch.as_tensor(cam.position, device=self.device), dim=-1)
                samples = p_world[:,None,:] + p_range * p_norm[:,None,:]
                samples = samples.view(-1, 3).float()
                
                # Compute density of samples
                d, _ = self.density_function(samples)
                d = d.reshape(-1, 21)
                
                # Retain densities above threshold
                under = (d - surface_level) < 0
                above = (d - surface_level) > 0
                _, first_d_above_idx = above.max(dim=-1, keepdim=True)
                empty_pix = ~under[..., 0] + (first_d_above_idx[..., 0] == 0)
                d = d[~empty_pix]
                p_range = p_range[~empty_pix][..., 0]
                
                # Compute densities and range before and after surface crossing
                idx = first_d_above_idx[~empty_pix]
                d_before = d.gather(dim=-1, index=idx-1).view(-1)
                t_before = p_range.gather(dim=-1, index=idx-1).view(-1)
                first_d_above = d.gather(dim=-1, index=idx).view(-1)
                first_t_above = p_range.gather(dim=-1, index=idx).view(-1)

                # Compute surface intersection point
                t_intersect = (surface_level - d_before) / \
                              (first_d_above - d_before) * \
                              (first_t_above - t_before) + \
                              t_before
                p_intersect = (p_world[~empty_pix] + t_intersect[:, None] * p_norm[~empty_pix])
                p_intersects.append(p_intersect)
            
        # Construct point cloud from level surface points
        p_intersects = torch.cat(p_intersects)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p_intersects.double().cpu().numpy())
        pcd.estimate_normals()

        # Remove outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
        pcd = pcd.select_by_index(ind)

        # Compute mesh
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=kwargs.get('poisson_depth', 9))

        # Remove low density vertices
        verts = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(verts)

        return mesh #, p_worlds

    def extract_mesh_marching_cubes(self):
        import itertools
        import mcubes
        import open3d as o3d
        from pytorch3d.io import save_obj

        # Surface 
        surface_level = 1
        resolution = 256
        radius = 10 # TODO: Compute extent of the scene 
        X = torch.linspace(-1, 1, resolution) * radius
        Y = torch.linspace(-1, 1, resolution) * radius
        Z = torch.linspace(-1, 1, resolution) * radius
        xx, yy, zz = torch.meshgrid(X, Y, Z)
        P = torch.stack((xx, yy, zz), dim=-1).view(-1, 3).to(self.device)

        # Compute density for every combination of xx, yy, and zz
        logging.debug("Computing density")
        N = 2_000_000
        D = torch.zeros(resolution**3).to(self.device)
        max_dist = np.sqrt(2*(((radius * 2) / resolution)**2))
        for i in range(0, P.shape[0], N):
            D[i:i+N] = self._density(P[i:i+N], max_dist)
        D = D.reshape(resolution, resolution, resolution)

        # Compute the mesh for the given isosurface
        logging.debug("Computing mesh")
        vertices, triangles = mcubes.marching_cubes(D.cpu().numpy(), surface_level)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()

        return mesh

    # TODO: Remove this function in favor of 'density_function' ?
    def _density(self, points, max_dist):
        neighbor_idxs = knn_points(points[None], self.means[None], K=16).idx[0]
        neighbor_means = self.means[neighbor_idxs].detach().squeeze()
        neighbor_opacity = torch.clone(self.opacities[neighbor_idxs]).detach().squeeze()

        # Zero out opacities of neighbors that are too far
        dists = torch.norm(points[:,None] - neighbor_means, dim=-1)
        idxs = dists > (max_dist * 10)
        neighbor_opacity[idxs] = 0

        # Compute density
        density = torch.sum(neighbor_opacity, dim=-1)

        return density

class GaussianDistribution:
    def __init__(self, model, mask):
        self.model = model
        self.mask = mask
        self.means = model.means[mask]
        self.scales = model.scales[mask]
        self.quats = model.quats[mask]
        self.colors_dc = model.colors_dc[mask]
        self.colors_rest = model.colors_rest[mask]
        self.opacities = model.opacities[mask]

    def sample(self, n_samples=2):
        device = self.means.device

        # Perturb the means
        n_splits = self.means.shape[0]

        if n_splits > 0:
            mean_pert = torch.normal(
                mean=torch.zeros(n_splits * n_samples, 3, device=device),
                std=torch.exp(self.scales.repeat(n_samples, 1))).to(device)
            rots = quat_to_rot_tensor(self.quats.repeat(n_samples, 1))
            split_means = torch.bmm(rots, mean_pert[..., None]).squeeze() + self.means.repeat(n_samples, 1)
            split_scales = torch.log(torch.exp(self.scales.repeat(n_samples, 1)) / 1.6)
            self.model.scales[self.mask] = torch.log(torch.exp(self.scales) / 1.6)
        else:
            split_means = self.means.repeat(n_samples, 1)
            split_scales = self.scales.repeat(n_samples, 1)

        samples = {
            "means": split_means,
            "scales": split_scales,
            "quats": self.quats.repeat(n_samples, 1),
            "colors_dc": self.colors_dc.repeat(n_samples, 1),
            "colors_rest": self.colors_rest.repeat(n_samples, 1, 1),
            "opacities": self.opacities.repeat(n_samples, 1),
        }

        # Return samples
        return samples

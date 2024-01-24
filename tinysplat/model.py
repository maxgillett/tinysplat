import os, logging

import torch
import numpy as np
from torch import nn, Tensor
from torch import dtype as dtypes
from torch import device
from torch.nn.parameter import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import SSIM
from sklearn.neighbors import NearestNeighbors
from gsplat.sh import num_sh_bases

from .scene import PointCloud
from .utils import random_quat_tensor, RGB2SH, SH2RGB

class GaussianModel(nn.Module):
    def __init__(self,
        pcd: PointCloud,
        device = device("cuda:0"),
        **kwargs
    ):
        super(GaussianModel, self).__init__()

        # Configuration
        self.max_sh_degree = kwargs['sh_degree']
        self.active_sh_degree = 1
        self.epsilon_alpha = kwargs['epsilon_alpha']
        self.tau_means = kwargs['tau_means']
        self.phi = kwargs['phi']
        self.device = device

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
        self.interval_opacity_reset = kwargs['interval_opacity_reset']

        # Quality metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

        dim_sh = num_sh_bases(self.max_sh_degree)
        self.background = torch.zeros(3, device=device)

        self.filepath = os.path.join(kwargs['splat_dir'], kwargs['splat_filename'])
        if os.path.exists(self.filepath):
            state_dict = torch.load(self.filepath)
            num_points = state_dict['means'].shape[0]

            self.means = Parameter(torch.zeros(num_points, 3, device=device))
            self.colors_dc = Parameter(torch.zeros(num_points, 3, device=device))
            self.colors_rest = Parameter(torch.zeros(num_points, dim_sh - 1, 3, device=device))
            self.scales = Parameter(torch.zeros(num_points, 3, device=device))
            self.quats = Parameter(torch.zeros(num_points, 4, device=device))
            self.opacities = Parameter(torch.zeros(num_points, 1, device=device))

            self.load_state_dict(state_dict)
        else:
            # Initialize colors (using spherical harmonics)
            num_points = pcd.xyz.shape[0]
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
            self.means = Parameter(pcd.xyz.float())
            self.colors_dc = Parameter(colors[:, 0, :])
            self.colors_rest = Parameter(colors[:, 1:, :])
            self.scales = Parameter(torch.as_tensor(scales, device=device))
            self.quats = Parameter(random_quat_tensor(num_points))
            self.opacities = Parameter(torch.logit(0.1 * torch.ones(num_points, 1, device=device)))

        # Accumulate the 'means' gradients for use in cloning/splitting/pruning
        self.means_grad_accum = torch.zeros(num_points, device=device).float()


    def parameters(self):
        return [
            {'params': [self.means], 'lr': self.lr_means, "name": "means"},
            {'params': [self.colors_dc], 'lr': self.lr_colors_dc, "name": "colors_dc"},
            {'params': [self.colors_rest], 'lr': self.lr_colors_rest, "name": "colors_rest"},
            {'params': [self.scales], 'lr': self.lr_scales, "name": "scales"},
            {'params': [self.quats], 'lr': self.lr_quats, "name": "quats"},
            {'params': [self.opacities], 'lr': self.lr_opacities, "name": "opacities"}
        ]

    def increment_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def update_learning_rate(self, optimizer, step):
        pass

    def update_grad_accum(self, step, extras):
        if step < self.warmup_grad: return
        self.means_grad_accum += extras['xys'].grad.norm(dim=-1)

    def reset_opacities(self, step):
        if step % self.interval_opacity_reset != 0: return
        self.opacities.data[:] = self.epsilon_alpha / 2

    def densify_and_prune(self, step, optim, extras):
        if step < self.warmup_densify or step % self.interval_densify != 0:
            return

        width = extras['camera']['width']
        height = extras['camera']['height']
        grad_norm_avg = self.means_grad_accum / self.interval_densify / 2 * max(width, height)
        grad_mask = grad_norm_avg >= self.tau_means

        # Prune Gaussians with low opacity
        prune_mask = (torch.sigmoid(self.opacities) > self.epsilon_alpha).squeeze()
        logging.debug("Pruned {} Gaussians".format((~prune_mask).sum()))

        # Temporarily limit number of Gaussians to 150k
        global_mask = torch.ones(self.means.shape[0]).bool().to(self.device)
        if prune_mask.sum() > 150000:
            global_mask[:] = False

        # Clone Gaussians, and move them in the direction of the gradient
        mask = self.scales.exp().max(dim=-1).values < 0.01
        mask &= prune_mask
        mask &= grad_mask
        mask &= global_mask
        cloned_means = self.means[mask]
        cloned_means = cloned_means
        cloned_colors_dc = self.colors_dc[mask]
        cloned_colors_rest = self.colors_rest[mask]
        cloned_scales = self.scales[mask]
        cloned_quats = self.quats[mask]
        cloned_opacities = self.opacities[mask]
        logging.debug("Cloned {} Gaussians".format(mask.sum()))

        # Split Gaussians (TODO: Sample from distribution)
        mask = self.scales.exp().max(dim=-1).values > 0.01
        mask &= prune_mask
        mask &= grad_mask
        mask &= global_mask
        split_means = self.means[mask]
        split_colors_dc = self.colors_dc[mask]
        split_colors_rest = self.colors_rest[mask]
        split_scales = torch.log(torch.exp(self.scales[mask]) / self.phi)
        split_quats = self.quats[mask]
        split_opacities = self.opacities[mask]
        logging.debug("Split {} Gaussians".format(mask.sum()))

        # Concatenate all newly created parameters
        new_means = torch.cat((cloned_means, split_means))
        new_colors_dc = torch.cat((cloned_colors_dc, split_colors_dc))
        new_colors_rest = torch.cat((cloned_colors_rest, split_colors_rest))
        new_scales = torch.cat((cloned_scales, split_scales))
        new_quats = torch.cat((cloned_quats, split_quats))
        new_opacities = torch.cat((cloned_opacities, split_opacities))

        # Replace optimizer state and parameters.
        # We do this instead of creating a new optimizer because we want to
        # keep the momentum information.
        params = {
            "means": (self.means[prune_mask, ...], new_means),
            "colors_dc": (self.colors_dc[prune_mask, ...], new_colors_dc),
            "colors_rest": (self.colors_rest[prune_mask, ...], new_colors_rest),
            "scales": (self.scales[prune_mask, ...], new_scales),
            "quats": (self.quats[prune_mask, ...], new_quats),
            "opacities": (self.opacities[prune_mask, ...], new_opacities),
        }
        for group in optim.param_groups:
            name = group["name"]
            param = torch.cat(params[name])
            new_param = params[name][1]

            # Mask internal state variables
            state = optim.state[group["params"][0]]
            state["exp_avg"] = state["exp_avg"][prune_mask, ...]
            state["exp_avg_sq"] = state["exp_avg_sq"][prune_mask, ...]

            # Concatenate interval state variables
            new_exp_avg = torch.zeros_like(new_param)
            new_exp_avg_sq = torch.zeros_like(new_param)
            state["exp_avg"] = torch.cat((state["exp_avg"], new_exp_avg))
            state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], new_exp_avg_sq))

            # Replace paramater in optimizer state and model 
            del optim.state[group["params"][0]]
            group["params"][0] = Parameter(param)
            optim.state[group["params"][0]] = state
            setattr(self, name, group["params"][0])
            setattr(self, "test_value", step)

        # Reset accumulated gradients
        self.means_grad_accum = torch.zeros(self.means.shape[0], device=self.device)

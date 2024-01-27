import os

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import minimize

class DepthEstimator:
    def __init__(self, scene, dataset, load_model=False, skip_init=False, **kwargs):
        self.scene = scene
        self.device = kwargs['device']

        # Load depth estimates if they already exist
        stored_depths = dict()
        dir_name = kwargs['depths_path']
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        elif os.path.exists(dir_name):
            for file_name in tqdm(os.listdir(dir_name)):
                if file_name.endswith('.npy'):
                    stored_depths[file_name[:-4]] = np.load(os.path.join(dir_name, file_name), allow_pickle=True)

        # Load the model if not all images have been processed
        if len(stored_depths) < len(scene.cameras) or load_model:
            self.load_model(kwargs['depth_model'])

        if skip_init: return

        for camera in tqdm(scene.cameras):
            depth = stored_depths.get(camera.name)
            if depth is not None:
                camera.estimated_depth = depth
            else:
                depth = self.estimate(camera, dataset)
                camera.estimated_depth = depth
                np.save(os.path.join(dir_name, camera.name + '.npy'), depth)

    def load_model(self, depth_model="depth_anything"):
        if depth_model == "midas":
            self.depth_model = MidasModel(self.device)
        elif depth_model == "zoe":
            self.depth_model = ZoeDepthModel(self.device)
        elif depth_model == "depth_anything":
            self.depth_model = DepthAnythingModel(self.device)
        else:
            raise ValueError("Unknown depth model type")

    def estimate(self, camera, dataset):
        "Returns a SfM scale-matched dense depth map for a chosen camera"

        # Estimate sparse and dense depth (and reprojection error) maps
        D_sparse, E_sparse = self._estimate_sparse(camera, dataset)
        D_dense = self._estimate_dense(camera)

        # MiDaS depth maps are in disparity space
        if self.depth_model.name == ["midas"]:
            D_dense = self._match_scale_disparity(D_dense, D_sparse, E_sparse)
        elif self.depth_model.name in ["zoe", "depth_anything"]:
            D_dense = self._match_scale(D_dense, D_sparse, E_sparse)

        return D_dense

    def _estimate_dense(self, camera):
        "Use a monocular depth estimation model to estimate depth"
        if self.depth_model is None:
            raise ValueError("No depth model loaded")
        return self.depth_model.predict(camera)

    def _estimate_sparse(self, camera, dataset):
        "Returns a depth map estimated from COLMAP data"
        # Get the visible 3D points for the chosen camera
        ids = torch.as_tensor(camera.visible_point_ids).to(self.device)
        xyz_world, _, errors = dataset.pcd.get_points(ids)
        
        # Transform world points to camera points 
        view_mat = camera.view_matrix.to(self.device).float()
        R = view_mat[:3, :3]
        t = view_mat[:, 3]
        xyz_cam = torch.matmul(R, xyz_world.t().float()) + t[:3, np.newaxis]
        xyz_cam = xyz_cam.t()
        xyz_cam = xyz_cam.cpu().numpy()
        
        # Normalize view coordinates
        z = xyz_cam[:,2]
        x = xyz_cam[:,0] / z
        y = xyz_cam[:,1] / z
        
        # Convert to image coordinates
        image_width, image_height = camera.width, camera.height
        f_x = camera.f_x
        f_y = camera.f_y
        c_x = image_width / 2
        c_y = image_height / 2
        x_2d = np.round(x * f_x + c_x).astype(np.int32)
        y_2d = np.round(y * f_y + c_y).astype(np.int32)
        
        # Create sparse depth and error maps
        D_sparse = lil_matrix((image_height, image_width))
        E_sparse = lil_matrix((image_height, image_width))
        for x_, y_, z_, e_ in zip(x_2d, y_2d, z, errors):
            if 0 <= x_ < image_width and 0 <= y_ < image_height:
                D_sparse[y_, x_] = z_
                E_sparse[y_, x_] = e_
        D_sparse = D_sparse.tocoo()
        E_sparse = E_sparse.tocoo()

        return D_sparse, E_sparse

    def _match_scale_disparity(self, D_disparity, D_sparse, E_sparse):
        "Matches the scale in the provided disparity map to that in the sparse map"
        i, j = D_sparse.row, D_sparse.col
        z_dense_inv = torch.as_tensor(D_disparity[i,j].data).to(self.device)
        z_sparse_inv = 1. / torch.as_tensor(D_sparse.data).to(self.device)
        e_sparse = torch.as_tensor(E_sparse.data).to(self.device)

        def func(args):
            s, t = args[0], args[1]
            z_dense_inv_adj = s * z_dense_inv + t
            return (1/e_sparse * (z_sparse_inv - z_dense_inv_adj)).abs().mean().cpu().numpy()
        res = minimize(func, x0=[-0.5,3], method='Nelder-Mead')

        s, t = res.x
        D_dense = 1. / (s * D_disparity + t)
        return D_dense

    def _match_scale(self, D_dense, D_sparse, E_sparse):
        "Matches the scale in the provided metric depth map to that in the sparse map"
        i, j = D_sparse.row, D_sparse.col
        z_dense = torch.as_tensor(D_dense[i,j].data).to(self.device)
        z_sparse = torch.as_tensor(D_sparse.data).to(self.device)
        e_sparse = torch.as_tensor(E_sparse.data).to(self.device)

        def func(args):
            s, t = args[0], args[1]
            z_dense_adj = s * z_dense + t
            return (1/e_sparse * (z_sparse - z_dense_adj)).abs().mean().cpu().numpy()
        res = minimize(func, x0=[-0.5,3], method='Nelder-Mead')

        s, t = res.x
        D_dense = s * D_dense + t
        return D_dense


class ZoeDepthModel:
    def __init__(self, device):
        self.name = "zoe"
        self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
        self.model.to(device)

    def predict(self, camera):
        img = camera.image.pil_image

        # Monkey patch interpolation function (see https://github.com/isl-org/ZoeDepth/pull/60#issuecomment-1894272730)
        # TODO: Use forked ZoeDepth repo with merged PR
        original_interpolate = F.interpolate
        def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
            if size is not None:
                size = tuple(int(s) for s in size)
            return original_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor)

        F.interpolate = patched_interpolate
        depth = self.model.infer_pil(img)
        F.interpolate = original_interpolate

        return depth


class DepthAnythingModel:
    def __init__(self, device):
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

        self.model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
                ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            ])

    def predict(self, camera):
        img = camera.get_original_image().cpu().numpy()
        img = transform({'image': img})['image']
        img = torch.from_numpy(img).unsqueeze(0)
        
        width, height = camera.width, camera.height
        depth = model(image)
        depth = F.interpolate(depth.unsqueeze(1), size=(height, width), mode="bicubic", align_corners=False).squeeze(1)
        depth = depth.cpu().numpy()
        return depth


class MidasModel:
    def __init__(self, device):
        self.name = "midas"
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.model.to(device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform

    def predict(self, camera):
        img = camera.get_original_image().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        return output



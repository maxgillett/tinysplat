from typing import List, Tuple, Optional
import math
import asyncio
import uuid

import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torchvision.transforms.functional as TF

from .utils import quat_to_rot_matrix

class LazyTensorImage:
    def __init__(self, pil_image, device="cuda:0"):
        self.pil_image = pil_image
        self.tensor = None
        self.device = device

    def to_tensor(self):
        if self.tensor is None:
            arr = np.array(self.pil_image)
            self.tensor = torch.tensor(arr) / 255
        return self.tensor


class Camera:
    def __init__(
        self, 
        position: Tensor,
        f_x: float,
        f_y: float,
        fov_x: float,
        fov_y: float,
        quat: Optional[Tensor] = None,
        view_matrix: Optional[Tensor] = None,
        proj_matrix: Optional[Tensor] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        visible_point_ids: Optional[List[int]] = None,
        image: Optional[Image.Image] = None,
        name: Optional[str] = None,
        device = "cuda:0"
    ):
        self.id = uuid.uuid4()
        self.device = device
        self.position = position
        self.view_matrix = view_matrix
        self.proj_matrix = proj_matrix
        self.f_x = f_x
        self.f_y = f_y
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.width = image.width
        self.height = image.height
        self.visible_point_ids = visible_point_ids
        self.image = LazyTensorImage(image, device)
        self.estimated_depth = None
        self.name = name

        if view_matrix is None:
            assert quat is not None
            self.update_view_matrix(position, quat)
        if proj_matrix is None:
            assert near is not None
            assert far is not None
            self.update_proj_matrix(fov_x, fov_y, near, far)

    def update_view_matrix(self, position: Tensor, quat):
        """
        View matrix (world to camera transform)

        The translation vector (tvec) can be computed from the rotation matrix
        R and camera position p according to: -R^T \cdot p.
        Note that inv(view_mat)[:3,3] == position.
        """
        rot_mat = quat_to_rot_matrix(quat)
        view_mat = np.zeros((4,4))
        view_mat[:3, :3] = rot_mat
        view_mat[:3, 3] = -rot_mat.dot(position)
        view_mat[3, 3] = 1
        view_mat = torch.as_tensor(view_mat, dtype=torch.float32)
        self.view_matrix = view_mat

    def update_proj_matrix(self, fov_x: float, fov_y: float, znear: float = 0.001, zfar: float = 1000):
        self.fov_x = fov_x
        self.fov_y = fov_y
        proj_mat = np.zeros((4,4))
        proj_mat[0, 0] = 1. / np.tan(fov_x / 2)
        proj_mat[1, 1] = 1. / np.tan(fov_y / 2)
        proj_mat[2, 2] = (zfar + znear) / (zfar - znear)
        proj_mat[2, 3] = -1. * zfar * znear / (zfar - znear)
        proj_mat[3, 2] = 1
        self.proj_matrix = torch.as_tensor(proj_mat, dtype=torch.float32)

    def rescale(self, factor: float):
        self.width = int(self.width * factor)
        self.height = int(self.height * factor)
        self.fov_x = self.fov_x * factor
        self.fov_y = self.fov_y * factor
        self.update_proj_matrix(self.fov_x, self.fov_y)

    def get_original_image(self, dims: Tuple[int, int] = None) -> Tensor:
        """Get the original image from the camera."""
        img = self.image.to_tensor().to(self.device)
        if dims is not None:
            img = TF.resize(img.permute(2, 0, 1), size=[dims[1], dims[0]], antialias=None)
            img = img.permute(1, 2, 0)
        return img

    def get_estimated_depth(self) -> Tensor:
        return self.estimated_depth


class Scene:
    def __init__(self, cameras, model, rasterizer):
        rng = np.random.default_rng()
        self.model = model
        self.rasterizer = rasterizer
        self.cameras = cameras
        self.camera_training_idxs = rng.permutation(len(self.cameras))
        self.current_camera_idx = 0

    def get_random_camera(self, step) -> Camera:
        """Get a random camera (without replacement) from the dataset."""
        if step % len(self.cameras) - 1:
            rng = np.random.default_rng()
            self.camera_training_idxs = rng.permutation(len(self.cameras))
            self.current_camera_idx = 0
        else:
            self.current_camera_idx += 1
        idx = self.camera_training_idxs[self.current_camera_idx]
        return self.cameras[idx]

    def rescale(self, factor: float):
        for camera in self.cameras:
            camera.rescale(factor)

    def render(self, camera: Camera, dims: Tuple[int, int] = None) -> Tensor:
        return self.rasterizer(camera, dims, self.model.active_sh_degree)


class PointCloud:
    def __init__(self, point_ids: Tensor, xyz: Tensor, colors: Tensor, errors: Tensor):
        idxs = torch.argsort(point_ids)
        self.point_ids = point_ids[idxs]
        self.xyz = xyz[idxs]
        self.colors = colors[idxs]
        self.errors = errors[idxs]

    def get_points(self, ids: Tensor) -> Tensor:
        indices = torch.searchsorted(self.point_ids, ids)
        xyz = self.xyz[indices]
        colors = self.colors[indices]
        errors = self.errors[indices]
        return xyz, colors, errors

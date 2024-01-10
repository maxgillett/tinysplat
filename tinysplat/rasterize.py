from typing import List, Tuple

from gsplat.sh import SphericalHarmonics, num_sh_bases
from gsplat import ProjectGaussians
import numpy as np
import torch
from torch import nn
from torch import device

from .scene import Camera
from .model import GaussianModel
from .cuda import RasterizeGaussiansMultiOutput

class GaussianRasterizer:
    def __init__(self,
        model: GaussianModel,
        cameras: List[Camera],
        device = device("cuda:0")
    ):
        self.BLOCK_X = 16
        self.BLOCK_Y = 16

        self.device = device
        self.model = model
        self.global_scale = torch.tensor([1.0])

    def __call__(self, camera: Camera, dims: Tuple[int, int], sh_degree: int):
        if dims is None:
            dims = (camera.width, camera.height)

        # 1. Project 3D gaussians to 2D using EWA splatting method
        inputs = self.project_forward_inputs(camera, dims)
        xys, depths, radii, conics, num_tiles, _ = ProjectGaussians.apply(*inputs)

        # 2. Compute spherical harmonics
        inputs = self.spherical_harmonics_inputs(camera)
        rgbs = SphericalHarmonics.apply(*inputs)
        rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)

        # 2. Rasterize
        # TODO: Rasterize depth and return as 'depth_out'
        inputs = self.rasterize_forward_inputs(
            xys, depths, radii, conics, num_tiles, rgbs, dims)
        img_out, _, _  = RasterizeGaussiansMultiOutput.apply(*inputs)

        return img_out

    def project_forward_inputs(self, camera, dims):
        model = self.model
        global_scale = 1.
        img_width, img_height = dims
        c_x = img_width / 2
        c_y = img_height / 2
        fov_x = camera.fov_x
        fov_y = camera.fov_y
        view_matrix = camera.view_matrix.to(self.device)
        proj_matrix = camera.proj_matrix.to(self.device)
        return [model.means, model.scales, global_scale, model.quats / model.quats.norm(dim=-1, keepdim=True), view_matrix[:3,:], proj_matrix @ view_matrix, fov_x, fov_y, c_x, c_y, img_height, img_width, self.tile_bounds(dims)]

    def spherical_harmonics_inputs(self, camera):
        n_coeffs = num_sh_bases(self.model.active_sh_degree)
        T = camera.view_matrix[:3,3].to(self.device)
        view_dirs = self.model.means - T # (N, 3)
        view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)
        coeffs = self.model.colors[:, :n_coeffs, :]
        return [self.model.active_sh_degree, view_dirs, coeffs]

    def rasterize_forward_inputs(self, xys, depths, radii, conics, num_tiles, rgbs, dims):
        model = self.model
        img_width, img_height = dims
        return [xys, depths, radii, conics, num_tiles, rgbs, torch.sigmoid(model.opacities), img_height, img_width, model.background]

    def tile_bounds(self, dims: Tuple[int, int]):
        img_width, img_height = dims
        return (
            (img_width + self.BLOCK_X - 1) // self.BLOCK_X,
            (img_height + self.BLOCK_Y - 1) // self.BLOCK_Y,
            1,
        )

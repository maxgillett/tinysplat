import copy
from typing import List, Tuple, Type

import ctypes
import numpy as np
from tinygrad.tensor import Tensor, Function
from tinygrad.helpers import compile_cuda_style, dtypes, flat_mv
from tinygrad.runtime.ops_cuda import CUDAProgram, CUDAAllocator, CUDADevice, check
from tinygrad.shape.shapetracker import ShapeTracker
import gpuctypes.cuda as cuda

from scene import Camera
from model import GaussianModel

BLOCK_X = 16
BLOCK_Y = 16
BLOCK_SIZE = BLOCK_X * BLOCK_Y
N_THREADS = 256

class GaussianRasterizer:
    def __init__(self,
        model: GaussianModel,
        cameras: List[Camera],
        clip_thresh: float = 1e-1,
    ):
        self.model = model
        self.clip_thresh = Tensor([clip_thresh])
        self.global_scale = Tensor([1.0])
        self.background = Tensor([0.0])

        # We assume for now that all images have the same FOV and dimensions,
        # so we can just use the first camera's parameters
        img_dims = (cameras[0].img_width, cameras[0].img_height)
        self.img_width, self.img_height = img_dims
        self.fov_x = 2 * np.arctan(self.img_width / 2)
        self.fov_y = 2 * np.arctan(self.img_height / 2)
        self.tile_bounds = (
            (self.img_width + BLOCK_X - 1) // BLOCK_X,
            (self.img_height + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )

    def __call__(self, camera: Camera, sh_degree: int):
        # 1. Project 3D gaussians to 2D using EWA splatting method
        (cov_3d, xys, depths, radii, conics, n_tiles, mask) = self._project_gaussians(camera)

        # 2. Compute cumulative intersections
        num_intersects = self._compute_cumulative_intersections()

        # 3. Compute tile ID and depth value for each intersection
        self._map_gaussian_intersections()

        # 4. Bin and sort gaussians
        self._bin_and_sort()

        # 5. Get tile bins
        self._get_tile_bins()

        # 6. Rasterize
        out_img, final_Ts, final_idx = self._rasterize()

        return out_img, final_Ts, final_idx

    def _project_gaussians(self, camera: Camera):
        means = self.model.means
        scales = self.model.scales
        quats = self.model.quats
        view_mat = camera.view_matrix
        proj_mat = camera.proj_matrix

        # Clip the near plane
        R = view_mat[..., :3, :3]
        T = view_mat[..., :3, 3]
        p_view = R.matmul(means[..., None])[..., 0] + T
        is_close = p_view[..., 2] < self.clip_thresh

        for i in range(self.model.means.shape[0]):
            # Compute the rotation matrix from the quaternion
            # TODO Make sure that quaternion is normalized in the camera class
            R = quat_to_mat(quats[i])

            # Compute the projected covariance
            M = R * self.global_scale * scales[..., None, :]
            cov_3d = M.matmul(M.transpose(-1, -2))

        return (cov_3d,)

    def _compute_cumulative_intersections(self):
        #cum_tiles_hit = num_tiles_hit.cumsum(axis=0)
        #num_intersects = cum_tiles_hit[-1]
        #return num_intersects
        pass

    def _map_gaussian_intersections(self):
        pass

    def _bin_and_sort(self):
        pass

    def _get_tile_bins(self):
        pass

    def _rasterize(self):
        out_img = Tensor(np.zeros((self.img_height, self.img_width, 3), dtype=np.float32))
        final_Ts = Tensor(np.zeros((self.img_height, self.img_width), dtype=np.float32))
        final_idx = Tensor(np.zeros((self.img_height, self.img_width), dtype=np.int32))
        return out_img, final_Ts, final_idx

def quat_to_mat(quat):
    w, x, y, z = quat[0], quats[1], quats[2], quats[3]
    R = Tensor.stack(
        [
            Tensor.stack(
                [
                    1 - 2 * (y**2 + z**2),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ],
                dim=-1,
            ),
            Tensor.stack(
                [
                    2 * (x * y + w * z),
                    1 - 2 * (x**2 + z**2),
                    2 * (y * z - w * x),
                ],
                dim=-1,
            ),
            Tensor.stack(
                [
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x**2 + y**2),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )
    return R

from typing import List, Tuple
import uuid

import numpy as np
from PIL import Image
from tinygrad import Tensor

class Camera:
    def __init__(self, position: Tensor, view_matrix: Tensor, proj_matrix: Tensor, fov_x: float, fov_y: float, image: Image, name: str):
        self.id = uuid.uuid4()
        self.position = position
        self.img_height = image.height
        self.img_width = image.width
        self.view_matrix = view_matrix
        self.proj_matrix = proj_matrix
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.image = Tensor(np.array(image))
        self.name = name

    def get_original_image(self) -> Tensor:
        """Get the original image from the camera."""
        return self.image


class PointCloud:
    def __init__(self, points: Tensor, colors: Tensor):
        self.points = points
        self.colors = colors


class Scene:
    def __init__(self, cameras, model, rasterizer):
        rng = np.random.default_rng()
        self.model = model
        self.rasterizer = rasterizer
        self.cameras = cameras
        self.camera_training_idxs = rng.permutation(len(self.cameras))
        self.current_camera_idx = 0

    def get_random_camera(self) -> Camera:
        """Get a random camera (without replacement) from the dataset."""
        if self.current_camera_idx == len(self.cameras):
            rng = np.random.default_rng()
            self.camera_training_idxs = rng.permutation(self.cameras.size)
            self.current_camera_idx = 0
        else:
            self.current_camera_idx += 1
        idx = self.camera_training_idxs[self.current_camera_idx]
        return self.cameras[idx]

    def render(self, camera: Camera) -> (Tensor, Tensor, Tensor):
        return self.rasterizer(camera, self.model.active_sh_degree)

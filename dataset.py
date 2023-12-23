import os

from PIL import Image
import numpy as np
import pycolmap
from tinygrad import Tensor

from scene import Camera, PointCloud

class Dataset:
    def __init__(self, colmap_path: str, images_path: str):
        """Load a dataset from a COLMAP reconstruction."""
        # TODO: Move to a "from_colmap" constructor

        reconstruction = pycolmap.Reconstruction(colmap_path)
        cameras = reconstruction.cameras
        images = list(reconstruction.images.values())
        points_3d = list(reconstruction.points3D.values())

        # Construct cameras
        self.cameras = []
        for img in images[:2]:
            image_path = os.path.join(images_path, img.name)
            image_name = os.path.basename(img.name)
            image = Image.open(image_path)

            # Camera center
            position = img.projection_center()

            # Focal length and field of view
            cam = cameras[img.camera_id]
            f = cam.params[0] # focal length
            fov_x = 2 * np.arctan(image.width / (2 * f))
            fov_y = 2 * np.arctan(image.height / (2 * f))

            # View matrix (note that inv(view_mat)[:3,3] == position)
            view_mat = np.zeros((4,4))
            view_mat[:3, :3] = img.rotation_matrix()
            view_mat[:3, 3] = img.tvec
            view_mat[3, 3] = 1
            view_mat = Tensor(view_mat)

            # Projection matrix
            znear, zfar = 0.001, 1000
            proj_mat = np.zeros((4,4))
            proj_mat[0, 0] = 1. / np.tan(fov_x / 2)
            proj_mat[1, 1] = 1. / np.tan(fov_y / 2)
            proj_mat[2, 2] = (zfar + znear) / (zfar - znear)
            proj_mat[2, 3] = -1. * zfar * znear / (zfar - znear)
            proj_mat[3, 2] = 1
            proj_mat = Tensor(proj_mat)

            camera = Camera(position, view_mat, proj_mat, fov_x, fov_y, image, name=image_name)
            self.cameras.append(camera)

        # Construct point cloud
        points = []
        colors = []
        for pt in points_3d:
            points.append(pt.xyz)
            colors.append(pt.color)
        self.pcd = PointCloud(points=Tensor(np.asarray(points)), colors=Tensor(np.asarray(colors)))


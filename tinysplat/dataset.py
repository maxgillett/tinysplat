import os

from PIL import Image
import numpy as np
import pycolmap
import torch
from torch import tensor
import viser.transforms as vtf

from .scene import Camera, PointCloud

class Dataset:
    def __init__(self,
        colmap_path: str,
        images_path: str,
        max_image_dimension: int = 512,
        device = "cuda:0",
    ):
        """Load a dataset from a COLMAP reconstruction."""

        reconstruction = pycolmap.Reconstruction(colmap_path)
        cameras = reconstruction.cameras
        images = list(reconstruction.images.values())
        points_3d = list(reconstruction.points3D.values())
        points_ids = list(reconstruction.points3D.keys())

        # Construct cameras
        self.cameras = []
        for img in images[:]:
            image_path = os.path.join(images_path, img.name)
            image_name = os.path.basename(img.name)
            image = Image.open(image_path)

            # Camera center
            position = img.projection_center()

            # Focal length and field of view
            cam = cameras[img.camera_id]
            if len(cam.focal_length_idxs()) == 2:
                f_x = cam.focal_length_x
                f_y = cam.focal_length_y
            else:
                f_x = cam.focal_length
                f_y = cam.focal_length
            #fov_x = 2 * np.arctan(image.width / (2*f_x))
            #fov_y = 2 * np.arctan(image.height / (2*f_y))
            fov_x = 2 * np.arctan(image.width / (f_x))
            fov_y = 2 * np.arctan(image.height / (f_y))

            # View matrix (world to camera transform)
            # Note that inv(view_mat)[:3,3] == position
            rot_mat = img.rotation_matrix()
            view_mat = np.zeros((4,4))
            view_mat[:3, :3] = rot_mat
            view_mat[:3, 3] = img.tvec
            view_mat[3, 3] = 1
            view_mat = tensor(view_mat, dtype=torch.float32, device=device)

            # Projection matrix
            znear, zfar = 0.001, 1000
            #proj_mat = np.zeros((4,4))
            #proj_mat[0, 0] = 1. / np.tan(fov_x / 2)
            #proj_mat[1, 1] = 1. / np.tan(fov_y / 2)
            #proj_mat[2, 2] = (zfar + znear) / (zfar - znear)
            #proj_mat[2, 3] = -1. * zfar * znear / (zfar - znear)
            #proj_mat[3, 2] = 1
            #proj_mat = tensor(proj_mat, dtype=torch.float32, device=device)
            #cam.update_proj_matrix(fov_x, fov_y, znear, zfar)

            # 3D points visible in this image
            visible_point_ids = torch.as_tensor([p.point3D_id for p in img.points2D if p.has_point3D()], device=device)

            camera = Camera(
                position=position, 
                view_matrix=view_mat,
                #proj_matrix=proj_mat,
                f_x=f_x,
                f_y=f_y,
                fov_x=fov_x,
                fov_y=fov_y,
                near=znear,
                far=zfar,
                image=image,
                visible_point_ids=visible_point_ids,
                name=image_name,
                device=device)
            self.cameras.append(camera)

        # Construct point cloud
        self.pcd = PointCloud(
            point_ids=torch.as_tensor(
                np.asarray(points_ids), device=device),
            xyz=torch.as_tensor(
                np.asarray([pt.xyz for pt in points_3d]), device=device),
            colors=torch.as_tensor(
                np.asarray([pt.color for pt in points_3d]), device=device),
            errors=torch.as_tensor(
                np.asarray([pt.error for pt in points_3d]), device=device)
        )


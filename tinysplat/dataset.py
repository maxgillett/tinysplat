import os

import cv2
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
            if (n_focal_length_idxs := len(cam.focal_length_idxs())) == 2:
                f_x = cam.focal_length_x
                f_y = cam.focal_length_y
            else:
                f_x = cam.focal_length
                f_y = cam.focal_length

            # Compare image width to cam width, and adjust focal length accordingly
            if (n_principal_point_idxs := len(cam.principal_point_idxs())) == 2:
                c_x = cam.principal_point_x 
                c_y = cam.principal_point_y
            else:
                c_x = cam.principal_point
                c_y = cam.principal_point
            f_x *= image.width / 2 / c_x
            f_y *= image.height / 2 / c_y

            # Undistort image
            n_intrinsic_params = n_focal_length_idxs + n_principal_point_idxs
            if len(cam.params) > n_intrinsic_params:
                cam_matrix = np.array([
                    [f_x, 0,   c_x],
                    [0,   f_y, c_y],
                    [0,   0,   1],
                ])
                k_params = cam.params[n_intrinsic_params:]
                k_params = np.pad(k_params, (0, 8 - len(k_params)), "constant")
                new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, k_params, (image.width, image.height), 0)
                f_x = new_cam_matrix[0,0]
                f_y = new_cam_matrix[1,1]
                c_x = new_cam_matrix[0,2]
                c_y = new_cam_matrix[1,2]
                image = cv2.undistort(np.array(image), cam_matrix, k_params, None, new_cam_matrix)
                x, y, w, h = roi
                image = image[y:y+h, x:x+w]
                image = Image.fromarray(image)

            # Compute field of view
            fov_x = 2 * np.arctan(image.width / (2 * f_x))
            fov_y = 2 * np.arctan(image.height / (2 * f_y))

            # 3D points visible in this image
            visible_point_ids = torch.as_tensor([p.point3D_id for p in img.points2D if p.has_point3D()], device=device)

            camera = Camera(
                position=position, 
                f_x=f_x,
                f_y=f_y,
                fov_x=fov_x,
                fov_y=fov_y,
                quat=torch.as_tensor(img.qvec),
                near=0.001,
                far=1000,
                image=image,
                visible_point_ids=visible_point_ids,
                name=image_name,
                device=device)
            self.cameras.append(camera)

        # Compute camera extent
        cam_positions = np.hstack([cam.position for cam in self.cameras])
        mean_position = np.mean(cam_positions)
        self.spatial_extent = np.max(np.linalg.norm(cam_positions - mean_position, axis=0)) * 1.1

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


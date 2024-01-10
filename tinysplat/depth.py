import os

import torch
from tqdm import tqdm

class DepthEstimator:
    def __init__(self, scene, **kwargs):
        # Load depth estimates if they already exist
        stored_depths = dict()
        dir_name = kwargs['depths_path']
        if os.path.exists(dir_name):
            for file_name in tqdm(os.listdir(dir_name)):
                if file_name.endswith('.npy'):
                    stored_depths[file_name[:-4]] = np.load(os.path.join(dir_name, file_name))

        # Load the model if not all images have been processed
        if len(stored_depths) < len(scene.cameras):
            self.load_model(args.midas_model_type)

        for camera in tqdm(scene.cameras):
            depth = stored_depths.get(camera.name)
            if depth:
                camera.estimated_depth = depth
            else:
                depth = self.estimate(self.get_original_image().cpu().numpy())
                camera.estimated_depth = depth
                np.save(os.path.join(dir_name, camera.name + '.npy'), depth)

    def load_model(self, model_type="MiDaS_small"):
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def estimate(self, img):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()

import sys, asyncio, logging
from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
from tqdm import tqdm

from tinysplat.dataset import Dataset
from tinysplat.depth import DepthEstimator
from tinysplat.model import GaussianModel
from tinysplat.rasterize import GaussianRasterizer
from tinysplat.scene import Scene
from tinysplat.viewer import Viewer

async def train(
    model: GaussianModel,
    scene: Scene,
    args,
):
    device = args.device
    metrics = Metrics(model, scene, args)
    optimizer = optim.Adam(model.parameters())

    for step in tqdm(range(1, args.max_iter)):
        # TODO: Rescale scene at desired steps
        #if step == 0:   scene.rescale(0.25)
        #if step == 250: scene.rescale(2)
        #if step == 500: scene.rescale(2)

        # 1. Update the learning rate of the gaussians
        # 2. Every N iterations, increase the spherical harmonic degree by one
        model.update_learning_rate(optimizer, step)
        if step % args.sh_increment_interval == 0:
            model.increment_sh_degree()
        model.background = torch.rand(3, device=device)

        # 3. Pick a random camera from the scene and render the viewpoint
        camera = scene.get_random_camera(step)
        rendered_image, extras = scene.render(camera)

        # 4. Compute the loss between the rendered image and the ground truth
        dims = None #(camera.width, camera.height)
        ground_truth_image = camera.get_original_image(dims).float()
        loss_l1 = (rendered_image - ground_truth_image).abs().mean()
        loss_ssim = 1 - model.ssim(
            rendered_image.permute(2, 0, 1).unsqueeze(0),
            ground_truth_image.permute(2, 0, 1).unsqueeze(0))
        loss = (1 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim

        if args.regularize_depth:
            estimated_depth = torch.as_tensor(camera.get_estimated_depth()).float().to(device)
            rendered_depth = extras['depth']
            loss_depth = (rendered_depth - estimated_depth).abs().mean()
            loss += args.lambda_depth * loss_depth

        # 5. Backpropagate the loss
        loss.backward()

        # 6. Perform optimization step
        optimizer.step()

        # 7. Densify and prune
        with torch.no_grad():
            model.update_grad_accum(step, extras)
            model.densify_and_prune(step, optimizer, extras)
        optimizer.zero_grad(set_to_none=True)

        # Record and display metrics
        metrics.update(step, 'PSNR', float(model.psnr(rendered_image, ground_truth_image).item()))
        metrics.update(step, 'Loss', loss.detach().cpu().numpy())
        metrics.update(step, 'Loss (image)', loss_l1.detach().cpu().numpy())
        metrics.update(step, 'Loss (dssim)', loss_ssim.detach().cpu().numpy())
        if args.regularize_depth:
            metrics.update(step, 'Loss (depth)', loss_depth.detach().cpu().numpy())
        metrics.log(step)

        # 8. Every M iterations, save checkpoint
        if model.filepath is not None and step % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), model.filepath)

        # Yield control back to the event loop
        if args.viewer: await asyncio.sleep(0)


class Metrics():
    def __init__(self, model, scene, args):
        self.num_cameras = num_cameras = len(scene.cameras)
        self.metrics = defaultdict(lambda: np.zeros((args.max_iter // num_cameras + 1, num_cameras)))
        self.model = model
        self.scene = scene
        self.args = args

    def update(self, step, key, value):
        cam_idx = self.scene.camera_training_idxs[self.scene.current_camera_idx]
        self.metrics[key][step // self.num_cameras + 1, cam_idx] = value

    def log(self, step):
        if step % self.num_cameras != 0: return
        str_out = ""
        for key, mat in self.metrics.items():
            row_mean = mat[(step // self.num_cameras) - 1,:].mean()
            str_out += f"{key}: {row_mean:<10.4f} | "
        str_out += f"N: {self.model.means.shape[0]:<10}"
        tqdm.write(str_out)


def arg_parser():
    parser = ArgumentParser(description='Configuration parameters')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--train', action=BooleanOptionalAction)
    parser.add_argument('--viewer', type=bool, default=True)
    parser.add_argument('--splat-dir', type=str, default='splats')
    parser.add_argument('--splat-filename', type=str, default='model.splat')
    parser.add_argument('--sh-degree', type=int, default=3)
    parser.add_argument('--max-iter', type=int, default=10_000)
    parser.add_argument('--sh-increment-interval', type=int, default=500)
    parser.add_argument('--checkpoint-interval', type=int, default=10_000)

    # Viewer
    parser_viewer = parser.add_argument_group('Viewer')
    parser_viewer.add_argument('--viewer-ip', type=str, default="127.0.0.1")
    parser_viewer.add_argument('--viewer-port', type=int, default=8765)

    # Dataset
    parser_dataset = parser.add_argument_group('Dataset')
    parser_dataset.add_argument('--dataset-dir', type=str, default='datasets/train')
    parser_dataset.add_argument('--colmap-path', type=str, default='colmap/sparse/0')
    parser_dataset.add_argument('--images-path', type=str, default='images')

    # Learning rates
    parser_lr = parser.add_argument_group('Learning rates')
    parser_lr.add_argument('--lr-means', type=float, default=0.00016)
    parser_lr.add_argument('--lr-colors-dc', type=float, default=0.0025)
    parser_lr.add_argument('--lr-colors-rest', type=float, default=0.000125)
    parser_lr.add_argument('--lr-scales', type=float, default=0.005)
    parser_lr.add_argument('--lr-quats', type=float, default=0.001)
    parser_lr.add_argument('--lr-opacities', type=float, default=0.05)

    # Regularization
    # TODO: Add schedulers (for depth, in particular)
    parser_lambda = parser.add_argument_group('Regularization')
    parser_lambda.add_argument('--regularize-depth', action=BooleanOptionalAction)
    parser_lambda.add_argument('--lambda-dssim', type=float, default=0.2)
    parser_lambda.add_argument('--lambda-depth', type=float, default=0.1)
    parser_lambda.add_argument('--lambda-smooth', type=float, default=0.2)

    # Gaussian densification
    parser_densify = parser.add_argument_group('Gaussian densification')
    parser_densify.add_argument('--warmup-densify', type=int, default=600)
    parser_densify.add_argument('--warmup-grad', type=int, default=500)
    parser_densify.add_argument('--interval-densify', type=int, default=100)
    parser_densify.add_argument('--interval-opacity-reset', type=int, default=3000)
    parser_densify.add_argument('--epsilon-alpha', type=float, default=0.005)
    parser_densify.add_argument('--tau-means', type=float, default=0.0002)
    parser_densify.add_argument('--phi', type=float, default=1.6)

    # Depth estimation
    parser_depth = parser.add_argument_group('Depth estimation')
    parser_depth.add_argument('--depths-path', type=str, default='depths')
    parser_depth.add_argument('--midas-model-type', type=str, default='MiDaS_small')

    return parser

async def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s')

    parser = arg_parser()
    args = parser.parse_args(sys.argv[1:])
    args.colmap_path = f'{args.dataset_dir}/{args.colmap_path}'
    args.images_path = f'{args.dataset_dir}/{args.images_path}'
    args.depths_path = f'{args.dataset_dir}/{args.depths_path}'
    coroutines = []

    device = torch.device(args.device)
    dataset = Dataset(
        colmap_path=args.colmap_path,
        images_path=args.images_path,
        device=device)
    model = GaussianModel(
        pcd=dataset.pcd,
        **vars(args)).to(device)
    rasterizer = GaussianRasterizer(model, dataset.cameras)
    scene = Scene(dataset.cameras, model, rasterizer)

    if args.viewer:
        # It would be nice to move the viewer to a separate process in the future
        viewer = Viewer(scene, args.viewer_ip, args.viewer_port, device=device)
        coroutines.append(viewer.run())

    if args.regularize_depth:
        depth_estimator = DepthEstimator(scene, dataset, **vars(args))

    if args.train:
        coroutines.append(train(model, scene, args))

    await asyncio.gather(*coroutines)

if __name__ == "__main__":
    asyncio.run(main())

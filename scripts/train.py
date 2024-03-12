import sys, asyncio, logging
from datetime import datetime
from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
from tqdm import tqdm

from tinysplat import GaussianModel, GaussianRasterizer
from tinysplat.dataset import Dataset
from tinysplat.depth import DepthEstimator
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

    # Schedulers
    regularize_depth_schedule = Scheduler(
        args.regularize_depth,
        args.regularize_depth_start,
        args.regularize_depth_end)
    regularize_opacity_schedule = Scheduler(
        args.regularize_opacity,
        args.regularize_opacity_start,
        args.regularize_opacity_end)
    regularize_density_schedule = Scheduler(
        args.regularize_density,
        args.regularize_density_start,
        args.regularize_density_end)

    # Checkpoint filepath
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    for step in tqdm(range(1, args.max_iter + 1)):
        # 1. Update the learning rate of the gaussians
        # 2. Every N iterations, increase the spherical harmonic degree by one
        model.update_learning_rate(step, optimizer)
        if step % args.sh_increment_interval == 0:
            model.increment_sh_degree()
        model.background = torch.rand(3, device=device)

        # 3. Pick a random camera from the scene and render the viewpoint
        camera = scene.get_random_camera(step)
        rendered_image, extras = scene.render(camera)

        # 4. Compute the loss between the rendered image and the ground truth
        ground_truth_image = camera.get_original_image().float()
        loss_l1 = (rendered_image - ground_truth_image).abs().mean()
        loss_ssim = 1 - model.ssim(
            rendered_image.permute(2, 0, 1).unsqueeze(0),
            ground_truth_image.permute(2, 0, 1).unsqueeze(0))
        loss = (1 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim

        if regularize_depth_schedule(step):
            estimated_depth = torch.as_tensor(camera.get_estimated_depth()).float().to(device)
            rendered_depth = extras['depth']
            loss_depth = (rendered_depth - estimated_depth).abs().mean()
            loss += args.lambda_depth * loss_depth

        if regularize_opacity_schedule(step):
            opacities = torch.sigmoid(model.opacities)
            loss_opacity = -(opacities * torch.log(opacities + 1e-10) + \
                           (1 - opacities) * torch.log(1 - opacities + 1e-10)).mean()
            loss += args.lambda_opacity * loss_opacity

        if regularize_density_schedule(step):
            update = (step == regularize_density_schedule.start or step % args.interval_densify == 1)
            if update:
                model.points, _ = model.sample_points(num_samples=100_000)
            points = model.points
            density, neighbor_idxs = model.density_function(points, update_neighbors=update)
            beta = torch.exp(model.scales).min(dim=-1)[0][neighbor_idxs].mean(dim=1)
            approx_val, mask = model.approximate_density_function(
                points, extras['depth'], camera, beta, return_sdf=args.regularize_sdf)
            if args.regularize_sdf:
                sdf = beta * torch.sqrt(-2 * torch.log(density.clamp(0.001, 0.999)))
                loss_density = (sdf[mask] - approx_val).abs().mean()
            else:
                loss_density = (density[mask] - approx_val).abs().mean()
            loss += args.lambda_density * loss_density

        # 5. Backpropagate the loss
        loss.backward(retain_graph=True)

        # 6. Perform optimization step
        optimizer.step()

        # 7. Densify and prune
        with torch.no_grad():
            model.update_grad_accum(step, extras)
            model.densify_and_prune(step, optimizer, extras)
            if regularize_density_schedule.start == step:
                opacity_mask = (torch.sigmoid(model.opacities) < 0.5).squeeze()
                model.update_state(optimizer, opacity_mask)
        optimizer.zero_grad(set_to_none=True)

        # Record and display metrics
        metrics.update(step, 'PSNR', float(model.psnr(rendered_image, ground_truth_image).item()))
        metrics.update(step, 'Loss', loss.detach().cpu().numpy())
        metrics.update(step, 'Loss (image)', loss_l1.detach().cpu().numpy())
        metrics.update(step, 'Loss (dssim)', loss_ssim.detach().cpu().numpy())
        if args.regularize_depth:
            metrics.update(step, 'Loss (depth)', loss_depth.detach().cpu().numpy())
        if regularize_opacity_schedule(step):
            metrics.update(step, 'Loss (opacity)', loss_opacity.detach().cpu().numpy())
        if regularize_density_schedule(step):
            metrics.update(step, 'Loss (density)', loss_density.detach().cpu().numpy())
        metrics.log(step)

        # 8. Every M iterations, save checkpoint (the filename should be the current date + step
        if args.save_checkpoints and step % args.checkpoint_interval == 0:
            filepath = f'{args.checkpoint_dir}/{timestamp}-{step}.pth'
            torch.save(model.state_dict(), filepath)

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
        self.metrics[key][step // self.num_cameras, cam_idx] = value

    def log(self, step):
        if step % self.num_cameras != 0: return
        str_out = ""
        for key, mat in self.metrics.items():
            row_mean = mat[(step // self.num_cameras) - 1,:].mean()
            str_out += f"{key}: {row_mean:<10.4f} | "
        str_out += f"N: {self.model.means.shape[0]:<10}"
        tqdm.write(str_out)


class Scheduler():
    def __init__(self, active, start, stop):
        self.active = active
        self.start = start
        self.stop = stop

    def __call__(self, step):
        return self.active and self.start <= step < self.stop


def arg_parser():
    parser = ArgumentParser(description='Configuration parameters')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--train', action=BooleanOptionalAction)
    parser.add_argument('--viewer', type=bool, default=True)
    parser.add_argument('--load-checkpoint', type=str)
    parser.add_argument('--save-checkpoints', action=BooleanOptionalAction)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
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
    # TODO: Add learning rate schedulers
    parser_lambda = parser.add_argument_group('Regularization weights')
    parser_lambda.add_argument('--lambda-dssim', type=float, default=0.2)
    parser_lambda.add_argument('--lambda-depth', type=float, default=0.2)
    parser_lambda.add_argument('--lambda-smooth', type=float, default=0.2)
    parser_lambda.add_argument('--lambda-opacity', type=float, default=0.2)
    parser_lambda.add_argument('--lambda-density', type=float, default=0.2)

    # Gaussian densification
    parser_densify = parser.add_argument_group('Gaussian densification')
    parser_densify.add_argument('--warmup-densify', type=int, default=600)
    parser_densify.add_argument('--warmup-grad', type=int, default=500)
    parser_densify.add_argument('--interval-densify', type=int, default=100)
    parser_densify.add_argument('--interval-opacity-reset', type=int, default=3000)
    parser_densify.add_argument('--densify-end', type=int, default=30000)
    parser_densify.add_argument('--epsilon-alpha', type=float, default=0.005)
    parser_densify.add_argument('--tau-means', type=float, default=0.0002)
    parser_densify.add_argument('--densify-scale-thresh', type=float, default=0.01)
    parser_densify.add_argument('--phi', type=float, default=1.6)

    # Semantic segmentation
    parser_semantic = parser.add_argument_group('Semantic segmentation')
    parser_semantic.add_argument('--semantic-path', type=str, default='semantic')
    parser_semantic.add_argument('--semantic-model', type=str, default='facebook/mask2former-swin-large-ade-semantic')

    # Depth estimation
    parser_depth_estimate = parser.add_argument_group('Depth estimation')
    parser_depth_estimate.add_argument('--depths-path', type=str, default='depths')
    parser_depth_estimate.add_argument('--depth-model', type=str, default='zoe')

    # Depth regularization
    parser_depth = parser.add_argument_group('Depth regularization')
    parser_depth.add_argument('--regularize-depth', action=BooleanOptionalAction)
    parser_depth.add_argument('--regularize-depth-start', type=int, default=1)
    parser_depth.add_argument('--regularize-depth-end', type=int, default=15000)

    # Opacity entropy regularization (useful for mesh reconstruction)
    parser_entropy = parser.add_argument_group('Opacity regularization')
    parser_entropy.add_argument('--regularize-opacity', action=BooleanOptionalAction)
    parser_entropy.add_argument('--regularize-opacity-start', type=int, default=7000)
    parser_entropy.add_argument('--regularize-opacity-end', type=int, default=9000)

    # Density regularization (useful for mesh reconstruction)
    parser_surface = parser.add_argument_group('SuGaR density regularization')
    parser_surface.add_argument('--regularize-density', action=BooleanOptionalAction)
    parser_surface.add_argument('--regularize-sdf', action=BooleanOptionalAction)
    parser_surface.add_argument('--regularize-density-start', type=int, default=9000)
    parser_surface.add_argument('--regularize-density-end', type=int, default=15000)

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

    if filepath := args.load_checkpoint:
        state_dict = torch.load(filepath)
        model = GaussianModel.from_state_checkpoint(
            state_dict,
            **vars(args)).to(device)
    else:
        model = GaussianModel.from_pcd(
            dataset.pcd,
            **vars(args)).to(device)
    rasterizer = GaussianRasterizer(model, dataset.cameras)
    scene = Scene(dataset.cameras, model, rasterizer)
    model.interval_densify = len(scene.cameras)

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

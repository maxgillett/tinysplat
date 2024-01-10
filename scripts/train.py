import os, sys
import asyncio
import logging
from argparse import ArgumentParser, BooleanOptionalAction

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
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
    filepath = None,
):
    device = args.device
    optimizer = optim.Adam(model.parameters())

    # Rescale images
    for camera in scene.cameras:
        camera.rescale(0.25)

    for step in tqdm(range(1, args.max_iter)):
        # 1. Update the learning rate of the gaussians
        # 2. Every N iterations, increase the spherical harmonic degree by one
        model.update_learning_rate(optimizer, step)
        if step % args.sh_increment_interval == 0:
            model.increment_sh_degree()

        # 3. Pick a random camera from the scene and render the viewpoint
        camera = scene.get_random_camera(step)
        rendered_image = scene.render(camera)

        # 4. Compute the loss between the rendered image and the ground truth
        dims = (camera.width, camera.height)
        ground_truth_image = camera.get_original_image(dims).float().to(device)
        loss_l1 = (rendered_image - ground_truth_image).abs().mean()
        loss_ssim = 1 - model.ssim(
            rendered_image.permute(2, 0, 1).unsqueeze(0),
            ground_truth_image.permute(2, 0, 1).unsqueeze(0))
        loss = (1 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim

        if args.regularize_depth:
            estimated_depth = camera.get_original_depth().float().to(device)
            loss_depth = (rendered_depth - ground_truth_depth).abs().mean()
            loss += args.lambda_depth * loss_depth

        # 5. Backpropagate the loss
        loss.backward()

        # 6. Perform optimization step
        optimizer.step()

        # 7. Densify and prune
        with torch.no_grad():
            model.update_grad_accum(step)
            model.densify_and_prune(step, optimizer)
            model.reset_opacities(step)
        optimizer.zero_grad(set_to_none=False)

        # Print metrics
        model.psnr.update(rendered_image, ground_truth_image)
        r = model.psnr.compute()
        tqdm.write(
            "PSNR:" + str(r.cpu().numpy()) + 
            ", Loss:" + str(loss_l1.detach().cpu().numpy()) +
            ", N:" + str(model.means.shape[0]))

        # 8. Every M iterations, save checkpoint
        if filepath is not None and step % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), filepath)

        # Rescale images
        if step == 250 or step == 500:
            for camera in scene.cameras:
                camera.rescale(2)

        # Yield control back to the event loop
        if args.viewer: await asyncio.sleep(0)

async def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s')

    parser = ArgumentParser(description='Training script parameters')
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
    parser_viewer = parser.add_argument_group('viewer')
    parser_viewer.add_argument('--viewer-ip', type=str, default="127.0.0.1")
    parser_viewer.add_argument('--viewer-port', type=int, default=8765)

    # Dataset
    parser_dataset = parser.add_argument_group('dataset')
    parser_dataset.add_argument('--colmap-path', type=str, 
                                default='datasets/train/colmap/sparse/0')
    parser_dataset.add_argument('--images-path', type=str,
                                default='datasets/train/images')

    # Learning rates
    parser_lr = parser.add_argument_group('lr')
    parser_lr.add_argument('--lr-means', type=float, default=0.00016)
    parser_lr.add_argument('--lr-colors', type=float, default=0.0025)
    parser_lr.add_argument('--lr-scales', type=float, default=0.005)
    parser_lr.add_argument('--lr-quats', type=float, default=0.001)
    parser_lr.add_argument('--lr-opacities', type=float, default=0.05)

    # Regularization
    parser_lambda = parser.add_argument_group('lambda')
    parser_lambda.add_argument('--regularize-depth', action=BooleanOptionalAction)
    parser_lambda.add_argument('--lambda-dssim', type=float, default=0.2)
    parser_lambda.add_argument('--lambda-depth', type=float, default=0.2)
    parser_lambda.add_argument('--lambda-smooth', type=float, default=0.2)

    # Gaussian densification
    parser_densify = parser.add_argument_group('densify')
    parser_densify.add_argument('--warmup-densify', type=int, default=600)
    parser_densify.add_argument('--warmup-grad', type=int, default=500)
    parser_densify.add_argument('--interval-densify', type=int, default=100)
    parser_densify.add_argument('--interval-opacity-reset', type=int, default=3000)
    parser_densify.add_argument('--epsilon-alpha', type=float, default=0.005)
    parser_densify.add_argument('--tau-means', type=float, default=0.0002)
    parser_densify.add_argument('--phi', type=float, default=1.6)

    # Depth estimation
    parser_depth = parser.add_argument_group('depth')
    parser_depth.add_argument('--depths-path', type=str, 
                              default='datasets/train/depths')
    parser_depth.add_argument('--midas-model-type', type=str, default='MiDaS_small')

    args = parser.parse_args(sys.argv[1:])
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

    filepath = os.path.join(args.splat_dir, args.splat_filename)
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))

    if args.viewer:
        # It would be nice to move the viewer to a separate process in the future
        viewer = Viewer(scene, args.viewer_ip, args.viewer_port, device=device)
        coroutines.append(viewer.run())

    if args.regularize_depth:
        depth_estimator = DepthEstimator(scene, **vars(args))
        depth_estimator.estimate()

    if args.train:
        coroutines.append(train(model, scene, args, filepath=filepath))

    await asyncio.gather(*coroutines)

if __name__ == "__main__":
    asyncio.run(main())

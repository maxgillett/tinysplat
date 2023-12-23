import os
from typing import List, Tuple

from tinygrad import Tensor
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from tinygrad.runtime.ops_cuda import compile_cuda, CUDAProgram, CUDADevice

from dataset import Dataset
from model import GaussianModel
from rasterize import GaussianRasterizer
from scene import Scene

def train(dataset: Dataset):
    device = CUDADevice("cuda:0")

    model = GaussianModel(sh_degree=3, pcd=dataset.pcd)
    rasterizer = GaussianRasterizer(model, dataset.cameras)
    scene = Scene(dataset.cameras, model, rasterizer)
    optimizer = optim.Adam(get_parameters(model), lr=1e-3)

    file_name = "gaussian.bin"
    if os.path.exists(file_name):
        model.load(file_name)

    for step in range(1000):
        print("step", step)

        # 1. Update the learning rate of the gaussians
        # 2. Every N iterations, increase the spherical harmonic degree by one
        model.update_learning_rate(optimizer, step)
        if step % 10 == 0:
            model.increment_sh_degree()

        # 3. Pick a random camera from the scene and render the viewpoint
        camera = scene.get_random_camera()
        (rendered_image, transmittances, visibilities) = scene.render(camera)

        # 4. Compute the loss between the rendered image and the ground truth
        ground_truth_image = camera.get_original_image()
        loss = (rendered_image - ground_truth_image).abs().mean()

        # 5. Backpropagate the loss
        loss.backward()

        # 6. Densify and prune
        model.densify_and_prune(transmittances, visibilities)

        # 7. Perform optimization step
        optimizer.step()
        optimizer.zero_grad()

        # 8. Every M iterations, save checkpoint
        if step % 100 == 0:
            model.save(file_name)

if __name__ == "__main__":
    dataset = Dataset(
        colmap_path="datasets/apartment/colmap/sparse/0",
        images_path="datasets/apartment/images")
    train(dataset)

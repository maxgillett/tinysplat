import torch
import numpy as np
from torch import Tensor
from torch import dtype as dtypes
from torch import device as CUDADevice
from sklearn.neighbors import NearestNeighbors

from scene import PointCloud
from utils import num_sh_bases, logit

class GaussianModel:
    def __init__(self,
        sh_degree: int,
        pcd: PointCloud,
        clip_thresh: float = 0.01,
        device = CUDADevice("cuda:0")
    ):
        """
        Initialize the model from a point cloud
        
        Args:
        sh_degree: The maximum spherical harmonic (SH) degree to use for the model
        num_points: The initial number of points to use for the model
        clip_thresh: Minimum z depth threshold
        """
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.epsilon = 0.005
        num_points = pcd.points.shape[0]
        dim_sh = num_sh_bases(sh_degree)

        self.means = torch.tensor(np.stack(pcd.points.numpy(), axis=0), dtype=torch.float32, requires_grad=True)

        # Initialize colors (with spherical harmonics)
        colors = np.zeros((num_points, dim_sh, 3))
        colors[:, 0, :] = np.stack(pcd.colors.numpy(), axis=0)
        self.colors = torch.tensor(colors, requires_grad = True)

        # Find the average of the three nearest neighbors for each point and 
        # use that as the scale
        nn = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='euclidean')
        nn = nn.fit(pcd.points.numpy())
        distances, indices = nn.kneighbors(pcd.points.numpy())
        log_mean_dist = np.log(np.mean(distances[:, 1:], axis=1))
        scales = np.repeat(log_mean_dist[:, np.newaxis], 3, axis=1)
        self.scales = torch.tensor(scales, requires_grad=True)

        quats = np.zeros((num_points, 4))
        quats[:, 3] = 1.0
        self.quats = torch.tensor(quats, requires_grad=True)
        self.opacities = logit(0.1 * torch.ones(num_points, 1, requires_grad=True))
        self.grad_norm = torch.empty(num_points, 3, requires_grad=False)

    def increment_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def update_learning_rate(self, optimizer, iteration):
        # TODO: Reduce the learning rate of the positions/means over time
        pass

    def densify_and_prune(self, transmittances: Tensor, visibilities: Tensor):
        # Update the gradients
        norm = self.means.grad.pow(2).sum(dim=1).sqrt()
        print("grads", self.means.grad.numpy())
        #self.grad_norm[visibilities] += norm

        # Clone Gaussians
        idxs = (self.grad_norm.square().sum(axis=1).sqrt() >= self.epsilon).where(True, False).numpy().nonzero()[0]
        if idxs.size > 0:
            cloned_means = self.means[idxs]
            cloned_scales = self.scales[idxs]
            cloned_rotations = self.rotations[idxs]
            cloned_opacities = self.opacities[idxs]

            # Extend tensors with cloned values
            self.means = torch.cat((self.means, cloned_means), dim=0)
            self.scales = torch.cat((self.scales, cloned_scales), dim=0)
            self.rotations = torch.cat((self.rotations, cloned_rotations), dim=0)
            self.opacities = torch.cat((self.opacities, cloned_opacities), dim=0)

        # TODO: Split Gaussians
        #stds = self.scales.exp()[idxs].repeat([2, 1])
        #Tensor.normal(Tensor.zeros((stds.size(0), 3)), stds, out=self.means)

        # TODO: Prune points
        # Remove Gaussians with opacity < \epsilon_

    @property
    def num_points(self):
        return self.means.shape[0]

    def load(self, path: str):
        with open(filename+'.npy', 'rb') as f:
            for name, par in self.named_parameters():
                try:
                    par.data = torch.from_numpy(np.load(f))
                    if torch.cuda.is_available():
                        par.data = par.data.to('cuda')
                except:
                    print('Could not load parameter')


    def save(self, path: str):
        with open(filename+'.npy', 'wb') as f:
            for name, par in self.named_parameters():
                np.save(f, par.data.cpu().numpy())

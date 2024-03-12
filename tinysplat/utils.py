import math
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def random_quat_tensor(N, **kwargs):
    u = torch.rand(N, **kwargs)
    v = torch.rand(N, **kwargs)
    w = torch.rand(N, **kwargs)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

def quat_to_rot_matrix(quat):
    """
    Convert a quaternion into a full three-dimensional rotation matrix.
    """
    q0, q1, q2, q3 = quat[0], quat[1], quat[2], quat[3]
    R = np.asarray([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q3*q0, 2*q1*q3 + 2*q2*q0],
        [2*q1*q2 + 2*q3*q0, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q1*q0],
        [2*q1*q3 - 2*q2*q0, 2*q2*q3 + 2*q1*q0, 1 - 2*q1**2 - 2*q2**2]
    ])
    return R

# Taken from gsplat
def quat_to_rot_tensor(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(F.normalize(quat, dim=-1), dim=-1)
    return torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (y**2 + z**2),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (x * y + w * z),
                    1 - 2 * (x**2 + z**2),
                    2 * (y * z - w * x),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x**2 + y**2),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )

## Modified from PixelNeRF (https://www.github.com/sxyu/pixel-nerf)
def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((-X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj

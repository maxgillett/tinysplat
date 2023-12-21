import copy
from typing import List, Tuple

import ctypes
from tinygrad.tensor import Tensor
from tinygrad.helpers import compile_cuda_style, dtypes, flat_mv
from tinygrad.runtime.ops_cuda import CUDAProgram, CUDAAllocator, CUDADevice, check
import gpuctypes.cuda as cuda

from scene import Camera
from model import GaussianModel

BLOCK_X = 16
BLOCK_Y = 16
BLOCK_SIZE = BLOCK_X * BLOCK_Y
N_THREADS = 256

# CUDA kernel code modified from gsplat (https://github.com/nerfstudio-project/gsplat)
# See the mathematical supplement for details (https://arxiv.org/abs/2312.02121)
kernel_names = [
    ("forward.cu", ["project_gaussians_forward_kernel", "map_gaussians_to_intersects", "get_tile_bin_edges", "rasterize_forward"]),
    ("backward.cu", ["project_gaussians_backward_kernel", "rasterize_backward_kernel"]),
]
lowered_names = dict([(name, None) for (_, names) in kernel_names for name in names])

def cuda_compile_program_outer(prog, *args):
    for name in lowered_names.keys():
        status = cuda.nvrtcAddNameExpression(prog, name.encode())
        if status != cuda.NVRTC_SUCCESS:
            raise Exception("Failed to add name expression:", name)
        
    status = cuda.nvrtcCompileProgram(prog, *args)
    if status != cuda.NVRTC_SUCCESS:
        raise Exception("Failed to compile", status)
    
    for name in lowered_names.keys():
        lowered_name_ptr = ctypes.POINTER(ctypes.c_char)()
        status = cuda.nvrtcGetLoweredName(prog, name.encode(), ctypes.byref(lowered_name_ptr))
        if status != cuda.NVRTC_SUCCESS:
            raise Exception("Failed to get lowered name:", name)
        lowered_names[name] = ctypes.cast(lowered_name_ptr, ctypes.c_char_p).value.decode()
                             
    return status

def compile_cuda(filepath):
    prg = open(filepath).read()
    code = compile_cuda_style(
        prg,
        [f'--gpu-architecture={CUDADevice.default_arch_name}',
         "-I/usr/local/cuda/include",
         "-I/usr/include"],
        cuda.nvrtcProgram,
        cuda.nvrtcCreateProgram,
        cuda_compile_program_outer,
        cuda.nvrtcGetPTX,
        cuda.nvrtcGetPTXSize,
        cuda.nvrtcGetProgramLog,
        cuda.nvrtcGetProgramLogSize,
        check)
    return code

def call_cuda_kernel(
        program: CUDAProgram,
        allocator: CUDAAllocator,
        inputs=[],
        outputs=[],
        global_size: Tuple[int, ...] = None,
        local_size: Tuple[int, ...] = None,
    ):
    bufs = []
    bufs.extend(inputs)
    bufs.extend(outputs)

    # Allocate items if not already allocated
    for i, buf in enumerate(bufs):
        if isinstance(buf, Tensor):
            bufs[i] = cuda_allocate(buf, allocator, copy_in=True)

    # Execute kernel
    program(*bufs, global_size=global_size, local_size=local_size, wait=True)
	
def cuda_allocate(item: Tensor, allocator: CUDAAllocator, copy_in=False):
    a = allocator.alloc(np.prod(item.shape)*item.dtype.itemsize)
    if copy_in:
        na = item.realize().lazydata.realized.toCPU()
        allocator.copyin(a, bytearray(na))
    return a


class GaussianRasterizer:
    def __init__(self,
        model: GaussianModel,
        cameras: List[Camera],
        device: CUDADevice,
        clip_thresh: float = 1e-1,
        max_num_points: int = 5_000_000,
    ):
        self.model = model
        self.clip_thresh = clip_thresh
        self.background = 0.0
        self.kernels = dict()
        self.allocator = CUDAAllocator(device)
        self.max_num_points = max_num_points

        # We assume for now that all images have the same FOV and dimensions,
        # so we can just use the first camera's parameters
        img_dims = (cameras[0].image_width, cameras[0].image_height)
        self.img_width, self.img_height = img_dims
        self.fov_x = 2 * np.arctan(self.img_width / 2)
        self.fov_y = 2 * np.arctan(self.img_height / 2)
        self.tile_bounds = (
            (self.img_width + BLOCK_X - 1) // BLOCK_X,
            (self.img_height + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        
        # Compile kernels
        for filename, names in kernel_names.items():
            code = compile_cuda("cuda/" + filename)
            for name in names:
                self.kernels[name] = CUDAProgram(device, name, code)

        # Allocate input and output buffers
        self.buffers = RasterizerBuffers(self, cameras)

    def __call__(self, camera: Camera, sh_degree: int):
        # Fetch the current camera parameters
        view_matrix = self.inputs.cameras[camera.id]["view_matrix"]
        proj_matrix = self.inputs.cameras[camera.id]["proj_matrix"]

        # Copy in model parameter data
        self.update_model_params()

        # 1. Project 3D gaussians to 2D using EWA splatting method
        call_cuda_kernel(
            program=self.kernels["project_gaussians_forward_kernel"],
            allocator=self.allocator,
            inputs=self.buffers.project_forward_inputs(
                view_matrix,
                proj_matrix),
            outputs=self.buffers.project_forward_outputs,
            global_size=(N + N_THREADS - 1) // N_THREADS,
            local_size=N_THREADS)

        # 2. Compute cumulative intersections
        num_tiles_hit = np.empty((N), dtype=np.int32)
        self.allocator.copyout(num_tiles_hit, self.outputs.num_tiles_hit)
        cum_tiles_hit = num_tiles_hit.cumsum(axis=0)
        num_intersects = cum_tiles_hit[-1].item()

        # 3. Compute tile ID and depth value for each intersection
        call_cuda_kernel(
            program=self.kernels["map_gaussians_to_intersects"],
            allocator=self.allocator,
            inputs=self.buffers.map_gaussians_inputs(
                num_intersects,
                cum_tiles_hit),
            outputs=self.buffers.map_gaussians_outputs,
            global_size=(N + N_THREADS - 1) // N_THREADS,
            local_size=N_THREADS)

        # 4. Bin and sort gaussians
        isect_ids = np.empty((num_intersects,), dtype=np.int32)
        self.allocator.copyout(isect_ids, self.buffers.isect_ids)
        sorted_indices = Tensor(isect_ids).argsort()
        isect_ids_sorted = isect_ids[sorted_indices]
        gaussian_ids_sorted = gaussian_ids.gather(sorted_indices, dim=0)

        # 5. Get tile bins
        call_cuda_kernel(
            program=self.kernels["get_tile_bin_edges"],
            allocator=self.allocator,
            inputs=[
                num_intersects,
                isect_ids_sorted],
            outputs=self.buffers.tile_bin_edges_outputs,
            global_size=(num_intersects + N_THREADS - 1) // N_THREADS,
            local_size=N_THREADS)

        # 6. Rasterize
        call_cuda_kernel(
            program=self.kernels["rasterize_forward"],
            allocator=self.allocator,
            inputs=self.inputs.rasterize_forward_inputs(gaussian_ids_sorted),
            outputs=self.outputs.rasterize_forward_outputs,
            global_size=tile_bounds,
            local_size=(BLOCK_X, BLOCK_Y))

        # Copy out the output image, transmittances, and depth values
        # TODO: Can we instead keep this on the GPU and initialize Tensors with it?
        out_img = np.empty((self.img_height, self.img_width, 3), dtype=np.float32)
        final_Ts = np.empty((self.img_height, self.img_width), dtype=np.float32)
        final_idx = np.empty((self.img_height, self.img_width), dtype=np.int32)
        allocator.copyout(out_img, self.outputs.out_img)
        allocator.copyout(final_Ts, self.outputs.final_Ts)
        allocator.copyout(final_idx, self.outputs.final_idx)

        return (out_img, final_Ts, final_idx)

    def update_model_params(self):
        model = self.rasterizer.model
        for (buf, tensor) in [
            (self.inputs.means_3d, model.means_3d),
            (self.inputs.scales, model.scales),
            (self.inputs.colors, model.colors),
            (self.inputs.rotations, model.rotations)
        ]:
            allocator.copyin(buf, bytearray(tensor.realize().lazydata.realized.toCPU()))


# TODO: Should we reallocate memory instead whenever the number of gaussians changes
# instead of assuming a fixed maximum number?
class RasterizerBuffers:
    def __init__(self, rasterizer: GaussianRasterizer, cameras: List[Camera]):
        allocator = rasterizer.allocator
        N = rasterizer.max_num_points
        img_width, img_height = rasterizer.img_width, rasterizer.img_height

        # Model parameter buffers
        self.means_3d = cuda_allocate(Tensor.empty(N, 3, dtype=dtypes.float32), allocator)
        self.scales = cuda_allocate(Tensor.empty(N, 3, dtype=dtypes.float32), allocator)
        self.colors = cuda_allocate(Tensor.empty(N, 3, dtype=dtypes.float32), allocator)
        self.rotations = cuda_allocate(Tensor.empty(N, 4, dtype=dtypes.float32), allocator)
        self.opacities = cuda_allocate(Tensor.empty(N, dtype=dtypes.float32), allocator)

        # Camera parameter buffers
        self.cameras = dict()
        for camera in cameras:
            self.cameras[camera.id] = {
                "view_matrix": cuda_allocate(camera.view_matrix, allocator, copy_in=True),
                "proj_matrix": cuda_allocate(camera.view_matrix, allocator, copy_in=True)
            }

        # Input buffers
        self.num_points = cuda_allocate(Tensor([N], dtype=dtypes.int32), allocator, copy_in=True)
        self.img_height = cuda_allocate(Tensor([rasterizer.img_height], dtype=dtypes.int32), allocator, copy_in=True)
        self.img_width = cuda_allocate(Tensor([img_width], dtype=dtypes.int32), allocator, copy_in=True)
        self.fov_x = cuda_allocate(Tensor([rasterizer.fov_x], dtype=dtypes.float32), allocator, copy_in=True)
        self.fov_y = cuda_allocate(Tensor([rasterizer.fov_y], dtype=dtypes.float32), allocator, copy_in=True)
        self.c_x = cuda_allocate(Tensor([img_width / 2], dtype=dtypes.float32), allocator, copy_in=True)
        self.c_y = cuda_allocate(Tensor([img_height / 2], dtype=dtypes.float32), allocator, copy_in=True)
        self.global_scale = cuda_allocate(Tensor([1.], dtype=dtypes.float32), allocator, copy_in=True)
        self.clip_thresh = cuda_allocate(Tensor(rasterizer.clip_thresh), allocator, copy_in=True)
        self.tile_bounds = cuda_allocate(Tensor(rasterizer.tile_bounds), allocator, copy_in=True)
        self.background = cuda_allocate(Tensor(rasterizer.background), allocator, copy_in=True)

        # Output buffers
        self.covs_3d = cuda_allocate(Tensor.empty(N, 6, dtype=dtypes.float32), allocator)
        self.xys = cuda_allocate(Tensor.empty(N, 2, dtype=dtypes.float32), allocator)
        self.depths = cuda_allocate(Tensor.empty(N, dtype=dtypes.float32), allocator)
        self.radii = cuda_allocate(Tensor.empty(1, dtype=dtypes.int32), allocator)
        self.conics = cuda_allocate(Tensor.empty(N, 3, dtype=dtypes.float32), allocator)
        self.num_tiles_hit = cuda_allocate(Tensor.empty(N, dtype=dtypes.int32), allocator)
        self.isect_ids = cuda_allocate(Tensor.empty(N, dtype=dtypes.int64), allocator)
        self.gaussian_ids = cuda_allocate(Tensor.empty(N, dtype=dtypes.int32), allocator)
        self.tile_bins = cuda_allocate(Tensor.empty(N, 2, dtype=dtypes.int32), allocator)
        self.final_Ts = cuda_allocate(Tensor.empty(img_height, img_width, dtype=dtypes.float32), allocator)
        self.final_idx = cuda_allocate(Tensor.empty(img_height, img_width, dtype=dtypes.int32), allocator)
        self.out_img = cuda_allocate(Tensor.empty(img_height, img_width, 3, dtype=dtypes.float32), allocator)

    # ***** input buffer methods *****

    def project_forward_inputs(self, view_matrix, proj_matrix):
        return [self.num_points, self.means_3d, self.scales, self.global_scale, self.rotations, view_matrix, proj_matrix, self.fov_x, self.fov_y, self.c_x, self.c_y, self.img_height, self.img_width, self.tile_bounds, self.clip_thresh]

    def map_gaussians_inputs(self, num_intersects, cum_tiles_hit):
        return [self.num_points, num_intersects, self.xys, self.depths, self.radii, cum_tiles_hit, self.tile_bounds]

    def rasterize_forward_inputs(self, gaussian_ids_sorted):
        return [self.tile_bounds, self.img_size, gaussian_ids_sorted, self.tile_bins, self.xys, self.conics, self.colors, self.opacities, self.background]

    # ***** output buffer methods *****

    @property
    def project_forward_outputs(self):
        return [self.covs_3d, self.xys, self.depths, self.radii, self.conics, self.num_tiles_hit]

    @property
    def map_gaussians_outputs(self):
        return [self.isect_ids, self.gaussian_ids]

    @property
    def tile_bin_edges_outputs(self):
        return [self.tile_bins]

    @property
    def rasterize_forward_outputs(self):
        return [self.final_Ts, self.final_idx, self.out_img]

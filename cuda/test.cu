#include <cuda_runtime.h>
#include <cooperative_groups.h>

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float3* __restrict__ means3d
    //const float3* __restrict__ scales,
    //const float glob_scale,
    //const float4* __restrict__ quats,
    //const float* __restrict__ viewmat,
    //const float* __restrict__ projmat,
    //const float4 intrins,
    //const dim3 img_size,
    //const dim3 tile_bounds,
    //const float clip_thresh,
    //float* __restrict__ covs3d,
    //float2* __restrict__ xys,
    //float* __restrict__ depths,
    //int* __restrict__ radii,
    //float3* __restrict__ conics,
    //int32_t* __restrict__ num_tiles_hit
) {
    covs3d[0] = 1.0;
    // TODO
}

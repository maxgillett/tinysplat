#include <cuda_runtime.h>
#include <cooperative_groups.h>
#define CHANNELS 3
namespace cg = cooperative_groups;

__device__ __constant__ float SH_C0 = 0.28209479177387814f;
__device__ __constant__ float SH_C1 = 0.4886025119029199f;
__device__ __constant__ float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
__device__ __constant__ float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
__device__ __constant__ float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

// This function is used in both host and device code
__host__ __device__ unsigned num_sh_bases(const unsigned degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

__device__ void sh_coeffs_to_color(
    const unsigned degree,
    const float3 &viewdir,
    const float *coeffs,
    float *colors
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    // expects CHANNELS * num_bases coefficients
    // supports up to num_bases = 25
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * coeffs[1 * CHANNELS + c] +
                              z * coeffs[2 * CHANNELS + c] -
                              x * coeffs[3 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * coeffs[4 * CHANNELS + c] +
             SH_C2[1] * yz * coeffs[5 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * coeffs[6 * CHANNELS + c] +
             SH_C2[3] * xz * coeffs[7 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * coeffs[8 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * xx - yy) * coeffs[9 * CHANNELS + c] +
             SH_C3[1] * xy * z * coeffs[10 * CHANNELS + c] +
             SH_C3[2] * y * (4.f * zz - xx - yy) * coeffs[11 * CHANNELS + c] +
             SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) *
                 coeffs[12 * CHANNELS + c] +
             SH_C3[4] * x * (4.f * zz - xx - yy) * coeffs[13 * CHANNELS + c] +
             SH_C3[5] * z * (xx - yy) * coeffs[14 * CHANNELS + c] +
             SH_C3[6] * x * (xx - 3.f * yy) * coeffs[15 * CHANNELS + c]);
        if (degree < 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (xx - yy) * coeffs[16 * CHANNELS + c] +
             SH_C4[1] * yz * (3.f * xx - yy) * coeffs[17 * CHANNELS + c] +
             SH_C4[2] * xy * (7.f * zz - 1.f) * coeffs[18 * CHANNELS + c] +
             SH_C4[3] * yz * (7.f * zz - 3.f) * coeffs[19 * CHANNELS + c] +
             SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) *
                 coeffs[20 * CHANNELS + c] +
             SH_C4[5] * xz * (7.f * zz - 3.f) * coeffs[21 * CHANNELS + c] +
             SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) *
                 coeffs[22 * CHANNELS + c] +
             SH_C4[7] * xz * (xx - 3.f * yy) * coeffs[23 * CHANNELS + c] +
             SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) *
                 coeffs[24 * CHANNELS + c]);
    }
}

__device__ void sh_coeffs_to_color_vjp(
    const unsigned degree,
    const float3 &viewdir,
    const float *v_colors,
    float *v_coeffs
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_coeffs[c] = SH_C0 * v_colors[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        float v1 = -SH_C1 * y;
        float v2 = SH_C1 * z;
        float v3 = -SH_C1 * x;
        v_coeffs[1 * CHANNELS + c] = v1 * v_colors[c];
        v_coeffs[2 * CHANNELS + c] = v2 * v_colors[c];
        v_coeffs[3 * CHANNELS + c] = v3 * v_colors[c];
        if (degree < 2) {
            continue;
        }
        float v4 = SH_C2[0] * xy;
        float v5 = SH_C2[1] * yz;
        float v6 = SH_C2[2] * (2.f * zz - xx - yy);
        float v7 = SH_C2[3] * xz;
        float v8 = SH_C2[4] * (xx - yy);
        v_coeffs[4 * CHANNELS + c] = v4 * v_colors[c];
        v_coeffs[5 * CHANNELS + c] = v5 * v_colors[c];
        v_coeffs[6 * CHANNELS + c] = v6 * v_colors[c];
        v_coeffs[7 * CHANNELS + c] = v7 * v_colors[c];
        v_coeffs[8 * CHANNELS + c] = v8 * v_colors[c];
        if (degree < 3) {
            continue;
        }
        float v9 = SH_C3[0] * y * (3.f * xx - yy);
        float v10 = SH_C3[1] * xy * z;
        float v11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float v12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float v13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float v14 = SH_C3[5] * z * (xx - yy);
        float v15 = SH_C3[6] * x * (xx - 3.f * yy);
        v_coeffs[9 * CHANNELS + c] = v9 * v_colors[c];
        v_coeffs[10 * CHANNELS + c] = v10 * v_colors[c];
        v_coeffs[11 * CHANNELS + c] = v11 * v_colors[c];
        v_coeffs[12 * CHANNELS + c] = v12 * v_colors[c];
        v_coeffs[13 * CHANNELS + c] = v13 * v_colors[c];
        v_coeffs[14 * CHANNELS + c] = v14 * v_colors[c];
        v_coeffs[15 * CHANNELS + c] = v15 * v_colors[c];
        if (degree < 4) {
            continue;
        }
        float v16 = SH_C4[0] * xy * (xx - yy);
        float v17 = SH_C4[1] * yz * (3.f * xx - yy);
        float v18 = SH_C4[2] * xy * (7.f * zz - 1.f);
        float v19 = SH_C4[3] * yz * (7.f * zz - 3.f);
        float v20 = SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f);
        float v21 = SH_C4[5] * xz * (7.f * zz - 3.f);
        float v22 = SH_C4[6] * (xx - yy) * (7.f * zz - 1.f);
        float v23 = SH_C4[7] * xz * (xx - 3.f * yy);
        float v24 = SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));
        v_coeffs[16 * CHANNELS + c] = v16 * v_colors[c];
        v_coeffs[17 * CHANNELS + c] = v17 * v_colors[c];
        v_coeffs[18 * CHANNELS + c] = v18 * v_colors[c];
        v_coeffs[19 * CHANNELS + c] = v19 * v_colors[c];
        v_coeffs[20 * CHANNELS + c] = v20 * v_colors[c];
        v_coeffs[21 * CHANNELS + c] = v21 * v_colors[c];
        v_coeffs[22 * CHANNELS + c] = v22 * v_colors[c];
        v_coeffs[23 * CHANNELS + c] = v23 * v_colors[c];
        v_coeffs[24 * CHANNELS + c] = v24 * v_colors[c];
    }
}

//__global__ void child(int *out, size_t n)                                
__global__ void child()                                
{                                                               
   //size_t tid = blockIdx.x * blockDim.x + threadIdx.x;          
   //if (tid < n) {                                               
   //   out[tid] = tid;                                           
   //}                                                            
}       

__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float* __restrict__ projmat,
    const float4 intrins,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float* __restrict__ covs3d,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int* __restrict__ num_tiles_hit
    //int *coeff,
    ////float coeff,
    ////float r,
    //int *v_coeffs
) {
    //size_t n = 2;
    //size_t numBlocks = 32;
    //size_t numThreads = 128;
    //dim3 numBlocks(1);  // Number of blocks
    //dim3 numThreads(256);  

    //child<<<numBlocks, numThreads>>>();                    
    //cudaDeviceSynchronize();   

    //v_coeffs[0] = coeff[0];
    //v_coeffs[1] = 3;

    radii[0] = 3;
}

__global__ void compute_sh_forward_kernel(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ coeffs,
    float* __restrict__ colors
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }
    const unsigned num_channels = 3;
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    sh_coeffs_to_color(
        degrees_to_use, viewdirs[idx], &(coeffs[idx_sh]), &(colors[idx_col])
    );
}

__global__ void compute_sh_backward_kernel(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ v_colors,
    float* __restrict__ v_coeffs
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }
    const unsigned num_channels = 3;
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    sh_coeffs_to_color_vjp(
        degrees_to_use, viewdirs[idx], &(v_colors[idx_col]), &(v_coeffs[idx_sh])
    );
}


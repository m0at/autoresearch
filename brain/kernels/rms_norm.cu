#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

// ── Warp reduction ──────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ── RMSNorm forward ─────────────────────────────────────────────────────────
// Warp-per-row: each warp of 32 threads processes one row.
// 8 warps per block = 8 rows per block. No shared memory, no __syncthreads.
// For D=512: 16 elements per lane, x cached in L1 for second pass.

__global__ void fused_rms_norm_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    uint32_t rows, uint32_t D, float eps)
{
    uint32_t lane = threadIdx.x & 31;
    uint32_t warp_id = threadIdx.x >> 5;
    uint32_t warps_per_block = blockDim.x >> 5;
    uint32_t row = blockIdx.x * warps_per_block + warp_id;
    if (row >= rows) return;

    const __nv_bfloat16* xr = x + (uint64_t)row * D;
    __nv_bfloat16* yr = y + (uint64_t)row * D;

    // Pass 1: sum of squares
    float local_sq = 0.0f;
    for (uint32_t i = lane; i < D; i += 32) {
        float v = __bfloat162float(xr[i]);
        local_sq += v * v;
    }
    float sum_sq = warp_reduce_sum(local_sq);
    float rrms = rsqrtf(sum_sq / (float)D + eps);

    // Pass 2: normalize (x re-read from L1 cache)
    for (uint32_t i = lane; i < D; i += 32) {
        yr[i] = __float2bfloat16(__bfloat162float(xr[i]) * rrms);
    }
}

// ── RMSNorm backward ────────────────────────────────────────────────────────
// Warp-per-row. Two reductions (sum_sq, dot) via warp shuffles only.
// No shared memory needed.

__global__ void fused_rms_norm_bwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ grad_out,
    __nv_bfloat16* __restrict__ grad_in,
    uint32_t rows, uint32_t D, float eps)
{
    uint32_t lane = threadIdx.x & 31;
    uint32_t warp_id = threadIdx.x >> 5;
    uint32_t warps_per_block = blockDim.x >> 5;
    uint32_t row = blockIdx.x * warps_per_block + warp_id;
    if (row >= rows) return;

    const __nv_bfloat16* xr  = x        + (uint64_t)row * D;
    const __nv_bfloat16* gor = grad_out + (uint64_t)row * D;
    __nv_bfloat16* gir       = grad_in  + (uint64_t)row * D;

    // Pass 1: compute sum_sq and dot(grad, x)
    float local_sq  = 0.0f;
    float local_dot = 0.0f;
    for (uint32_t i = lane; i < D; i += 32) {
        float xi = __bfloat162float(xr[i]);
        float gi = __bfloat162float(gor[i]);
        local_sq  += xi * xi;
        local_dot += gi * xi;
    }
    float sum_sq = warp_reduce_sum(local_sq);
    float dot_gx = warp_reduce_sum(local_dot);

    float rrms  = rsqrtf(sum_sq / (float)D + eps);
    float coeff = dot_gx / (float)D * rrms * rrms;

    // Pass 2: compute grad_in (x and grad re-read from L1 cache)
    for (uint32_t i = lane; i < D; i += 32) {
        float xi = __bfloat162float(xr[i]);
        float gi = __bfloat162float(gor[i]);
        gir[i] = __float2bfloat16(rrms * (gi - xi * coeff));
    }
}

// ── C entry points ──────────────────────────────────────────────────────────

extern "C" void fused_rms_norm_fwd(
    const void* x, void* y, uint32_t rows, uint32_t D, float eps, cudaStream_t stream)
{
    uint32_t warps_per_block = 8;  // 8 rows per block
    uint32_t threads = warps_per_block * 32;  // 256
    uint32_t blocks = (rows + warps_per_block - 1) / warps_per_block;
    fused_rms_norm_fwd_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (__nv_bfloat16*)y, rows, D, eps);
}

extern "C" void fused_rms_norm_bwd(
    const void* x, const void* grad_out, void* grad_in,
    uint32_t rows, uint32_t D, float eps, cudaStream_t stream)
{
    uint32_t warps_per_block = 8;
    uint32_t threads = warps_per_block * 32;
    uint32_t blocks = (rows + warps_per_block - 1) / warps_per_block;
    fused_rms_norm_bwd_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)grad_out,
        (__nv_bfloat16*)grad_in, rows, D, eps);
}

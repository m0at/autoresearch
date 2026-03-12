#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ── Warp reduction ──────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum_rs(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Forward: out[i] = lambda_r * x[i] + lambda_0 * x0[i]
// Vectorized: each thread processes 8 bf16 elements via uint4
__global__ void residual_scale_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ x0,
    const __nv_bfloat16* __restrict__ lambda_r_ptr,
    const __nv_bfloat16* __restrict__ lambda_0_ptr,
    __nv_bfloat16* __restrict__ out,
    int N)
{
    float lr = __bfloat162float(lambda_r_ptr[0]);
    float l0 = __bfloat162float(lambda_0_ptr[0]);

    int n_vec = N >> 3;  // number of uint4 chunks
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n_vec) {
        uint4 xv  = ((const uint4*)x)[gid];
        uint4 x0v = ((const uint4*)x0)[gid];

        const __nv_bfloat16* xp  = (const __nv_bfloat16*)&xv;
        const __nv_bfloat16* x0p = (const __nv_bfloat16*)&x0v;

        uint4 ov;
        __nv_bfloat16* op = (__nv_bfloat16*)&ov;

        #pragma unroll
        for (int j = 0; j < 8; j++)
            op[j] = __float2bfloat16(lr * __bfloat162float(xp[j]) + l0 * __bfloat162float(x0p[j]));

        ((uint4*)out)[gid] = ov;
    }

    // Handle tail elements (N not divisible by 8)
    if (gid == 0) {
        for (int i = n_vec * 8; i < N; i++)
            out[i] = __float2bfloat16(lr * __bfloat162float(x[i]) + l0 * __bfloat162float(x0[i]));
    }
}

// Backward:
//   d_x[i]  = lambda_r * grad[i]
//   d_x0[i] += lambda_0 * grad[i]   (accumulate)
//   d_lambda_r = sum(x[i] * grad[i])  (float, atomicAdd across blocks)
//   d_lambda_0 = sum(x0[i] * grad[i]) (float, atomicAdd across blocks)
// Vectorized: each thread processes 8 bf16 elements via uint4
__global__ void residual_scale_bwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ x0,
    const __nv_bfloat16* __restrict__ grad,
    const __nv_bfloat16* __restrict__ lambda_r_ptr,
    const __nv_bfloat16* __restrict__ lambda_0_ptr,
    __nv_bfloat16* __restrict__ d_x,
    __nv_bfloat16* __restrict__ d_x0,
    float* __restrict__ d_lambda_r,
    float* __restrict__ d_lambda_0,
    int N)
{
    float lr = __bfloat162float(lambda_r_ptr[0]);
    float l0 = __bfloat162float(lambda_0_ptr[0]);

    int n_vec = N >> 3;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float local_dlr = 0.0f;
    float local_dl0 = 0.0f;

    if (gid < n_vec) {
        uint4 xv   = ((const uint4*)x)[gid];
        uint4 x0v  = ((const uint4*)x0)[gid];
        uint4 gv   = ((const uint4*)grad)[gid];
        uint4 dx0v = ((const uint4*)d_x0)[gid];

        const __nv_bfloat16* xp   = (const __nv_bfloat16*)&xv;
        const __nv_bfloat16* x0p  = (const __nv_bfloat16*)&x0v;
        const __nv_bfloat16* gp   = (const __nv_bfloat16*)&gv;
        __nv_bfloat16*       dx0p = (__nv_bfloat16*)&dx0v;

        uint4 dxv;
        __nv_bfloat16* dxp = (__nv_bfloat16*)&dxv;

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float xi  = __bfloat162float(xp[j]);
            float x0i = __bfloat162float(x0p[j]);
            float gi  = __bfloat162float(gp[j]);
            dxp[j]  = __float2bfloat16(lr * gi);
            dx0p[j] = __float2bfloat16(__bfloat162float(dx0p[j]) + l0 * gi);
            local_dlr += xi * gi;
            local_dl0 += x0i * gi;
        }

        ((uint4*)d_x)[gid]  = dxv;
        ((uint4*)d_x0)[gid] = dx0v;
    }

    // Handle tail elements (thread 0 only)
    if (gid == 0) {
        for (int i = n_vec * 8; i < N; i++) {
            float xi  = __bfloat162float(x[i]);
            float x0i = __bfloat162float(x0[i]);
            float gi  = __bfloat162float(grad[i]);
            d_x[i]  = __float2bfloat16(lr * gi);
            float old_dx0 = __bfloat162float(d_x0[i]);
            d_x0[i] = __float2bfloat16(old_dx0 + l0 * gi);
            local_dlr += xi * gi;
            local_dl0 += x0i * gi;
        }
    }

    // Warp-level reduction
    local_dlr = warp_reduce_sum_rs(local_dlr);
    local_dl0 = warp_reduce_sum_rs(local_dl0);

    // Cross-warp reduction via shared memory
    __shared__ float shared_dlr[32];
    __shared__ float shared_dl0[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        shared_dlr[warp_id] = local_dlr;
        shared_dl0[warp_id] = local_dl0;
    }
    __syncthreads();

    // Thread 0 sums across warps and atomicAdds to global
    if (threadIdx.x == 0) {
        uint32_t n_warps = (blockDim.x + 31) / 32;
        float block_dlr = 0.0f;
        float block_dl0 = 0.0f;
        for (uint32_t i = 0; i < n_warps; i++) {
            block_dlr += shared_dlr[i];
            block_dl0 += shared_dl0[i];
        }
        atomicAdd(d_lambda_r, block_dlr);
        atomicAdd(d_lambda_0, block_dl0);
    }
}

extern "C" void residual_scale_fwd(
    const void* x, const void* x0,
    const void* lambda_r_ptr, const void* lambda_0_ptr,
    void* out, int N, cudaStream_t stream)
{
    int threads = 256;
    int n_vec = N >> 3;
    int blocks = (n_vec + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    if (blocks > 2048) blocks = 2048;
    residual_scale_fwd_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)x0,
        (const __nv_bfloat16*)lambda_r_ptr, (const __nv_bfloat16*)lambda_0_ptr,
        (__nv_bfloat16*)out, N);
}

extern "C" void residual_scale_bwd(
    const void* x, const void* x0, const void* grad,
    const void* lambda_r_ptr, const void* lambda_0_ptr,
    void* d_x, void* d_x0,
    float* d_lambda_r, float* d_lambda_0,
    int N, cudaStream_t stream)
{
    int threads = 256;
    int n_vec = N >> 3;
    int blocks = (n_vec + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    if (blocks > 2048) blocks = 2048;
    residual_scale_bwd_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)x0,
        (const __nv_bfloat16*)grad,
        (const __nv_bfloat16*)lambda_r_ptr, (const __nv_bfloat16*)lambda_0_ptr,
        (__nv_bfloat16*)d_x, (__nv_bfloat16*)d_x0,
        d_lambda_r, d_lambda_0, N);
}

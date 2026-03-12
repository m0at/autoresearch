#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Forward: y = cap * tanh(x / cap)
__global__ void softcap_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    float cap, int N)
{
    int vec_N = N >> 3;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_N;
         i += blockDim.x * gridDim.x) {
        uint4 xv = ((const uint4*)x)[i];
        __nv_bfloat16* xp = (__nv_bfloat16*)&xv;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float v = __bfloat162float(xp[j]);
            xp[j] = __float2bfloat16(cap * tanhf(v / cap));
        }
        ((uint4*)y)[i] = xv;
    }
    int tail_start = vec_N * 8;
    for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        float v = __bfloat162float(x[i]);
        y[i] = __float2bfloat16(cap * tanhf(v / cap));
    }
}

// Backward: dx = (1 - tanh(x/cap)^2) * grad
// NOTE: `y` is the POST-softcap value (y = cap * tanh(raw/cap)).
// We recover tanh(raw/cap) = y/cap, so the derivative is 1 - (y/cap)^2.
// This avoids needing to store pre-softcap logits.
__global__ void softcap_bwd_kernel(
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ dx,
    float cap, int N)
{
    int vec_N = N >> 3;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_N;
         i += blockDim.x * gridDim.x) {
        uint4 yv = ((const uint4*)y)[i];
        uint4 gv = ((const uint4*)grad)[i];
        __nv_bfloat16* yp = (__nv_bfloat16*)&yv;
        __nv_bfloat16* gp = (__nv_bfloat16*)&gv;
        uint4 dv;
        __nv_bfloat16* dp = (__nv_bfloat16*)&dv;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float t = __bfloat162float(yp[j]) / cap;
            float g = __bfloat162float(gp[j]);
            dp[j] = __float2bfloat16((1.0f - t * t) * g);
        }
        ((uint4*)dx)[i] = dv;
    }
    int tail_start = vec_N * 8;
    for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        float t = __bfloat162float(y[i]) / cap;
        float g = __bfloat162float(grad[i]);
        dx[i] = __float2bfloat16((1.0f - t * t) * g);
    }
}

extern "C" void softcap_fwd(
    const void* x, void* y, float cap, int N, cudaStream_t stream)
{
    int threads = 256;
    int vec_N = N >> 3;
    int blocks = (vec_N + threads - 1) / threads;
    if (blocks > 2048) blocks = 2048;
    softcap_fwd_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (__nv_bfloat16*)y, cap, N);
}

extern "C" void softcap_bwd(
    const void* x, const void* grad, void* dx, float cap, int N, cudaStream_t stream)
{
    int threads = 256;
    int vec_N = N >> 3;
    int blocks = (vec_N + threads - 1) / threads;
    if (blocks > 2048) blocks = 2048;
    softcap_bwd_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)grad,
        (__nv_bfloat16*)dx, cap, N);
}

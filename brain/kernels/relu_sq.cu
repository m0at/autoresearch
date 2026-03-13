#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Forward: y = max(0, x)^2
// Vectorized uint4 (128-bit) loads: 8 bf16 values per memory transaction.
__global__ void relu_sq_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N)
{
    const int VEC = 8;
    int n_vec = N / VEC;

    // Vectorized path: grid-stride over uint4 elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
         i += blockDim.x * gridDim.x) {
        uint4 data = ((const uint4*)x)[i];
        __nv_bfloat16* vals = (__nv_bfloat16*)&data;
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            float v = fmaxf(0.0f, __bfloat162float(vals[j]));
            vals[j] = __float2bfloat16(v * v);
        }
        ((uint4*)y)[i] = data;
    }

    // Scalar tail: elements N - (N % 8) .. N-1
    int tail_start = n_vec * VEC;
    for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        float v = fmaxf(0.0f, __bfloat162float(x[i]));
        y[i] = __float2bfloat16(v * v);
    }
}

// Backward: dx = 2 * max(0, x) * grad
// Vectorized uint4 (128-bit) loads: 8 bf16 values per memory transaction.
__global__ void relu_sq_bwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ dx,
    int N)
{
    const int VEC = 8;
    int n_vec = N / VEC;

    // Vectorized path: grid-stride over uint4 elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
         i += blockDim.x * gridDim.x) {
        uint4 xd = ((const uint4*)x)[i];
        uint4 gd = ((const uint4*)grad)[i];
        __nv_bfloat16* xv = (__nv_bfloat16*)&xd;
        __nv_bfloat16* gv = (__nv_bfloat16*)&gd;
        uint4 out;
        __nv_bfloat16* ov = (__nv_bfloat16*)&out;
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            float v = fmaxf(0.0f, __bfloat162float(xv[j]));
            float g = __bfloat162float(gv[j]);
            ov[j] = __float2bfloat16(2.0f * v * g);
        }
        ((uint4*)dx)[i] = out;
    }

    // Scalar tail
    int tail_start = n_vec * VEC;
    for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        float v = fmaxf(0.0f, __bfloat162float(x[i]));
        float g = __bfloat162float(grad[i]);
        dx[i] = __float2bfloat16(2.0f * v * g);
    }
}

extern "C" void relu_sq_fwd(
    const void* x, void* y, int N, cudaStream_t stream)
{
    int threads = 256;
    int n_vec = N / 8;
    int blocks = (n_vec + threads - 1) / threads;
    if (blocks > 2048) blocks = 2048;
    if (blocks < 1) blocks = 1;
    relu_sq_fwd_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (__nv_bfloat16*)y, N);
}

extern "C" void relu_sq_bwd(
    const void* x, const void* grad, void* dx, int N, cudaStream_t stream)
{
    int threads = 256;
    int n_vec = N / 8;
    int blocks = (n_vec + threads - 1) / threads;
    if (blocks > 2048) blocks = 2048;
    if (blocks < 1) blocks = 1;
    relu_sq_bwd_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)grad,
        (__nv_bfloat16*)dx, N);
}

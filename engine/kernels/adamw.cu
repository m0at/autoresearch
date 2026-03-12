#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// Fused AdamW step for BF16 params/grads/moments.
// All arithmetic in F32, stores back to BF16.
__global__ void adamw_step_bf16_kernel(
    __nv_bfloat16* __restrict__ params,
    const __nv_bfloat16* __restrict__ grads,
    __nv_bfloat16* __restrict__ exp_avg,
    __nv_bfloat16* __restrict__ exp_avg_sq,
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2,
    int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    float p = __bfloat162float(params[gid]);
    float g = __bfloat162float(grads[gid]);
    float m = __bfloat162float(exp_avg[gid]);
    float v = __bfloat162float(exp_avg_sq[gid]);

    // Decoupled weight decay
    p -= lr * wd * p;

    // Momentum update
    m = beta1 * m + (1.0f - beta1) * g;

    // Second moment update
    v = beta2 * v + (1.0f - beta2) * g * g;

    // Bias-corrected estimates
    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;

    // Parameter update
    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    params[gid]     = __float2bfloat16(p);
    exp_avg[gid]    = __float2bfloat16(m);
    exp_avg_sq[gid] = __float2bfloat16(v);
}

// F32 version for scalar params (resid_lambdas, x0_lambdas).
__global__ void adamw_step_f32_kernel(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2,
    int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    float p = params[gid];
    float g = grads[gid];
    float m = exp_avg[gid];
    float v = exp_avg_sq[gid];

    p -= lr * wd * p;
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;
    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    params[gid]     = p;
    exp_avg[gid]    = m;
    exp_avg_sq[gid] = v;
}

extern "C" void adamw_step_bf16(
    void* params, const void* grads,
    void* exp_avg, void* exp_avg_sq,
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2,
    int N, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    adamw_step_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)params, (const __nv_bfloat16*)grads,
        (__nv_bfloat16*)exp_avg, (__nv_bfloat16*)exp_avg_sq,
        lr, beta1, beta2, eps, wd,
        bias_correction1, bias_correction2, N);
}

extern "C" void adamw_step_f32(
    float* params, const float* grads,
    float* exp_avg, float* exp_avg_sq,
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2,
    int N, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    adamw_step_f32_kernel<<<blocks, threads, 0, stream>>>(
        params, grads, exp_avg, exp_avg_sq,
        lr, beta1, beta2, eps, wd,
        bias_correction1, bias_correction2, N);
}

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ── Fused RoPE forward ──────────────────────────────────────────────────────
// Grid-stride loop with in-place safety (__syncthreads between reads/writes).
// Each iteration processes blockDim elements = complete heads (hdim divides blockDim).

__global__ void fused_rope_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_t,
    const __nv_bfloat16* __restrict__ sin_t,
    __nv_bfloat16* __restrict__ out,
    uint32_t N, uint32_t T, uint32_t n_head, uint32_t hdim)
{
    uint32_t half_d = hdim / 2;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t base = blockIdx.x * blockDim.x; base < N; base += stride) {
        uint32_t gid = base + threadIdx.x;
        bool active = gid < N;

        float v1 = 0.f, v2 = 0.f, c = 1.f, s = 0.f;
        uint32_t d = 0;
        if (active) {
            d = gid % hdim;
            uint32_t t = (gid / hdim / n_head) % T;
            if (d < half_d) {
                uint32_t cs_idx = t * half_d + d;
                c = __bfloat162float(cos_t[cs_idx]);
                s = __bfloat162float(sin_t[cs_idx]);
                v1 = __bfloat162float(x[gid]);
                v2 = __bfloat162float(x[gid + half_d]);
            } else {
                uint32_t d2 = d - half_d;
                uint32_t cs_idx = t * half_d + d2;
                c = __bfloat162float(cos_t[cs_idx]);
                s = __bfloat162float(sin_t[cs_idx]);
                v1 = __bfloat162float(x[gid - half_d]);
                v2 = __bfloat162float(x[gid]);
            }
        }

        __syncthreads();

        if (active) {
            if (d < half_d)
                out[gid] = __float2bfloat16(v1 * c + v2 * s);
            else
                out[gid] = __float2bfloat16(-v1 * s + v2 * c);
        }

        __syncthreads();
    }
}

// ── Fused RoPE backward ─────────────────────────────────────────────────────
// In-place safe: reads both partner elements before __syncthreads, then writes.
// Same pattern as the forward kernel.

__global__ void fused_rope_bwd_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ cos_t,
    const __nv_bfloat16* __restrict__ sin_t,
    __nv_bfloat16* __restrict__ grad_in,
    uint32_t N, uint32_t T, uint32_t n_head, uint32_t hdim)
{
    uint32_t half_d = hdim / 2;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t base = blockIdx.x * blockDim.x; base < N; base += stride) {
        uint32_t gid = base + threadIdx.x;
        bool active = gid < N;

        float result = 0.f;
        if (active) {
            uint32_t d = gid % hdim;
            uint32_t t = (gid / hdim / n_head) % T;

            if (d < half_d) {
                uint32_t cs_idx = t * half_d + d;
                float c = __bfloat162float(cos_t[cs_idx]);
                float s = __bfloat162float(sin_t[cs_idx]);
                float g1 = __bfloat162float(grad_out[gid]);
                float g2 = __bfloat162float(grad_out[gid + half_d]);
                result = g1 * c - g2 * s;
            } else {
                uint32_t d2 = d - half_d;
                uint32_t cs_idx = t * half_d + d2;
                float c = __bfloat162float(cos_t[cs_idx]);
                float s = __bfloat162float(sin_t[cs_idx]);
                float g1 = __bfloat162float(grad_out[gid - half_d]);
                float g2 = __bfloat162float(grad_out[gid]);
                result = g1 * s + g2 * c;
            }
        }

        __syncthreads();

        if (active) {
            grad_in[gid] = __float2bfloat16(result);
        }

        __syncthreads();
    }
}

// ── C entry points ──────────────────────────────────────────────────────────

extern "C" void fused_rope_fwd(
    const void* x, const void* cos_t, const void* sin_t, void* out,
    uint32_t N, uint32_t T, uint32_t n_head, uint32_t hdim, cudaStream_t stream)
{
    uint32_t threads = 512;
    uint32_t blocks = (N + threads - 1) / threads;
    if (blocks > 2048) blocks = 2048;
    fused_rope_fwd_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)cos_t,
        (const __nv_bfloat16*)sin_t, (__nv_bfloat16*)out,
        N, T, n_head, hdim);
}

extern "C" void fused_rope_bwd(
    const void* grad_out, const void* cos_t, const void* sin_t, void* grad_in,
    uint32_t N, uint32_t T, uint32_t n_head, uint32_t hdim, cudaStream_t stream)
{
    uint32_t threads = 512;
    uint32_t blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    fused_rope_bwd_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)grad_out, (const __nv_bfloat16*)cos_t,
        (const __nv_bfloat16*)sin_t, (__nv_bfloat16*)grad_in,
        N, T, n_head, hdim);
}

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

// ── Warp reduction helpers ──────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();
    uint32_t n_warps = (blockDim.x + 31) / 32;
    float total = 0.0f;
    for (uint32_t i = 0; i < n_warps; i++) total += shared[i];
    return total;
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();
    uint32_t n_warps = (blockDim.x + 31) / 32;
    float m = -CUDART_INF_F;
    for (uint32_t i = 0; i < n_warps; i++) m = fmaxf(m, shared[i]);
    return m;
}

// Helper: apply softcap in f32. cap=0 means no softcap.
__device__ __forceinline__ float apply_softcap(float v, float cap) {
    return (cap > 0.0f) ? cap * tanhf(v / cap) : v;
}

// ── Cross-entropy forward (with fused softcap) ─────────────────────────────
// Caches softcapped values in shared memory to avoid redundant tanhf calls.
// Pass 1: softcap + cache + find max.  Pass 2: read cache + sum exp.
// Eliminates 1 tanhf per vocab element vs the naive two-pass approach.
//
// Shared memory layout: [V floats for softcap cache | 32 floats for reduction]
// V=8192 → 32KB cache + 128B reduction = ~32.1KB total.

__global__ void fused_cross_entropy_fwd_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const uint32_t* __restrict__ targets,
    float* __restrict__ losses,
    float* __restrict__ loss_sum,
    uint32_t V, float softcap)
{
    // Dynamic shared memory: first V floats = softcap cache, last 32 = reduction buf.
    extern __shared__ float smem[];
    float* s_cached = smem;           // [V] softcapped values
    float* shared   = smem + V;       // [32] warp reduction scratch

    const __nv_bfloat16* xr = logits + (uint64_t)blockIdx.x * V;
    uint32_t tgt = targets[blockIdx.x];

    // pass 1: softcap once, cache result, track max
    float local_max = -CUDART_INF_F;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
        float v = apply_softcap(__bfloat162float(xr[i]), softcap);
        s_cached[i] = v;
        local_max = fmaxf(local_max, v);
    }
    float m = block_reduce_max(local_max, shared);
    __syncthreads();  // ensure s_cached writes visible to all threads

    // pass 2: sum of exp(cached_softcap - max) — no tanhf here
    float local_sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
        local_sum += expf(s_cached[i] - m);
    }
    float total = block_reduce_sum(local_sum, shared);

    // loss = -(softcap(logit[target]) - max) + log(sum_exp)
    if (threadIdx.x == 0) {
        float logit_tgt = s_cached[tgt];   // cached — no extra tanhf
        float loss = -(logit_tgt - m) + logf(total);
        if (losses) losses[blockIdx.x] = loss;
        if (loss_sum) atomicAdd(loss_sum, loss);
    }
}

// ── Cross-entropy backward (with fused softcap) ────────────────────────────
// Same caching trick: softcap computed once in pass 1, reused in passes 2+3.
// Eliminates 2 tanhf per vocab element vs the naive three-pass approach.

__global__ void fused_cross_entropy_bwd_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const uint32_t* __restrict__ targets,
    const float* __restrict__ grad_res,
    __nv_bfloat16* __restrict__ grad_in,
    uint32_t V, float softcap)
{
    extern __shared__ float smem[];
    float* s_cached = smem;
    float* shared   = smem + V;

    const __nv_bfloat16* xr = logits + (uint64_t)blockIdx.x * V;
    __nv_bfloat16* gr = grad_in + (uint64_t)blockIdx.x * V;
    uint32_t tgt = targets[blockIdx.x];
    float scale = grad_res[blockIdx.x];

    // pass 1: softcap once, cache, find max
    float local_max = -CUDART_INF_F;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
        float v = apply_softcap(__bfloat162float(xr[i]), softcap);
        s_cached[i] = v;
        local_max = fmaxf(local_max, v);
    }
    float m = block_reduce_max(local_max, shared);
    __syncthreads();

    // pass 2: sum exp — no tanhf
    float local_sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
        local_sum += expf(s_cached[i] - m);
    }
    float total = block_reduce_sum(local_sum, shared);
    float inv_sum = 1.0f / total;
    __syncthreads();

    // pass 3: write grad — reads from cache, no tanhf needed for s or its derivative
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
        float s = s_cached[i];   // cached softcapped value
        float p = expf(s - m) * inv_sum;
        float indicator = (i == tgt) ? 1.0f : 0.0f;
        float ce_grad = (p - indicator) * scale;
        // softcap derivative: 1 - (s/cap)^2  (s already = cap*tanh(x/cap))
        float softcap_deriv = (softcap > 0.0f) ? (1.0f - (s / softcap) * (s / softcap)) : 1.0f;
        gr[i] = __float2bfloat16(ce_grad * softcap_deriv);
    }
}

// ── C entry points ──────────────────────────────────────────────────────────
// smem = V floats (softcap cache) + 32 floats (reduction) = (V+32)*4 bytes.

extern "C" void fused_cross_entropy_fwd(
    const void* logits, const void* targets, void* losses,
    void* loss_sum, uint32_t N, uint32_t V, float softcap, cudaStream_t stream)
{
    dim3 grid(N);
    dim3 block(256);
    size_t smem = (V + 32) * sizeof(float);
    fused_cross_entropy_fwd_kernel<<<grid, block, smem, stream>>>(
        (const __nv_bfloat16*)logits, (const uint32_t*)targets,
        (float*)losses, (float*)loss_sum, V, softcap);
}

extern "C" void fused_cross_entropy_bwd(
    const void* logits, const void* targets, const void* grad_res,
    void* grad_in, uint32_t N, uint32_t V, float softcap, cudaStream_t stream)
{
    dim3 grid(N);
    dim3 block(256);
    size_t smem = (V + 32) * sizeof(float);
    fused_cross_entropy_bwd_kernel<<<grid, block, smem, stream>>>(
        (const __nv_bfloat16*)logits, (const uint32_t*)targets,
        (const float*)grad_res, (__nv_bfloat16*)grad_in, V, softcap);
}

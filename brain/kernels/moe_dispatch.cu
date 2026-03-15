// moe_dispatch.cu — MoE token dispatch kernels for top-2 routing
// All inputs/outputs bf16 unless noted. Intermediates in f32.
// N_EXPERTS=8, TOP_K=2, D_MODEL=1024

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>          // Driver API: cuMemAlloc, cuMemFree, cuMemcpyDtoH, cuMemcpyDtoDAsync
#include <stdint.h>
using bf16 = nv_bfloat16;

// ─── Constants ───────────────────────────────────────────────────────────────

static constexpr int N_EXPERTS = 8;
static constexpr int TOP_K    = 2;
static constexpr int D_MODEL  = 1024;

// ─── Static cuMemAlloc buffers (cudarc pool allocs can't be zeroed) ──────────
// Per-GPU arrays to avoid cross-device illegal address on multi-GPU
#define MAX_GPUS 8
static int* s_expert_counts[MAX_GPUS]  = {};
static int* s_expert_offsets[MAX_GPUS] = {};
static int* s_write_scratch[MAX_GPUS]  = {};

// D_MODEL in bf16 = 2048 bytes = 128 uint4 (16 bytes each)
static constexpr int VEC_PER_TOKEN = D_MODEL * sizeof(bf16) / sizeof(uint4); // 128


// ─── 1. moe_router_softmax_topk ─────────────────────────────────────────────
// Fused softmax over 8 experts + top-2 selection with renormalized gates.
//
// Grid:  (BT, 1, 1)
// Block: (32, 1, 1)  — one warp per token, 8 active lanes
//
// Input:  router_logits [BT, 8] bf16
// Output: probs         [BT, 8] f32  (full softmax for aux loss)
//         gates         [BT, 2] f32  (renormalized top-2 probs)
//         indices       [BT, 2] i32  (expert indices for top-2)

extern "C" __global__ void moe_router_softmax_topk(
    const bf16* __restrict__ router_logits,  // [BT, 8]
    float*      __restrict__ probs,          // [BT, 8]
    float*      __restrict__ gates,          // [BT, 2]
    int*        __restrict__ indices,        // [BT, 2]
    int bt)
{
    int token = blockIdx.x;
    if (token >= bt) return;
    int lane = threadIdx.x; // 0..31, only 0..7 active for loads

    // Load logits — lanes 0..7 each load one expert, lanes 8..31 get -inf
    float val = (lane < N_EXPERTS)
        ? __bfloat162float(router_logits[token * N_EXPERTS + lane])
        : -1e30f;

    // Warp-wide max for numerical stability (all 32 lanes participate)
    float max_val = val;
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));

    // exp(x - max)
    float exp_val = (lane < N_EXPERTS) ? expf(val - max_val) : 0.0f;

    // Warp-wide sum
    float sum_exp = exp_val;
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1)
        sum_exp += __shfl_xor_sync(0xFFFFFFFF, sum_exp, offset);

    float prob = exp_val / sum_exp;

    // Store full probs (lanes 0..7)
    if (lane < N_EXPERTS)
        probs[token * N_EXPERTS + lane] = prob;

    // ── Top-2 selection via warp shuffles ──
    // Pack (prob, lane_id) and do two passes of argmax

    // First top-1: warp-wide argmax
    float best1_val = prob;
    int   best1_idx = lane;

    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xFFFFFFFF, best1_val, offset);
        int   other_idx = __shfl_xor_sync(0xFFFFFFFF, best1_idx, offset);
        if (other_val > best1_val || (other_val == best1_val && other_idx < best1_idx)) {
            best1_val = other_val;
            best1_idx = other_idx;
        }
    }
    // Now all lanes agree on best1_val, best1_idx

    // Second top-2: mask out the winner, repeat argmax
    float masked_prob = (lane == best1_idx) ? -1.0f : prob;
    float best2_val = masked_prob;
    int   best2_idx = lane;

    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xFFFFFFFF, best2_val, offset);
        int   other_idx = __shfl_xor_sync(0xFFFFFFFF, best2_idx, offset);
        if (other_val > best2_val || (other_val == best2_val && other_idx < best2_idx)) {
            best2_val = other_val;
            best2_idx = other_idx;
        }
    }

    // Renormalize gates: gate_k = prob_k / (prob_top1 + prob_top2)
    float gate_sum = best1_val + best2_val;
    float rcp_gate_sum = 1.0f / gate_sum;

    // Lane 0 writes output
    if (lane == 0) {
        gates[token * TOP_K + 0]   = best1_val * rcp_gate_sum;
        gates[token * TOP_K + 1]   = best2_val * rcp_gate_sum;
        indices[token * TOP_K + 0] = best1_idx;
        indices[token * TOP_K + 1] = best2_idx;
    }
}

// Host launcher
extern "C" void launch_moe_router_softmax_topk(
    const void* router_logits,
    void*       probs,
    void*       gates,
    void*       indices,
    int bt,
    cudaStream_t stream)
{
    dim3 grid(bt);
    dim3 block(32); // one warp per token
    moe_router_softmax_topk<<<grid, block, 0, stream>>>(
        (const bf16*)router_logits,
        (float*)probs,
        (float*)gates,
        (int*)indices,
        bt);
}


// ─── 2. moe_permute_tokens ──────────────────────────────────────────────────
// Build dispatch tables: sort token slots by expert.
//
// Two-pass approach:
//   Pass 1: atomicAdd to count tokens per expert
//   Pass 2: prefix sum on expert_counts → expert_offsets, then scatter
//
// We split this into two kernels + a tiny serial prefix-sum kernel.

// Pass 1: count tokens per expert
// Grid:  (ceil(BT*TOP_K / 256), 1, 1)
// Block: (256, 1, 1)
extern "C" __global__ void moe_permute_count(
    const int* __restrict__ expert_indices,  // [BT, TOP_K] flattened = BT*TOP_K
    int*       __restrict__ expert_counts,   // [N_EXPERTS], zeroed before launch
    int total_slots)                         // BT * TOP_K
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;
    int expert = expert_indices[slot];
    atomicAdd(&expert_counts[expert], 1);
}

// Prefix sum: single-warp kernel, N_EXPERTS=8 elements
// Grid:  (1, 1, 1)
// Block: (32, 1, 1)  — only lane 0 does work (8 elements, not worth parallelizing)
extern "C" __global__ void moe_prefix_sum(
    const int* __restrict__ expert_counts,   // [N_EXPERTS]
    int*       __restrict__ expert_offsets)   // [N_EXPERTS + 1]
{
    if (threadIdx.x != 0) return;
    int running = 0;
    expert_offsets[0] = 0;
    #pragma unroll
    for (int e = 0; e < N_EXPERTS; e++) {
        running += expert_counts[e];
        expert_offsets[e + 1] = running;
    }
}

// Pass 2: scatter token indices into permutation array
// Each slot atomically claims a position within its expert's range.
// Grid:  (ceil(BT*TOP_K / 256), 1, 1)
// Block: (256, 1, 1)
//
// We use a scratch copy of expert_counts (zeroed) as running write pointers.
extern "C" __global__ void moe_permute_scatter(
    const int* __restrict__ expert_indices,   // [BT * TOP_K]
    const int* __restrict__ expert_offsets,    // [N_EXPERTS + 1]
    int*       __restrict__ write_counters,    // [N_EXPERTS], zeroed before launch
    int*       __restrict__ token_perm,        // [BT * TOP_K]
    int total_slots)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;
    int expert = expert_indices[slot];
    int pos = expert_offsets[expert] + atomicAdd(&write_counters[expert], 1);
    token_perm[pos] = slot;
}

// Zero N ints — replaces cudaMemsetAsync which fails with cuMemAllocAsync pointers
__global__ void moe_zero_ints(int* data, int n) {
    int i = threadIdx.x;
    if (i < n) data[i] = 0;
}

// Combined host launcher for moe_permute_tokens
// Caller must provide:
//   - expert_counts  [N_EXPERTS]     (will be zeroed internally)
//   - expert_offsets [N_EXPERTS + 1]
//   - write_scratch  [N_EXPERTS]     (scratch for scatter pass write counters)
//   - token_perm     [BT * TOP_K]
extern "C" void launch_moe_permute_tokens(
    const void* expert_indices,
    void*       token_perm,
    void*       expert_counts,
    void*       expert_offsets,
    void*       write_scratch,
    int bt,
    cudaStream_t stream)
{
    int dev; cudaGetDevice(&dev);

    // Allocate static buffers on first call via driver API (cudarc pool allocs
    // cannot be zeroed by any memset API — only driver-allocated buffers work)
    if (!s_expert_counts[dev]) {
        CUdeviceptr tmp;
        cuMemAlloc(&tmp, N_EXPERTS * sizeof(int));
        s_expert_counts[dev] = (int*)tmp;
        cuMemAlloc(&tmp, (N_EXPERTS + 1) * sizeof(int));
        s_expert_offsets[dev] = (int*)tmp;
        cuMemAlloc(&tmp, N_EXPERTS * sizeof(int));
        s_write_scratch[dev] = (int*)tmp;
    }

    int total_slots = bt * TOP_K;
    int threads = 256;
    int blocks = (total_slots + threads - 1) / threads;

    // Zero static buffers via kernel launch (memset APIs fail on cudarc pool memory)
    moe_zero_ints<<<1, N_EXPERTS, 0, stream>>>(s_expert_counts[dev], N_EXPERTS);
    moe_zero_ints<<<1, N_EXPERTS, 0, stream>>>(s_write_scratch[dev], N_EXPERTS);

    // Pass 1: count
    moe_permute_count<<<blocks, threads, 0, stream>>>(
        (const int*)expert_indices,
        s_expert_counts[dev],
        total_slots);

    // Prefix sum
    moe_prefix_sum<<<1, 32, 0, stream>>>(
        s_expert_counts[dev],
        s_expert_offsets[dev]);

    // Pass 2: scatter
    moe_permute_scatter<<<blocks, threads, 0, stream>>>(
        (const int*)expert_indices,
        (const int*)s_expert_offsets[dev],
        s_write_scratch[dev],
        (int*)token_perm,
        total_slots);

    // Copy results back to caller's buffers (may be pool-allocated)
    cuMemcpyDtoDAsync((CUdeviceptr)expert_offsets, (CUdeviceptr)s_expert_offsets[dev],
                      (N_EXPERTS + 1) * sizeof(int), stream);
    cuMemcpyDtoDAsync((CUdeviceptr)expert_counts, (CUdeviceptr)s_expert_counts[dev],
                      N_EXPERTS * sizeof(int), stream);
}


// ─── 3. moe_gather_tokens ───────────────────────────────────────────────────
// Gather: x_expert[i] = x[perm[i] / TOP_K]  (perm stores slot indices into
// the [BT, TOP_K] table; the source token is slot / TOP_K).
//
// Vectorized with uint4: D_MODEL=1024 bf16 = 2048 bytes = 128 uint4.
//
// Grid:  (n_dispatch, 1, 1)   where n_dispatch = BT * TOP_K
// Block: (128, 1, 1)          one thread per uint4

extern "C" __global__ void moe_gather_tokens(
    const bf16* __restrict__ x,           // [BT, D_MODEL]
    const int*  __restrict__ token_perm,  // [n_dispatch]
    bf16*       __restrict__ x_gathered,  // [n_dispatch, D_MODEL]
    int n_dispatch)
{
    int dispatch_idx = blockIdx.x;
    if (dispatch_idx >= n_dispatch) return;

    int slot = token_perm[dispatch_idx];
    int src_token = slot / TOP_K;  // map slot back to token index

    int vec_lane = threadIdx.x; // 0..127

    const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)src_token * D_MODEL);
    uint4*       dst = reinterpret_cast<uint4*>(x_gathered + (size_t)dispatch_idx * D_MODEL);

    dst[vec_lane] = src[vec_lane];
}

// Host launcher
extern "C" void launch_moe_gather_tokens(
    const void* x,
    const void* token_perm,
    void*       x_gathered,
    int n_dispatch,
    cudaStream_t stream)
{
    dim3 grid(n_dispatch);
    dim3 block(VEC_PER_TOKEN); // 128
    moe_gather_tokens<<<grid, block, 0, stream>>>(
        (const bf16*)x,
        (const int*)token_perm,
        (bf16*)x_gathered,
        n_dispatch);
}


// ─── 4. moe_scatter_k ──────────────────────────────────────────────────────
// Two-pass scatter for combining expert outputs back to token positions.
//
// Pass 1 (k=0, beta=0): output[token] = gate * expert_out[dispatch_idx]
// Pass 2 (k=1, beta=1): output[token] += gate * expert_out[dispatch_idx]
//
// Each token appears exactly once per k-slot, so no write conflicts within
// a single pass. The k parameter selects which top-k slot we're scattering.
//
// Grid:  (n_tokens_this_pass, 1, 1)
// Block: (256, 1, 1)  — each thread handles D_MODEL/256 = 4 elements
//
// n_tokens_this_pass = number of dispatch entries with the given k-index.
// For simplicity, we iterate all dispatch entries and filter by k.
//
// Actually, cleaner: process ALL dispatch entries, use beta to decide
// write vs accumulate based on whether this token has been written before.
// But the plan says two-pass by k-slot. Let's do that.

// Single-pass kernel parameterized by beta (0.0 = write, 1.0 = accumulate)
// Grid:  (BT, 1, 1)
// Block: (256, 1, 1)
//
// For k-th pass, we process expert_indices[:, k] and gates[:, k].
// expert_out is indexed via dispatch table: for token t with k-th expert,
// the dispatched output lives at the position in the permuted array.
//
// Simpler approach: iterate over dispatch slots belonging to k-th selection.
// The token_perm array is sorted by expert. We need to map back from
// (token, k) to dispatch position. Instead, iterate all BT tokens,
// look up their k-th dispatch slot, find it in expert_out.
//
// Actually, the cleanest: for each token, we know indices[token, k] and
// gates[token, k]. The expert_out for this token's k-th selection is at
// the position where token_perm maps it. We need the inverse mapping.
//
// Let's use a direct approach: iterate BT tokens, for k-th pass,
// the dispatch slot is (token * TOP_K + k). We need to know where
// slot (token*TOP_K + k) ended up in the permuted array (i.e., which
// position in x_gathered corresponds to this slot).
//
// We need an inverse permutation: inv_perm[slot] = dispatch_position.
// The gather kernel already gathered x[slot/TOP_K] into x_gathered[dispatch_pos]
// where token_perm[dispatch_pos] = slot.
// So inv_perm is the inverse of token_perm.
//
// Rather than building inv_perm, let's restructure: process dispatch entries
// directly (not tokens), and use beta to handle the write/accumulate.
//
// For each dispatch entry d (0..BT*TOP_K-1):
//   slot = token_perm[d]
//   token = slot / TOP_K
//   k = slot % TOP_K
//   gate = gates[token * TOP_K + k]
//   output[token] = beta * output[token] + gate * expert_out[d]
//
// Problem: two dispatch entries can map to the same token (k=0 and k=1),
// and they might execute concurrently → race condition.
//
// Solution: two-pass. Filter by k.
//   Pass with beta=0: process only entries where slot%TOP_K == 0
//   Pass with beta=1: process only entries where slot%TOP_K == 1
//
// But dispatch entries for the same token with different k values might not
// be adjacent. We'd need to iterate all entries and skip non-matching ones.
//
// Better: just iterate BT tokens per pass. For pass k:
//   slot = token * TOP_K + k
//   We need to know where this slot is in the permuted expert_out.
//   This requires inv_perm[slot] → position in expert_out.
//
// Let's build the inverse permutation in the permute step. It's cheap.

// Kernel to build inverse permutation
// Grid:  (ceil(n_dispatch / 256), 1, 1)
// Block: (256, 1, 1)
extern "C" __global__ void moe_build_inv_perm(
    const int* __restrict__ token_perm,    // [n_dispatch]
    int*       __restrict__ inv_perm,      // [n_dispatch]: inv_perm[token_perm[i]] = i
    int n_dispatch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dispatch) return;
    inv_perm[token_perm[i]] = i;
}

extern "C" void launch_moe_build_inv_perm(
    const void* token_perm,
    void*       inv_perm,
    int n_dispatch,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n_dispatch + threads - 1) / threads;
    moe_build_inv_perm<<<blocks, threads, 0, stream>>>(
        (const int*)token_perm,
        (int*)inv_perm,
        n_dispatch);
}


// Scatter kernel: one block per token, 256 threads, each handles 4 bf16 elements.
// D_MODEL=1024, 256 threads → 4 elements/thread.
//
// Grid:  (BT, 1, 1)
// Block: (256, 1, 1)
//
// beta=0: write.  beta=1: accumulate.
extern "C" __global__ void moe_scatter_k(
    const bf16*  __restrict__ expert_out,   // [BT*TOP_K, D_MODEL]
    const float* __restrict__ gates,        // [BT, TOP_K]
    const int*   __restrict__ inv_perm,     // [BT * TOP_K]
    bf16*        __restrict__ output,       // [BT, D_MODEL]
    int bt,
    int k,       // which top-k slot (0 or 1)
    float beta)  // 0.0 = write, 1.0 = accumulate
{
    int token = blockIdx.x;
    if (token >= bt) return;

    int slot = token * TOP_K + k;
    int dispatch_pos = inv_perm[slot];
    float gate = gates[token * TOP_K + k];

    // Each thread processes 4 bf16 elements (1024 / 256 = 4)
    int tid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int col = tid * 4 + i;
        float e = __bfloat162float(expert_out[(size_t)dispatch_pos * D_MODEL + col]);
        float scaled = gate * e;

        if (beta == 0.0f) {
            output[(size_t)token * D_MODEL + col] = __float2bfloat16(scaled);
        } else {
            float prev = __bfloat162float(output[(size_t)token * D_MODEL + col]);
            output[(size_t)token * D_MODEL + col] = __float2bfloat16(prev + scaled);
        }
    }
}

// Host launcher — calls scatter for k=0 (write) then k=1 (accumulate)
extern "C" void launch_moe_scatter(
    const void* expert_out,
    const void* gates,
    const void* inv_perm,
    void*       output,
    int bt,
    cudaStream_t stream)
{
    dim3 grid(bt);
    dim3 block(256);

    // Pass 1: k=0, beta=0 (write)
    moe_scatter_k<<<grid, block, 0, stream>>>(
        (const bf16*)expert_out,
        (const float*)gates,
        (const int*)inv_perm,
        (bf16*)output,
        bt, 0, 0.0f);

    // Pass 2: k=1, beta=1 (accumulate)
    moe_scatter_k<<<grid, block, 0, stream>>>(
        (const bf16*)expert_out,
        (const float*)gates,
        (const int*)inv_perm,
        (bf16*)output,
        bt, 1, 1.0f);
}

// Copy expert_offsets from device to host — must be called from the same
// runtime context as launch_moe_permute_tokens.
extern "C" void moe_copy_offsets_to_host(
    const void* expert_offsets_dev,
    void*       expert_offsets_host,
    cudaStream_t stream)
{
    int dev; cudaGetDevice(&dev);
    // Use BOTH runtime and driver sync to ensure all pending work completes
    cudaStreamSynchronize(stream);
    cuStreamSynchronize(stream);
    // Read from static cuMemAlloc buffer (the caller's pointer may be pool-allocated
    // and unreliable for DtoH copies)
    const void* src = s_expert_offsets[dev] ? (const void*)s_expert_offsets[dev] : expert_offsets_dev;
    cuMemcpyDtoH(expert_offsets_host, (CUdeviceptr)src,
                 (N_EXPERTS + 1) * sizeof(int));
}

// Force kernel registration on current device and pre-allocate per-GPU buffers
extern "C" void moe_dispatch_init() {
    int dev; cudaGetDevice(&dev);
    CUdeviceptr dummy;
    cuMemAlloc(&dummy, sizeof(int));
    moe_zero_ints<<<1, 1, 0, 0>>>((int*)dummy, 1);
    cudaDeviceSynchronize();
    cuMemFree(dummy);

    // Pre-allocate per-GPU static buffers
    if (!s_expert_counts[dev]) {
        CUdeviceptr tmp;
        cuMemAlloc(&tmp, N_EXPERTS * sizeof(int));
        s_expert_counts[dev] = (int*)tmp;
        cuMemAlloc(&tmp, (N_EXPERTS + 1) * sizeof(int));
        s_expert_offsets[dev] = (int*)tmp;
        cuMemAlloc(&tmp, N_EXPERTS * sizeof(int));
        s_write_scratch[dev] = (int*)tmp;
    }
}

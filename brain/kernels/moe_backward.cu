#include <cuda.h>
// MoE backward kernels for dispatch operations.
// D_MODEL=1024, N_EXPERTS=8, TOP_K=2.
// bf16 for tensor I/O, f32 for all reductions and intermediate math.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

// ============================================================================
// Kernel 1: moe_router_softmax_topk_bwd
//
// Backward through softmax + top-k routing.
// Combines d_gates (from expert output backward) and d_probs (from aux loss)
// to produce d_router_logits.
//
// Math:
//   For each token t, experts e in {0..7}:
//     sum_sel = Σ_{k=0,1} gate[t,k] * d_gate[t,k]   (dot over selected experts)
//     d_softmax[t,e] = probs[t,e] * (d_selected[t,e] - sum_sel)
//       where d_selected[t,e] = d_gate[t,k] if expert e was selected as slot k, else 0
//     d_logits[t,e] = d_softmax[t,e] + d_probs[t,e]  (aux loss grad added pre-softmax-bwd)
//
//   Actually: aux loss gradient goes through softmax backward too.
//   Full derivation:
//     The upstream gradient on probs has two sources:
//       1) top-k selection: d_probs_from_gates[t,e] = d_gate[t,k] * (1/renorm) stuff
//       2) aux loss: d_probs_aux[t,e]
//     Since top-k with renormalization:
//       gate[k] = probs[idx[k]] / (probs[idx[0]] + probs[idx[1]])
//     Let S = probs[idx[0]] + probs[idx[1]].
//     d_probs[t, idx[k]] += d_gate[k] * (S - probs[idx[k]]) / S^2
//                           + Σ_{j≠k} d_gate[j] * (-probs[idx[j]]) / S^2
//     Simplification: d_probs[t, idx[k]] += (d_gate[k] - gate[k] * Σ_j gate[j]*d_gate[j]) / S
//     Wait — with renormalization, gate[k] = probs[idx[k]] / S, so:
//       d_probs[idx[k]] = (d_gate[k] - gate[k] * (gate[0]*d_gate[0] + gate[1]*d_gate[1])) / S
//     This is just softmax-backward on the 2-element selected subset.
//
//   Then softmax backward on the full 8-way softmax:
//     d_logits[e] = probs[e] * (d_probs_total[e] - Σ_j probs[j] * d_probs_total[j])
//
// One warp (32 threads) per token. Only 8 experts, so lanes 0-7 are active.
// ============================================================================

__global__ void __launch_bounds__(256)
moe_router_softmax_topk_bwd_kernel(
    const float* __restrict__ probs,           // [BT, 8] f32
    const float* __restrict__ gates,           // [BT, 2] f32 (renormalized)
    const int*   __restrict__ indices,         // [BT, 2] i32
    const float* __restrict__ d_gates,         // [BT, 2] f32
    const float* __restrict__ d_probs_aux,     // [BT, 8] f32 (from aux loss)
    bf16*        __restrict__ d_router_logits, // [BT, 8] bf16
    int bt)
{
    // 8 warps per block, each warp handles one token
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int token = blockIdx.x * 8 + warp_id; // 8 warps/block
    if (token >= bt || lane >= 8) return;

    // Load this token's routing data
    const float p = probs[token * 8 + lane];
    const int idx0 = indices[token * 2 + 0];
    const int idx1 = indices[token * 2 + 1];
    const float g0 = gates[token * 2 + 0];
    const float g1 = gates[token * 2 + 1];
    const float dg0 = d_gates[token * 2 + 0];
    const float dg1 = d_gates[token * 2 + 1];

    // Step 1: Backward through renormalization (2-element softmax on selected probs)
    // S = probs[idx0] + probs[idx1]
    // gate[k] = probs[idx[k]] / S
    // d_probs[idx[k]] = (d_gate[k] - gate[k] * dot(gate, d_gate)) / S
    float dot_gd = g0 * dg0 + g1 * dg1;
    float S = __shfl_sync(0xFF, p, idx0) + __shfl_sync(0xFF, p, idx1);

    float d_probs_routing = 0.0f;
    if (lane == idx0) {
        d_probs_routing = (dg0 - g0 * dot_gd) / S;
    }
    if (lane == idx1) {
        d_probs_routing = (dg1 - g1 * dot_gd) / S;
    }

    // Step 2: Combine routing grad with aux loss grad
    float d_probs_total = d_probs_routing + d_probs_aux[token * 8 + lane];

    // Step 3: Backward through 8-way softmax
    // d_logits[e] = probs[e] * (d_probs_total[e] - Σ_j probs[j] * d_probs_total[j])
    float pd = p * d_probs_total;
    // Reduce pd across lanes 0-7
    #pragma unroll
    for (int offset = 4; offset >= 1; offset >>= 1) {
        pd += __shfl_xor_sync(0xFF, pd, offset);
    }
    // pd now holds Σ_j probs[j] * d_probs_total[j] in all 8 lanes

    float d_logit = p * (d_probs_total - pd);

    d_router_logits[token * 8 + lane] = __float2bfloat16(d_logit);
}

extern "C" void moe_router_softmax_topk_bwd(
    const void* probs,
    const void* gates,
    const void* indices,
    const void* d_gates,
    const void* d_probs_aux,
    void* d_router_logits,
    int bt,
    void* stream)
{
    // 8 warps per block, 256 threads/block
    int blocks = (bt + 7) / 8;
    moe_router_softmax_topk_bwd_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float*)probs,
        (const float*)gates,
        (const int*)indices,
        (const float*)d_gates,
        (const float*)d_probs_aux,
        (bf16*)d_router_logits,
        bt);
}

// ============================================================================
// Kernel 2: moe_scatter_bwd
//
// Backward of the two-pass scatter (forward: output[tok] = Σ_k gate[k]*expert_out[slot_k]).
// Computes:
//   d_expert_output[slot] = gate_value * d_output[token]
//   d_gate[slot] = dot(expert_output[slot], d_output[token])  (reduction over D)
//
// Each slot in [0, BT*2) maps to one (token, k) pair.
// token_perm[slot] encodes which token this slot serves.
// We process per-expert: expert_offsets[e] .. expert_offsets[e+1].
//
// Grid: one block per slot. Each block has 256 threads handling D_MODEL=1024 elements.
// 256 threads, 4 elements/thread for D=1024.
// ============================================================================

__global__ void __launch_bounds__(256)
moe_scatter_bwd_kernel(
    const bf16*  __restrict__ expert_output,  // [BT*2, D] bf16
    const bf16*  __restrict__ d_output,       // [BT, D] bf16
    const float* __restrict__ gates,          // [BT, 2] f32
    const int*   __restrict__ expert_indices, // [BT, 2] i32
    const int*   __restrict__ token_perm,     // [BT*2] i32 — encodes (token_id * 2 + k)
    bf16*        __restrict__ d_expert_output,// [BT*2, D] bf16
    float*       __restrict__ d_gate,         // [BT*2] f32
    int D)
{
    const int slot = blockIdx.x;
    const int tid = threadIdx.x;

    // Decode which token and which k-slot this permutation entry refers to.
    // token_perm stores the flat index into [BT, 2]: value = token_id * 2 + k
    int perm_val = token_perm[slot];
    int token_id = perm_val >> 1;   // perm_val / 2
    int k_slot   = perm_val & 1;    // perm_val % 2

    float g = gates[token_id * 2 + k_slot];

    // Compute d_expert_output and partial d_gate
    float local_dot = 0.0f;
    // 4 elements per thread, stride by blockDim
    for (int i = tid; i < D; i += 256) {
        float eo = __bfloat162float(expert_output[slot * D + i]);
        float dout = __bfloat162float(d_output[token_id * D + i]);

        // d_expert_output = gate * d_output
        d_expert_output[slot * D + i] = __float2bfloat16(g * dout);

        // accumulate dot product for d_gate
        local_dot += eo * dout;
    }

    // Warp-level reduction for d_gate
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        local_dot += __shfl_xor_sync(0xFFFFFFFF, local_dot, offset);
    }

    // Block-level reduction across warps (8 warps in 256 threads)
    __shared__ float warp_sums[8];
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) {
        warp_sums[warp_id] = local_dot;
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            sum += warp_sums[w];
        }
        d_gate[slot] = sum;
    }
}

extern "C" void moe_scatter_bwd(
    const void* expert_output,
    const void* d_output,
    const void* gates,
    const void* expert_indices,
    const void* token_perm,
    void* d_expert_output,
    void* d_gate,
    int bt,
    int D,
    void* stream)
{
    int total_slots = bt * 2; // TOP_K = 2
    moe_scatter_bwd_kernel<<<total_slots, 256, 0, (cudaStream_t)stream>>>(
        (const bf16*)expert_output,
        (const bf16*)d_output,
        (const float*)gates,
        (const int*)expert_indices,
        (const int*)token_perm,
        (bf16*)d_expert_output,
        (float*)d_gate,
        D);
}

// ============================================================================
// Kernel 3: moe_gather_bwd
//
// Backward of gather: forward was x_expert[slot] = xn[token_perm[slot] / 2]
// Backward scatters d_gathered back to d_xn.
//
// Same two-pass strategy as forward moe_scatter_k to avoid atomics:
//   Pass 0 (k=0, beta=0): d_xn[token] = d_gathered[slot_for_k0]   (write)
//   Pass 1 (k=1, beta=1): d_xn[token] += d_gathered[slot_for_k1]  (accumulate)
//
// Each pass processes slots belonging to that k-value.
// Since each token appears exactly once per k-slot, no write conflicts.
//
// We launch one block per slot, 256 threads, vectorized with uint4 (128-bit).
// D_MODEL=1024 bf16 = 2048 bytes = 128 uint4.
// 256 threads, each handles 128/256 = not evenly — use loop.
// Actually: 1024 bf16 elements, 256 threads → 4 elements/thread.
// For vectorized: 1024 bf16 = 512 bf16_2 = 256 uint2. 256 threads, 1 uint2/thread.
// Or: 1024 bf16 = 128 uint4 (16 bytes each). 256 threads → not clean.
// Simplest: 4 bf16 elements per thread, no vectorization needed for correctness.
// ============================================================================

__global__ void __launch_bounds__(256)
moe_gather_bwd_kernel(
    const bf16* __restrict__ d_gathered,   // [BT*2, D] bf16
    const int*  __restrict__ token_perm,   // [BT*2] — value = token_id * 2 + k
    bf16*       __restrict__ d_xn,         // [BT, D] bf16
    int D,
    int pass_k,     // 0 or 1
    int total_slots)
{
    const int slot = blockIdx.x;
    if (slot >= total_slots) return;

    int perm_val = token_perm[slot];
    int k_slot = perm_val & 1;
    // Only process slots matching this pass
    if (k_slot != pass_k) return;

    int token_id = perm_val >> 1;
    const int tid = threadIdx.x;

    for (int i = tid; i < D; i += 256) {
        float val = __bfloat162float(d_gathered[slot * D + i]);
        if (pass_k == 0) {
            // Write
            d_xn[token_id * D + i] = __float2bfloat16(val);
        } else {
            // Accumulate
            float existing = __bfloat162float(d_xn[token_id * D + i]);
            d_xn[token_id * D + i] = __float2bfloat16(existing + val);
        }
    }
}

// Optimized version: separate k=0 and k=1 slots using expert_offsets to avoid
// branch divergence. But since the permutation is sorted by expert (not by k),
// we can't trivially split by k. Instead, we launch all slots and branch.
//
// Better approach: launch all slots, each checks its k. Blocks where k doesn't
// match exit early. With TOP_K=2 and random distribution, ~50% of blocks exit
// each pass. This is acceptable for a memory-bound kernel.
//
// Even better: pre-sort is not needed. The forward gather was a simple
// x_expert[slot] = xn[perm[slot]/2]. The backward just needs to write/accumulate
// d_xn[perm[slot]/2] = d_gathered[slot]. We do two passes so that for each
// token, the k=0 slot writes and k=1 slot accumulates. No conflicts.

extern "C" void moe_gather_bwd(
    const void* d_gathered,
    const void* token_perm,
    void* d_xn,
    int bt,
    int D,
    void* stream)
{
    int total_slots = bt * 2;
    cudaStream_t s = (cudaStream_t)stream;

    // Pass 0: write (k=0 slots)
    moe_gather_bwd_kernel<<<total_slots, 256, 0, s>>>(
        (const bf16*)d_gathered,
        (const int*)token_perm,
        (bf16*)d_xn,
        D,
        0,
        total_slots);

    // Pass 1: accumulate (k=1 slots)
    moe_gather_bwd_kernel<<<total_slots, 256, 0, s>>>(
        (const bf16*)d_gathered,
        (const int*)token_perm,
        (bf16*)d_xn,
        D,
        1,
        total_slots);
}

// Force kernel registration on current device — launch ALL kernels from this TU
extern "C" void moe_backward_init() {
    CUdeviceptr dummy;
    cuMemAlloc(&dummy, 4096);
    moe_router_softmax_topk_bwd_kernel<<<1, 32, 0, 0>>>((float*)dummy, (float*)dummy, (int*)dummy, (float*)dummy, (float*)dummy, (nv_bfloat16*)dummy, 1);
    moe_scatter_bwd_kernel<<<1, 256, 0, 0>>>((nv_bfloat16*)dummy, (nv_bfloat16*)dummy, (float*)dummy, (int*)dummy, (int*)dummy, (nv_bfloat16*)dummy, (float*)dummy, 1);
    moe_gather_bwd_kernel<<<1, 256, 0, 0>>>((nv_bfloat16*)dummy, (int*)dummy, (nv_bfloat16*)dummy, 1, 1024, 0);
    cudaDeviceSynchronize();
    cuMemFree(dummy);
}

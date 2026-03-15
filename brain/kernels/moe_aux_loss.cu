#include <cuda.h>
// MoE auxiliary load-balancing loss (forward + backward)
// L_aux = alpha * N * sum_i(f_i * P_i)
//   f_i = expert_counts[i] / (BT * TOP_K)   — fraction of tokens routed to expert i
//   P_i = mean(probs[:, i])                  — average router probability for expert i
// N = N_EXPERTS = 8, TOP_K = 2

#define N_EXPERTS 8
#define TOP_K 2
#define WARP_SIZE 32

extern "C" {

// Forward: compute L_aux and atomicAdd into loss scalar.
// Grid:  (N_EXPERTS,)  i.e. 8 blocks, one per expert
// Block: min(BT, 1024) threads — each thread reduces a slice of probs[:, expert]
//
// Inputs:
//   router_probs [BT, 8]  — f32, full softmax probabilities
//   expert_counts [8]     — i32, number of tokens dispatched to each expert
//   BT                    — total tokens (B * T)
//   coeff                 — alpha (aux loss coefficient, e.g. 0.01)
// Output:
//   loss                  — f32 scalar, atomicAdd'd (caller must zero before launch)
__global__ void load_balance_loss_fwd(
    const float* __restrict__ router_probs,   // [BT, N_EXPERTS]
    const int*   __restrict__ expert_counts,   // [N_EXPERTS]
    float*                    loss,            // scalar
    int                       BT,
    float                     coeff
) {
    int e = blockIdx.x;  // expert index [0, 8)

    // f_e: fraction of token-slots routed to this expert
    float f_e = (float)expert_counts[e] / (float)(BT * TOP_K);

    // P_e: mean of router_probs[:, e] — parallel reduction across BT tokens
    float sum = 0.0f;
    for (int t = threadIdx.x; t < BT; t += blockDim.x) {
        sum += router_probs[t * N_EXPERTS + e];
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block-level reduction via shared memory (up to 32 warps = 1024 threads)
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int nwarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp == 0) {
        sum = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    // Thread 0 of each block contributes alpha * N * f_e * P_e
    if (threadIdx.x == 0) {
        float P_e = sum / (float)BT;
        float contrib = coeff * (float)N_EXPERTS * f_e * P_e;
        atomicAdd(loss, contrib);
    }
}

// Backward: accumulate gradient into d_router_probs.
// d_probs[t, e] += alpha * N * f_e / BT
//   f_e is stop-gradient (derived from expert_counts, not differentiable through argmax)
//   The 1/BT factor comes from d/d(probs[t,e]) of P_e = mean(probs[:, e])
//
// Grid:  (ceil(BT / 256),)
// Block: 256 threads
//
// Inputs:
//   expert_counts [8]     — i32
//   BT                    — total tokens
//   coeff                 — alpha
// Output:
//   d_router_probs [BT, 8] — f32, accumulated (+=)
__global__ void load_balance_loss_bwd(
    const int* __restrict__ expert_counts,     // [N_EXPERTS]
    float*                   d_router_probs,    // [BT, N_EXPERTS]
    int                      BT,
    float                    coeff
) {
    // Preload per-expert scale factors into shared memory
    __shared__ float scale[N_EXPERTS];
    if (threadIdx.x < N_EXPERTS) {
        // d L_aux / d probs[t, e] = alpha * N * f_e / BT
        float f_e = (float)expert_counts[threadIdx.x] / (float)(BT * TOP_K);
        scale[threadIdx.x] = coeff * (float)N_EXPERTS * f_e / (float)BT;
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= BT) return;

    // Accumulate gradient for all 8 experts at this token position
    // Unrolled — N_EXPERTS is a compile-time constant (8)
    float* row = d_router_probs + t * N_EXPERTS;
    #pragma unroll
    for (int e = 0; e < N_EXPERTS; e++) {
        row[e] += scale[e];
    }
}

} // extern "C"

// Launch configurations:
//
// load_balance_loss_fwd:
//   dim3 grid(N_EXPERTS);          // 8 blocks
//   dim3 block(min(BT, 1024));     // up to 1024 threads per block
//   cudaMemsetAsync(loss, 0, sizeof(float), stream);
//   load_balance_loss_fwd<<<grid, block, 0, stream>>>(router_probs, expert_counts, loss, BT, coeff);
//
// load_balance_loss_bwd:
//   int threads = 256;
//   int blocks = (BT + threads - 1) / threads;
//   load_balance_loss_bwd<<<blocks, threads, 0, stream>>>(expert_counts, d_router_probs, BT, coeff);

// Force kernel registration on current device
extern "C" void moe_aux_loss_init() {
    CUdeviceptr dummy;
    cuMemAlloc(&dummy, 128);
    load_balance_loss_fwd<<<1, 1, 0, 0>>>((float*)dummy, (int*)dummy, (float*)dummy, 1, 0.01f);
    cudaDeviceSynchronize();
    cuMemFree(dummy);
}

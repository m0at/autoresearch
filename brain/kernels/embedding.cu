#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Forward: out[i, d] = weight[idx[i], d]
// Vectorized: each thread copies 8 bf16 elements via uint4
__global__ void embedding_fwd_kernel(
    const unsigned int* __restrict__ idx,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int BT, int D)
{
    int total_vec = (BT * D) >> 3;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec;
         i += blockDim.x * gridDim.x) {
        int elem = i << 3;          // first bf16 index in this chunk
        int row  = elem / D;
        int col  = elem % D;
        unsigned int tok = idx[row];
        uint4 wv = ((const uint4*)(weight + (uint64_t)tok * D + col))[0];
        ((uint4*)(out + (uint64_t)row * D + col))[0] = wv;
    }

    // Tail elements (BT*D not divisible by 8)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int tail_start = (total_vec << 3);
        int total = BT * D;
        for (int i = tail_start; i < total; i++) {
            int d = i % D;
            int r = i / D;
            unsigned int tok = idx[r];
            out[i] = weight[(uint64_t)tok * D + d];
        }
    }
}

// ---------------------------------------------------------------------------
// Backward: shared-memory accumulation (one thread block per column)
//
// Each block handles one column d in [0, D).
// All BT tokens are scanned: gradients are accumulated into a shared-memory
// array of V fp32 entries using shared-memory atomicAdd (near-zero contention).
// The result is added to the existing d_weight (for gradient accumulation
// across micro-batches).
// ---------------------------------------------------------------------------

__global__ void embedding_bwd_kernel(
    const unsigned int* __restrict__ idx,
    const __nv_bfloat16* __restrict__ d_out,
    __nv_bfloat16* __restrict__ d_weight,
    int BT, int V, int D)
{
    extern __shared__ float s_acc[];  // V floats

    int d = blockIdx.x;  // one block per column
    if (d >= D) return;

    // Zero shared memory
    for (int v = threadIdx.x; v < V; v += blockDim.x)
        s_acc[v] = 0.0f;
    __syncthreads();

    // Accumulate gradients in shared memory
    for (int i = threadIdx.x; i < BT; i += blockDim.x) {
        unsigned int tok = idx[i];
        float grad = __bfloat162float(d_out[(uint64_t)i * D + d]);
        atomicAdd(&s_acc[tok], grad);
    }
    __syncthreads();

    // Add accumulated results to global d_weight
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float val = s_acc[v];
        if (val != 0.0f) {
            uint64_t off = (uint64_t)v * D + d;
            d_weight[off] = __float2bfloat16(__bfloat162float(d_weight[off]) + val);
        }
    }
}

extern "C" void embedding_fwd(
    const void* idx, const void* weight, void* out,
    int BT, int D, cudaStream_t stream)
{
    int total_vec = (BT * D) >> 3;
    int threads = 256;
    int blocks = (total_vec + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    if (blocks > 2048) blocks = 2048;
    embedding_fwd_kernel<<<blocks, threads, 0, stream>>>(
        (const unsigned int*)idx, (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)out, BT, D);
}

extern "C" void embedding_bwd(
    const void* idx, const void* d_out, void* d_weight,
    int BT, int V, int D, cudaStream_t stream)
{
    int threads = 256;
    int smem = V * sizeof(float);  // 8192 * 4 = 32KB < 48KB default
    embedding_bwd_kernel<<<D, threads, smem, stream>>>(
        (const unsigned int*)idx, (const __nv_bfloat16*)d_out,
        (__nv_bfloat16*)d_weight, BT, V, D);
}

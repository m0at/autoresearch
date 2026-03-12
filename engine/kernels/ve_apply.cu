#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Forward: v[i] += 2 * sigmoid(gate_expanded[i]) * ve[i]
// gate is [BT, N_KV_HEAD], expanded to match v's [BT, N_KV_HEAD * HEAD_DIM]
// Vectorized: uint4 loads (8 bf16 per load). Safe because D = N_KV_HEAD * HEAD_DIM
// is divisible by 8, so vec loads never cross row or head boundaries.
__global__ void ve_apply_fwd_kernel(
    __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ ve,
    const __nv_bfloat16* __restrict__ gate,
    int BT, int N_KV_HEAD, int HEAD_DIM)
{
    int D = N_KV_HEAD * HEAD_DIM;
    int total_vec = (BT * D) >> 3;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec;
         i += blockDim.x * gridDim.x) {
        int elem_idx = i << 3;
        int d = elem_idx % D;
        int bt = elem_idx / D;
        int h = d / HEAD_DIM;

        float g = __bfloat162float(gate[bt * N_KV_HEAD + h]);
        float sig = 1.0f / (1.0f + expf(-g));
        float two_sig = 2.0f * sig;

        uint4 vv = ((uint4*)v)[i];
        uint4 vev = ((const uint4*)ve)[i];
        __nv_bfloat16* vp = (__nv_bfloat16*)&vv;
        const __nv_bfloat16* vep = (const __nv_bfloat16*)&vev;

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float vi = __bfloat162float(vp[j]);
            float vei = __bfloat162float(vep[j]);
            vp[j] = __float2bfloat16(vi + two_sig * vei);
        }
        ((uint4*)v)[i] = vv;
    }
}

// Backward:
// d_ve[i] = 2 * sigmoid(gate_expanded[i]) * d_v[i]
// d_gate[bt, h] = sum_over_d(2 * sig * (1-sig) * ve[bt,h,d] * d_v[bt,h,d])
// One block per (bt, h) pair, HEAD_DIM threads for reduction.
__global__ void ve_apply_bwd_kernel(
    const __nv_bfloat16* __restrict__ d_v,
    const __nv_bfloat16* __restrict__ ve,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ d_ve,
    __nv_bfloat16* __restrict__ d_gate,
    int BT, int N_KV_HEAD, int HEAD_DIM)
{
    // blockIdx.x = bt * N_KV_HEAD + h
    int bth = blockIdx.x;
    if (bth >= BT * N_KV_HEAD) return;
    int bt = bth / N_KV_HEAD;
    int h = bth % N_KV_HEAD;
    int D = N_KV_HEAD * HEAD_DIM;

    float g = __bfloat162float(gate[bt * N_KV_HEAD + h]);
    float sig = 1.0f / (1.0f + expf(-g));

    int base = bt * D + h * HEAD_DIM;

    float local_dg = 0.0f;

    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        int idx = base + d;
        float dv = __bfloat162float(d_v[idx]);
        float vei = __bfloat162float(ve[idx]);

        d_ve[idx] = __float2bfloat16(2.0f * sig * dv);
        local_dg += 2.0f * sig * (1.0f - sig) * vei * dv;
    }

    // Warp reduction for d_gate
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_dg += __shfl_xor_sync(0xffffffff, local_dg, offset);

    // Cross-warp reduction via shared memory (if HEAD_DIM > 32)
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) shared[warp_id] = local_dg;
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t n_warps = (blockDim.x + 31) / 32;
        float total = 0.0f;
        for (uint32_t i = 0; i < n_warps; i++) total += shared[i];
        d_gate[bt * N_KV_HEAD + h] = __float2bfloat16(total);
    }
}

extern "C" void ve_apply_fwd(
    void* v, const void* ve, const void* gate,
    int BT, int N_KV_HEAD, int HEAD_DIM, cudaStream_t stream)
{
    int total_vec = (BT * N_KV_HEAD * HEAD_DIM) >> 3;
    int threads = 256;
    int blocks = (total_vec + threads - 1) / threads;
    ve_apply_fwd_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)v, (const __nv_bfloat16*)ve,
        (const __nv_bfloat16*)gate, BT, N_KV_HEAD, HEAD_DIM);
}

extern "C" void ve_apply_bwd(
    const void* d_v, const void* ve, const void* gate,
    void* d_ve, void* d_gate,
    int BT, int N_KV_HEAD, int HEAD_DIM, cudaStream_t stream)
{
    int n_blocks = BT * N_KV_HEAD;
    // Use min(HEAD_DIM, 256) threads -- HEAD_DIM is typically 128
    int threads = HEAD_DIM < 256 ? HEAD_DIM : 256;
    // Round up to warp boundary
    threads = ((threads + 31) / 32) * 32;
    ve_apply_bwd_kernel<<<n_blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)d_v, (const __nv_bfloat16*)ve,
        (const __nv_bfloat16*)gate,
        (__nv_bfloat16*)d_ve, (__nv_bfloat16*)d_gate,
        BT, N_KV_HEAD, HEAD_DIM);
}

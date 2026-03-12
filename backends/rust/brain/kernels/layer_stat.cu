#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ── layer_l2_norm_bf16 ───────────────────────────────────────────────────────

__global__ void layer_l2_norm_bf16_kernel(
    const __nv_bfloat16* x, int n, float* out, int out_idx)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float v = __bfloat162float(x[i]);
        sum += v * v;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Shared memory reduction across warps (block=256 → max 8 warps)
    __shared__ float smem[8];
    int lane  = threadIdx.x & 31;
    int warp  = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        sum = (lane < nwarps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) atomicAdd(&out[out_idx], sum);
    }
}

__global__ void inplace_sqrt_f32(float* out, int idx) {
    out[idx] = sqrtf(out[idx]);
}

extern "C" void layer_l2_norm_bf16(
    const __nv_bfloat16* x, int n, float* out, int out_idx, cudaStream_t stream)
{
    cudaMemsetAsync(&out[out_idx], 0, sizeof(float), stream);
    int block = 256;
    int grid  = min((n + block - 1) / block, 1024);
    layer_l2_norm_bf16_kernel<<<grid, block, 0, stream>>>(x, n, out, out_idx);
    inplace_sqrt_f32<<<1, 1, 0, stream>>>(out, out_idx);
}

// ── layer_scale_bf16 ─────────────────────────────────────────────────────────

__global__ void layer_scale_bf16_kernel(
    const float* scale, int scale_idx, __nv_bfloat16* x, int n)
{
    float s = scale[scale_idx];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] = __float2bfloat16(__bfloat162float(x[i]) * s);
    }
}

extern "C" void layer_scale_bf16(
    const float* scale, int scale_idx, __nv_bfloat16* x, int n, cudaStream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    layer_scale_bf16_kernel<<<grid, block, 0, stream>>>(scale, scale_idx, x, n);
}

// ── neuron_act_norm_bf16 ──────────────────────────────────────────────────────

__global__ void neuron_act_norm_bf16_kernel(
    const __nv_bfloat16* h_act, int bt, int mlp_dim, float* out, int layer_offset)
{
    int neuron = blockIdx.x;  // one block per neuron
    float sum = 0.0f;
    for (int i = threadIdx.x; i < bt; i += blockDim.x) {
        float v = __bfloat162float(h_act[i * mlp_dim + neuron]);
        sum += fabsf(v);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Shared memory reduction across warps (block=256 → max 8 warps)
    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        sum = (lane < nwarps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) out[layer_offset + neuron] = sum / (float)bt;
    }
}

extern "C" void neuron_act_norm_bf16(
    const __nv_bfloat16* h_act, int bt, int mlp_dim,
    float* out, int layer, cudaStream_t stream)
{
    int layer_offset = layer * mlp_dim;
    neuron_act_norm_bf16_kernel<<<mlp_dim, 256, 0, stream>>>(
        h_act, bt, mlp_dim, out, layer_offset);
}

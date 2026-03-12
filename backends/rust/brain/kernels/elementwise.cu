#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Vectorized elementwise kernels — 128-bit (uint4) loads/stores
// Each uint4 holds 8 bf16 values (16 bytes). 256 threads, grid-stride, max 2048 blocks.
// ============================================================================

static inline __device__ int min_int(int a, int b) { return a < b ? a : b; }

// ---------------------------------------------------------------------------
// residual_add: x[i] += y[i]
// ---------------------------------------------------------------------------
__global__ void residual_add_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    int N)
{
    int vec_N = N >> 3;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_N;
         i += blockDim.x * gridDim.x) {
        uint4 xv = ((const uint4*)x)[i];
        uint4 yv = ((const uint4*)y)[i];
        __nv_bfloat16* xp = (__nv_bfloat16*)&xv;
        const __nv_bfloat16* yp = (const __nv_bfloat16*)&yv;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            xp[j] = __float2bfloat16(__bfloat162float(xp[j]) + __bfloat162float(yp[j]));
        }
        ((uint4*)x)[i] = xv;
    }
    // Tail elements
    for (int i = vec_N * 8 + blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(y[i]));
    }
}

// ---------------------------------------------------------------------------
// three_way_add: out[i] = a[i] + b[i] + c[i]
// ---------------------------------------------------------------------------
__global__ void three_way_add_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ c,
    __nv_bfloat16* __restrict__ out,
    int N)
{
    int vec_N = N >> 3;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_N;
         i += blockDim.x * gridDim.x) {
        uint4 av = ((const uint4*)a)[i];
        uint4 bv = ((const uint4*)b)[i];
        uint4 cv = ((const uint4*)c)[i];
        uint4 ov;
        const __nv_bfloat16* ap = (const __nv_bfloat16*)&av;
        const __nv_bfloat16* bp = (const __nv_bfloat16*)&bv;
        const __nv_bfloat16* cp = (const __nv_bfloat16*)&cv;
        __nv_bfloat16* op = (__nv_bfloat16*)&ov;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            op[j] = __float2bfloat16(
                __bfloat162float(ap[j]) + __bfloat162float(bp[j]) + __bfloat162float(cp[j]));
        }
        ((uint4*)out)[i] = ov;
    }
    for (int i = vec_N * 8 + blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        out[i] = __float2bfloat16(
            __bfloat162float(a[i]) + __bfloat162float(b[i]) + __bfloat162float(c[i]));
    }
}

// ---------------------------------------------------------------------------
// cast_bf16_to_f32: load uint4 (8 bf16), store as 2×float4 (8 f32)
// ---------------------------------------------------------------------------
__global__ void cast_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ in,
    float* __restrict__ out,
    int N)
{
    int vec_N = N >> 3;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_N;
         i += blockDim.x * gridDim.x) {
        uint4 inv = ((const uint4*)in)[i];
        const __nv_bfloat16* ip = (const __nv_bfloat16*)&inv;
        float4 o0, o1;
        o0.x = __bfloat162float(ip[0]);
        o0.y = __bfloat162float(ip[1]);
        o0.z = __bfloat162float(ip[2]);
        o0.w = __bfloat162float(ip[3]);
        o1.x = __bfloat162float(ip[4]);
        o1.y = __bfloat162float(ip[5]);
        o1.z = __bfloat162float(ip[6]);
        o1.w = __bfloat162float(ip[7]);
        ((float4*)out)[i * 2]     = o0;
        ((float4*)out)[i * 2 + 1] = o1;
    }
    for (int i = vec_N * 8 + blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        out[i] = __bfloat162float(in[i]);
    }
}

// ---------------------------------------------------------------------------
// cast_f32_to_bf16: load 2×float4 (8 f32), store as uint4 (8 bf16)
// ---------------------------------------------------------------------------
__global__ void cast_f32_to_bf16_kernel(
    const float* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int N)
{
    int vec_N = N >> 3;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_N;
         i += blockDim.x * gridDim.x) {
        float4 i0 = ((const float4*)in)[i * 2];
        float4 i1 = ((const float4*)in)[i * 2 + 1];
        uint4 ov;
        __nv_bfloat16* op = (__nv_bfloat16*)&ov;
        op[0] = __float2bfloat16(i0.x);
        op[1] = __float2bfloat16(i0.y);
        op[2] = __float2bfloat16(i0.z);
        op[3] = __float2bfloat16(i0.w);
        op[4] = __float2bfloat16(i1.x);
        op[5] = __float2bfloat16(i1.y);
        op[6] = __float2bfloat16(i1.z);
        op[7] = __float2bfloat16(i1.w);
        ((uint4*)out)[i] = ov;
    }
    for (int i = vec_N * 8 + blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        out[i] = __float2bfloat16(in[i]);
    }
}

// ---------------------------------------------------------------------------
// slice_cols: extract first out_cols columns from (rows, in_cols)
// Vectorized path when out_cols % 8 == 0 AND in_cols % 8 == 0
// ---------------------------------------------------------------------------
__global__ void slice_cols_kernel_vec(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int rows, int in_cols_vec, int out_cols_vec)
{
    // in_cols_vec = in_cols/8, out_cols_vec = out_cols/8
    int total = rows * out_cols_vec;
    for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total;
         gid += blockDim.x * gridDim.x) {
        int row = gid / out_cols_vec;
        int col = gid % out_cols_vec;
        ((uint4*)output)[gid] = ((const uint4*)input)[row * in_cols_vec + col];
    }
}

__global__ void slice_cols_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int rows, int in_cols, int out_cols)
{
    int total = rows * out_cols;
    for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total;
         gid += blockDim.x * gridDim.x) {
        int row = gid / out_cols;
        int col = gid % out_cols;
        output[gid] = input[row * in_cols + col];
    }
}

// ---------------------------------------------------------------------------
// add_slice_cols: dst[:, :src_cols] += src
// Vectorized path when src_cols % 8 == 0 AND dst_cols % 8 == 0
// ---------------------------------------------------------------------------
__global__ void add_slice_cols_kernel_vec(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int rows, int dst_cols_vec, int src_cols_vec)
{
    // dst_cols_vec = dst_cols/8, src_cols_vec = src_cols/8
    int total = rows * src_cols_vec;
    for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total;
         gid += blockDim.x * gridDim.x) {
        int row = gid / src_cols_vec;
        int col = gid % src_cols_vec;
        int dst_vi = row * dst_cols_vec + col;
        uint4 dv = ((const uint4*)dst)[dst_vi];
        uint4 sv = ((const uint4*)src)[gid];
        __nv_bfloat16* dp = (__nv_bfloat16*)&dv;
        const __nv_bfloat16* sp = (const __nv_bfloat16*)&sv;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            dp[j] = __float2bfloat16(__bfloat162float(dp[j]) + __bfloat162float(sp[j]));
        }
        ((uint4*)dst)[dst_vi] = dv;
    }
}

__global__ void add_slice_cols_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int rows, int dst_cols, int src_cols)
{
    int total = rows * src_cols;
    for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total;
         gid += blockDim.x * gridDim.x) {
        int row = gid / src_cols;
        int col = gid % src_cols;
        int dst_idx = row * dst_cols + col;
        dst[dst_idx] = __float2bfloat16(
            __bfloat162float(dst[dst_idx]) + __bfloat162float(src[gid]));
    }
}

// ============================================================================
// extern "C" entry points — signatures unchanged
// ============================================================================

extern "C" void residual_add(void* x, const void* y, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = min((N / 8 + threads - 1) / threads, 2048);
    if (blocks < 1) blocks = 1;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)x, (const __nv_bfloat16*)y, N);
}

extern "C" void three_way_add(
    const void* a, const void* b, const void* c, void* out, int N, cudaStream_t stream)
{
    int threads = 256;
    int blocks = min((N / 8 + threads - 1) / threads, 2048);
    if (blocks < 1) blocks = 1;
    three_way_add_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
        (const __nv_bfloat16*)c, (__nv_bfloat16*)out, N);
}

extern "C" void cast_bf16_to_f32(const void* in, void* out, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = min((N / 8 + threads - 1) / threads, 2048);
    if (blocks < 1) blocks = 1;
    cast_bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)in, (float*)out, N);
}

extern "C" void cast_f32_to_bf16(const void* in, void* out, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = min((N / 8 + threads - 1) / threads, 2048);
    if (blocks < 1) blocks = 1;
    cast_f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)in, (__nv_bfloat16*)out, N);
}

extern "C" void slice_cols(
    const void* input, void* output,
    int rows, int in_cols, int out_cols, cudaStream_t stream)
{
    int threads = 256;
    if ((out_cols & 7) == 0 && (in_cols & 7) == 0) {
        int total = rows * (out_cols >> 3);
        int blocks = min((total + threads - 1) / threads, 2048);
        if (blocks < 1) blocks = 1;
        slice_cols_kernel_vec<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
            rows, in_cols >> 3, out_cols >> 3);
    } else {
        int total = rows * out_cols;
        int blocks = min((total + threads - 1) / threads, 2048);
        if (blocks < 1) blocks = 1;
        slice_cols_kernel<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
            rows, in_cols, out_cols);
    }
}

extern "C" void add_slice_cols(
    void* dst, const void* src,
    int rows, int dst_cols, int src_cols, cudaStream_t stream)
{
    int threads = 256;
    if ((src_cols & 7) == 0 && (dst_cols & 7) == 0) {
        int total = rows * (src_cols >> 3);
        int blocks = min((total + threads - 1) / threads, 2048);
        if (blocks < 1) blocks = 1;
        add_slice_cols_kernel_vec<<<blocks, threads, 0, stream>>>(
            (__nv_bfloat16*)dst, (const __nv_bfloat16*)src,
            rows, dst_cols >> 3, src_cols >> 3);
    } else {
        int total = rows * src_cols;
        int blocks = min((total + threads - 1) / threads, 2048);
        if (blocks < 1) blocks = 1;
        add_slice_cols_kernel<<<blocks, threads, 0, stream>>>(
            (__nv_bfloat16*)dst, (const __nv_bfloat16*)src,
            rows, dst_cols, src_cols);
    }
}

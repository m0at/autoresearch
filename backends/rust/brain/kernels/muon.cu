#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

// ── Helpers ──────────────────────────────────────────────────────────────────

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);
}

// ── Nesterov momentum (lerp-based, matching Python) ──────────────────────────
// Python equivalent:
//   momentum_buffer.lerp_(grad, 1 - momentum)
//     => buf = momentum * buf + (1-momentum) * grad
//   g = grad.lerp_(buf, momentum)
//     => out = (1-momentum) * grad + momentum * buf
//
// This formulation has proper (1-momentum) scaling to avoid gradient amplification.

__global__ void muon_nesterov_kernel(
    __nv_bfloat16* buf, const __nv_bfloat16* grad, __nv_bfloat16* out,
    float momentum, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = bf16_to_f32(grad[i]);
    float b = momentum * bf16_to_f32(buf[i]) + (1.0f - momentum) * g;
    buf[i] = f32_to_bf16(b);
    out[i] = f32_to_bf16((1.0f - momentum) * g + momentum * b);
}

extern "C" void muon_nesterov_bf16(
    void* buf, const void* grad, void* out, float momentum, int N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    muon_nesterov_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)buf, (const __nv_bfloat16*)grad,
        (__nv_bfloat16*)out, momentum, N);
}

// ── Nesterov with f32 momentum buffer (mixed precision) ──────────────────────
// Same math as above but momentum buffer is f32, grad/out are bf16.
// This matches Python where momentum_buffer is f32 (nn.Linear params are f32).

__global__ void muon_nesterov_f32buf_kernel(
    float* buf, const __nv_bfloat16* grad, __nv_bfloat16* out,
    float momentum, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = bf16_to_f32(grad[i]);
    float b = momentum * buf[i] + (1.0f - momentum) * g;
    buf[i] = b;
    out[i] = f32_to_bf16((1.0f - momentum) * g + momentum * b);
}

extern "C" void muon_nesterov_f32buf(
    void* buf, const void* grad, void* out, float momentum, int N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    muon_nesterov_f32buf_kernel<<<blocks, threads, 0, stream>>>(
        (float*)buf, (const __nv_bfloat16*)grad,
        (__nv_bfloat16*)out, momentum, N);
}

// ── Weight update with f32 master weights ────────────────────────────────────
// Updates f32 master weight, then copies to bf16 compute weight.
// Python: params (f32) -= lr * g + lr * wd * params * mask

__global__ void muon_weight_update_f32_kernel(
    float* master, __nv_bfloat16* compute, const __nv_bfloat16* update,
    float lr, float wd, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float p = master[i];
    float g = bf16_to_f32(update[i]);
    float mask = (g * p >= 0.0f) ? 1.0f : 0.0f;
    p -= lr * g + lr * wd * p * mask;
    master[i] = p;
    compute[i] = f32_to_bf16(p);
}

extern "C" void muon_weight_update_f32(
    void* master, void* compute, const void* update, float lr, float wd, int N,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    muon_weight_update_f32_kernel<<<blocks, threads, 0, stream>>>(
        (float*)master, (__nv_bfloat16*)compute, (const __nv_bfloat16*)update,
        lr, wd, N);
}

// ── Copy bf16 → f32 (init master from bf16 weights) ─────────────────────────

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* src, float* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = bf16_to_f32(src[i]);
}

extern "C" void copy_bf16_to_f32(const void* src, void* dst, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>((const __nv_bfloat16*)src, (float*)dst, N);
}

// ── Frobenius normalize ──────────────────────────────────────────────────────
// out[i] = x[i] / frobenius_norm(x), where frobenius_norm = sqrt(sum(x^2))
// Two-pass: first reduce sum of squares, then scale.

__global__ void frob_sum_sq_kernel(
    const __nv_bfloat16* x, float* partial_sums, int N
) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float val = (i < N) ? bf16_to_f32(x[i]) : 0.0f;
    smem[tid] = val * val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(partial_sums, smem[0]);
}

__global__ void frob_scale_kernel(
    const __nv_bfloat16* x, __nv_bfloat16* out, const float* norm_sq, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Python: X / (X.norm() * 1.02 + 1e-6)
    float norm = sqrtf(*norm_sq);
    float scale = 1.0f / (norm * 1.02f + 1e-6f);
    out[i] = f32_to_bf16(bf16_to_f32(x[i]) * scale);
}

extern "C" void frob_normalize_bf16(const void* x, void* out, int N, float* scratch,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    // scratch must point to at least 1 float of device memory
    cudaMemsetAsync(scratch, 0, sizeof(float), stream);
    frob_sum_sq_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, scratch, N);
    frob_scale_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (__nv_bfloat16*)out, scratch, N);
}

// ── Weight update with cautious decay ──────────────────────────────────────────
// Python: mask = (g * params) >= 0
//         params -= lr * g + lr * wd * params * mask

__global__ void muon_weight_update_kernel(
    __nv_bfloat16* param, const __nv_bfloat16* update,
    float lr, float wd, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float p = bf16_to_f32(param[i]);
    float g = bf16_to_f32(update[i]);
    float mask = (g * p >= 0.0f) ? 1.0f : 0.0f;
    p -= lr * g + lr * wd * p * mask;
    param[i] = f32_to_bf16(p);
}

extern "C" void muon_weight_update_bf16(
    void* param, const void* update, float lr, float wd, int N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    muon_weight_update_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)param, (const __nv_bfloat16*)update, lr, wd, N);
}

// ── Scale in-place ───────────────────────────────────────────────────────────
// x[i] *= scale

__global__ void scale_kernel(__nv_bfloat16* x, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] = f32_to_bf16(bf16_to_f32(x[i]) * scale);
}

extern "C" void scale_bf16(void* x, float scale, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    scale_kernel<<<blocks, threads, 0, stream>>>((__nv_bfloat16*)x, scale, N);
}

// ── Identity matrix ──────────────────────────────────────────────────────────
// out = I_d (d x d identity, stored row-major)

__global__ void eye_kernel(__nv_bfloat16* out, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d * d) return;
    int row = i / d;
    int col = i % d;
    out[i] = (row == col) ? f32_to_bf16(1.0f) : f32_to_bf16(0.0f);
}

extern "C" void eye_bf16(void* out, int d, cudaStream_t stream) {
    int N = d * d;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    eye_kernel<<<blocks, threads, 0, stream>>>((__nv_bfloat16*)out, d);
}

// ── NorMuon variance reduction ───────────────────────────────────────────
// Applies per-row (or per-col) second-moment normalization after Polar Express.
// 3 internal kernels: row_variance -> update_ema -> apply_scale.

// Kernel 1: Compute v_mean[i] = mean(g[i,:]^2) (reduce_cols=1) or mean(g[:,i]^2) (reduce_cols=0).
// Also atomicAdd v_mean[i] * red_dim_size into v_norm_sq scalar.
__global__ void normuon_row_variance_kernel(
    const __nv_bfloat16* g, float* v_mean, float* v_norm_sq,
    int m, int n, int reduce_cols
) {
    int idx = blockIdx.x; // one block per reduction group
    int red_dim = reduce_cols ? n : m;
    int tid = threadIdx.x;

    __shared__ float smem[256];
    float acc = 0.0f;

    for (int j = tid; j < red_dim; j += blockDim.x) {
        int row, col;
        if (reduce_cols) { row = idx; col = j; }
        else             { row = j;   col = idx; }
        float val = bf16_to_f32(g[row * n + col]);
        acc += val * val;
    }
    smem[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        float mean_val = smem[0] / (float)red_dim;
        v_mean[idx] = mean_val;
        atomicAdd(v_norm_sq, mean_val * (float)red_dim);
    }
}

// Kernel 2: Update EMA, compute step_size, accumulate v_norm_new_sq.
__global__ void normuon_update_ema_kernel(
    float* v_mean, float* second_mom, float* v_norm_new_sq,
    int num_groups, int red_dim, float beta2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_groups) return;

    float vm = v_mean[i];
    float sm = beta2 * second_mom[i] + (1.0f - beta2) * vm;
    second_mom[i] = sm;
    float step_sq = 1.0f / fmaxf(sm, 1e-10f);
    float partial = vm * (float)red_dim * step_sq;
    atomicAdd(v_norm_new_sq, partial);
}

// Kernel 3: Apply final scale to each element of g.
__global__ void normuon_apply_scale_kernel(
    __nv_bfloat16* g, const float* second_mom,
    const float* v_norm_sq, const float* v_norm_new_sq,
    int m, int n, int reduce_cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (i >= total) return;

    int row = i / n;
    int col = i % n;
    int idx = reduce_cols ? row : col;

    float step_size = rsqrtf(fmaxf(second_mom[idx], 1e-10f));
    float v_norm = sqrtf(*v_norm_sq);
    float v_norm_new = sqrtf(*v_norm_new_sq);
    float final_scale = step_size * (v_norm / fmaxf(v_norm_new, 1e-10f));
    g[i] = f32_to_bf16(bf16_to_f32(g[i]) * final_scale);
}

extern "C" void normuon_step_bf16(
    void* g, float* second_mom, int m, int n, int reduce_cols, float beta2,
    float* scratch, cudaStream_t stream
) {
    int red_dim = reduce_cols ? n : m;
    int num_groups = reduce_cols ? m : n;

    // scratch layout: [v_mean (num_groups floats)] [v_norm_sq (1 float)] [v_norm_new_sq (1 float)]
    float* d_v_mean = scratch;
    float* d_v_norm_sq = scratch + num_groups;
    float* d_v_norm_new_sq = scratch + num_groups + 1;
    cudaMemsetAsync(d_v_mean, 0, (num_groups + 2) * sizeof(float), stream);

    // Kernel 1: row variance
    int k1_threads = 256;
    normuon_row_variance_kernel<<<num_groups, k1_threads, 0, stream>>>(
        (__nv_bfloat16*)g, d_v_mean, d_v_norm_sq, m, n, reduce_cols);

    // Kernel 2: update EMA
    int k2_threads = 256;
    int k2_blocks = (num_groups + k2_threads - 1) / k2_threads;
    normuon_update_ema_kernel<<<k2_blocks, k2_threads, 0, stream>>>(
        d_v_mean, second_mom, d_v_norm_new_sq, num_groups, red_dim, beta2);

    // Kernel 3: apply scale
    int total = m * n;
    int k3_threads = 256;
    int k3_blocks = (total + k3_threads - 1) / k3_threads;
    normuon_apply_scale_kernel<<<k3_blocks, k3_threads, 0, stream>>>(
        (__nv_bfloat16*)g, second_mom, d_v_norm_sq, d_v_norm_new_sq,
        m, n, reduce_cols);
}

// ── axpby: out = alpha * x + beta * y ────────────────────────────────────────

__global__ void axpby_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* y, __nv_bfloat16* out,
    float alpha, float beta, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = f32_to_bf16(alpha * bf16_to_f32(x[i]) + beta * bf16_to_f32(y[i]));
}

extern "C" void axpby_bf16(
    const void* x, const void* y, void* out,
    float alpha, float beta, int N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    axpby_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)y,
        (__nv_bfloat16*)out, alpha, beta, N);
}

// ── Fused NS combined: out = ca*I + cb*A + cc*A^2 (batched) ──────────────────
// Replaces: batch × eye_bf16 + scale_bf16 + 2 × axpby_bf16 per NS iteration.
// Each batch element is a d×d matrix at stride d*d.
// Total elements: batch * d * d.

__global__ void ns_combined_batched_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ A2,
    __nv_bfloat16* __restrict__ out,
    float ca, float cb, float cc,
    int d, int batch
) {
    int total = batch * d * d;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += blockDim.x * gridDim.x) {
        int local = i % (d * d);
        int row = local / d;
        int col = local % d;
        float eye_val = (row == col) ? 1.0f : 0.0f;
        float val = ca * eye_val + cb * bf16_to_f32(A[i]) + cc * bf16_to_f32(A2[i]);
        out[i] = f32_to_bf16(val);
    }
}

extern "C" void ns_combined_batched(
    const void* A, const void* A2, void* out,
    float ca, float cb, float cc,
    int d, int batch, cudaStream_t stream
) {
    int total = batch * d * d;
    int threads = 512;
    int blocks = min((total + threads - 1) / threads, 1024);
    ns_combined_batched_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)A, (const __nv_bfloat16*)A2,
        (__nv_bfloat16*)out, ca, cb, cc, d, batch);
}

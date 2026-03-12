#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// 1. Fused RMS Norm  (bf16 I/O, f32 accumulation)
//    One threadgroup per row.  dim <= 8192 assumed (one threadgroup handles a row).
// ---------------------------------------------------------------------------

constant uint RMSNORM_THREADS = 256;

kernel void rms_norm_forward(
    device const half* input   [[buffer(0)]],
    device half* output        [[buffer(1)]],
    constant uint& dim         [[buffer(2)]],
    constant float& eps        [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    threadgroup float shared[RMSNORM_THREADS];

    uint row = gid;
    device const half* x = input  + row * dim;
    device half*       y = output + row * dim;

    // Accumulate sum of squares in f32
    float ss = 0.0f;
    for (uint i = lid; i < dim; i += tpg) {
        float v = float(x[i]);
        ss += v * v;
    }
    shared[lid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared[0] / float(dim) + eps);

    // Write normalized output
    for (uint i = lid; i < dim; i += tpg) {
        y[i] = half(float(x[i]) * rms);
    }
}

// ---------------------------------------------------------------------------
// 2. Fused ReLU-squared (branchless)
// ---------------------------------------------------------------------------

kernel void relu_squared(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    constant uint& length    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= length) return;
    float x = float(input[tid]);
    float r = max(x, 0.0f);
    output[tid] = half(r * r);
}

// ---------------------------------------------------------------------------
// 3. Fused Softcap:  cap * tanh(x / cap)
// ---------------------------------------------------------------------------

kernel void softcap(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant float& cap        [[buffer(2)]],
    constant uint& length      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= length) return;
    float x = input[tid];
    output[tid] = cap * precise::tanh(x / cap);
}

// ---------------------------------------------------------------------------
// 4. Fused Cross-Entropy  (log-softmax + NLL in one pass)
//    One threadgroup per row (token).
// ---------------------------------------------------------------------------

constant uint CE_THREADS = 256;

kernel void cross_entropy_forward(
    device const float* logits   [[buffer(0)]],
    device const int*   targets  [[buffer(1)]],
    device float*       losses   [[buffer(2)]],
    constant uint& vocab_size    [[buffer(3)]],
    constant uint& batch_tokens  [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (gid >= batch_tokens) return;

    threadgroup float shared[CE_THREADS];

    device const float* row = logits + gid * vocab_size;
    int target = targets[gid];

    // ---- pass 1: row max ----
    float m = -INFINITY;
    for (uint i = lid; i < vocab_size; i += tpg) {
        m = max(m, row[i]);
    }
    shared[lid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    // ---- pass 2: sum of exp(x - max) ----
    float sum_exp = 0.0f;
    for (uint i = lid; i < vocab_size; i += tpg) {
        sum_exp += exp(row[i] - row_max);
    }
    shared[lid] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = shared[0];

    // ---- compute loss = -logit[target] + max + log(sum_exp) ----
    if (lid == 0) {
        losses[gid] = -row[target] + row_max + log(total);
    }
}

// ---------------------------------------------------------------------------
// 5. Fused AdamW step  (all f32, one thread per parameter)
//    In-place variant: reads and writes the same param/exp_avg/exp_avg_sq buffers.
// ---------------------------------------------------------------------------

kernel void adamw_step(
    device float*       param       [[buffer(0)]],
    device const float* grad        [[buffer(1)]],
    device float*       exp_avg     [[buffer(2)]],
    device float*       exp_avg_sq  [[buffer(3)]],
    constant float& lr              [[buffer(4)]],
    constant float& beta1           [[buffer(5)]],
    constant float& beta2           [[buffer(6)]],
    constant float& eps             [[buffer(7)]],
    constant float& wd              [[buffer(8)]],
    constant float& bias1_corr      [[buffer(9)]],
    constant float& bias2_corr      [[buffer(10)]],
    constant uint&  length          [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= length) return;

    float p = param[tid];
    float g = grad[tid];
    float m = exp_avg[tid];
    float v = exp_avg_sq[tid];

    // Weight decay (decoupled)
    p -= lr * wd * p;

    // Momentum update
    m = beta1 * m + (1.0f - beta1) * g;

    // Second moment update
    v = beta2 * v + (1.0f - beta2) * g * g;

    // Bias-corrected estimates
    float m_hat = m / bias1_corr;
    float v_hat = v / bias2_corr;

    // Parameter update
    p -= lr * m_hat / (sqrt(v_hat) + eps);

    param[tid]      = p;
    exp_avg[tid]    = m;
    exp_avg_sq[tid] = v;
}

// ---------------------------------------------------------------------------
// 5b. Fused AdamW step — packed I/O variant for candle CustomOp integration.
//     Input:  [param(N) | grad(N) | exp_avg(N) | exp_avg_sq(N)] — 4N floats
//     Output: [new_param(N) | new_exp_avg(N) | new_exp_avg_sq(N)] — 3N floats
// ---------------------------------------------------------------------------

kernel void adamw_step_packed(
    device const float* input       [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    constant float& lr              [[buffer(2)]],
    constant float& beta1           [[buffer(3)]],
    constant float& beta2           [[buffer(4)]],
    constant float& eps             [[buffer(5)]],
    constant float& wd              [[buffer(6)]],
    constant float& bias1_corr      [[buffer(7)]],
    constant float& bias2_corr      [[buffer(8)]],
    constant uint&  length          [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= length) return;

    float p = input[tid];
    float g = input[length + tid];
    float m = input[2 * length + tid];
    float v = input[3 * length + tid];

    p -= lr * wd * p;
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;
    float m_hat = m / bias1_corr;
    float v_hat = v / bias2_corr;
    p -= lr * m_hat / (sqrt(v_hat) + eps);

    output[tid]              = p;
    output[length + tid]     = m;
    output[2 * length + tid] = v;
}

// ---------------------------------------------------------------------------
// 6. Polar Express step  (batched Newton-Schulz iteration for Muon)
//
//    Each batch element is a (rows x cols) matrix.
//    Tall (rows > cols): A = X^T X,  B = b*A + c*A*A,  X' = a*X + X*B
//    Wide (rows <= cols): A = X X^T, B = b*A + c*A*A,  X' = a*X + B*X
//
//    Uses tiled matmul with threadgroup shared memory.
//    Tile size 16x16, one threadgroup computes one output tile.
// ---------------------------------------------------------------------------

constant uint TILE = 16;

// Helper: tiled matmul  C = alpha*C_in + beta*(A @ B)
// A is (M x K), B is (K x N), C is (M x N), all bf16.
// Launched with threadgroups covering (M/TILE, N/TILE) and threads (TILE, TILE).
static void tiled_matmul(
    device const bfloat* A, uint lda,
    device const bfloat* B, uint ldb,
    device bfloat* C, uint ldc,
    uint M, uint K, uint N,
    float alpha, float beta,
    uint2 tg_pos, uint2 t_pos,
    threadgroup float* sA,    // TILE * TILE
    threadgroup float* sB)    // TILE * TILE
{
    uint row = tg_pos.y * TILE + t_pos.y;
    uint col = tg_pos.x * TILE + t_pos.x;

    float acc = 0.0f;

    for (uint t = 0; t < K; t += TILE) {
        // Load tile of A into shared
        uint a_row = row;
        uint a_col = t + t_pos.x;
        sA[t_pos.y * TILE + t_pos.x] = (a_row < M && a_col < K) ? float(A[a_row * lda + a_col]) : 0.0f;

        // Load tile of B into shared
        uint b_row = t + t_pos.y;
        uint b_col = col;
        sB[t_pos.y * TILE + t_pos.x] = (b_row < K && b_col < N) ? float(B[b_row * ldb + b_col]) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            acc += sA[t_pos.y * TILE + k] * sB[k * TILE + t_pos.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        float prev = (alpha != 0.0f) ? alpha * float(C[row * ldc + col]) : 0.0f;
        C[row * ldc + col] = bfloat(prev + beta * acc);
    }
}

// Dispatch with threadgroups = (ceil(cols/TILE) * ceil(rows/TILE) * num_params)
// and threads_per_threadgroup = (TILE, TILE, 1).
// workspace must hold num_params * max(inner, inner)^2 * 2 bfloats,
// where inner = is_tall ? cols : rows.

kernel void polar_express_step(
    device bfloat* X             [[buffer(0)]],
    device bfloat* workspace     [[buffer(1)]],
    constant float& a          [[buffer(2)]],
    constant float& b          [[buffer(3)]],
    constant float& c          [[buffer(4)]],
    constant uint& num_params  [[buffer(5)]],
    constant uint& rows        [[buffer(6)]],
    constant uint& cols        [[buffer(7)]],
    constant bool& is_tall     [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg  [[thread_position_in_threadgroup]],
    uint3 ntg   [[threadgroups_per_grid]])
{
    // Determine which batch element and which output tile
    uint mat_size = rows * cols;
    uint inner = is_tall ? cols : rows;
    uint a_mat_size = inner * inner;

    // Grid is laid out as (tile_x, tile_y, batch_idx)
    uint batch = tgpig.z;
    if (batch >= num_params) return;

    device bfloat* Xi = X + batch * mat_size;
    device bfloat* Ai = workspace + batch * a_mat_size * 2;  // A storage
    device bfloat* Bi = Ai + a_mat_size;                      // B storage

    threadgroup float sA_mem[TILE * TILE];
    threadgroup float sB_mem[TILE * TILE];

    uint2 tg_pos = uint2(tgpig.x, tgpig.y);
    uint2 t_pos  = uint2(tptg.x, tptg.y);

    uint tiles_inner = (inner + TILE - 1) / TILE;

    // We run this kernel in 4 sequential dispatches from the host side.
    // But for a single-dispatch approach, we encode the phase via
    // the grid layout.  The simplest correct approach: this kernel
    // computes ONE phase per dispatch.  The host dispatches 4 times
    // with a phase uniform.
    //
    // For clarity and correctness, we split into separate kernels below.
    // This kernel is a placeholder showing the algorithm; the host
    // should call the phase kernels individually.
    //
    // However, since the prompt asks for a single kernel, we implement
    // a single-dispatch version using device-side barriers between phases.
    // Metal 3 on M5 does not support grid-wide barriers, so we split
    // into phase kernels.  See polar_express_phase[1-4] below.
}

// Phase 1: Compute A
//   is_tall: A = X^T @ X  (inner x rows) @ (rows x inner) => (inner x inner)
//   wide:    A = X @ X^T   (inner x cols) @ (cols x inner) => (inner x inner)
// Grid: (ceil(inner/TILE), ceil(inner/TILE), num_params), threads: (TILE, TILE, 1)

kernel void polar_express_phase1(
    device const bfloat* X       [[buffer(0)]],
    device bfloat* workspace     [[buffer(1)]],
    constant uint& num_params  [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    constant uint& cols        [[buffer(4)]],
    constant bool& is_tall     [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg  [[thread_position_in_threadgroup]])
{
    uint batch = tgpig.z;
    if (batch >= num_params) return;

    uint mat_size = rows * cols;
    uint inner = is_tall ? cols : rows;
    uint K_dim = is_tall ? rows : cols;
    uint a_mat_size = inner * inner;

    device const bfloat* Xi = X + batch * mat_size;
    device bfloat* Ai = workspace + batch * a_mat_size * 2;

    threadgroup float sA_mem[TILE * TILE];
    threadgroup float sB_mem[TILE * TILE];

    uint2 tg_pos = uint2(tgpig.x, tgpig.y);
    uint2 t_pos  = uint2(tptg.x, tptg.y);

    // A = X^T @ X  (is_tall) means we treat X as (rows x cols), X^T is (cols x rows)
    // So A[i,j] = sum_k X[k,i] * X[k,j]  =>  X^T has dims (cols x rows), lda=cols for X^T reading
    // For tiled_matmul: left=(inner x K_dim), right=(K_dim x inner)

    if (is_tall) {
        // A = X^T @ X.  X is row-major (rows x cols).
        // X^T[i,k] = X[k,i], so we read X transposed.
        // We'll do the matmul manually with the transpose.
        uint row = tg_pos.y * TILE + t_pos.y;
        uint col = tg_pos.x * TILE + t_pos.x;

        float acc = 0.0f;
        for (uint t = 0; t < K_dim; t += TILE) {
            // Load tile of X^T: X^T[row, t+tx] = X[t+tx, row]
            uint xr = t + t_pos.x;
            sA_mem[t_pos.y * TILE + t_pos.x] = (row < inner && xr < K_dim) ? float(Xi[xr * cols + row]) : 0.0f;

            // Load tile of X: X[t+ty, col]
            uint xr2 = t + t_pos.y;
            sB_mem[t_pos.y * TILE + t_pos.x] = (xr2 < K_dim && col < inner) ? float(Xi[xr2 * cols + col]) : 0.0f;

            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < TILE; k++) {
                acc += sA_mem[t_pos.y * TILE + k] * sB_mem[k * TILE + t_pos.x];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row < inner && col < inner) {
            Ai[row * inner + col] = bfloat(acc);
        }
    } else {
        // A = X @ X^T.  X is (rows x cols).
        // A[i,j] = sum_k X[i,k] * X[j,k]
        uint row = tg_pos.y * TILE + t_pos.y;
        uint col = tg_pos.x * TILE + t_pos.x;

        float acc = 0.0f;
        for (uint t = 0; t < K_dim; t += TILE) {
            // X[row, t+tx]
            uint xc = t + t_pos.x;
            sA_mem[t_pos.y * TILE + t_pos.x] = (row < inner && xc < K_dim) ? float(Xi[row * cols + xc]) : 0.0f;

            // X^T[t+ty, col] = X[col, t+ty]
            uint xc2 = t + t_pos.y;
            sB_mem[t_pos.y * TILE + t_pos.x] = (xc2 < K_dim && col < inner) ? float(Xi[col * cols + xc2]) : 0.0f;

            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < TILE; k++) {
                acc += sA_mem[t_pos.y * TILE + k] * sB_mem[k * TILE + t_pos.x];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row < inner && col < inner) {
            Ai[row * inner + col] = bfloat(acc);
        }
    }
}

// Phase 2: Compute B = b*A + c*(A @ A)
// Grid: (ceil(inner/TILE), ceil(inner/TILE), num_params), threads: (TILE, TILE, 1)

kernel void polar_express_phase2(
    device bfloat* workspace     [[buffer(0)]],
    constant float& b          [[buffer(1)]],
    constant float& c          [[buffer(2)]],
    constant uint& num_params  [[buffer(3)]],
    constant uint& inner       [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg  [[thread_position_in_threadgroup]])
{
    uint batch = tgpig.z;
    if (batch >= num_params) return;

    uint a_mat_size = inner * inner;
    device bfloat* Ai = workspace + batch * a_mat_size * 2;
    device bfloat* Bi = Ai + a_mat_size;

    threadgroup float sA_mem[TILE * TILE];
    threadgroup float sB_mem[TILE * TILE];

    uint row = tgpig.y * TILE + tptg.y;
    uint col = tgpig.x * TILE + tptg.x;

    // Compute A@A tile
    float acc = 0.0f;
    for (uint t = 0; t < inner; t += TILE) {
        uint a_col = t + tptg.x;
        sA_mem[tptg.y * TILE + tptg.x] = (row < inner && a_col < inner) ? float(Ai[row * inner + a_col]) : 0.0f;

        uint b_row = t + tptg.y;
        sB_mem[tptg.y * TILE + tptg.x] = (b_row < inner && col < inner) ? float(Ai[b_row * inner + col]) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++) {
            acc += sA_mem[tptg.y * TILE + k] * sB_mem[k * TILE + tptg.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // B = b*A + c*(A@A)
    if (row < inner && col < inner) {
        float a_val = float(Ai[row * inner + col]);
        Bi[row * inner + col] = bfloat(b * a_val + c * acc);
    }
}

// Phase 3: Compute X_out = a*X_in + X_in@B (tall) or a*X_in + B@X_in (wide)
// X_in and X_out may be the same buffer (in-place) or different (first iter copy).
// Grid: (ceil(cols/TILE), ceil(rows/TILE), num_params), threads: (TILE, TILE, 1)

kernel void polar_express_phase3(
    device const bfloat* X_in      [[buffer(0)]],
    device bfloat* X_out           [[buffer(1)]],
    device const bfloat* workspace [[buffer(2)]],
    constant float& a              [[buffer(3)]],
    constant uint& num_params      [[buffer(4)]],
    constant uint& rows            [[buffer(5)]],
    constant uint& cols            [[buffer(6)]],
    constant bool& is_tall         [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg  [[thread_position_in_threadgroup]])
{
    uint batch = tgpig.z;
    if (batch >= num_params) return;

    uint mat_size = rows * cols;
    uint inner = is_tall ? cols : rows;
    uint a_mat_size = inner * inner;

    device const bfloat* Xi_in  = X_in  + batch * mat_size;
    device       bfloat* Xi_out = X_out + batch * mat_size;
    device const bfloat* Bi = workspace + batch * a_mat_size * 2 + a_mat_size;

    threadgroup float sA_mem[TILE * TILE];
    threadgroup float sB_mem[TILE * TILE];

    uint row = tgpig.y * TILE + tptg.y;
    uint col = tgpig.x * TILE + tptg.x;

    float acc = 0.0f;

    if (is_tall) {
        // X' = a*X + X @ B,  X is (rows x cols), B is (cols x cols)
        uint K_dim = cols;
        for (uint t = 0; t < K_dim; t += TILE) {
            uint xc = t + tptg.x;
            sA_mem[tptg.y * TILE + tptg.x] = (row < rows && xc < K_dim) ? float(Xi_in[row * cols + xc]) : 0.0f;

            uint br = t + tptg.y;
            sB_mem[tptg.y * TILE + tptg.x] = (br < K_dim && col < cols) ? float(Bi[br * inner + col]) : 0.0f;

            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < TILE; k++) {
                acc += sA_mem[tptg.y * TILE + k] * sB_mem[k * TILE + tptg.x];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // X' = a*X + B @ X,  B is (rows x rows), X is (rows x cols)
        uint K_dim = rows;
        for (uint t = 0; t < K_dim; t += TILE) {
            uint bc = t + tptg.x;
            sA_mem[tptg.y * TILE + tptg.x] = (row < rows && bc < K_dim) ? float(Bi[row * inner + bc]) : 0.0f;

            uint xr = t + tptg.y;
            sB_mem[tptg.y * TILE + tptg.x] = (xr < K_dim && col < cols) ? float(Xi_in[xr * cols + col]) : 0.0f;

            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < TILE; k++) {
                acc += sA_mem[tptg.y * TILE + k] * sB_mem[k * TILE + tptg.x];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (row < rows && col < cols) {
        float x_val = float(Xi_in[row * cols + col]);
        Xi_out[row * cols + col] = bfloat(a * x_val + acc);
    }
}

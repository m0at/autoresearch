#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Fused causal flash attention with sliding window (Flash Attention 2 style)
//
// Layout: Q (B, T, n_head, D), K/V (B, T, n_kv_head, D), O (B, T, n_head, D)
// bf16 inputs/outputs, f32 accumulation.
// RoPE and QK-norm are applied upstream — Q, K arrive ready.
// GQA: multiple query heads share one KV head (n_head / n_kv_head groups).
// ---------------------------------------------------------------------------

constant constexpr uint Br = 16;      // query block rows (reduced for 32KB threadgroup limit)
constant constexpr uint Bc = 16;      // key/value block cols
constant constexpr uint Br_bwd = 16;  // backward Q-block rows
constant constexpr uint Bc_bwd = 8;   // backward KV-block cols (smaller: bwd needs dk/dv accumulators)

// Causal + sliding window mask: returns true if position should be masked (set to -inf)
inline bool is_masked(uint q_pos, uint k_pos, uint window_size) {
    if (k_pos > q_pos) return true;  // causal: can't attend to future
    if (window_size > 0u && (q_pos - k_pos >= window_size)) return true;  // sliding window
    return false;
}

// Helper: bf16 dot product with f32 accumulation over `len` elements
inline float dot_f32(threadgroup const bfloat* a, device const bfloat* b, uint len) {
    float acc = 0.0f;
    for (uint i = 0; i < len; i += 4) {
        acc += float(a[i])     * float(b[i]);
        acc += float(a[i + 1]) * float(b[i + 1]);
        acc += float(a[i + 2]) * float(b[i + 2]);
        acc += float(a[i + 3]) * float(b[i + 3]);
    }
    return acc;
}

inline float dot_f32_tg(threadgroup const bfloat* a, threadgroup const bfloat* b, uint len) {
    float acc = 0.0f;
    for (uint i = 0; i < len; i += 4) {
        acc += float(a[i])     * float(b[i]);
        acc += float(a[i + 1]) * float(b[i + 1]);
        acc += float(a[i + 2]) * float(b[i + 2]);
        acc += float(a[i + 3]) * float(b[i + 3]);
    }
    return acc;
}

kernel void flash_attention_forward(
    device const bfloat* Q         [[buffer(0)]],   // (B, T, n_head, D)
    device const bfloat* K         [[buffer(1)]],   // (B, T, n_kv_head, D)
    device const bfloat* V         [[buffer(2)]],   // (B, T, n_kv_head, D)
    device bfloat* O               [[buffer(3)]],   // (B, T, n_head, D)
    constant uint& B             [[buffer(4)]],
    constant uint& T             [[buffer(5)]],
    constant uint& n_head        [[buffer(6)]],
    constant uint& n_kv_head     [[buffer(7)]],
    constant uint& head_dim      [[buffer(8)]],
    constant uint& window_size   [[buffer(9)]],   // 0 = full causal (no window)
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint3 tid3                   [[thread_position_in_threadgroup]],
    uint3 tg_size3               [[threads_per_threadgroup]]
) {
    const uint tid = tid3.x;
    const uint tg_size = tg_size3.x;
    // -----------------------------------------------------------------------
    // Grid mapping: tgid.x = q-block index, tgid.y = head, tgid.z = batch
    // Each threadgroup processes one (batch, head, q_block).
    // -----------------------------------------------------------------------
    const uint batch_idx = tgid.z;
    const uint head_idx  = tgid.y;
    const uint q_block   = tgid.x;

    if (batch_idx >= B || head_idx >= n_head) return;

    const uint kv_head_idx  = head_idx / (n_head / n_kv_head);  // GQA mapping
    const float scale       = rsqrt(float(head_dim));
    const uint q_start      = q_block * Br;
    const uint q_end        = min(q_start + Br, T);
    const uint q_count      = q_end - q_start;

    // Strides for (B, T, n_head/n_kv_head, D) layout
    const uint q_batch_stride  = T * n_head * head_dim;
    const uint q_seq_stride    = n_head * head_dim;
    const uint kv_batch_stride = T * n_kv_head * head_dim;
    const uint kv_seq_stride   = n_kv_head * head_dim;

    // Base pointers for this batch and head
    device const bfloat* Q_base = Q + batch_idx * q_batch_stride + head_idx * head_dim;
    device const bfloat* K_base = K + batch_idx * kv_batch_stride + kv_head_idx * head_dim;
    device const bfloat* V_base = V + batch_idx * kv_batch_stride + kv_head_idx * head_dim;
    device bfloat*       O_base = O + batch_idx * q_batch_stride + head_idx * head_dim;

    // -----------------------------------------------------------------------
    // Threadgroup shared memory
    //   q_tile: Br x head_dim  (query tile)
    //   k_tile: Bc x head_dim  (key tile)
    //   v_tile: Bc x head_dim  (value tile)
    //   s_tile: Br x Bc        (attention scores, f32)
    //   o_acc:  Br x head_dim  (output accumulator, f32)
    //   m_row:  Br             (running row max, f32)
    //   l_row:  Br             (running row sum, f32)
    // -----------------------------------------------------------------------
    threadgroup bfloat  q_tile[Br * 128];       // max head_dim = 128: 16*128*2 = 4096
    threadgroup bfloat  k_tile[Bc * 128];       // 4096
    threadgroup bfloat  v_tile[Bc * 128];       // 4096
    threadgroup float s_tile[Br * Bc];        // 16*16*4 = 1024
    threadgroup float o_acc [Br * 128];       // 16*128*4 = 8192
    threadgroup float m_row [Br];             // 64
    threadgroup float l_row [Br];             // 64  = 21504 total

    // -----------------------------------------------------------------------
    // Load Q tile into threadgroup memory
    // -----------------------------------------------------------------------
    const uint total_q_elems = q_count * head_dim;
    for (uint i = tid; i < total_q_elems; i += tg_size) {
        uint row = i / head_dim;
        uint col = i % head_dim;
        q_tile[row * head_dim + col] = Q_base[(q_start + row) * q_seq_stride + col];
    }

    // Initialize accumulators
    for (uint i = tid; i < q_count * head_dim; i += tg_size) {
        o_acc[i] = 0.0f;
    }
    for (uint i = tid; i < q_count; i += tg_size) {
        m_row[i] = -INFINITY;
        l_row[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Determine K/V iteration range
    // For causal attention, we only need kv positions <= max query position.
    // With sliding window, we also skip positions too far back.
    // -----------------------------------------------------------------------
    const uint max_q_pos = q_end - 1;
    const uint kv_end    = min(max_q_pos + 1, T);  // causal: can't attend past our position

    uint kv_start = 0;
    if (window_size > 0 && q_start >= window_size) {
        kv_start = q_start - window_size + 1;
    }
    // Align kv_start down to Bc boundary for tiling
    kv_start = (kv_start / Bc) * Bc;

    // -----------------------------------------------------------------------
    // Inner loop: iterate over K, V blocks
    // -----------------------------------------------------------------------
    const uint num_kv_blocks = (kv_end + Bc - 1) / Bc;
    const uint first_kv_block = kv_start / Bc;

    for (uint kv_block = first_kv_block; kv_block < num_kv_blocks; kv_block++) {
        const uint k_start = kv_block * Bc;
        const uint k_end   = min(k_start + Bc, T);
        const uint k_count = k_end - k_start;

        // Load K tile
        const uint total_k_elems = k_count * head_dim;
        for (uint i = tid; i < total_k_elems; i += tg_size) {
            uint row = i / head_dim;
            uint col = i % head_dim;
            k_tile[row * head_dim + col] = K_base[(k_start + row) * kv_seq_stride + col];
        }

        // Load V tile
        for (uint i = tid; i < total_k_elems; i += tg_size) {
            uint row = i / head_dim;
            uint col = i % head_dim;
            v_tile[row * head_dim + col] = V_base[(k_start + row) * kv_seq_stride + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -------------------------------------------------------------------
        // Compute S = Q_tile @ K_tile^T (Br x Bc), with masking
        // Each thread computes one or more (i, j) entries
        // -------------------------------------------------------------------
        const uint s_count = q_count * k_count;
        for (uint idx = tid; idx < s_count; idx += tg_size) {
            uint i = idx / k_count;  // row in q_tile
            uint j = idx % k_count;  // col in k_tile

            uint q_pos = q_start + i;
            uint k_pos = k_start + j;

            // Dot product Q[i] . K[j] with f32 accumulation
            float dot = dot_f32_tg(
                q_tile + i * head_dim,
                k_tile + j * head_dim,
                head_dim
            );
            dot *= scale;

            if (is_masked(q_pos, k_pos, window_size)) dot = -INFINITY;
            s_tile[i * Bc + j] = dot;
        }

        // Pad unused entries in s_tile to -inf for partial blocks
        for (uint idx = tid; idx < Br * Bc; idx += tg_size) {
            uint i = idx / Bc;
            uint j = idx % Bc;
            if (i >= q_count || j >= k_count) {
                s_tile[idx] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -------------------------------------------------------------------
        // Online softmax update (Flash Attention 2 algorithm)
        //   16 threads per query row (256 threads / 16 rows = 16 lanes),
        //   each lane handles head_dim/16 = 8 elements of the O accumulator.
        //   row_max/row_sum computed redundantly by all 16 lanes (Bc=16 is tiny).
        //   exp() values computed inline to avoid s_tile write races.
        // -------------------------------------------------------------------
        {
            const uint row_id = tid / 16;   // which query row (0..15)
            const uint lane   = tid % 16;   // lane within that row (0..15)
            const uint d_start = lane * 8;
            const uint d_end   = d_start + 8;

            if (row_id < q_count) {
                // Redundant per-lane: find row max
                float row_max = -INFINITY;
                for (uint j = 0; j < Bc; j++) {
                    row_max = max(row_max, s_tile[row_id * Bc + j]);
                }

                float m_old = m_row[row_id];
                float m_new = max(m_old, row_max);

                // Guard: if m_new is still -inf, all scores (this tile and prior)
                // are masked — skip to avoid exp(-inf - (-inf)) = NaN.
                if (m_new > -INFINITY) {
                    // When m_old was -inf (first tile with valid scores), correction
                    // is 0 — o_acc is all zeros so o_acc * 0 = 0 (correct).
                    float correction = (m_old > -INFINITY) ? exp(m_old - m_new) : 0.0f;

                    // Redundant per-lane: compute row sum
                    float row_sum = 0.0f;
                    for (uint j = 0; j < k_count; j++) {
                        row_sum += exp(s_tile[row_id * Bc + j] - m_new);
                    }

                    // Rescale existing accumulator
                    float l_old = l_row[row_id];
                    float l_new = l_old * correction + row_sum;

                    // Update O for this lane's 8 elements: O = O * correction + P_row @ V
                    // Compute exp() inline — do NOT write back to s_tile (race-free)
                    for (uint d = d_start; d < d_end && d < head_dim; d++) {
                        float o_val = o_acc[row_id * head_dim + d] * correction;
                        float v_sum = 0.0f;
                        for (uint j = 0; j < k_count; j++) {
                            v_sum += exp(s_tile[row_id * Bc + j] - m_new) * float(v_tile[j * head_dim + d]);
                        }
                        o_acc[row_id * head_dim + d] = o_val + v_sum;
                    }

                    // Only lane 0 writes m_row and l_row
                    if (lane == 0) {
                        m_row[row_id] = m_new;
                        l_row[row_id] = l_new;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -----------------------------------------------------------------------
    // Final rescale: O[i] /= l_row[i] and write to global memory
    // 16 threads per query row, each handling 8 elements of head_dim
    // -----------------------------------------------------------------------
    {
        const uint row_id = tid / 16;
        const uint lane   = tid % 16;
        const uint d_start = lane * 8;
        const uint d_end   = d_start + 8;

        if (row_id < q_count) {
            float inv_sum = select(1.0f / l_row[row_id], 0.0f, l_row[row_id] == 0.0f);
            for (uint d = d_start; d < d_end && d < head_dim; d++) {
                O_base[(q_start + row_id) * q_seq_stride + d] =
                    bfloat(o_acc[row_id * head_dim + d] * inv_sum);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flash Attention 2 — Backward pass
//
// Split into two kernels to avoid atomics on M5 (no FP32 atomic_add):
//   1. flash_attention_bwd_dq:  grid over Q blocks, accumulates dQ per block
//   2. flash_attention_bwd_dkv: grid over KV blocks, accumulates dK/dV per block
//
// Both kernels recompute the attention matrix S from Q,K in tiles, apply
// softmax (recomputed via online softmax), and compute the relevant gradients.
// ---------------------------------------------------------------------------

// Helper: dot product from device memory with f32 accumulation
inline float dot_f32_dev(device const bfloat* a, device const bfloat* b, uint len) {
    float acc = 0.0f;
    for (uint i = 0; i < len; i += 4) {
        acc += float(a[i])     * float(b[i]);
        acc += float(a[i + 1]) * float(b[i + 1]);
        acc += float(a[i + 2]) * float(b[i + 2]);
        acc += float(a[i + 3]) * float(b[i + 3]);
    }
    return acc;
}

inline float dot_f32_tg_dev(threadgroup const bfloat* a, device const bfloat* b, uint len) {
    float acc = 0.0f;
    for (uint i = 0; i < len; i += 4) {
        acc += float(a[i])     * float(b[i]);
        acc += float(a[i + 1]) * float(b[i + 1]);
        acc += float(a[i + 2]) * float(b[i + 2]);
        acc += float(a[i + 3]) * float(b[i + 3]);
    }
    return acc;
}

// =========================================================================
// Kernel 1: Compute dQ
//   Grid: (num_q_blocks, n_head, B)
//   Each threadgroup owns one Q block, iterates over KV blocks.
//   dQ is accumulated in threadgroup f32, written once — no atomics.
// =========================================================================
kernel void flash_attention_bwd_dq(
    device const bfloat* Q          [[buffer(0)]],
    device const bfloat* K          [[buffer(1)]],
    device const bfloat* V          [[buffer(2)]],
    device const bfloat* O          [[buffer(3)]],
    device const bfloat* dO         [[buffer(4)]],
    device bfloat* dQ               [[buffer(5)]],
    constant uint& B              [[buffer(6)]],
    constant uint& T              [[buffer(7)]],
    constant uint& n_head         [[buffer(8)]],
    constant uint& n_kv_head      [[buffer(9)]],
    constant uint& head_dim       [[buffer(10)]],
    constant uint& window_size    [[buffer(11)]],
    uint3 tgid                    [[threadgroup_position_in_grid]],
    uint3 tid3                    [[thread_position_in_threadgroup]],
    uint3 tg_size3                [[threads_per_threadgroup]]
) {
    const uint tid = tid3.x;
    const uint tg_size = tg_size3.x;
    const uint batch_idx = tgid.z;
    const uint head_idx  = tgid.y;
    const uint q_block   = tgid.x;

    if (batch_idx >= B || head_idx >= n_head) return;

    const uint kv_head_idx = head_idx / (n_head / n_kv_head);
    const float scale      = rsqrt(float(head_dim));
    const uint q_start     = q_block * Br_bwd;
    const uint q_end       = min(q_start + Br_bwd, T);
    const uint q_count     = q_end - q_start;

    const uint q_batch_stride  = T * n_head * head_dim;
    const uint q_seq_stride    = n_head * head_dim;
    const uint kv_batch_stride = T * n_kv_head * head_dim;
    const uint kv_seq_stride   = n_kv_head * head_dim;

    device const bfloat* Q_base  = Q  + batch_idx * q_batch_stride + head_idx * head_dim;
    device const bfloat* K_base  = K  + batch_idx * kv_batch_stride + kv_head_idx * head_dim;
    device const bfloat* V_base  = V  + batch_idx * kv_batch_stride + kv_head_idx * head_dim;
    device const bfloat* O_base  = O  + batch_idx * q_batch_stride + head_idx * head_dim;
    device const bfloat* dO_base = dO + batch_idx * q_batch_stride + head_idx * head_dim;
    device bfloat*       dQ_base = dQ + batch_idx * q_batch_stride + head_idx * head_dim;

    // Shared memory — sized with Br_bwd/Bc_bwd to fit within 32KB
    threadgroup bfloat  q_tile [Br_bwd * 128];
    threadgroup bfloat  k_tile [Bc_bwd * 128];
    threadgroup bfloat  v_tile [Bc_bwd * 128];
    threadgroup bfloat  o_tile [Br_bwd * 128];
    threadgroup bfloat  do_tile[Br_bwd * 128];
    threadgroup float s_tile [Br_bwd * Bc_bwd];
    threadgroup float dq_acc [Br_bwd * 128];  // f32 dQ accumulator
    threadgroup float m_row  [Br_bwd];        // softmax row max
    threadgroup float l_row  [Br_bwd];        // softmax row sum
    threadgroup float D_row  [Br_bwd];        // D_i = rowsum(dO_i * O_i)
    threadgroup float scratch[Br_bwd * 16];   // lane-parallel reduction scratch

    // 16 threads per query row: row_id = tid/16, lane = tid%16
    // Each lane handles head_dim/16 elements (4 for d=64, 8 for d=128)
    constexpr uint LANES = 16;
    const uint elems_per_lane = head_dim / LANES;
    const uint row_id = tid / LANES;
    const uint lane   = tid % LANES;

    // Load Q, O, dO tiles
    for (uint i = tid; i < q_count * head_dim; i += tg_size) {
        uint row = i / head_dim;
        uint col = i % head_dim;
        q_tile [row * head_dim + col] = Q_base [(q_start + row) * q_seq_stride + col];
        o_tile [row * head_dim + col] = O_base [(q_start + row) * q_seq_stride + col];
        do_tile[row * head_dim + col] = dO_base[(q_start + row) * q_seq_stride + col];
    }

    // Init dQ accumulator to zero
    for (uint i = tid; i < q_count * head_dim; i += tg_size) {
        dq_acc[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute D_i = rowsum(dO_i * O_i) — 16 lanes per row, reduce via scratch
    if (row_id < q_count) {
        float partial = 0.0f;
        const uint d_base = lane * elems_per_lane;
        for (uint dd = d_base; dd < d_base + elems_per_lane; dd++) {
            partial += float(do_tile[row_id * head_dim + dd]) * float(o_tile[row_id * head_dim + dd]);
        }
        scratch[row_id * LANES + lane] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row_id < q_count && lane == 0) {
        float d = 0.0f;
        for (uint ll = 0; ll < LANES; ll++) {
            d += scratch[row_id * LANES + ll];
        }
        D_row[row_id] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Pass 1: Compute softmax stats (m, l) — identical to forward
    // -----------------------------------------------------------------------
    for (uint i = tid; i < q_count; i += tg_size) {
        m_row[i] = -INFINITY;
        l_row[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Determine KV range (same logic as forward)
    const uint max_q_pos = q_end - 1;
    const uint kv_end    = min(max_q_pos + 1, T);
    uint kv_start_pos = 0;
    if (window_size > 0 && q_start >= window_size) {
        kv_start_pos = q_start - window_size + 1;
    }
    kv_start_pos = (kv_start_pos / Bc_bwd) * Bc_bwd;
    const uint num_kv_blocks  = (kv_end + Bc_bwd - 1) / Bc_bwd;
    const uint first_kv_block = kv_start_pos / Bc_bwd;

    // Pass 1: compute m_row, l_row by iterating over KV blocks
    for (uint kv_block = first_kv_block; kv_block < num_kv_blocks; kv_block++) {
        const uint k_start = kv_block * Bc_bwd;
        const uint k_end_b = min(k_start + Bc_bwd, T);
        const uint k_count = k_end_b - k_start;

        // Load K tile
        for (uint i = tid; i < k_count * head_dim; i += tg_size) {
            uint row = i / head_dim;
            uint col = i % head_dim;
            k_tile[row * head_dim + col] = K_base[(k_start + row) * kv_seq_stride + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute S = Q @ K^T with masking
        const uint s_count = q_count * k_count;
        for (uint idx = tid; idx < s_count; idx += tg_size) {
            uint i = idx / k_count;
            uint j = idx % k_count;
            uint q_pos = q_start + i;
            uint k_pos = k_start + j;

            float dot = dot_f32_tg(q_tile + i * head_dim, k_tile + j * head_dim, head_dim);
            dot *= scale;
            if (is_masked(q_pos, k_pos, window_size)) dot = -INFINITY;
            s_tile[i * Bc_bwd + j] = dot;
        }
        // Pad unused entries
        for (uint idx = tid; idx < Br_bwd * Bc_bwd; idx += tg_size) {
            uint i = idx / Bc_bwd;
            uint j = idx % Bc_bwd;
            if (i >= q_count || j >= k_count) s_tile[idx] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax stats update — all 16 lanes per row redundantly
        // compute row max/sum (Bc_bwd=8 is tiny), lane 0 writes stats
        if (row_id < q_count) {
            float row_max = -INFINITY;
            for (uint j = 0; j < Bc_bwd; j++) {
                row_max = max(row_max, s_tile[row_id * Bc_bwd + j]);
            }
            float m_old = m_row[row_id];
            float m_new = max(m_old, row_max);
            // Guard: skip if m_new is -inf (all scores masked)
            if (m_new > -INFINITY) {
                float correction = (m_old > -INFINITY) ? exp(m_old - m_new) : 0.0f;
                float row_sum = 0.0f;
                for (uint j = 0; j < k_count; j++) {
                    row_sum += exp(s_tile[row_id * Bc_bwd + j] - m_new);
                }
                if (lane == 0) {
                    l_row[row_id] = l_row[row_id] * correction + row_sum;
                    m_row[row_id] = m_new;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -----------------------------------------------------------------------
    // Pass 2: Compute dQ by iterating over KV blocks again
    // For each KV block, recompute P_ij = softmax(S_ij) using stored m, l
    // then dS_ij = P_ij * (dO @ V^T - D_i), dQ += dS @ K * scale
    //
    // 16 threads per query row. Each lane:
    //   - Computes partial dot products over its elems_per_lane head_dim elements
    //   - Accumulates dQ[i, lane*epl..(lane+1)*epl] directly
    // -----------------------------------------------------------------------
    for (uint kv_block = first_kv_block; kv_block < num_kv_blocks; kv_block++) {
        const uint k_start = kv_block * Bc_bwd;
        const uint k_end_b = min(k_start + Bc_bwd, T);
        const uint k_count = k_end_b - k_start;

        // Load K, V tiles
        for (uint i = tid; i < k_count * head_dim; i += tg_size) {
            uint row = i / head_dim;
            uint col = i % head_dim;
            k_tile[row * head_dim + col] = K_base[(k_start + row) * kv_seq_stride + col];
            v_tile[row * head_dim + col] = V_base[(k_start + row) * kv_seq_stride + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Fused per-row: recompute P, compute dS, accumulate dQ
        // 4 barriers per j iteration to protect scratch reuse
        if (row_id < q_count) {
            const uint d_start = lane * elems_per_lane;
            for (uint j = 0; j < k_count; j++) {
                uint q_pos = q_start + row_id;
                uint k_pos = k_start + j;

                // Step 1: Recompute S_ij via lane-parallel dot, reduce for P_ij
                float partial_qk = 0.0f;
                for (uint dd = d_start; dd < d_start + elems_per_lane; dd++) {
                    partial_qk += float(q_tile[row_id * head_dim + dd]) * float(k_tile[j * head_dim + dd]);
                }
                scratch[row_id * LANES + lane] = partial_qk;
                threadgroup_barrier(mem_flags::mem_threadgroup);  // 1: scratch readable

                float dot_qk = 0.0f;
                for (uint ll = 0; ll < LANES; ll++) {
                    dot_qk += scratch[row_id * LANES + ll];
                }
                dot_qk *= scale;
                // Guard: if m_row is -inf (all scores masked for this row),
                // or position is masked, p_ij = 0.
                float p_ij = (is_masked(q_pos, k_pos, window_size) || m_row[row_id] == -INFINITY)
                    ? 0.0f
                    : exp(dot_qk - m_row[row_id]) / l_row[row_id];

                threadgroup_barrier(mem_flags::mem_threadgroup);  // 2: safe to reuse scratch

                // Step 2: dP_ij = dO[i].V[j] via lane-parallel dot, reduce
                float partial_dp = 0.0f;
                for (uint dd = d_start; dd < d_start + elems_per_lane; dd++) {
                    partial_dp += float(do_tile[row_id * head_dim + dd]) * float(v_tile[j * head_dim + dd]);
                }
                scratch[row_id * LANES + lane] = partial_dp;
                threadgroup_barrier(mem_flags::mem_threadgroup);  // 3: scratch readable

                float dp_ij = 0.0f;
                for (uint ll = 0; ll < LANES; ll++) {
                    dp_ij += scratch[row_id * LANES + ll];
                }

                // Step 3: dS_ij = P_ij * (dP_ij - D_i)
                float ds_ij = p_ij * (dp_ij - D_row[row_id]);

                // Step 4: dQ[i,d] += dS_ij * K[j,d] * scale — each lane owns its d's
                float ds_scaled = ds_ij * scale;
                for (uint dd = d_start; dd < d_start + elems_per_lane; dd++) {
                    dq_acc[row_id * head_dim + dd] += ds_scaled * float(k_tile[j * head_dim + dd]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);  // 4: safe for next iteration
            }
        } else {
            // Threads beyond q_count*16 still need to hit all 4 barriers per j
            for (uint j = 0; j < k_count; j++) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // Write dQ to global memory — 16 lanes per row, each writes its elements
    if (row_id < q_count) {
        const uint d_start = lane * elems_per_lane;
        for (uint dd = d_start; dd < d_start + elems_per_lane; dd++) {
            dQ_base[(q_start + row_id) * q_seq_stride + dd] = bfloat(dq_acc[row_id * head_dim + dd]);
        }
    }
}

// =========================================================================
// Kernel 2: Compute dK, dV
//   Grid: (num_kv_blocks, n_kv_head, B)
//   Each threadgroup owns one KV block, iterates over Q blocks.
//   For GQA, iterates over all query heads that map to this KV head.
//   dK/dV accumulated in threadgroup f32, written once — no atomics.
// =========================================================================
kernel void flash_attention_bwd_dkv(
    device const bfloat* Q          [[buffer(0)]],
    device const bfloat* K          [[buffer(1)]],
    device const bfloat* V          [[buffer(2)]],
    device const bfloat* O          [[buffer(3)]],
    device const bfloat* dO         [[buffer(4)]],
    device bfloat* dK               [[buffer(5)]],
    device bfloat* dV               [[buffer(6)]],
    constant uint& B              [[buffer(7)]],
    constant uint& T              [[buffer(8)]],
    constant uint& n_head         [[buffer(9)]],
    constant uint& n_kv_head      [[buffer(10)]],
    constant uint& head_dim       [[buffer(11)]],
    constant uint& window_size    [[buffer(12)]],
    uint3 tgid                    [[threadgroup_position_in_grid]],
    uint3 tid3                    [[thread_position_in_threadgroup]],
    uint3 tg_size3                [[threads_per_threadgroup]]
) {
    const uint tid = tid3.x;
    const uint tg_size = tg_size3.x;
    const uint batch_idx   = tgid.z;
    const uint kv_head_idx = tgid.y;
    const uint kv_block    = tgid.x;

    if (batch_idx >= B || kv_head_idx >= n_kv_head) return;

    const float scale      = rsqrt(float(head_dim));
    const uint k_start     = kv_block * Bc_bwd;
    const uint k_end       = min(k_start + Bc_bwd, T);
    const uint k_count     = k_end - k_start;

    const uint heads_per_kv = n_head / n_kv_head;
    const uint first_q_head = kv_head_idx * heads_per_kv;

    const uint q_batch_stride  = T * n_head * head_dim;
    const uint q_seq_stride    = n_head * head_dim;
    const uint kv_batch_stride = T * n_kv_head * head_dim;
    const uint kv_seq_stride   = n_kv_head * head_dim;

    device const bfloat* K_base  = K  + batch_idx * kv_batch_stride + kv_head_idx * head_dim;
    device const bfloat* V_base  = V  + batch_idx * kv_batch_stride + kv_head_idx * head_dim;
    device bfloat*       dK_base = dK + batch_idx * kv_batch_stride + kv_head_idx * head_dim;
    device bfloat*       dV_base = dV + batch_idx * kv_batch_stride + kv_head_idx * head_dim;

    // Thread utilization: 16 threads per Q row (256 threads / 16 Q rows)
    // row = tid / 16, lane = tid % 16, each lane handles head_dim/16 = 8 elements
    const uint THREADS_PER_ROW = 16;  // 256 / Br_bwd
    const uint row_id = tid / THREADS_PER_ROW;     // which Q row this thread works on
    const uint lane_id = tid % THREADS_PER_ROW;    // lane within the row team

    // Shared memory — sized with Br_bwd/Bc_bwd to fit within 32KB
    threadgroup bfloat  k_tile [Bc_bwd * 128];
    threadgroup bfloat  v_tile [Bc_bwd * 128];
    threadgroup bfloat  q_tile [Br_bwd * 128];
    threadgroup bfloat  do_tile[Br_bwd * 128];
    threadgroup bfloat  o_tile [Br_bwd * 128];
    threadgroup float s_tile [Br_bwd * Bc_bwd];  // attention scores / softmax probs
    threadgroup float dk_acc [Bc_bwd * 128];      // f32 dK accumulator
    threadgroup float dv_acc [Bc_bwd * 128];      // f32 dV accumulator
    threadgroup float m_row  [Br_bwd];            // softmax row max (per Q row)
    threadgroup float l_row  [Br_bwd];            // softmax row sum
    threadgroup float D_row  [Br_bwd];            // D_i = rowsum(dO * O)
    threadgroup float reduce_buf[Br_bwd * THREADS_PER_ROW]; // partial sums for reduction

    // Load K, V tiles (constant across all Q blocks and query heads)
    for (uint i = tid; i < k_count * head_dim; i += tg_size) {
        uint row = i / head_dim;
        uint col = i % head_dim;
        k_tile[row * head_dim + col] = K_base[(k_start + row) * kv_seq_stride + col];
        v_tile[row * head_dim + col] = V_base[(k_start + row) * kv_seq_stride + col];
    }

    // Init dK, dV accumulators to zero
    for (uint i = tid; i < k_count * head_dim; i += tg_size) {
        dk_acc[i] = 0.0f;
        dv_acc[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over all query heads that map to this KV head
    for (uint qh = 0; qh < heads_per_kv; qh++) {
        const uint head_idx = first_q_head + qh;

        device const bfloat* Q_base  = Q  + batch_idx * q_batch_stride + head_idx * head_dim;
        device const bfloat* O_base  = O  + batch_idx * q_batch_stride + head_idx * head_dim;
        device const bfloat* dO_base = dO + batch_idx * q_batch_stride + head_idx * head_dim;

        // Determine which Q blocks can attend to this KV block
        // A Q row at position q_pos attends to k_pos if:
        //   q_pos >= k_pos (causal) AND (window_size==0 OR q_pos - k_pos < window_size)
        // So Q rows that attend to k_start..k_end are in range [k_start, ...].
        // The earliest Q that could attend to k_end-1 is k_end-1.
        // The latest Q that could attend to the last K in this block (k_end-1)
        // is k_end - 1 + window_size - 1 = k_end + window_size - 2, so
        // q_range_end (exclusive) = k_end + window_size - 1.
        const uint q_range_start = k_start;  // causal: q >= k
        uint q_range_end = T;
        if (window_size > 0) {
            q_range_end = min(T, k_end + window_size - 1);
        }
        if (q_range_start >= q_range_end) continue;

        const uint first_q_block = q_range_start / Br_bwd;
        const uint last_q_block  = (q_range_end + Br_bwd - 1) / Br_bwd;

        // For each Q block, we need the full softmax stats for those Q rows.
        // We must do a two-pass approach: first compute m,l over ALL KV blocks
        // for each Q block, then use those to compute dK/dV contributions.
        //
        // However, to avoid a third pass, we do it per-Q-block:
        //   For each Q block, compute m_row/l_row across all KV, then compute
        //   P and dK/dV contributions from just this KV block.

        for (uint qb = first_q_block; qb < last_q_block; qb++) {
            const uint q_start_b = qb * Br_bwd;
            const uint q_end_b   = min(q_start_b + Br_bwd, T);
            const uint q_count   = q_end_b - q_start_b;

            // Load Q, O, dO tiles for this Q block
            for (uint i = tid; i < q_count * head_dim; i += tg_size) {
                uint row = i / head_dim;
                uint col = i % head_dim;
                q_tile [row * head_dim + col] = Q_base [(q_start_b + row) * q_seq_stride + col];
                o_tile [row * head_dim + col] = O_base [(q_start_b + row) * q_seq_stride + col];
                do_tile[row * head_dim + col] = dO_base[(q_start_b + row) * q_seq_stride + col];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute D_i = rowsum(dO_i * O_i) — 16 threads per row
            // Each lane handles head_dim/16 = 8 elements, then reduce
            if (row_id < q_count) {
                const uint elems_per_lane = head_dim / THREADS_PER_ROW;  // 128/16 = 8
                const uint base = row_id * head_dim + lane_id * elems_per_lane;
                float partial = 0.0f;
                for (uint dd = 0; dd < elems_per_lane; dd++) {
                    partial += float(do_tile[base + dd]) * float(o_tile[base + dd]);
                }
                reduce_buf[row_id * THREADS_PER_ROW + lane_id] = partial;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Lane 0 of each row reduces and writes D_row
            if (row_id < q_count && lane_id == 0) {
                float d = 0.0f;
                for (uint ll = 0; ll < THREADS_PER_ROW; ll++) {
                    d += reduce_buf[row_id * THREADS_PER_ROW + ll];
                }
                D_row[row_id] = d;
            }

            // Init softmax stats
            for (uint i = tid; i < q_count; i += tg_size) {
                m_row[i] = -INFINITY;
                l_row[i] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Pass 1: compute m_row, l_row over ALL KV blocks for this Q block
            uint kv_start_pos = 0;
            const uint max_q_pos = q_end_b - 1;
            const uint kv_end_pos = min(max_q_pos + 1, T);
            if (window_size > 0 && q_start_b >= window_size) {
                kv_start_pos = q_start_b - window_size + 1;
            }
            kv_start_pos = (kv_start_pos / Bc_bwd) * Bc_bwd;
            const uint num_kv_blocks  = (kv_end_pos + Bc_bwd - 1) / Bc_bwd;
            const uint first_kv_block = kv_start_pos / Bc_bwd;

            // We need a temporary tile for K during softmax stats computation
            // but k_tile is occupied with our target KV block. Use s_tile area
            // creatively, or do the computation using device memory directly.
            // For simplicity and correctness, compute softmax stats from device mem.
            for (uint kvb = first_kv_block; kvb < num_kv_blocks; kvb++) {
                const uint ks = kvb * Bc_bwd;
                const uint ke = min(ks + Bc_bwd, T);
                const uint kc = ke - ks;

                // Compute S = Q_tile @ K_block^T using device memory for K
                // For the target KV block, we can use k_tile; otherwise device mem.
                const uint sc = q_count * kc;
                for (uint idx = tid; idx < sc; idx += tg_size) {
                    uint i = idx / kc;
                    uint j = idx % kc;
                    uint q_pos = q_start_b + i;
                    uint k_pos = ks + j;

                    float dot;
                    if (kvb == kv_block) {
                        dot = dot_f32_tg(q_tile + i * head_dim,
                                         k_tile + j * head_dim, head_dim);
                    } else {
                        dot = dot_f32_tg_dev(q_tile + i * head_dim,
                                             K_base + (ks + j) * kv_seq_stride,
                                             head_dim);
                    }
                    dot *= scale;
                    if (is_masked(q_pos, k_pos, window_size)) dot = -INFINITY;
                    s_tile[i * Bc_bwd + j] = dot;
                }
                for (uint idx = tid; idx < Br_bwd * Bc_bwd; idx += tg_size) {
                    uint i = idx / Bc_bwd;
                    uint j = idx % Bc_bwd;
                    if (i >= q_count || j >= kc) s_tile[idx] = -INFINITY;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Update m_row, l_row — 16 threads per Q row, all 256 active
                // Each lane computes partial max/sum over its share of Bc_bwd columns
                if (row_id < q_count) {
                    // Distribute Bc_bwd (=8) columns across 16 lanes:
                    // lanes 0..7 each handle 1 column, lanes 8..15 are idle for col work
                    float lane_max = -INFINITY;
                    if (lane_id < kc) {
                        lane_max = s_tile[row_id * Bc_bwd + lane_id];
                    }
                    // Reduce max across lanes into reduce_buf
                    reduce_buf[row_id * THREADS_PER_ROW + lane_id] = lane_max;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // Lane 0 computes final max and writes m_new, then computes exp sums
                if (row_id < q_count && lane_id == 0) {
                    float row_max = -INFINITY;
                    for (uint ll = 0; ll < THREADS_PER_ROW; ll++) {
                        row_max = max(row_max, reduce_buf[row_id * THREADS_PER_ROW + ll]);
                    }
                    float m_old = m_row[row_id];
                    float m_new = max(m_old, row_max);
                    // Guard: skip if m_new is -inf (all scores masked)
                    if (m_new > -INFINITY) {
                        float correction = (m_old > -INFINITY) ? exp(m_old - m_new) : 0.0f;
                        float row_sum = 0.0f;
                        for (uint j = 0; j < kc; j++) {
                            row_sum += exp(s_tile[row_id * Bc_bwd + j] - m_new);
                        }
                        l_row[row_id] = l_row[row_id] * correction + row_sum;
                        m_row[row_id] = m_new;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Pass 2: Recompute P for THIS KV block, compute dK/dV contributions
            // Compute S = Q_tile @ K_tile^T (using our loaded k_tile)
            const uint sc2 = q_count * k_count;
            for (uint idx = tid; idx < sc2; idx += tg_size) {
                uint i = idx / k_count;
                uint j = idx % k_count;
                uint q_pos = q_start_b + i;
                uint k_pos = k_start + j;

                float dot = dot_f32_tg(q_tile + i * head_dim,
                                       k_tile + j * head_dim, head_dim);
                dot *= scale;
                if (is_masked(q_pos, k_pos, window_size)) dot = -INFINITY;

                // P_ij = exp(S_ij - m_i) / l_i
                // Guard: if m_row is -inf (all masked), p = 0
                float p = (is_masked(q_pos, k_pos, window_size) || m_row[i] == -INFINITY)
                    ? 0.0f
                    : exp(dot - m_row[i]) / l_row[i];
                s_tile[i * Bc_bwd + j] = p;
            }
            for (uint idx = tid; idx < Br_bwd * Bc_bwd; idx += tg_size) {
                uint i = idx / Bc_bwd;
                uint j = idx % Bc_bwd;
                if (i >= q_count || j >= k_count) s_tile[idx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // dV_acc += P^T @ dO  — for each (j, d): sum_i P[i,j] * dO[i,d]
            for (uint idx = tid; idx < k_count * head_dim; idx += tg_size) {
                uint j = idx / head_dim;
                uint d = idx % head_dim;
                float acc = 0.0f;
                for (uint i = 0; i < q_count; i++) {
                    acc += s_tile[i * Bc_bwd + j] * float(do_tile[i * head_dim + d]);
                }
                dv_acc[j * head_dim + d] += acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup); // sync before dS overwrites s_tile

            // Compute dS_ij = P_ij * (dO[i] @ V[j]^T - D_i)
            for (uint idx = tid; idx < sc2; idx += tg_size) {
                uint i = idx / k_count;
                uint j = idx % k_count;

                float dp = dot_f32_tg(do_tile + i * head_dim,
                                      v_tile + j * head_dim, head_dim);
                float ds = s_tile[i * Bc_bwd + j] * (dp - D_row[i]);
                s_tile[i * Bc_bwd + j] = ds;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // dK_acc += dS^T @ Q * scale — for each (j, d): sum_i dS[i,j] * Q[i,d]
            for (uint idx = tid; idx < k_count * head_dim; idx += tg_size) {
                uint j = idx / head_dim;
                uint d = idx % head_dim;
                float acc = 0.0f;
                for (uint i = 0; i < q_count; i++) {
                    acc += s_tile[i * Bc_bwd + j] * float(q_tile[i * head_dim + d]);
                }
                dk_acc[j * head_dim + d] += acc * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } // end Q block loop
    } // end query head loop

    // Write dK, dV to global memory
    for (uint i = tid; i < k_count * head_dim; i += tg_size) {
        uint row = i / head_dim;
        uint col = i % head_dim;
        dK_base[(k_start + row) * kv_seq_stride + col] = bfloat(dk_acc[i]);
        dV_base[(k_start + row) * kv_seq_stride + col] = bfloat(dv_acc[i]);
    }
}

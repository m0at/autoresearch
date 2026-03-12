#include <metal_stdlib>
using namespace metal;

// ── helpers ──────────────────────────────────────────────────────────────────
// SIMD-first reductions: 1 barrier instead of 2. Valid for threadgroups <= 256
// (i.e. <= 8 SIMD groups, so the final simd_sum covers all partial sums).

inline float reduce_sum(float local_val, uint tid, uint tg_size,
                        threadgroup float* shared) {
    float val = simd_sum(local_val);
    if ((tid & 31) == 0) shared[tid >> 5] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_groups = (tg_size + 31) / 32;
    float total = 0.0f;
    for (uint i = 0; i < n_groups; i++) total += shared[i];
    return total;
}

inline float reduce_max(float local_val, uint tid, uint tg_size,
                        threadgroup float* shared) {
    float val = simd_max(local_val);
    if ((tid & 31) == 0) shared[tid >> 5] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_groups = (tg_size + 31) / 32;
    float m = -INFINITY;
    for (uint i = 0; i < n_groups; i++) m = max(m, shared[i]);
    return m;
}

// ── 1. RMSNorm forward ──────────────────────────────────────────────────────

kernel void fused_rms_norm_fwd(device const bfloat* x        [[buffer(0)]],
                               device       bfloat* y        [[buffer(1)]],
                               constant     uint&   D        [[buffer(2)]],
                               constant     float&  eps      [[buffer(3)]],
                               uint row     [[threadgroup_position_in_grid]],
                               uint tid     [[thread_index_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
    device const bfloat* xr = x + row * D;
    device       bfloat* yr = y + row * D;

    float local_sq = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float v = float(xr[i]);
        local_sq += v * v;
    }

    threadgroup float shared[32];
    float sum_sq = reduce_sum(local_sq, tid, tg_size, shared);

    float rrms = rsqrt(sum_sq / float(D) + eps);

    for (uint i = tid; i < D; i += tg_size) {
        yr[i] = bfloat(float(xr[i]) * rrms);
    }
}

// ── 2. RMSNorm backward ─────────────────────────────────────────────────────

kernel void fused_rms_norm_bwd(device const bfloat* x        [[buffer(0)]],
                               device const bfloat* grad_out [[buffer(1)]],
                               device       bfloat* grad_in  [[buffer(2)]],
                               constant     uint&   D        [[buffer(3)]],
                               constant     float&  eps      [[buffer(4)]],
                               uint row     [[threadgroup_position_in_grid]],
                               uint tid     [[thread_index_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
    device const bfloat* xr  = x        + row * D;
    device const bfloat* gor = grad_out + row * D;
    device       bfloat* gir = grad_in  + row * D;

    float local_sq  = 0.0f;
    float local_dot = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float xi = float(xr[i]);
        float gi = float(gor[i]);
        local_sq  += xi * xi;
        local_dot += gi * xi;
    }

    threadgroup float shared[32];
    float sum_sq = reduce_sum(local_sq, tid, tg_size, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot_gx = reduce_sum(local_dot, tid, tg_size, shared);

    float rrms  = rsqrt(sum_sq / float(D) + eps);
    float coeff = dot_gx / float(D) * rrms * rrms;

    for (uint i = tid; i < D; i += tg_size) {
        float xi = float(xr[i]);
        float gi = float(gor[i]);
        gir[i] = bfloat(rrms * (gi - xi * coeff));
    }
}

// ── 3. 2*sigmoid forward ────────────────────────────────────────────────────

kernel void fused_sigmoid_2x_fwd(device const bfloat* x [[buffer(0)]],
                                 device       bfloat* y [[buffer(1)]],
                                 constant     uint&   N [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    float v = float(x[gid]);
    y[gid] = bfloat(2.0f / (1.0f + exp(-v)));
}

// ── 4. 2*sigmoid backward ───────────────────────────────────────────────────

kernel void fused_sigmoid_2x_bwd(device const bfloat* x        [[buffer(0)]],
                                 device const bfloat* grad_out [[buffer(1)]],
                                 device       bfloat* grad_in  [[buffer(2)]],
                                 constant     uint&   N        [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    float v   = float(x[gid]);
    float sig = 1.0f / (1.0f + exp(-v));
    grad_in[gid] = bfloat(2.0f * sig * (1.0f - sig) * float(grad_out[gid]));
}

// ── 5. ReLU² forward ─────────────────────────────────────────────────────────
// out = max(0, x)²  — fuses relu + square into a single dispatch

kernel void fused_relu_sq_fwd(device const bfloat* x [[buffer(0)]],
                              device       bfloat* y [[buffer(1)]],
                              constant     uint&   N [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    float v = max(0.0f, float(x[gid]));
    y[gid] = bfloat(v * v);
}

// ── 6. ReLU² backward ────────────────────────────────────────────────────────
// grad_in = grad_out * 2 * max(0, x)   (d/dx[relu(x)²] = 2*relu(x)*step(x) = 2*max(0,x))

kernel void fused_relu_sq_bwd(device const bfloat* x        [[buffer(0)]],
                              device const bfloat* grad_out [[buffer(1)]],
                              device       bfloat* grad_in  [[buffer(2)]],
                              constant     uint&   N        [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    float v = max(0.0f, float(x[gid]));
    grad_in[gid] = bfloat(2.0f * v * float(grad_out[gid]));
}

// ── 7. Residual lambda scaling forward ────────────────────────────────────
// out = lambda_r * x + lambda_0 * x0

kernel void fused_residual_scale_fwd(device const bfloat* x        [[buffer(0)]],
                                     device const bfloat* x0       [[buffer(1)]],
                                     device       bfloat* out      [[buffer(2)]],
                                     constant     float&  lambda_r [[buffer(3)]],
                                     constant     float&  lambda_0 [[buffer(4)]],
                                     constant     uint&   N        [[buffer(5)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    out[gid] = bfloat(lambda_r * float(x[gid]) + lambda_0 * float(x0[gid]));
}

// ── 8. Residual lambda scaling backward ──────────────────────────────────
// grad_x  = lambda * grad_out  (reused for both x and x0 grads)

kernel void fused_scale_bwd(device const bfloat* grad_out [[buffer(0)]],
                            device       bfloat* grad_in  [[buffer(1)]],
                            constant     float&  lambda   [[buffer(2)]],
                            constant     uint&   N        [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    grad_in[gid] = bfloat(lambda * float(grad_out[gid]));
}

// ── 9. RoPE forward ─────────────────────────────────────────────────────────
// x:   (B, T, n_head, head_dim) contiguous, flat index N = B*T*n_head*head_dim
// cos: (T, half)                contiguous (squeezed from (1,T,1,half))
// sin: (T, half)                contiguous
// half = head_dim / 2
// For d < half:  out[..d]      = x[..d]*cos[t,d]      + x[..d+half]*sin[t,d]
// For d >= half: out[..d] = -x[..d-half]*sin[t,d-half] + x[..d]*cos[t,d-half]

kernel void fused_rope_fwd(device const bfloat* x      [[buffer(0)]],
                            device const bfloat* cos_t  [[buffer(1)]],
                            device const bfloat* sin_t  [[buffer(2)]],
                            device       bfloat* out    [[buffer(3)]],
                            constant     uint&   N      [[buffer(4)]],
                            constant     uint&   T      [[buffer(5)]],
                            constant     uint&   n_head [[buffer(6)]],
                            constant     uint&   hdim   [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    uint half_d = hdim / 2;
    uint d = gid % hdim;
    uint rem = gid / hdim;
    // skip h: rem = rem / n_head; but we just need t
    rem = rem / n_head;
    uint t = rem % T;
    if (d < half_d) {
        uint cs_idx = t * half_d + d;
        float c = float(cos_t[cs_idx]);
        float s = float(sin_t[cs_idx]);
        float x1 = float(x[gid]);
        float x2 = float(x[gid + half_d]);
        out[gid] = bfloat(x1 * c + x2 * s);
    } else {
        uint d2 = d - half_d;
        uint cs_idx = t * half_d + d2;
        float c = float(cos_t[cs_idx]);
        float s = float(sin_t[cs_idx]);
        float x1 = float(x[gid - half_d]);
        float x2 = float(x[gid]);
        out[gid] = bfloat(-x1 * s + x2 * c);
    }
}

// ── 10. RoPE backward ───────────────────────────────────────────────────────
// Inverse rotation: negate sin. Same structure as forward.

kernel void fused_rope_bwd(device const bfloat* grad_out [[buffer(0)]],
                            device const bfloat* cos_t    [[buffer(1)]],
                            device const bfloat* sin_t    [[buffer(2)]],
                            device       bfloat* grad_in  [[buffer(3)]],
                            constant     uint&   N        [[buffer(4)]],
                            constant     uint&   T        [[buffer(5)]],
                            constant     uint&   n_head   [[buffer(6)]],
                            constant     uint&   hdim     [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    uint half_d = hdim / 2;
    uint d = gid % hdim;
    uint rem = gid / hdim;
    rem = rem / n_head;
    uint t = rem % T;
    if (d < half_d) {
        uint cs_idx = t * half_d + d;
        float c = float(cos_t[cs_idx]);
        float s = float(sin_t[cs_idx]);
        float g1 = float(grad_out[gid]);
        float g2 = float(grad_out[gid + half_d]);
        grad_in[gid] = bfloat(g1 * c - g2 * s);
    } else {
        uint d2 = d - half_d;
        uint cs_idx = t * half_d + d2;
        float c = float(cos_t[cs_idx]);
        float s = float(sin_t[cs_idx]);
        float g1 = float(grad_out[gid - half_d]);
        float g2 = float(grad_out[gid]);
        grad_in[gid] = bfloat(g1 * s + g2 * c);
    }
}

// ── 11. QK-Norm forward ──────────────────────────────────────────────────────
// Normalizes Q (q_rows rows) and K (k_rows rows) in a single dispatch.
// Output layout: [normed_q (q_rows * D) | normed_k (k_rows * D)]

kernel void fused_qk_norm_fwd(device const bfloat* q         [[buffer(0)]],
                               device const bfloat* k         [[buffer(1)]],
                               device       bfloat* out       [[buffer(2)]],
                               constant     uint&   D         [[buffer(3)]],
                               constant     uint&   q_rows    [[buffer(4)]],
                               constant     float&  eps       [[buffer(5)]],
                               uint row     [[threadgroup_position_in_grid]],
                               uint tid     [[thread_index_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
    device const bfloat* src = (row < q_rows) ? (q + row * D) : (k + (row - q_rows) * D);
    device       bfloat* dst = out + row * D;

    float local_sq = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float v = float(src[i]);
        local_sq += v * v;
    }

    threadgroup float shared[32];
    float sum_sq = reduce_sum(local_sq, tid, tg_size, shared);
    float rrms = rsqrt(sum_sq / float(D) + eps);

    for (uint i = tid; i < D; i += tg_size) {
        dst[i] = bfloat(float(src[i]) * rrms);
    }
}

// ── 12. QK-Norm backward ─────────────────────────────────────────────────────
// Inputs q, k (original), grad_out (concatenated [grad_nq | grad_nk]).
// Output: concatenated [grad_q | grad_k] in a single buffer.

kernel void fused_qk_norm_bwd(device const bfloat* q         [[buffer(0)]],
                               device const bfloat* k         [[buffer(1)]],
                               device const bfloat* grad_out  [[buffer(2)]],
                               device       bfloat* grad_in   [[buffer(3)]],
                               constant     uint&   D         [[buffer(4)]],
                               constant     uint&   q_rows    [[buffer(5)]],
                               constant     float&  eps       [[buffer(6)]],
                               uint row     [[threadgroup_position_in_grid]],
                               uint tid     [[thread_index_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
    device const bfloat* src = (row < q_rows) ? (q + row * D) : (k + (row - q_rows) * D);
    device const bfloat* gor = grad_out + row * D;
    device       bfloat* gir = grad_in + row * D;

    float local_sq  = 0.0f;
    float local_dot = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float xi = float(src[i]);
        float gi = float(gor[i]);
        local_sq  += xi * xi;
        local_dot += gi * xi;
    }

    threadgroup float shared[32];
    float sum_sq = reduce_sum(local_sq, tid, tg_size, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot_gx = reduce_sum(local_dot, tid, tg_size, shared);

    float rrms  = rsqrt(sum_sq / float(D) + eps);
    float coeff = dot_gx / float(D) * rrms * rrms;

    for (uint i = tid; i < D; i += tg_size) {
        float xi = float(src[i]);
        float gi = float(gor[i]);
        gir[i] = bfloat(rrms * (gi - xi * coeff));
    }
}

// ── 13. Softmax forward ──────────────────────────────────────────────────────

kernel void fused_softmax_fwd(device const bfloat* x        [[buffer(0)]],
                              device       bfloat* y        [[buffer(1)]],
                              constant     uint&   D        [[buffer(2)]],
                              uint row     [[threadgroup_position_in_grid]],
                              uint tid     [[thread_index_in_threadgroup]],
                              uint tg_size [[threads_per_threadgroup]]) {
    device const bfloat* xr = x + row * D;
    device       bfloat* yr = y + row * D;

    threadgroup float shared[32];

    // pass 1: find max
    float local_max = -INFINITY;
    for (uint i = tid; i < D; i += tg_size) {
        local_max = max(local_max, float(xr[i]));
    }
    float m = reduce_max(local_max, tid, tg_size, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup); // sync before reusing shared[]

    // pass 2: exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        local_sum += exp(float(xr[i]) - m);
    }
    float total = reduce_sum(local_sum, tid, tg_size, shared);
    float inv_sum = 1.0f / total;

    // pass 3: normalize
    for (uint i = tid; i < D; i += tg_size) {
        yr[i] = bfloat(exp(float(xr[i]) - m) * inv_sum);
    }
}

// ── 12. Softmax backward ───────────────────────────────────────────────────
// grad_in_i = s_i * (g_i - dot(s, g))
// where s = softmax output, g = grad_out

kernel void fused_softmax_bwd(device const bfloat* s        [[buffer(0)]],
                              device const bfloat* grad_out [[buffer(1)]],
                              device       bfloat* grad_in  [[buffer(2)]],
                              constant     uint&   D        [[buffer(3)]],
                              uint row     [[threadgroup_position_in_grid]],
                              uint tid     [[thread_index_in_threadgroup]],
                              uint tg_size [[threads_per_threadgroup]]) {
    device const bfloat* sr  = s        + row * D;
    device const bfloat* gor = grad_out + row * D;
    device       bfloat* gir = grad_in  + row * D;

    // pass 1: compute dot(s, g)
    float local_dot = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        local_dot += float(sr[i]) * float(gor[i]);
    }

    threadgroup float shared[32];
    float dot_sg = reduce_sum(local_dot, tid, tg_size, shared);

    // pass 2: grad_in_i = s_i * (g_i - dot_sg)
    for (uint i = tid; i < D; i += tg_size) {
        float si = float(sr[i]);
        float gi = float(gor[i]);
        gir[i] = bfloat(si * (gi - dot_sg));
    }
}

// ── 15. Fused token + position embedding forward ────────────────────────────
// out[i * D + d] = tok_emb[token_ids[i] * D + d] + pos_emb[(seq_offset + i % T) * D + d]
// Grid: N = B * T * D total elements, one thread per element.

kernel void fused_embed_fwd(device const uint*   token_ids [[buffer(0)]],
                            device const bfloat* tok_emb   [[buffer(1)]],
                            device const bfloat* pos_emb   [[buffer(2)]],
                            device       bfloat* out       [[buffer(3)]],
                            constant     uint&   T         [[buffer(4)]],
                            constant     uint&   D         [[buffer(5)]],
                            constant     uint&   N         [[buffer(6)]],
                            constant     uint&   seq_off   [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    uint d = gid % D;
    uint i = gid / D;            // token index in flattened (B*T)
    uint pos = seq_off + (i % T);
    uint tok_id = token_ids[i];
    out[gid] = bfloat(float(tok_emb[tok_id * D + d]) + float(pos_emb[pos * D + d]));
}

// ── 16. Fused embed backward: scatter grad to token embedding table ─────────
// For each element in grad_out (B*T, D), atomically add to grad_tok_emb[token_ids[i], d].
// Uses float atomics (accumulate in F32 buffer, caller casts back).

kernel void fused_embed_bwd_tok(device const uint*   token_ids  [[buffer(0)]],
                                device const bfloat* grad_out   [[buffer(1)]],
                                device atomic<float>* grad_tok  [[buffer(2)]],
                                constant     uint&   D          [[buffer(3)]],
                                constant     uint&   N          [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    uint d = gid % D;
    uint i = gid / D;
    uint tok_id = token_ids[i];
    float g = float(grad_out[gid]);
    atomic_fetch_add_explicit(&grad_tok[tok_id * D + d], g, memory_order_relaxed);
}

// ── 17. Fused embed backward: scatter grad to position embedding table ──────
// For each element in grad_out (B*T, D), atomically add to grad_pos_emb[pos, d].

kernel void fused_embed_bwd_pos(device const bfloat* grad_out   [[buffer(0)]],
                                device atomic<float>* grad_pos  [[buffer(1)]],
                                constant     uint&   T          [[buffer(2)]],
                                constant     uint&   D          [[buffer(3)]],
                                constant     uint&   N          [[buffer(4)]],
                                constant     uint&   seq_off    [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    uint d = gid % D;
    uint i = gid / D;
    uint pos = seq_off + (i % T);
    float g = float(grad_out[gid]);
    atomic_fetch_add_explicit(&grad_pos[pos * D + d], g, memory_order_relaxed);
}

// ── 15. Fused cross-entropy forward ────────────────────────────────────────
// One threadgroup per row (token). Computes per-token CE loss in F32.
// logits: (N, V) bfloat, targets: (N,) uint32, losses: (N,) float32

kernel void fused_cross_entropy_fwd(device const bfloat*  logits  [[buffer(0)]],
                                    device const uint*    targets [[buffer(1)]],
                                    device       float*   losses  [[buffer(2)]],
                                    constant     uint&    V       [[buffer(3)]],
                                    uint row     [[threadgroup_position_in_grid]],
                                    uint tid     [[thread_index_in_threadgroup]],
                                    uint tg_size [[threads_per_threadgroup]]) {
    device const bfloat* xr = logits + row * V;
    uint tgt = targets[row];
    threadgroup float shared[32];

    // pass 1: find max
    float local_max = -INFINITY;
    for (uint i = tid; i < V; i += tg_size) {
        local_max = max(local_max, float(xr[i]));
    }
    float m = reduce_max(local_max, tid, tg_size, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup); // sync before reusing shared[]

    // pass 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (uint i = tid; i < V; i += tg_size) {
        local_sum += exp(float(xr[i]) - m);
    }
    float total = reduce_sum(local_sum, tid, tg_size, shared);

    // loss = -(logit[target] - max) + log(sum_exp)
    if (tid == 0) {
        float logit_tgt = float(xr[tgt]);
        losses[row] = -(logit_tgt - m) + log(total);
    }
}

// ── 18. Fused cross-entropy backward ───────────────────────────────────────
// grad_logits[row,j] = (softmax(logits)[row,j] - (j == target[row])) * grad_res[row]

kernel void fused_cross_entropy_bwd(device const bfloat*  logits     [[buffer(0)]],
                                    device const uint*    targets    [[buffer(1)]],
                                    device const float*   grad_res   [[buffer(2)]],
                                    device       bfloat*  grad_in    [[buffer(3)]],
                                    constant     uint&    V          [[buffer(4)]],
                                    uint row     [[threadgroup_position_in_grid]],
                                    uint tid     [[thread_index_in_threadgroup]],
                                    uint tg_size [[threads_per_threadgroup]]) {
    device const bfloat* xr = logits + row * V;
    device       bfloat* gr = grad_in + row * V;
    uint tgt = targets[row];
    float scale = grad_res[row];
    threadgroup float shared[32];

    // pass 1: find max
    float local_max = -INFINITY;
    for (uint i = tid; i < V; i += tg_size) {
        local_max = max(local_max, float(xr[i]));
    }
    float m = reduce_max(local_max, tid, tg_size, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup); // sync before reusing shared[]

    // pass 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (uint i = tid; i < V; i += tg_size) {
        local_sum += exp(float(xr[i]) - m);
    }
    float total = reduce_sum(local_sum, tid, tg_size, shared);
    float inv_sum = 1.0f / total;

    // pass 3: write grad = (softmax - one_hot) * scale
    for (uint i = tid; i < V; i += tg_size) {
        float p = exp(float(xr[i]) - m) * inv_sum;
        float indicator = (i == tgt) ? 1.0f : 0.0f;
        gr[i] = bfloat((p - indicator) * scale);
    }
}

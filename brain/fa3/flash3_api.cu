// Flash Attention 3 (Hopper) C wrapper.
// Provides extern "C" entry points for Rust FFI, filling FA3's
// Flash_fwd_params / Flash_bwd_params from flat arguments.
//
// Only hdim128, bf16, sm90 instantiations are linked.
// Causal / local attention handled at runtime (not compile-time).

#include "flash.h"
#include <cutlass/numeric_types.h>

#include <cstring>
#include <cstdio>
#include <algorithm>

// ── Forward template instantiation (from flash_fwd_hdim128_bf16_sm90.cu) ────
// Template: <Arch, T, kHeadDim, kHeadDimV, Split, PagedKVNonTMA, Has_softcap, PackGQA>
template<>
void run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, false>(
    Flash_fwd_params &params, cudaStream_t stream);

// ── Backward template instantiation (from flash_bwd_hdim128_bf16_sm90.cu) ───
// Template: <Arch, T, kHeadDim, Has_softcap>
template<>
void run_mha_bwd_<90, cutlass::bfloat16_t, 128, false>(
    Flash_bwd_params &params, cudaStream_t stream);

// ── prepare_varlen_num_blocks (from flash_prepare_scheduler.cu) ─────────────
// Already declared in flash.h

// ═════════════════════════════════════════════════════════════════════════════
// Forward
// ═════════════════════════════════════════════════════════════════════════════

extern "C" void run_mha_v3(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,

    // Scheduler metadata buffer (pre-allocated by Rust, >= 1024 int32s)
    int  *scheduler_meta_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,

    uint32_t seqlen_q,
    uint32_t seqlen_k,

    int is_bf16,
    int is_causal,

    int window_size_left,
    int window_size_right,

    float softcap,

    int num_sm,
    cudaStream_t stream
) {
    Flash_fwd_params params = {};

    // Pointers
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;

    // Strides (u32 -> i64 widening)
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;
    params.v_dim_stride = 1;  // contiguous last dimension

    // Dimensions
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q;  // 2048 is already multiple of 128
    params.seqlen_k_rounded = seqlen_k;
    params.d = d;
    params.d_rounded = d_rounded;
    params.dv = d;         // V headdim == Q/K headdim
    params.dv_rounded = d_rounded;
    params.total_q = b * seqlen_q;
    params.total_k = b * seqlen_k;

    // Scale
    params.scale_softmax = softmax_scale;
    params.softcap = (softcap > 0.0f) ? 1.0f : 0.0f;  // FA3 uses 0/1 flag

    // Causal / local attention
    params.is_causal = (is_causal != 0);
    params.is_local = (window_size_left >= 0 || window_size_right >= 0);
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.attention_chunk = 0;

    // Data type
    params.is_bf16 = (is_bf16 != 0);

    // No dropout
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;

    // Hardware
    params.arch = 90;
    params.num_sm = num_sm;

    // No split-KV, no paged KV
    params.num_splits = 1;
    params.pack_gqa = false;
    params.page_size = 1;

    // ── Scheduler metadata setup (required by FA3 persistent kernels) ────────
    // Layout in scheduler_meta_ptr:
    //   [0..b_rounded)                    = num_splits_dynamic
    //   [b_rounded..2*b_rounded)          = num_m_blocks
    //   [2*b_rounded..3*b_rounded)        = varlen_batch_idx (if sort_batches)
    //   [3*b_rounded..4*b_rounded)        = num_nheads_in_l2 (if head_swizzle)
    //   [tile_count_semaphore_offset]      = tile_count_semaphore
    auto round_up4 = [](int x) { return (x + 3) / 4 * 4; };
    int b_rounded = round_up4(b);

    params.varlen_sort_batches = !params.is_local;
    params.head_swizzle = params.is_causal || params.is_local;

    int num_vectors = 2;  // num_splits_dynamic + num_m_blocks
    if (params.varlen_sort_batches) num_vectors++;
    if (params.head_swizzle) num_vectors++;
    int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
    int tile_count_semaphore_offset = b_rounded * num_vectors;

    params.num_splits_dynamic_ptr = scheduler_meta_ptr;
    params.num_m_blocks_ptr = scheduler_meta_ptr + b_rounded;
    params.varlen_batch_idx_ptr = params.varlen_sort_batches
        ? scheduler_meta_ptr + b_rounded * 2 : nullptr;
    params.num_nheads_in_l2_ptr = params.head_swizzle
        ? scheduler_meta_ptr + head_swizzle_offset : nullptr;
    params.tile_count_semaphore = scheduler_meta_ptr + tile_count_semaphore_offset;

    params.skip_scheduler_metadata_computation = false;
    params.prepare_varlen_pdl = false;  // PDL not needed for non-varlen

    // Run scheduler to compute metadata (kBlockM=128, kBlockN=128 for hdim128 bf16)
    int kBlockM = 128;
    int kBlockN = 128;
    prepare_varlen_num_blocks(params, stream, params.pack_gqa, kBlockM, kBlockN, false /*enable_pdl*/);

    // Run FA3 forward
    run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, false>(params, stream);
}

// ═════════════════════════════════════════════════════════════════════════════
// Backward (no scheduler metadata needed)
// ═════════════════════════════════════════════════════════════════════════════

extern "C" void run_mha_backward_v3(
    void *dout_ptr,
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *out_ptr,
    void *softmax_lse_ptr,
    void *dq_ptr,
    void *dk_ptr,
    void *dv_ptr,
    void *dq_accum_ptr,
    void *dsoftmax_sum_ptr,

    // FA3-specific backward buffers
    void *softmax_lse_log2_ptr,
    int  *dq_semaphore_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t do_batch_stride,
    uint32_t dq_batch_stride,
    uint32_t dk_batch_stride,
    uint32_t dv_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,
    uint32_t do_row_stride,
    uint32_t dq_row_stride,
    uint32_t dk_row_stride,
    uint32_t dv_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,
    uint32_t do_head_stride,
    uint32_t dq_head_stride,
    uint32_t dk_head_stride,
    uint32_t dv_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,

    uint32_t seqlen_q,
    uint32_t seqlen_k,

    int is_bf16,
    int is_causal,

    int window_size_left,
    int window_size_right,

    float softcap,
    int deterministic,

    int num_sm,
    cudaStream_t stream
) {
    Flash_bwd_params params = {};

    // ── Forward params (reused by backward) ─────────────────────────────────
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = out_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;

    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;
    params.v_dim_stride = 1;

    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q;
    params.seqlen_k_rounded = seqlen_k;
    params.d = d;
    params.d_rounded = d_rounded;
    params.dv = d;
    params.dv_rounded = d_rounded;
    params.total_q = b * seqlen_q;
    params.total_k = b * seqlen_k;

    params.scale_softmax = softmax_scale;
    params.softcap = (softcap > 0.0f) ? 1.0f : 0.0f;

    params.is_causal = (is_causal != 0);
    params.is_local = (window_size_left >= 0 || window_size_right >= 0);
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.attention_chunk = 0;

    params.is_bf16 = (is_bf16 != 0);

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;

    params.arch = 90;
    params.num_sm = num_sm;

    params.num_splits = 1;
    params.pack_gqa = false;

    // ── Backward-specific params ────────────────────────────────────────────
    params.do_ptr = dout_ptr;
    params.dq_ptr = dq_ptr;
    params.dk_ptr = dk_ptr;
    params.dv_ptr = dv_ptr;

    params.do_batch_stride = do_batch_stride;
    params.do_row_stride = do_row_stride;
    params.do_head_stride = do_head_stride;
    params.dq_batch_stride = dq_batch_stride;
    params.dk_batch_stride = dk_batch_stride;
    params.dv_batch_stride = dv_batch_stride;
    params.dq_row_stride = dq_row_stride;
    params.dk_row_stride = dk_row_stride;
    params.dv_row_stride = dv_row_stride;
    params.dq_head_stride = dq_head_stride;
    params.dk_head_stride = dk_head_stride;
    params.dv_head_stride = dv_head_stride;

    // Accumulators (f32)
    params.dq_accum_ptr = dq_accum_ptr;
    params.dk_accum_ptr = nullptr;  // not GQA (h == h_k)
    params.dv_accum_ptr = nullptr;  // not GQA (h == h_k)

    // Softmax backward
    params.dsoftmax_sum = dsoftmax_sum_ptr;
    params.softmax_lse_log2_ptr = softmax_lse_log2_ptr;

    // Semaphores
    params.dq_semaphore = dq_semaphore_ptr;
    // dk/dv semaphores: nullptr (not GQA, not deterministic-GQA)

    params.deterministic = (deterministic != 0);
    params.dq_accum_split_stride = h * d_rounded;

    run_mha_bwd_<90, cutlass::bfloat16_t, 128, false>(params, stream);
}

use std::ffi::c_void;

// CUstream is a pointer type in the CUDA driver API.
// cudaStream_t (runtime API) is the same type.
pub type CudaStream = *mut c_void;

unsafe extern "C" {
    // ── embedding.cu ────────────────────────────────────────────────────────
    pub fn embedding_fwd(
        idx: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        bt: i32,
        d: i32,
        stream: CudaStream,
    );
    pub fn embedding_bwd(
        idx: *const c_void,
        d_out: *const c_void,
        d_weight: *mut c_void,
        bt: i32,
        v: i32,
        d: i32,
        stream: CudaStream,
    );

    // ── relu_sq.cu ──────────────────────────────────────────────────────────
    pub fn relu_sq_fwd(x: *const c_void, y: *mut c_void, n: i32, stream: CudaStream);
    pub fn relu_sq_bwd(x: *const c_void, grad: *const c_void, dx: *mut c_void, n: i32, stream: CudaStream);

    // ── residual_scale.cu ───────────────────────────────────────────────────
    pub fn residual_scale_fwd(
        x: *const c_void,
        x0: *const c_void,
        lambda_r_ptr: *const c_void,
        lambda_0_ptr: *const c_void,
        out: *mut c_void,
        n: i32,
        stream: CudaStream,
    );
    pub fn residual_scale_bwd(
        x: *const c_void,
        x0: *const c_void,
        grad: *const c_void,
        lambda_r_ptr: *const c_void,
        lambda_0_ptr: *const c_void,
        d_x: *mut c_void,
        d_x0: *mut c_void,
        d_lambda_r: *mut f32,
        d_lambda_0: *mut f32,
        n: i32,
        stream: CudaStream,
    );

    // ── softcap.cu ──────────────────────────────────────────────────────────
    pub fn softcap_fwd(x: *const c_void, y: *mut c_void, cap: f32, n: i32, stream: CudaStream);
    pub fn softcap_bwd(
        x: *const c_void,
        grad: *const c_void,
        dx: *mut c_void,
        cap: f32,
        n: i32,
        stream: CudaStream,
    );

    // ── ve_apply.cu ─────────────────────────────────────────────────────────
    pub fn ve_apply_fwd(
        v: *mut c_void,
        ve: *const c_void,
        gate: *const c_void,
        bt: i32,
        n_kv_head: i32,
        head_dim: i32,
        stream: CudaStream,
    );
    pub fn ve_apply_bwd(
        d_v: *const c_void,
        ve: *const c_void,
        gate: *const c_void,
        d_ve: *mut c_void,
        d_gate: *mut c_void,
        bt: i32,
        n_kv_head: i32,
        head_dim: i32,
        stream: CudaStream,
    );

    // ── elementwise.cu ──────────────────────────────────────────────────────
    pub fn residual_add(x: *mut c_void, y: *const c_void, n: i32, stream: CudaStream);
    pub fn three_way_add(
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        out: *mut c_void,
        n: i32,
        stream: CudaStream,
    );
    pub fn cast_bf16_to_f32(input: *const c_void, output: *mut c_void, n: i32, stream: CudaStream);
    pub fn cast_f32_to_bf16(input: *const c_void, output: *mut c_void, n: i32, stream: CudaStream);
    pub fn slice_cols(
        input: *const c_void, output: *mut c_void,
        rows: i32, in_cols: i32, out_cols: i32,
        stream: CudaStream,
    );
    pub fn add_slice_cols(
        dst: *mut c_void, src: *const c_void,
        rows: i32, dst_cols: i32, src_cols: i32,
        stream: CudaStream,
    );

    // ── adamw.cu ────────────────────────────────────────────────────────────
    pub fn adamw_step_bf16(
        params: *mut c_void,
        grads: *const c_void,
        exp_avg: *mut c_void,
        exp_avg_sq: *mut c_void,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        n: i32,
        stream: CudaStream,
    );
    pub fn adamw_step_f32(
        params: *mut f32,
        grads: *const f32,
        exp_avg: *mut f32,
        exp_avg_sq: *mut f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        n: i32,
        stream: CudaStream,
    );

    // ── muon.cu ───────────────────────────────────────────────────────────
    pub fn muon_nesterov_bf16(
        buf: *mut c_void, grad: *const c_void, out: *mut c_void,
        momentum: f32, n: i32, stream: CudaStream,
    );
    pub fn muon_nesterov_f32buf(
        buf: *mut f32, grad: *const c_void, out: *mut c_void,
        momentum: f32, n: i32, stream: CudaStream,
    );
    pub fn frob_normalize_bf16(x: *const c_void, out: *mut c_void, n: i32, scratch: *mut f32, stream: CudaStream);
    pub fn muon_weight_update_bf16(
        param: *mut c_void, update: *const c_void,
        lr: f32, wd: f32, n: i32, stream: CudaStream,
    );
    pub fn muon_weight_update_f32(
        master: *mut f32, compute: *mut c_void, update: *const c_void,
        lr: f32, wd: f32, n: i32, stream: CudaStream,
    );
    pub fn copy_bf16_to_f32(src: *const c_void, dst: *mut f32, n: i32, stream: CudaStream);
    pub fn normuon_step_bf16(
        g: *mut c_void,
        second_mom: *mut f32,
        m: i32, n: i32,
        reduce_cols: i32,
        beta2: f32,
        scratch: *mut f32,
        stream: CudaStream,
    );
    pub fn scale_bf16(x: *mut c_void, scale: f32, n: i32, stream: CudaStream);
    pub fn eye_bf16(out: *mut c_void, d: i32, stream: CudaStream);
    pub fn axpby_bf16(
        x: *const c_void, y: *const c_void, out: *mut c_void,
        alpha: f32, beta: f32, n: i32, stream: CudaStream,
    );
    pub fn ns_combined_batched(
        a: *const c_void, a2: *const c_void, out: *mut c_void,
        ca: f32, cb: f32, cc: f32,
        d: i32, batch: i32, stream: CudaStream,
    );

    // ── fused_norm_residual.cu ─────────────────────────────────────────────

    // Fused residual_scale + RMSNorm forward:
    //   scaled_x = lambda_r * x + lambda_0 * x0;  norm_out = rms_norm(scaled_x)
    pub fn fused_residual_norm_fwd(
        x: *const c_void,
        x0: *const c_void,
        lambda_r_ptr: *const c_void,
        lambda_0_ptr: *const c_void,
        scaled_x_out: *mut c_void,
        norm_out: *mut c_void,
        rows: u32,
        d: u32,
        eps: f32,
        stream: CudaStream,
    );

    // Fused rms_norm_bwd + residual_add + residual_scale_bwd
    pub fn fused_residual_norm_bwd(
        grad_out: *const c_void,
        d_x_in: *const c_void,
        x_pre: *const c_void,
        x0: *const c_void,
        lambda_r_ptr: *const c_void,
        lambda_0_ptr: *const c_void,
        d_x_out: *mut c_void,
        d_x0: *mut c_void,
        d_lambda_r: *mut f32,
        d_lambda_0: *mut f32,
        rows: u32,
        d: u32,
        eps: f32,
        stream: CudaStream,
    );

    // Fused residual_add + RMSNorm forward:
    //   x += proj;  y = rms_norm(x)
    pub fn fused_residual_add_rms_norm_fwd(
        x: *mut c_void,
        proj: *const c_void,
        y: *mut c_void,
        rows: u32,
        d: u32,
        eps: f32,
        stream: CudaStream,
    );

    // Fused rms_norm_bwd + residual accumulate:
    //   d_x += rms_norm_bwd(x_pre_norm, grad_out)
    pub fn fused_rms_norm_bwd_residual_add(
        x: *const c_void,
        grad_out: *const c_void,
        d_x: *mut c_void,
        rows: u32,
        d: u32,
        eps: f32,
        stream: CudaStream,
    );

    // Fused RoPE + per-head RMSNorm forward:
    //   out = rms_norm_per_head(rope(x, cos, sin))
    pub fn fused_rope_rms_norm_fwd(
        x: *const c_void,
        cos_t: *const c_void,
        sin_t: *const c_void,
        out: *mut c_void,
        rows: u32,
        t: u32,
        n_head: u32,
        hdim: u32,
        eps: f32,
        stream: CudaStream,
    );

    // ── rms_norm.cu ─────────────────────────────────────────────────────────
    pub fn fused_rms_norm_fwd(
        x: *const c_void,
        y: *mut c_void,
        rows: u32,
        d: u32,
        eps: f32,
        stream: CudaStream,
    );
    pub fn fused_rms_norm_bwd(
        x: *const c_void,
        grad_out: *const c_void,
        grad_in: *mut c_void,
        rows: u32,
        d: u32,
        eps: f32,
        stream: CudaStream,
    );

    // ── layer_stat.cu ────────────────────────────────────────────────────────
    pub fn layer_l2_norm_bf16(x: *const c_void, n: i32, out: *mut f32, out_idx: i32, stream: CudaStream);
    pub fn layer_scale_bf16(scale: *const f32, scale_idx: i32, x: *mut c_void, n: i32, stream: CudaStream);
    pub fn neuron_act_norm_bf16(h_act: *const c_void, bt: i32, mlp_dim: i32, out: *mut f32, layer: i32, stream: CudaStream);

    // ── rope.cu ─────────────────────────────────────────────────────────────
    pub fn fused_rope_fwd(
        x: *const c_void,
        cos_t: *const c_void,
        sin_t: *const c_void,
        out: *mut c_void,
        n: u32,
        t: u32,
        n_head: u32,
        hdim: u32,
        stream: CudaStream,
    );
    pub fn fused_rope_bwd(
        grad_out: *const c_void,
        cos_t: *const c_void,
        sin_t: *const c_void,
        grad_in: *mut c_void,
        n: u32,
        t: u32,
        n_head: u32,
        hdim: u32,
        stream: CudaStream,
    );

    // ── cross_entropy.cu (with fused softcap) ─────────────────────────────
    pub fn fused_cross_entropy_fwd(
        logits: *const c_void,
        targets: *const c_void,
        losses: *mut c_void,
        loss_sum: *mut f32,
        n: u32,
        v: u32,
        softcap: f32,
        stream: CudaStream,
    );
    pub fn fused_cross_entropy_bwd(
        logits: *const c_void,
        targets: *const c_void,
        grad_res: *const c_void,
        grad_in: *mut c_void,
        n: u32,
        v: u32,
        softcap: f32,
        stream: CudaStream,
    );

    // ── flash-attn v3 (Hopper, prebuilt libflashattention3.a) ──────────────
    pub fn run_mha_v3(
        q_ptr: *mut c_void, k_ptr: *mut c_void, v_ptr: *mut c_void, o_ptr: *mut c_void,
        softmax_lse_ptr: *mut c_void,
        scheduler_meta_ptr: *mut i32,
        q_batch_stride: u32, k_batch_stride: u32, v_batch_stride: u32, o_batch_stride: u32,
        q_row_stride: u32, k_row_stride: u32, v_row_stride: u32, o_row_stride: u32,
        q_head_stride: u32, k_head_stride: u32, v_head_stride: u32, o_head_stride: u32,
        b: u32, h: u32, h_k: u32, d: u32, d_rounded: u32,
        softmax_scale: f32,
        seqlen_q: u32, seqlen_k: u32,
        is_bf16: i32, is_causal: i32,
        window_size_left: i32, window_size_right: i32,
        softcap: f32,
        num_sm: i32,
        stream: CudaStream,
    );

    pub fn run_mha_backward_v3(
        dout_ptr: *mut c_void, q_ptr: *mut c_void, k_ptr: *mut c_void, v_ptr: *mut c_void,
        out_ptr: *mut c_void, softmax_lse_ptr: *mut c_void,
        dq_ptr: *mut c_void, dk_ptr: *mut c_void, dv_ptr: *mut c_void,
        dq_accum_ptr: *mut c_void, dsoftmax_sum_ptr: *mut c_void,
        softmax_lse_log2_ptr: *mut c_void,
        dq_semaphore_ptr: *mut i32,
        q_batch_stride: u32, k_batch_stride: u32, v_batch_stride: u32, o_batch_stride: u32,
        do_batch_stride: u32, dq_batch_stride: u32, dk_batch_stride: u32, dv_batch_stride: u32,
        q_row_stride: u32, k_row_stride: u32, v_row_stride: u32, o_row_stride: u32,
        do_row_stride: u32, dq_row_stride: u32, dk_row_stride: u32, dv_row_stride: u32,
        q_head_stride: u32, k_head_stride: u32, v_head_stride: u32, o_head_stride: u32,
        do_head_stride: u32, dq_head_stride: u32, dk_head_stride: u32, dv_head_stride: u32,
        b: u32, h: u32, h_k: u32, d: u32, d_rounded: u32,
        softmax_scale: f32,
        seqlen_q: u32, seqlen_k: u32,
        is_bf16: i32, is_causal: i32,
        window_size_left: i32, window_size_right: i32,
        softcap: f32, deterministic: i32,
        num_sm: i32,
        stream: CudaStream,
    );
}

use std::ffi::c_void;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, DevicePtr};
use half::bf16;

use crate::buffer::BufferManager;
use crate::config::*;
use crate::ffi;
use crate::gemm::GemmRunner;

// ---------------------------------------------------------------------------
// Raw pointer helpers (same as forward.rs)
// ---------------------------------------------------------------------------

/// Extract the raw device pointer from a CudaSlice via the DevicePtr trait.
#[inline(always)]
fn dptr<T>(buf: &CudaSlice<T>) -> CUdeviceptr {
    let (ptr, _sync) = buf.device_ptr(buf.stream());
    ptr
}

/// Device pointer offset by `offset` elements of type T.
#[inline(always)]
fn dptr_at<T>(buf: &CudaSlice<T>, offset: usize) -> CUdeviceptr {
    dptr(buf) + (offset * std::mem::size_of::<T>()) as u64
}

/// Full backward pass. Gradients accumulate (+=) into bufs.layer_grads,
/// bufs.wte_grad, bufs.lm_head_grad, bufs.resid_lambdas_grad, bufs.x0_lambdas_grad.
/// Caller must call bufs.zero_gradients() before the first micro-step.
///
/// # Forward-pass buffer contract
///
/// The forward pass must populate:
/// - `bufs.targets`
/// - (logits are recomputed per-chunk in the CE backward loop)
/// - `bufs.emb` -- embedding output
/// - `bufs.x` -- final hidden state (pre-final-norm, post-last-layer)
/// - `bufs.x0` -- initial hidden state (= rms_norm(emb), immutable after init)
/// - `bufs.saved_x_pre_attn_norm[i]` -- x before residual_scale at layer i
/// - `bufs.saved_x_pre_mlp_norm[i]` -- x before MLP layer-norm at layer i
/// - `bufs.saved_xn[i]` -- attention-normed xn at layer i (for QKV/VE-gate dW)
/// - (h_pre_act recomputed from saved_x_pre_mlp_norm via rms_norm + FC1)
/// - (q/k recomputed in backward from saved_xn via wq/wk + fused_rope_rms_norm)
/// - `bufs.saved_v[i]` -- v after VE-apply (if applicable)
/// - `bufs.saved_attn_out[i]` -- flash attention output (pre-Wo projection)
/// - `bufs.saved_softmax_lse[i]` -- flash attention log-sum-exp (f32)
pub fn backward(bufs: &mut BufferManager, gemm: &GemmRunner, grad_accum_steps: usize) {
    bufs.stream.context().bind_to_thread().expect("bind CUDA context");

    let b = bufs.batch_size;
    let bt = b * SEQ;
    let n_btd = (bt * D_MODEL) as i32;
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;

    // Zero d_x0 before per-layer loop (it accumulates via read-modify-write
    // in residual_scale_bwd, so must start at zero each backward call).
    crate::forward::raw_zero(&bufs.stream, &mut bufs.d_x0);

    // =====================================================================
    //  Post-loop backward
    // =====================================================================

    // Recompute xn_final = rms_norm(bufs.x) into bufs.xn (needed for lm_head dW).
    unsafe {
        ffi::fused_rms_norm_fwd(
            dptr(&bufs.x) as *const c_void,
            dptr(&bufs.xn) as *mut c_void,
            bt as u32, D_MODEL as u32, EPS, stream,
        );
    }

    // [CHUNKED CE BWD + LM HEAD BWD] Process CE_CHUNK tokens at a time.
    // For each chunk:
    //   (a) Recompute logits chunk: xn[offset..] @ lm_head^T -> logits (CE_CHUNK buf)
    //   (b) CE bwd: logits, targets[offset..] -> d_logits (CE_CHUNK buf)
    //   (c) LM head dX: d_logits @ lm_head -> d_xn[offset..] (beta=0, non-overlapping)
    //   (d) LM head dW: d_logits^T @ xn[offset..] -> d_lm_head (accumulate)
    let grad_scale = 1.0f32 / (bt * grad_accum_steps) as f32;
    {
        let handle = *gemm.blas().handle();
        let logits_ptr = dptr(&bufs.logits);
        let d_logits_ptr = dptr(&bufs.d_logits);
        let lm_head_ptr = dptr(&bufs.lm_head);
        let lm_head_grad_ptr = dptr(&bufs.lm_head_grad);
        let xn_base = dptr(&bufs.xn);
        let d_xn_base = dptr(&bufs.d_xn);
        let targets_base = dptr(&bufs.targets);
        let grad_res_ptr = dptr(&bufs.h_act);
        let d = D_MODEL;

        let mut offset = 0usize;
        while offset < bt {
            let cur = CE_CHUNK.min(bt - offset);
            let xn_off = xn_base + (offset * d * std::mem::size_of::<bf16>()) as u64;
            let d_xn_off = d_xn_base + (offset * d * std::mem::size_of::<bf16>()) as u64;
            let targets_off = targets_base + (offset * std::mem::size_of::<i32>()) as u64;
            let grad_res_off = grad_res_ptr + (offset * std::mem::size_of::<f32>()) as u64;

            // (a) Recompute logits: xn[offset..][cur, d] @ lm_head[VOCAB, d]^T -> logits[cur, VOCAB]
            unsafe {
                crate::forward::raw_gemm_matmul(
                    handle, xn_off, lm_head_ptr, logits_ptr,
                    cur, VOCAB, d, 0.0,
                );
            }

            // (b) CE bwd: logits, targets -> d_logits
            //     grad_res scratch at h_act[offset..] filled with grad_scale
            unsafe {
                cudarc::driver::sys::cuMemsetD32Async(
                    grad_res_off,
                    grad_scale.to_bits(),
                    cur,
                    bufs.stream.cu_stream(),
                );
                ffi::fused_cross_entropy_bwd(
                    logits_ptr as *const c_void,
                    targets_off as *const c_void,
                    grad_res_off as *const c_void,
                    d_logits_ptr as *mut c_void,
                    cur as u32, VOCAB as u32,
                    SOFTCAP, stream,
                );
            }

            // (c) LM head dX: d_logits[cur, VOCAB] @ lm_head[VOCAB, d] -> d_xn[offset..][cur, d]
            unsafe {
                crate::forward::raw_gemm_bwd_x(
                    handle, d_logits_ptr, lm_head_ptr, d_xn_off,
                    cur, VOCAB, d, 0.0,
                );
            }

            // (d) LM head dW: d_logits[cur, VOCAB]^T @ xn[offset..][cur, d] -> d_lm_head (accumulate)
            unsafe {
                crate::forward::raw_gemm_matmul_acc(
                    handle, d_logits_ptr, xn_off, lm_head_grad_ptr,
                    cur, VOCAB, d,
                );
            }

            offset += cur;
        }
    }

    // rms_norm_bwd (final norm): x_pre_norm=bufs.x, grad=d_xn --> d_x
    unsafe {
        ffi::fused_rms_norm_bwd(
            dptr(&bufs.x) as *const c_void,
            dptr(&bufs.d_xn) as *const c_void,
            dptr(&bufs.d_x) as *mut c_void,
            bt as u32, D_MODEL as u32, EPS, stream,
        );
    }

    // =====================================================================
    //  Per-layer backward (N_LAYER-1 --> 0)
    // =====================================================================

    for layer in (0..N_LAYER).rev() {
        let ve_layer = has_ve(layer);
        let n_mlp = (bt * MLP_DIM) as i32;

        // =================================================================
        //  MLP backward
        // =================================================================

        // Recompute MLP-normed xn = rms_norm(saved_x_pre_mlp_norm) into bufs.xn
        unsafe {
            ffi::fused_rms_norm_fwd(
                dptr(&bufs.saved_x_pre_mlp_norm[layer]) as *const c_void,
                dptr(&bufs.xn) as *mut c_void,
                bt as u32, D_MODEL as u32, EPS, stream,
            );
        }

        // Recompute h_pre_act = xn @ Wfc into bufs.d_h (scratch, [BT, MLP_DIM])
        gemm.matmul(
            &bufs.xn, &bufs.layer_weights[layer].wfc,
            &mut bufs.d_h, bt, MLP_DIM, D_MODEL,
        );

        // Recompute h_act = relu_sq(h_pre_act) into bufs.h (scratch)
        unsafe {
            ffi::relu_sq_fwd(
                dptr(&bufs.d_h) as *const c_void,
                dptr(&bufs.h) as *mut c_void,
                n_mlp, stream,
            );
        }

        // FC2 dW: d_x^T @ h_act --> d_Wdn (accumulate)
        gemm.matmul_acc(&bufs.d_x, &bufs.h, &mut bufs.layer_grads[layer].wdn, bt, D_MODEL, MLP_DIM);

        // FC2 dX: d_x(BT, D_MODEL) @ Wdn(D_MODEL, MLP_DIM) --> d_h_act(BT, MLP_DIM)
        gemm.matmul_bwd_x(&bufs.d_x, &bufs.layer_weights[layer].wdn, &mut bufs.h_act, bt, D_MODEL, MLP_DIM);

        // relu_sq_bwd: d_h_act, h_pre_act(bufs.d_h) --> d_h (overwrites bufs.d_h in-place)
        unsafe {
            ffi::relu_sq_bwd(
                dptr(&bufs.d_h) as *const c_void,
                dptr(&bufs.h_act) as *const c_void,
                dptr(&bufs.d_h) as *mut c_void,
                n_mlp, stream,
            );
        }

        // FC1 dW: d_h^T @ xn_mlp --> d_Wfc (accumulate)
        gemm.matmul_acc(&bufs.d_h, &bufs.xn, &mut bufs.layer_grads[layer].wfc, bt, MLP_DIM, D_MODEL);

        // FC1 dX: d_h(BT, MLP_DIM) @ Wfc(MLP_DIM, D_MODEL) --> d_xn(BT, D_MODEL)
        gemm.matmul_bwd_x(&bufs.d_h, &bufs.layer_weights[layer].wfc, &mut bufs.d_xn, bt, MLP_DIM, D_MODEL);


        // [FUSED] rms_norm_bwd + residual_add:
        //   d_x += rms_norm_bwd(saved_x_pre_mlp_norm, d_xn)
        unsafe {
            ffi::fused_rms_norm_bwd_residual_add(
                dptr(&bufs.saved_x_pre_mlp_norm[layer]) as *const c_void,
                dptr(&bufs.d_xn) as *const c_void,
                dptr(&bufs.d_x) as *mut c_void,
                bt as u32, D_MODEL as u32, EPS, stream,
            );
        }

        // =================================================================
        //  Attention backward
        // =================================================================

        // Out proj dW: d_x^T @ saved_attn_out --> d_Wo (accumulate)
        gemm.matmul_acc(
            &bufs.d_x, &bufs.saved_attn_out[layer],
            &mut bufs.layer_grads[layer].wo, bt, D_MODEL, D_MODEL,
        );

        // Out proj dX: d_x(BT, D_MODEL) @ Wo(D_MODEL, D_MODEL) --> d_attn_out
        gemm.matmul_bwd_x(
            &bufs.d_x, &bufs.layer_weights[layer].wo,
            &mut bufs.attn_out, bt, D_MODEL, D_MODEL,
        );

        // Recompute post-RoPE+QKnorm q and k for FA3 backward (avoids saving them)
        gemm.matmul(&bufs.saved_xn[layer], &bufs.layer_weights[layer].wq, &mut bufs.q, bt, D_MODEL, D_MODEL);
        unsafe {
            ffi::fused_rope_rms_norm_fwd(
                dptr(&bufs.q) as *const c_void,
                dptr(&bufs.cos) as *const c_void,
                dptr(&bufs.sin) as *const c_void,
                dptr(&bufs.q) as *mut c_void,
                (bt * N_HEAD) as u32,
                SEQ as u32, N_HEAD as u32, HEAD_DIM as u32, EPS, stream,
            );
        }
        gemm.matmul(&bufs.saved_xn[layer], &bufs.layer_weights[layer].wk, &mut bufs.k, bt, D_MODEL, D_MODEL);
        unsafe {
            ffi::fused_rope_rms_norm_fwd(
                dptr(&bufs.k) as *const c_void,
                dptr(&bufs.cos) as *const c_void,
                dptr(&bufs.sin) as *const c_void,
                dptr(&bufs.k) as *mut c_void,
                (bt * N_KV_HEAD) as u32,
                SEQ as u32, N_KV_HEAD as u32, HEAD_DIM as u32, EPS, stream,
            );
        }

        // Flash attention v3 backward: d_attn_out, recomputed q/k, saved v/out/lse --> d_q, d_k, d_v
        // Zero flash scratch buffers
        crate::forward::raw_zero(&bufs.stream, &mut bufs.flash_dq_accum);
        crate::forward::raw_zero(&bufs.stream, &mut bufs.flash_dsoftmax_sum);
        crate::forward::raw_zero(&bufs.stream, &mut bufs.fa3_dq_semaphore);
        {
            let batch_stride = (SEQ * N_HEAD * HEAD_DIM) as u32;
            let row_stride = (N_HEAD * HEAD_DIM) as u32;
            let head_stride = HEAD_DIM as u32;
            let softmax_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

            let window_sz = WINDOW_SIZES[layer];
            let mut window_size_left: i32 = if window_sz >= SEQ { -1 } else { window_sz as i32 };
            let window_size_right: i32 = 0;

            let is_causal: i32 = 1; // always causal
            if window_size_left < 0 && window_size_right >= 0 {
                window_size_left = SEQ as i32;
            }

            // FA3 writes into separate d_q/d_k/d_v (needed for QK-norm/RoPE bwd below)
            unsafe {
                ffi::run_mha_backward_v3(
                    dptr(&bufs.attn_out) as *mut _,         // dout (grad of attn output)
                    dptr(&bufs.q) as *mut _,                // q (recomputed)
                    dptr(&bufs.k) as *mut _,                // k (recomputed)
                    dptr(&bufs.saved_v[layer]) as *mut _,   // v
                    dptr(&bufs.saved_attn_out[layer]) as *mut _, // out (forward output)
                    dptr(&bufs.saved_softmax_lse[layer]) as *mut _, // softmax_lse
                    dptr(&bufs.d_q) as *mut _,              // dq
                    dptr(&bufs.d_k) as *mut _,              // dk
                    dptr(&bufs.d_v) as *mut _,              // dv
                    dptr(&bufs.flash_dq_accum) as *mut _,   // dq_accum (f32)
                    dptr(&bufs.flash_dsoftmax_sum) as *mut _, // dsoftmax_sum (f32)
                    dptr(&bufs.fa3_softmax_lse_log2) as *mut _, // FA3: softmax_lse_log2
                    dptr(&bufs.fa3_dq_semaphore) as *mut _, // FA3: dq_semaphore
                    batch_stride,                           // q_batch_stride
                    batch_stride,                           // k_batch_stride
                    batch_stride,                           // v_batch_stride
                    batch_stride,                           // o_batch_stride
                    batch_stride,                           // do_batch_stride
                    batch_stride,                           // dq_batch_stride
                    batch_stride,                           // dk_batch_stride
                    batch_stride,                           // dv_batch_stride
                    row_stride,                             // q_row_stride
                    row_stride,                             // k_row_stride
                    row_stride,                             // v_row_stride
                    row_stride,                             // o_row_stride
                    row_stride,                             // do_row_stride
                    row_stride,                             // dq_row_stride
                    row_stride,                             // dk_row_stride
                    row_stride,                             // dv_row_stride
                    head_stride,                            // q_head_stride
                    head_stride,                            // k_head_stride
                    head_stride,                            // v_head_stride
                    head_stride,                            // o_head_stride
                    head_stride,                            // do_head_stride
                    head_stride,                            // dq_head_stride
                    head_stride,                            // dk_head_stride
                    head_stride,                            // dv_head_stride
                    b as u32,                               // b
                    N_HEAD as u32,                          // h
                    N_KV_HEAD as u32,                       // h_k
                    HEAD_DIM as u32,                        // d
                    HEAD_DIM as u32,                        // d_rounded
                    softmax_scale,
                    SEQ as u32,                             // seqlen_q
                    SEQ as u32,                             // seqlen_k
                    1,                                      // is_bf16
                    is_causal,
                    window_size_left,
                    window_size_right,
                    0.0,                                    // softcap
                    0,                                      // deterministic
                    132,                                    // num_sm (H100 SXM)
                    stream,
                );
            }
        }

        // QK-Norm + RoPE backward:
        gemm.matmul(&bufs.saved_xn[layer], &bufs.layer_weights[layer].wq, &mut bufs.q, bt, D_MODEL, D_MODEL);
        unsafe {
            ffi::fused_rope_fwd(
                dptr(&bufs.q) as *const c_void,
                dptr(&bufs.cos) as *const c_void,
                dptr(&bufs.sin) as *const c_void,
                dptr(&bufs.q) as *mut c_void, // in-place
                (bt * N_HEAD * HEAD_DIM) as u32,
                SEQ as u32, N_HEAD as u32, HEAD_DIM as u32, stream,
            );
        }

        gemm.matmul(&bufs.saved_xn[layer], &bufs.layer_weights[layer].wk, &mut bufs.k, bt, D_MODEL, D_MODEL);
        unsafe {
            ffi::fused_rope_fwd(
                dptr(&bufs.k) as *const c_void,
                dptr(&bufs.cos) as *const c_void,
                dptr(&bufs.sin) as *const c_void,
                dptr(&bufs.k) as *mut c_void, // in-place
                (bt * N_KV_HEAD * HEAD_DIM) as u32,
                SEQ as u32, N_KV_HEAD as u32, HEAD_DIM as u32, stream,
            );
        }

        // QK-norm bwd: per-head rms_norm_bwd on d_q (rows = BT*N_HEAD, cols = HEAD_DIM)
        unsafe {
            ffi::fused_rms_norm_bwd(
                dptr(&bufs.q) as *const c_void,
                dptr(&bufs.d_q) as *const c_void,
                dptr(&bufs.d_q) as *mut c_void,
                (bt * N_HEAD) as u32, HEAD_DIM as u32, EPS, stream,
            );
            ffi::fused_rms_norm_bwd(
                dptr(&bufs.k) as *const c_void,
                dptr(&bufs.d_k) as *const c_void,
                dptr(&bufs.d_k) as *mut c_void,
                (bt * N_KV_HEAD) as u32, HEAD_DIM as u32, EPS, stream,
            );
        }

        // RoPE bwd (in-place inverse rotation)
        unsafe {
            ffi::fused_rope_bwd(
                dptr(&bufs.d_q) as *const c_void,
                dptr(&bufs.cos) as *const c_void,
                dptr(&bufs.sin) as *const c_void,
                dptr(&bufs.d_q) as *mut c_void, // in-place
                (bt * N_HEAD * HEAD_DIM) as u32,
                SEQ as u32, N_HEAD as u32, HEAD_DIM as u32, stream,
            );
            ffi::fused_rope_bwd(
                dptr(&bufs.d_k) as *const c_void,
                dptr(&bufs.cos) as *const c_void,
                dptr(&bufs.sin) as *const c_void,
                dptr(&bufs.d_k) as *mut c_void, // in-place
                (bt * N_KV_HEAD * HEAD_DIM) as u32,
                SEQ as u32, N_KV_HEAD as u32, HEAD_DIM as u32, stream,
            );
        }

        // =============================================================
        //  VE backward (odd layers)
        // =============================================================
        if ve_layer {
            let ve_w_ptr = dptr(bufs.layer_weights[layer].ve_weight.as_ref().unwrap());
            let ve_gate_w_ptr = dptr(bufs.layer_weights[layer].ve_gate.as_ref().unwrap());
            let d_ve_w_ptr = dptr(bufs.layer_grads[layer].ve_weight.as_ref().unwrap());
            let d_ve_gate_w_ptr = dptr(bufs.layer_grads[layer].ve_gate.as_ref().unwrap());

            // Recompute VE embeddings: lookup ve_weight[input_ids] --> bufs.ve
            unsafe {
                ffi::embedding_fwd(
                    dptr(&bufs.input_ids) as *const c_void,
                    ve_w_ptr as *const c_void,
                    dptr(&bufs.ve) as *mut c_void,
                    bt as i32, D_MODEL as i32, stream,
                );
            }

            // Recompute gate: extract xn[:, :VE_GATE_CH] into contiguous scratch
            unsafe {
                ffi::slice_cols(
                    dptr(&bufs.saved_xn[layer]) as *const c_void,
                    dptr(&bufs.h) as *mut c_void,
                    bt as i32, D_MODEL as i32, VE_GATE_CH as i32, stream,
                );
            }
            // gate(BT, N_KV_HEAD) = xn_slice(BT, VE_GATE_CH) @ Wgate(N_KV_HEAD, VE_GATE_CH)^T
            {
                let ve_gate_w = bufs.layer_weights[layer].ve_gate.as_ref().unwrap();
                gemm.matmul(&bufs.h, ve_gate_w, &mut bufs.gate, bt, N_KV_HEAD, VE_GATE_CH);
            }

            // ve_apply_bwd: reads d_v, ve, gate --> writes d_ve (bufs.ve), d_gate (bufs.gate)
            unsafe {
                ffi::ve_apply_bwd(
                    dptr(&bufs.d_v) as *const c_void,
                    dptr(&bufs.ve) as *const c_void,
                    dptr(&bufs.gate) as *const c_void,
                    dptr(&bufs.ve) as *mut c_void,
                    dptr(&bufs.gate) as *mut c_void,
                    bt as i32, N_KV_HEAD as i32, HEAD_DIM as i32, stream,
                );
            }

            // VE gate dW: d_gate^T @ xn_slice --> d_Wgate (accumulate)
            {
                let d_ve_gate_w = bufs.layer_grads[layer].ve_gate.as_mut().unwrap();
                gemm.matmul_acc(&bufs.gate, &bufs.h, d_ve_gate_w, bt, N_KV_HEAD, VE_GATE_CH);
            }

            // VE gate dX: d_gate(bt, N_KV_HEAD) @ Wgate(N_KV_HEAD, VE_GATE_CH) -> d_xn_gate_slice(bt, VE_GATE_CH)
            {
                let ve_gate_w = bufs.layer_weights[layer].ve_gate.as_ref().unwrap();
                gemm.matmul_bwd_x(&bufs.gate, ve_gate_w, &mut bufs.h, bt, N_KV_HEAD, VE_GATE_CH);
            }

            // VE embedding bwd: scatter d_ve into d_ve_weight (atomicAdd)
            unsafe {
                ffi::embedding_bwd(
                    dptr(&bufs.input_ids) as *const c_void,
                    dptr(&bufs.ve) as *const c_void,
                    d_ve_w_ptr as *mut c_void,
                    bt as i32, VOCAB as i32, D_MODEL as i32, stream,
                );
            }
        }

        // =============================================================
        //  QKV backward (packed batched GEMMs)
        // =============================================================

        // Pack d_q, d_k, d_v (post QK-norm/RoPE bwd) into d_qkv = [d_q|d_k|d_v]
        {
            let btd_bytes = (bt * D_MODEL * std::mem::size_of::<bf16>()) as usize;
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.d_qkv), dptr(&bufs.d_q), btd_bytes, bufs.stream.cu_stream(),
                );
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr_at(&bufs.d_qkv, bt * D_MODEL), dptr(&bufs.d_k), btd_bytes, bufs.stream.cu_stream(),
                );
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr_at(&bufs.d_qkv, 2 * bt * D_MODEL), dptr(&bufs.d_v), btd_bytes, bufs.stream.cu_stream(),
                );
            }
        }

        // QKV dX: d_xn = d_q @ wq + d_k @ wk + d_v @ wv — one batched call with shared output
        crate::forward::raw_zero(&bufs.stream, &mut bufs.d_xn);
        gemm.matmul_batched_bwd_x_shared_out(
            &bufs.d_qkv, &bufs.layer_weights[layer].wqkv, &mut bufs.d_xn,
            bt, D_MODEL, D_MODEL, 3,
        );

        // Add VE gate gradient to d_xn (stored in bufs.h from VE gate dX above)
        if ve_layer {
            unsafe {
                ffi::add_slice_cols(
                    dptr(&bufs.d_xn) as *mut c_void,
                    dptr(&bufs.h) as *const c_void,
                    bt as i32, D_MODEL as i32, VE_GATE_CH as i32, stream,
                );
            }
        }

        // QKV dW: d_wqkv += d_qkv^T @ saved_xn — one batched call with shared X
        gemm.matmul_shared_x_batched_acc(
            &bufs.d_qkv, &bufs.saved_xn[layer], &mut bufs.layer_grads[layer].wqkv,
            bt, D_MODEL, D_MODEL, 3,
        );
        // Split wqkv_grad back into wq/wk/wv grads (D2D copies for Muon compatibility)
        {
            let dd_bytes = (D_MODEL * D_MODEL * std::mem::size_of::<bf16>()) as usize;
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.layer_grads[layer].wq), dptr(&bufs.layer_grads[layer].wqkv), dd_bytes, bufs.stream.cu_stream(),
                );
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.layer_grads[layer].wk), dptr_at(&bufs.layer_grads[layer].wqkv, D_MODEL * D_MODEL), dd_bytes, bufs.stream.cu_stream(),
                );
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.layer_grads[layer].wv), dptr_at(&bufs.layer_grads[layer].wqkv, 2 * D_MODEL * D_MODEL), dd_bytes, bufs.stream.cu_stream(),
                );
            }
        }

        // =============================================================
        //  [FUSED] Attention norm backward + residual add + residual scale backward
        // =============================================================
        unsafe {
            ffi::fused_residual_norm_bwd(
                dptr(&bufs.d_xn) as *const c_void,                        // grad w.r.t. normed output
                dptr(&bufs.d_x) as *const c_void,                         // incoming residual grad
                dptr(&bufs.saved_x_pre_attn_norm[layer]) as *const c_void, // x before scale
                dptr(&bufs.x0) as *const c_void,
                dptr_at(&bufs.resid_lambdas, layer) as *const c_void,
                dptr_at(&bufs.x0_lambdas, layer) as *const c_void,
                dptr(&bufs.d_x) as *mut c_void,                           // d_x out (overwritten)
                dptr(&bufs.d_x0) as *mut c_void,                          // d_x0 (accumulated)
                dptr_at(&bufs.resid_lambdas_grad, layer) as *mut f32,
                dptr_at(&bufs.x0_lambdas_grad, layer) as *mut f32,
                bt as u32, D_MODEL as u32, EPS, stream,
            );
        }
    }

    // =====================================================================
    //  Pre-loop backward
    // =====================================================================

    // d_x0 accumulated the x0-branch gradients (lambda_0 * grad) across all layers.
    // d_x after layer 0's residual_scale_bwd holds the gradient w.r.t. the initial x,
    // which is the SAME tensor as x0 (forward: x0 = x = rms_norm(emb)).
    // Total gradient w.r.t. rms_norm(emb) = d_x + d_x0.
    unsafe {
        ffi::residual_add(
            dptr(&bufs.d_x0) as *mut c_void,
            dptr(&bufs.d_x) as *const c_void,
            n_btd, stream,
        );
    }

    // rms_norm_bwd: emb, d_x0 --> d_emb (into bufs.xn scratch)
    unsafe {
        ffi::fused_rms_norm_bwd(
            dptr(&bufs.emb) as *const c_void,
            dptr(&bufs.d_x0) as *const c_void,
            dptr(&bufs.xn) as *mut c_void,
            bt as u32, D_MODEL as u32, EPS, stream,
        );
    }

    // embedding_bwd: scatter d_emb into d_wte (atomicAdd)
    unsafe {
        ffi::embedding_bwd(
            dptr(&bufs.input_ids) as *const c_void,
            dptr(&bufs.xn) as *const c_void,
            dptr(&bufs.wte_grad) as *mut c_void,
            bt as i32, VOCAB as i32, D_MODEL as i32, stream,
        );
    }
}

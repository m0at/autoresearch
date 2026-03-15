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

    backward_layer_range(bufs, gemm, 0, N_LAYER);

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

/// Run the per-layer backward pass for layers [start, end) in reverse order.
/// Extracted so both backward() and backward_staged() can share the layer body.
fn backward_layer_range(bufs: &mut BufferManager, gemm: &GemmRunner, start: usize, end: usize) {
    let b = bufs.batch_size;
    let bt = b * SEQ;
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;
    // moe_idx tracks position in moe_weights/moe_grads/saved_* Vecs.
    // For staged mode these only contain entries for owned MoE layers.
    // Start at count of owned MoE layers (we decrement before use since loop is reversed).
    let mut moe_idx = (start..end).filter(|i| is_moe_layer(*i)).count();

    for layer in (start..end).rev() {
        let ve_layer = has_ve(layer);
        let n_mlp = (bt * MLP_DIM) as i32;

        // =================================================================
        //  MLP backward (MoE or dense)
        // =================================================================

        if is_moe_layer(layer) {
            // =============================================================
            //  MoE backward (steps 1-12 from MOE_PLAN.md §7)
            // =============================================================
            moe_idx -= 1; // decrement before use (reverse iteration)

            let handle = *gemm.blas().handle();

            // Step 1: Recompute xn_mlp = rms_norm(saved_x_pre_mlp_norm) → bufs.xn
            unsafe {
                ffi::fused_rms_norm_fwd(
                    dptr(&bufs.saved_x_pre_mlp_norm[layer]) as *const c_void,
                    dptr(&bufs.xn) as *mut c_void,
                    bt as u32, D_MODEL as u32, EPS, stream,
                );
            }

            // Step 2: Reload saved dispatch tables
            let perm_ptr = dptr(&bufs.saved_token_perm[moe_idx]);
            let offsets_ptr = dptr(&bufs.saved_expert_offsets[moe_idx]);
            let gates_ptr = dptr(&bufs.saved_gate_values[moe_idx]);
            let indices_ptr = dptr(&bufs.saved_expert_indices[moe_idx]);
            let saved_probs_ptr = dptr(&bufs.saved_router_probs[moe_idx]);

            // DtoH copy of expert_offsets via C-side (same CUDA context as kernels)
            unsafe {
                ffi::moe_copy_offsets_to_host(
                    offsets_ptr as *const c_void,
                    bufs.moe_expert_offsets_host.as_mut_ptr() as *mut c_void,
                    bufs.stream.cu_stream() as ffi::CudaStream,
                );
            }

            let n_dispatched = bufs.moe_expert_offsets_host[N_EXPERTS] as usize; // BT * TOP_K

            // Step 3: Re-gather xn tokens using saved perm → bufs.h (reuse as gathered, [n_dispatched, D])
            // bufs.h is [BT, MLP_DIM] = BT*4096 bf16 slots; n_dispatched = BT*TOP_K = BT*2,
            // gathered needs n_dispatched * D_MODEL = BT*2*1024 bf16 slots = BT*2048 < BT*4096. Fits.
            unsafe {
                ffi::launch_moe_gather_tokens(
                    dptr(&bufs.xn) as *const c_void,
                    perm_ptr as *const c_void,
                    dptr(&bufs.h) as *mut c_void,
                    n_dispatched as i32,
                    stream,
                );
            }

            // Step 4: Recompute expert forward (FC1 → ReLU² → FC2) to get expert_output
            // Expert output stored in bufs.h_act (reuse [BT, MLP_DIM] = BT*4096 bf16;
            // expert_output needs n_dispatched * D_MODEL = BT*2048. Fits.)
            //
            // We also need h_pre_act for ReLU² backward (step 6), stored in bufs.d_h.
            // bufs.d_h is [BT, MLP_DIM] = BT*4096 bf16; we need n_dispatched * MLP_DIM_E
            // = BT*2*1024 = BT*2048. Fits.
            //
            // h_act_expert stored in bufs.d_xn temporarily (BT*D_MODEL = BT*1024 bf16;
            // expert h_act needs n_dispatched * MLP_DIM_E = BT*2048... doesn't fit in d_xn).
            // Instead, reuse bufs.q for expert h_act scratch (BT*D_MODEL = BT*1024;
            // BT*2*1024 = BT*2048 > BT*1024... also doesn't fit).
            //
            // Better approach: process experts sequentially with per-expert recompute
            // inline with backward (steps 4+5+6 merged per expert).
            // But the plan says: recompute full forward first, then scatter_bwd, then per-expert bwd.
            //
            // Let's follow the plan exactly. For intermediate storage:
            // - gathered:      bufs.h       [0 .. n_dispatched * D_MODEL)
            // - h_pre_act:     bufs.d_h     [0 .. n_dispatched * MLP_DIM_E)  (BT*2048, fits BT*4096)
            // - h_act_expert:  bufs.h_act   [0 .. n_dispatched * MLP_DIM_E)  (BT*2048, fits BT*4096)
            // - expert_output: bufs.attn_out [0 .. n_dispatched * D_MODEL)   (BT*1024, fits BT*1024)
            //   Wait, attn_out is [BT, D_MODEL] = BT*1024, and n_dispatched*D = BT*2*1024... doesn't fit.
            //
            // The simplest correct approach: use the same qkv buffer for expert_output.
            // bufs.qkv is [3*BT, D_MODEL] = 3*BT*1024 bf16 slots. n_dispatched*D = 2*BT*1024. Fits.
            let gathered_ptr = dptr(&bufs.h);                     // [n_dispatched, D_MODEL]
            let h_pre_act_ptr = dptr(&bufs.d_h);                 // [n_dispatched, MLP_DIM_E]
            let h_act_ptr = dptr(&bufs.h_act);                   // [n_dispatched, MLP_DIM_E]
            let expert_out_ptr = dptr(&bufs.qkv);                // [n_dispatched, D_MODEL]

            // Per-expert forward recompute
            for e in 0..N_EXPERTS {
                let off = bufs.moe_expert_offsets_host[e] as usize;
                let n_e = (bufs.moe_expert_offsets_host[e + 1] - bufs.moe_expert_offsets_host[e]) as usize;
                if n_e == 0 { continue; }

                let gathered_e = gathered_ptr + (off * D_MODEL * std::mem::size_of::<bf16>()) as u64;
                let h_pre_e = h_pre_act_ptr + (off * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;
                let h_act_e = h_act_ptr + (off * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;
                let out_e = expert_out_ptr + (off * D_MODEL * std::mem::size_of::<bf16>()) as u64;

                // Expert FC1 weight slice: expert_wfc[e*MLP_DIM_E..(e+1)*MLP_DIM_E, D_MODEL]
                let wfc_e = dptr(&bufs.moe_weights[moe_idx].expert_wfc)
                    + (e * MLP_DIM_E * D_MODEL * std::mem::size_of::<bf16>()) as u64;
                let wdn_e = dptr(&bufs.moe_weights[moe_idx].expert_wdn)
                    + (e * D_MODEL * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;

                // FC1: gathered_e[n_e, D] @ wfc_e[MLP_DIM_E, D]^T → h_pre_e[n_e, MLP_DIM_E]
                unsafe {
                    crate::forward::raw_gemm_matmul(
                        handle, gathered_e, wfc_e, h_pre_e,
                        n_e, MLP_DIM_E, D_MODEL, 0.0,
                    );
                }

                // ReLU²: h_pre_e → h_act_e
                unsafe {
                    ffi::relu_sq_fwd(
                        h_pre_e as *const c_void,
                        h_act_e as *mut c_void,
                        (n_e * MLP_DIM_E) as i32, stream,
                    );
                }

                // FC2: h_act_e[n_e, MLP_DIM_E] @ wdn_e[D, MLP_DIM_E]^T → out_e[n_e, D]
                unsafe {
                    crate::forward::raw_gemm_matmul(
                        handle, h_act_e, wdn_e, out_e,
                        n_e, D_MODEL, MLP_DIM_E, 0.0,
                    );
                }
            }

            // Step 5: Scatter backward — compute d_expert_output and d_gate
            // d_expert_output → bufs.d_qkv [n_dispatched, D_MODEL] (reuse, 3*BT*D slots, fits)
            // d_gates → bufs.moe_gate_values [BT, TOP_K] (f32 scratch, reused)
            let d_expert_out_ptr = dptr(&bufs.d_qkv);
            let d_gates_ptr = dptr(&bufs.moe_gate_values);

            unsafe {
                ffi::moe_scatter_bwd(
                    expert_out_ptr as *const c_void,        // expert_out (recomputed)
                    dptr(&bufs.d_x) as *const c_void,       // d_output (incoming grad)
                    gates_ptr as *const c_void,              // saved gates
                    indices_ptr as *const c_void,            // saved expert_indices
                    perm_ptr as *const c_void,               // saved token_perm
                    d_expert_out_ptr as *mut c_void,         // d_expert_out (output)
                    d_gates_ptr as *mut c_void,              // d_gates (output, f32)
                    bt as i32,
                    D_MODEL as i32,
                    stream,
                );
            }

            // Step 6: Per-expert backward
            for e in 0..N_EXPERTS {
                let off = bufs.moe_expert_offsets_host[e] as usize;
                let n_e = (bufs.moe_expert_offsets_host[e + 1] - bufs.moe_expert_offsets_host[e]) as usize;
                if n_e == 0 { continue; }

                let gathered_e = gathered_ptr + (off * D_MODEL * std::mem::size_of::<bf16>()) as u64;
                let h_pre_e = h_pre_act_ptr + (off * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;
                let h_act_e = h_act_ptr + (off * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;
                let d_expert_e = d_expert_out_ptr + (off * D_MODEL * std::mem::size_of::<bf16>()) as u64;

                let wfc_e = dptr(&bufs.moe_weights[moe_idx].expert_wfc)
                    + (e * MLP_DIM_E * D_MODEL * std::mem::size_of::<bf16>()) as u64;
                let wdn_e = dptr(&bufs.moe_weights[moe_idx].expert_wdn)
                    + (e * D_MODEL * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;
                let d_wfc_e = dptr(&bufs.moe_grads[moe_idx].expert_wfc)
                    + (e * MLP_DIM_E * D_MODEL * std::mem::size_of::<bf16>()) as u64;
                let d_wdn_e = dptr(&bufs.moe_grads[moe_idx].expert_wdn)
                    + (e * D_MODEL * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;

                // Scratch for d_h_act_e and d_h_e: reuse bufs.qkv tail region
                // bufs.qkv has 3*BT*D slots. expert_out used [0..2*BT*D).
                // d_expert_out uses d_qkv [0..2*BT*D). We need per-expert scratch
                // of size n_e * MLP_DIM_E. Use the v-slice region of qkv (offset 2*BT*D).
                // That's BT*D = BT*1024 bf16 slots. n_e * MLP_DIM_E ≤ BT*1024. Fits.
                let d_h_act_e = dptr(&bufs.qkv) + (2 * bt * D_MODEL * std::mem::size_of::<bf16>()) as u64;

                // FC2 dW: d_expert_e[n_e, D]^T @ h_act_e[n_e, MLP_DIM_E] → d_wdn_e (accumulate)
                unsafe {
                    crate::forward::raw_gemm_matmul_acc(
                        handle, d_expert_e, h_act_e, d_wdn_e,
                        n_e, D_MODEL, MLP_DIM_E,
                    );
                }

                // FC2 dX: d_expert_e[n_e, D] @ wdn_e[D, MLP_DIM_E] → d_h_act_e[n_e, MLP_DIM_E]
                unsafe {
                    crate::forward::raw_gemm_bwd_x(
                        handle, d_expert_e, wdn_e, d_h_act_e,
                        n_e, D_MODEL, MLP_DIM_E, 0.0,
                    );
                }

                // ReLU² backward: d_h_act_e, h_pre_act_e → d_h_e (overwrite h_pre_e in-place)
                unsafe {
                    ffi::relu_sq_bwd(
                        h_pre_e as *const c_void,
                        d_h_act_e as *const c_void,
                        h_pre_e as *mut c_void,   // reuse h_pre_e for d_h_e
                        (n_e * MLP_DIM_E) as i32, stream,
                    );
                }
                let d_h_e = h_pre_e; // now contains d_h

                // FC1 dW: d_h_e[n_e, MLP_DIM_E]^T @ gathered_e[n_e, D] → d_wfc_e (accumulate)
                unsafe {
                    crate::forward::raw_gemm_matmul_acc(
                        handle, d_h_e, gathered_e, d_wfc_e,
                        n_e, MLP_DIM_E, D_MODEL,
                    );
                }

                // FC1 dX: d_h_e[n_e, MLP_DIM_E] @ wfc_e[MLP_DIM_E, D] → d_gathered_e[n_e, D]
                // Write back into gathered buffer (bufs.h) — it's no longer needed for forward.
                unsafe {
                    crate::forward::raw_gemm_bwd_x(
                        handle, d_h_e, wfc_e, gathered_e,
                        n_e, MLP_DIM_E, D_MODEL, 0.0,
                    );
                }
            }

            // Step 7: Gather backward — scatter d_gathered (in bufs.h) → d_xn
            // Kernel handles both top-k passes internally (k=0 write, k=1 accumulate)
            unsafe {
                ffi::moe_gather_bwd(
                    dptr(&bufs.h) as *const c_void,     // d_gathered
                    perm_ptr as *const c_void,           // token_perm
                    dptr(&bufs.d_xn) as *mut c_void,    // d_xn (output)
                    bt as i32,
                    D_MODEL as i32,
                    stream,
                );
            }

            // Step 8: Aux loss backward → d_router_probs
            // d_probs written into bufs.moe_router_probs (f32 scratch, [BT, N_EXPERTS])
            crate::forward::raw_zero(&bufs.stream, &mut bufs.moe_router_probs);
            unsafe {
                ffi::load_balance_loss_bwd(
                    dptr(&bufs.moe_router_probs) as *mut c_void,  // d_probs (output)
                    dptr(&bufs.saved_expert_offsets[moe_idx]) as *const c_void, // counts from offsets
                    bt as i32,
                    N_EXPERTS as i32,
                    AUX_LOSS_COEFF,
                    stream,
                );
            }

            // Step 9: Router softmax+topk backward
            // (d_gates, d_router_probs) → d_router_logits (bf16, in bufs.moe_router_logits scratch)
            unsafe {
                ffi::moe_router_softmax_topk_bwd(
                    saved_probs_ptr as *const c_void,           // saved probs (bf16 → cast in kernel)
                    gates_ptr as *const c_void,                  // saved gates (f32)
                    indices_ptr as *const c_void,                // saved indices
                    d_gates_ptr as *const c_void,                // d_gates (f32, from step 5)
                    dptr(&bufs.moe_router_probs) as *const c_void, // d_probs (f32, from step 8)
                    dptr(&bufs.moe_router_logits) as *mut c_void,  // d_logits (bf16, output)
                    bt as i32,
                    stream,
                );
            }

            // Step 10: Router dW: d_router_logits[BT, N_EXPERTS]^T @ xn[BT, D] → d_w_router (accumulate)
            unsafe {
                crate::forward::raw_gemm_matmul_acc(
                    handle,
                    dptr(&bufs.moe_router_logits),  // d_logits [BT, N_EXPERTS]
                    dptr(&bufs.xn),                 // xn [BT, D_MODEL]
                    dptr(&bufs.moe_grads[moe_idx].w_router), // d_w_router [N_EXPERTS, D_MODEL]
                    bt, N_EXPERTS, D_MODEL,
                );
            }

            // Step 11: Router dX: d_router_logits[BT, N_EXPERTS] @ w_router[N_EXPERTS, D] → d_xn (accumulate)
            // d_xn already has gather-bwd contribution; accumulate router grad with beta=1.0
            unsafe {
                crate::forward::raw_gemm_bwd_x(
                    handle,
                    dptr(&bufs.moe_router_logits),              // [BT, N_EXPERTS]
                    dptr(&bufs.moe_weights[moe_idx].w_router),    // [N_EXPERTS, D_MODEL]
                    dptr(&bufs.d_xn),                           // [BT, D_MODEL]
                    bt, N_EXPERTS, D_MODEL, 1.0,                // beta=1.0 to accumulate
                );
            }

            // Step 12: rms_norm_bwd + residual into d_x
            unsafe {
                ffi::fused_rms_norm_bwd_residual_add(
                    dptr(&bufs.saved_x_pre_mlp_norm[layer]) as *const c_void,
                    dptr(&bufs.d_xn) as *const c_void,
                    dptr(&bufs.d_x) as *mut c_void,
                    bt as u32, D_MODEL as u32, EPS, stream,
                );
            }

        } else {
            // =============================================================
            //  Dense MLP backward (unchanged)
            // =============================================================

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
                &bufs.xn, bufs.layer_weights[layer].wfc.as_ref().unwrap(),
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
            gemm.matmul_acc(&bufs.d_x, &bufs.h, bufs.layer_grads[layer].wdn.as_mut().unwrap(), bt, D_MODEL, MLP_DIM);

            // FC2 dX: d_x(BT, D_MODEL) @ Wdn(D_MODEL, MLP_DIM) --> d_h_act(BT, MLP_DIM)
            gemm.matmul_bwd_x(&bufs.d_x, bufs.layer_weights[layer].wdn.as_ref().unwrap(), &mut bufs.h_act, bt, D_MODEL, MLP_DIM);

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
            gemm.matmul_acc(&bufs.d_h, &bufs.xn, bufs.layer_grads[layer].wfc.as_mut().unwrap(), bt, MLP_DIM, D_MODEL);

            // FC1 dX: d_h(BT, MLP_DIM) @ Wfc(MLP_DIM, D_MODEL) --> d_xn(BT, D_MODEL)
            gemm.matmul_bwd_x(&bufs.d_h, bufs.layer_weights[layer].wfc.as_ref().unwrap(), &mut bufs.d_xn, bt, MLP_DIM, D_MODEL);

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
            let is_local = window_sz < SEQ;
            let window_size_left: i32 = if is_local { window_sz as i32 } else { -1 };
            let window_size_right: i32 = if is_local { 0 } else { -1 };
            let is_causal: i32 = 1; // always causal

            // FA3 backward: writes into separate d_q/d_k/d_v
            unsafe {
                ffi::run_mha_backward_v3(
                    dptr(&bufs.attn_out) as *mut _,
                    dptr(&bufs.q) as *mut _,
                    dptr(&bufs.k) as *mut _,
                    dptr(&bufs.saved_v[layer]) as *mut _,
                    dptr(&bufs.saved_attn_out[layer]) as *mut _,
                    dptr(&bufs.saved_softmax_lse[layer]) as *mut _,
                    dptr(&bufs.d_q) as *mut _,
                    dptr(&bufs.d_k) as *mut _,
                    dptr(&bufs.d_v) as *mut _,
                    dptr(&bufs.flash_dq_accum) as *mut _,
                    dptr(&bufs.flash_dsoftmax_sum) as *mut _,
                    dptr(&bufs.fa3_softmax_lse_log2) as *mut _,
                    dptr(&bufs.fa3_dq_semaphore) as *mut _,
                    batch_stride, batch_stride, batch_stride, batch_stride, batch_stride,
                    batch_stride, batch_stride, batch_stride,
                    row_stride, row_stride, row_stride, row_stride, row_stride,
                    row_stride, row_stride, row_stride,
                    head_stride, head_stride, head_stride, head_stride, head_stride,
                    head_stride, head_stride, head_stride,
                    b as u32, N_HEAD as u32, N_KV_HEAD as u32,
                    HEAD_DIM as u32, HEAD_DIM as u32,
                    softmax_scale,
                    SEQ as u32, SEQ as u32,
                    1, is_causal, window_size_left, window_size_right,
                    0.0, 0, 132, stream,
                );
            }

            // Check for FA3 backward errors
            unsafe {
                let err = cudarc::driver::sys::cuCtxSynchronize();
                if err != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    eprintln!("[FA3 BWD] CUDA error after backward layer {layer}: {:?}", err);
                }
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
}

/// Pipeline-parallel backward pass. Runs only the stages this GPU owns:
/// - If has_head: CE loss backward + lm_head grad (post-loop)
/// - Per-layer backward for layers [layer_start, layer_end) in reverse
/// - If has_embedding: embedding gradient (pre-loop)
/// Staged backward: like `backward()` but only processes layers owned by this
/// stage, gated by `has_head` and `has_embedding`. When `skip_embedding_bwd` is
/// true, the embedding backward (pre-loop section) is skipped — the caller is
/// responsible for running `embedding_backward()` after accumulating d_x0 from
/// all pipeline stages.
pub fn backward_staged_ex(bufs: &mut BufferManager, gemm: &GemmRunner, grad_accum_steps: usize, skip_embedding_bwd: bool) {
    unsafe { crate::ffi::cuda_set_device(bufs.stream.context().cu_device()); }
    bufs.stream.context().bind_to_thread().expect("bind CUDA context");

    let b = bufs.batch_size;
    let bt = b * SEQ;
    let n_btd = (bt * D_MODEL) as i32;
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;

    // Zero d_x0 before per-layer loop (it accumulates via read-modify-write
    // in residual_scale_bwd, so must start at zero each backward call).
    crate::forward::raw_zero(&bufs.stream, &mut bufs.d_x0);

    // =====================================================================
    //  Post-loop backward (last stage only)
    // =====================================================================

    if bufs.has_head {
        // Recompute xn_final = rms_norm(bufs.x) into bufs.xn (needed for lm_head dW).
        unsafe {
            ffi::fused_rms_norm_fwd(
                dptr(&bufs.x) as *const c_void,
                dptr(&bufs.xn) as *mut c_void,
                bt as u32, D_MODEL as u32, EPS, stream,
            );
        }

        // [CHUNKED CE BWD + LM HEAD BWD]
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

                // (a) Recompute logits
                unsafe {
                    crate::forward::raw_gemm_matmul(
                        handle, xn_off, lm_head_ptr, logits_ptr,
                        cur, VOCAB, d, 0.0,
                    );
                }

                // (b) CE bwd
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

                // (c) LM head dX
                unsafe {
                    crate::forward::raw_gemm_bwd_x(
                        handle, d_logits_ptr, lm_head_ptr, d_xn_off,
                        cur, VOCAB, d, 0.0,
                    );
                }

                // (d) LM head dW (accumulate)
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
    }

    // =====================================================================
    //  Per-layer backward (layer_end-1 --> layer_start)
    // =====================================================================

    backward_layer_range(bufs, gemm, bufs.layer_start, bufs.layer_end);

    // =====================================================================
    //  Pre-loop backward (first stage only)
    // =====================================================================

    if bufs.has_embedding && !skip_embedding_bwd {
        embedding_backward(bufs);
    }
}

/// Original backward_staged entry point (backward compat). Does NOT skip
/// embedding backward.
pub fn backward_staged(bufs: &mut BufferManager, gemm: &GemmRunner, grad_accum_steps: usize) {
    backward_staged_ex(bufs, gemm, grad_accum_steps, false);
}

/// Run ONLY the embedding backward on stage 0 (pre-loop section).
/// Expects d_x0 to contain the fully-accumulated gradient from ALL pipeline
/// stages, and d_x to contain the gradient from layer 0's residual_scale_bwd.
pub fn embedding_backward(bufs: &mut BufferManager) {
    assert!(bufs.has_embedding, "embedding_backward called on non-embedding stage");
    bufs.stream.context().bind_to_thread().expect("bind CUDA context");

    let bt = bufs.batch_size * SEQ;
    let n_btd = (bt * D_MODEL) as i32;
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;

    unsafe {
        ffi::residual_add(
            dptr(&bufs.d_x0) as *mut c_void,
            dptr(&bufs.d_x) as *const c_void,
            n_btd, stream,
        );
    }

    unsafe {
        ffi::fused_rms_norm_bwd(
            dptr(&bufs.emb) as *const c_void,
            dptr(&bufs.d_x0) as *const c_void,
            dptr(&bufs.xn) as *mut c_void,
            bt as u32, D_MODEL as u32, EPS, stream,
        );
    }

    unsafe {
        ffi::embedding_bwd(
            dptr(&bufs.input_ids) as *const c_void,
            dptr(&bufs.xn) as *const c_void,
            dptr(&bufs.wte_grad) as *mut c_void,
            bt as i32, VOCAB as i32, D_MODEL as i32, stream,
        );
    }
}

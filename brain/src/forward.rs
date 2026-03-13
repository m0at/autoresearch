use cudarc::cublas::safe::{Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::bf16;

use crate::buffer::BufferManager;
use crate::config::*;
use crate::ffi;
use crate::gemm::GemmRunner;

// ---------------------------------------------------------------------------
// Raw pointer helpers.
//
// We use cudarc's DevicePtr trait to extract the raw CUdeviceptr. CudaSlice
// is NOT #[repr(C)] so we cannot assume field layout.
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

/// Raw device-to-device copy — bypasses cudarc's event tracking so it works
/// inside CUDA graph capture (no cross-stream dependency).
#[inline(always)]
fn raw_dtod<T>(stream: &CudaStream, src: &CudaSlice<T>, dst: &mut CudaSlice<T>) {
    let nbytes = src.len() * std::mem::size_of::<T>();
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dptr(dst), dptr(src), nbytes, stream.cu_stream());
    }
}

/// Raw memset to zero — bypasses cudarc's event tracking for graph capture.
#[inline(always)]
pub fn raw_zero<T>(stream: &CudaStream, dst: &mut CudaSlice<T>) {
    let nbytes = dst.len() * std::mem::size_of::<T>();
    unsafe {
        cudarc::driver::sys::cuMemsetD8Async(dptr(dst), 0, nbytes, stream.cu_stream());
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

/// Run the complete forward pass. Per-token CE losses are left on device in
/// `bufs.h_act` (as f32), and the total loss (sum of per-token losses) is
/// atomically accumulated into `bufs.loss[0]` (f32, device). No host sync.
///
/// The caller must zero `bufs.loss` once before the gradient-accumulation
/// loop so that the sum spans all micro-steps. After sync, divide
/// `bufs.loss[0]` by `(bt * grad_accum_steps)` to get the mean.
pub fn forward(bufs: &mut BufferManager, gemm: &GemmRunner) {
    // Bind CUDA context to this thread — required for raw driver API calls.
    bufs.stream.context().bind_to_thread().expect("bind CUDA context");

    let b = bufs.batch_size;
    let t = SEQ;
    let bt = b * t;
    let d = D_MODEL;
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;

    // Use the shared cuBLAS handle for VE-gate strided GEMM.
    let ve_blas = gemm.blas();

    // =====================================================================
    // PRE-LOOP
    // =====================================================================

    // [EMB] Embedding lookup: input_ids[B*T] -> emb[B*T, D_MODEL]
    unsafe {
        ffi::embedding_fwd(
            dptr(&bufs.input_ids) as *const _,
            dptr(&bufs.wte) as *const _,
            dptr(&bufs.emb) as *mut _,
            bt as i32, d as i32, stream,
        );
    }

    // [NORM0] RMSNorm: emb -> x
    unsafe {
        ffi::fused_rms_norm_fwd(
            dptr(&bufs.emb) as *const _,
            dptr(&bufs.x) as *mut _,
            bt as u32, d as u32, EPS, stream,
        );
    }

    // [COPY] x0 = x (initial hidden state, preserved across all layers)
    raw_dtod(&bufs.stream, &bufs.x, &mut bufs.x0);

    // =====================================================================
    // PER-LAYER LOOP
    // =====================================================================
    for layer in 0..N_LAYER {

        // Save x pre attn-norm for backward
        raw_dtod(&bufs.stream, &bufs.x, &mut bufs.saved_x_pre_attn_norm[layer]);

        // [RSCALE+NORM1] Fused residual_scale + RMSNorm:
        //   x = lambda_r * x_pre + lambda_0 * x0;  xn = rms_norm(x)
        // Norm output written directly to saved_xn[layer] (eliminates dtod copy).
        unsafe {
            ffi::fused_residual_norm_fwd(
                dptr(&bufs.x) as *const _,
                dptr(&bufs.x0) as *const _,
                dptr_at(&bufs.resid_lambdas, layer) as *const _,
                dptr_at(&bufs.x0_lambdas, layer) as *const _,
                dptr(&bufs.x) as *mut _,              // scaled x (for residual path)
                dptr(&bufs.saved_xn[layer]) as *mut _, // normed output → directly to save buffer
                bt as u32, d as u32, EPS, stream,
            );
        }

        // =============================================================
        // ATTENTION
        // =============================================================

        // [QKV] Packed QKV: single batched GEMM with shared xn input.
        // wqkv = [wq; wk; wv] stacked [3D, D]. qkv output = [q|k|v] blocked [3*BT, D].
        // Reads saved_xn[layer] once instead of 3x — same as torch.compile's QKV fusion.
        gemm.matmul_shared_x_batched(
            &bufs.saved_xn[layer],
            &bufs.layer_weights[layer].wqkv,
            &mut bufs.qkv,
            bt, d, d, 3,
        );

        // [VE] Value Embeddings (odd layers only)
        if has_ve(layer) {
            let ve_w = bufs.layer_weights[layer].ve_weight.as_ref().unwrap();
            let ve_g = bufs.layer_weights[layer].ve_gate.as_ref().unwrap();

            // VE lookup: input_ids -> ve[bt, d]
            unsafe {
                ffi::embedding_fwd(
                    dptr(&bufs.input_ids) as *const _,
                    dptr(ve_w) as *const _,
                    dptr(&bufs.ve) as *mut _,
                    bt as i32, d as i32, stream,
                );
            }

            // Gate: saved_xn[layer][:, :VE_GATE_CH] @ Wgate[N_KV_HEAD, VE_GATE_CH]^T -> gate[bt, N_KV_HEAD]
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: N_KV_HEAD as i32,
                n: bt as i32,
                k: VE_GATE_CH as i32,
                alpha: bf16::from_f32(1.0),
                lda: VE_GATE_CH as i32,
                ldb: d as i32,
                beta: bf16::from_f32(0.0),
                ldc: N_KV_HEAD as i32,
            };
            unsafe {
                ve_blas.gemm(cfg, ve_g, &bufs.saved_xn[layer], &mut bufs.gate)
            }.expect("VE gate GEMM failed");

            // Copy v from qkv to saved_v, then VE-apply in-place on saved_v
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.saved_v[layer]),
                    dptr_at(&bufs.qkv, 2 * bt * d),
                    bt * d * std::mem::size_of::<bf16>(),
                    bufs.stream.cu_stream(),
                );
            }

            // [VE_APPLY] saved_v += 2 * sigmoid(gate).unsqueeze(-1) * ve
            unsafe {
                ffi::ve_apply_fwd(
                    dptr(&bufs.saved_v[layer]) as *mut _,
                    dptr(&bufs.ve) as *const _,
                    dptr(&bufs.gate) as *const _,
                    bt as i32, N_KV_HEAD as i32, HEAD_DIM as i32, stream,
                );
            }
        } else {
            // Non-VE layers: copy v from qkv to saved_v
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.saved_v[layer]),
                    dptr_at(&bufs.qkv, 2 * bt * d),
                    bt * d * std::mem::size_of::<bf16>(),
                    bufs.stream.cu_stream(),
                );
            }
        }

        // [ROPE+QKNORM] Fused RoPE + per-head RMSNorm on q and k.
        // Read from qkv slices, write directly to saved_q/saved_k (eliminates dtod copies).
        unsafe {
            ffi::fused_rope_rms_norm_fwd(
                dptr(&bufs.qkv) as *const _,               // q at offset 0 in qkv
                dptr(&bufs.cos) as *const _,
                dptr(&bufs.sin) as *const _,
                dptr(&bufs.saved_q[layer]) as *mut _,       // write directly to save buffer
                (bt * N_HEAD) as u32,
                t as u32, N_HEAD as u32, HEAD_DIM as u32, EPS, stream,
            );
            ffi::fused_rope_rms_norm_fwd(
                dptr_at(&bufs.qkv, bt * d) as *const _,    // k at offset BT*D in qkv
                dptr(&bufs.cos) as *const _,
                dptr(&bufs.sin) as *const _,
                dptr(&bufs.saved_k[layer]) as *mut _,       // write directly to save buffer
                (bt * N_KV_HEAD) as u32,
                t as u32, N_KV_HEAD as u32, HEAD_DIM as u32, EPS, stream,
            );
        }

        // [FATTN] Flash attention forward — reads from saved_q/k/v, writes to saved_attn_out
        flash_attn_fwd(bufs, b, t, layer, layer,
            dptr(&bufs.saved_q[layer]),
            dptr(&bufs.saved_k[layer]),
            dptr(&bufs.saved_v[layer]),
            dptr(&bufs.saved_attn_out[layer]),
        );

        // [OPROJ] Out projection: saved_attn_out[bt,d] @ Wo[d,d]^T -> xn (scratch)
        gemm.matmul(
            &bufs.saved_attn_out[layer], &bufs.layer_weights[layer].wo,
            &mut bufs.xn, bt, d, d,
        );

        // [RESID1+NORM2] Fused residual add + RMSNorm:
        //   x += out_proj_result;  xn = rms_norm(x)
        // Saves one full BT*D read pass vs separate kernels.
        unsafe {
            ffi::fused_residual_add_rms_norm_fwd(
                dptr(&bufs.x) as *mut _,
                dptr(&bufs.xn) as *const _,
                dptr(&bufs.xn) as *mut _, // reuse xn: proj read first, then normed output written
                bt as u32, d as u32, EPS, stream,
            );
        }

        // =============================================================
        // MLP
        // =============================================================

        // Save x (post-residual-add) pre mlp-norm for backward
        raw_dtod(&bufs.stream, &bufs.x, &mut bufs.saved_x_pre_mlp_norm[layer]);

        // [FC1] xn[bt,d] @ Wfc[MLP_DIM,d]^T -> saved_h_pre_act (eliminates dtod copy)
        gemm.matmul(
            &bufs.xn, &bufs.layer_weights[layer].wfc,
            &mut bufs.saved_h_pre_act[layer], bt, MLP_DIM, d,
        );

        // [RELU2] ReLU^2: saved_h_pre_act -> h_act
        unsafe {
            ffi::relu_sq_fwd(
                dptr(&bufs.saved_h_pre_act[layer]) as *const _,
                dptr(&bufs.h_act) as *mut _,
                (bt * MLP_DIM) as i32, stream,
            );
        }

        // [FC2] h_act[bt,MLP_DIM] @ Wdn[d,MLP_DIM]^T -> xn[bt,d]
        gemm.matmul(
            &bufs.h_act, &bufs.layer_weights[layer].wdn,
            &mut bufs.xn, bt, d, MLP_DIM,
        );

        // [LAYER STAT] Capture per-neuron mean absolute activation (for neuron rinsing)
        unsafe {
            ffi::neuron_act_norm_bf16(
                dptr(&bufs.h_act) as *const std::ffi::c_void,
                bt as i32,
                MLP_DIM as i32,
                dptr(&bufs.layer_neuron_act_norms) as *mut f32,
                layer as i32,
                stream,
            );
        }

        // [RESID2] x += mlp_result
        unsafe {
            ffi::residual_add(
                dptr(&bufs.x) as *mut _,
                dptr(&bufs.xn) as *const _,
                (bt * d) as i32, stream,
            );
        }
    }

    // =====================================================================
    // POST-LOOP
    // =====================================================================

    // [NORMF] Final RMSNorm: x -> xn
    unsafe {
        ffi::fused_rms_norm_fwd(
            dptr(&bufs.x) as *const _,
            dptr(&bufs.xn) as *mut _,
            bt as u32, d as u32, EPS, stream,
        );
    }

    // [LMHEAD] xn[bt,d] @ lm_head[VOCAB,d]^T -> logits[bt,VOCAB]
    gemm.matmul(&bufs.xn, &bufs.lm_head, &mut bufs.logits, bt, VOCAB, d);

    // [CELOSS] Cross-entropy with fused softcap: raw bf16 logits are cast to
    // f32 and softcapped (cap*tanh(x/cap)) inside the kernel before the
    // softmax, matching Python's `.float()` cast. Per-token losses written to
    // h_act scratch; the loss sum is atomically accumulated into bufs.loss[0].
    // The caller zeros bufs.loss once before the grad-accum loop so that the
    // sum spans all micro-steps.
    let per_token_loss_ptr = dptr(&bufs.h_act);
    unsafe {
        ffi::fused_cross_entropy_fwd(
            dptr(&bufs.logits) as *const _,
            dptr(&bufs.targets) as *const _,
            per_token_loss_ptr as *mut _,
            dptr(&bufs.loss) as *mut f32,
            bt as u32,
            VOCAB as u32,
            SOFTCAP, stream,
        );
    }

    // Per-token losses reside on device at per_token_loss_ptr (in h_act).
    // The running loss sum is in bufs.loss[0] (f32, device).
    // No host sync — backward uses d_logits computed from logits+targets, not
    // the scalar loss. The caller reads back bufs.loss when needed for logging.
}

/// Eval-only forward pass: identical to `forward()` but skips all
/// activation saves (saved_x_pre_attn_norm, saved_xn, saved_v, saved_q/k,
/// saved_attn_out, saved_x_pre_mlp_norm, saved_h_pre_act).  Softmax LSE
/// is written to slot 0 as a scratch buffer since backward is never called.
/// Per-token CE losses are still written to bufs.h_act for the caller.
pub fn forward_eval(bufs: &mut BufferManager, gemm: &GemmRunner) {
    bufs.stream.context().bind_to_thread().expect("bind CUDA context");

    let b = bufs.batch_size;
    let t = SEQ;
    let bt = b * t;
    let d = D_MODEL;
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;

    let ve_blas = gemm.blas();

    // [EMB] Embedding lookup
    unsafe {
        ffi::embedding_fwd(
            dptr(&bufs.input_ids) as *const _,
            dptr(&bufs.wte) as *const _,
            dptr(&bufs.emb) as *mut _,
            bt as i32, d as i32, stream,
        );
    }

    // [NORM0] RMSNorm: emb -> x
    unsafe {
        ffi::fused_rms_norm_fwd(
            dptr(&bufs.emb) as *const _,
            dptr(&bufs.x) as *mut _,
            bt as u32, d as u32, EPS, stream,
        );
    }

    // [COPY] x0 = x
    raw_dtod(&bufs.stream, &bufs.x, &mut bufs.x0);

    for layer in 0..N_LAYER {
        // [RSCALE+NORM1] Fused residual_scale + RMSNorm (no save of pre-norm x)
        unsafe {
            ffi::fused_residual_norm_fwd(
                dptr(&bufs.x) as *const _,
                dptr(&bufs.x0) as *const _,
                dptr_at(&bufs.resid_lambdas, layer) as *const _,
                dptr_at(&bufs.x0_lambdas, layer) as *const _,
                dptr(&bufs.x) as *mut _,
                dptr(&bufs.xn) as *mut _,
                bt as u32, d as u32, EPS, stream,
            );
        }

        // [QKV] Packed QKV GEMM (eval path — same batched call, no saves)
        gemm.matmul_shared_x_batched(
            &bufs.xn,
            &bufs.layer_weights[layer].wqkv,
            &mut bufs.qkv,
            bt, d, d, 3,
        );

        // [VE] Value Embeddings (odd layers only)
        // For VE layers, copy v-slice from qkv to bufs.v, then apply VE in-place.
        // For non-VE layers, v stays in qkv and is passed as raw pointer to FA3.
        if has_ve(layer) {
            let ve_w = bufs.layer_weights[layer].ve_weight.as_ref().unwrap();
            let ve_g = bufs.layer_weights[layer].ve_gate.as_ref().unwrap();
            unsafe {
                ffi::embedding_fwd(
                    dptr(&bufs.input_ids) as *const _,
                    dptr(ve_w) as *const _,
                    dptr(&bufs.ve) as *mut _,
                    bt as i32, d as i32, stream,
                );
            }
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: N_KV_HEAD as i32,
                n: bt as i32,
                k: VE_GATE_CH as i32,
                alpha: bf16::from_f32(1.0),
                lda: VE_GATE_CH as i32,
                ldb: d as i32,
                beta: bf16::from_f32(0.0),
                ldc: N_KV_HEAD as i32,
            };
            unsafe {
                ve_blas.gemm(cfg, ve_g, &bufs.xn, &mut bufs.gate)
            }.expect("VE gate GEMM failed");
            // Copy v from qkv to bufs.v for VE in-place modification
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.v),
                    dptr_at(&bufs.qkv, 2 * bt * d),
                    bt * d * std::mem::size_of::<bf16>(),
                    bufs.stream.cu_stream(),
                );
            }
            unsafe {
                ffi::ve_apply_fwd(
                    dptr(&bufs.v) as *mut _,
                    dptr(&bufs.ve) as *const _,
                    dptr(&bufs.gate) as *const _,
                    bt as i32, N_KV_HEAD as i32, HEAD_DIM as i32, stream,
                );
            }
        }

        // [ROPE+QKNORM] In-place on qkv q/k slices (no save needed for eval)
        unsafe {
            ffi::fused_rope_rms_norm_fwd(
                dptr(&bufs.qkv) as *const _,
                dptr(&bufs.cos) as *const _,
                dptr(&bufs.sin) as *const _,
                dptr(&bufs.qkv) as *mut _,              // in-place on q slice
                (bt * N_HEAD) as u32,
                t as u32, N_HEAD as u32, HEAD_DIM as u32, EPS, stream,
            );
            ffi::fused_rope_rms_norm_fwd(
                dptr_at(&bufs.qkv, bt * d) as *const _,
                dptr(&bufs.cos) as *const _,
                dptr(&bufs.sin) as *const _,
                dptr_at(&bufs.qkv, bt * d) as *mut _,   // in-place on k slice
                (bt * N_KV_HEAD) as u32,
                t as u32, N_KV_HEAD as u32, HEAD_DIM as u32, EPS, stream,
            );
        }

        // [FATTN] Flash attention — reuse lse slot 0 as scratch
        // For VE layers, v is in bufs.v (post VE-apply); otherwise v is in qkv v-slice
        let v_ptr = if has_ve(layer) {
            dptr(&bufs.v)
        } else {
            dptr_at(&bufs.qkv, 2 * bt * d)
        };
        flash_attn_fwd(bufs, b, t, layer, 0,
            dptr(&bufs.qkv),                   // q at offset 0
            dptr_at(&bufs.qkv, bt * d),        // k at offset BT*D
            v_ptr,
            dptr(&bufs.attn_out),
        );

        // [OPROJ] Out projection (no save of attn_out)
        gemm.matmul(
            &bufs.attn_out, &bufs.layer_weights[layer].wo,
            &mut bufs.xn, bt, d, d,
        );

        // [RESID1+NORM2]
        unsafe {
            ffi::fused_residual_add_rms_norm_fwd(
                dptr(&bufs.x) as *mut _,
                dptr(&bufs.xn) as *const _,
                dptr(&bufs.xn) as *mut _,
                bt as u32, d as u32, EPS, stream,
            );
        }

        // [FC1] (no save of x pre-mlp-norm)
        gemm.matmul(
            &bufs.xn, &bufs.layer_weights[layer].wfc,
            &mut bufs.h, bt, MLP_DIM, d,
        );

        // [RELU2] (no save of h pre-act)
        unsafe {
            ffi::relu_sq_fwd(
                dptr(&bufs.h) as *const _,
                dptr(&bufs.h_act) as *mut _,
                (bt * MLP_DIM) as i32, stream,
            );
        }

        // [FC2]
        gemm.matmul(
            &bufs.h_act, &bufs.layer_weights[layer].wdn,
            &mut bufs.xn, bt, d, MLP_DIM,
        );

        // [LAYER STAT] Capture per-neuron mean absolute activation (for neuron rinsing)
        unsafe {
            ffi::neuron_act_norm_bf16(
                dptr(&bufs.h_act) as *const std::ffi::c_void,
                bt as i32,
                MLP_DIM as i32,
                dptr(&bufs.layer_neuron_act_norms) as *mut f32,
                layer as i32,
                stream,
            );
        }

        // [RESID2]
        unsafe {
            ffi::residual_add(
                dptr(&bufs.x) as *mut _,
                dptr(&bufs.xn) as *const _,
                (bt * d) as i32, stream,
            );
        }
    }

    // [NORMF] Final RMSNorm
    unsafe {
        ffi::fused_rms_norm_fwd(
            dptr(&bufs.x) as *const _,
            dptr(&bufs.xn) as *mut _,
            bt as u32, d as u32, EPS, stream,
        );
    }

    // [LMHEAD]
    gemm.matmul(&bufs.xn, &bufs.lm_head, &mut bufs.logits, bt, VOCAB, d);

    // [CELOSS] Per-token losses -> h_act; no loss accumulation needed for eval
    let per_token_loss_ptr = dptr(&bufs.h_act);
    unsafe {
        ffi::fused_cross_entropy_fwd(
            dptr(&bufs.logits) as *const _,
            dptr(&bufs.targets) as *const _,
            per_token_loss_ptr as *mut _,
            dptr(&bufs.loss) as *mut f32,
            bt as u32,
            VOCAB as u32,
            SOFTCAP, stream,
        );
    }
}

// ---------------------------------------------------------------------------
// Flash attention v3 forward (Hopper, prebuilt libflashattention3.a)
// ---------------------------------------------------------------------------

fn flash_attn_fwd(
    bufs: &mut BufferManager, b: usize, t: usize, layer: usize, lse_layer: usize,
    q_ptr: CUdeviceptr, k_ptr: CUdeviceptr, v_ptr: CUdeviceptr, o_ptr: CUdeviceptr,
) {
    let batch_stride = (t * N_HEAD * HEAD_DIM) as u32;
    let row_stride = (N_HEAD * HEAD_DIM) as u32;
    let head_stride = HEAD_DIM as u32;
    let softmax_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;

    let window = WINDOW_SIZES[layer];
    let mut window_size_left: i32 = if window >= t { -1 } else { window as i32 };
    let window_size_right: i32 = 0;

    // Always causal for language modeling. Local window controlled separately via is_local.
    let is_causal: i32 = 1;
    // Clamp window_size_left for local attention (FA3 handles this via is_local flag)
    if window_size_left < 0 && window_size_right >= 0 {
        window_size_left = t as i32;
    }

    unsafe {
        ffi::run_mha_v3(
            q_ptr as *mut _,
            k_ptr as *mut _,
            v_ptr as *mut _,
            o_ptr as *mut _,
            dptr(&bufs.saved_softmax_lse[lse_layer]) as *mut _,
            dptr(&bufs.fa3_scheduler_meta) as *mut _,
            batch_stride,               // q_batch_stride
            batch_stride,               // k_batch_stride
            batch_stride,               // v_batch_stride
            batch_stride,               // o_batch_stride
            row_stride,                 // q_row_stride
            row_stride,                 // k_row_stride
            row_stride,                 // v_row_stride
            row_stride,                 // o_row_stride
            head_stride,                // q_head_stride
            head_stride,                // k_head_stride
            head_stride,                // v_head_stride
            head_stride,                // o_head_stride
            b as u32,                   // b
            N_HEAD as u32,              // h
            N_KV_HEAD as u32,           // h_k
            HEAD_DIM as u32,            // d
            HEAD_DIM as u32,            // d_rounded
            softmax_scale,
            t as u32,                   // seqlen_q
            t as u32,                   // seqlen_k
            1,                          // is_bf16
            is_causal,
            window_size_left,
            window_size_right,
            0.0,                        // softcap (handled separately in CE loss)
            132,                        // num_sm (H100 SXM)
            stream,
        );
    }
}

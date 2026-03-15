use cudarc::cublas::safe::{Gemm, GemmConfig};
use cudarc::cublas::sys::{self, cublasOperation_t};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, ValidAsZeroBits};
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

/// Memset to zero using cudarc's stream method (handles pool-allocated buffers).
#[inline(always)]
pub fn raw_zero<T: cudarc::driver::DeviceRepr + ValidAsZeroBits>(
    stream: &std::sync::Arc<CudaStream>,
    dst: &mut CudaSlice<T>,
) {
    stream.memset_zeros(dst).unwrap();
}

/// Raw cublasGemmEx call with explicit device pointers (no CudaSlice needed).
/// Computes row-major: Y(m_rows, n_cols) = X(m_rows, k_cols) @ W(n_cols, k_cols)^T
/// Same convention as GemmRunner::matmul but with raw CUdeviceptr.
#[inline(always)]
pub unsafe fn raw_gemm_matmul(
    handle: sys::cublasHandle_t,
    x_ptr: CUdeviceptr, // (m, k) row-major
    w_ptr: CUdeviceptr, // (n, k) row-major
    y_ptr: CUdeviceptr, // (m, n) row-major
    m: usize, n: usize, k: usize,
    beta: f32,
) {
    let alpha: f32 = 1.0;
    unsafe {
        sys::cublasGemmEx(
            handle,
            cublasOperation_t::CUBLAS_OP_T,  // transa
            cublasOperation_t::CUBLAS_OP_N,  // transb
            n as i32,                         // m_cublas
            m as i32,                         // n_cublas
            k as i32,                         // k_cublas
            &alpha as *const f32 as *const _,
            w_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            k as i32,                         // lda
            x_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            k as i32,                         // ldb
            &beta as *const f32 as *const _,
            y_ptr as *mut _,
            sys::cudaDataType_t::CUDA_R_16BF,
            n as i32,                         // ldc
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        );
    }
}

/// Raw cublasGemmEx: dW(n,k) += dY(m,n)^T @ X(m,k)
/// Same convention as GemmRunner::matmul_acc but with raw CUdeviceptr.
#[inline(always)]
pub unsafe fn raw_gemm_matmul_acc(
    handle: sys::cublasHandle_t,
    dy_ptr: CUdeviceptr, // (m, n) row-major
    x_ptr: CUdeviceptr,  // (m, k) row-major
    dw_ptr: CUdeviceptr, // (n, k) row-major, accumulated
    m: usize, n: usize, k: usize,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;
    unsafe {
        sys::cublasGemmEx(
            handle,
            cublasOperation_t::CUBLAS_OP_N,  // transa
            cublasOperation_t::CUBLAS_OP_T,  // transb
            k as i32,                         // m_cublas
            n as i32,                         // n_cublas
            m as i32,                         // k_cublas
            &alpha as *const f32 as *const _,
            x_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            k as i32,                         // lda
            dy_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            n as i32,                         // ldb
            &beta as *const f32 as *const _,
            dw_ptr as *mut _,
            sys::cudaDataType_t::CUDA_R_16BF,
            k as i32,                         // ldc
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        );
    }
}

/// Raw cublasGemmEx: dX(m,k) = dY(m,n) @ W(n,k)
/// Same convention as GemmRunner::matmul_bwd_x but with raw CUdeviceptr.
#[inline(always)]
pub unsafe fn raw_gemm_bwd_x(
    handle: sys::cublasHandle_t,
    dy_ptr: CUdeviceptr, // (m, n) row-major
    w_ptr: CUdeviceptr,  // (n, k) row-major
    dx_ptr: CUdeviceptr, // (m, k) row-major
    m: usize, n: usize, k: usize,
    beta: f32,
) {
    let alpha: f32 = 1.0;
    unsafe {
        sys::cublasGemmEx(
            handle,
            cublasOperation_t::CUBLAS_OP_N,  // transa
            cublasOperation_t::CUBLAS_OP_N,  // transb
            k as i32,                         // m_cublas
            m as i32,                         // n_cublas
            n as i32,                         // k_cublas
            &alpha as *const f32 as *const _,
            w_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            k as i32,                         // lda
            dy_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            n as i32,                         // ldb
            &beta as *const f32 as *const _,
            dx_ptr as *mut _,
            sys::cudaDataType_t::CUDA_R_16BF,
            k as i32,                         // ldc
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        );
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
    let mut moe_idx = 0usize;
    for layer in 0..N_LAYER {
        forward_single_layer(bufs, gemm, layer, &mut moe_idx);
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

    // [LMHEAD + CELOSS] Chunked: process CE_CHUNK tokens at a time.
    // This avoids materializing the full [BT, VOCAB] logits tensor.
    // logits buffer is only CE_CHUNK * VOCAB elements.
    {
        let handle = *gemm.blas().handle();
        let per_token_loss_ptr = dptr(&bufs.h_act);
        let logits_ptr = dptr(&bufs.logits);
        let lm_head_ptr = dptr(&bufs.lm_head);
        let xn_base = dptr(&bufs.xn);
        let targets_base = dptr(&bufs.targets);

        let mut offset = 0usize;
        while offset < bt {
            let cur = CE_CHUNK.min(bt - offset);

            // LM head GEMM: logits[cur, VOCAB] = xn[offset..][cur, d] @ lm_head[VOCAB, d]^T
            unsafe {
                raw_gemm_matmul(
                    handle,
                    xn_base + (offset * d * std::mem::size_of::<bf16>()) as u64,
                    lm_head_ptr,
                    logits_ptr,
                    cur, VOCAB, d, 0.0,
                );
            }

            // CE loss: logits[cur, VOCAB], targets[offset..] -> per-token losses, loss accumulation
            unsafe {
                ffi::fused_cross_entropy_fwd(
                    logits_ptr as *const _,
                    (targets_base + (offset * std::mem::size_of::<i32>()) as u64) as *const _,
                    (per_token_loss_ptr + (offset * std::mem::size_of::<f32>()) as u64) as *mut _,
                    dptr(&bufs.loss) as *mut f32,
                    cur as u32,
                    VOCAB as u32,
                    SOFTCAP, stream,
                );
            }

            offset += cur;
        }
    }
}

/// Pipeline-parallel forward: run layers [bufs.layer_start, bufs.layer_end).
/// If bufs.has_embedding, runs embedding + norm0 + copy x0.
/// If bufs.has_head, runs final norm + lm_head + CE loss.
/// Activation tensors `x` and `x0` are expected to be populated by P2P transfer
/// from the previous stage (or by the embedding step if first stage).
pub fn forward_staged(bufs: &mut BufferManager, gemm: &GemmRunner) {
    // Set runtime device FIRST, then bind driver context
    unsafe { crate::ffi::cuda_set_device(bufs.stream.context().cu_device()); }
    bufs.stream.context().bind_to_thread().expect("bind CUDA context");

    let b = bufs.batch_size;
    let t = SEQ;
    let bt = b * t;
    let d = D_MODEL;
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;
    let ve_blas = gemm.blas();

    // PRE-LOOP: embedding (first stage only)
    if bufs.has_embedding {
        unsafe {
            ffi::embedding_fwd(
                dptr(&bufs.input_ids) as *const _,
                dptr(&bufs.wte) as *const _,
                dptr(&bufs.emb) as *mut _,
                bt as i32, d as i32, stream,
            );
            ffi::fused_rms_norm_fwd(
                dptr(&bufs.emb) as *const _,
                dptr(&bufs.x) as *mut _,
                bt as u32, d as u32, EPS, stream,
            );
        }
        raw_dtod(&bufs.stream, &bufs.x, &mut bufs.x0);
    }

    // PER-LAYER LOOP: only this stage's layers
    // Delegate to the same layer body as forward() — the layer loop code is
    // identical, just bounded to [layer_start, layer_end).
    forward_layer_range(bufs, gemm, bufs.layer_start, bufs.layer_end);

    // POST-LOOP: final norm + head (last stage only)
    if bufs.has_head {
        unsafe {
            ffi::fused_rms_norm_fwd(
                dptr(&bufs.x) as *const _,
                dptr(&bufs.xn) as *mut _,
                bt as u32, d as u32, EPS, stream,
            );
        }

        let handle = *gemm.blas().handle();
        let per_token_loss_ptr = dptr(&bufs.h_act);
        let logits_ptr = dptr(&bufs.logits);
        let lm_head_ptr = dptr(&bufs.lm_head);
        let xn_base = dptr(&bufs.xn);
        let targets_base = dptr(&bufs.targets);

        let mut offset = 0usize;
        while offset < bt {
            let cur = CE_CHUNK.min(bt - offset);
            unsafe {
                raw_gemm_matmul(
                    handle,
                    xn_base + (offset * d * std::mem::size_of::<bf16>()) as u64,
                    lm_head_ptr,
                    logits_ptr,
                    cur, VOCAB, d, 0.0,
                );
                ffi::fused_cross_entropy_fwd(
                    logits_ptr as *const _,
                    (targets_base + (offset * std::mem::size_of::<i32>()) as u64) as *const _,
                    (per_token_loss_ptr + (offset * std::mem::size_of::<f32>()) as u64) as *mut _,
                    dptr(&bufs.loss) as *mut f32,
                    cur as u32, VOCAB as u32, SOFTCAP, stream,
                );
            }
            offset += cur;
        }
    }
}

/// Run a single transformer layer's forward pass (attention + MLP/MoE + residuals).
/// Called from both forward() and forward_layer_range().
/// `moe_idx` tracks which MoE weight set to use; incremented when this layer is MoE.
fn forward_single_layer(bufs: &mut BufferManager, gemm: &GemmRunner, layer: usize, moe_idx: &mut usize) {
    let b = bufs.batch_size;
    let t = SEQ;
    let bt = b * t;
    let d = D_MODEL;
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;
    let ve_blas = gemm.blas();

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
    // Write to scratch q/k buffers (backward recomputes q/k from saved qkv via saved_xn).
    unsafe {
        ffi::fused_rope_rms_norm_fwd(
            dptr(&bufs.qkv) as *const _,               // q at offset 0 in qkv
            dptr(&bufs.cos) as *const _,
            dptr(&bufs.sin) as *const _,
            dptr(&bufs.q) as *mut _,                    // scratch buffer
            (bt * N_HEAD) as u32,
            t as u32, N_HEAD as u32, HEAD_DIM as u32, EPS, stream,
        );
        ffi::fused_rope_rms_norm_fwd(
            dptr_at(&bufs.qkv, bt * d) as *const _,    // k at offset BT*D in qkv
            dptr(&bufs.cos) as *const _,
            dptr(&bufs.sin) as *const _,
            dptr(&bufs.k) as *mut _,                    // scratch buffer
            (bt * N_KV_HEAD) as u32,
            t as u32, N_KV_HEAD as u32, HEAD_DIM as u32, EPS, stream,
        );
    }

    // [FATTN] Flash attention forward — reads from q/k scratch + saved_v, writes to saved_attn_out
    flash_attn_fwd(bufs, b, t, layer, layer,
        dptr(&bufs.q),
        dptr(&bufs.k),
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
    // MLP (dense) or MoE (conditional per layer)
    // =============================================================

    // Save x (post-residual-add) pre mlp-norm for backward
    raw_dtod(&bufs.stream, &bufs.x, &mut bufs.saved_x_pre_mlp_norm[layer]);

    if is_moe_layer(layer) {
        // ---------------------------------------------------------
        // MoE forward path
        // ---------------------------------------------------------
        let handle = *gemm.blas().handle();
        let moe_w = &bufs.moe_weights[*moe_idx];
        let n_dispatch = (bt * TOP_K) as i32;

        // [ROUTER] xn[BT, D] @ w_router[N_EXPERTS, D]^T → router_logits[BT, N_EXPERTS]
        unsafe {
            raw_gemm_matmul(
                handle,
                dptr(&bufs.xn),
                dptr(&moe_w.w_router),
                dptr(&bufs.moe_router_logits),
                bt, N_EXPERTS, d, 0.0,
            );
        }

        // [TOPK] Fused softmax + top-2 → probs[BT,8] (f32), gates[BT,2] (f32),
        // indices[BT,2] (i32). N_EXPERTS and TOP_K are compile-time constants in the kernel.
        unsafe {
            ffi::launch_moe_router_softmax_topk(
                dptr(&bufs.moe_router_logits) as *const _,
                dptr(&bufs.moe_router_probs) as *mut _,
                dptr(&bufs.moe_gate_values) as *mut _,
                dptr(&bufs.moe_expert_indices) as *mut _,
                bt as i32, stream,
            );
        }

        // Zero expert_counts and expert_offsets from Rust (cudarc pool-aware)
        // before the permute kernel, since in-kernel zeroing fails with pool alloc
        bufs.stream.memset_zeros(&mut bufs.moe_expert_counts).unwrap();
        bufs.stream.memset_zeros(&mut bufs.moe_expert_offsets).unwrap();

        // [PERMUTE] Build dispatch tables
        unsafe {
            ffi::launch_moe_permute_tokens(
                dptr(&bufs.moe_expert_indices) as *const _,
                dptr(&bufs.moe_token_perm) as *mut _,
                dptr(&bufs.moe_expert_counts) as *mut _,
                dptr(&bufs.moe_expert_offsets) as *mut _,
                dptr(&bufs.moe_expert_counts) as *mut _,
                bt as i32, stream,
            );
        }

        // [AUX] Load-balance loss: L_aux = coeff * N * sum(f_i * P_i), atomicAdd to loss.
        // Reads expert_counts produced by permute and router_probs from softmax.
        unsafe {
            ffi::load_balance_loss_fwd(
                dptr(&bufs.moe_router_probs) as *const _,
                dptr(&bufs.moe_expert_counts) as *const _,
                dptr(&bufs.loss) as *mut f32,
                bt as i32, N_EXPERTS as i32,
                AUX_LOSS_COEFF, stream,
            );
        }

        // [SAVE] Save routing state for backward (per-layer async copies)
        {
            let stream_ptr = bufs.stream.cu_stream();
            let mi = *moe_idx;
            unsafe {
                let nbytes = bufs.moe_router_probs.len() * std::mem::size_of::<f32>();
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.saved_router_probs[mi]), dptr(&bufs.moe_router_probs), nbytes, stream_ptr);
                let nbytes = bufs.moe_gate_values.len() * std::mem::size_of::<f32>();
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.saved_gate_values[mi]), dptr(&bufs.moe_gate_values), nbytes, stream_ptr);
                let nbytes = bufs.moe_expert_indices.len() * std::mem::size_of::<i32>();
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.saved_expert_indices[mi]), dptr(&bufs.moe_expert_indices), nbytes, stream_ptr);
                let nbytes = bufs.moe_token_perm.len() * std::mem::size_of::<i32>();
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.saved_token_perm[mi]), dptr(&bufs.moe_token_perm), nbytes, stream_ptr);
                let nbytes = bufs.moe_expert_offsets.len() * std::mem::size_of::<i32>();
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dptr(&bufs.saved_expert_offsets[mi]), dptr(&bufs.moe_expert_offsets), nbytes, stream_ptr);
            }
        }

        // [DtoH] expert_offsets to host via C-side (same CUDA context as kernels)
        unsafe {
            ffi::moe_copy_offsets_to_host(
                dptr(&bufs.moe_expert_offsets) as *const _,
                bufs.moe_expert_offsets_host.as_mut_ptr() as *mut _,
                stream,
            );
        }

        // [GATHER] Permute xn tokens into contiguous per-expert blocks.
        // Reuses h buffer: BT*TOP_K*D = BT*2048 bf16 elements fits in h's BT*4096 capacity.
        unsafe {
            ffi::launch_moe_gather_tokens(
                dptr(&bufs.xn) as *const _,
                dptr(&bufs.moe_token_perm) as *const _,
                dptr(&bufs.h) as *mut _,
                n_dispatch, stream,
            );
        }

        // [INV_PERM] Build inverse permutation for scatter.
        // Reuses moe_router_logits as raw i32 storage (BT*N_EXPERTS*2 bytes >= BT*TOP_K*4 bytes).
        // Router logits are no longer needed after softmax_topk.
        let inv_perm_ptr = dptr(&bufs.moe_router_logits);
        unsafe {
            ffi::launch_moe_build_inv_perm(
                dptr(&bufs.moe_token_perm) as *const _,
                inv_perm_ptr as *mut _,
                n_dispatch, stream,
            );
        }

        // [EXPERT LOOP] Process each expert sequentially with cuBLAS.
        // Buffer reuse:
        //   h[0..n_dispatch*D]  — gathered tokens (input), overwritten with expert output
        //   h_act[0..n_e*MLP_DIM_E]             — FC1 scratch (per expert, reused)
        //   h_act[n_e*MLP_DIM_E..2*n_e*MLP_DIM_E] — ReLU² scratch (per expert, reused)
        let gathered_base = dptr(&bufs.h);
        let fc1_scratch = dptr(&bufs.h_act);
        let offsets = &bufs.moe_expert_offsets_host;
        let wfc_base = dptr(&moe_w.expert_wfc);
        let wdn_base = dptr(&moe_w.expert_wdn);

        for e in 0..N_EXPERTS {
            let off = offsets[e] as usize;
            let n_e = (offsets[e + 1] - offsets[e]) as usize;
            if n_e == 0 { continue; }

            let gathered_e = gathered_base + (off * d * std::mem::size_of::<bf16>()) as u64;
            let wfc_e = wfc_base + (e * MLP_DIM_E * d * std::mem::size_of::<bf16>()) as u64;
            let wdn_e = wdn_base + (e * d * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;
            // FC1 and ReLU² use non-overlapping regions at the start of h_act
            let h_e = fc1_scratch;
            let h_act_e = fc1_scratch + (n_e * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;

            assert!(
                2 * n_e * MLP_DIM_E <= bufs.h_act.len(),
                "MoE expert {e}: h_act buffer overflow — need {} bf16 elems \
                 (2 * {n_e} * {MLP_DIM_E}) but h_act has {}",
                2 * n_e * MLP_DIM_E, bufs.h_act.len(),
            );

            // FC1: gathered_e[n_e, D] @ wfc_e[MLP_DIM_E, D]^T → h_e[n_e, MLP_DIM_E]
            unsafe {
                raw_gemm_matmul(handle, gathered_e, wfc_e, h_e, n_e, MLP_DIM_E, d, 0.0);
            }

            // ReLU²: h_e → h_act_e
            unsafe {
                ffi::relu_sq_fwd(
                    h_e as *const _, h_act_e as *mut _,
                    (n_e * MLP_DIM_E) as i32, stream,
                );
            }

            // FC2: h_act_e[n_e, MLP_DIM_E] @ wdn_e[D, MLP_DIM_E]^T → gathered_e[n_e, D]
            // Overwrites gathered input (already consumed by FC1)
            unsafe {
                raw_gemm_matmul(handle, h_act_e, wdn_e, gathered_e, n_e, d, MLP_DIM_E, 0.0);
            }
        }

        // [SCATTER] Gate-weighted scatter: xn[BT, D] = sum_k gate_k * expert_out[perm_k]
        // Uses inv_perm to map dispatch slots back to original token positions.
        // Output goes to xn (scratch), same buffer as dense MLP path.
        unsafe {
            ffi::launch_moe_scatter(
                dptr(&bufs.h) as *const _,
                dptr(&bufs.moe_gate_values) as *const _,
                inv_perm_ptr as *const _,
                dptr(&bufs.xn) as *mut _,
                bt as i32, stream,
            );
        }

        // [RESID2] x += moe_output
        unsafe {
            ffi::residual_add(
                dptr(&bufs.x) as *mut _,
                dptr(&bufs.xn) as *const _,
                (bt * d) as i32, stream,
            );
        }

        *moe_idx += 1;
    } else {
        // ---------------------------------------------------------
        // Dense MLP forward path (unchanged)
        // ---------------------------------------------------------

        // [FC1] xn[bt,d] @ Wfc[MLP_DIM,d]^T -> h (scratch; backward recomputes h_pre_act)
        gemm.matmul(
            &bufs.xn, bufs.layer_weights[layer].wfc.as_ref().unwrap(),
            &mut bufs.h, bt, MLP_DIM, d,
        );

        // [RELU2] ReLU^2: h -> h_act
        unsafe {
            ffi::relu_sq_fwd(
                dptr(&bufs.h) as *const _,
                dptr(&bufs.h_act) as *mut _,
                (bt * MLP_DIM) as i32, stream,
            );
        }

        // [FC2] h_act[bt,MLP_DIM] @ Wdn[d,MLP_DIM]^T -> xn[bt,d]
        gemm.matmul(
            &bufs.h_act, bufs.layer_weights[layer].wdn.as_ref().unwrap(),
            &mut bufs.xn, bt, d, MLP_DIM,
        );

        // [RESID2] x += mlp_result
        unsafe {
            ffi::residual_add(
                dptr(&bufs.x) as *mut _,
                dptr(&bufs.xn) as *const _,
                (bt * d) as i32, stream,
            );
        }
    }
}

/// Run the per-layer forward pass for layers [start, end).
/// Extracted so both forward() and forward_staged() can share the layer body.
fn forward_layer_range(bufs: &mut BufferManager, gemm: &GemmRunner, start: usize, end: usize) {
    let mut moe_idx = 0usize; // moe_weights only has entries for this stage's owned layers
    for layer in start..end {
        forward_single_layer(bufs, gemm, layer, &mut moe_idx);
    }
}

/// Eval-only forward pass: identical to `forward()` but skips all
/// activation saves (saved_x_pre_attn_norm, saved_xn, saved_v,
/// saved_attn_out, saved_x_pre_mlp_norm).  Softmax LSE
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

        if is_moe_layer(layer) {
            // ---------------------------------------------------------
            // MoE forward path (eval — no saves)
            // ---------------------------------------------------------
            let handle = *gemm.blas().handle();
            let moe_w = &bufs.moe_weights[layer];
            let n_dispatch = (bt * TOP_K) as i32;

            // [ROUTER] xn[BT, D] @ w_router[N_EXPERTS, D]^T → router_logits[BT, N_EXPERTS]
            unsafe {
                raw_gemm_matmul(
                    handle,
                    dptr(&bufs.xn),
                    dptr(&moe_w.w_router),
                    dptr(&bufs.moe_router_logits),
                    bt, N_EXPERTS, d, 0.0,
                );
            }

            // [TOPK] Softmax + top-2 → probs, gates, indices
            unsafe {
                ffi::launch_moe_router_softmax_topk(
                    dptr(&bufs.moe_router_logits) as *const _,
                    dptr(&bufs.moe_router_probs) as *mut _,
                    dptr(&bufs.moe_gate_values) as *mut _,
                    dptr(&bufs.moe_expert_indices) as *mut _,
                    bt as i32, stream,
                );
            }

            // [PERMUTE] Build dispatch tables
            unsafe {
                ffi::launch_moe_permute_tokens(
                    dptr(&bufs.moe_expert_indices) as *const _,
                    dptr(&bufs.moe_token_perm) as *mut _,
                    dptr(&bufs.moe_expert_counts) as *mut _,
                    dptr(&bufs.moe_expert_offsets) as *mut _,
                    dptr(&bufs.moe_expert_counts) as *mut _,
                    bt as i32, stream,
                );
            }

            // [AUX] Load-balance loss (still computed during eval for monitoring)
            unsafe {
                ffi::load_balance_loss_fwd(
                    dptr(&bufs.moe_router_probs) as *const _,
                    dptr(&bufs.moe_expert_counts) as *const _,
                    dptr(&bufs.loss) as *mut f32,
                    bt as i32, N_EXPERTS as i32,
                    AUX_LOSS_COEFF, stream,
                );
            }

            // [DtoH] expert_offsets to host via C-side (same CUDA context as kernels)
            unsafe {
                ffi::moe_copy_offsets_to_host(
                    dptr(&bufs.moe_expert_offsets) as *const _,
                    bufs.moe_expert_offsets_host.as_mut_ptr() as *mut _,
                    stream,
                );
            }

            // [GATHER] Permute xn tokens into per-expert blocks in h
            unsafe {
                ffi::launch_moe_gather_tokens(
                    dptr(&bufs.xn) as *const _,
                    dptr(&bufs.moe_token_perm) as *const _,
                    dptr(&bufs.h) as *mut _,
                    n_dispatch, stream,
                );
            }

            // [INV_PERM] Build inverse permutation (reuse moe_router_logits as raw storage)
            let inv_perm_ptr = dptr(&bufs.moe_router_logits);
            unsafe {
                ffi::launch_moe_build_inv_perm(
                    dptr(&bufs.moe_token_perm) as *const _,
                    inv_perm_ptr as *mut _,
                    n_dispatch, stream,
                );
            }

            // [EXPERT LOOP]
            let gathered_base = dptr(&bufs.h);
            let fc1_scratch = dptr(&bufs.h_act);
            let offsets = &bufs.moe_expert_offsets_host;
            let wfc_base = dptr(&moe_w.expert_wfc);
            let wdn_base = dptr(&moe_w.expert_wdn);

            for e in 0..N_EXPERTS {
                let off = offsets[e] as usize;
                let n_e = (offsets[e + 1] - offsets[e]) as usize;
                if n_e == 0 { continue; }

                let gathered_e = gathered_base + (off * d * std::mem::size_of::<bf16>()) as u64;
                let wfc_e = wfc_base + (e * MLP_DIM_E * d * std::mem::size_of::<bf16>()) as u64;
                let wdn_e = wdn_base + (e * d * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;
                let h_e = fc1_scratch;
                let h_act_e = fc1_scratch + (n_e * MLP_DIM_E * std::mem::size_of::<bf16>()) as u64;

                unsafe {
                    raw_gemm_matmul(handle, gathered_e, wfc_e, h_e, n_e, MLP_DIM_E, d, 0.0);
                    ffi::relu_sq_fwd(h_e as *const _, h_act_e as *mut _, (n_e * MLP_DIM_E) as i32, stream);
                    raw_gemm_matmul(handle, h_act_e, wdn_e, gathered_e, n_e, d, MLP_DIM_E, 0.0);
                }
            }

            // [SCATTER] Gate-weighted scatter → xn
            unsafe {
                ffi::launch_moe_scatter(
                    dptr(&bufs.h) as *const _,
                    dptr(&bufs.moe_gate_values) as *const _,
                    inv_perm_ptr as *const _,
                    dptr(&bufs.xn) as *mut _,
                    bt as i32, stream,
                );
            }

            // [RESID2] x += moe_output
            unsafe {
                ffi::residual_add(
                    dptr(&bufs.x) as *mut _,
                    dptr(&bufs.xn) as *const _,
                    (bt * d) as i32, stream,
                );
            }
        } else {
            // ---------------------------------------------------------
            // Dense MLP forward path (eval, unchanged)
            // ---------------------------------------------------------

            // [FC1] (no save of x pre-mlp-norm)
            gemm.matmul(
                &bufs.xn, bufs.layer_weights[layer].wfc.as_ref().unwrap(),
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
                &bufs.h_act, bufs.layer_weights[layer].wdn.as_ref().unwrap(),
                &mut bufs.xn, bt, d, MLP_DIM,
            );

            // [RESID2]
            unsafe {
                ffi::residual_add(
                    dptr(&bufs.x) as *mut _,
                    dptr(&bufs.xn) as *const _,
                    (bt * d) as i32, stream,
                );
            }
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

    // [LMHEAD + CELOSS] Chunked: process CE_CHUNK tokens at a time.
    {
        let handle = *gemm.blas().handle();
        let per_token_loss_ptr = dptr(&bufs.h_act);
        let logits_ptr = dptr(&bufs.logits);
        let lm_head_ptr = dptr(&bufs.lm_head);
        let xn_base = dptr(&bufs.xn);
        let targets_base = dptr(&bufs.targets);

        let mut offset = 0usize;
        while offset < bt {
            let cur = CE_CHUNK.min(bt - offset);

            unsafe {
                raw_gemm_matmul(
                    handle,
                    xn_base + (offset * d * std::mem::size_of::<bf16>()) as u64,
                    lm_head_ptr,
                    logits_ptr,
                    cur, VOCAB, d, 0.0,
                );
            }

            unsafe {
                ffi::fused_cross_entropy_fwd(
                    logits_ptr as *const _,
                    (targets_base + (offset * std::mem::size_of::<i32>()) as u64) as *const _,
                    (per_token_loss_ptr + (offset * std::mem::size_of::<f32>()) as u64) as *mut _,
                    dptr(&bufs.loss) as *mut f32,
                    cur as u32,
                    VOCAB as u32,
                    SOFTCAP, stream,
                );
            }

            offset += cur;
        }
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
    let is_local = window < t;
    let window_size_left: i32 = if is_local { window as i32 } else { -1 };
    let window_size_right: i32 = if is_local { 0 } else { -1 };
    let is_causal: i32 = 1;

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

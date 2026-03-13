use std::ffi::c_void;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, DevicePtr};
use half::bf16;

use crate::buffer::BufferManager;
use crate::config::*;
use crate::ffi;
use crate::gemm::GemmRunner;

// ── Polar Express coefficients (matching Python exactly) ───────────────────
const NS_COEFFS: [(f32, f32, f32); 5] = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
];
const NS_ITERS: usize = 5;

// ── LR schedule defaults (ported from exp/H100/mar8 winning recipe) ─────────
const DEFAULT_WARMUP_RATIO: f64 = 0.0;
const DEFAULT_WARMDOWN_RATIO: f64 = 0.75;
const DEFAULT_FINAL_LR_FRAC: f64 = 0.05;

// Base learning rates (defaults)
const DEFAULT_EMBEDDING_LR: f64 = 0.9;
const DEFAULT_UNEMBEDDING_LR: f64 = 0.005;
const DEFAULT_MATRIX_LR: f64 = 0.04;
const DEFAULT_SCALAR_LR: f64 = 0.5;
const DEFAULT_WEIGHT_DECAY: f64 = 0.2;

// Weight decay for embedding/output params (AdamW, separate from Muon WD)
const WTE_WEIGHT_DECAY: f32 = 0.001;
const LM_HEAD_WEIGHT_DECAY: f32 = 0.01;
const VE_WEIGHT_DECAY: f32 = 0.003;

// AdamW hyperparams
const ADAM_BETA1: f32 = 0.8;
const ADAM_BETA2: f32 = 0.95;
const ADAM_EPS: f32 = 1e-10;

// ── Schedule type ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Schedule {
    Linear,
    Cosine,
}

impl Schedule {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cosine" | "cos" => Schedule::Cosine,
            _ => Schedule::Linear,
        }
    }
}

impl std::fmt::Display for Schedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Schedule::Linear => write!(f, "linear"),
            Schedule::Cosine => write!(f, "cosine"),
        }
    }
}

// ── Runtime schedule config (parsed from env vars) ─────────────────────────

#[derive(Debug, Clone)]
pub struct ScheduleConfig {
    pub peak_lr: f64,         // MATRIX_LR for Muon
    pub warmdown_ratio: f64,
    pub weight_decay: f64,
    pub schedule: Schedule,
    pub final_lr_frac: f64,   // fraction of peak LR at end of training
    // Derived LRs scaled relative to peak_lr / DEFAULT_MATRIX_LR
    pub embedding_lr: f64,
    pub unembedding_lr: f64,
    pub scalar_lr: f64,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            peak_lr: DEFAULT_MATRIX_LR,
            warmdown_ratio: DEFAULT_WARMDOWN_RATIO,
            weight_decay: DEFAULT_WEIGHT_DECAY,
            schedule: Schedule::Linear,
            final_lr_frac: DEFAULT_FINAL_LR_FRAC,
            embedding_lr: DEFAULT_EMBEDDING_LR,
            unembedding_lr: DEFAULT_UNEMBEDDING_LR,
            scalar_lr: DEFAULT_SCALAR_LR,
        }
    }
}

// ── FFI: Muon elementwise kernels used directly (without ffi:: prefix) ──────
// NOTE: these duplicate ffi.rs declarations but with a local CudaStream type alias.
type CudaStreamPtr = *mut c_void;
unsafe extern "C" {
    fn frob_normalize_bf16(x: *const c_void, out: *mut c_void, n: i32, scratch: *mut f32, stream: CudaStreamPtr);
    fn scale_bf16(x: *mut c_void, scale: f32, n: i32, stream: CudaStreamPtr);
    fn normuon_step_bf16(
        g: *mut c_void, second_mom: *mut f32,
        m: i32, n: i32, reduce_cols: i32, beta2: f32, scratch: *mut f32,
        stream: CudaStreamPtr,
    );
}

// ── Raw pointer helpers ────────────────────────────────────────────────────
//
// Extract raw CUdeviceptr directly from CudaSlice layout to avoid cudarc's
// event-tracking overhead and borrow-checker conflicts. Same pattern as forward.rs.
//
#[inline(always)]
fn dptr<T>(buf: &CudaSlice<T>) -> CUdeviceptr {
    let (ptr, _sync) = buf.device_ptr(buf.stream());
    ptr
}

#[inline(always)]
fn vptr<T>(buf: &CudaSlice<T>) -> *const c_void {
    dptr(buf) as *const c_void
}

#[inline(always)]
fn vptr_mut<T>(buf: &CudaSlice<T>) -> *mut c_void {
    dptr(buf) as *mut c_void
}

// ── LR schedule ────────────────────────────────────────────────────────────

fn dmodel_lr_scale() -> f32 {
    (D_MODEL as f64 / 768.0).powf(-0.5) as f32
}

fn get_lr_multiplier(progress: f64, cfg: &ScheduleConfig) -> f64 {
    let warmup = DEFAULT_WARMUP_RATIO;
    let warmdown = cfg.warmdown_ratio;
    let final_frac = cfg.final_lr_frac;

    if progress < warmup {
        if warmup > 0.0 {
            progress / warmup
        } else {
            1.0
        }
    } else if warmdown <= 0.0 || progress < 1.0 - warmdown {
        1.0
    } else {
        let cooldown = (1.0 - progress) / warmdown;
        match cfg.schedule {
            Schedule::Linear => cooldown + (1.0 - cooldown) * final_frac,
            Schedule::Cosine => {
                // cosine decay: 1 -> final_frac as cooldown goes 1 -> 0
                let t = 1.0 - cooldown; // 0 at warmdown start, 1 at end
                let cosine_mult = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
                cosine_mult + (1.0 - cosine_mult) * final_frac
            }
        }
    }
}

/// Compute the LR multiplier from training progress [0, 1].
pub fn lr_multiplier(progress: f64, cfg: &ScheduleConfig) -> f32 {
    get_lr_multiplier(progress, cfg) as f32
}

fn muon_momentum(step: usize) -> f32 {
    let frac = (step as f64 / 200.0).min(1.0);
    ((1.0 - frac) * 0.85 + frac * 0.95) as f32
}

fn muon_weight_decay(progress: f64, cfg: &ScheduleConfig) -> f32 {
    (cfg.weight_decay * (1.0 - progress)) as f32
}

// ── Main optimizer step ────────────────────────────────────────────────────

/// Run one complete optimizer step on all parameters.
///
/// 1. AdamW for embeddings (wte, lm_head), scalar params (resid/x0 lambdas), VE weights
/// 2. Muon with Polar Express for block matrix weights (N_LAYER*6) + ve_gate (VE_LAYERS) total
///
/// Uses backward scratch buffers (d_x, d_x0, d_q, d_k, d_v) as temporary workspace
/// for the Muon/Polar Express matmuls -- these are unused during the optimizer step.
///
/// `step` is 1-indexed (first training step is step 1).
/// `progress` is training progress [0, 1] = min(total_training_time / time_budget, 1).
pub fn optimizer_step(
    bufs: &mut BufferManager,
    gemm: &GemmRunner,
    step: usize,
    progress: f64,
    cfg: &ScheduleConfig,
) {
    let lr_mult = lr_multiplier(progress, cfg);
    let dscale = dmodel_lr_scale();
    let momentum = muon_momentum(step);
    let wd = muon_weight_decay(progress, cfg);

    adamw_embeddings(bufs, step, lr_mult, dscale, cfg);
    adamw_scalars(bufs, step, lr_mult, cfg);
    adamw_ve_weights(bufs, step, lr_mult, dscale, cfg);
    muon_all_layers(bufs, gemm, lr_mult, momentum, wd, step, cfg);
    muon_ve_gates(bufs, gemm, lr_mult, momentum, wd, cfg);

    // Refresh packed wqkv from updated wq/wk/wv (Muon updates individual bf16 weights)
    pack_wqkv(bufs);
}

/// Compute Frobenius norm of a bf16 buffer on GPU (sync — debug only).
fn debug_bf16_norm(buf: *const c_void, n: i32, scratch: *mut f32) -> f32 {
    unsafe {
        cudarc::driver::sys::cuMemsetD32_v2(scratch as CUdeviceptr, 0, 1);
        frob_normalize_bf16(buf, buf as *mut c_void, 0, scratch, std::ptr::null_mut()); // abuse: N=0 won't launch
        // Actually just call the sum_sq kernel directly — but we can't from here easily.
        // Instead, use the scratch buffer approach: zero it, run frob, read back.
        // Simpler: copy to host and compute. But that's slow.
        // Let's use a different approach: call frob_normalize with a throwaway and read scratch.
    }
    // Use cuMemsetD32 + frob_sum_sq via frob_normalize trick:
    // Actually the simplest: just read the buffer to host.
    let mut host = vec![half::bf16::ZERO; n as usize];
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoH_v2(
            host.as_mut_ptr() as *mut std::ffi::c_void,
            buf as CUdeviceptr,
            n as usize * 2,
        );
    }
    let sum_sq: f64 = host.iter().map(|x| { let v = x.to_f32() as f64; v * v }).sum();
    sum_sq.sqrt() as f32
}

// ── AdamW: embedding params ────────────────────────────────────────────────

fn adamw_embeddings(bufs: &mut BufferManager, step: usize, lr_mult: f32, dscale: f32, cfg: &ScheduleConfig) {
    let bc1 = 1.0 - ADAM_BETA1.powi(step as i32);
    let bc2 = 1.0 - ADAM_BETA2.powi(step as i32);
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;

    // wte: lr = embedding_lr * dmodel_scale * lr_mult, wd=0.001
    {
        let lr = cfg.embedding_lr as f32 * dscale * lr_mult;
        let n = bufs.wte.len() as i32;
        unsafe {
            ffi::adamw_step_bf16(
                vptr_mut(&bufs.wte), vptr(&bufs.wte_grad),
                vptr_mut(&bufs.adamw.wte_exp_avg), vptr_mut(&bufs.adamw.wte_exp_avg_sq),
                lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, WTE_WEIGHT_DECAY, bc1, bc2, n, stream,
            );
        }
    }

    // lm_head: lr = unembedding_lr * dmodel_scale * lr_mult, wd=0.01
    // Python keeps lm_head in f32 — use f32 master + f32 moments, copy back to bf16
    {
        let lr = cfg.unembedding_lr as f32 * dscale * lr_mult;
        let n = bufs.lm_head.len() as i32;
        unsafe {
            // Cast bf16 grads to f32 scratch (reuse d_xn as scratch)
            let grad_f32 = dptr(&bufs.d_xn) as *mut f32;
            ffi::cast_bf16_to_f32(vptr(&bufs.lm_head_grad), grad_f32 as *mut c_void, n, stream);
            // Run f32 AdamW on f32 master weights
            ffi::adamw_step_f32(
                dptr(&bufs.lm_head_f32) as *mut f32,
                grad_f32 as *const f32,
                dptr(&bufs.adamw.lm_head_exp_avg) as *mut f32,
                dptr(&bufs.adamw.lm_head_exp_avg_sq) as *mut f32,
                lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, LM_HEAD_WEIGHT_DECAY, bc1, bc2, n, stream,
            );
            // Copy f32 master back to bf16 compute weight
            ffi::cast_f32_to_bf16(dptr(&bufs.lm_head_f32) as *const c_void, vptr_mut(&bufs.lm_head), n, stream);
        }
    }
}

// ── AdamW: scalar params (resid_lambdas, x0_lambdas) ──────────────────────
//
// Params stored BF16, grads and optimizer state in F32.
// Cast param BF16 -> F32 into scratch, run F32 AdamW, cast back.
// Scratch: first N_LAYER floats of d_xn (BF16 buf reinterpreted as F32 bytes).

fn adamw_scalars(bufs: &mut BufferManager, step: usize, lr_mult: f32, cfg: &ScheduleConfig) {
    let scratch_f32 = dptr(&bufs.d_xn) as *mut f32;
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;

    // resid_lambdas: lr = scalar_lr * 0.01 * lr_mult, betas=(0.8, 0.95), wd=0
    {
        let lr = cfg.scalar_lr as f32 * 0.01 * lr_mult;
        let n = bufs.resid_lambdas.len() as i32;
        let bc1 = 1.0 - ADAM_BETA1.powi(step as i32);
        let bc2 = 1.0 - ADAM_BETA2.powi(step as i32);

        unsafe {
            ffi::cast_bf16_to_f32(
                vptr(&bufs.resid_lambdas),
                scratch_f32 as *mut c_void,
                n, stream,
            );
            ffi::adamw_step_f32(
                scratch_f32,
                dptr(&bufs.resid_lambdas_grad) as *const f32,
                dptr(&bufs.adamw.resid_lambdas_exp_avg) as *mut f32,
                dptr(&bufs.adamw.resid_lambdas_exp_avg_sq) as *mut f32,
                lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, 0.0, bc1, bc2, n, stream,
            );
            ffi::cast_f32_to_bf16(
                scratch_f32 as *const c_void,
                vptr_mut(&bufs.resid_lambdas),
                n, stream,
            );
        }
    }

    // x0_lambdas: lr = scalar_lr * lr_mult, betas=(0.96, 0.95), wd=0
    {
        let lr = cfg.scalar_lr as f32 * lr_mult;
        let x0_beta1: f32 = 0.96;
        let x0_beta2: f32 = 0.95;
        let n = bufs.x0_lambdas.len() as i32;
        let bc1 = 1.0 - x0_beta1.powi(step as i32);
        let bc2 = 1.0 - x0_beta2.powi(step as i32);

        unsafe {
            ffi::cast_bf16_to_f32(
                vptr(&bufs.x0_lambdas),
                scratch_f32 as *mut c_void,
                n, stream,
            );
            ffi::adamw_step_f32(
                scratch_f32,
                dptr(&bufs.x0_lambdas_grad) as *const f32,
                dptr(&bufs.adamw.x0_lambdas_exp_avg) as *mut f32,
                dptr(&bufs.adamw.x0_lambdas_exp_avg_sq) as *mut f32,
                lr, x0_beta1, x0_beta2, ADAM_EPS, 0.0, bc1, bc2, n, stream,
            );
            ffi::cast_f32_to_bf16(
                scratch_f32 as *const c_void,
                vptr_mut(&bufs.x0_lambdas),
                n, stream,
            );
        }
    }
}

// ── AdamW: VE embedding weights ──────────────────────────────────────────

fn adamw_ve_weights(bufs: &mut BufferManager, step: usize, lr_mult: f32, dscale: f32, cfg: &ScheduleConfig) {
    let lr = cfg.embedding_lr as f32 * dscale * lr_mult;
    let bc1 = 1.0 - ADAM_BETA1.powi(step as i32);
    let bc2 = 1.0 - ADAM_BETA2.powi(step as i32);
    let stream = bufs.stream.cu_stream() as ffi::CudaStream;

    let mut ve_idx = 0usize;
    for layer in 0..N_LAYER {
        if !has_ve(layer) {
            continue;
        }

        if let (Some(w), Some(g)) = (
            bufs.layer_weights[layer].ve_weight.as_ref(),
            bufs.layer_grads[layer].ve_weight.as_ref(),
        ) {
            let n = w.len() as i32;
            unsafe {
                ffi::adamw_step_bf16(
                    vptr_mut(w), vptr(g),
                    vptr_mut(&bufs.adamw.ve_exp_avg[ve_idx]),
                    vptr_mut(&bufs.adamw.ve_exp_avg_sq[ve_idx]),
                    lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, VE_WEIGHT_DECAY, bc1, bc2, n, stream,
                );
            }
        }

        ve_idx += 1;
    }
}

// ── Muon: VE gate weights ───────────────────────────────────────────────
//
// Python includes ve_gate in `matrix_params` (transformer.h.parameters()),
// so ve_gate gets Muon treatment: Nesterov momentum, Polar Express (NS),
// NorMuon variance reduction, and cautious weight update.
//
// ve_gate shape: [N_KV_HEAD, VE_GATE_CH] — wide (m < n).
// Gram matrix: X @ X^T = [N_KV_HEAD, N_KV_HEAD]. NS iterations on small matrices.
// Aspect ratio: max(1, N_KV_HEAD/VE_GATE_CH)^0.5 = 1.0 (no scaling since m < n).
// LR: MATRIX_LR * lr_mult (same as block matrices).

fn muon_ve_gates(
    bufs: &mut BufferManager,
    gemm: &GemmRunner,
    lr_mult: f32,
    momentum: f32,
    wd: f32,
    cfg: &ScheduleConfig,
) {
    let m = N_KV_HEAD;
    let n = VE_GATE_CH;
    let mn = (m * n) as i32;
    let d = m; // min(m, n) = N_KV_HEAD (wide matrix)
    let dd = (d * d) as i32;

    // LR: no aspect scaling since m < n
    let aspect_lr = cfg.peak_lr as f32 * lr_mult;

    // Scratch buffers: reuse d_x, d_x0, d_q, d_k, d_v (all large enough for N_KV_HEAD*VE_GATE_CH elems)
    let x_a = vptr_mut(&bufs.d_x);
    let x_b = vptr_mut(&bufs.d_x0);
    let gram = vptr_mut(&bufs.d_q);
    let sq = vptr_mut(&bufs.d_k);
    let comb = vptr_mut(&bufs.d_v);

    let stream = bufs.stream.cu_stream() as ffi::CudaStream;
    let mut ve_idx = 0usize;
    for layer in 0..N_LAYER {
        if !has_ve(layer) {
            continue;
        }

        let (param, grad) = match (
            bufs.layer_weights[layer].ve_gate.as_ref(),
            bufs.layer_grads[layer].ve_gate.as_ref(),
        ) {
            (Some(w), Some(g)) => (vptr_mut(w), vptr(g)),
            _ => { ve_idx += 1; continue; }
        };

        let mom = dptr(&bufs.muon.ve_gate_momentum[ve_idx]) as *mut f32;
        let master = bufs.layer_weights[layer].ve_gate_f32.as_ref()
            .map(|b| dptr(b) as *mut f32).unwrap();

        // ── 1. Nesterov momentum (f32 buf, bf16 grad/out) ──
        unsafe {
            ffi::muon_nesterov_f32buf(mom, grad as *const c_void, x_a, momentum, mn, stream);
        }

        // ── 2. Polar Express orthogonalization ──
        let scratch = dptr(&bufs.muon.scratch) as *mut f32;
        unsafe {
            frob_normalize_bf16(x_a as *const c_void, x_b, mn, scratch, stream);
        }

        let mut src_is_b = true;
        for &(ca, cb, cc) in NS_COEFFS[..NS_ITERS].iter() {
            let src = if src_is_b { x_b } else { x_a };
            let dst = if src_is_b { x_a } else { x_b };

            unsafe {
                // Wide matrix (m < n): A(m,m) = X(m,n) @ X^T(n,m)
                // matmul(x, w, y, M, N, K): y(M,N) = x(M,K) @ w(N,K)^T
                // A(m,m) = X(m,n) @ X(m,n)^T => M=m, N=m, K=n
                if src_is_b {
                    gemm.matmul(&bufs.d_x0, &bufs.d_x0, &mut bufs.d_q, m, m, n);
                } else {
                    gemm.matmul(&bufs.d_x, &bufs.d_x, &mut bufs.d_q, m, m, n);
                }

                // A^2(d,d) = A(d,d) @ A(d,d)
                gemm.matmul_bwd_x(&bufs.d_q, &bufs.d_q, &mut bufs.d_k, d, d, d);

                // Fused: combined = ca*I + cb*A + cc*A^2
                ffi::ns_combined_batched(
                    gram as *const c_void, sq as *const c_void, comb,
                    ca, cb, cc, d as i32, 1, stream,
                );

                // Wide: dst(m,n) = combined(m,m) @ src(m,n)
                if src_is_b {
                    gemm.matmul_bwd_x(&bufs.d_v, &bufs.d_x0, &mut bufs.d_x, m, m, n);
                } else {
                    gemm.matmul_bwd_x(&bufs.d_v, &bufs.d_x, &mut bufs.d_x0, m, m, n);
                }
            }

            src_is_b = !src_is_b;
            let _ = (src, dst);
        }

        // After 5 iterations (odd), result is in x_a (d_x)
        let result = if src_is_b { x_b } else { x_a };

        // ── 2.5 NorMuon variance reduction ──
        // Wide (m < n): reduce_cols=0
        let second_mom = dptr(&bufs.muon.ve_gate_second_momentum[ve_idx]) as *mut f32;
        unsafe {
            normuon_step_bf16(result, second_mom, m as i32, n as i32, 0, 0.95, scratch, stream);
        }

        // ── 3. Weight update (f32 master → bf16 compute) ──
        unsafe {
            ffi::muon_weight_update_f32(master, param, result as *const c_void, aspect_lr, wd, mn, stream);
        }

        ve_idx += 1;
    }
}

// ── Muon for block matrix weights (batched NS) ─────────────────────────────
//
// 6 matrices per layer (wq, wk, wv, wo, wfc, wdn), N_LAYER layers.
// Grouped by shape for batched Newton-Schulz GEMMs:
//   - Square (D_MODELxD_MODEL): wq, wk, wv, wo x N_LAYER layers
//   - Tall   (MLP_DIMxD_MODEL): wfc x N_LAYER layers
//   - Wide   (D_MODELxMLP_DIM): wdn x N_LAYER layers
//
// Each group processes:
//   1. Nesterov momentum per-matrix (into stacked buffer)
//   2. Frobenius normalize per-matrix
//   3. Batched Newton-Schulz iterations (3 batched GEMMs per iter x 5 iters = 15 total)
//   4. NorMuon + weight update per-matrix (scatter back)
//
// Scratch buffers (backward pass, unused during optimizer):
//   d_x     (B*T*D bf16) -> x_a  (ping-pong, holds stacked m*n matrices)
//   d_x0    (B*T*D bf16) -> x_b  (ping-pong)
//   d_q     (B*T*D bf16) -> gram (stacked d*d Gram matrices)
//   d_k     (B*T*D bf16) -> sq   (stacked d*d A^2 matrices)
//   d_v     (B*T*D bf16) -> comb (stacked d*d combined matrices)
//   d_h     (B*T*MLP bf16) -> extra scratch for tall/wide stacking if needed

/// Info about one matrix in a shape group, needed for scatter after batched NS.
struct MatrixInfo {
    mom_idx: usize,    // index into bufs.muon.momentum / second_momentum
    layer: usize,
    mat_idx: usize,    // 0..6 (wq, wk, wv, wo, wfc, wdn)
}

fn muon_all_layers(
    bufs: &mut BufferManager,
    gemm: &GemmRunner,
    lr_mult: f32,
    momentum: f32,
    wd: f32,
    _step: usize,
    cfg: &ScheduleConfig,
) {
    let matrix_lr_base = cfg.peak_lr as f32 * lr_mult;
    let scratch = dptr(&bufs.muon.scratch) as *mut f32;

    // Build groups: square (mat_idx 0..4), tall (mat_idx 4), wide (mat_idx 5)
    let mut square_mats: Vec<MatrixInfo> = Vec::with_capacity(N_LAYER * 4);
    let mut tall_mats: Vec<MatrixInfo> = Vec::with_capacity(N_LAYER);
    let mut wide_mats: Vec<MatrixInfo> = Vec::with_capacity(N_LAYER);

    for layer in 0..N_LAYER {
        for mat_idx in 0..6usize {
            let info = MatrixInfo {
                mom_idx: layer * 6 + mat_idx,
                layer,
                mat_idx,
            };
            match mat_idx {
                0..=3 => square_mats.push(info),
                4 => tall_mats.push(info),
                5 => wide_mats.push(info),
                _ => unreachable!(),
            }
        }
    }

    // Process each shape group with batched NS
    muon_group_batched(
        bufs, gemm, &square_mats,
        D_MODEL, D_MODEL, // square: D_MODEL x D_MODEL
        matrix_lr_base, momentum, wd, scratch,
    );
    muon_group_batched(
        bufs, gemm, &tall_mats,
        MLP_DIM, D_MODEL, // tall: MLP_DIM x D_MODEL
        matrix_lr_base, momentum, wd, scratch,
    );
    muon_group_batched(
        bufs, gemm, &wide_mats,
        D_MODEL, MLP_DIM, // wide: D_MODEL x MLP_DIM
        matrix_lr_base, momentum, wd, scratch,
    );
}

/// Process one shape group: gather -> batch NS -> scatter.
fn muon_group_batched(
    bufs: &mut BufferManager,
    gemm: &GemmRunner,
    mats: &[MatrixInfo],
    m: usize,
    n: usize,
    matrix_lr_base: f32,
    momentum: f32,
    wd: f32,
    scratch: *mut f32,
) {
    let mn = m * n;
    let tall = m > n;
    let d = if tall { n } else { m };  // d = min(m, n)
    let dd = d * d;
    let batch = mats.len();
    let aspect_lr = matrix_lr_base * (1.0f64.max(m as f64 / n as f64)).sqrt() as f32;

    // x_a = d_x (stacked m*n matrices), x_b = d_x0
    let x_a_base = dptr(&bufs.d_x);
    let x_b_base = dptr(&bufs.d_x0);
    // gram = d_q (stacked d*d), sq = d_k (stacked d*d), comb = d_v (stacked d*d)
    let gram_base = dptr(&bufs.d_q);
    let sq_base = dptr(&bufs.d_k);
    let comb_base = dptr(&bufs.d_v);

    let stream = bufs.stream.cu_stream() as ffi::CudaStream;

    // ── 1. Nesterov momentum + Frobenius normalize per matrix ──
    // Write Nesterov output into x_a at stride offsets, then frob_normalize into x_b.
    for (i, info) in mats.iter().enumerate() {
        let offset_bytes = i * mn * 2; // bf16 = 2 bytes
        let x_a_i = (x_a_base + offset_bytes as u64) as *mut c_void;
        let x_b_i = (x_b_base + offset_bytes as u64) as *mut c_void;

        let grad = block_grad_ptr(bufs, info.layer, info.mat_idx);
        let mom = dptr(&bufs.muon.momentum[info.mom_idx]) as *mut f32;

        unsafe {
            ffi::muon_nesterov_f32buf(mom, grad as *const c_void, x_a_i, momentum, mn as i32, stream);
            frob_normalize_bf16(x_a_i as *const c_void, x_b_i, mn as i32, scratch, stream);
        }
    }

    // ── 2. Batched Newton-Schulz iterations ──
    // x_b holds initial normalized matrices. Ping-pong between x_b (src) and x_a (dst).
    let mut src_is_b = true;

    for &(ca, cb, cc) in NS_COEFFS[..NS_ITERS].iter() {
        // Compute batched Gram matrix A into gram buffer
        unsafe {
            if tall {
                // A(d,d) = X^T(d,m) @ X(m,d)  [here d=n for tall]
                // Using matmul_acc semantics: dW(N,K) += dY(M,N)^T @ X(M,K)
                // A(n,n) += X(m,n)^T @ X(m,n) => M=m, N=n, K=n
                // Zero gram first (beta=1 accumulate)
                ffi::scale_bf16(gram_base as *mut c_void, 0.0, (batch * dd) as i32, stream);
                if src_is_b {
                    gemm.batched_matmul_acc(&bufs.d_x0, &bufs.d_x0, &mut bufs.d_q, m, n, n, batch);
                } else {
                    gemm.batched_matmul_acc(&bufs.d_x, &bufs.d_x, &mut bufs.d_q, m, n, n, batch);
                }
            } else {
                // A(d,d) = X(d,n) @ X^T(n,d)  [here d=m for square/wide]
                // Using matmul semantics: Y(M,N) = X(M,K) @ W(N,K)^T
                // A(m,m) = X(m,n) @ X(m,n)^T => M=m, N=m, K=n
                if src_is_b {
                    gemm.batched_matmul(&bufs.d_x0, &bufs.d_x0, &mut bufs.d_q, m, m, n, batch);
                } else {
                    gemm.batched_matmul(&bufs.d_x, &bufs.d_x, &mut bufs.d_q, m, m, n, batch);
                }
            }

            // Batched A^2(d,d) = A(d,d) @ A(d,d) into sq
            // matmul_bwd_x: dX(M,K) = dY(M,N) @ W(N,K) => M=d, N=d, K=d
            gemm.batched_matmul_bwd_x(&bufs.d_q, &bufs.d_q, &mut bufs.d_k, d, d, d, batch);

            // Fused: combined = ca*I + cb*A + cc*A^2 (single kernel for all batch elements)
            ffi::ns_combined_batched(
                gram_base as *const c_void,
                sq_base as *const c_void,
                comb_base as *mut c_void,
                ca, cb, cc,
                d as i32, batch as i32, stream,
            );

            // Batched apply: X_new = X @ combined (tall) or combined @ X (square/wide)
            if tall {
                // dst(m,n) = src(m,n) @ combined(n,n)
                // matmul_bwd_x: dX(M,K) = dY(M,N) @ W(N,K), M=m, N=n, K=n
                if src_is_b {
                    gemm.batched_matmul_bwd_x(&bufs.d_x0, &bufs.d_v, &mut bufs.d_x, m, n, n, batch);
                } else {
                    gemm.batched_matmul_bwd_x(&bufs.d_x, &bufs.d_v, &mut bufs.d_x0, m, n, n, batch);
                }
            } else {
                // dst(m,n) = combined(m,m) @ src(m,n)
                // matmul_bwd_x: dX(M,K) = dY(M,N) @ W(N,K), M=m, N=m, K=n
                if src_is_b {
                    gemm.batched_matmul_bwd_x(&bufs.d_v, &bufs.d_x0, &mut bufs.d_x, m, m, n, batch);
                } else {
                    gemm.batched_matmul_bwd_x(&bufs.d_v, &bufs.d_x, &mut bufs.d_x0, m, m, n, batch);
                }
            }
        }

        src_is_b = !src_is_b;
    }

    // After 5 (odd) iterations, result is in x_a (d_x).
    let result_base = if src_is_b { x_b_base } else { x_a_base };

    // ── 3. NorMuon + weight update per matrix (scatter) ──
    let reduce_cols = if m >= n { 1 } else { 0 };

    for (i, info) in mats.iter().enumerate() {
        let offset_bytes = i * mn * 2;
        let result_i = (result_base + offset_bytes as u64) as *mut c_void;

        let second_mom = dptr(&bufs.muon.second_momentum[info.mom_idx]) as *mut f32;
        unsafe {
            normuon_step_bf16(result_i, second_mom, m as i32, n as i32, reduce_cols, 0.95, scratch, stream);
        }

        let param = block_weight_ptr(bufs, info.layer, info.mat_idx);
        let master = block_master_ptr(bufs, info.layer, info.mat_idx);
        unsafe {
            ffi::muon_weight_update_f32(master, param, result_i as *const c_void, aspect_lr, wd, mn as i32, stream);
        }
    }
}

/// Raw device pointer to block weight matrix (bf16 compute copy).
fn block_weight_ptr(bufs: &BufferManager, layer: usize, mat_idx: usize) -> *mut c_void {
    match mat_idx {
        0 => vptr_mut(&bufs.layer_weights[layer].wq),
        1 => vptr_mut(&bufs.layer_weights[layer].wk),
        2 => vptr_mut(&bufs.layer_weights[layer].wv),
        3 => vptr_mut(&bufs.layer_weights[layer].wo),
        4 => vptr_mut(&bufs.layer_weights[layer].wfc),
        5 => vptr_mut(&bufs.layer_weights[layer].wdn),
        _ => unreachable!(),
    }
}

/// Raw device pointer to block weight f32 master copy.
fn block_master_ptr(bufs: &BufferManager, layer: usize, mat_idx: usize) -> *mut f32 {
    match mat_idx {
        0 => dptr(&bufs.layer_weights[layer].wq_f32) as *mut f32,
        1 => dptr(&bufs.layer_weights[layer].wk_f32) as *mut f32,
        2 => dptr(&bufs.layer_weights[layer].wv_f32) as *mut f32,
        3 => dptr(&bufs.layer_weights[layer].wo_f32) as *mut f32,
        4 => dptr(&bufs.layer_weights[layer].wfc_f32) as *mut f32,
        5 => dptr(&bufs.layer_weights[layer].wdn_f32) as *mut f32,
        _ => unreachable!(),
    }
}

/// Repack wq/wk/wv into wqkv = [wq; wk; wv] for all layers (after optimizer step).
fn pack_wqkv(bufs: &BufferManager) {
    bufs.pack_wqkv();
}

/// Raw device pointer to block gradient matrix.
fn block_grad_ptr(bufs: &BufferManager, layer: usize, mat_idx: usize) -> *mut c_void {
    match mat_idx {
        0 => vptr_mut(&bufs.layer_grads[layer].wq),
        1 => vptr_mut(&bufs.layer_grads[layer].wk),
        2 => vptr_mut(&bufs.layer_grads[layer].wv),
        3 => vptr_mut(&bufs.layer_grads[layer].wo),
        4 => vptr_mut(&bufs.layer_grads[layer].wfc),
        5 => vptr_mut(&bufs.layer_grads[layer].wdn),
        _ => unreachable!(),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> ScheduleConfig {
        ScheduleConfig::default()
    }

    #[test]
    fn lr_multiplier_at_zero() {
        assert!((get_lr_multiplier(0.0, &default_cfg()) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn lr_multiplier_mid_plateau() {
        // warmdown_ratio=0.75, plateau is [0, 0.25)
        assert!((get_lr_multiplier(0.1, &default_cfg()) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn lr_multiplier_warmdown_boundary() {
        // 1.0 - warmdown_ratio = 0.25
        assert!((get_lr_multiplier(0.25, &default_cfg()) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn lr_multiplier_mid_warmdown() {
        // progress=0.625: cooldown = (1.0 - 0.625) / 0.75 = 0.5
        // linear: lr = 0.5 + 0.5 * 0.05 = 0.525
        assert!((get_lr_multiplier(0.625, &default_cfg()) - 0.525).abs() < 1e-9);
    }

    #[test]
    fn lr_multiplier_at_end() {
        assert!((get_lr_multiplier(1.0, &default_cfg()) - DEFAULT_FINAL_LR_FRAC).abs() < 1e-9);
    }

    #[test]
    fn lr_multiplier_cosine_at_end() {
        let cfg = ScheduleConfig { schedule: Schedule::Cosine, ..default_cfg() };
        assert!((get_lr_multiplier(1.0, &cfg) - DEFAULT_FINAL_LR_FRAC).abs() < 1e-9);
    }

    #[test]
    fn lr_multiplier_cosine_mid_warmdown() {
        let cfg = ScheduleConfig { schedule: Schedule::Cosine, ..default_cfg() };
        // progress=0.625: cooldown=0.5, t=0.5
        // cosine_mult = 0.5*(1+cos(pi*0.5)) = 0.5*(1+0) = 0.5
        // result = 0.5 + 0.5*0.05 = 0.525
        assert!((get_lr_multiplier(0.625, &cfg) - 0.525).abs() < 1e-9);
    }

    #[test]
    fn lr_multiplier_cosine_at_warmdown_start() {
        let cfg = ScheduleConfig { schedule: Schedule::Cosine, ..default_cfg() };
        // progress=0.25: cooldown=1.0, t=0.0, cosine_mult=0.5*(1+cos(0))=1.0
        assert!((get_lr_multiplier(0.25, &cfg) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn momentum_ramp() {
        assert!((muon_momentum(0) - 0.85).abs() < 1e-6);
        assert!((muon_momentum(100) - 0.90).abs() < 1e-6);
        assert!((muon_momentum(200) - 0.95).abs() < 1e-6);
        assert!((muon_momentum(1000) - 0.95).abs() < 1e-6);
    }

    #[test]
    fn weight_decay_schedule() {
        let cfg = default_cfg();
        assert!((muon_weight_decay(0.0, &cfg) - DEFAULT_WEIGHT_DECAY as f32).abs() < 1e-6);
        assert!((muon_weight_decay(0.5, &cfg) - 0.1).abs() < 1e-6);
        assert!((muon_weight_decay(1.0, &cfg) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn dmodel_scale_value() {
        // (D_MODEL/768)^(-0.5) = sqrt(768/D_MODEL)
        let scale = dmodel_lr_scale();
        let expected = (D_MODEL as f64 / 768.0).powf(-0.5) as f32;
        assert!((scale - expected).abs() < 1e-4);
    }

    #[test]
    fn ns_coefficients_match() {
        assert_eq!(NS_COEFFS.len(), 5);
        assert!((NS_COEFFS[0].0 - 8.15655).abs() < 0.001);
        assert!((NS_COEFFS[4].2 - 0.42324).abs() < 0.001);
    }

    #[test]
    fn bias_corrections() {
        let bc1 = 1.0 - ADAM_BETA1.powi(1);
        assert!((bc1 - 0.2).abs() < 1e-6);
        let bc2 = 1.0 - ADAM_BETA2.powi(1);
        assert!((bc2 - 0.05).abs() < 1e-6);
    }

    #[test]
    fn muon_matrix_dims() {
        // All block matrices have d = min(m,n) = D_MODEL
        for &(m, n) in &[
            (D_MODEL, D_MODEL),
            (MLP_DIM, D_MODEL),
            (D_MODEL, MLP_DIM),
        ] {
            assert_eq!(m.min(n), D_MODEL);
        }
    }

    #[test]
    fn aspect_ratio_scaling() {
        let d = D_MODEL as f64;
        let mlp = MLP_DIM as f64;
        // Square: 1.0
        assert!((1.0f64.max(d / d)).sqrt() - 1.0 < 1e-9);
        // Tall (MLP_DIM x D_MODEL): sqrt(MLP_DIM / D_MODEL)
        let tall_aspect = (1.0f64.max(mlp / d)).sqrt();
        assert!((tall_aspect - (mlp / d).sqrt()).abs() < 1e-9);
        // Wide (D_MODEL x MLP_DIM): 1.0
        assert!((1.0f64.max(d / mlp)).sqrt() - 1.0 < 1e-9);
    }

    #[test]
    fn ping_pong_final_buffer() {
        // After 5 (odd) NS iterations, src_is_b toggles from true 5 times -> false.
        // Result is in x_a (d_x).
        let mut src_is_b = true;
        for _ in 0..NS_ITERS {
            src_is_b = !src_is_b;
        }
        assert!(!src_is_b);
    }

    #[test]
    fn momentum_buffer_count() {
        // 6 matrices per layer (wq, wk, wv, wo, wfc, wdn) x N_LAYER
        let expected = N_LAYER * 6;
        assert!(expected > 0);
        assert_eq!(expected, 54); // 9 * 6
    }

    #[test]
    fn ve_layer_count() {
        assert_eq!(VE_LAYERS.len(), (0..N_LAYER).filter(|&l| has_ve(l)).count());
    }
}

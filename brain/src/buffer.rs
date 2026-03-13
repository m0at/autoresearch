use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, ValidAsZeroBits};
use half::bf16;

use crate::config::*;

/// Per-layer weight buffers.
pub struct LayerWeights {
    pub wq: CudaSlice<bf16>,  // [D_MODEL, D_MODEL]
    pub wk: CudaSlice<bf16>,  // [D_MODEL, D_MODEL]
    pub wv: CudaSlice<bf16>,  // [D_MODEL, D_MODEL]
    pub wqkv: CudaSlice<bf16>, // [3*D_MODEL, D_MODEL] packed [wq; wk; wv]
    pub wo: CudaSlice<bf16>,  // [D_MODEL, D_MODEL]
    pub wfc: CudaSlice<bf16>, // [MLP_DIM, D_MODEL]
    pub wdn: CudaSlice<bf16>, // [D_MODEL, MLP_DIM]
    pub ve_weight: Option<CudaSlice<bf16>>, // [VOCAB, D_MODEL]
    pub ve_gate: Option<CudaSlice<bf16>>,   // [N_KV_HEAD, VE_GATE_CH]
    // f32 master copies for mixed-precision Muon (matching Python's f32 nn.Linear params)
    pub wq_f32: CudaSlice<f32>,
    pub wk_f32: CudaSlice<f32>,
    pub wv_f32: CudaSlice<f32>,
    pub wo_f32: CudaSlice<f32>,
    pub wfc_f32: CudaSlice<f32>,
    pub wdn_f32: CudaSlice<f32>,
    pub ve_gate_f32: Option<CudaSlice<f32>>,
}

/// Per-layer gradient buffers (same shapes as weights).
pub struct LayerGrads {
    pub wq: CudaSlice<bf16>,
    pub wk: CudaSlice<bf16>,
    pub wv: CudaSlice<bf16>,
    pub wqkv: CudaSlice<bf16>, // [3*D_MODEL, D_MODEL] packed grad
    pub wo: CudaSlice<bf16>,
    pub wfc: CudaSlice<bf16>,
    pub wdn: CudaSlice<bf16>,
    pub ve_weight: Option<CudaSlice<bf16>>,
    pub ve_gate: Option<CudaSlice<bf16>>,
}

/// AdamW optimizer state for embedding/scalar params.
pub struct AdamWState {
    pub wte_exp_avg: CudaSlice<bf16>,
    pub wte_exp_avg_sq: CudaSlice<bf16>,
    pub lm_head_exp_avg: CudaSlice<f32>,
    pub lm_head_exp_avg_sq: CudaSlice<f32>,
    pub resid_lambdas_exp_avg: CudaSlice<f32>,
    pub resid_lambdas_exp_avg_sq: CudaSlice<f32>,
    pub x0_lambdas_exp_avg: CudaSlice<f32>,
    pub x0_lambdas_exp_avg_sq: CudaSlice<f32>,
    // Per VE layer (VE_LAYERS.len() entries)
    pub ve_exp_avg: Vec<CudaSlice<bf16>>,
    pub ve_exp_avg_sq: Vec<CudaSlice<bf16>>,
}

/// Muon optimizer state for block matrix weights.
/// Includes ve_gate weights which Python also treats with Muon (not AdamW).
pub struct MuonState {
    pub momentum: Vec<CudaSlice<f32>>,       // f32 momentum (matching Python's f32 params)
    pub second_momentum: Vec<CudaSlice<f32>>, // NorMuon EMA buffers
    // ve_gate: [N_KV_HEAD, VE_GATE_CH] per VE layer, treated with Muon (matching Python)
    pub ve_gate_momentum: Vec<CudaSlice<f32>>,        // f32, VE_LAYERS.len() entries
    pub ve_gate_second_momentum: Vec<CudaSlice<f32>>, // VE_LAYERS.len() entries (reduce_cols=0, size=VE_GATE_CH)
    // Pre-allocated scratch for frob_normalize (1 float) and normuon_step (num_groups + 2 floats).
    // Max num_groups = max(MLP_DIM, D_MODEL). Allocated as MLP_DIM + 3.
    pub scratch: CudaSlice<f32>,
}

/// All GPU memory for training, allocated once at init.
pub struct BufferManager {
    pub stream: Arc<CudaStream>,
    pub batch_size: usize,

    // ── Weights (bf16) ──
    pub wte: CudaSlice<bf16>,              // [VOCAB, D_MODEL]
    pub lm_head: CudaSlice<bf16>,          // [VOCAB, D_MODEL]
    pub lm_head_f32: CudaSlice<f32>,       // f32 master for AdamW (Python keeps lm_head in f32)
    pub resid_lambdas: CudaSlice<bf16>,    // [N_LAYER]
    pub x0_lambdas: CudaSlice<bf16>,       // [N_LAYER]
    pub layer_weights: Vec<LayerWeights>,

    // ── Gradients (same layout as weights) ──
    pub wte_grad: CudaSlice<bf16>,
    pub lm_head_grad: CudaSlice<bf16>,
    pub resid_lambdas_grad: CudaSlice<f32>, // scalar grads in f32
    pub x0_lambdas_grad: CudaSlice<f32>,
    pub layer_grads: Vec<LayerGrads>,

    // ── Optimizer state ──
    pub adamw: AdamWState,
    pub muon: MuonState,

    // ── Activations (reused across layers) ──
    pub emb: CudaSlice<bf16>,      // [B, T, D_MODEL]
    pub x: CudaSlice<bf16>,        // [B, T, D_MODEL]
    pub x0: CudaSlice<bf16>,       // [B, T, D_MODEL]
    pub xn: CudaSlice<bf16>,       // [B, T, D_MODEL]
    pub q: CudaSlice<bf16>,        // [B*T, D_MODEL]  (reshaped as [B,T,N_HEAD,HEAD_DIM])
    pub k: CudaSlice<bf16>,        // [B*T, D_MODEL]
    pub v: CudaSlice<bf16>,        // [B*T, D_MODEL]
    pub qkv: CudaSlice<bf16>,      // [3*B*T, D_MODEL] packed [q|k|v] for batched QKV GEMM
    pub ve: CudaSlice<bf16>,       // [B, T, D_MODEL]
    pub gate: CudaSlice<bf16>,     // [B*T, N_KV_HEAD]
    pub attn_out: CudaSlice<bf16>, // [B, T, D_MODEL]
    pub h: CudaSlice<bf16>,        // [B*T, MLP_DIM]
    pub h_act: CudaSlice<bf16>,    // [B*T, MLP_DIM]
    pub logits: CudaSlice<bf16>,   // [B*T, VOCAB]
    pub loss: CudaSlice<f32>,      // [1]

    // ── Saved for backward (per-layer) ──
    pub saved_x_pre_attn_norm: Vec<CudaSlice<bf16>>,  // [N_LAYER] x [B, T, D_MODEL]
    pub saved_x_pre_mlp_norm: Vec<CudaSlice<bf16>>,   // [N_LAYER] x [B, T, D_MODEL]
    pub saved_h_pre_act: Vec<CudaSlice<bf16>>,         // [N_LAYER] x [B*T, MLP_DIM]
    pub saved_xn: Vec<CudaSlice<bf16>>,                // [N_LAYER] x [B*T, D_MODEL]  normed (for dW)
    pub saved_q: Vec<CudaSlice<bf16>>,                 // [N_LAYER] x [B*T, D_MODEL]
    pub saved_k: Vec<CudaSlice<bf16>>,                 // [N_LAYER] x [B*T, D_MODEL]
    pub saved_v: Vec<CudaSlice<bf16>>,                 // [N_LAYER] x [B*T, D_MODEL]
    pub saved_attn_out: Vec<CudaSlice<bf16>>,          // [N_LAYER] x [B*T, D_MODEL]
    pub saved_softmax_lse: Vec<CudaSlice<f32>>,       // [N_LAYER] x [B, N_HEAD, SEQ] flash attn LSE

    // ── Backward scratch (reused across layers) ──
    pub d_x: CudaSlice<bf16>,      // [B, T, D_MODEL]
    pub d_x0: CudaSlice<bf16>,     // [B, T, D_MODEL]
    pub d_q: CudaSlice<bf16>,      // [B*T, D_MODEL]
    pub d_k: CudaSlice<bf16>,      // [B*T, D_MODEL]
    pub d_v: CudaSlice<bf16>,      // [B*T, D_MODEL]
    pub d_qkv: CudaSlice<bf16>,    // [3*B*T, D_MODEL] packed [d_q|d_k|d_v] for batched bwd
    pub d_h: CudaSlice<bf16>,      // [B*T, MLP_DIM]
    pub d_logits: CudaSlice<bf16>, // [B*T, VOCAB]
    pub d_xn: CudaSlice<bf16>,     // [B*T, D_MODEL]  scratch for 3-way add

    // ── Flash attention backward scratch (f32) ──
    pub flash_dq_accum: CudaSlice<f32>,    // [B, N_HEAD, SEQ, HEAD_DIM]
    pub flash_dsoftmax_sum: CudaSlice<f32>, // [B, N_HEAD, SEQ]
    // FA3-specific backward scratch
    pub fa3_softmax_lse_log2: CudaSlice<f32>, // [B, N_HEAD, SEQ]
    pub fa3_dq_semaphore: CudaSlice<i32>,     // [ceil(SEQ/128), B, N_HEAD]
    pub fa3_scheduler_meta: CudaSlice<i32>,   // scheduler metadata for FA3 forward

    // ── Fixed buffers ──
    pub cos: CudaSlice<bf16>,      // [T, HEAD_DIM/2]
    pub sin: CudaSlice<bf16>,      // [T, HEAD_DIM/2]
    pub input_ids: CudaSlice<u32>,   // [B, T]
    pub targets: CudaSlice<u32>,     // [B, T]
    pub input_ids_b: CudaSlice<u32>, // [B, T] double-buffer for async H2D
    pub targets_b: CudaSlice<u32>,   // [B, T] double-buffer for async H2D

    // ── Dynamic layer importance ──
    pub layer_act_norms: CudaSlice<f32>,      // [N_LAYER] MLP output L2 norm per layer
    pub layer_neuron_act_norms: CudaSlice<f32>,  // [N_LAYER × MLP_DIM] per-neuron mean abs activation
    pub layer_grad_norms: CudaSlice<f32>,     // [N_LAYER] gradient L2 norm per layer (train)
    pub layer_val_grad_norms: CudaSlice<f32>, // [N_LAYER] gradient L2 norm per layer (val)
    pub layer_dynamic_scale: CudaSlice<f32>,  // [N_LAYER] learned importance scale, init=1.0
}

fn alloc<T: cudarc::driver::DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    count: usize,
) -> Result<CudaSlice<T>> {
    Ok(stream.alloc_zeros::<T>(count)?)
}

fn alloc_bf16(stream: &Arc<CudaStream>, count: usize) -> Result<CudaSlice<bf16>> {
    alloc::<bf16>(stream, count)
}

fn alloc_f32(stream: &Arc<CudaStream>, count: usize) -> Result<CudaSlice<f32>> {
    alloc::<f32>(stream, count)
}

fn alloc_u32(stream: &Arc<CudaStream>, count: usize) -> Result<CudaSlice<u32>> {
    alloc::<u32>(stream, count)
}

impl BufferManager {
    /// Allocate all GPU memory for training. `batch_size` = device batch size (B).
    pub fn new(stream: Arc<CudaStream>, batch_size: usize) -> Result<Self> {
        let b = batch_size;
        let t = SEQ;
        let bt = b * t;
        let d = D_MODEL;
        let mlp = MLP_DIM;

        // ── Weights ──
        let wte = alloc_bf16(&stream, VOCAB * d)?;
        let lm_head = alloc_bf16(&stream, VOCAB * d)?;
        let lm_head_f32 = alloc_f32(&stream, VOCAB * d)?;
        let resid_lambdas = alloc_bf16(&stream, N_LAYER)?;
        let x0_lambdas = alloc_bf16(&stream, N_LAYER)?;

        let mut layer_weights = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            let ve = has_ve(i);
            layer_weights.push(LayerWeights {
                wq: alloc_bf16(&stream, d * d)?,
                wk: alloc_bf16(&stream, d * d)?,
                wv: alloc_bf16(&stream, d * d)?,
                wqkv: alloc_bf16(&stream, 3 * d * d)?,
                wo: alloc_bf16(&stream, d * d)?,
                wfc: alloc_bf16(&stream, mlp * d)?,
                wdn: alloc_bf16(&stream, d * mlp)?,
                ve_weight: if ve { Some(alloc_bf16(&stream, VOCAB * d)?) } else { None },
                ve_gate: if ve { Some(alloc_bf16(&stream, N_KV_HEAD * VE_GATE_CH)?) } else { None },
                // f32 master copies (mixed-precision Muon)
                wq_f32: alloc_f32(&stream, d * d)?,
                wk_f32: alloc_f32(&stream, d * d)?,
                wv_f32: alloc_f32(&stream, d * d)?,
                wo_f32: alloc_f32(&stream, d * d)?,
                wfc_f32: alloc_f32(&stream, mlp * d)?,
                wdn_f32: alloc_f32(&stream, d * mlp)?,
                ve_gate_f32: if ve { Some(alloc_f32(&stream, N_KV_HEAD * VE_GATE_CH)?) } else { None },
            });
        }

        // ── Gradients ──
        let wte_grad = alloc_bf16(&stream, VOCAB * d)?;
        let lm_head_grad = alloc_bf16(&stream, VOCAB * d)?;
        let resid_lambdas_grad = alloc_f32(&stream, N_LAYER)?;
        let x0_lambdas_grad = alloc_f32(&stream, N_LAYER)?;

        let mut layer_grads = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            let ve = has_ve(i);
            layer_grads.push(LayerGrads {
                wq: alloc_bf16(&stream, d * d)?,
                wk: alloc_bf16(&stream, d * d)?,
                wv: alloc_bf16(&stream, d * d)?,
                wqkv: alloc_bf16(&stream, 3 * d * d)?,
                wo: alloc_bf16(&stream, d * d)?,
                wfc: alloc_bf16(&stream, mlp * d)?,
                wdn: alloc_bf16(&stream, d * mlp)?,
                ve_weight: if ve { Some(alloc_bf16(&stream, VOCAB * d)?) } else { None },
                ve_gate: if ve { Some(alloc_bf16(&stream, N_KV_HEAD * VE_GATE_CH)?) } else { None },
            });
        }

        // ── Optimizer state ──
        let mut ve_exp_avg = Vec::with_capacity(VE_LAYERS.len());
        let mut ve_exp_avg_sq = Vec::with_capacity(VE_LAYERS.len());
        for _ in &VE_LAYERS {
            ve_exp_avg.push(alloc_bf16(&stream, VOCAB * d)?);
            ve_exp_avg_sq.push(alloc_bf16(&stream, VOCAB * d)?);
        }

        let adamw = AdamWState {
            wte_exp_avg: alloc_bf16(&stream, VOCAB * d)?,
            wte_exp_avg_sq: alloc_bf16(&stream, VOCAB * d)?,
            lm_head_exp_avg: alloc_f32(&stream, VOCAB * d)?,
            lm_head_exp_avg_sq: alloc_f32(&stream, VOCAB * d)?,
            resid_lambdas_exp_avg: alloc_f32(&stream, N_LAYER)?,
            resid_lambdas_exp_avg_sq: alloc_f32(&stream, N_LAYER)?,
            x0_lambdas_exp_avg: alloc_f32(&stream, N_LAYER)?,
            x0_lambdas_exp_avg_sq: alloc_f32(&stream, N_LAYER)?,
            ve_exp_avg,
            ve_exp_avg_sq,
        };

        // Muon: one momentum buffer per block matrix weight (6 per layer)
        // f32 momentum to match Python (nn.Linear params are f32, momentum_buffer is f32)
        let mut momentum = Vec::with_capacity(N_LAYER * 6);
        let mut second_momentum = Vec::with_capacity(N_LAYER * 6);
        for _ in 0..N_LAYER {
            // wq, wk, wv, wo: [D_MODEL, D_MODEL] -> reduce_cols=1, second_mom size = D_MODEL
            for _ in 0..4 {
                momentum.push(alloc_f32(&stream, d * d)?);
                second_momentum.push(alloc_f32(&stream, d)?);
            }
            // wfc: [MLP_DIM, D_MODEL] -> reduce_cols=1, second_mom size = MLP_DIM
            momentum.push(alloc_f32(&stream, mlp * d)?);
            second_momentum.push(alloc_f32(&stream, mlp)?);
            // wdn: [D_MODEL, MLP_DIM] -> reduce_cols=0, second_mom size = MLP_DIM
            momentum.push(alloc_f32(&stream, d * mlp)?);
            second_momentum.push(alloc_f32(&stream, mlp)?);
        }
        // ve_gate: [N_KV_HEAD, VE_GATE_CH] per VE layer — Muon, matching Python
        // m < n (4 < 32), so reduce_cols=0, second_mom size = VE_GATE_CH
        let mut ve_gate_momentum = Vec::with_capacity(VE_LAYERS.len());
        let mut ve_gate_second_momentum = Vec::with_capacity(VE_LAYERS.len());
        for _ in &VE_LAYERS {
            ve_gate_momentum.push(alloc_f32(&stream, N_KV_HEAD * VE_GATE_CH)?);
            ve_gate_second_momentum.push(alloc_f32(&stream, VE_GATE_CH)?);
        }
        // Scratch for frob_normalize (1 float) and normuon_step (num_groups + 2 floats).
        // Max num_groups = max(MLP_DIM, D_MODEL). Allocate MLP_DIM + 3 to cover all cases.
        let muon_scratch = alloc_f32(&stream, MLP_DIM + 3)?;
        let muon = MuonState {
            momentum, second_momentum,
            ve_gate_momentum, ve_gate_second_momentum,
            scratch: muon_scratch,
        };

        // ── Activations ──
        let emb = alloc_bf16(&stream, bt * d)?;
        let x = alloc_bf16(&stream, bt * d)?;
        let x0 = alloc_bf16(&stream, bt * d)?;
        let xn = alloc_bf16(&stream, bt * d)?;
        let q = alloc_bf16(&stream, bt * d)?;
        let k = alloc_bf16(&stream, bt * d)?;
        let v = alloc_bf16(&stream, bt * d)?;
        let qkv = alloc_bf16(&stream, 3 * bt * d)?;
        let ve_buf = alloc_bf16(&stream, bt * d)?;
        let gate = alloc_bf16(&stream, bt * N_KV_HEAD)?;
        let attn_out = alloc_bf16(&stream, bt * d)?;
        let h = alloc_bf16(&stream, bt * mlp)?;
        let h_act = alloc_bf16(&stream, bt * mlp)?;
        let logits = alloc_bf16(&stream, bt * VOCAB)?;
        let loss = alloc_f32(&stream, 1)?;

        // ── Saved for backward ──
        let mut saved_x_pre_attn_norm = Vec::with_capacity(N_LAYER);
        let mut saved_x_pre_mlp_norm = Vec::with_capacity(N_LAYER);
        let mut saved_h_pre_act = Vec::with_capacity(N_LAYER);
        let mut saved_xn = Vec::with_capacity(N_LAYER);
        let mut saved_q = Vec::with_capacity(N_LAYER);
        let mut saved_k = Vec::with_capacity(N_LAYER);
        let mut saved_v = Vec::with_capacity(N_LAYER);
        let mut saved_attn_out = Vec::with_capacity(N_LAYER);
        for _ in 0..N_LAYER {
            saved_x_pre_attn_norm.push(alloc_bf16(&stream, bt * d)?);
            saved_x_pre_mlp_norm.push(alloc_bf16(&stream, bt * d)?);
            saved_h_pre_act.push(alloc_bf16(&stream, bt * mlp)?);
            saved_xn.push(alloc_bf16(&stream, bt * d)?);
            saved_q.push(alloc_bf16(&stream, bt * d)?);
            saved_k.push(alloc_bf16(&stream, bt * d)?);
            saved_v.push(alloc_bf16(&stream, bt * d)?);
            saved_attn_out.push(alloc_bf16(&stream, bt * d)?);
        }
        let mut saved_softmax_lse = Vec::with_capacity(N_LAYER);
        for _ in 0..N_LAYER {
            saved_softmax_lse.push(alloc_f32(&stream, b * N_HEAD * t)?);
        }

        // ── Backward scratch ──
        let d_x = alloc_bf16(&stream, bt * d)?;
        let d_x0 = alloc_bf16(&stream, bt * d)?;
        let d_q = alloc_bf16(&stream, bt * d)?;
        let d_k = alloc_bf16(&stream, bt * d)?;
        let d_v = alloc_bf16(&stream, bt * d)?;
        let d_qkv = alloc_bf16(&stream, 3 * bt * d)?;
        let d_h = alloc_bf16(&stream, bt * mlp)?;
        let d_logits = alloc_bf16(&stream, bt * VOCAB)?;
        let d_xn = alloc_bf16(&stream, bt * d)?;

        // ── Flash attention backward scratch (f32) ──
        let flash_dq_accum = alloc_f32(&stream, b * N_HEAD * t * HEAD_DIM)?;
        let flash_dsoftmax_sum = alloc_f32(&stream, b * N_HEAD * t)?;
        // FA3-specific: softmax_lse_log2 [B, N_HEAD, SEQ] and dq_semaphore [ceil(SEQ/128), B, N_HEAD]
        let fa3_softmax_lse_log2 = alloc_f32(&stream, b * N_HEAD * t)?;
        let dq_sem_blocks = (t + 127) / 128; // kBlockM = 128 for hdim128
        let fa3_dq_semaphore = alloc::<i32>(&stream, dq_sem_blocks * b * N_HEAD)?;
        // FA3 scheduler metadata: round_up(B,4)*4 + 1 ints (tile_count_semaphore + metadata vectors)
        let fa3_scheduler_meta = alloc::<i32>(&stream, 2048)?; // generous, ~8KB

        // ── Fixed buffers ──
        let cos = alloc_bf16(&stream, t * (HEAD_DIM / 2))?;
        let sin = alloc_bf16(&stream, t * (HEAD_DIM / 2))?;
        let input_ids = alloc_u32(&stream, bt)?;
        let targets = alloc_u32(&stream, bt)?;
        let input_ids_b = alloc_u32(&stream, bt)?;
        let targets_b = alloc_u32(&stream, bt)?;

        // ── Dynamic layer importance ──
        let layer_act_norms = alloc_f32(&stream, N_LAYER)?;
        let layer_neuron_act_norms = alloc_f32(&stream, N_LAYER * MLP_DIM)?;
        let layer_grad_norms = alloc_f32(&stream, N_LAYER)?;
        let layer_val_grad_norms = alloc_f32(&stream, N_LAYER)?;
        // Initialize dynamic_scale to 1.0 (alloc_zeros gives 0.0, need 1.0)
        let layer_dynamic_scale_host = vec![1.0f32; N_LAYER];
        let mut layer_dynamic_scale = alloc_f32(&stream, N_LAYER)?;
        stream.memcpy_htod(&layer_dynamic_scale_host, &mut layer_dynamic_scale)?;

        Ok(Self {
            stream,
            batch_size: b,
            wte,
            lm_head,
            lm_head_f32,
            resid_lambdas,
            x0_lambdas,
            layer_weights,
            wte_grad,
            lm_head_grad,
            resid_lambdas_grad,
            x0_lambdas_grad,
            layer_grads,
            adamw,
            muon,
            emb,
            x,
            x0,
            xn,
            q,
            k,
            v,
            qkv,
            ve: ve_buf,
            gate,
            attn_out,
            h,
            h_act,
            logits,
            loss,
            saved_x_pre_attn_norm,
            saved_x_pre_mlp_norm,
            saved_h_pre_act,
            saved_xn,
            saved_q,
            saved_k,
            saved_v,
            saved_attn_out,
            saved_softmax_lse,
            d_x,
            d_x0,
            d_q,
            d_k,
            d_v,
            d_qkv,
            d_h,
            d_logits,
            d_xn,
            flash_dq_accum,
            flash_dsoftmax_sum,
            fa3_softmax_lse_log2,
            fa3_dq_semaphore,
            fa3_scheduler_meta,
            cos,
            sin,
            input_ids,
            targets,
            input_ids_b,
            targets_b,
            layer_act_norms,
            layer_neuron_act_norms,
            layer_grad_norms,
            layer_val_grad_norms,
            layer_dynamic_scale,
        })
    }

    /// Zero all gradient buffers. Call before each training step (or set of micro-steps).
    pub fn zero_gradients(&mut self) -> Result<()> {
        let s = &self.stream;
        zero(s, &mut self.wte_grad)?;
        zero(s, &mut self.lm_head_grad)?;
        zero_f32(s, &mut self.resid_lambdas_grad)?;
        zero_f32(s, &mut self.x0_lambdas_grad)?;
        for lg in &mut self.layer_grads {
            zero(s, &mut lg.wq)?;
            zero(s, &mut lg.wk)?;
            zero(s, &mut lg.wv)?;
            zero(s, &mut lg.wqkv)?;
            zero(s, &mut lg.wo)?;
            zero(s, &mut lg.wfc)?;
            zero(s, &mut lg.wdn)?;
            if let Some(ref mut ve) = lg.ve_weight {
                zero(s, ve)?;
            }
            if let Some(ref mut g) = lg.ve_gate {
                zero(s, g)?;
            }
        }
        Ok(())
    }

    /// Pack wq/wk/wv into wqkv = [wq; wk; wv] for all layers.
    /// Must be called after loading/initializing weights and after each optimizer step.
    pub fn pack_wqkv(&self) {
        let dd_bytes = D_MODEL * D_MODEL * std::mem::size_of::<bf16>();
        let stream_ptr = self.stream.cu_stream();
        for layer in 0..N_LAYER {
            let lw = &self.layer_weights[layer];
            let wqkv_base = {
                let (ptr, _sync) = lw.wqkv.device_ptr(lw.wqkv.stream());
                ptr
            };
            let wq_ptr = {
                let (ptr, _sync) = lw.wq.device_ptr(lw.wq.stream());
                ptr
            };
            let wk_ptr = {
                let (ptr, _sync) = lw.wk.device_ptr(lw.wk.stream());
                ptr
            };
            let wv_ptr = {
                let (ptr, _sync) = lw.wv.device_ptr(lw.wv.stream());
                ptr
            };
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    wqkv_base, wq_ptr, dd_bytes, stream_ptr,
                );
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    wqkv_base + dd_bytes as u64, wk_ptr, dd_bytes, stream_ptr,
                );
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    wqkv_base + 2 * dd_bytes as u64, wv_ptr, dd_bytes, stream_ptr,
                );
            }
        }
    }

    /// Total bytes allocated across all buffers.
    pub fn total_bytes(&self) -> usize {
        let b = self.batch_size;
        let t = SEQ;
        let bt = b * t;
        let d = D_MODEL;
        let mlp = MLP_DIM;
        let bf16_sz = std::mem::size_of::<bf16>();
        let f32_sz = std::mem::size_of::<f32>();
        let u32_sz = std::mem::size_of::<u32>();

        let mut total: usize = 0;

        // Weights
        total += 2 * VOCAB * d * bf16_sz;           // wte + lm_head
        total += 2 * N_LAYER * bf16_sz;              // resid_lambdas + x0_lambdas
        total += N_LAYER * (4 * d * d + 2 * mlp * d) * bf16_sz; // per-layer block weights
        total += VE_LAYERS.len() * VOCAB * d * bf16_sz;         // ve_weight
        total += VE_LAYERS.len() * N_KV_HEAD * VE_GATE_CH * bf16_sz; // ve_gate

        // Gradients (same as weights, except scalar grads are f32)
        total += 2 * VOCAB * d * bf16_sz;
        total += 2 * N_LAYER * f32_sz;
        total += N_LAYER * (4 * d * d + 2 * mlp * d) * bf16_sz;
        total += VE_LAYERS.len() * VOCAB * d * bf16_sz;
        total += VE_LAYERS.len() * N_KV_HEAD * VE_GATE_CH * bf16_sz;

        // Optimizer state
        // AdamW: wte, lm_head (2 bufs each), scalars (f32), ve weights (2 bufs each)
        total += 4 * VOCAB * d * bf16_sz; // wte + lm_head exp_avg/sq
        total += 4 * N_LAYER * f32_sz;    // resid/x0 lambdas exp_avg/sq
        total += VE_LAYERS.len() * 2 * VOCAB * d * bf16_sz; // ve exp_avg/sq
        // Muon: momentum for 6 matrices per layer
        total += N_LAYER * (4 * d * d + 2 * mlp * d) * bf16_sz;
        // Muon: second_momentum (NorMuon EMA, f32) for 6 matrices per layer
        // wq/wk/wv/wo: reduce_cols=1, size=D_MODEL; wfc: reduce_cols=1, size=MLP_DIM; wdn: reduce_cols=0, size=MLP_DIM
        total += N_LAYER * (4 * d + 2 * mlp) * f32_sz;
        // Muon: ve_gate momentum + second_momentum
        total += VE_LAYERS.len() * N_KV_HEAD * VE_GATE_CH * bf16_sz; // ve_gate_momentum
        total += VE_LAYERS.len() * VE_GATE_CH * f32_sz;               // ve_gate_second_momentum
        // Muon: scratch for frob_normalize + normuon_step
        total += (mlp + 3) * f32_sz;

        // Activations (reused)
        // emb, x, x0, xn, q, k, v, ve, attn_out = 9 * bt*d
        total += 9 * bt * d * bf16_sz;
        total += bt * N_KV_HEAD * bf16_sz; // gate
        total += 2 * bt * mlp * bf16_sz;   // h, h_act
        total += bt * VOCAB * bf16_sz;     // logits
        total += f32_sz;                   // loss

        // Saved for backward (per-layer)
        // x_pre_attn_norm, x_pre_mlp_norm, xn, q, k, v, attn_out = 7 * N_LAYER * bt*d
        total += 7 * N_LAYER * bt * d * bf16_sz;
        // h_pre_act = N_LAYER * bt * mlp
        total += N_LAYER * bt * mlp * bf16_sz;
        // softmax_lse = N_LAYER * b * N_HEAD * t
        total += N_LAYER * b * N_HEAD * t * f32_sz;

        // Backward scratch
        // d_x, d_x0, d_q, d_k, d_v, d_xn = 6 * bt*d
        total += 6 * bt * d * bf16_sz;
        total += bt * mlp * bf16_sz;   // d_h
        total += bt * VOCAB * bf16_sz; // d_logits
        // Flash attention backward scratch (f32)
        total += b * N_HEAD * t * HEAD_DIM * f32_sz; // flash_dq_accum
        total += b * N_HEAD * t * f32_sz;             // flash_dsoftmax_sum
        total += b * N_HEAD * t * f32_sz;             // fa3_softmax_lse_log2
        total += ((t + 127) / 128) * b * N_HEAD * u32_sz; // fa3_dq_semaphore

        // Fixed
        total += 2 * t * (HEAD_DIM / 2) * bf16_sz; // cos, sin
        total += 4 * bt * u32_sz;                    // input_ids, targets (x2 for double buffer)

        total
    }
}

fn zero(stream: &Arc<CudaStream>, buf: &mut CudaSlice<bf16>) -> Result<()> {
    stream.memset_zeros(buf)?;
    Ok(())
}

fn zero_f32(stream: &Arc<CudaStream>, buf: &mut CudaSlice<f32>) -> Result<()> {
    stream.memset_zeros(buf)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_bytes_b128() {
        let b = 128;
        let t = SEQ;
        let bt = b * t;
        let d = D_MODEL;
        let mlp = MLP_DIM;
        let bf16_sz = 2usize;
        let n_ve = VE_LAYERS.len();

        let weight_elems = 2 * VOCAB * d      // wte + lm_head
            + 2 * N_LAYER                      // scalars
            + N_LAYER * (4 * d * d + 2 * mlp * d) // block weights
            + n_ve * VOCAB * d                  // ve_weight
            + n_ve * N_KV_HEAD * VE_GATE_CH;    // ve_gate
        let weight_bytes = weight_elems * bf16_sz;

        // Sanity: weight bytes should be positive and < 2 GB
        let weight_mb = weight_bytes as f64 / (1024.0 * 1024.0);
        assert!(weight_mb > 0.0, "weight_mb must be positive");
        assert!(weight_mb < 2048.0, "weight_mb {weight_mb:.1} unreasonably large");

        // Verify saved-for-backward: 7*N_LAYER*bt*d + N_LAYER*bt*mlp
        let saved_elems = 7 * N_LAYER * bt * d + N_LAYER * bt * mlp;
        let saved_bytes = saved_elems * bf16_sz;
        let saved_mb = saved_bytes as f64 / (1024.0 * 1024.0);
        assert!(saved_mb > 0.0);
    }
}

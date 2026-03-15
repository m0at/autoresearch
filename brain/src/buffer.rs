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
    pub wfc: Option<CudaSlice<bf16>>, // [MLP_DIM, D_MODEL] — None for MoE layers
    pub wdn: Option<CudaSlice<bf16>>, // [D_MODEL, MLP_DIM] — None for MoE layers
    pub ve_weight: Option<CudaSlice<bf16>>, // [VOCAB, D_MODEL]
    pub ve_gate: Option<CudaSlice<bf16>>,   // [N_KV_HEAD, VE_GATE_CH]
    // f32 master copies for mixed-precision Muon (matching Python's f32 nn.Linear params)
    pub wq_f32: CudaSlice<f32>,
    pub wk_f32: CudaSlice<f32>,
    pub wv_f32: CudaSlice<f32>,
    pub wo_f32: CudaSlice<f32>,
    pub wfc_f32: Option<CudaSlice<f32>>,  // None for MoE layers
    pub wdn_f32: Option<CudaSlice<f32>>,  // None for MoE layers
    pub ve_gate_f32: Option<CudaSlice<f32>>,
}

/// Per-MoE-layer weight buffers.
pub struct MoeLayerWeights {
    pub w_router: CudaSlice<bf16>,      // [N_EXPERTS, D_MODEL]
    pub expert_wfc: CudaSlice<bf16>,    // [N_EXPERTS * MLP_DIM_E, D_MODEL]
    pub expert_wdn: CudaSlice<bf16>,    // [N_EXPERTS * D_MODEL, MLP_DIM_E]
    pub w_router_f32: CudaSlice<f32>,
    pub expert_wfc_f32: CudaSlice<f32>,
    pub expert_wdn_f32: CudaSlice<f32>,
}

/// Per-MoE-layer gradient buffers.
pub struct MoeLayerGrads {
    pub w_router: CudaSlice<bf16>,      // [N_EXPERTS, D_MODEL]
    pub expert_wfc: CudaSlice<bf16>,    // [N_EXPERTS * MLP_DIM_E, D_MODEL]
    pub expert_wdn: CudaSlice<bf16>,    // [N_EXPERTS * D_MODEL, MLP_DIM_E]
}

/// Per-layer gradient buffers (same shapes as weights).
pub struct LayerGrads {
    pub wq: CudaSlice<bf16>,
    pub wk: CudaSlice<bf16>,
    pub wv: CudaSlice<bf16>,
    pub wqkv: CudaSlice<bf16>, // [3*D_MODEL, D_MODEL] packed grad
    pub wo: CudaSlice<bf16>,
    pub wfc: Option<CudaSlice<bf16>>,  // None for MoE layers
    pub wdn: Option<CudaSlice<bf16>>,  // None for MoE layers
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
    // MoE router weights: [N_EXPERTS, D_MODEL] per MoE layer, AdamW (f32)
    pub moe_router_exp_avg: Vec<CudaSlice<f32>>,
    pub moe_router_exp_avg_sq: Vec<CudaSlice<f32>>,
}

/// Muon optimizer state for block matrix weights.
/// Includes ve_gate weights which Python also treats with Muon (not AdamW).
pub struct MuonState {
    pub momentum: Vec<CudaSlice<f32>>,       // f32 momentum (matching Python's f32 params)
    pub second_momentum: Vec<CudaSlice<f32>>, // NorMuon EMA buffers
    // ve_gate: [N_KV_HEAD, VE_GATE_CH] per VE layer, treated with Muon (matching Python)
    pub ve_gate_momentum: Vec<CudaSlice<f32>>,        // f32, VE_LAYERS.len() entries
    pub ve_gate_second_momentum: Vec<CudaSlice<f32>>, // VE_LAYERS.len() entries (reduce_cols=0, size=VE_GATE_CH)
    // MoE expert matrices: [MLP_DIM_E, D_MODEL], N_EXPERTS*2 per MoE layer
    pub moe_expert_momentum: Vec<CudaSlice<f32>>,
    pub moe_expert_second_momentum: Vec<CudaSlice<f32>>,
    // Pre-allocated scratch for frob_normalize (1 float) and normuon_step (num_groups + 2 floats).
    // Max num_groups = max(MLP_DIM, D_MODEL). Allocated as MLP_DIM + 3.
    pub scratch: CudaSlice<f32>,
}

/// All GPU memory for training, allocated once at init.
pub struct BufferManager {
    pub stream: Arc<CudaStream>,
    pub batch_size: usize,
    /// Pipeline stage: which global layers this stage owns. Default: 0..N_LAYER.
    pub layer_start: usize,
    pub layer_end: usize,
    /// True if this stage owns the embedding layer (first stage).
    pub has_embedding: bool,
    /// True if this stage owns the lm_head + loss (last stage).
    pub has_head: bool,

    // ── Weights (bf16) ──
    pub wte: CudaSlice<bf16>,              // [VOCAB, D_MODEL]
    pub lm_head: CudaSlice<bf16>,          // [VOCAB, D_MODEL]
    pub lm_head_f32: CudaSlice<f32>,       // f32 master for AdamW (Python keeps lm_head in f32)
    pub resid_lambdas: CudaSlice<bf16>,    // [N_LAYER]
    pub x0_lambdas: CudaSlice<bf16>,       // [N_LAYER]
    pub layer_weights: Vec<LayerWeights>,

    // ── MoE weights + grads ──
    pub moe_weights: Vec<MoeLayerWeights>,   // N_LAYER entries (all layers are MoE)
    pub moe_grads: Vec<MoeLayerGrads>,       // N_LAYER entries

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
    pub logits: CudaSlice<bf16>,   // [CE_CHUNK, VOCAB]
    pub loss: CudaSlice<f32>,      // [1]

    // ── MoE scratch (reused across layers) ──
    pub moe_router_logits: CudaSlice<bf16>,     // [BT, N_EXPERTS]
    pub moe_router_probs: CudaSlice<f32>,       // [BT, N_EXPERTS]
    pub moe_gate_values: CudaSlice<f32>,        // [BT, TOP_K]
    pub moe_expert_indices: CudaSlice<i32>,     // [BT, TOP_K]
    pub moe_token_perm: CudaSlice<i32>,         // [BT * TOP_K]
    pub moe_expert_counts: CudaSlice<i32>,      // [N_EXPERTS]
    pub moe_expert_offsets: CudaSlice<i32>,     // [N_EXPERTS + 1]
    pub moe_expert_offsets_host: Vec<i32>,       // pinned host copy

    // ── Saved for backward (MoE, per-layer) ──
    pub saved_router_probs: Vec<CudaSlice<f32>>,    // N_LAYER × [BT, N_EXPERTS]
    pub saved_gate_values: Vec<CudaSlice<f32>>,     // N_LAYER × [BT, TOP_K]
    pub saved_expert_indices: Vec<CudaSlice<i32>>,  // N_LAYER × [BT, TOP_K]
    pub saved_token_perm: Vec<CudaSlice<i32>>,      // N_LAYER × [BT * TOP_K]
    pub saved_expert_offsets: Vec<CudaSlice<i32>>,  // N_LAYER × [N_EXPERTS + 1]

    // ── Saved for backward (per-layer) ──
    pub saved_x_pre_attn_norm: Vec<CudaSlice<bf16>>,  // [N_LAYER] x [B, T, D_MODEL]
    pub saved_x_pre_mlp_norm: Vec<CudaSlice<bf16>>,   // [N_LAYER] x [B, T, D_MODEL]

    pub saved_xn: Vec<CudaSlice<bf16>>,                // [N_LAYER] x [B*T, D_MODEL]  normed (for dW)
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
    pub d_logits: CudaSlice<bf16>, // [CE_CHUNK, VOCAB]
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

        let mlp_e = MLP_DIM_E;
        let n_exp = N_EXPERTS;

        let mut layer_weights = Vec::with_capacity(N_LAYER);
        let mut moe_weights_vec = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            let ve = has_ve(i);
            let moe = is_moe_layer(i);
            layer_weights.push(LayerWeights {
                wq: alloc_bf16(&stream, d * d)?,
                wk: alloc_bf16(&stream, d * d)?,
                wv: alloc_bf16(&stream, d * d)?,
                wqkv: alloc_bf16(&stream, 3 * d * d)?,
                wo: alloc_bf16(&stream, d * d)?,
                wfc: if !moe { Some(alloc_bf16(&stream, mlp * d)?) } else { None },
                wdn: if !moe { Some(alloc_bf16(&stream, d * mlp)?) } else { None },
                ve_weight: if ve { Some(alloc_bf16(&stream, VOCAB * d)?) } else { None },
                ve_gate: if ve { Some(alloc_bf16(&stream, N_KV_HEAD * VE_GATE_CH)?) } else { None },
                wq_f32: alloc_f32(&stream, d * d)?,
                wk_f32: alloc_f32(&stream, d * d)?,
                wv_f32: alloc_f32(&stream, d * d)?,
                wo_f32: alloc_f32(&stream, d * d)?,
                wfc_f32: if !moe { Some(alloc_f32(&stream, mlp * d)?) } else { None },
                wdn_f32: if !moe { Some(alloc_f32(&stream, d * mlp)?) } else { None },
                ve_gate_f32: if ve { Some(alloc_f32(&stream, N_KV_HEAD * VE_GATE_CH)?) } else { None },
            });
            if moe {
                moe_weights_vec.push(MoeLayerWeights {
                    w_router: alloc_bf16(&stream, n_exp * d)?,
                    expert_wfc: alloc_bf16(&stream, n_exp * mlp_e * d)?,
                    expert_wdn: alloc_bf16(&stream, n_exp * d * mlp_e)?,
                    w_router_f32: alloc_f32(&stream, n_exp * d)?,
                    expert_wfc_f32: alloc_f32(&stream, n_exp * mlp_e * d)?,
                    expert_wdn_f32: alloc_f32(&stream, n_exp * d * mlp_e)?,
                });
            }
        }

        // ── Gradients ──
        let wte_grad = alloc_bf16(&stream, VOCAB * d)?;
        let lm_head_grad = alloc_bf16(&stream, VOCAB * d)?;
        let resid_lambdas_grad = alloc_f32(&stream, N_LAYER)?;
        let x0_lambdas_grad = alloc_f32(&stream, N_LAYER)?;

        let mut layer_grads = Vec::with_capacity(N_LAYER);
        let mut moe_grads_vec = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            let ve = has_ve(i);
            let moe = is_moe_layer(i);
            layer_grads.push(LayerGrads {
                wq: alloc_bf16(&stream, d * d)?,
                wk: alloc_bf16(&stream, d * d)?,
                wv: alloc_bf16(&stream, d * d)?,
                wqkv: alloc_bf16(&stream, 3 * d * d)?,
                wo: alloc_bf16(&stream, d * d)?,
                wfc: if !moe { Some(alloc_bf16(&stream, mlp * d)?) } else { None },
                wdn: if !moe { Some(alloc_bf16(&stream, d * mlp)?) } else { None },
                ve_weight: if ve { Some(alloc_bf16(&stream, VOCAB * d)?) } else { None },
                ve_gate: if ve { Some(alloc_bf16(&stream, N_KV_HEAD * VE_GATE_CH)?) } else { None },
            });
            if moe {
                moe_grads_vec.push(MoeLayerGrads {
                    w_router: alloc_bf16(&stream, n_exp * d)?,
                    expert_wfc: alloc_bf16(&stream, n_exp * mlp_e * d)?,
                    expert_wdn: alloc_bf16(&stream, n_exp * d * mlp_e)?,
                });
            }
        }

        // ── Optimizer state ──
        let mut ve_exp_avg = Vec::with_capacity(VE_LAYERS.len());
        let mut ve_exp_avg_sq = Vec::with_capacity(VE_LAYERS.len());
        for _ in &VE_LAYERS {
            ve_exp_avg.push(alloc_bf16(&stream, VOCAB * d)?);
            ve_exp_avg_sq.push(alloc_bf16(&stream, VOCAB * d)?);
        }

        let n_moe_layers_opt = (0..N_LAYER).filter(|i| is_moe_layer(*i)).count();
        let mut moe_router_exp_avg = Vec::with_capacity(n_moe_layers_opt);
        let mut moe_router_exp_avg_sq = Vec::with_capacity(n_moe_layers_opt);
        for _ in 0..n_moe_layers_opt {
            moe_router_exp_avg.push(alloc_f32(&stream, n_exp * d)?);
            moe_router_exp_avg_sq.push(alloc_f32(&stream, n_exp * d)?);
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
            moe_router_exp_avg,
            moe_router_exp_avg_sq,
        };

        // Muon: one momentum buffer per block matrix weight
        // f32 momentum to match Python (nn.Linear params are f32, momentum_buffer is f32)
        let mut momentum = Vec::new();
        let mut second_momentum = Vec::new();
        for i in 0..N_LAYER {
            let moe = is_moe_layer(i);
            // wq, wk, wv, wo: [D_MODEL, D_MODEL] -> reduce_cols=1, second_mom size = D_MODEL
            for _ in 0..4 {
                momentum.push(alloc_f32(&stream, d * d)?);
                second_momentum.push(alloc_f32(&stream, d)?);
            }
            if !moe {
                // Dense MLP: wfc [MLP_DIM, D_MODEL], wdn [D_MODEL, MLP_DIM]
                momentum.push(alloc_f32(&stream, mlp * d)?);
                second_momentum.push(alloc_f32(&stream, mlp)?);
                momentum.push(alloc_f32(&stream, d * mlp)?);
                second_momentum.push(alloc_f32(&stream, mlp)?);
            }
        }
        // MoE expert Muon: expert_wfc and expert_wdn are [MLP_DIM_E, D_MODEL]
        // N_EXPERTS * 2 matrices per MoE layer
        let n_moe_layers = (0..N_LAYER).filter(|i| is_moe_layer(*i)).count();
        let mut moe_expert_momentum = Vec::with_capacity(n_moe_layers * n_exp * 2);
        let mut moe_expert_second_momentum = Vec::with_capacity(n_moe_layers * n_exp * 2);
        for _ in 0..n_moe_layers {
            for _ in 0..(n_exp * 2) {
                // [MLP_DIM_E, D_MODEL]
                moe_expert_momentum.push(alloc_f32(&stream, mlp_e * d)?);
                moe_expert_second_momentum.push(alloc_f32(&stream, mlp_e)?);
            }
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
        let muon_scratch = alloc_f32(&stream, MLP_DIM.max(MLP_DIM_E) + 3)?;
        let muon = MuonState {
            momentum, second_momentum,
            ve_gate_momentum, ve_gate_second_momentum,
            moe_expert_momentum, moe_expert_second_momentum,
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
        let mlp_buf = mlp.max(mlp_e);
        // h: gathered tokens [n_dispatch, D] — n_dispatch = BT*TOP_K worst case
        let h = alloc_bf16(&stream, (bt * TOP_K * d).max(bt * mlp))?;
        // h_act: expert intermediates 2 * n_e * MLP_DIM_E; worst case n_e = BT*TOP_K
        let h_act = alloc_bf16(&stream, 2 * bt * TOP_K * mlp_e.max(mlp))?;
        let logits = alloc_bf16(&stream, CE_CHUNK * VOCAB)?;
        let loss = alloc_f32(&stream, 1)?;

        // ── MoE scratch (reused across layers) ──
        let moe_router_logits = alloc_bf16(&stream, bt * N_EXPERTS)?;
        let moe_router_probs = alloc_f32(&stream, bt * N_EXPERTS)?;
        let moe_gate_values = alloc_f32(&stream, bt * TOP_K)?;
        let moe_expert_indices = alloc::<i32>(&stream, bt * TOP_K)?;
        let moe_token_perm = alloc::<i32>(&stream, bt * TOP_K)?;
        let moe_expert_counts = alloc::<i32>(&stream, N_EXPERTS)?;
        let moe_expert_offsets = alloc::<i32>(&stream, N_EXPERTS + 1)?;
        let moe_expert_offsets_host = vec![0i32; N_EXPERTS + 1];

        // ── Saved for backward (MoE, per-layer) ──
        let mut saved_router_probs = Vec::with_capacity(N_LAYER);
        let mut saved_gate_values = Vec::with_capacity(N_LAYER);
        let mut saved_expert_indices = Vec::with_capacity(N_LAYER);
        let mut saved_token_perm = Vec::with_capacity(N_LAYER);
        let mut saved_expert_offsets = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            if is_moe_layer(i) {
                saved_router_probs.push(alloc_f32(&stream, bt * N_EXPERTS)?);
                saved_gate_values.push(alloc_f32(&stream, bt * TOP_K)?);
                saved_expert_indices.push(alloc::<i32>(&stream, bt * TOP_K)?);
                saved_token_perm.push(alloc::<i32>(&stream, bt * TOP_K)?);
                saved_expert_offsets.push(alloc::<i32>(&stream, N_EXPERTS + 1)?);
            }
        }

        // ── Saved for backward ──
        let mut saved_x_pre_attn_norm = Vec::with_capacity(N_LAYER);
        let mut saved_x_pre_mlp_norm = Vec::with_capacity(N_LAYER);

        let mut saved_xn = Vec::with_capacity(N_LAYER);
        let mut saved_v = Vec::with_capacity(N_LAYER);
        let mut saved_attn_out = Vec::with_capacity(N_LAYER);
        for _ in 0..N_LAYER {
            saved_x_pre_attn_norm.push(alloc_bf16(&stream, bt * d)?);
            saved_x_pre_mlp_norm.push(alloc_bf16(&stream, bt * d)?);

            saved_xn.push(alloc_bf16(&stream, bt * d)?);
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
        let d_h = alloc_bf16(&stream, 2 * bt * TOP_K * mlp_e.max(mlp))?;
        let d_logits = alloc_bf16(&stream, CE_CHUNK * VOCAB)?;
        let d_xn = alloc_bf16(&stream, bt * d)?;

        // ── Flash attention backward scratch (f32) ──
        let flash_dq_accum = alloc_f32(&stream, b * N_HEAD * t * HEAD_DIM)?;
        let flash_dsoftmax_sum = alloc_f32(&stream, b * N_HEAD * t)?;
        // FA3-specific: softmax_lse_log2 [B, N_HEAD, SEQ_ROUNDED] and dq_semaphore [ceil(SEQ/kBlockM), B, N_HEAD]
        let fa3_softmax_lse_log2 = alloc_f32(&stream, b * N_HEAD * t)?;
        let dq_sem_blocks = (t + 63) / 64; // kBlockM = 64 for hdim128 causal bwd on sm90
        let fa3_dq_semaphore = alloc::<i32>(&stream, dq_sem_blocks * b * N_HEAD)?;
        // FA3 scheduler metadata: round_up(B,4)*4 + 1 ints (tile_count_semaphore + metadata vectors)
        let fa3_scheduler_meta = alloc::<i32>(&stream, 32768)?; // generous for T=32768

        // ── Fixed buffers ──
        let cos = alloc_bf16(&stream, t * (HEAD_DIM / 2))?;
        let sin = alloc_bf16(&stream, t * (HEAD_DIM / 2))?;
        let input_ids = alloc_u32(&stream, bt)?;
        let targets = alloc_u32(&stream, bt)?;
        let input_ids_b = alloc_u32(&stream, bt)?;
        let targets_b = alloc_u32(&stream, bt)?;

        Ok(Self {
            stream,
            batch_size: b,
            layer_start: 0,
            layer_end: N_LAYER,
            has_embedding: true,
            has_head: true,
            wte,
            lm_head,
            lm_head_f32,
            resid_lambdas,
            x0_lambdas,
            layer_weights,
            moe_weights: moe_weights_vec,
            moe_grads: moe_grads_vec,
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
            moe_router_logits,
            moe_router_probs,
            moe_gate_values,
            moe_expert_indices,
            moe_token_perm,
            moe_expert_counts,
            moe_expert_offsets,
            moe_expert_offsets_host,
            saved_router_probs,
            saved_gate_values,
            saved_expert_indices,
            saved_token_perm,
            saved_expert_offsets,
            saved_x_pre_attn_norm,
            saved_x_pre_mlp_norm,

            saved_xn,
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
            if let Some(ref mut w) = lg.wfc { zero(s, w)?; }
            if let Some(ref mut w) = lg.wdn { zero(s, w)?; }
            if let Some(ref mut ve) = lg.ve_weight {
                zero(s, ve)?;
            }
            if let Some(ref mut g) = lg.ve_gate {
                zero(s, g)?;
            }
        }
        for mg in &mut self.moe_grads {
            zero(s, &mut mg.w_router)?;
            zero(s, &mut mg.expert_wfc)?;
            zero(s, &mut mg.expert_wdn)?;
        }
        Ok(())
    }

    /// Pack wq/wk/wv into wqkv = [wq; wk; wv] for all layers.
    /// Must be called after loading/initializing weights and after each optimizer step.
    pub fn pack_wqkv(&self) {
        let dd_bytes = D_MODEL * D_MODEL * std::mem::size_of::<bf16>();
        let stream_ptr = self.stream.cu_stream();
        for layer in self.layer_start..self.layer_end {
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

    /// Allocate GPU memory for a pipeline stage that owns layers [layer_start, layer_end).
    /// Non-owned layers get 1-element dummy allocations. Embedding/head are conditional.
    pub fn new_staged(
        stream: Arc<CudaStream>,
        batch_size: usize,
        layer_start: usize,
        layer_end: usize,
        has_embedding: bool,
        has_head: bool,
    ) -> Result<Self> {
        let b = batch_size;
        let t = SEQ;
        let bt = b * t;
        let d = D_MODEL;
        let mlp = MLP_DIM;
        let mlp_e = MLP_DIM_E;
        let n_exp = N_EXPERTS;

        let owns = |i: usize| i >= layer_start && i < layer_end;

        // ── Weights ──
        let embed_sz = if has_embedding { VOCAB * d } else { 1 };
        let head_sz = if has_head { VOCAB * d } else { 1 };
        let wte = alloc_bf16(&stream, embed_sz)?;
        let lm_head = alloc_bf16(&stream, head_sz)?;
        let lm_head_f32 = alloc_f32(&stream, head_sz)?;
        let resid_lambdas = alloc_bf16(&stream, N_LAYER)?; // small, always full
        let x0_lambdas = alloc_bf16(&stream, N_LAYER)?;

        let mut layer_weights = Vec::with_capacity(N_LAYER);
        let mut moe_weights_vec = Vec::with_capacity(layer_end - layer_start);
        for i in 0..N_LAYER {
            let own = owns(i);
            let ve = has_ve(i) && own;
            let moe = is_moe_layer(i) && own;
            let sz = if own { d * d } else { 1 };
            let mlp_sz = if own { mlp * d } else { 1 };
            let ve_sz = if ve { VOCAB * d } else { 1 };
            let ve_gate_sz = if ve { N_KV_HEAD * VE_GATE_CH } else { 1 };
            layer_weights.push(LayerWeights {
                wq: alloc_bf16(&stream, sz)?,
                wk: alloc_bf16(&stream, sz)?,
                wv: alloc_bf16(&stream, sz)?,
                wqkv: alloc_bf16(&stream, if own { 3 * d * d } else { 1 })?,
                wo: alloc_bf16(&stream, sz)?,
                wfc: if is_moe_layer(i) { None } else if own { Some(alloc_bf16(&stream, mlp_sz)?) } else { None },
                wdn: if is_moe_layer(i) { None } else if own { Some(alloc_bf16(&stream, mlp_sz)?) } else { None },
                ve_weight: if ve { Some(alloc_bf16(&stream, ve_sz)?) } else { None },
                ve_gate: if ve { Some(alloc_bf16(&stream, ve_gate_sz)?) } else { None },
                wq_f32: alloc_f32(&stream, sz)?,
                wk_f32: alloc_f32(&stream, sz)?,
                wv_f32: alloc_f32(&stream, sz)?,
                wo_f32: alloc_f32(&stream, sz)?,
                wfc_f32: if is_moe_layer(i) { None } else if own { Some(alloc_f32(&stream, mlp_sz)?) } else { None },
                wdn_f32: if is_moe_layer(i) { None } else if own { Some(alloc_f32(&stream, mlp_sz)?) } else { None },
                ve_gate_f32: if ve { Some(alloc_f32(&stream, ve_gate_sz)?) } else { None },
            });
            if moe {
                moe_weights_vec.push(MoeLayerWeights {
                    w_router: alloc_bf16(&stream, n_exp * d)?,
                    expert_wfc: alloc_bf16(&stream, n_exp * mlp_e * d)?,
                    expert_wdn: alloc_bf16(&stream, n_exp * d * mlp_e)?,
                    w_router_f32: alloc_f32(&stream, n_exp * d)?,
                    expert_wfc_f32: alloc_f32(&stream, n_exp * mlp_e * d)?,
                    expert_wdn_f32: alloc_f32(&stream, n_exp * d * mlp_e)?,
                });
            }
        }

        // ── Gradients ──
        let wte_grad = alloc_bf16(&stream, embed_sz)?;
        let lm_head_grad = alloc_bf16(&stream, head_sz)?;
        let resid_lambdas_grad = alloc_f32(&stream, N_LAYER)?;
        let x0_lambdas_grad = alloc_f32(&stream, N_LAYER)?;

        let mut layer_grads = Vec::with_capacity(N_LAYER);
        let mut moe_grads_vec = Vec::with_capacity(layer_end - layer_start);
        for i in 0..N_LAYER {
            let own = owns(i);
            let ve = has_ve(i) && own;
            let moe = is_moe_layer(i) && own;
            let sz = if own { d * d } else { 1 };
            layer_grads.push(LayerGrads {
                wq: alloc_bf16(&stream, sz)?,
                wk: alloc_bf16(&stream, sz)?,
                wv: alloc_bf16(&stream, sz)?,
                wqkv: alloc_bf16(&stream, if own { 3 * d * d } else { 1 })?,
                wo: alloc_bf16(&stream, sz)?,
                wfc: if is_moe_layer(i) { None } else if own { Some(alloc_bf16(&stream, if own { mlp * d } else { 1 })?) } else { None },
                wdn: if is_moe_layer(i) { None } else if own { Some(alloc_bf16(&stream, if own { d * mlp } else { 1 })?) } else { None },
                ve_weight: if ve { Some(alloc_bf16(&stream, VOCAB * d)?) } else { None },
                ve_gate: if ve { Some(alloc_bf16(&stream, N_KV_HEAD * VE_GATE_CH)?) } else { None },
            });
            if moe {
                moe_grads_vec.push(MoeLayerGrads {
                    w_router: alloc_bf16(&stream, n_exp * d)?,
                    expert_wfc: alloc_bf16(&stream, n_exp * mlp_e * d)?,
                    expert_wdn: alloc_bf16(&stream, n_exp * d * mlp_e)?,
                });
            }
        }

        // ── Optimizer state ──
        let n_owned_ve = VE_LAYERS.iter().filter(|&&l| owns(l)).count();
        let mut ve_exp_avg = Vec::with_capacity(n_owned_ve);
        let mut ve_exp_avg_sq = Vec::with_capacity(n_owned_ve);
        for &l in &VE_LAYERS {
            if owns(l) {
                ve_exp_avg.push(alloc_bf16(&stream, VOCAB * d)?);
                ve_exp_avg_sq.push(alloc_bf16(&stream, VOCAB * d)?);
            }
        }

        let embed_opt_sz = if has_embedding { VOCAB * d } else { 1 };
        let head_opt_sz = if has_head { VOCAB * d } else { 1 };
        let n_owned_moe = (layer_start..layer_end).filter(|i| is_moe_layer(*i)).count();

        let mut moe_router_exp_avg_a = Vec::with_capacity(n_owned_moe);
        let mut moe_router_exp_avg_sq_a = Vec::with_capacity(n_owned_moe);
        for i in layer_start..layer_end {
            if is_moe_layer(i) {
                moe_router_exp_avg_a.push(alloc_f32(&stream, n_exp * d)?);
                moe_router_exp_avg_sq_a.push(alloc_f32(&stream, n_exp * d)?);
            }
        }

        let adamw = AdamWState {
            wte_exp_avg: alloc_bf16(&stream, embed_opt_sz)?,
            wte_exp_avg_sq: alloc_bf16(&stream, embed_opt_sz)?,
            lm_head_exp_avg: alloc_f32(&stream, head_opt_sz)?,
            lm_head_exp_avg_sq: alloc_f32(&stream, head_opt_sz)?,
            resid_lambdas_exp_avg: alloc_f32(&stream, N_LAYER)?,
            resid_lambdas_exp_avg_sq: alloc_f32(&stream, N_LAYER)?,
            x0_lambdas_exp_avg: alloc_f32(&stream, N_LAYER)?,
            x0_lambdas_exp_avg_sq: alloc_f32(&stream, N_LAYER)?,
            ve_exp_avg,
            ve_exp_avg_sq,
            moe_router_exp_avg: moe_router_exp_avg_a,
            moe_router_exp_avg_sq: moe_router_exp_avg_sq_a,
        };

        // Muon momentum (only for owned layers)
        let mut momentum = Vec::new();
        let mut second_momentum = Vec::new();
        for i in 0..N_LAYER {
            let own = owns(i);
            let moe = is_moe_layer(i);
            let attn_sz = if own { d * d } else { 1 };
            let attn_sm = if own { d } else { 1 };
            for _ in 0..4 {
                momentum.push(alloc_f32(&stream, attn_sz)?);
                second_momentum.push(alloc_f32(&stream, attn_sm)?);
            }
            if !moe && own {
                momentum.push(alloc_f32(&stream, mlp * d)?);
                second_momentum.push(alloc_f32(&stream, mlp)?);
                momentum.push(alloc_f32(&stream, d * mlp)?);
                second_momentum.push(alloc_f32(&stream, mlp)?);
            } else if !moe {
                // Non-owned dense layer: dummy
                for _ in 0..2 {
                    momentum.push(alloc_f32(&stream, 1)?);
                    second_momentum.push(alloc_f32(&stream, 1)?);
                }
            }
        }

        let mut moe_expert_momentum = Vec::with_capacity(n_owned_moe * n_exp * 2);
        let mut moe_expert_second_momentum = Vec::with_capacity(n_owned_moe * n_exp * 2);
        for i in 0..N_LAYER {
            if is_moe_layer(i) && owns(i) {
                for _ in 0..(n_exp * 2) {
                    moe_expert_momentum.push(alloc_f32(&stream, mlp_e * d)?);
                    moe_expert_second_momentum.push(alloc_f32(&stream, mlp_e)?);
                }
            }
        }

        let mut ve_gate_momentum = Vec::with_capacity(n_owned_ve);
        let mut ve_gate_second_momentum = Vec::with_capacity(n_owned_ve);
        for &l in &VE_LAYERS {
            if owns(l) {
                ve_gate_momentum.push(alloc_f32(&stream, N_KV_HEAD * VE_GATE_CH)?);
                ve_gate_second_momentum.push(alloc_f32(&stream, VE_GATE_CH)?);
            }
        }

        let mlp_buf = mlp.max(mlp_e);
        let muon_scratch = alloc_f32(&stream, mlp_buf + 3)?;
        let muon = MuonState {
            momentum, second_momentum,
            ve_gate_momentum, ve_gate_second_momentum,
            moe_expert_momentum, moe_expert_second_momentum,
            scratch: muon_scratch,
        };

        // ── Activations (full size — reused across layers) ──
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
        let h = alloc_bf16(&stream, (bt * TOP_K * d).max(bt * mlp))?;
        let h_act = alloc_bf16(&stream, 2 * bt * TOP_K * mlp_e.max(mlp))?;
        let logits = if has_head { alloc_bf16(&stream, CE_CHUNK * VOCAB)? } else { alloc_bf16(&stream, 1)? };
        let loss = alloc_f32(&stream, 1)?;

        // ── MoE scratch ──
        let moe_router_logits = alloc_bf16(&stream, bt * N_EXPERTS)?;
        let moe_router_probs = alloc_f32(&stream, bt * N_EXPERTS)?;
        let moe_gate_values = alloc_f32(&stream, bt * TOP_K)?;
        let moe_expert_indices = alloc::<i32>(&stream, bt * TOP_K)?;
        let moe_token_perm = alloc::<i32>(&stream, bt * TOP_K)?;
        let moe_expert_counts = alloc::<i32>(&stream, N_EXPERTS)?;
        let moe_expert_offsets = alloc::<i32>(&stream, N_EXPERTS + 1)?;
        let moe_expert_offsets_host = vec![0i32; N_EXPERTS + 1];

        // ── Saved for backward (only for owned layers) ──
        let n_owned = layer_end - layer_start;
        let mut saved_router_probs = Vec::with_capacity(n_owned);
        let mut saved_gate_values = Vec::with_capacity(n_owned);
        let mut saved_expert_indices = Vec::with_capacity(n_owned);
        let mut saved_token_perm = Vec::with_capacity(n_owned);
        let mut saved_expert_offsets = Vec::with_capacity(n_owned);
        let mut saved_x_pre_attn_norm = Vec::with_capacity(N_LAYER);
        let mut saved_x_pre_mlp_norm = Vec::with_capacity(N_LAYER);
        let mut saved_xn = Vec::with_capacity(N_LAYER);
        let mut saved_v = Vec::with_capacity(N_LAYER);
        let mut saved_attn_out = Vec::with_capacity(N_LAYER);
        let mut saved_softmax_lse = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            let own = owns(i);
            let sz = if own { bt * d } else { 1 };
            saved_x_pre_attn_norm.push(alloc_bf16(&stream, sz)?);
            saved_x_pre_mlp_norm.push(alloc_bf16(&stream, sz)?);
            saved_xn.push(alloc_bf16(&stream, sz)?);
            saved_v.push(alloc_bf16(&stream, sz)?);
            saved_attn_out.push(alloc_bf16(&stream, sz)?);
            saved_softmax_lse.push(alloc_f32(&stream, if own { b * N_HEAD * t } else { 1 })?);
            if is_moe_layer(i) && own {
                saved_router_probs.push(alloc_f32(&stream, bt * N_EXPERTS)?);
                saved_gate_values.push(alloc_f32(&stream, bt * TOP_K)?);
                saved_expert_indices.push(alloc::<i32>(&stream, bt * TOP_K)?);
                saved_token_perm.push(alloc::<i32>(&stream, bt * TOP_K)?);
                saved_expert_offsets.push(alloc::<i32>(&stream, N_EXPERTS + 1)?);
            }
        }

        // ── Backward scratch ──
        let d_x = alloc_bf16(&stream, bt * d)?;
        let d_x0 = alloc_bf16(&stream, bt * d)?;
        let d_q = alloc_bf16(&stream, bt * d)?;
        let d_k = alloc_bf16(&stream, bt * d)?;
        let d_v_buf = alloc_bf16(&stream, bt * d)?;
        let d_qkv = alloc_bf16(&stream, 3 * bt * d)?;
        let d_h = alloc_bf16(&stream, 2 * bt * TOP_K * mlp_e.max(mlp))?;
        let d_logits = if has_head { alloc_bf16(&stream, CE_CHUNK * VOCAB)? } else { alloc_bf16(&stream, 1)? };
        let d_xn = alloc_bf16(&stream, bt * d)?;
        let flash_dq_accum = alloc_f32(&stream, b * N_HEAD * t * HEAD_DIM)?;
        let flash_dsoftmax_sum = alloc_f32(&stream, b * N_HEAD * t)?;
        let fa3_softmax_lse_log2 = alloc_f32(&stream, b * N_HEAD * t)?;
        let dq_sem_blocks = (t + 63) / 64; // kBlockM = 64 for hdim128 causal bwd on sm90
        let fa3_dq_semaphore = alloc::<i32>(&stream, dq_sem_blocks * b * N_HEAD)?;
        let fa3_scheduler_meta = alloc::<i32>(&stream, 32768)?;

        // ── Fixed ──
        let cos = alloc_bf16(&stream, t * (HEAD_DIM / 2))?;
        let sin = alloc_bf16(&stream, t * (HEAD_DIM / 2))?;
        let input_ids = alloc_u32(&stream, bt)?;
        let targets = alloc_u32(&stream, bt)?;
        let input_ids_b = alloc_u32(&stream, bt)?;
        let targets_b = alloc_u32(&stream, bt)?;

        Ok(Self {
            stream,
            batch_size: b,
            layer_start,
            layer_end,
            has_embedding,
            has_head,
            wte, lm_head, lm_head_f32, resid_lambdas, x0_lambdas,
            layer_weights, moe_weights: moe_weights_vec, moe_grads: moe_grads_vec,
            wte_grad, lm_head_grad, resid_lambdas_grad, x0_lambdas_grad, layer_grads,
            adamw, muon,
            emb, x, x0, xn, q, k, v, qkv, ve: ve_buf, gate, attn_out, h, h_act, logits, loss,
            moe_router_logits, moe_router_probs, moe_gate_values, moe_expert_indices,
            moe_token_perm, moe_expert_counts, moe_expert_offsets, moe_expert_offsets_host,
            saved_router_probs, saved_gate_values, saved_expert_indices, saved_token_perm, saved_expert_offsets,
            saved_x_pre_attn_norm, saved_x_pre_mlp_norm, saved_xn, saved_v, saved_attn_out, saved_softmax_lse,
            d_x, d_x0, d_q, d_k, d_v: d_v_buf, d_qkv, d_h, d_logits, d_xn,
            flash_dq_accum, flash_dsoftmax_sum, fa3_softmax_lse_log2, fa3_dq_semaphore, fa3_scheduler_meta,
            cos, sin, input_ids, targets, input_ids_b, targets_b,
        })
    }

    /// Check if this stage owns a given global layer index.
    pub fn owns_layer(&self, layer: usize) -> bool {
        layer >= self.layer_start && layer < self.layer_end
    }

    /// Raw device pointer to the `x` activation buffer (for P2P send).
    pub fn x_device_ptr(&self) -> u64 {
        let (ptr, _) = self.x.device_ptr(self.x.stream());
        ptr
    }

    /// Raw device pointer to the `x` activation buffer (for P2P recv).
    pub fn x_device_ptr_mut(&self) -> u64 {
        let (ptr, _) = self.x.device_ptr(self.x.stream());
        ptr
    }

    /// Raw device pointer to the `d_x` gradient buffer (for P2P backward send).
    pub fn dx_device_ptr(&self) -> u64 {
        let (ptr, _) = self.d_x.device_ptr(self.d_x.stream());
        ptr
    }

    /// Raw device pointer to the `d_x` gradient buffer (for P2P backward recv).
    pub fn dx_device_ptr_mut(&self) -> u64 {
        let (ptr, _) = self.d_x.device_ptr(self.d_x.stream());
        ptr
    }

    /// Raw device pointer to the `x0` activation buffer (for P2P send).
    pub fn x0_device_ptr(&self) -> u64 {
        let (ptr, _) = self.x0.device_ptr(self.x0.stream());
        ptr
    }

    /// Raw device pointer to the `x0` activation buffer (for P2P recv).
    pub fn x0_device_ptr_mut(&self) -> u64 {
        let (ptr, _) = self.x0.device_ptr(self.x0.stream());
        ptr
    }

    /// Raw device pointer to the `input_ids` buffer.
    pub fn input_ids_device_ptr(&self) -> u64 {
        let (ptr, _) = self.input_ids.device_ptr(self.input_ids.stream());
        ptr
    }

    /// Raw device pointer to the `targets` buffer.
    pub fn targets_device_ptr(&self) -> u64 {
        let (ptr, _) = self.targets.device_ptr(self.targets.stream());
        ptr
    }

    /// Raw device pointer to the `loss` buffer.
    pub fn loss_device_ptr(&self) -> u64 {
        let (ptr, _) = self.loss.device_ptr(self.loss.stream());
        ptr
    }

    /// Total bytes allocated across all buffers.
    pub fn total_bytes(&self) -> usize {
        let b = self.batch_size;
        let t = SEQ;
        let bt = b * t;
        let d = D_MODEL;
        let mlp = MLP_DIM;
        let mlp_e = MLP_DIM_E;
        let n_exp = N_EXPERTS;
        let bf16_sz = std::mem::size_of::<bf16>();
        let f32_sz = std::mem::size_of::<f32>();
        let i32_sz = std::mem::size_of::<i32>();
        let u32_sz = std::mem::size_of::<u32>();

        let owns = |i: usize| i >= self.layer_start && i < self.layer_end;
        let n_owned = self.layer_end - self.layer_start;
        let n_non_owned = N_LAYER - n_owned;
        let n_owned_moe = (self.layer_start..self.layer_end).filter(|i| is_moe_layer(*i)).count();
        let n_owned_dense = n_owned - n_owned_moe;
        let n_non_owned_moe = (0..N_LAYER).filter(|i| is_moe_layer(*i) && !owns(*i)).count();
        let n_non_owned_dense = n_non_owned - n_non_owned_moe;
        let n_owned_ve = VE_LAYERS.iter().filter(|&&l| owns(l)).count();

        let embed_sz = if self.has_embedding { VOCAB * d } else { 1 };
        let head_sz = if self.has_head { VOCAB * d } else { 1 };

        let mut total: usize = 0;

        // ── Weights ──
        total += embed_sz * bf16_sz;                  // wte
        total += head_sz * bf16_sz;                   // lm_head
        total += head_sz * f32_sz;                    // lm_head_f32
        total += 2 * N_LAYER * bf16_sz;              // resid_lambdas + x0_lambdas (always full)
        // Per-layer attention weights: owned get full size, non-owned get 1-element dummies
        // wq, wk, wv, wo each d*d; wqkv 3*d*d; plus f32 copies of wq/wk/wv/wo
        total += n_owned * (4 * d * d + 3 * d * d) * bf16_sz;
        total += n_non_owned * (4 + 1) * bf16_sz;    // 4x 1-elem + 1x 1-elem wqkv dummies
        total += n_owned * 4 * d * d * f32_sz;       // wq/wk/wv/wo _f32
        total += n_non_owned * 4 * f32_sz;            // 4x 1-elem f32 dummies
        // Dense MLP weights: only owned dense layers get full alloc; non-owned dense get None (0)
        // wfc, wdn are Option: Some for owned dense, None for MoE or non-owned
        total += n_owned_dense * 2 * mlp * d * bf16_sz;
        total += n_owned_dense * 2 * mlp * d * f32_sz;
        // VE weights: only owned VE layers
        total += n_owned_ve * VOCAB * d * bf16_sz;    // ve_weight
        total += n_owned_ve * N_KV_HEAD * VE_GATE_CH * bf16_sz; // ve_gate
        total += n_owned_ve * N_KV_HEAD * VE_GATE_CH * f32_sz;  // ve_gate_f32
        // MoE weights: only owned MoE layers
        total += n_owned_moe * n_exp * d * bf16_sz;              // w_router
        total += n_owned_moe * 2 * n_exp * mlp_e * d * bf16_sz; // expert_wfc + expert_wdn
        total += n_owned_moe * n_exp * d * f32_sz;               // w_router_f32
        total += n_owned_moe * 2 * n_exp * mlp_e * d * f32_sz;  // expert_wfc/wdn _f32

        // ── Gradients ──
        total += embed_sz * bf16_sz;                  // wte_grad
        total += head_sz * bf16_sz;                   // lm_head_grad
        total += 2 * N_LAYER * f32_sz;               // resid/x0 lambdas grads (always full)
        // Per-layer attention grads: same pattern as weights
        total += n_owned * (4 * d * d + 3 * d * d) * bf16_sz;
        total += n_non_owned * (4 + 1) * bf16_sz;
        // Dense MLP grads: owned dense only
        total += n_owned_dense * 2 * mlp * d * bf16_sz;
        // VE grads: owned VE only
        total += n_owned_ve * VOCAB * d * bf16_sz;
        total += n_owned_ve * N_KV_HEAD * VE_GATE_CH * bf16_sz;
        // MoE grads: owned MoE only
        total += n_owned_moe * n_exp * d * bf16_sz;
        total += n_owned_moe * 2 * n_exp * mlp_e * d * bf16_sz;

        // ── Optimizer state ──
        let embed_opt_sz = if self.has_embedding { VOCAB * d } else { 1 };
        let head_opt_sz = if self.has_head { VOCAB * d } else { 1 };
        total += 2 * embed_opt_sz * bf16_sz;          // wte AdamW (exp_avg + exp_avg_sq)
        total += 2 * head_opt_sz * f32_sz;            // lm_head AdamW (f32)
        total += 4 * N_LAYER * f32_sz;               // resid/x0 lambdas AdamW (always full)
        total += n_owned_ve * 2 * VOCAB * d * bf16_sz; // ve AdamW
        // Router AdamW (f32, owned MoE only)
        total += n_owned_moe * 2 * n_exp * d * f32_sz;
        // Muon: attention momentum — owned get d*d, non-owned get 1-elem dummies
        total += n_owned * 4 * d * d * f32_sz;
        total += n_non_owned * 4 * f32_sz;
        // Muon: attention second_momentum — owned get d, non-owned get 1
        total += n_owned * 4 * d * f32_sz;
        total += n_non_owned * 4 * f32_sz;
        // Muon: dense MLP momentum — owned dense get full, non-owned dense get 1-elem dummies
        total += n_owned_dense * 2 * mlp * d * f32_sz;
        total += n_non_owned_dense * 2 * f32_sz;
        // Muon: dense MLP second_momentum
        total += n_owned_dense * 2 * mlp * f32_sz;
        total += n_non_owned_dense * 2 * f32_sz;
        // Muon: MoE expert momentum (owned MoE only, no dummies for non-owned MoE)
        total += n_owned_moe * n_exp * 2 * mlp_e * d * f32_sz;
        total += n_owned_moe * n_exp * 2 * d * f32_sz;
        // Muon: ve_gate (owned VE only)
        total += n_owned_ve * N_KV_HEAD * VE_GATE_CH * f32_sz;
        total += n_owned_ve * VE_GATE_CH * f32_sz;
        let mlp_buf = mlp.max(mlp_e);
        total += (mlp_buf + 3) * f32_sz;              // muon scratch

        // ── Activations (reused, always full size) ──
        total += 9 * bt * d * bf16_sz;               // emb,x,x0,xn,q,k,v,ve,attn_out
        total += 3 * bt * d * bf16_sz;               // qkv
        total += bt * N_KV_HEAD * bf16_sz;           // gate
        // h: max(bt*TOP_K*d, bt*mlp) for gathered tokens
        // h_act: 2*bt*TOP_K*max(mlp_e,mlp) for expert intermediates
        total += (bt * TOP_K * d).max(bt * mlp) * bf16_sz; // h
        total += 2 * bt * TOP_K * mlp_e.max(mlp) * bf16_sz; // h_act
        let logits_sz = if self.has_head { CE_CHUNK * VOCAB } else { 1 };
        total += logits_sz * bf16_sz;                 // logits
        total += f32_sz;                              // loss

        // ── MoE scratch (reused, always full size) ──
        total += bt * n_exp * bf16_sz;               // moe_router_logits
        total += bt * n_exp * f32_sz;                // moe_router_probs
        total += bt * TOP_K * f32_sz;                // moe_gate_values
        total += bt * TOP_K * i32_sz;                // moe_expert_indices
        total += bt * TOP_K * i32_sz;                // moe_token_perm
        total += n_exp * i32_sz;                      // moe_expert_counts
        total += (n_exp + 1) * i32_sz;               // moe_expert_offsets

        // ── Saved for backward ──
        // MoE saved: only for owned MoE layers
        total += n_owned_moe * bt * n_exp * f32_sz;   // saved_router_probs
        total += n_owned_moe * bt * TOP_K * f32_sz;   // saved_gate_values
        total += n_owned_moe * bt * TOP_K * i32_sz;   // saved_expert_indices
        total += n_owned_moe * bt * TOP_K * i32_sz;   // saved_token_perm
        total += n_owned_moe * (n_exp + 1) * i32_sz;  // saved_expert_offsets
        // Per-layer saved: owned get bt*d, non-owned get 1-elem dummies
        total += 5 * n_owned * bt * d * bf16_sz;      // x_pre_attn/mlp_norm, xn, v, attn_out
        total += 5 * n_non_owned * bf16_sz;           // 5x 1-elem dummies
        total += n_owned * b * N_HEAD * t * f32_sz;   // softmax_lse
        total += n_non_owned * f32_sz;                // 1-elem dummies

        // ── Backward scratch (always full size) ──
        total += 6 * bt * d * bf16_sz;               // d_x, d_x0, d_q, d_k, d_v, d_xn
        total += 3 * bt * d * bf16_sz;               // d_qkv
        total += 2 * bt * TOP_K * mlp_e.max(mlp) * bf16_sz; // d_h
        let d_logits_sz = if self.has_head { CE_CHUNK * VOCAB } else { 1 };
        total += d_logits_sz * bf16_sz;               // d_logits
        total += b * N_HEAD * t * HEAD_DIM * f32_sz; // flash_dq_accum
        total += b * N_HEAD * t * f32_sz;             // flash_dsoftmax_sum
        total += b * N_HEAD * t * f32_sz;             // fa3_softmax_lse_log2
        total += ((t + 63) / 64) * b * N_HEAD * u32_sz; // fa3_dq_semaphore (kBlockM=64 causal)
        total += 32768 * i32_sz;                      // fa3_scheduler_meta

        // ── Fixed ──
        total += 2 * t * (HEAD_DIM / 2) * bf16_sz;  // cos, sin
        total += 4 * bt * u32_sz;                     // input_ids, targets (x2 double buffer)

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

        // Verify saved-for-backward: 5*N_LAYER*bt*d (q/k removed — recomputed in bwd)
        let saved_elems = 5 * N_LAYER * bt * d;
        let saved_bytes = saved_elems * bf16_sz;
        let saved_mb = saved_bytes as f64 / (1024.0 * 1024.0);
        assert!(saved_mb > 0.0);
    }
}

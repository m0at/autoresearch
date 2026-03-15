use std::fs;
use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use safetensors::SafeTensors;

use crate::buffer::BufferManager;
use crate::config::*;

// ---------------------------------------------------------------------------
// Safetensors tensor name mapping
// ---------------------------------------------------------------------------
//
// This loader handles three checkpoint formats:
//
// 1. Python state_dict names (from torch model.state_dict()) — DENSE:
//    "transformer.wte.weight"                   -> bufs.wte
//    "lm_head.weight"                           -> bufs.lm_head
//    "resid_lambdas"                            -> bufs.resid_lambdas
//    "x0_lambdas"                               -> bufs.x0_lambdas
//    "transformer.h.{i}.attn.c_q.weight"        -> bufs.layer_weights[i].wq
//    "transformer.h.{i}.attn.c_k.weight"        -> bufs.layer_weights[i].wk
//    "transformer.h.{i}.attn.c_v.weight"        -> bufs.layer_weights[i].wv
//    "transformer.h.{i}.attn.c_proj.weight"     -> bufs.layer_weights[i].wo
//    "transformer.h.{i}.mlp.c_fc.weight"        -> bufs.layer_weights[i].wfc
//    "transformer.h.{i}.mlp.c_proj.weight"      -> bufs.layer_weights[i].wdn
//    "value_embeds.{i}.weight"                  -> bufs.layer_weights[i].ve_weight
//    "transformer.h.{i}.attn.ve_gate.weight"    -> bufs.layer_weights[i].ve_gate
//
// 2. Engine checkpoint names — DENSE (no MoE keys):
//    "wte.weight"                               -> bufs.wte
//    "lm_head.weight"                           -> bufs.lm_head
//    "h.{i}.mlp.c_fc.weight"                    -> warm-start into MoE experts
//    "h.{i}.mlp.c_proj.weight"                  -> warm-start into MoE experts
//    (etc.)
//
// 3. Engine checkpoint names — MOE (has moe keys):
//    "h.{i}.moe.router.weight"                  -> bufs.moe_weights[i].w_router
//    "h.{i}.moe.expert.{e}.fc.weight"           -> bufs.moe_weights[i].expert_wfc slice
//    "h.{i}.moe.expert.{e}.proj.weight"         -> bufs.moe_weights[i].expert_wdn slice

// ---------------------------------------------------------------------------
// Local xorshift64 RNG (same algorithm as train.rs, separate state)
// ---------------------------------------------------------------------------

struct Rng64(u64);

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_f64(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }

    /// Box-Muller: returns a single N(0,1) sample.
    fn normal(&mut self) -> f64 {
        use std::f64::consts::TAU;
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

/// Load model weights from a safetensors file into the engine's GPU buffers.
///
/// Detects checkpoint type automatically:
/// - MoE checkpoint (has "h.0.moe.router.weight") -> direct load
/// - Dense checkpoint (no MoE keys) -> warm-start into MoE buffers
///
/// All tensors are expected to be bf16. f32 master copies are NOT loaded
/// here -- call `init_f32_masters()` after this to populate them from bf16.
pub fn load_weights_from_safetensors(
    path: &str,
    bufs: &mut BufferManager,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let data = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    // Detect naming convention
    let is_python = tensors.tensor("transformer.wte.weight").is_ok();
    let prefix = if is_python { "transformer." } else { "" };

    // Detect if this is an MoE checkpoint
    let is_moe_ckpt = tensors.tensor("h.0.moe.router.weight").is_ok();

    // Detect vocab size from checkpoint for warm-start embedding expansion
    let ckpt_wte_name = format!("{prefix}wte.weight");
    let ckpt_vocab = {
        let t = tensors.tensor(&ckpt_wte_name)
            .map_err(|_| anyhow::anyhow!("tensor {:?} not found", ckpt_wte_name))?;
        t.shape()[0]
    };

    let mut rng = Rng64::new(0xDEAD_BEEF_CAFE_1337);
    let mut loaded = 0usize;

    // -- wte --
    if ckpt_vocab == VOCAB {
        upload_bf16(&tensors, &ckpt_wte_name, &mut bufs.wte, stream)?;
    } else {
        upload_bf16_expanded(&tensors, &ckpt_wte_name, &mut bufs.wte, ckpt_vocab, VOCAB, D_MODEL, &mut rng, stream)?;
        println!("[init_weights] wte: expanded vocab {ckpt_vocab} -> {VOCAB} (new rows init N(0, 0.02))");
    }
    loaded += 1;

    // -- lm_head --
    if ckpt_vocab == VOCAB {
        upload_bf16(&tensors, "lm_head.weight", &mut bufs.lm_head, stream)?;
    } else {
        upload_bf16_expanded(&tensors, "lm_head.weight", &mut bufs.lm_head, ckpt_vocab, VOCAB, D_MODEL, &mut rng, stream)?;
        println!("[init_weights] lm_head: expanded vocab {ckpt_vocab} -> {VOCAB}");
    }
    loaded += 1;

    // -- resid_lambdas --
    upload_bf16(&tensors, "resid_lambdas", &mut bufs.resid_lambdas, stream)?;
    loaded += 1;

    // -- x0_lambdas --
    upload_bf16(&tensors, "x0_lambdas", &mut bufs.x0_lambdas, stream)?;
    loaded += 1;

    // -- per-layer weights --
    for i in 0..N_LAYER {
        let lw = &mut bufs.layer_weights[i];
        let h_prefix = format!("{prefix}h.{i}");

        // Attention: always loaded unchanged
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_q.weight"), &mut lw.wq, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_k.weight"), &mut lw.wk, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_v.weight"), &mut lw.wv, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_proj.weight"), &mut lw.wo, stream)?;
        loaded += 4;

        // Dense MLP (wfc/wdn) — only present for non-MoE layers
        if let Some(ref mut wfc) = lw.wfc {
            upload_bf16(&tensors, &format!("{h_prefix}.mlp.c_fc.weight"), wfc, stream)?;
            loaded += 1;
        }
        if let Some(ref mut wdn) = lw.wdn {
            upload_bf16(&tensors, &format!("{h_prefix}.mlp.c_proj.weight"), wdn, stream)?;
            loaded += 1;
        }

        // MoE weights
        if is_moe_layer(i) && i < bufs.moe_weights.len() {
            let mw = &mut bufs.moe_weights[i];

            if is_moe_ckpt {
                loaded += load_moe_direct(&tensors, i, mw, stream)?;
            } else {
                let dense_wfc_name = format!("{h_prefix}.mlp.c_fc.weight");
                let dense_wdn_name = format!("{h_prefix}.mlp.c_proj.weight");
                loaded += warm_start_moe(&tensors, &dense_wfc_name, &dense_wdn_name, mw, &mut rng, stream)?;
                if i == 0 {
                    println!("[init_weights] warm-starting MoE from dense checkpoint");
                }
            }
        }

        // VE weight
        if let Some(ref mut ve_w) = lw.ve_weight {
            let ve_name = if is_python {
                format!("value_embeds.{i}.weight")
            } else {
                format!("ve.{i}.weight")
            };
            if ckpt_vocab == VOCAB {
                upload_bf16(&tensors, &ve_name, ve_w, stream)?;
            } else {
                upload_bf16_expanded(&tensors, &ve_name, ve_w, ckpt_vocab, VOCAB, D_MODEL, &mut rng, stream)?;
            }
            loaded += 1;
        }

        // VE gate
        if let Some(ref mut ve_g) = lw.ve_gate {
            upload_bf16(&tensors, &format!("{h_prefix}.attn.ve_gate.weight"), ve_g, stream)?;
            loaded += 1;
        }
    }

    stream.synchronize()?;
    let fmt = match (is_python, is_moe_ckpt) {
        (true, _) => "Python dense",
        (false, true) => "engine MoE",
        (false, false) => "engine dense -> MoE warm-start",
    };
    println!("[init_weights] loaded {loaded} tensors from {path} ({fmt})");

    Ok(())
}

/// Staged (pipeline-parallel) variant: only loads weights for layers this stage owns.
///
/// - Embedding/lm_head loaded only if has_embedding/has_head
/// - resid_lambdas/x0_lambdas always loaded (small, all stages need them)
/// - Layer weights only for [layer_start, layer_end)
/// - moe_weights[0] corresponds to the first MoE layer this stage owns
pub fn load_weights_staged(
    path: &str,
    bufs: &mut BufferManager,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let data = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    let is_python = tensors.tensor("transformer.wte.weight").is_ok();
    let prefix = if is_python { "transformer." } else { "" };
    let is_moe_ckpt = tensors.tensor("h.0.moe.router.weight").is_ok();

    let ckpt_wte_name = format!("{prefix}wte.weight");
    let ckpt_vocab = {
        let t = tensors.tensor(&ckpt_wte_name)
            .map_err(|_| anyhow::anyhow!("tensor {:?} not found", ckpt_wte_name))?;
        t.shape()[0]
    };

    let mut rng = Rng64::new(0xDEAD_BEEF_CAFE_1337);
    let mut loaded = 0usize;

    // -- wte (only if this stage owns the embedding) --
    if bufs.has_embedding {
        if ckpt_vocab == VOCAB {
            upload_bf16(&tensors, &ckpt_wte_name, &mut bufs.wte, stream)?;
        } else {
            upload_bf16_expanded(&tensors, &ckpt_wte_name, &mut bufs.wte, ckpt_vocab, VOCAB, D_MODEL, &mut rng, stream)?;
            println!("[init_weights_staged] wte: expanded vocab {ckpt_vocab} -> {VOCAB}");
        }
        loaded += 1;
    }

    // -- lm_head (only if this stage owns the head) --
    if bufs.has_head {
        if ckpt_vocab == VOCAB {
            upload_bf16(&tensors, "lm_head.weight", &mut bufs.lm_head, stream)?;
        } else {
            upload_bf16_expanded(&tensors, "lm_head.weight", &mut bufs.lm_head, ckpt_vocab, VOCAB, D_MODEL, &mut rng, stream)?;
            println!("[init_weights_staged] lm_head: expanded vocab {ckpt_vocab} -> {VOCAB}");
        }
        loaded += 1;
    }

    // -- resid_lambdas & x0_lambdas (always, small) --
    upload_bf16(&tensors, "resid_lambdas", &mut bufs.resid_lambdas, stream)?;
    upload_bf16(&tensors, "x0_lambdas", &mut bufs.x0_lambdas, stream)?;
    loaded += 2;

    // -- per-layer weights (only owned layers) --
    let mut moe_idx = 0usize; // index into bufs.moe_weights for owned MoE layers
    for i in bufs.layer_start..bufs.layer_end {
        let lw = &mut bufs.layer_weights[i];
        let h_prefix = format!("{prefix}h.{i}");

        // Attention
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_q.weight"), &mut lw.wq, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_k.weight"), &mut lw.wk, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_v.weight"), &mut lw.wv, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_proj.weight"), &mut lw.wo, stream)?;
        loaded += 4;

        // Dense MLP
        if let Some(ref mut wfc) = lw.wfc {
            upload_bf16(&tensors, &format!("{h_prefix}.mlp.c_fc.weight"), wfc, stream)?;
            loaded += 1;
        }
        if let Some(ref mut wdn) = lw.wdn {
            upload_bf16(&tensors, &format!("{h_prefix}.mlp.c_proj.weight"), wdn, stream)?;
            loaded += 1;
        }

        // MoE weights — moe_weights is indexed locally (0 = first owned MoE layer)
        if is_moe_layer(i) {
            let mw = &mut bufs.moe_weights[moe_idx];
            moe_idx += 1;

            if is_moe_ckpt {
                loaded += load_moe_direct(&tensors, i, mw, stream)?;
            } else {
                let dense_wfc_name = format!("{h_prefix}.mlp.c_fc.weight");
                let dense_wdn_name = format!("{h_prefix}.mlp.c_proj.weight");
                loaded += warm_start_moe(&tensors, &dense_wfc_name, &dense_wdn_name, mw, &mut rng, stream)?;
                if moe_idx == 1 {
                    println!("[init_weights_staged] warm-starting MoE from dense checkpoint");
                }
            }
        }

        // VE weight
        if let Some(ref mut ve_w) = lw.ve_weight {
            let ve_name = if is_python {
                format!("value_embeds.{i}.weight")
            } else {
                format!("ve.{i}.weight")
            };
            if ckpt_vocab == VOCAB {
                upload_bf16(&tensors, &ve_name, ve_w, stream)?;
            } else {
                upload_bf16_expanded(&tensors, &ve_name, ve_w, ckpt_vocab, VOCAB, D_MODEL, &mut rng, stream)?;
            }
            loaded += 1;
        }

        // VE gate
        if let Some(ref mut ve_g) = lw.ve_gate {
            upload_bf16(&tensors, &format!("{h_prefix}.attn.ve_gate.weight"), ve_g, stream)?;
            loaded += 1;
        }
    }

    stream.synchronize()?;
    let fmt = match (is_python, is_moe_ckpt) {
        (true, _) => "Python dense",
        (false, true) => "engine MoE",
        (false, false) => "engine dense -> MoE warm-start",
    };
    println!(
        "[init_weights_staged] loaded {loaded} tensors from {path} ({fmt}), layers {}..{}",
        bufs.layer_start, bufs.layer_end
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// MoE loading helpers
// ---------------------------------------------------------------------------

/// Load MoE weights directly from an MoE checkpoint.
fn load_moe_direct(
    tensors: &SafeTensors,
    layer: usize,
    mw: &mut crate::buffer::MoeLayerWeights,
    stream: &Arc<CudaStream>,
) -> Result<usize> {
    let router_name = format!("h.{layer}.moe.router.weight");
    upload_bf16(tensors, &router_name, &mut mw.w_router, stream)?;

    let expert_numel = MLP_DIM_E * D_MODEL;
    let mut packed_wfc = Vec::with_capacity(N_EXPERTS * expert_numel);
    let mut packed_wdn = Vec::with_capacity(N_EXPERTS * expert_numel);

    for e in 0..N_EXPERTS {
        let fc_name = format!("h.{layer}.moe.expert.{e}.fc.weight");
        let proj_name = format!("h.{layer}.moe.expert.{e}.proj.weight");

        let fc_host = extract_bf16_host(tensors, &fc_name)?;
        let dn_host = extract_bf16_host(tensors, &proj_name)?;
        ensure!(fc_host.len() == expert_numel, "tensor {fc_name}: expected {expert_numel} elements, got {}", fc_host.len());
        ensure!(dn_host.len() == expert_numel, "tensor {proj_name}: expected {expert_numel} elements, got {}", dn_host.len());

        packed_wfc.extend_from_slice(&fc_host);
        packed_wdn.extend_from_slice(&dn_host);
    }

    stream.memcpy_htod(&packed_wfc, &mut mw.expert_wfc)?;
    stream.memcpy_htod(&packed_wdn, &mut mw.expert_wdn)?;

    Ok(1 + N_EXPERTS * 2)
}

/// Warm-start MoE weights from a dense checkpoint.
///
/// - Expert 0: exact copy of pretrained dense wfc/wdn (first MLP_DIM_E rows/cols)
/// - Experts 1-7: copy + N(0, 0.01 * ||W||_F / sqrt(numel)) perturbation
/// - Router: zero-init (uniform routing initially)
fn warm_start_moe(
    tensors: &SafeTensors,
    wfc_name: &str,
    wdn_name: &str,
    mw: &mut crate::buffer::MoeLayerWeights,
    rng: &mut Rng64,
    stream: &Arc<CudaStream>,
) -> Result<usize> {
    let expert_numel = MLP_DIM_E * D_MODEL;

    // Load dense weights to host
    let dense_wfc = extract_bf16_host(tensors, wfc_name)?;
    let dense_wdn = extract_bf16_host(tensors, wdn_name)?;

    // wfc: [MLP_DIM, D_MODEL] -> take first MLP_DIM_E rows -> [MLP_DIM_E, D_MODEL]
    let wfc_slice: Vec<bf16> = dense_wfc[..expert_numel].to_vec();

    // wdn: [D_MODEL, MLP_DIM] -> take first MLP_DIM_E cols -> [D_MODEL, MLP_DIM_E]
    let wdn_slice: Vec<bf16> = if MLP_DIM_E == MLP_DIM {
        dense_wdn[..expert_numel].to_vec()
    } else {
        let mut out = Vec::with_capacity(expert_numel);
        for row in 0..D_MODEL {
            let row_start = row * MLP_DIM;
            out.extend_from_slice(&dense_wdn[row_start..row_start + MLP_DIM_E]);
        }
        out
    };

    // Frobenius norms for perturbation scaling
    let wfc_fro = frobenius_norm(&wfc_slice);
    let wdn_fro = frobenius_norm(&wdn_slice);
    let wfc_sigma = 0.01 * wfc_fro / (expert_numel as f64).sqrt();
    let wdn_sigma = 0.01 * wdn_fro / (expert_numel as f64).sqrt();

    // Build packed expert buffers [N_EXPERTS * MLP_DIM_E, D_MODEL]
    let mut packed_wfc = Vec::with_capacity(N_EXPERTS * expert_numel);
    let mut packed_wdn = Vec::with_capacity(N_EXPERTS * expert_numel);

    for e in 0..N_EXPERTS {
        if e == 0 {
            // Expert 0: exact copy
            packed_wfc.extend_from_slice(&wfc_slice);
            packed_wdn.extend_from_slice(&wdn_slice);
        } else {
            // Experts 1-7: copy + perturbation
            for &w in &wfc_slice {
                let perturb = rng.normal() * wfc_sigma;
                packed_wfc.push(bf16::from_f64(w.to_f64() + perturb));
            }
            for &w in &wdn_slice {
                let perturb = rng.normal() * wdn_sigma;
                packed_wdn.push(bf16::from_f64(w.to_f64() + perturb));
            }
        }
    }

    stream.memcpy_htod(&packed_wfc, &mut mw.expert_wfc)?;
    stream.memcpy_htod(&packed_wdn, &mut mw.expert_wdn)?;

    // Router: zero-init
    let zeros = vec![bf16::ZERO; N_EXPERTS * D_MODEL];
    stream.memcpy_htod(&zeros, &mut mw.w_router)?;

    Ok(1 + N_EXPERTS * 2)
}

// ---------------------------------------------------------------------------
// Tensor upload helpers
// ---------------------------------------------------------------------------

/// Upload a bf16 tensor from safetensors to a GPU buffer (exact size match).
fn upload_bf16(
    tensors: &SafeTensors,
    name: &str,
    buf: &mut CudaSlice<bf16>,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let t = tensors.tensor(name)
        .map_err(|_| anyhow::anyhow!("tensor {name:?} not found in safetensors file"))?;
    let bytes = t.data();

    match t.dtype() {
        safetensors::Dtype::BF16 => {
            let host: &[bf16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const bf16, bytes.len() / 2)
            };
            ensure!(
                host.len() == buf.len(),
                "tensor {name}: safetensors has {} elements, buffer has {}",
                host.len(), buf.len()
            );
            stream.memcpy_htod(host, buf)?;
        }
        safetensors::Dtype::F32 => {
            let f32_host: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };
            ensure!(
                f32_host.len() == buf.len(),
                "tensor {name}: safetensors has {} f32 elements, buffer has {}",
                f32_host.len(), buf.len()
            );
            let bf16_host: Vec<bf16> = f32_host.iter().map(|&x| bf16::from_f32(x)).collect();
            stream.memcpy_htod(&bf16_host, buf)?;
        }
        other => {
            anyhow::bail!("tensor {name}: unsupported dtype {other:?} (expected BF16 or F32)");
        }
    }

    Ok(())
}

/// Upload with vocab expansion: copy ckpt_vocab rows, init new rows with N(0, 0.02).
fn upload_bf16_expanded(
    tensors: &SafeTensors,
    name: &str,
    buf: &mut CudaSlice<bf16>,
    ckpt_vocab: usize,
    target_vocab: usize,
    dim: usize,
    rng: &mut Rng64,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let host_ckpt = extract_bf16_host(tensors, name)?;
    ensure!(
        host_ckpt.len() == ckpt_vocab * dim,
        "tensor {name}: expected {} elements for vocab {ckpt_vocab}, got {}",
        ckpt_vocab * dim, host_ckpt.len()
    );

    let mut host = Vec::with_capacity(target_vocab * dim);
    host.extend_from_slice(&host_ckpt);

    // New rows: N(0, 0.02)
    let new_elems = (target_vocab - ckpt_vocab) * dim;
    for _ in 0..new_elems {
        host.push(bf16::from_f64(rng.normal() * 0.02));
    }

    ensure!(
        host.len() == buf.len(),
        "tensor {name}: expanded to {} elements, buffer has {}",
        host.len(), buf.len()
    );
    stream.memcpy_htod(&host, buf)?;
    Ok(())
}

/// Extract a tensor as Vec<bf16> on host (handles bf16 and f32 source dtypes).
fn extract_bf16_host(tensors: &SafeTensors, name: &str) -> Result<Vec<bf16>> {
    let t = tensors.tensor(name)
        .map_err(|_| anyhow::anyhow!("tensor {name:?} not found in safetensors file"))?;
    let bytes = t.data();

    match t.dtype() {
        safetensors::Dtype::BF16 => {
            let host: &[bf16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const bf16, bytes.len() / 2)
            };
            Ok(host.to_vec())
        }
        safetensors::Dtype::F32 => {
            let f32_host: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };
            Ok(f32_host.iter().map(|&x| bf16::from_f32(x)).collect())
        }
        other => {
            anyhow::bail!("tensor {name}: unsupported dtype {other:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Frobenius norm of a bf16 slice (accumulated in f64).
fn frobenius_norm(data: &[bf16]) -> f64 {
    let sum_sq: f64 = data.iter().map(|x| {
        let v = x.to_f64();
        v * v
    }).sum();
    sum_sq.sqrt()
}

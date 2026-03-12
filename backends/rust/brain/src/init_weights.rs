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
// This loader handles two naming conventions:
//
// 1. Python state_dict names (from torch model.state_dict()):
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
// 2. Engine checkpoint names (from save_checkpoint):
//    "wte.weight"                               -> bufs.wte
//    "lm_head.weight"                           -> bufs.lm_head
//    "resid_lambdas"                            -> bufs.resid_lambdas
//    "x0_lambdas"                               -> bufs.x0_lambdas
//    "h.{i}.attn.c_q.weight"                    -> bufs.layer_weights[i].wq
//    "h.{i}.attn.c_k.weight"                    -> bufs.layer_weights[i].wk
//    "h.{i}.attn.c_v.weight"                    -> bufs.layer_weights[i].wv
//    "h.{i}.attn.c_proj.weight"                 -> bufs.layer_weights[i].wo
//    "h.{i}.mlp.c_fc.weight"                    -> bufs.layer_weights[i].wfc
//    "h.{i}.mlp.c_proj.weight"                  -> bufs.layer_weights[i].wdn
//    "ve.{i}.weight"                            -> bufs.layer_weights[i].ve_weight
//    "h.{i}.attn.ve_gate.weight"                -> bufs.layer_weights[i].ve_gate

/// Load model weights from a safetensors file into the engine's GPU buffers.
///
/// Handles both Python (torch state_dict) and engine checkpoint naming.
/// All tensors are expected to be bf16. f32 master copies are NOT loaded
/// here -- call `init_f32_masters()` after this to populate them from bf16.
pub fn load_weights_from_safetensors(
    path: &str,
    bufs: &mut BufferManager,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let data = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    // Detect naming convention by checking for "transformer.wte.weight"
    let is_python = tensors.tensor("transformer.wte.weight").is_ok();
    let prefix = if is_python { "transformer." } else { "" };

    let mut loaded = 0usize;

    // -- wte --
    let wte_name = format!("{prefix}wte.weight");
    upload_bf16(&tensors, &wte_name, &mut bufs.wte, stream)?;
    loaded += 1;

    // -- lm_head --
    upload_bf16(&tensors, "lm_head.weight", &mut bufs.lm_head, stream)?;
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

        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_q.weight"), &mut lw.wq, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_k.weight"), &mut lw.wk, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_v.weight"), &mut lw.wv, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.attn.c_proj.weight"), &mut lw.wo, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.mlp.c_fc.weight"), &mut lw.wfc, stream)?;
        upload_bf16(&tensors, &format!("{h_prefix}.mlp.c_proj.weight"), &mut lw.wdn, stream)?;
        loaded += 6;

        // VE weight: Python uses "value_embeds.{i}.weight", engine uses "ve.{i}.weight"
        if let Some(ref mut ve_w) = lw.ve_weight {
            let ve_name = if is_python {
                format!("value_embeds.{i}.weight")
            } else {
                format!("ve.{i}.weight")
            };
            upload_bf16(&tensors, &ve_name, ve_w, stream)?;
            loaded += 1;
        }

        // VE gate
        if let Some(ref mut ve_g) = lw.ve_gate {
            upload_bf16(&tensors, &format!("{h_prefix}.attn.ve_gate.weight"), ve_g, stream)?;
            loaded += 1;
        }
    }

    stream.synchronize()?;
    println!("[init_weights] loaded {loaded} tensors from {path} ({})",
        if is_python { "Python format" } else { "engine format" });

    Ok(())
}

/// Upload a bf16 tensor from safetensors to a GPU buffer.
/// Handles both bf16 and f32 source tensors (f32 gets converted to bf16).
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
            // Convert f32 -> bf16 on CPU
            let bf16_host: Vec<bf16> = f32_host.iter().map(|&x| bf16::from_f32(x)).collect();
            stream.memcpy_htod(&bf16_host, buf)?;
        }
        other => {
            anyhow::bail!("tensor {name}: unsupported dtype {other:?} (expected BF16 or F32)");
        }
    }

    Ok(())
}

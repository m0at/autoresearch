use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const OPS_METAL_SOURCE: &str = include_str!("../../metal/ops.metal");

static OPS_PIPELINE_CACHE: OnceLock<Mutex<HashMap<String, ComputePipeline>>> = OnceLock::new();

fn get_ops_pipeline(
    device: &candle_metal_kernels::metal::Device,
    kernel_name: &str,
) -> Result<ComputePipeline> {
    let cache = OPS_PIPELINE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache
        .lock()
        .map_err(|e| candle_core::Error::Msg(format!("lock: {e}")))?;
    if let Some(p) = map.get(kernel_name) {
        return Ok(p.clone());
    }
    let library = device
        .new_library_with_source(OPS_METAL_SOURCE, None)
        .map_err(|e| candle_core::Error::Msg(format!("compile ops.metal: {e}")))?;
    let function = library
        .get_function(kernel_name, None)
        .map_err(|e| candle_core::Error::Msg(format!("get_function({kernel_name}): {e}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Msg(format!("pipeline({kernel_name}): {e}")))?;
    map.insert(kernel_name.to_string(), pipeline.clone());
    Ok(pipeline)
}

// ── Fused AdamW ─────────────────────────────────────────────────────────────
//
// Uses the packed kernel variant: adamw_step_packed
// Input:  [param(N) | grad(N) | exp_avg(N) | exp_avg_sq(N)] -- 4N f32
// Output: [new_param(N) | new_exp_avg(N) | new_exp_avg_sq(N)] -- 3N f32
//
// Reduces ~8 separate GPU kernel dispatches (mul, sub, lerp, sqr, div, sqrt,
// etc.) to a single fused dispatch.

struct FusedAdamWOp {
    n: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    bias1_corr: f32,
    bias2_corr: f32,
}

impl CustomOp1 for FusedAdamWOp {
    fn name(&self) -> &'static str {
        "fused_adamw_step"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("non-contiguous".into()))?;
        let data = match storage {
            CpuStorage::F32(d) => &d[start..end],
            _ => return Err(candle_core::Error::Msg("expected F32".into())),
        };
        let n = self.n;
        let param = &data[0..n];
        let grad = &data[n..2 * n];
        let exp_avg = &data[2 * n..3 * n];
        let exp_avg_sq = &data[3 * n..4 * n];

        let mut out = vec![0f32; 3 * n];
        for i in 0..n {
            let mut p = param[i];
            let g = grad[i];
            let mut m = exp_avg[i];
            let mut v = exp_avg_sq[i];

            p -= self.lr * self.wd * p;
            m = self.beta1 * m + (1.0 - self.beta1) * g;
            v = self.beta2 * v + (1.0 - self.beta2) * g * g;
            let m_hat = m / self.bias1_corr;
            let v_hat = v / self.bias2_corr;
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);

            out[i] = p;
            out[n + i] = m;
            out[2 * n + i] = v;
        }
        Ok((CpuStorage::F32(out), Shape::from_dims(&[3 * n])))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();
        let n = self.n;
        let input_offset = layout.start_offset() * storage.dtype().size_in_bytes();

        let out_buf = device.new_buffer(3 * n, DType::F32, "adamw_out")?;

        let pipeline = get_ops_pipeline(device.device(), "adamw_step_packed")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(storage.buffer()), input_offset);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_bytes(2, &self.lr);
        encoder.set_bytes(3, &self.beta1);
        encoder.set_bytes(4, &self.beta2);
        encoder.set_bytes(5, &self.eps);
        encoder.set_bytes(6, &self.wd);
        encoder.set_bytes(7, &self.bias1_corr);
        encoder.set_bytes(8, &self.bias2_corr);
        encoder.set_bytes(9, &(n as u32));
        let groups = (n + 255) / 256;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.use_resource(storage.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);

        Ok((
            MetalStorage::new(out_buf, device.clone(), 3 * n, DType::F32),
            Shape::from_dims(&[3 * n]),
        ))
    }
}

/// Fused AdamW step on Metal. Returns (new_param, new_exp_avg, new_exp_avg_sq).
/// Reduces ~8 separate GPU kernel dispatches to 1 fused dispatch.
pub fn fused_adamw_step(
    param: &Tensor,
    grad: &Tensor,
    exp_avg: &Tensor,
    exp_avg_sq: &Tensor,
    step: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let n = param.elem_count();
    let bias1_corr = 1.0 - beta1.powi(step as i32);
    let bias2_corr = 1.0 - beta2.powi(step as i32);

    let param_f = param.to_dtype(DType::F32)?.contiguous()?;
    let grad_f = grad.to_dtype(DType::F32)?.contiguous()?;
    let ea_f = exp_avg.to_dtype(DType::F32)?.contiguous()?;
    let eas_f = exp_avg_sq.to_dtype(DType::F32)?.contiguous()?;

    let packed = Tensor::cat(
        &[
            &param_f.flatten_all()?,
            &grad_f.flatten_all()?,
            &ea_f.flatten_all()?,
            &eas_f.flatten_all()?,
        ],
        0,
    )?;

    let op = FusedAdamWOp {
        n,
        lr: lr as f32,
        beta1: beta1 as f32,
        beta2: beta2 as f32,
        eps: eps as f32,
        wd: wd as f32,
        bias1_corr: bias1_corr as f32,
        bias2_corr: bias2_corr as f32,
    };

    let result = packed.apply_op1_no_bwd(&op)?;

    let new_param = result.narrow(0, 0, n)?.reshape(param.shape())?;
    let new_ea = result.narrow(0, n, n)?.reshape(exp_avg.shape())?;
    let new_eas = result.narrow(0, 2 * n, n)?.reshape(exp_avg_sq.shape())?;

    let new_param = new_param.to_dtype(param.dtype())?;

    Ok((new_param, new_ea, new_eas))
}

// ── Fused Polar Express ─────────────────────────────────────────────────────
//
// Dispatches 3 Metal kernels per Newton-Schulz iteration:
//   Phase 1: A = X^T @ X (tall) or X @ X^T (wide)
//   Phase 2: B = b*A + c*(A@A)
//   Phase 3: X_out = a*X_in + X_in@B (tall) or a*X_in + B@X_in (wide)
//
// Phase 3 uses separate X_in / X_out buffers. On iteration 0, X_in is the
// immutable input storage and X_out is a fresh buffer. On iterations 1+,
// both point to the same buffer (in-place update).

const POLAR_EXPRESS_COEFFS: [(f64, f64, f64); 5] = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
];

struct FusedPolarExpressOp {
    rows: usize,
    cols: usize,
    num_params: usize,
    ns_steps: usize,
}

impl CustomOp1 for FusedPolarExpressOp {
    fn name(&self) -> &'static str {
        "fused_polar_express"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "fused_polar_express: CPU not supported, use candle ops fallback".into(),
        ))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();
        let rows = self.rows as u32;
        let cols = self.cols as u32;
        let num_params = self.num_params as u32;
        let is_tall: u8 = if self.rows > self.cols { 1 } else { 0 };
        let inner = if self.rows > self.cols {
            self.cols
        } else {
            self.rows
        };

        let total_elems = self.num_params * self.rows * self.cols;
        let shape = layout.shape().clone();
        let input_offset = layout.start_offset() * storage.dtype().size_in_bytes();

        // Output buffer: phase 3 writes here. After iter 0 it also serves as input.
        let x_buf = device.new_buffer(total_elems, DType::BF16, "pe_x")?;

        // Workspace: holds A and B matrices per batch element.
        let workspace_elems = self.num_params * inner * inner * 2;
        let ws_buf = device.new_buffer(workspace_elems, DType::BF16, "pe_workspace")?;

        let tile = 16usize;
        let tiles_inner = (inner + tile - 1) / tile;
        let tiles_cols = (self.cols + tile - 1) / tile;
        let tiles_rows = (self.rows + tile - 1) / tile;

        let p1 = get_ops_pipeline(device.device(), "polar_express_phase1")?;
        let p2 = get_ops_pipeline(device.device(), "polar_express_phase2")?;
        let p3 = get_ops_pipeline(device.device(), "polar_express_phase3")?;

        for (iter_idx, &(a, b, c)) in POLAR_EXPRESS_COEFFS[..self.ns_steps].iter().enumerate() {
            let a = a as f32;
            let b = b as f32;
            let c = c as f32;
            let inner_u32 = inner as u32;

            // X source: input on first iteration, x_buf (output of prior phase3) after
            let (x_in_buf, x_in_offset) = if iter_idx == 0 {
                (storage.buffer(), input_offset)
            } else {
                (&*x_buf, 0usize)
            };

            // Phase 1: A = X^T@X or X@X^T (read-only on X)
            {
                let encoder = device.command_encoder()?;
                encoder.set_compute_pipeline_state(&p1);
                encoder.set_buffer(0, Some(x_in_buf), x_in_offset);
                encoder.set_buffer(1, Some(&ws_buf), 0);
                encoder.set_bytes(2, &num_params);
                encoder.set_bytes(3, &rows);
                encoder.set_bytes(4, &cols);
                encoder.set_bytes(5, &is_tall);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: tiles_inner,
                        height: tiles_inner,
                        depth: self.num_params,
                    },
                    MTLSize {
                        width: tile,
                        height: tile,
                        depth: 1,
                    },
                );
                encoder.use_resource(x_in_buf, objc2_metal::MTLResourceUsage::Read);
                encoder.use_resource(&*ws_buf, objc2_metal::MTLResourceUsage::Write);
            }

            // Phase 2: B = b*A + c*(A@A) (workspace only)
            {
                let encoder = device.command_encoder()?;
                encoder.set_compute_pipeline_state(&p2);
                encoder.set_buffer(0, Some(&ws_buf), 0);
                encoder.set_bytes(1, &b);
                encoder.set_bytes(2, &c);
                encoder.set_bytes(3, &num_params);
                encoder.set_bytes(4, &inner_u32);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: tiles_inner,
                        height: tiles_inner,
                        depth: self.num_params,
                    },
                    MTLSize {
                        width: tile,
                        height: tile,
                        depth: 1,
                    },
                );
                encoder.use_resource(
                    &*ws_buf,
                    objc2_metal::MTLResourceUsage::Read | objc2_metal::MTLResourceUsage::Write,
                );
            }

            // Phase 3: X_out = a*X_in + X_in@B (or B@X_in)
            // buffer(0) = X_in (read), buffer(1) = X_out (write), buffer(2) = workspace (read)
            {
                let encoder = device.command_encoder()?;
                encoder.set_compute_pipeline_state(&p3);
                encoder.set_buffer(0, Some(x_in_buf), x_in_offset); // X_in
                encoder.set_buffer(1, Some(&x_buf), 0); // X_out
                encoder.set_buffer(2, Some(&ws_buf), 0); // workspace
                encoder.set_bytes(3, &a);
                encoder.set_bytes(4, &num_params);
                encoder.set_bytes(5, &rows);
                encoder.set_bytes(6, &cols);
                encoder.set_bytes(7, &is_tall);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: tiles_cols,
                        height: tiles_rows,
                        depth: self.num_params,
                    },
                    MTLSize {
                        width: tile,
                        height: tile,
                        depth: 1,
                    },
                );
                encoder.use_resource(x_in_buf, objc2_metal::MTLResourceUsage::Read);
                encoder.use_resource(
                    &*x_buf,
                    objc2_metal::MTLResourceUsage::Write,
                );
                encoder.use_resource(&*ws_buf, objc2_metal::MTLResourceUsage::Read);
            }
        }

        Ok((
            MetalStorage::new(x_buf, device.clone(), total_elems, DType::BF16),
            shape,
        ))
    }
}

/// Fused Polar Express (Newton-Schulz orthogonalization) on Metal.
/// Input g has shape (..., rows, cols) with optional batch dimensions.
/// Returns BF16 orthogonalized tensor, same shape.
/// Replaces 5 iterations * ~4 matmuls = 20+ kernel dispatches with
/// 5 iterations * 3 phases = 15 tightly scheduled dispatches.
pub fn fused_polar_express(g: &Tensor, ns_steps: usize) -> anyhow::Result<Tensor> {
    let ndim = g.dims().len();
    let rows = g.dims()[ndim - 2];
    let cols = g.dims()[ndim - 1];
    let num_params: usize = g.dims()[..ndim - 2].iter().product::<usize>().max(1);

    // Normalize: x = g / (1.02 * frobenius_norm(g) + 1e-6)
    let x = g.to_dtype(DType::BF16)?;
    let norm = ((super::muon_adamw::frobenius_norm_keepdim(&x)? * 1.02)? + 1e-6)?;
    let x = x.broadcast_div(&norm)?;
    let x = x.contiguous()?;

    let flat_shape = if num_params > 1 {
        vec![num_params, rows, cols]
    } else {
        vec![rows, cols]
    };
    let x_flat = x.reshape(&flat_shape[..])?;

    let op = FusedPolarExpressOp {
        rows,
        cols,
        num_params,
        ns_steps,
    };
    let result = x_flat.apply_op1_no_bwd(&op)?;
    let result = result.reshape(g.shape())?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_fused_adamw_cpu() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let param = Tensor::new(&[1.0f32, 2.0, 3.0], &dev)?;
        let grad = Tensor::new(&[0.1f32, 0.2, 0.3], &dev)?;
        let exp_avg = param.zeros_like()?;
        let exp_avg_sq = param.zeros_like()?;

        let (new_p, new_ea, new_eas) =
            fused_adamw_step(&param, &grad, &exp_avg, &exp_avg_sq, 1, 1e-3, 0.9, 0.999, 1e-8, 0.0)?;

        let vals: Vec<f32> = new_p.to_vec1()?;
        let old_vals: Vec<f32> = param.to_vec1()?;
        for (n, o) in vals.iter().zip(old_vals.iter()) {
            assert!(n.is_finite());
            assert!((n - o).abs() > 1e-12, "param should change");
        }

        let ea_vals: Vec<f32> = new_ea.to_vec1()?;
        assert!(ea_vals[0].abs() > 1e-10, "exp_avg should be non-zero");

        let eas_vals: Vec<f32> = new_eas.to_vec1()?;
        assert!(eas_vals[0] > 1e-10, "exp_avg_sq should be positive");

        Ok(())
    }

    #[test]
    fn test_fused_adamw_matches_reference() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let param = Tensor::new(&[1.5f32, -0.5, 2.0, 0.1], &dev)?;
        let grad = Tensor::new(&[0.3f32, -0.1, 0.5, 0.2], &dev)?;
        let exp_avg = param.zeros_like()?;
        let exp_avg_sq = param.zeros_like()?;

        // Reference: candle ops path
        let (ref_p, ref_ea, ref_eas) = super::super::muon_adamw::adamw_step(
            &param, &grad, &exp_avg, &exp_avg_sq, 1, 1e-3, 0.9, 0.999, 1e-8, 0.01,
        )?;

        // Fused path
        let (fused_p, fused_ea, fused_eas) =
            fused_adamw_step(&param, &grad, &exp_avg, &exp_avg_sq, 1, 1e-3, 0.9, 0.999, 1e-8, 0.01)?;

        let ref_vals: Vec<f32> = ref_p.to_vec1()?;
        let fused_vals: Vec<f32> = fused_p.to_vec1()?;
        for (r, f) in ref_vals.iter().zip(fused_vals.iter()) {
            assert!(
                (r - f).abs() < 1e-5,
                "param mismatch: ref={r} fused={f}"
            );
        }

        let ref_ea: Vec<f32> = ref_ea.to_vec1()?;
        let fused_ea: Vec<f32> = fused_ea.to_vec1()?;
        for (r, f) in ref_ea.iter().zip(fused_ea.iter()) {
            assert!((r - f).abs() < 1e-5, "exp_avg mismatch: ref={r} fused={f}");
        }

        let ref_eas: Vec<f32> = ref_eas.to_vec1()?;
        let fused_eas: Vec<f32> = fused_eas.to_vec1()?;
        for (r, f) in ref_eas.iter().zip(fused_eas.iter()) {
            assert!(
                (r - f).abs() < 1e-5,
                "exp_avg_sq mismatch: ref={r} fused={f}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_fused_adamw_weight_decay() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let param = Tensor::new(&[2.0f32, 3.0, 4.0], &dev)?;
        let grad = param.zeros_like()?;
        let exp_avg = param.zeros_like()?;
        let exp_avg_sq = param.zeros_like()?;

        let (new_p, _, _) =
            fused_adamw_step(&param, &grad, &exp_avg, &exp_avg_sq, 1, 0.01, 0.9, 0.999, 1e-8, 0.1)?;

        let new_vals: Vec<f32> = new_p.to_vec1()?;
        let old_vals: Vec<f32> = param.to_vec1()?;
        for (n, o) in new_vals.iter().zip(old_vals.iter()) {
            assert!(n.abs() < o.abs(), "weight decay should shrink: {n} vs {o}");
        }
        Ok(())
    }

    #[test]
    fn test_fused_adamw_multi_step() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let mut param = Tensor::new(&[1.0f32, 2.0, 3.0], &dev)?;
        let grad = Tensor::new(&[0.5f32, 0.5, 0.5], &dev)?;
        let mut ea = param.zeros_like()?;
        let mut eas = param.zeros_like()?;

        for step in 1..=5 {
            let (np, ne, nes) =
                fused_adamw_step(&param, &grad, &ea, &eas, step, 1e-3, 0.9, 0.999, 1e-8, 0.0)?;
            param = np;
            ea = ne;
            eas = nes;
        }

        let vals: Vec<f32> = param.to_vec1()?;
        for v in &vals {
            assert!(v.is_finite(), "param should remain finite after 5 steps");
        }
        Ok(())
    }

    // ── Metal GPU tests ──────────────────────────────────────────────────

    #[test]
    fn test_fused_adamw_metal() -> anyhow::Result<()> {
        let dev = Device::new_metal(0)?;
        let param = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &dev)?;
        let grad = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4], &dev)?;
        let exp_avg = param.zeros_like()?;
        let exp_avg_sq = param.zeros_like()?;

        let (new_p, new_ea, new_eas) =
            fused_adamw_step(&param, &grad, &exp_avg, &exp_avg_sq, 1, 1e-3, 0.9, 0.999, 1e-8, 0.0)?;

        let vals: Vec<f32> = new_p.to_vec1()?;
        let old_vals: Vec<f32> = param.to_vec1()?;
        for (n, o) in vals.iter().zip(old_vals.iter()) {
            assert!(n.is_finite(), "Metal adamw output should be finite");
            assert!((n - o).abs() > 1e-12, "Metal adamw should change param");
        }

        let ea_vals: Vec<f32> = new_ea.to_vec1()?;
        assert!(ea_vals[0].abs() > 1e-10);

        let eas_vals: Vec<f32> = new_eas.to_vec1()?;
        assert!(eas_vals[0] > 1e-10);

        Ok(())
    }

    #[test]
    fn test_fused_adamw_metal_matches_cpu() -> anyhow::Result<()> {
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;

        let p_cpu = Tensor::new(&[1.5f32, -0.5, 2.0, 0.1, -1.0, 3.0, 0.5, -2.0], &cpu)?;
        let g_cpu = Tensor::new(&[0.3f32, -0.1, 0.5, 0.2, -0.3, 0.1, 0.4, -0.2], &cpu)?;
        let ea_cpu = p_cpu.zeros_like()?;
        let eas_cpu = p_cpu.zeros_like()?;

        let p_metal = p_cpu.to_device(&metal)?;
        let g_metal = g_cpu.to_device(&metal)?;
        let ea_metal = p_metal.zeros_like()?;
        let eas_metal = p_metal.zeros_like()?;

        let (cpu_p, cpu_ea, cpu_eas) =
            fused_adamw_step(&p_cpu, &g_cpu, &ea_cpu, &eas_cpu, 1, 1e-3, 0.9, 0.999, 1e-8, 0.01)?;
        let (metal_p, metal_ea, metal_eas) =
            fused_adamw_step(&p_metal, &g_metal, &ea_metal, &eas_metal, 1, 1e-3, 0.9, 0.999, 1e-8, 0.01)?;

        let cpu_vals: Vec<f32> = cpu_p.to_vec1()?;
        let metal_vals: Vec<f32> = metal_p.to_vec1()?;
        for (c, m) in cpu_vals.iter().zip(metal_vals.iter()) {
            assert!(
                (c - m).abs() < 1e-5,
                "Metal/CPU param mismatch: cpu={c} metal={m}"
            );
        }

        let cpu_ea: Vec<f32> = cpu_ea.to_vec1()?;
        let metal_ea: Vec<f32> = metal_ea.to_vec1()?;
        for (c, m) in cpu_ea.iter().zip(metal_ea.iter()) {
            assert!((c - m).abs() < 1e-5, "Metal/CPU exp_avg mismatch");
        }

        let cpu_eas: Vec<f32> = cpu_eas.to_vec1()?;
        let metal_eas: Vec<f32> = metal_eas.to_vec1()?;
        for (c, m) in cpu_eas.iter().zip(metal_eas.iter()) {
            assert!((c - m).abs() < 1e-5, "Metal/CPU exp_avg_sq mismatch");
        }

        Ok(())
    }

    #[test]
    fn test_fused_adamw_metal_multi_step() -> anyhow::Result<()> {
        let dev = Device::new_metal(0)?;
        let mut param = Tensor::new(&[1.0f32, 2.0, 3.0], &dev)?;
        let grad = Tensor::new(&[0.5f32, 0.5, 0.5], &dev)?;
        let mut ea = param.zeros_like()?;
        let mut eas = param.zeros_like()?;

        for step in 1..=10 {
            let (np, ne, nes) =
                fused_adamw_step(&param, &grad, &ea, &eas, step, 1e-3, 0.9, 0.999, 1e-8, 0.0)?;
            param = np;
            ea = ne;
            eas = nes;
        }

        let vals: Vec<f32> = param.to_vec1()?;
        for v in &vals {
            assert!(v.is_finite(), "Metal adamw should remain finite after 10 steps");
        }
        Ok(())
    }

    #[test]
    fn test_fused_polar_express_metal() -> anyhow::Result<()> {
        let dev = Device::new_metal(0)?;
        // Wide matrix (4 < 8)
        let g = Tensor::randn(0f32, 1.0, (4, 8), &dev)?.to_dtype(DType::BF16)?;
        let x = fused_polar_express(&g, 5)?;

        assert_eq!(x.dims(), &[4, 8]);
        assert_eq!(x.dtype(), DType::BF16);

        // Check orthogonality: X @ X^T should approximate identity
        let x_f32 = x.to_dtype(DType::F32)?;
        let xxt = x_f32.matmul(&x_f32.t()?)?;
        let xxt_vals: Vec<Vec<f32>> = xxt.to_vec2()?;
        for i in 0..4 {
            assert!(
                xxt_vals[i][i] > 0.3,
                "diagonal X@X^T[{i},{i}] = {} should be positive",
                xxt_vals[i][i]
            );
            for j in 0..4 {
                if i != j {
                    assert!(
                        xxt_vals[i][j].abs() < 0.5,
                        "off-diag X@X^T[{i},{j}] = {} should be small",
                        xxt_vals[i][j]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_fused_polar_express_metal_tall() -> anyhow::Result<()> {
        let dev = Device::new_metal(0)?;
        // Tall matrix (8 > 4)
        let g = Tensor::randn(0f32, 1.0, (8, 4), &dev)?.to_dtype(DType::BF16)?;
        let x = fused_polar_express(&g, 5)?;

        assert_eq!(x.dims(), &[8, 4]);
        assert_eq!(x.dtype(), DType::BF16);

        // For tall: X^T @ X should approximate identity (4x4)
        let x_f32 = x.to_dtype(DType::F32)?;
        let xtx = x_f32.t()?.matmul(&x_f32)?;
        let xtx_vals: Vec<Vec<f32>> = xtx.to_vec2()?;
        for i in 0..4 {
            assert!(
                xtx_vals[i][i] > 0.3,
                "diagonal X^T@X[{i},{i}] = {} should be positive",
                xtx_vals[i][i]
            );
            for j in 0..4 {
                if i != j {
                    assert!(
                        xtx_vals[i][j].abs() < 0.5,
                        "off-diag X^T@X[{i},{j}] = {} should be small",
                        xtx_vals[i][j]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_fused_polar_express_metal_batched() -> anyhow::Result<()> {
        let dev = Device::new_metal(0)?;
        // Batched: 3 matrices of shape (4, 8)
        let g = Tensor::randn(0f32, 1.0, (3, 4, 8), &dev)?.to_dtype(DType::BF16)?;
        let x = fused_polar_express(&g, 5)?;

        assert_eq!(x.dims(), &[3, 4, 8]);
        assert_eq!(x.dtype(), DType::BF16);

        // Check each batch element
        for b in 0..3 {
            let xi = x.get(b)?.to_dtype(DType::F32)?;
            let xxt = xi.matmul(&xi.t()?)?;
            let xxt_vals: Vec<Vec<f32>> = xxt.to_vec2()?;
            for i in 0..4 {
                assert!(
                    xxt_vals[i][i] > 0.3,
                    "batch {b} diagonal [{i},{i}] = {} should be positive",
                    xxt_vals[i][i]
                );
            }
        }
        Ok(())
    }
}

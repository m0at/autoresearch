use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor,
};
use half::bf16;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;

const EPS: f32 = 1e-6;
const METAL_SOURCE: &str = include_str!("../metal/fused_ops.metal");

// Pipeline cache: compile each kernel once per process lifetime.
static PIPELINE_CACHE: OnceLock<Mutex<HashMap<String, ComputePipeline>>> = OnceLock::new();

fn get_pipeline(
    device: &candle_metal_kernels::metal::Device,
    kernel_name: &str,
) -> Result<ComputePipeline> {
    let cache = PIPELINE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache
        .lock()
        .map_err(|e| candle_core::Error::Msg(format!("lock: {e}")))?;
    if let Some(p) = map.get(kernel_name) {
        return Ok(p.clone());
    }
    let library = device
        .new_library_with_source(METAL_SOURCE, None)
        .map_err(|e| candle_core::Error::Msg(format!("compile fused_ops.metal: {e}")))?;
    let function = library
        .get_function(kernel_name, None)
        .map_err(|e| candle_core::Error::Msg(format!("get_function({kernel_name}): {e}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Msg(format!("pipeline({kernel_name}): {e}")))?;
    map.insert(kernel_name.to_string(), pipeline.clone());
    Ok(pipeline)
}

// ── CPU helpers ─────────────────────────────────────────────────────────────

fn extract_f32_data(storage: &CpuStorage, layout: &Layout) -> Result<Vec<f32>> {
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("non-contiguous layout in cpu_fwd".into()))?;
    match storage {
        CpuStorage::F32(data) => Ok(data[start..end].to_vec()),
        CpuStorage::BF16(data) => Ok(data[start..end].iter().map(|v| v.to_f32()).collect()),
        _ => Err(candle_core::Error::Msg(
            "unsupported dtype for fused op".into(),
        )),
    }
}

fn to_cpu_storage(data: Vec<f32>, dtype: DType) -> CpuStorage {
    match dtype {
        DType::F32 => CpuStorage::F32(data),
        DType::BF16 => CpuStorage::BF16(data.into_iter().map(bf16::from_f32).collect()),
        _ => CpuStorage::F32(data),
    }
}

fn storage_dtype(storage: &CpuStorage) -> DType {
    match storage {
        CpuStorage::F32(_) => DType::F32,
        CpuStorage::BF16(_) => DType::BF16,
        _ => DType::F32,
    }
}

// ── FusedRmsNormOp ──────────────────────────────────────────────────────────

struct FusedRmsNormOp;

impl CustomOp1 for FusedRmsNormOp {
    fn name(&self) -> &'static str {
        "fused_rms_norm_fwd"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(storage);
        let data = extract_f32_data(storage, layout)?;
        let shape = layout.shape().clone();
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = data.len() / d;
        let mut out = vec![0f32; data.len()];
        for r in 0..rows {
            let row = &data[r * d..(r + 1) * d];
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let rrms = 1.0 / (sum_sq / d as f32 + EPS).sqrt();
            for i in 0..d {
                out[r * d + i] = row[i] * rrms;
            }
        }
        Ok((to_cpu_storage(out, dtype), shape))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();
        let shape = layout.shape().clone();
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = shape.elem_count() / d;
        let out_buf = device.new_buffer(shape.elem_count(), storage.dtype(), "rms_norm_out")?;
        let pipeline = get_pipeline(device.device(), "fused_rms_norm_fwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let offset = layout.start_offset() * storage.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(storage.buffer()), offset);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_bytes(2, &(d as u32));
        encoder.set_bytes(3, &EPS);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: rows,
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
        let count = shape.elem_count();
        Ok((
            MetalStorage::new(out_buf, device.clone(), count, storage.dtype()),
            shape,
        ))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let x = if arg.is_contiguous() {
            arg.clone()
        } else {
            arg.contiguous()?
        };
        let g = if grad_res.is_contiguous() {
            grad_res.clone()
        } else {
            grad_res.contiguous()?
        };
        let grad_in = x.apply_op2_no_bwd(&g, &FusedRmsNormBwdOp)?;
        Ok(Some(grad_in))
    }
}

// ── FusedRmsNormBwdOp ───────────────────────────────────────────────────────

struct FusedRmsNormBwdOp;

impl CustomOp2 for FusedRmsNormBwdOp {
    fn name(&self) -> &'static str {
        "fused_rms_norm_bwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(s1);
        let x = extract_f32_data(s1, l1)?;
        let grad_out = extract_f32_data(s2, l2)?;
        let shape = l1.shape().clone();
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = x.len() / d;
        let mut grad_in = vec![0f32; x.len()];
        for r in 0..rows {
            let xr = &x[r * d..(r + 1) * d];
            let gr = &grad_out[r * d..(r + 1) * d];
            let sum_sq: f32 = xr.iter().map(|v| v * v).sum();
            let dot_gx: f32 = xr.iter().zip(gr.iter()).map(|(a, b)| a * b).sum();
            let rrms = 1.0 / (sum_sq / d as f32 + EPS).sqrt();
            let coeff = dot_gx / d as f32 * rrms * rrms;
            for i in 0..d {
                grad_in[r * d + i] = rrms * (gr[i] - xr[i] * coeff);
            }
        }
        Ok((to_cpu_storage(grad_in, dtype), shape))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device();
        let shape = l1.shape().clone();
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = shape.elem_count() / d;
        let out_buf = device.new_buffer(shape.elem_count(), s1.dtype(), "rms_norm_bwd_out")?;
        let pipeline = get_pipeline(device.device(), "fused_rms_norm_bwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(s1.buffer()), off1);
        encoder.set_buffer(1, Some(s2.buffer()), off2);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_bytes(3, &(d as u32));
        encoder.set_bytes(4, &EPS);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        let count = shape.elem_count();
        Ok((
            MetalStorage::new(out_buf, device.clone(), count, s1.dtype()),
            shape,
        ))
    }
}

// ── FusedSigmoid2xOp ───────────────────────────────────────────────────────

struct FusedSigmoid2xOp;

impl CustomOp1 for FusedSigmoid2xOp {
    fn name(&self) -> &'static str {
        "fused_sigmoid_2x_fwd"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(storage);
        let data = extract_f32_data(storage, layout)?;
        let shape = layout.shape().clone();
        let out: Vec<f32> = data.iter().map(|&v| 2.0 / (1.0 + (-v).exp())).collect();
        Ok((to_cpu_storage(out, dtype), shape))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();
        let shape = layout.shape().clone();
        let n = shape.elem_count();
        let out_buf = device.new_buffer(n, storage.dtype(), "sigmoid_2x_out")?;
        let pipeline = get_pipeline(device.device(), "fused_sigmoid_2x_fwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let offset = layout.start_offset() * storage.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(storage.buffer()), offset);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_bytes(2, &(n as u32));
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
            MetalStorage::new(out_buf, device.clone(), n, storage.dtype()),
            shape,
        ))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let x = if arg.is_contiguous() {
            arg.clone()
        } else {
            arg.contiguous()?
        };
        let g = if grad_res.is_contiguous() {
            grad_res.clone()
        } else {
            grad_res.contiguous()?
        };
        let grad_in = x.apply_op2_no_bwd(&g, &FusedSigmoid2xBwdOp)?;
        Ok(Some(grad_in))
    }
}

// ── FusedSigmoid2xBwdOp ────────────────────────────────────────────────────

struct FusedSigmoid2xBwdOp;

impl CustomOp2 for FusedSigmoid2xBwdOp {
    fn name(&self) -> &'static str {
        "fused_sigmoid_2x_bwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(s1);
        let x = extract_f32_data(s1, l1)?;
        let grad_out = extract_f32_data(s2, l2)?;
        let shape = l1.shape().clone();
        let out: Vec<f32> = x
            .iter()
            .zip(grad_out.iter())
            .map(|(&v, &g)| {
                let sig = 1.0 / (1.0 + (-v).exp());
                2.0 * sig * (1.0 - sig) * g
            })
            .collect();
        Ok((to_cpu_storage(out, dtype), shape))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device();
        let shape = l1.shape().clone();
        let n = shape.elem_count();
        let out_buf = device.new_buffer(n, s1.dtype(), "sigmoid_2x_bwd_out")?;
        let pipeline = get_pipeline(device.device(), "fused_sigmoid_2x_bwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(s1.buffer()), off1);
        encoder.set_buffer(1, Some(s2.buffer()), off2);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_bytes(3, &(n as u32));
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
        encoder.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(out_buf, device.clone(), n, s1.dtype()),
            shape,
        ))
    }
}

// ── FusedReluSqOp ───────────────────────────────────────────────────────────

struct FusedReluSqOp;

impl CustomOp1 for FusedReluSqOp {
    fn name(&self) -> &'static str {
        "fused_relu_sq_fwd"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(storage);
        let data = extract_f32_data(storage, layout)?;
        let shape = layout.shape().clone();
        let out: Vec<f32> = data
            .iter()
            .map(|&v| {
                let r = v.max(0.0);
                r * r
            })
            .collect();
        Ok((to_cpu_storage(out, dtype), shape))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();
        let shape = layout.shape().clone();
        let n = shape.elem_count();
        let out_buf = device.new_buffer(n, storage.dtype(), "relu_sq_out")?;
        let pipeline = get_pipeline(device.device(), "fused_relu_sq_fwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let offset = layout.start_offset() * storage.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(storage.buffer()), offset);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_bytes(2, &(n as u32));
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
            MetalStorage::new(out_buf, device.clone(), n, storage.dtype()),
            shape,
        ))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let x = if arg.is_contiguous() {
            arg.clone()
        } else {
            arg.contiguous()?
        };
        let g = if grad_res.is_contiguous() {
            grad_res.clone()
        } else {
            grad_res.contiguous()?
        };
        let grad_in = x.apply_op2_no_bwd(&g, &FusedReluSqBwdOp)?;
        Ok(Some(grad_in))
    }
}

// ── FusedReluSqBwdOp ────────────────────────────────────────────────────────

struct FusedReluSqBwdOp;

impl CustomOp2 for FusedReluSqBwdOp {
    fn name(&self) -> &'static str {
        "fused_relu_sq_bwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(s1);
        let x = extract_f32_data(s1, l1)?;
        let grad_out = extract_f32_data(s2, l2)?;
        let shape = l1.shape().clone();
        let out: Vec<f32> = x
            .iter()
            .zip(grad_out.iter())
            .map(|(&v, &g)| 2.0 * v.max(0.0) * g)
            .collect();
        Ok((to_cpu_storage(out, dtype), shape))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device();
        let shape = l1.shape().clone();
        let n = shape.elem_count();
        let out_buf = device.new_buffer(n, s1.dtype(), "relu_sq_bwd_out")?;
        let pipeline = get_pipeline(device.device(), "fused_relu_sq_bwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(s1.buffer()), off1);
        encoder.set_buffer(1, Some(s2.buffer()), off2);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_bytes(3, &(n as u32));
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
        encoder.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(out_buf, device.clone(), n, s1.dtype()),
            shape,
        ))
    }
}

// ── FusedScaleBwdOp ─────────────────────────────────────────────────────
// grad_in = lambda * grad_out  (used for both x and x0 backward paths)

struct FusedScaleBwdOp {
    lambda: f32,
}

impl CustomOp1 for FusedScaleBwdOp {
    fn name(&self) -> &'static str {
        "fused_scale_bwd"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(storage);
        let data = extract_f32_data(storage, layout)?;
        let shape = layout.shape().clone();
        let s = self.lambda;
        let out: Vec<f32> = data.iter().map(|&v| s * v).collect();
        Ok((to_cpu_storage(out, dtype), shape))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();
        let shape = layout.shape().clone();
        let n = shape.elem_count();
        let out_buf = device.new_buffer(n, storage.dtype(), "scale_bwd_out")?;
        let pipeline = get_pipeline(device.device(), "fused_scale_bwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let offset = layout.start_offset() * storage.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(storage.buffer()), offset);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_bytes(2, &self.lambda);
        encoder.set_bytes(3, &(n as u32));
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
            MetalStorage::new(out_buf, device.clone(), n, storage.dtype()),
            shape,
        ))
    }
}

// ── FusedResidualScaleWithGrad ──────────────────────────────────────────
// Wrapper that provides backward through apply_op2.

struct FusedResidualScaleWithGrad {
    lambda_r: f32,
    lambda_0: f32,
}

impl CustomOp2 for FusedResidualScaleWithGrad {
    fn name(&self) -> &'static str {
        "fused_residual_scale"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(s1);
        let x = extract_f32_data(s1, l1)?;
        let x0 = extract_f32_data(s2, l2)?;
        let shape = l1.shape().clone();
        let lr = self.lambda_r;
        let l0 = self.lambda_0;
        let out: Vec<f32> = x
            .iter()
            .zip(x0.iter())
            .map(|(&a, &b)| lr * a + l0 * b)
            .collect();
        Ok((to_cpu_storage(out, dtype), shape))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device();
        let shape = l1.shape().clone();
        let n = shape.elem_count();
        let out_buf = device.new_buffer(n, s1.dtype(), "residual_scale_out")?;
        let pipeline = get_pipeline(device.device(), "fused_residual_scale_fwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(s1.buffer()), off1);
        encoder.set_buffer(1, Some(s2.buffer()), off2);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_bytes(3, &self.lambda_r);
        encoder.set_bytes(4, &self.lambda_0);
        encoder.set_bytes(5, &(n as u32));
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
        encoder.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(out_buf, device.clone(), n, s1.dtype()),
            shape,
        ))
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let g = if grad_res.is_contiguous() {
            grad_res.clone()
        } else {
            grad_res.contiguous()?
        };
        let grad_x = g.apply_op1_no_bwd(&FusedScaleBwdOp {
            lambda: self.lambda_r,
        })?;
        let grad_x0 = g.apply_op1_no_bwd(&FusedScaleBwdOp {
            lambda: self.lambda_0,
        })?;
        let _ = (arg1, arg2); // not needed for linear backward
        Ok((Some(grad_x), Some(grad_x0)))
    }
}

// ── FusedSoftmaxOp ──────────────────────────────────────────────────────────

struct FusedSoftmaxOp;

impl CustomOp1 for FusedSoftmaxOp {
    fn name(&self) -> &'static str {
        "fused_softmax_fwd"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(storage);
        let data = extract_f32_data(storage, layout)?;
        let shape = layout.shape().clone();
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = data.len() / d;
        let mut out = vec![0f32; data.len()];
        for r in 0..rows {
            let row = &data[r * d..(r + 1) * d];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for i in 0..d {
                let e = (row[i] - m).exp();
                out[r * d + i] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for i in 0..d {
                out[r * d + i] *= inv;
            }
        }
        Ok((to_cpu_storage(out, dtype), shape))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();
        let shape = layout.shape().clone();
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = shape.elem_count() / d;
        let out_buf = device.new_buffer(shape.elem_count(), storage.dtype(), "softmax_out")?;
        let pipeline = get_pipeline(device.device(), "fused_softmax_fwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let offset = layout.start_offset() * storage.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(storage.buffer()), offset);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_bytes(2, &(d as u32));
        encoder.dispatch_thread_groups(
            MTLSize {
                width: rows,
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
        let count = shape.elem_count();
        Ok((
            MetalStorage::new(out_buf, device.clone(), count, storage.dtype()),
            shape,
        ))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let s = if res.is_contiguous() {
            res.clone()
        } else {
            res.contiguous()?
        };
        let g = if grad_res.is_contiguous() {
            grad_res.clone()
        } else {
            grad_res.contiguous()?
        };
        let grad_in = s.apply_op2_no_bwd(&g, &FusedSoftmaxBwdOp)?;
        Ok(Some(grad_in))
    }
}

// ── FusedSoftmaxBwdOp ──────────────────────────────────────────────────────

struct FusedSoftmaxBwdOp;

impl CustomOp2 for FusedSoftmaxBwdOp {
    fn name(&self) -> &'static str {
        "fused_softmax_bwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(s1);
        let s = extract_f32_data(s1, l1)?;
        let grad_out = extract_f32_data(s2, l2)?;
        let shape = l1.shape().clone();
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = s.len() / d;
        let mut grad_in = vec![0f32; s.len()];
        for r in 0..rows {
            let sr = &s[r * d..(r + 1) * d];
            let gr = &grad_out[r * d..(r + 1) * d];
            let dot_sg: f32 = sr.iter().zip(gr.iter()).map(|(a, b)| a * b).sum();
            for i in 0..d {
                grad_in[r * d + i] = sr[i] * (gr[i] - dot_sg);
            }
        }
        Ok((to_cpu_storage(grad_in, dtype), shape))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device();
        let shape = l1.shape().clone();
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = shape.elem_count() / d;
        let out_buf = device.new_buffer(shape.elem_count(), s1.dtype(), "softmax_bwd_out")?;
        let pipeline = get_pipeline(device.device(), "fused_softmax_bwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(s1.buffer()), off1);
        encoder.set_buffer(1, Some(s2.buffer()), off2);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_bytes(3, &(d as u32));
        encoder.dispatch_thread_groups(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        let count = shape.elem_count();
        Ok((
            MetalStorage::new(out_buf, device.clone(), count, s1.dtype()),
            shape,
        ))
    }
}

// ── FusedRopeOp ─────────────────────────────────────────────────────────────

struct FusedRopeOp;

impl CustomOp3 for FusedRopeOp {
    fn name(&self) -> &'static str {
        "fused_rope_fwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(s1);
        let x = extract_f32_data(s1, l1)?;
        let cos_data = extract_f32_data(s2, l2)?;
        let sin_data = extract_f32_data(s3, l3)?;
        let shape = l1.shape().clone();
        let dims = shape.dims();
        let (b, t, n_head, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
        let half = head_dim / 2;
        let mut out = vec![0f32; x.len()];
        for bi in 0..b {
            for ti in 0..t {
                for hi in 0..n_head {
                    let base = ((bi * t + ti) * n_head + hi) * head_dim;
                    let cs_base = ti * half;
                    for d in 0..half {
                        let c = cos_data[cs_base + d];
                        let s = sin_data[cs_base + d];
                        let x1 = x[base + d];
                        let x2 = x[base + d + half];
                        out[base + d] = x1 * c + x2 * s;
                        out[base + d + half] = -x1 * s + x2 * c;
                    }
                }
            }
        }
        Ok((to_cpu_storage(out, dtype), shape))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
        s3: &MetalStorage,
        l3: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device();
        let shape = l1.shape().clone();
        let dims = shape.dims();
        let (t, n_head, head_dim) = (dims[1], dims[2], dims[3]);
        let n = shape.elem_count();
        let out_buf = device.new_buffer(n, s1.dtype(), "rope_fwd_out")?;
        let pipeline = get_pipeline(device.device(), "fused_rope_fwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            Some(s1.buffer()),
            l1.start_offset() * s1.dtype().size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(s2.buffer()),
            l2.start_offset() * s2.dtype().size_in_bytes(),
        );
        encoder.set_buffer(
            2,
            Some(s3.buffer()),
            l3.start_offset() * s3.dtype().size_in_bytes(),
        );
        encoder.set_buffer(3, Some(&out_buf), 0);
        encoder.set_bytes(4, &(n as u32));
        encoder.set_bytes(5, &(t as u32));
        encoder.set_bytes(6, &(n_head as u32));
        encoder.set_bytes(7, &(head_dim as u32));
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
        encoder.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s3.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(out_buf, device.clone(), n, s1.dtype()),
            shape,
        ))
    }

    fn bwd(
        &self,
        _x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        _res: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let g = if grad.is_contiguous() {
            grad.clone()
        } else {
            grad.contiguous()?
        };
        let c = if cos.is_contiguous() {
            cos.clone()
        } else {
            cos.contiguous()?
        };
        let s = if sin.is_contiguous() {
            sin.clone()
        } else {
            sin.contiguous()?
        };
        let grad_x = g.apply_op3_no_bwd(&c, &s, &FusedRopeBwdOp)?;
        Ok((Some(grad_x), None, None))
    }
}

struct FusedRopeBwdOp;

impl CustomOp3 for FusedRopeBwdOp {
    fn name(&self) -> &'static str {
        "fused_rope_bwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(s1);
        let g = extract_f32_data(s1, l1)?;
        let cos_data = extract_f32_data(s2, l2)?;
        let sin_data = extract_f32_data(s3, l3)?;
        let shape = l1.shape().clone();
        let dims = shape.dims();
        let (b, t, n_head, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
        let half = head_dim / 2;
        let mut grad_in = vec![0f32; g.len()];
        for bi in 0..b {
            for ti in 0..t {
                for hi in 0..n_head {
                    let base = ((bi * t + ti) * n_head + hi) * head_dim;
                    let cs_base = ti * half;
                    for d in 0..half {
                        let c = cos_data[cs_base + d];
                        let s = sin_data[cs_base + d];
                        let g1 = g[base + d];
                        let g2 = g[base + d + half];
                        grad_in[base + d] = g1 * c - g2 * s;
                        grad_in[base + d + half] = g1 * s + g2 * c;
                    }
                }
            }
        }
        Ok((to_cpu_storage(grad_in, dtype), shape))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
        s3: &MetalStorage,
        l3: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device();
        let shape = l1.shape().clone();
        let dims = shape.dims();
        let (t, n_head, head_dim) = (dims[1], dims[2], dims[3]);
        let n = shape.elem_count();
        let out_buf = device.new_buffer(n, s1.dtype(), "rope_bwd_out")?;
        let pipeline = get_pipeline(device.device(), "fused_rope_bwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            Some(s1.buffer()),
            l1.start_offset() * s1.dtype().size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(s2.buffer()),
            l2.start_offset() * s2.dtype().size_in_bytes(),
        );
        encoder.set_buffer(
            2,
            Some(s3.buffer()),
            l3.start_offset() * s3.dtype().size_in_bytes(),
        );
        encoder.set_buffer(3, Some(&out_buf), 0);
        encoder.set_bytes(4, &(n as u32));
        encoder.set_bytes(5, &(t as u32));
        encoder.set_bytes(6, &(n_head as u32));
        encoder.set_bytes(7, &(head_dim as u32));
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
        encoder.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s3.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(out_buf, device.clone(), n, s1.dtype()),
            shape,
        ))
    }
}

// ── FusedQkNormOp ────────────────────────────────────────────────────────────
// Fused RMS normalization of Q and K in a single kernel dispatch.
// CustomOp2: arg1 = q_flat (q_rows, D), arg2 = k_flat (k_rows, D)
// Output: concatenated (q_rows + k_rows, D), caller splits.

struct FusedQkNormOp;

impl CustomOp2 for FusedQkNormOp {
    fn name(&self) -> &'static str {
        "fused_qk_norm_fwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(s1);
        let q = extract_f32_data(s1, l1)?;
        let k = extract_f32_data(s2, l2)?;
        let d = *l1.shape().dims().last().unwrap();
        let q_rows = q.len() / d;
        let k_rows = k.len() / d;
        let total_rows = q_rows + k_rows;
        let mut out = vec![0f32; total_rows * d];
        for r in 0..total_rows {
            let src = if r < q_rows {
                &q[r * d..(r + 1) * d]
            } else {
                &k[(r - q_rows) * d..(r - q_rows + 1) * d]
            };
            let sum_sq: f32 = src.iter().map(|v| v * v).sum();
            let rrms = 1.0 / (sum_sq / d as f32 + EPS).sqrt();
            for i in 0..d {
                out[r * d + i] = src[i] * rrms;
            }
        }
        let shape = Shape::from_dims(&[total_rows, d]);
        Ok((to_cpu_storage(out, dtype), shape))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device();
        let d = *l1.shape().dims().last().unwrap();
        let q_rows = l1.shape().elem_count() / d;
        let k_rows = l2.shape().elem_count() / d;
        let total_rows = q_rows + k_rows;
        let total_elems = total_rows * d;
        let out_buf = device.new_buffer(total_elems, s1.dtype(), "qk_norm_out")?;
        let pipeline = get_pipeline(device.device(), "fused_qk_norm_fwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(s1.buffer()), off1);
        encoder.set_buffer(1, Some(s2.buffer()), off2);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_bytes(3, &(d as u32));
        encoder.set_bytes(4, &(q_rows as u32));
        encoder.set_bytes(5, &EPS);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: total_rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        let shape = Shape::from_dims(&[total_rows, d]);
        Ok((
            MetalStorage::new(out_buf, device.clone(), total_elems, s1.dtype()),
            shape,
        ))
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let q = if arg1.is_contiguous() {
            arg1.clone()
        } else {
            arg1.contiguous()?
        };
        let k = if arg2.is_contiguous() {
            arg2.clone()
        } else {
            arg2.contiguous()?
        };
        let g = if grad_res.is_contiguous() {
            grad_res.clone()
        } else {
            grad_res.contiguous()?
        };
        let q_rows = q.dim(0)?;
        // Concatenate originals, apply rms_norm backward on all rows, split
        let original = Tensor::cat(&[&q, &k], 0)?;
        let grad_concat = original.apply_op2_no_bwd(&g, &FusedRmsNormBwdOp)?;
        let grad_q = grad_concat.narrow(0, 0, q_rows)?;
        let grad_k = grad_concat.narrow(0, q_rows, grad_concat.dim(0)? - q_rows)?;
        Ok((Some(grad_q.contiguous()?), Some(grad_k.contiguous()?)))
    }
}

// ── FusedEmbedOp ────────────────────────────────────────────────────────────

struct FusedEmbedOp {
    seq_len: usize,
    seq_offset: usize,
}

fn extract_u32_data(storage: &CpuStorage, layout: &Layout) -> Result<Vec<u32>> {
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("non-contiguous layout for u32".into()))?;
    match storage {
        CpuStorage::U32(data) => Ok(data[start..end].to_vec()),
        _ => Err(candle_core::Error::Msg(
            "expected U32 storage for token_ids".into(),
        )),
    }
}

impl CustomOp3 for FusedEmbedOp {
    fn name(&self) -> &'static str {
        "fused_embed_fwd"
    }

    fn cpu_fwd(
        &self,
        s_ids: &CpuStorage,
        l_ids: &Layout,
        s_tok: &CpuStorage,
        l_tok: &Layout,
        s_pos: &CpuStorage,
        l_pos: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(s_tok);
        let ids = extract_u32_data(s_ids, l_ids)?;
        let tok = extract_f32_data(s_tok, l_tok)?;
        let pos = extract_f32_data(s_pos, l_pos)?;
        let d = l_tok.shape().dims().last().copied().unwrap_or(0);
        let bt = ids.len();
        let t = self.seq_len;
        let mut out = vec![0f32; bt * d];
        for i in 0..bt {
            let tok_id = ids[i] as usize;
            let p = self.seq_offset + (i % t);
            for j in 0..d {
                out[i * d + j] = tok[tok_id * d + j] + pos[p * d + j];
            }
        }
        Ok((to_cpu_storage(out, dtype), Shape::from_dims(&[bt, d])))
    }

    fn metal_fwd(
        &self,
        s_ids: &MetalStorage,
        l_ids: &Layout,
        s_tok: &MetalStorage,
        l_tok: &Layout,
        s_pos: &MetalStorage,
        l_pos: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s_tok.device();
        let d = l_tok.shape().dims().last().copied().unwrap_or(0);
        let bt = l_ids.shape().elem_count();
        let n = bt * d;
        let out_shape = Shape::from_dims(&[bt, d]);
        let out_buf = device.new_buffer(n, s_tok.dtype(), "embed_out")?;
        let pipeline = get_pipeline(device.device(), "fused_embed_fwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            Some(s_ids.buffer()),
            l_ids.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(s_tok.buffer()),
            l_tok.start_offset() * s_tok.dtype().size_in_bytes(),
        );
        encoder.set_buffer(
            2,
            Some(s_pos.buffer()),
            l_pos.start_offset() * s_pos.dtype().size_in_bytes(),
        );
        encoder.set_buffer(3, Some(&out_buf), 0);
        encoder.set_bytes(4, &(self.seq_len as u32));
        encoder.set_bytes(5, &(d as u32));
        encoder.set_bytes(6, &(n as u32));
        encoder.set_bytes(7, &(self.seq_offset as u32));
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
        encoder.use_resource(s_ids.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s_tok.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s_pos.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(out_buf, device.clone(), n, s_tok.dtype()),
            out_shape,
        ))
    }

    fn bwd(
        &self,
        ids: &Tensor,
        tok_emb: &Tensor,
        pos_emb: &Tensor,
        _res: &Tensor,
        grad_out: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let grad_out = if grad_out.is_contiguous() {
            grad_out.clone()
        } else {
            grad_out.contiguous()?
        };
        let ids = if ids.is_contiguous() {
            ids.clone()
        } else {
            ids.contiguous()?
        };
        let vocab_size = tok_emb.dim(0)?;
        let seq_max = pos_emb.dim(0)?;
        let d = tok_emb.dim(1)?;
        let bt = ids.elem_count();
        let t = self.seq_len;
        let ids_flat = ids.flatten_all()?;
        let grad_flat = grad_out.reshape((bt, d))?;
        let grad_tok = Tensor::zeros((vocab_size, d), grad_out.dtype(), grad_out.device())?;
        let grad_tok = grad_tok.index_add(&ids_flat, &grad_flat, 0)?;
        let positions: Vec<u32> = (0..bt)
            .map(|i| (self.seq_offset + (i % t)) as u32)
            .collect();
        let pos_ids = Tensor::new(positions, grad_out.device())?;
        let grad_pos = Tensor::zeros((seq_max, d), grad_out.dtype(), grad_out.device())?;
        let grad_pos = grad_pos.index_add(&pos_ids, &grad_flat, 0)?;
        Ok((None, Some(grad_tok), Some(grad_pos)))
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

pub fn fused_qk_norm(q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
    let q_shape = q.shape().clone();
    let k_shape = k.shape().clone();
    let d = *q_shape.dims().last().unwrap();
    let q_rows = q_shape.elem_count() / d;
    let k_rows = k_shape.elem_count() / d;

    let q_flat = if q.is_contiguous() {
        q.reshape((q_rows, d))?
    } else {
        q.contiguous()?.reshape((q_rows, d))?
    };
    let k_flat = if k.is_contiguous() {
        k.reshape((k_rows, d))?
    } else {
        k.contiguous()?.reshape((k_rows, d))?
    };

    let concat_out = q_flat.apply_op2(&k_flat, FusedQkNormOp)?;

    let normed_q = concat_out.narrow(0, 0, q_rows)?.reshape(q_shape)?;
    let normed_k = concat_out.narrow(0, q_rows, k_rows)?.reshape(k_shape)?;
    Ok((normed_q, normed_k))
}

pub fn fused_softmax(x: &Tensor) -> Result<Tensor> {
    let x = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()?
    };
    x.apply_op1(FusedSoftmaxOp)
}

pub fn fused_rms_norm(x: &Tensor) -> Result<Tensor> {
    let x = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()?
    };
    x.apply_op1(FusedRmsNormOp)
}

pub fn fused_sigmoid_2x(x: &Tensor) -> Result<Tensor> {
    let x = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()?
    };
    x.apply_op1(FusedSigmoid2xOp)
}

pub fn fused_relu_sq(x: &Tensor) -> Result<Tensor> {
    let x = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()?
    };
    x.apply_op1(FusedReluSqOp)
}

pub fn fused_residual_scale(
    x: &Tensor,
    x0: &Tensor,
    lambda_r: f32,
    lambda_0: f32,
) -> Result<Tensor> {
    let x = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()?
    };
    let x0 = if x0.is_contiguous() {
        x0.clone()
    } else {
        x0.contiguous()?
    };
    x.apply_op2(&x0, FusedResidualScaleWithGrad { lambda_r, lambda_0 })
}

pub fn fused_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let x = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()?
    };
    let cos_shape = cos.dims().to_vec();
    let t = cos_shape[1];
    let half = *cos_shape.last().unwrap();
    let cos = cos.reshape((t, half))?;
    let sin = sin.reshape((t, half))?;
    let cos = if cos.is_contiguous() {
        cos
    } else {
        cos.contiguous()?
    };
    let sin = if sin.is_contiguous() {
        sin
    } else {
        sin.contiguous()?
    };
    x.apply_op3(&cos, &sin, FusedRopeOp)
}

// ── FusedCrossEntropyOp ─────────────────────────────────────────────────

struct FusedCrossEntropyOp {
    vocab_size: usize,
}

impl CustomOp2 for FusedCrossEntropyOp {
    fn name(&self) -> &'static str {
        "fused_cross_entropy_fwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let logits = extract_f32_data(s1, l1)?;
        let targets = extract_u32_data(s2, l2)?;
        let v = self.vocab_size;
        let n = targets.len();
        let mut losses = vec![0f32; n];
        for row in 0..n {
            let xr = &logits[row * v..(row + 1) * v];
            let tgt = targets[row] as usize;
            let max_val = xr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = xr.iter().map(|&x| (x - max_val).exp()).sum();
            losses[row] = -(xr[tgt] - max_val) + sum_exp.ln();
        }
        Ok((CpuStorage::F32(losses), Shape::from_dims(&[n])))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device();
        let n = l2.shape().elem_count();
        let v = self.vocab_size as u32;
        let out_buf = device.new_buffer(n, DType::F32, "ce_loss_out")?;
        let pipeline = get_pipeline(device.device(), "fused_cross_entropy_fwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(s1.buffer()), off1);
        encoder.set_buffer(1, Some(s2.buffer()), off2);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_bytes(3, &v);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: n,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(out_buf, device.clone(), n, DType::F32),
            Shape::from_dims(&[n]),
        ))
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let logits = if arg1.is_contiguous() {
            arg1.clone()
        } else {
            arg1.contiguous()?
        };
        let targets = if arg2.is_contiguous() {
            arg2.clone()
        } else {
            arg2.contiguous()?
        };
        let g = if grad_res.is_contiguous() {
            grad_res.clone()
        } else {
            grad_res.contiguous()?
        };
        let grad_logits = logits.apply_op3_no_bwd(
            &targets,
            &g,
            &FusedCrossEntropyBwdOp {
                vocab_size: self.vocab_size,
            },
        )?;
        Ok((Some(grad_logits), None))
    }
}

// ── FusedCrossEntropyBwdOp ──────────────────────────────────────────────

struct FusedCrossEntropyBwdOp {
    vocab_size: usize,
}

impl CustomOp3 for FusedCrossEntropyBwdOp {
    fn name(&self) -> &'static str {
        "fused_cross_entropy_bwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = storage_dtype(s1);
        let logits = extract_f32_data(s1, l1)?;
        let targets = extract_u32_data(s2, l2)?;
        let grad_res = extract_f32_data(s3, l3)?;
        let v = self.vocab_size;
        let n = targets.len();
        let mut grad_in = vec![0f32; n * v];
        for row in 0..n {
            let xr = &logits[row * v..(row + 1) * v];
            let tgt = targets[row] as usize;
            let scale = grad_res[row];
            let max_val = xr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = xr.iter().map(|&x| (x - max_val).exp()).sum();
            let inv_sum = 1.0 / sum_exp;
            for j in 0..v {
                let p = (xr[j] - max_val).exp() * inv_sum;
                let indicator = if j == tgt { 1.0 } else { 0.0 };
                grad_in[row * v + j] = (p - indicator) * scale;
            }
        }
        Ok((to_cpu_storage(grad_in, dtype), l1.shape().clone()))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
        s3: &MetalStorage,
        l3: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device();
        let shape = l1.shape().clone();
        let n = l2.shape().elem_count();
        let v = self.vocab_size as u32;
        let out_buf = device.new_buffer(shape.elem_count(), s1.dtype(), "ce_grad_out")?;
        let pipeline = get_pipeline(device.device(), "fused_cross_entropy_bwd")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        let off3 = l3.start_offset() * s3.dtype().size_in_bytes();
        encoder.set_buffer(0, Some(s1.buffer()), off1);
        encoder.set_buffer(1, Some(s2.buffer()), off2);
        encoder.set_buffer(2, Some(s3.buffer()), off3);
        encoder.set_buffer(3, Some(&out_buf), 0);
        encoder.set_bytes(4, &v);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: n,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(s3.buffer(), objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(out_buf, device.clone(), shape.elem_count(), s1.dtype()),
            shape,
        ))
    }
}

/// Fused cross-entropy loss: returns scalar mean loss.
/// logits: (N, V) BF16 or F32, targets: (N,) U32
pub fn fused_cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let logits = if logits.is_contiguous() {
        logits.clone()
    } else {
        logits.contiguous()?
    };
    let targets = if targets.is_contiguous() {
        targets.clone()
    } else {
        targets.contiguous()?
    };
    let v = logits.dim(1)?;
    let per_row = logits.apply_op2(&targets, FusedCrossEntropyOp { vocab_size: v })?;
    per_row.mean_all()
}

/// Fused token + position embedding lookup.
/// token_ids: (B, T) U32, tok_emb: (V, D), pos_emb: (S, D)
/// Returns: (B*T, D) = tok_emb[ids] + pos_emb[positions]
pub fn fused_embed(
    token_ids: &Tensor,
    tok_emb: &Tensor,
    pos_emb: &Tensor,
    seq_offset: usize,
) -> Result<Tensor> {
    let (_b, t) = token_ids.dims2()?;
    let ids_flat = token_ids.flatten_all()?;
    let ids_flat = if ids_flat.is_contiguous() {
        ids_flat
    } else {
        ids_flat.contiguous()?
    };
    let tok_emb = if tok_emb.is_contiguous() {
        tok_emb.clone()
    } else {
        tok_emb.contiguous()?
    };
    let pos_emb = if pos_emb.is_contiguous() {
        pos_emb.clone()
    } else {
        pos_emb.contiguous()?
    };
    ids_flat.apply_op3(
        &tok_emb,
        &pos_emb,
        FusedEmbedOp {
            seq_len: t,
            seq_offset,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};

    #[test]
    fn test_rms_norm_basic() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?.reshape((1, 4))?;
        let y = fused_rms_norm(&x)?;
        let y_data: Vec<f32> = y.flatten_all()?.to_vec1()?;
        let sum_sq: f32 = [1.0, 4.0, 9.0, 16.0].iter().sum();
        let rrms = 1.0 / (sum_sq / 4.0 + EPS).sqrt();
        for (i, &v) in y_data.iter().enumerate() {
            let expected = (i + 1) as f32 * rrms;
            assert!(
                (v - expected).abs() < 1e-5,
                "mismatch at {i}: {v} vs {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_rms_norm_grad() -> Result<()> {
        let dev = &Device::Cpu;
        let x = candle_core::Var::new(&[[1.0f32, 2.0, 3.0, 4.0]], dev)?;
        let y = fused_rms_norm(x.as_tensor())?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;
        let grad = grads.get(x.as_tensor()).expect("grad should exist");
        let g: Vec<f32> = grad.flatten_all()?.to_vec1()?;
        assert_eq!(g.len(), 4);
        for &v in &g {
            assert!(v.is_finite(), "gradient should be finite: {v}");
        }
        Ok(())
    }

    #[test]
    fn test_sigmoid_2x_basic() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0, 5.0], dev)?;
        let y = fused_sigmoid_2x(&x)?;
        let y_data: Vec<f32> = y.to_vec1()?;
        let expected: Vec<f32> = [0.0f32, 1.0, -1.0, 5.0]
            .iter()
            .map(|&v| 2.0 / (1.0 + (-v).exp()))
            .collect();
        for (i, (&got, &exp)) in y_data.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "mismatch at {i}: {got} vs {exp}");
        }
        Ok(())
    }

    #[test]
    fn test_sigmoid_2x_grad() -> Result<()> {
        let dev = &Device::Cpu;
        let x = candle_core::Var::new(&[0.0f32, 1.0, -1.0], dev)?;
        let y = fused_sigmoid_2x(x.as_tensor())?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;
        let grad = grads.get(x.as_tensor()).expect("grad should exist");
        let g: Vec<f32> = grad.flatten_all()?.to_vec1()?;
        assert_eq!(g.len(), 3);
        for (i, &v) in g.iter().enumerate() {
            let xv = [0.0f32, 1.0, -1.0][i];
            let sig = 1.0 / (1.0 + (-xv).exp());
            let expected = 2.0 * sig * (1.0 - sig);
            assert!(
                (v - expected).abs() < 1e-5,
                "grad mismatch at {i}: {v} vs {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_relu_sq_basic() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0], dev)?;
        let y = fused_relu_sq(&x)?;
        let y_data: Vec<f32> = y.to_vec1()?;
        let expected = [0.0, 0.0, 0.0, 1.0, 4.0, 9.0];
        for (i, (&got, &exp)) in y_data.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "mismatch at {i}: {got} vs {exp}");
        }
        Ok(())
    }

    #[test]
    fn test_relu_sq_grad() -> Result<()> {
        let dev = &Device::Cpu;
        let x = candle_core::Var::new(&[-1.0f32, 0.0, 1.0, 2.0], dev)?;
        let y = fused_relu_sq(x.as_tensor())?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;
        let grad = grads.get(x.as_tensor()).expect("grad should exist");
        let g: Vec<f32> = grad.flatten_all()?.to_vec1()?;
        // d/dx[relu(x)^2] = 2*max(0,x)
        let expected = [0.0, 0.0, 2.0, 4.0];
        for (i, (&got, &exp)) in g.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "grad mismatch at {i}: {got} vs {exp}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_residual_scale_basic() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
        let x0 = Tensor::new(&[0.5f32, 1.0, 1.5, 2.0], dev)?;
        let y = fused_residual_scale(&x, &x0, 0.8, 0.2)?;
        let y_data: Vec<f32> = y.to_vec1()?;
        for (i, &v) in y_data.iter().enumerate() {
            let xi = (i + 1) as f32;
            let x0i = (i + 1) as f32 * 0.5;
            let expected = 0.8 * xi + 0.2 * x0i;
            assert!(
                (v - expected).abs() < 1e-5,
                "mismatch at {i}: {v} vs {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_residual_scale_grad() -> Result<()> {
        let dev = &Device::Cpu;
        let x = candle_core::Var::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?;
        let x0 = candle_core::Var::new(&[0.5f32, 1.0, 1.5, 2.0], dev)?;
        let y = fused_residual_scale(x.as_tensor(), x0.as_tensor(), 0.8, 0.2)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;
        let gx: Vec<f32> = grads
            .get(x.as_tensor())
            .expect("grad x")
            .flatten_all()?
            .to_vec1()?;
        let gx0: Vec<f32> = grads
            .get(x0.as_tensor())
            .expect("grad x0")
            .flatten_all()?
            .to_vec1()?;
        for (i, &v) in gx.iter().enumerate() {
            assert!((v - 0.8).abs() < 1e-5, "grad_x mismatch at {i}: {v} vs 0.8");
        }
        for (i, &v) in gx0.iter().enumerate() {
            assert!(
                (v - 0.2).abs() < 1e-5,
                "grad_x0 mismatch at {i}: {v} vs 0.2"
            );
        }
        Ok(())
    }

    #[test]
    fn test_rms_norm_multi_row() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], dev)?.reshape((2, 3))?;
        let y = fused_rms_norm(&x)?;
        assert_eq!(y.dims(), &[2, 3]);
        let y_data: Vec<f32> = y.flatten_all()?.to_vec1()?;
        // Row 0: [1, 0, 0], sum_sq=1, rrms=sqrt(3/(1+eps))
        let rrms0 = (3.0 / (1.0 + EPS)).sqrt();
        assert!((y_data[0] - rrms0).abs() < 1e-4);
        assert!(y_data[1].abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_qk_norm_basic() -> Result<()> {
        let dev = &Device::Cpu;
        let q = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dev)?.reshape((2, 4))?;
        let k = Tensor::new(&[0.5f32, 1.5, 2.5, 3.5], dev)?.reshape((1, 4))?;
        let (nq, nk) = fused_qk_norm(&q, &k)?;
        assert_eq!(nq.dims(), &[2, 4]);
        assert_eq!(nk.dims(), &[1, 4]);
        // Compare against separate rms_norm
        let nq_ref = fused_rms_norm(&q)?;
        let nk_ref = fused_rms_norm(&k)?;
        let nq_data: Vec<f32> = nq.flatten_all()?.to_vec1()?;
        let nq_ref_data: Vec<f32> = nq_ref.flatten_all()?.to_vec1()?;
        let nk_data: Vec<f32> = nk.flatten_all()?.to_vec1()?;
        let nk_ref_data: Vec<f32> = nk_ref.flatten_all()?.to_vec1()?;
        for (i, (&a, &b)) in nq_data.iter().zip(nq_ref_data.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "q mismatch at {i}: {a} vs {b}");
        }
        for (i, (&a, &b)) in nk_data.iter().zip(nk_ref_data.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "k mismatch at {i}: {a} vs {b}");
        }
        Ok(())
    }

    #[test]
    fn test_qk_norm_grad() -> Result<()> {
        let dev = &Device::Cpu;
        let q_var = candle_core::Var::new(&[[1.0f32, 2.0, 3.0, 4.0]], dev)?;
        let k_var = candle_core::Var::new(&[[0.5f32, 1.5, 2.5, 3.5]], dev)?;
        let (nq, nk) = fused_qk_norm(q_var.as_tensor(), k_var.as_tensor())?;
        let loss = (nq.sum_all()? + nk.sum_all()?)?;
        let grads = loss.backward()?;
        let gq = grads.get(q_var.as_tensor()).expect("q grad should exist");
        let gk = grads.get(k_var.as_tensor()).expect("k grad should exist");
        let gq_data: Vec<f32> = gq.flatten_all()?.to_vec1()?;
        let gk_data: Vec<f32> = gk.flatten_all()?.to_vec1()?;
        for &v in &gq_data {
            assert!(v.is_finite(), "q gradient should be finite: {v}");
        }
        for &v in &gk_data {
            assert!(v.is_finite(), "k gradient should be finite: {v}");
        }
        Ok(())
    }

    #[test]
    fn test_qk_norm_4d() -> Result<()> {
        let dev = &Device::Cpu;
        // Simulate (B, T, n_head, head_dim) = (1, 2, 2, 4)
        let q = Tensor::randn(0f32, 1f32, (1, 2, 2, 4), dev)?;
        let k = Tensor::randn(0f32, 1f32, (1, 2, 1, 4), dev)?;
        let (nq, nk) = fused_qk_norm(&q, &k)?;
        assert_eq!(nq.dims(), &[1, 2, 2, 4]);
        assert_eq!(nk.dims(), &[1, 2, 1, 4]);
        // Compare against separate rms_norm
        let nq_ref = fused_rms_norm(&q)?;
        let nk_ref = fused_rms_norm(&k)?;
        let nq_data: Vec<f32> = nq.flatten_all()?.to_vec1()?;
        let nq_ref_data: Vec<f32> = nq_ref.flatten_all()?.to_vec1()?;
        for (i, (&a, &b)) in nq_data.iter().zip(nq_ref_data.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "4d q mismatch at {i}: {a} vs {b}");
        }
        let nk_data: Vec<f32> = nk.flatten_all()?.to_vec1()?;
        let nk_ref_data: Vec<f32> = nk_ref.flatten_all()?.to_vec1()?;
        for (i, (&a, &b)) in nk_data.iter().zip(nk_ref_data.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "4d k mismatch at {i}: {a} vs {b}");
        }
        Ok(())
    }

    #[test]
    fn test_softmax_basic() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], dev)?.reshape((1, 4))?;
        let y = fused_softmax(&x)?;
        let y_data: Vec<f32> = y.flatten_all()?.to_vec1()?;
        let sum: f32 = y_data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );
        for i in 1..4 {
            assert!(y_data[i] > y_data[i - 1], "softmax should be monotonic");
        }
        Ok(())
    }

    #[test]
    fn test_softmax_multi_row() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0], dev)?.reshape((2, 3))?;
        let y = fused_softmax(&x)?;
        let y_data: Vec<f32> = y.flatten_all()?.to_vec1()?;
        for i in 3..6 {
            assert!(
                (y_data[i] - 1.0 / 3.0).abs() < 1e-5,
                "uniform row: got {}",
                y_data[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_softmax_grad() -> Result<()> {
        let dev = &Device::Cpu;
        let x = candle_core::Var::new(&[[1.0f32, 2.0, 3.0, 4.0]], dev)?;
        let y = fused_softmax(x.as_tensor())?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;
        let grad = grads.get(x.as_tensor()).expect("grad should exist");
        let g: Vec<f32> = grad.flatten_all()?.to_vec1()?;
        assert_eq!(g.len(), 4);
        for (i, &v) in g.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "softmax grad of sum should be ~0, got {v} at {i}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_softmax_grad_nontrivial() -> Result<()> {
        let dev = &Device::Cpu;
        let x = candle_core::Var::new(&[[1.0f32, 2.0, 3.0]], dev)?;
        let y = fused_softmax(x.as_tensor())?;
        let loss = y.i((.., 0..1))?.sum_all()?;
        let grads = loss.backward()?;
        let grad = grads.get(x.as_tensor()).expect("grad should exist");
        let g: Vec<f32> = grad.flatten_all()?.to_vec1()?;
        for v in &g {
            assert!(v.is_finite(), "gradient should be finite: {v}");
        }
        assert!(g[0] > 0.0f32, "grad[0] should be positive");
        assert!(g[1] < 0.0f32, "grad[1] should be negative");
        assert!(g[2] < 0.0f32, "grad[2] should be negative");
        Ok(())
    }

    #[test]
    fn test_rope_fwd_identity() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, t, nh, hd) = (1, 4, 2, 8);
        let half = hd / 2;
        let x = Tensor::new(
            &(0..b * t * nh * hd).map(|i| i as f32).collect::<Vec<_>>()[..],
            dev,
        )?
        .reshape((b, t, nh, hd))?;
        let cos = Tensor::ones((1, t, 1, half), DType::F32, dev)?;
        let sin = Tensor::zeros((1, t, 1, half), DType::F32, dev)?;
        let y = fused_rotary_emb(&x, &cos, &sin)?;
        assert_eq!(y.dims(), &[b, t, nh, hd]);
        let x_data: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let y_data: Vec<f32> = y.flatten_all()?.to_vec1()?;
        for (i, (&xv, &yv)) in x_data.iter().zip(y_data.iter()).enumerate() {
            assert!(
                (xv - yv).abs() < 1e-5,
                "identity mismatch at {i}: {xv} vs {yv}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_rope_fwd_matches_naive() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, t, nh, hd) = (2, 4, 2, 8);
        let half = hd / 2;
        let x = Tensor::randn(0f32, 1f32, (b, t, nh, hd), dev)?;
        let cos = Tensor::randn(0f32, 1f32, (1, t, 1, half), dev)?;
        let sin = Tensor::randn(0f32, 1f32, (1, t, 1, half), dev)?;
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;
        let y1_naive = (x1.broadcast_mul(&cos)? + x2.broadcast_mul(&sin)?)?;
        let y2_naive = (x1.broadcast_mul(&sin.neg()?)? + x2.broadcast_mul(&cos)?)?;
        let y_naive = Tensor::cat(&[&y1_naive, &y2_naive], 3)?;
        let y_fused = fused_rotary_emb(&x, &cos, &sin)?;
        let n: Vec<f32> = y_naive.flatten_all()?.to_vec1()?;
        let f: Vec<f32> = y_fused.flatten_all()?.to_vec1()?;
        for (i, (&nv, &fv)) in n.iter().zip(f.iter()).enumerate() {
            assert!(
                (nv - fv).abs() < 1e-5,
                "naive vs fused at {i}: {nv} vs {fv}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_rope_grad() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, t, nh, hd) = (1, 4, 2, 8);
        let half = hd / 2;
        let x_data: Vec<f32> = (0..b * t * nh * hd).map(|i| (i as f32) * 0.01).collect();
        let x = candle_core::Var::from_tensor(
            &Tensor::new(&x_data[..], dev)?.reshape((b, t, nh, hd))?,
        )?;
        let cos = Tensor::ones((1, t, 1, half), DType::F32, dev)?;
        let sin = Tensor::zeros((1, t, 1, half), DType::F32, dev)?;
        let y = fused_rotary_emb(x.as_tensor(), &cos, &sin)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;
        let grad = grads.get(x.as_tensor()).expect("grad should exist");
        let g: Vec<f32> = grad.flatten_all()?.to_vec1()?;
        assert_eq!(g.len(), b * t * nh * hd);
        for (i, &v) in g.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-5, "grad mismatch at {i}: {v} vs 1.0");
        }
        Ok(())
    }

    #[test]
    fn test_rope_grad_nontrivial() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, t, nh, hd) = (1, 2, 1, 4);
        let half = hd / 2;
        let x_var =
            candle_core::Var::from_tensor(&Tensor::randn(0f32, 1f32, (b, t, nh, hd), dev)?)?;
        let cos = Tensor::randn(0f32, 1f32, (1, t, 1, half), dev)?;
        let sin = Tensor::randn(0f32, 1f32, (1, t, 1, half), dev)?;
        let y = fused_rotary_emb(x_var.as_tensor(), &cos, &sin)?;
        let loss = y.sqr()?.sum_all()?;
        let grads_fused = loss.backward()?;
        let g_fused: Vec<f32> = grads_fused
            .get(x_var.as_tensor())
            .unwrap()
            .flatten_all()?
            .to_vec1()?;
        let x_var2 = candle_core::Var::from_tensor(x_var.as_tensor())?;
        let x1 = x_var2.as_tensor().narrow(3, 0, half)?;
        let x2 = x_var2.as_tensor().narrow(3, half, half)?;
        let y1 = (x1.broadcast_mul(&cos)? + x2.broadcast_mul(&sin)?)?;
        let y2 = (x1.broadcast_mul(&sin.neg()?)? + x2.broadcast_mul(&cos)?)?;
        let y_naive = Tensor::cat(&[&y1, &y2], 3)?;
        let loss_naive = y_naive.sqr()?.sum_all()?;
        let grads_naive = loss_naive.backward()?;
        let g_naive: Vec<f32> = grads_naive
            .get(x_var2.as_tensor())
            .unwrap()
            .flatten_all()?
            .to_vec1()?;
        for (i, (&fv, &nv)) in g_fused.iter().zip(g_naive.iter()).enumerate() {
            assert!(
                (fv - nv).abs() < 1e-4,
                "grad fused vs naive at {i}: {fv} vs {nv}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_cross_entropy_basic() -> Result<()> {
        let dev = &Device::Cpu;
        let logits = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dev)?;
        let targets = Tensor::new(&[3u32, 0], dev)?;
        let loss = fused_cross_entropy(&logits, &targets)?;
        let loss_val = loss.to_scalar::<f32>()?;
        let ref_loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
        let ref_val = ref_loss.to_scalar::<f32>()?;
        assert!(
            (loss_val - ref_val).abs() < 1e-5,
            "fused={loss_val} vs ref={ref_val}"
        );
        Ok(())
    }

    #[test]
    fn test_cross_entropy_grad() -> Result<()> {
        let dev = &Device::Cpu;
        let logits = candle_core::Var::new(&[[1.0f32, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dev)?;
        let targets = Tensor::new(&[3u32, 0], dev)?;
        let loss = fused_cross_entropy(logits.as_tensor(), &targets)?;
        let grads = loss.backward()?;
        let grad = grads.get(logits.as_tensor()).expect("grad should exist");
        let g: Vec<f32> = grad.flatten_all()?.to_vec1()?;
        assert_eq!(g.len(), 8);
        let ref_loss = candle_nn::loss::cross_entropy(logits.as_tensor(), &targets)?;
        let ref_grads = ref_loss.backward()?;
        let ref_grad = ref_grads.get(logits.as_tensor()).expect("ref grad");
        let rg: Vec<f32> = ref_grad.flatten_all()?.to_vec1()?;
        for (i, (&got, &exp)) in g.iter().zip(rg.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "grad mismatch at {i}: fused={got} vs ref={exp}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_cross_entropy_uniform() -> Result<()> {
        let dev = &Device::Cpu;
        let v = 8usize;
        let logits = Tensor::zeros((4, v), DType::F32, dev)?;
        let targets = Tensor::new(&[0u32, 1, 2, 3], dev)?;
        let loss = fused_cross_entropy(&logits, &targets)?;
        let loss_val = loss.to_scalar::<f32>()?;
        let expected = (v as f32).ln();
        assert!(
            (loss_val - expected).abs() < 1e-5,
            "uniform loss: got {loss_val}, expected {expected}"
        );
        Ok(())
    }

    #[test]
    fn test_fused_embed_basic() -> Result<()> {
        let dev = &Device::Cpu;
        let vocab = 8;
        let d = 4;
        let seq = 3;
        let tok_data: Vec<f32> = (0..vocab * d).map(|i| i as f32 * 0.1).collect();
        let pos_data: Vec<f32> = (0..seq * d).map(|i| i as f32 * 0.01).collect();
        let tok_emb = Tensor::new(&tok_data[..], dev)?.reshape((vocab, d))?;
        let pos_emb = Tensor::new(&pos_data[..], dev)?.reshape((seq, d))?;
        let ids = Tensor::new(&[2u32, 5, 1], dev)?.reshape((1, 3))?;
        let out = fused_embed(&ids, &tok_emb, &pos_emb, 0)?;
        let out_data: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert_eq!(out_data.len(), 3 * d);
        for j in 0..d {
            let expected = tok_data[2 * d + j] + pos_data[0 * d + j];
            assert!(
                (out_data[0 * d + j] - expected).abs() < 1e-5,
                "row0[{j}]: got {}, expected {expected}",
                out_data[0 * d + j]
            );
        }
        for j in 0..d {
            let expected = tok_data[5 * d + j] + pos_data[1 * d + j];
            assert!(
                (out_data[1 * d + j] - expected).abs() < 1e-5,
                "row1[{j}]: got {}, expected {expected}",
                out_data[1 * d + j]
            );
        }
        Ok(())
    }

    #[test]
    fn test_fused_embed_grad() -> Result<()> {
        let dev = &Device::Cpu;
        let vocab = 8;
        let d = 4;
        let seq = 3;
        let tok_data: Vec<f32> = (0..vocab * d).map(|i| (i as f32) * 0.1).collect();
        let pos_data: Vec<f32> = (0..seq * d).map(|i| (i as f32) * 0.01).collect();
        let tok_var =
            candle_core::Var::from_tensor(&Tensor::new(&tok_data[..], dev)?.reshape((vocab, d))?)?;
        let pos_var =
            candle_core::Var::from_tensor(&Tensor::new(&pos_data[..], dev)?.reshape((seq, d))?)?;
        let ids = Tensor::new(&[2u32, 5, 1], dev)?.reshape((1, 3))?;
        let out = fused_embed(&ids, tok_var.as_tensor(), pos_var.as_tensor(), 0)?;
        let loss = out.sum_all()?;
        let grads = loss.backward()?;
        let g_tok = grads.get(tok_var.as_tensor()).expect("tok grad");
        let g_pos = grads.get(pos_var.as_tensor()).expect("pos grad");
        assert_eq!(g_tok.dims(), &[vocab, d]);
        assert_eq!(g_pos.dims(), &[seq, d]);
        let gt: Vec<f32> = g_tok.flatten_all()?.to_vec1()?;
        for v in 0..vocab {
            for j in 0..d {
                let expected = if v == 2 || v == 5 || v == 1 { 1.0 } else { 0.0 };
                assert!(
                    (gt[v * d + j] - expected).abs() < 1e-5,
                    "tok grad[{v},{j}]: got {}, expected {expected}",
                    gt[v * d + j]
                );
            }
        }
        let gp: Vec<f32> = g_pos.flatten_all()?.to_vec1()?;
        for p in 0..seq {
            for j in 0..d {
                assert!(
                    (gp[p * d + j] - 1.0).abs() < 1e-5,
                    "pos grad[{p},{j}]: got {}, expected 1.0",
                    gp[p * d + j]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_fused_embed_batch() -> Result<()> {
        let dev = &Device::Cpu;
        let vocab = 8;
        let d = 4;
        let seq = 2;
        let tok_data: Vec<f32> = (0..vocab * d).map(|i| i as f32 * 0.1).collect();
        let pos_data: Vec<f32> = (0..seq * d).map(|i| i as f32 * 0.01).collect();
        let tok_emb = Tensor::new(&tok_data[..], dev)?.reshape((vocab, d))?;
        let pos_emb = Tensor::new(&pos_data[..], dev)?.reshape((seq, d))?;
        let ids = Tensor::new(&[3u32, 7, 0, 4], dev)?.reshape((2, 2))?;
        let out = fused_embed(&ids, &tok_emb, &pos_emb, 0)?;
        let out_data: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert_eq!(out_data.len(), 4 * d);
        for j in 0..d {
            let expected = tok_data[3 * d + j] + pos_data[0 * d + j];
            assert!((out_data[0 * d + j] - expected).abs() < 1e-5);
        }
        for j in 0..d {
            let expected = tok_data[0 * d + j] + pos_data[0 * d + j];
            assert!((out_data[2 * d + j] - expected).abs() < 1e-5);
        }
        Ok(())
    }
}

// ── Metal GPU tests ─────────────────────────────────────────────────────────
// Run fused ops on actual Metal GPU, compare against CPU reference.
// BF16 tolerance: rtol ~1e-2, atol ~1e-2.

#[cfg(test)]
#[cfg(feature = "metal")]
mod metal_tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor};

    const ATOL: f64 = 1e-2;
    const RTOL: f64 = 1e-2;

    fn metal_device() -> Device {
        Device::new_metal(0).expect("Metal device required")
    }

    fn assert_close(got: &[f32], expected: &[f32], label: &str) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut worst = 0;
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let ae = (g as f64 - e as f64).abs();
            let re = if e.abs() > 1e-6 {
                ae / e.abs() as f64
            } else {
                ae
            };
            if ae > max_abs {
                max_abs = ae;
                worst = i;
            }
            if re > max_rel {
                max_rel = re;
            }
            assert!(
                ae < ATOL || re < RTOL,
                "{label}[{i}]: got {g}, expected {e}, abs={ae:.6}, rel={re:.6}"
            );
        }
        eprintln!("  {label}: max_abs={max_abs:.6}@[{worst}], max_rel={max_rel:.6}");
    }

    fn pseudo_random(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as i32) as f32 / i32::MAX as f32 * scale
            })
            .collect()
    }

    fn to_f32_cpu(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    // ── RMS Norm forward ────────────────────────────────────────────────────

    #[test]
    fn metal_rms_norm_fwd_small() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0];
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((2, 4))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_rms_norm(&x_cpu)?;
        let y_metal = fused_rms_norm(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "rms_norm_fwd_small",
        );
        Ok(())
    }

    #[test]
    fn metal_rms_norm_fwd_128x512() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(128 * 512, 42, 1.0);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((128, 512))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_rms_norm(&x_cpu)?;
        let y_metal = fused_rms_norm(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "rms_norm_fwd_128x512",
        );
        Ok(())
    }

    #[test]
    fn metal_rms_norm_fwd_16384x512() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(16384 * 512, 123, 0.5);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((16384, 512))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_rms_norm(&x_cpu)?;
        let y_metal = fused_rms_norm(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "rms_norm_fwd_16384x512",
        );
        Ok(())
    }

    // ── RMS Norm backward ───────────────────────────────────────────────────

    #[test]
    fn metal_rms_norm_bwd_small() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let x_cpu =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &Device::Cpu)?.reshape((1, 4))?)?;
        let loss_cpu = fused_rms_norm(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &metal)?.reshape((1, 4))?)?;
        let loss_metal = fused_rms_norm(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        assert_close(&to_f32_cpu(&gm), &to_f32_cpu(&gc), "rms_norm_bwd_small");
        Ok(())
    }

    #[test]
    fn metal_rms_norm_bwd_128x512() -> Result<()> {
        let metal = metal_device();
        let (rows, d) = (128, 512);
        let data = pseudo_random(rows * d, 99, 1.0);

        let x_cpu = candle_core::Var::from_tensor(
            &Tensor::new(&data[..], &Device::Cpu)?.reshape((rows, d))?,
        )?;
        let loss_cpu = fused_rms_norm(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &metal)?.reshape((rows, d))?)?;
        let loss_metal = fused_rms_norm(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        assert_close(&to_f32_cpu(&gm), &to_f32_cpu(&gc), "rms_norm_bwd_128x512");
        Ok(())
    }

    // ── Sigmoid 2x forward ──────────────────────────────────────────────────

    #[test]
    fn metal_sigmoid_2x_fwd_small() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![0.0, 1.0, -1.0, 5.0, -5.0, 0.1, -0.1, 3.0];
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_sigmoid_2x(&x_cpu)?;
        let y_metal = fused_sigmoid_2x(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "sigmoid_2x_fwd_small",
        );
        Ok(())
    }

    #[test]
    fn metal_sigmoid_2x_fwd_65536() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(128 * 512, 77, 2.0);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_sigmoid_2x(&x_cpu)?;
        let y_metal = fused_sigmoid_2x(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "sigmoid_2x_fwd_65536",
        );
        Ok(())
    }

    #[test]
    fn metal_sigmoid_2x_fwd_8m() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(16384 * 512, 200, 1.5);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_sigmoid_2x(&x_cpu)?;
        let y_metal = fused_sigmoid_2x(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "sigmoid_2x_fwd_8M",
        );
        Ok(())
    }

    // ── Sigmoid 2x backward ─────────────────────────────────────────────────

    #[test]
    fn metal_sigmoid_2x_bwd_small() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![0.0, 1.0, -1.0, 5.0, -5.0, 0.1];

        let x_cpu = candle_core::Var::new(&data[..], &Device::Cpu)?;
        let loss_cpu = fused_sigmoid_2x(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal = candle_core::Var::new(&data[..], &metal)?;
        let loss_metal = fused_sigmoid_2x(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        assert_close(&to_f32_cpu(&gm), &to_f32_cpu(&gc), "sigmoid_2x_bwd_small");
        Ok(())
    }

    #[test]
    fn metal_sigmoid_2x_bwd_65536() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(128 * 512, 55, 2.0);

        let x_cpu = candle_core::Var::new(&data[..], &Device::Cpu)?;
        let loss_cpu = fused_sigmoid_2x(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal = candle_core::Var::new(&data[..], &metal)?;
        let loss_metal = fused_sigmoid_2x(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        assert_close(&to_f32_cpu(&gm), &to_f32_cpu(&gc), "sigmoid_2x_bwd_65536");
        Ok(())
    }

    // ── ReLU^2 forward ──────────────────────────────────────────────────────

    #[test]
    fn metal_relu_sq_fwd_small() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_relu_sq(&x_cpu)?;
        let y_metal = fused_relu_sq(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "relu_sq_fwd_small",
        );
        Ok(())
    }

    #[test]
    fn metal_relu_sq_fwd_65536() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(128 * 512, 66, 3.0);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_relu_sq(&x_cpu)?;
        let y_metal = fused_relu_sq(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "relu_sq_fwd_65536",
        );
        Ok(())
    }

    #[test]
    fn metal_relu_sq_fwd_8m() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(16384 * 512, 201, 2.0);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_relu_sq(&x_cpu)?;
        let y_metal = fused_relu_sq(&x_cpu.to_device(&metal)?)?;
        assert_close(&to_f32_cpu(&y_metal), &to_f32_cpu(&y_cpu), "relu_sq_fwd_8M");
        Ok(())
    }

    // ── ReLU^2 backward ─────────────────────────────────────────────────────

    #[test]
    fn metal_relu_sq_bwd_small() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];

        let x_cpu = candle_core::Var::new(&data[..], &Device::Cpu)?;
        let loss_cpu = fused_relu_sq(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal = candle_core::Var::new(&data[..], &metal)?;
        let loss_metal = fused_relu_sq(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        assert_close(&to_f32_cpu(&gm), &to_f32_cpu(&gc), "relu_sq_bwd_small");
        Ok(())
    }

    #[test]
    fn metal_relu_sq_bwd_65536() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(128 * 512, 67, 3.0);

        let x_cpu = candle_core::Var::new(&data[..], &Device::Cpu)?;
        let loss_cpu = fused_relu_sq(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal = candle_core::Var::new(&data[..], &metal)?;
        let loss_metal = fused_relu_sq(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        assert_close(&to_f32_cpu(&gm), &to_f32_cpu(&gc), "relu_sq_bwd_65536");
        Ok(())
    }

    // ── Edge cases: zeros ────────────────────────────────────────────────────

    #[test]
    fn metal_rms_norm_zeros() -> Result<()> {
        let metal = metal_device();
        let data = vec![0.0f32; 512];
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((1, 512))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_rms_norm(&x_cpu)?;
        let y_metal = fused_rms_norm(&x_cpu.to_device(&metal)?)?;
        let mv = to_f32_cpu(&y_metal);
        for (i, &v) in mv.iter().enumerate() {
            assert!(v.abs() < 1e-3, "zeros: metal[{i}] = {v}, expected ~0");
        }
        assert_close(&mv, &to_f32_cpu(&y_cpu), "rms_norm_zeros");
        Ok(())
    }

    #[test]
    fn metal_sigmoid_2x_zeros() -> Result<()> {
        let metal = metal_device();
        let data = vec![0.0f32; 1024];
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_sigmoid_2x(&x_cpu)?;
        let y_metal = fused_sigmoid_2x(&x_cpu.to_device(&metal)?)?;
        let mv = to_f32_cpu(&y_metal);
        for (i, &v) in mv.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-2,
                "zeros: metal[{i}] = {v}, expected ~1.0"
            );
        }
        assert_close(&mv, &to_f32_cpu(&y_cpu), "sigmoid_2x_zeros");
        Ok(())
    }

    #[test]
    fn metal_relu_sq_zeros() -> Result<()> {
        let metal = metal_device();
        let data = vec![0.0f32; 1024];
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_relu_sq(&x_cpu)?;
        let y_metal = fused_relu_sq(&x_cpu.to_device(&metal)?)?;
        let mv = to_f32_cpu(&y_metal);
        for (i, &v) in mv.iter().enumerate() {
            assert!(v.abs() < 1e-6, "zeros: metal[{i}] = {v}, expected 0");
        }
        assert_close(&mv, &to_f32_cpu(&y_cpu), "relu_sq_zeros");
        Ok(())
    }

    // ── Edge cases: large values ─────────────────────────────────────────────

    #[test]
    fn metal_rms_norm_large_values() -> Result<()> {
        let metal = metal_device();
        let mut data = pseudo_random(4 * 512, 333, 50.0);
        data[0] = 100.0;
        data[1] = -100.0;
        data[512] = 80.0;
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((4, 512))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_rms_norm(&x_cpu)?;
        let y_metal = fused_rms_norm(&x_cpu.to_device(&metal)?)?;
        let mv = to_f32_cpu(&y_metal);
        for &v in &mv {
            assert!(v.is_finite(), "large: got {v}");
        }
        assert_close(&mv, &to_f32_cpu(&y_cpu), "rms_norm_large");
        Ok(())
    }

    #[test]
    fn metal_sigmoid_2x_extreme() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![50.0, -50.0, 100.0, -100.0, 0.0, 1.0, -1.0, 10.0];
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_sigmoid_2x(&x_cpu)?;
        let y_metal = fused_sigmoid_2x(&x_cpu.to_device(&metal)?)?;
        let mv = to_f32_cpu(&y_metal);
        assert!(
            (mv[0] - 2.0).abs() < 1e-2,
            "sigmoid_2x(50) ~ 2.0, got {}",
            mv[0]
        );
        assert!(mv[1].abs() < 1e-2, "sigmoid_2x(-50) ~ 0.0, got {}", mv[1]);
        for &v in &mv {
            assert!(v.is_finite(), "extreme: got {v}");
        }
        assert_close(&mv, &to_f32_cpu(&y_cpu), "sigmoid_2x_extreme");
        Ok(())
    }

    #[test]
    fn metal_relu_sq_negative_only() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = (-100..0).map(|i| i as f32 * 0.1).collect();
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_relu_sq(&x_cpu)?;
        let y_metal = fused_relu_sq(&x_cpu.to_device(&metal)?)?;
        let mv = to_f32_cpu(&y_metal);
        for (i, &v) in mv.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "negative_only: metal[{i}] = {v}, expected 0"
            );
        }
        assert_close(&mv, &to_f32_cpu(&y_cpu), "relu_sq_negative_only");
        Ok(())
    }

    // ── Edge cases: small values ─────────────────────────────────────────────

    #[test]
    fn metal_rms_norm_small_values() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(4 * 512, 444, 1e-3);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((4, 512))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_rms_norm(&x_cpu)?;
        let y_metal = fused_rms_norm(&x_cpu.to_device(&metal)?)?;
        let mv = to_f32_cpu(&y_metal);
        for &v in &mv {
            assert!(v.is_finite(), "small: got {v}");
        }
        assert_close(&mv, &to_f32_cpu(&y_cpu), "rms_norm_small");
        Ok(())
    }

    // ── Non-power-of-2 / odd dimensions ──────────────────────────────────────

    #[test]
    fn metal_rms_norm_non_pow2_dim() -> Result<()> {
        let metal = metal_device();
        let d = 300;
        let rows = 16;
        let data = pseudo_random(rows * d, 22, 1.0);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((rows, d))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_rms_norm(&x_cpu)?;
        let y_metal = fused_rms_norm(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "rms_norm_non_pow2",
        );
        Ok(())
    }

    #[test]
    fn metal_rms_norm_single_row() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(512, 11, 1.0);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((1, 512))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_rms_norm(&x_cpu)?;
        let y_metal = fused_rms_norm(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "rms_norm_single_row",
        );
        Ok(())
    }

    #[test]
    fn metal_sigmoid_2x_single_element() -> Result<()> {
        let metal = metal_device();
        let x_cpu = Tensor::new(&[0.5f32], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_sigmoid_2x(&x_cpu)?;
        let y_metal = fused_sigmoid_2x(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "sigmoid_2x_single",
        );
        Ok(())
    }

    #[test]
    fn metal_sigmoid_2x_non_multiple_256() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(1000, 88, 3.0);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_sigmoid_2x(&x_cpu)?;
        let y_metal = fused_sigmoid_2x(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "sigmoid_2x_non_256",
        );
        Ok(())
    }

    #[test]
    fn metal_relu_sq_single_element() -> Result<()> {
        let metal = metal_device();
        let x_cpu = Tensor::new(&[2.5f32], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_relu_sq(&x_cpu)?;
        let y_metal = fused_relu_sq(&x_cpu.to_device(&metal)?)?;
        assert_close(&to_f32_cpu(&y_metal), &to_f32_cpu(&y_cpu), "relu_sq_single");
        Ok(())
    }

    #[test]
    fn metal_relu_sq_non_multiple_256() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(999, 89, 5.0);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_relu_sq(&x_cpu)?;
        let y_metal = fused_relu_sq(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "relu_sq_non_256",
        );
        Ok(())
    }

    // ── Backward edge cases ──────────────────────────────────────────────────

    #[test]
    fn metal_rms_norm_bwd_zeros() -> Result<()> {
        let metal = metal_device();
        let data = vec![0.0f32; 512];

        let x_cpu = candle_core::Var::from_tensor(
            &Tensor::new(&data[..], &Device::Cpu)?.reshape((1, 512))?,
        )?;
        let loss_cpu = fused_rms_norm(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &metal)?.reshape((1, 512))?)?;
        let loss_metal = fused_rms_norm(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        let gmv = to_f32_cpu(&gm);
        for &v in &gmv {
            assert!(v.is_finite(), "bwd zeros: got {v}");
        }
        assert_close(&gmv, &to_f32_cpu(&gc), "rms_norm_bwd_zeros");
        Ok(())
    }

    #[test]
    fn metal_sigmoid_2x_bwd_extreme() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![50.0, -50.0, 0.0, 1.0, -1.0, 10.0, -10.0, 0.001];

        let x_cpu = candle_core::Var::new(&data[..], &Device::Cpu)?;
        let loss_cpu = fused_sigmoid_2x(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal = candle_core::Var::new(&data[..], &metal)?;
        let loss_metal = fused_sigmoid_2x(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        let gmv = to_f32_cpu(&gm);
        for &v in &gmv {
            assert!(v.is_finite(), "bwd extreme: got {v}");
        }
        // Saturated sigmoid -> grad ~ 0
        assert!(gmv[0].abs() < 1e-2, "grad at x=50 should be ~0");
        assert!(gmv[1].abs() < 1e-2, "grad at x=-50 should be ~0");
        assert_close(&gmv, &to_f32_cpu(&gc), "sigmoid_2x_bwd_extreme");
        Ok(())
    }

    #[test]
    fn metal_relu_sq_bwd_at_zero() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

        let x_cpu = candle_core::Var::new(&data[..], &Device::Cpu)?;
        let loss_cpu = fused_relu_sq(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal = candle_core::Var::new(&data[..], &metal)?;
        let loss_metal = fused_relu_sq(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        let gmv = to_f32_cpu(&gm);
        for (i, &v) in gmv.iter().enumerate() {
            assert!(v.abs() < 1e-6, "relu_sq bwd at zero: [{i}] = {v}");
        }
        assert_close(&gmv, &to_f32_cpu(&gc), "relu_sq_bwd_at_zero");
        Ok(())
    }

    // ── Residual scale forward ─────────────────────────────────────────────

    #[test]
    fn metal_residual_scale_fwd_small() -> Result<()> {
        let metal = metal_device();
        let xd: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0];
        let x0d: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0, -0.5, 0.25, 1.25, -1.5];
        let x_cpu = Tensor::new(&xd[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let x0_cpu = Tensor::new(&x0d[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_residual_scale(&x_cpu, &x0_cpu, 0.9, 0.1)?;
        let y_metal = fused_residual_scale(
            &x_cpu.to_device(&metal)?,
            &x0_cpu.to_device(&metal)?,
            0.9,
            0.1,
        )?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "residual_scale_fwd_small",
        );
        Ok(())
    }

    #[test]
    fn metal_residual_scale_fwd_large() -> Result<()> {
        let metal = metal_device();
        let n = 16384 * 512;
        let xd = pseudo_random(n, 42, 1.0);
        let x0d = pseudo_random(n, 99, 0.5);
        let x_cpu = Tensor::new(&xd[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let x0_cpu = Tensor::new(&x0d[..], &Device::Cpu)?.to_dtype(DType::BF16)?;
        let y_cpu = fused_residual_scale(&x_cpu, &x0_cpu, 1.0, 0.1)?;
        let y_metal = fused_residual_scale(
            &x_cpu.to_device(&metal)?,
            &x0_cpu.to_device(&metal)?,
            1.0,
            0.1,
        )?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "residual_scale_fwd_large",
        );
        Ok(())
    }

    // ── Residual scale backward ────────────────────────────────────────────

    #[test]
    fn metal_residual_scale_bwd_small() -> Result<()> {
        let metal = metal_device();
        let xd: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x0d: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];

        let x_cpu = candle_core::Var::new(&xd[..], &Device::Cpu)?;
        let x0_cpu = candle_core::Var::new(&x0d[..], &Device::Cpu)?;
        let loss_cpu = fused_residual_scale(
            &x_cpu.to_dtype(DType::BF16)?,
            &x0_cpu.to_dtype(DType::BF16)?,
            0.8,
            0.2,
        )?
        .to_dtype(DType::F32)?
        .sum_all()?;
        let grads_cpu = loss_cpu.backward()?;
        let gc_x = grads_cpu.get(&x_cpu).expect("cpu grad x").clone();
        let gc_x0 = grads_cpu.get(&x0_cpu).expect("cpu grad x0").clone();

        let x_metal = candle_core::Var::new(&xd[..], &metal)?;
        let x0_metal = candle_core::Var::new(&x0d[..], &metal)?;
        let loss_metal = fused_residual_scale(
            &x_metal.to_dtype(DType::BF16)?,
            &x0_metal.to_dtype(DType::BF16)?,
            0.8,
            0.2,
        )?
        .to_dtype(DType::F32)?
        .sum_all()?;
        let grads_metal = loss_metal.backward()?;
        let gm_x = grads_metal.get(&x_metal).expect("metal grad x").clone();
        let gm_x0 = grads_metal.get(&x0_metal).expect("metal grad x0").clone();

        assert_close(
            &to_f32_cpu(&gm_x),
            &to_f32_cpu(&gc_x),
            "residual_scale_bwd_x",
        );
        assert_close(
            &to_f32_cpu(&gm_x0),
            &to_f32_cpu(&gc_x0),
            "residual_scale_bwd_x0",
        );
        Ok(())
    }

    #[test]
    fn metal_residual_scale_bwd_65536() -> Result<()> {
        let metal = metal_device();
        let n = 128 * 512;
        let xd = pseudo_random(n, 77, 1.0);
        let x0d = pseudo_random(n, 88, 0.5);

        let x_cpu = candle_core::Var::new(&xd[..], &Device::Cpu)?;
        let x0_cpu = candle_core::Var::new(&x0d[..], &Device::Cpu)?;
        let loss_cpu = fused_residual_scale(
            &x_cpu.to_dtype(DType::BF16)?,
            &x0_cpu.to_dtype(DType::BF16)?,
            1.0,
            0.1,
        )?
        .to_dtype(DType::F32)?
        .sum_all()?;
        let grads_cpu = loss_cpu.backward()?;
        let gc_x = grads_cpu.get(&x_cpu).expect("cpu grad x").clone();
        let gc_x0 = grads_cpu.get(&x0_cpu).expect("cpu grad x0").clone();

        let x_metal = candle_core::Var::new(&xd[..], &metal)?;
        let x0_metal = candle_core::Var::new(&x0d[..], &metal)?;
        let loss_metal = fused_residual_scale(
            &x_metal.to_dtype(DType::BF16)?,
            &x0_metal.to_dtype(DType::BF16)?,
            1.0,
            0.1,
        )?
        .to_dtype(DType::F32)?
        .sum_all()?;
        let grads_metal = loss_metal.backward()?;
        let gm_x = grads_metal.get(&x_metal).expect("metal grad x").clone();
        let gm_x0 = grads_metal.get(&x0_metal).expect("metal grad x0").clone();

        assert_close(
            &to_f32_cpu(&gm_x),
            &to_f32_cpu(&gc_x),
            "residual_scale_bwd_x_65536",
        );
        assert_close(
            &to_f32_cpu(&gm_x0),
            &to_f32_cpu(&gc_x0),
            "residual_scale_bwd_x0_65536",
        );
        Ok(())
    }

    // ── Softmax forward ────────────────────────────────────────────────────

    #[test]
    fn metal_softmax_fwd_small() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((2, 4))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_softmax(&x_cpu)?;
        let y_metal = fused_softmax(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "softmax_fwd_small",
        );
        Ok(())
    }

    #[test]
    fn metal_softmax_fwd_128x512() -> Result<()> {
        let metal = metal_device();
        let data = pseudo_random(128 * 512, 42, 2.0);
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((128, 512))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_softmax(&x_cpu)?;
        let y_metal = fused_softmax(&x_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&y_metal),
            &to_f32_cpu(&y_cpu),
            "softmax_fwd_128x512",
        );
        Ok(())
    }

    // ── Softmax backward ───────────────────────────────────────────────────

    #[test]
    fn metal_softmax_bwd_small() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let x_cpu =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &Device::Cpu)?.reshape((1, 4))?)?;
        let loss_cpu = fused_softmax(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .i((.., 0..1))?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &metal)?.reshape((1, 4))?)?;
        let loss_metal = fused_softmax(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .i((.., 0..1))?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        assert_close(&to_f32_cpu(&gm), &to_f32_cpu(&gc), "softmax_bwd_small");
        Ok(())
    }

    #[test]
    fn metal_softmax_bwd_128x512() -> Result<()> {
        let metal = metal_device();
        let (rows, d) = (128, 512);
        let data = pseudo_random(rows * d, 99, 2.0);

        let x_cpu = candle_core::Var::from_tensor(
            &Tensor::new(&data[..], &Device::Cpu)?.reshape((rows, d))?,
        )?;
        let loss_cpu = fused_softmax(&x_cpu.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .i((.., 0..1))?
            .sum_all()?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &metal)?.reshape((rows, d))?)?;
        let loss_metal = fused_softmax(&x_metal.to_dtype(DType::BF16)?)?
            .to_dtype(DType::F32)?
            .i((.., 0..1))?
            .sum_all()?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        assert_close(&to_f32_cpu(&gm), &to_f32_cpu(&gc), "softmax_bwd_128x512");
        Ok(())
    }

    // ── Softmax edge cases ─────────────────────────────────────────────────

    #[test]
    fn metal_softmax_uniform() -> Result<()> {
        let metal = metal_device();
        let data = vec![0.0f32; 512];
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((1, 512))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_softmax(&x_cpu)?;
        let y_metal = fused_softmax(&x_cpu.to_device(&metal)?)?;
        let mv = to_f32_cpu(&y_metal);
        for (i, &v) in mv.iter().enumerate() {
            assert!(
                (v - 1.0 / 512.0).abs() < 1e-2,
                "uniform: metal[{i}] = {v}, expected ~1/512"
            );
        }
        assert_close(&mv, &to_f32_cpu(&y_cpu), "softmax_uniform");
        Ok(())
    }

    #[test]
    fn metal_softmax_large_values() -> Result<()> {
        let metal = metal_device();
        let mut data = vec![0.0f32; 512];
        data[0] = 100.0;
        let x_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((1, 512))?
            .to_dtype(DType::BF16)?;
        let y_cpu = fused_softmax(&x_cpu)?;
        let y_metal = fused_softmax(&x_cpu.to_device(&metal)?)?;
        let mv = to_f32_cpu(&y_metal);
        for &v in &mv {
            assert!(v.is_finite(), "large: got {v}");
        }
        assert_close(&mv, &to_f32_cpu(&y_cpu), "softmax_large");
        Ok(())
    }

    // ── QK-Norm forward ────────────────────────────────────────────────────

    #[test]
    fn metal_qk_norm_fwd_small() -> Result<()> {
        let metal = metal_device();
        let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0];
        let k_data: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
        let q_cpu = Tensor::new(&q_data[..], &Device::Cpu)?
            .reshape((2, 4))?
            .to_dtype(DType::BF16)?;
        let k_cpu = Tensor::new(&k_data[..], &Device::Cpu)?
            .reshape((1, 4))?
            .to_dtype(DType::BF16)?;
        let (nq_cpu, nk_cpu) = fused_qk_norm(&q_cpu, &k_cpu)?;
        let q_metal = q_cpu.to_device(&metal)?;
        let k_metal = k_cpu.to_device(&metal)?;
        let (nq_metal, nk_metal) = fused_qk_norm(&q_metal, &k_metal)?;
        assert_close(
            &to_f32_cpu(&nq_metal),
            &to_f32_cpu(&nq_cpu),
            "qk_norm_fwd_q_small",
        );
        assert_close(
            &to_f32_cpu(&nk_metal),
            &to_f32_cpu(&nk_cpu),
            "qk_norm_fwd_k_small",
        );
        Ok(())
    }

    #[test]
    fn metal_qk_norm_fwd_4d() -> Result<()> {
        let metal = metal_device();
        let q_data = pseudo_random(8 * 32 * 4 * 128, 42, 1.0);
        let k_data = pseudo_random(8 * 32 * 4 * 128, 99, 1.0);
        let q_cpu = Tensor::new(&q_data[..], &Device::Cpu)?
            .reshape((8, 32, 4, 128))?
            .to_dtype(DType::BF16)?;
        let k_cpu = Tensor::new(&k_data[..], &Device::Cpu)?
            .reshape((8, 32, 4, 128))?
            .to_dtype(DType::BF16)?;
        let (nq_cpu, nk_cpu) = fused_qk_norm(&q_cpu, &k_cpu)?;
        let (nq_metal, nk_metal) =
            fused_qk_norm(&q_cpu.to_device(&metal)?, &k_cpu.to_device(&metal)?)?;
        assert_close(
            &to_f32_cpu(&nq_metal),
            &to_f32_cpu(&nq_cpu),
            "qk_norm_fwd_q_4d",
        );
        assert_close(
            &to_f32_cpu(&nk_metal),
            &to_f32_cpu(&nk_cpu),
            "qk_norm_fwd_k_4d",
        );
        Ok(())
    }

    // ── QK-Norm backward ───────────────────────────────────────────────────

    #[test]
    fn metal_qk_norm_bwd_small() -> Result<()> {
        let metal = metal_device();
        let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let k_data: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];

        let q_cpu = candle_core::Var::from_tensor(
            &Tensor::new(&q_data[..], &Device::Cpu)?.reshape((1, 4))?,
        )?;
        let k_cpu = candle_core::Var::from_tensor(
            &Tensor::new(&k_data[..], &Device::Cpu)?.reshape((1, 4))?,
        )?;
        let (nq, nk) = fused_qk_norm(&q_cpu.to_dtype(DType::BF16)?, &k_cpu.to_dtype(DType::BF16)?)?;
        let loss_cpu =
            (nq.to_dtype(DType::F32)?.sum_all()? + nk.to_dtype(DType::F32)?.sum_all()?)?;
        let grads_cpu = loss_cpu.backward()?;
        let gq_cpu = grads_cpu.get(&q_cpu).expect("cpu q grad").clone();
        let gk_cpu = grads_cpu.get(&k_cpu).expect("cpu k grad").clone();

        let q_metal =
            candle_core::Var::from_tensor(&Tensor::new(&q_data[..], &metal)?.reshape((1, 4))?)?;
        let k_metal =
            candle_core::Var::from_tensor(&Tensor::new(&k_data[..], &metal)?.reshape((1, 4))?)?;
        let (nq, nk) = fused_qk_norm(
            &q_metal.to_dtype(DType::BF16)?,
            &k_metal.to_dtype(DType::BF16)?,
        )?;
        let loss_metal =
            (nq.to_dtype(DType::F32)?.sum_all()? + nk.to_dtype(DType::F32)?.sum_all()?)?;
        let grads_metal = loss_metal.backward()?;
        let gq_metal = grads_metal.get(&q_metal).expect("metal q grad").clone();
        let gk_metal = grads_metal.get(&k_metal).expect("metal k grad").clone();

        assert_close(
            &to_f32_cpu(&gq_metal),
            &to_f32_cpu(&gq_cpu),
            "qk_norm_bwd_q_small",
        );
        assert_close(
            &to_f32_cpu(&gk_metal),
            &to_f32_cpu(&gk_cpu),
            "qk_norm_bwd_k_small",
        );
        Ok(())
    }

    // ── Cross-entropy forward ─────────────────────────────────────────────

    #[test]
    fn metal_cross_entropy_fwd_small() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let logits_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((2, 4))?
            .to_dtype(DType::BF16)?;
        let targets = Tensor::new(&[3u32, 0], &Device::Cpu)?;
        let loss_cpu = fused_cross_entropy(&logits_cpu, &targets)?;
        let loss_metal =
            fused_cross_entropy(&logits_cpu.to_device(&metal)?, &targets.to_device(&metal)?)?;
        let cpu_val = to_f32_cpu(&loss_cpu);
        let metal_val = to_f32_cpu(&loss_metal);
        assert_close(&metal_val, &cpu_val, "ce_fwd_small");
        Ok(())
    }

    #[test]
    fn metal_cross_entropy_fwd_large() -> Result<()> {
        let metal = metal_device();
        let n = 1024;
        let v = 8192;
        let data = pseudo_random(n * v, 42, 2.0);
        let logits_cpu = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?;
        let tgt_data: Vec<u32> = (0..n).map(|i| (i * 7 % v) as u32).collect();
        let targets = Tensor::new(&tgt_data[..], &Device::Cpu)?;
        let loss_cpu = fused_cross_entropy(&logits_cpu, &targets)?;
        let loss_metal =
            fused_cross_entropy(&logits_cpu.to_device(&metal)?, &targets.to_device(&metal)?)?;
        let cpu_val = to_f32_cpu(&loss_cpu);
        let metal_val = to_f32_cpu(&loss_metal);
        assert_close(&metal_val, &cpu_val, "ce_fwd_large");
        Ok(())
    }

    // ── Cross-entropy backward ────────────────────────────────────────────

    #[test]
    fn metal_cross_entropy_bwd_small() -> Result<()> {
        let metal = metal_device();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let targets_data = vec![3u32, 0];

        let x_cpu =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &Device::Cpu)?.reshape((2, 4))?)?;
        let tgt_cpu = Tensor::new(&targets_data[..], &Device::Cpu)?;
        let loss_cpu = fused_cross_entropy(&x_cpu.to_dtype(DType::BF16)?, &tgt_cpu)?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &metal)?.reshape((2, 4))?)?;
        let tgt_metal = Tensor::new(&targets_data[..], &metal)?;
        let loss_metal = fused_cross_entropy(&x_metal.to_dtype(DType::BF16)?, &tgt_metal)?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        assert_close(&to_f32_cpu(&gm), &to_f32_cpu(&gc), "ce_bwd_small");
        Ok(())
    }

    #[test]
    fn metal_cross_entropy_bwd_1024x8192() -> Result<()> {
        let metal = metal_device();
        let n = 1024;
        let v = 8192;
        let data = pseudo_random(n * v, 55, 2.0);
        let tgt_data: Vec<u32> = (0..n).map(|i| (i * 13 % v) as u32).collect();

        let x_cpu =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &Device::Cpu)?.reshape((n, v))?)?;
        let tgt_cpu = Tensor::new(&tgt_data[..], &Device::Cpu)?;
        let loss_cpu = fused_cross_entropy(&x_cpu.to_dtype(DType::BF16)?, &tgt_cpu)?;
        let gc = loss_cpu.backward()?.get(&x_cpu).expect("cpu grad").clone();

        let x_metal =
            candle_core::Var::from_tensor(&Tensor::new(&data[..], &metal)?.reshape((n, v))?)?;
        let tgt_metal = Tensor::new(&tgt_data[..], &metal)?;
        let loss_metal = fused_cross_entropy(&x_metal.to_dtype(DType::BF16)?, &tgt_metal)?;
        let gm = loss_metal
            .backward()?
            .get(&x_metal)
            .expect("metal grad")
            .clone();

        assert_close(&to_f32_cpu(&gm), &to_f32_cpu(&gc), "ce_bwd_1024x8192");
        Ok(())
    }

    // ── Cross-entropy uniform ─────────────────────────────────────────────

    #[test]
    fn metal_cross_entropy_uniform() -> Result<()> {
        let metal = metal_device();
        let v = 8192usize;
        let n = 64;
        let logits_cpu = Tensor::zeros((n, v), DType::BF16, &Device::Cpu)?;
        let targets = Tensor::new(&vec![0u32; n][..], &Device::Cpu)?;
        let loss_cpu = fused_cross_entropy(&logits_cpu, &targets)?;
        let loss_metal =
            fused_cross_entropy(&logits_cpu.to_device(&metal)?, &targets.to_device(&metal)?)?;
        let cpu_val = to_f32_cpu(&loss_cpu)[0];
        let metal_val = to_f32_cpu(&loss_metal)[0];
        let expected = (v as f32).ln();
        assert!(
            (cpu_val - expected).abs() < 0.05,
            "cpu uniform loss: {cpu_val} vs {expected}"
        );
        assert!(
            (metal_val - expected).abs() < 0.05,
            "metal uniform loss: {metal_val} vs {expected}"
        );
        Ok(())
    }

    // ── BF16 NaN diagnostic tests ───────────────────────────────────────

    /// Generate logits mimicking real model output: mostly in [-2, 2] with
    /// occasional outliers up to ±scale_outlier.
    fn realistic_logits(n: usize, v: usize, seed: u64, scale_outlier: f32) -> Vec<f32> {
        let mut s = seed;
        let mut next = || -> f32 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as i32) as f32 / i32::MAX as f32
        };
        (0..n * v)
            .map(|i| {
                let base = next() * 2.0; // [-2, 2]
                // Every 1000th element gets an outlier
                if i % 1000 == 0 {
                    base + next() * scale_outlier
                } else {
                    base
                }
            })
            .collect()
    }

    #[test]
    fn metal_cross_entropy_bf16_nan_check() -> Result<()> {
        let metal = metal_device();
        let n = 128usize;
        let v = 8192usize;

        // --- Test 1: Realistic logits with moderate outliers (±10) ---
        let data = realistic_logits(n, v, 777, 10.0);
        let tgt_data: Vec<u32> = (0..n).map(|i| ((i * 17 + 3) % v) as u32).collect();

        let logits_bf16 = Tensor::new(&data[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let targets = Tensor::new(&tgt_data[..], &metal)?;

        let loss = fused_cross_entropy(&logits_bf16, &targets)?;
        let loss_val = to_f32_cpu(&loss);
        let has_nan_moderate = loss_val.iter().any(|v| v.is_nan());
        eprintln!(
            "  BF16 moderate outliers (±10): loss={:.4}, nan={}",
            loss_val[0], has_nan_moderate
        );
        assert!(
            !has_nan_moderate,
            "NaN in BF16 cross-entropy with moderate outliers (±10)"
        );

        // --- Test 2: Large outliers (±50, exceeds soft-cap range) ---
        let data_large = realistic_logits(n, v, 888, 50.0);
        let logits_bf16_large = Tensor::new(&data_large[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;

        let loss_large = fused_cross_entropy(&logits_bf16_large, &targets)?;
        let loss_large_val = to_f32_cpu(&loss_large);
        let has_nan_large = loss_large_val.iter().any(|v| v.is_nan());
        eprintln!(
            "  BF16 large outliers (±50): loss={:.4}, nan={}",
            loss_large_val[0], has_nan_large
        );

        if has_nan_large {
            // Retry with F32 to see if it's a BF16 precision issue
            let logits_f32 = Tensor::new(&data_large[..], &Device::Cpu)?
                .reshape((n, v))?
                .to_device(&metal)?;
            let loss_f32 = fused_cross_entropy(&logits_f32, &targets)?;
            let loss_f32_val = to_f32_cpu(&loss_f32);
            let f32_nan = loss_f32_val.iter().any(|v| v.is_nan());
            eprintln!(
                "  F32 large outliers (±50): loss={:.4}, nan={}",
                loss_f32_val[0], f32_nan
            );
            if f32_nan {
                panic!("NaN in BOTH BF16 and F32 — kernel bug, not precision issue");
            } else {
                panic!("NaN in BF16 but NOT F32 — BF16 precision issue with large logits");
            }
        }

        // --- Test 3: Extreme values near BF16 limits ---
        // BF16 max is ~65504. Values of ±100 should still be representable,
        // but exp(100) overflows F32. The kernel uses max-subtraction, so this
        // tests whether the max reduction works correctly with BF16 input.
        let data_extreme = realistic_logits(n, v, 999, 100.0);
        let logits_bf16_extreme = Tensor::new(&data_extreme[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;

        let loss_extreme = fused_cross_entropy(&logits_bf16_extreme, &targets)?;
        let loss_extreme_val = to_f32_cpu(&loss_extreme);
        let has_nan_extreme = loss_extreme_val.iter().any(|v| v.is_nan());
        eprintln!(
            "  BF16 extreme outliers (±100): loss={:.4}, nan={}",
            loss_extreme_val[0], has_nan_extreme
        );

        if has_nan_extreme {
            let logits_f32_e = Tensor::new(&data_extreme[..], &Device::Cpu)?
                .reshape((n, v))?
                .to_device(&metal)?;
            let loss_f32_e = fused_cross_entropy(&logits_f32_e, &targets)?;
            let f32_nan_e = to_f32_cpu(&loss_f32_e).iter().any(|v| v.is_nan());
            eprintln!("  F32 extreme outliers (±100): nan={}", f32_nan_e);
            if f32_nan_e {
                panic!("NaN in BOTH BF16 and F32 with extreme values — kernel bug");
            } else {
                panic!("NaN in BF16 but NOT F32 with extreme values — BF16 precision issue");
            }
        }

        Ok(())
    }

    #[test]
    fn metal_cross_entropy_bf16_adversarial() -> Result<()> {
        let metal = metal_device();
        let n = 128usize;
        let v = 8192usize;
        let tgt_data: Vec<u32> = (0..n).map(|i| ((i * 17 + 3) % v) as u32).collect();
        let targets = Tensor::new(&tgt_data[..], &metal)?;

        // Case 1: All logits identical (max subtraction yields all zeros, sum_exp = V)
        // This stresses the log(sum_exp) path with exact cancellation.
        let all_same = vec![1.0f32; n * v];
        let logits = Tensor::new(&all_same[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let loss = fused_cross_entropy(&logits, &targets)?;
        let val = to_f32_cpu(&loss);
        eprintln!(
            "  BF16 all-same logits: loss={:.4}, nan={}",
            val[0],
            val[0].is_nan()
        );
        assert!(!val[0].is_nan(), "NaN with all-same logits");

        // Case 2: One logit is max by exactly 1 ULP in BF16
        // BF16 at value 30.0 has ULP ~0.125. Set one logit to 30.0, rest to 29.875.
        let mut near_same = vec![29.875f32; n * v];
        for row in 0..n {
            near_same[row * v + tgt_data[row] as usize] = 30.0;
        }
        let logits = Tensor::new(&near_same[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let loss = fused_cross_entropy(&logits, &targets)?;
        let val = to_f32_cpu(&loss);
        eprintln!(
            "  BF16 near-identical (30±ULP): loss={:.4}, nan={}",
            val[0],
            val[0].is_nan()
        );
        assert!(!val[0].is_nan(), "NaN with near-identical logits at 30.0");

        // Case 3: Max logit is much larger, rest are tiny — tests exp underflow
        // With max=100, exp(0 - 100) = exp(-100) ≈ 0 in F32, so sum_exp ≈ 1.0
        let mut spike = vec![-100.0f32; n * v];
        for row in 0..n {
            spike[row * v] = 100.0; // first logit is the spike
        }
        let logits = Tensor::new(&spike[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let loss = fused_cross_entropy(&logits, &targets)?;
        let val = to_f32_cpu(&loss);
        eprintln!(
            "  BF16 spike (max=100, rest=-100): loss={:.4}, nan={}",
            val[0],
            val[0].is_nan()
        );
        assert!(!val[0].is_nan(), "NaN with spike logits");

        // Case 4: All zeros — trivial but catches division edge cases
        let logits = Tensor::zeros((n, v), DType::BF16, &metal)?;
        let loss = fused_cross_entropy(&logits, &targets)?;
        let val = to_f32_cpu(&loss);
        let expected = (v as f32).ln();
        eprintln!(
            "  BF16 all-zero: loss={:.4} (expected {:.4}), nan={}",
            val[0],
            expected,
            val[0].is_nan()
        );
        assert!(!val[0].is_nan(), "NaN with all-zero logits");

        // Case 5: Logits at BF16 max representable (~65504)
        let mut at_max = vec![0.0f32; n * v];
        for row in 0..n {
            at_max[row * v] = 65504.0;
        }
        let logits = Tensor::new(&at_max[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let loss = fused_cross_entropy(&logits, &targets)?;
        let val = to_f32_cpu(&loss);
        eprintln!(
            "  BF16 at-max (65504): loss={:.4}, nan={}",
            val[0],
            val[0].is_nan()
        );
        // exp(0 - 65504) underflows to 0, so sum_exp = 1.0 (just the max entry)
        // loss for target=tgt: -(0 - 65504) + log(1) = 65504 if tgt != 0,
        //                      -(65504 - 65504) + log(1) = 0 if tgt == 0
        assert!(!val[0].is_nan(), "NaN with logits at BF16 max");

        // Case 6: Negative BF16 max
        let mut at_neg_max = vec![-65504.0f32; n * v];
        for row in 0..n {
            at_neg_max[row * v] = 0.0; // one logit at 0
        }
        let logits = Tensor::new(&at_neg_max[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let loss = fused_cross_entropy(&logits, &targets)?;
        let val = to_f32_cpu(&loss);
        eprintln!(
            "  BF16 neg-max (-65504): loss={:.4}, nan={}",
            val[0],
            val[0].is_nan()
        );
        assert!(!val[0].is_nan(), "NaN with negative BF16 max logits");

        Ok(())
    }

    #[test]
    fn metal_cross_entropy_softcapped_bf16_nan() -> Result<()> {
        let metal = metal_device();
        let n = 128usize;
        let v = 8192usize;
        let softcap = 30.0f64;

        let tgt_data: Vec<u32> = (0..n).map(|i| ((i * 17 + 3) % v) as u32).collect();
        let targets = Tensor::new(&tgt_data[..], &metal)?;

        // --- Step A: Generate raw logits with large values (pre-softcap) ---
        // In the model, logits come from a matmul and can be large before capping.
        let data_raw = realistic_logits(n, v, 1234, 80.0);

        // --- Step B: Apply soft-capping in BF16 on Metal ---
        let raw_bf16 = Tensor::new(&data_raw[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;

        let capped_bf16 = ((raw_bf16.clone() / softcap)?.tanh()? * softcap)?;

        // Check if soft-capping itself produced NaN
        let capped_vals = to_f32_cpu(&capped_bf16);
        let cap_has_nan = capped_vals.iter().any(|v| v.is_nan());
        let cap_has_inf = capped_vals.iter().any(|v| v.is_infinite());
        let cap_min = capped_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let cap_max = capped_vals
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "  Soft-capped BF16: nan={}, inf={}, range=[{:.2}, {:.2}]",
            cap_has_nan, cap_has_inf, cap_min, cap_max
        );

        if cap_has_nan {
            // Test soft-capping in F32
            let raw_f32 = Tensor::new(&data_raw[..], &Device::Cpu)?
                .reshape((n, v))?
                .to_device(&metal)?;
            let capped_f32 = ((raw_f32.clone() / softcap)?.tanh()? * softcap)?;
            let f32_cap_nan = to_f32_cpu(&capped_f32).iter().any(|v| v.is_nan());
            eprintln!("  Soft-capped F32: nan={}", f32_cap_nan);
            panic!(
                "NaN in soft-capping step itself (BF16={}, F32={})",
                cap_has_nan, f32_cap_nan
            );
        }

        // --- Step C: Cross-entropy on soft-capped BF16 logits ---
        let loss = fused_cross_entropy(&capped_bf16, &targets)?;
        let loss_val = to_f32_cpu(&loss);
        let ce_has_nan = loss_val.iter().any(|v| v.is_nan());
        eprintln!(
            "  Cross-entropy on soft-capped BF16: loss={:.4}, nan={}",
            loss_val[0], ce_has_nan
        );

        if ce_has_nan {
            // Test CE with F32 soft-capped logits
            let raw_f32 = Tensor::new(&data_raw[..], &Device::Cpu)?
                .reshape((n, v))?
                .to_device(&metal)?;
            let capped_f32 = ((raw_f32 / softcap)?.tanh()? * softcap)?;
            let loss_f32 = fused_cross_entropy(&capped_f32, &targets)?;
            let f32_ce_nan = to_f32_cpu(&loss_f32).iter().any(|v| v.is_nan());
            eprintln!("  Cross-entropy on soft-capped F32: nan={}", f32_ce_nan);
            panic!(
                "NaN in cross-entropy after soft-cap: BF16={}, F32={} — {}",
                ce_has_nan,
                f32_ce_nan,
                if f32_ce_nan {
                    "kernel bug"
                } else {
                    "BF16 precision issue in CE kernel"
                }
            );
        }

        // --- Step D: Even larger pre-softcap logits (±500) ---
        // tanh saturates, so output should be near ±30, but BF16 intermediate
        // values in the division step (logits/30) could be imprecise.
        let data_huge = realistic_logits(n, v, 5678, 500.0);
        let raw_bf16_huge = Tensor::new(&data_huge[..], &Device::Cpu)?
            .reshape((n, v))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;

        let capped_huge = ((raw_bf16_huge / softcap)?.tanh()? * softcap)?;
        let capped_huge_vals = to_f32_cpu(&capped_huge);
        let huge_cap_nan = capped_huge_vals.iter().any(|v| v.is_nan());
        let huge_cap_inf = capped_huge_vals.iter().any(|v| v.is_infinite());
        eprintln!(
            "  Soft-capped BF16 (±500 input): nan={}, inf={}, range=[{:.2}, {:.2}]",
            huge_cap_nan,
            huge_cap_inf,
            capped_huge_vals
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min),
            capped_huge_vals
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        );

        if huge_cap_nan {
            panic!("NaN in soft-capping with ±500 input logits");
        }

        let loss_huge = fused_cross_entropy(&capped_huge, &targets)?;
        let loss_huge_val = to_f32_cpu(&loss_huge);
        let huge_ce_nan = loss_huge_val.iter().any(|v| v.is_nan());
        eprintln!(
            "  Cross-entropy on soft-capped BF16 (±500): loss={:.4}, nan={}",
            loss_huge_val[0], huge_ce_nan
        );

        if huge_ce_nan {
            panic!("NaN in cross-entropy with ±500 pre-softcap logits");
        }

        Ok(())
    }
}

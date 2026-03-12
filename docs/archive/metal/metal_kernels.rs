// NOTE: Add `pub mod metal_kernels;` to main.rs or lib.rs to use this module.

use anyhow::{Result, anyhow};
use candle_core::{DType, Storage, Tensor};
use metal::*;
use std::collections::HashMap;
use std::mem;

pub struct MetalKernels {
    device: Device,
    queue: CommandQueue,
    pipelines: HashMap<String, ComputePipelineState>,
}

impl MetalKernels {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow!("No Metal device found"))?;
        let queue = device.new_command_queue();

        let mut kernels = MetalKernels {
            device,
            queue,
            pipelines: HashMap::new(),
        };
        kernels.compile_shaders()?;
        Ok(kernels)
    }

    fn compile_shaders(&mut self) -> Result<()> {
        let attention_src = include_str!("../metal/attention.metal");
        let ops_src = include_str!("../metal/ops.metal");

        let options = CompileOptions::new();

        // Compile attention library
        let attention_lib = self
            .device
            .new_library_with_source(attention_src, &options)
            .map_err(|e| anyhow!("Failed to compile attention.metal: {}", e))?;

        let attention_kernels = ["flash_attention_forward"];
        for name in attention_kernels {
            let function = attention_lib
                .get_function(name, None)
                .map_err(|e| anyhow!("Kernel '{}' not found: {}", name, e))?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| anyhow!("Pipeline for '{}' failed: {}", name, e))?;
            self.pipelines.insert(name.to_string(), pipeline);
        }

        // Compile ops library
        let ops_lib = self
            .device
            .new_library_with_source(ops_src, &options)
            .map_err(|e| anyhow!("Failed to compile ops.metal: {}", e))?;

        let ops_kernels = [
            "rms_norm_bf16",
            "relu_squared_bf16",
            "softcap_bf16",
            "cross_entropy_bf16",
            "adamw_step_bf16",
        ];
        for name in ops_kernels {
            let function = ops_lib
                .get_function(name, None)
                .map_err(|e| anyhow!("Kernel '{}' not found: {}", name, e))?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| anyhow!("Pipeline for '{}' failed: {}", name, e))?;
            self.pipelines.insert(name.to_string(), pipeline);
        }

        Ok(())
    }

    fn get_kernel(&self, name: &str) -> Result<&ComputePipelineState> {
        self.pipelines
            .get(name)
            .ok_or_else(|| anyhow!("Kernel '{}' not found in pipeline cache", name))
    }
}

/// Extract a `&metal::BufferRef` and byte offset from a candle Tensor on the Metal device.
///
/// Candle's Metal backend stores data in `candle_metal_kernels::metal::Buffer` (wrapping
/// `objc2-metal`), while this module uses the `metal` crate (gfx-rs/metal-rs).  Both are
/// thin wrappers around the same ObjC `id<MTLBuffer>` pointer, so we transmute the
/// reference.  The returned reference is valid as long as the tensor (and its underlying
/// `Arc<Buffer>`) is alive — which is guaranteed by the caller holding `&Tensor`.
///
/// # Safety
/// The caller must ensure the tensor outlives any GPU commands referencing the returned buffer.
fn extract_metal_buffer(tensor: &Tensor) -> Result<(&BufferRef, usize)> {
    let (storage, layout) = tensor.storage_and_layout();
    let metal_storage = match &*storage {
        Storage::Metal(ms) => ms,
        _ => return Err(anyhow!("Tensor not on Metal device")),
    };
    let candle_buf = metal_storage.buffer();
    let byte_offset = layout.start_offset() * tensor.dtype().size_in_bytes();
    // Safety: candle's `candle_metal_kernels::metal::Buffer` and the `metal` crate's
    // `BufferRef` are both thin pointer-wrappers around the same ObjC `id<MTLBuffer>`.
    // `candle_buf.as_ref()` yields `&ProtocolObject<dyn MTLBuffer>` — a single pointer to
    // the ObjC object — identical in layout to `&metal::BufferRef`.
    // We extend the lifetime past the RwLockReadGuard because the tensor's `Arc<Buffer>`
    // keeps the underlying allocation alive for the duration of the borrow of `&Tensor`.
    let buf_ref: &BufferRef = unsafe { &*(candle_buf.as_ref() as *const _ as *const BufferRef) };
    Ok((buf_ref, byte_offset))
}

// ============================================================
// Flash Attention
// ============================================================

impl MetalKernels {
    /// Flash attention forward pass.
    /// q: (B, T, n_head, head_dim) bf16
    /// k: (B, T, n_kv_head, head_dim) bf16
    /// v: (B, T, n_kv_head, head_dim) bf16
    /// Returns output tensor (B, T, n_head, head_dim) bf16.
    pub fn flash_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window_size: usize,
    ) -> Result<Tensor> {
        let q_dims = q.dims();
        assert_eq!(q_dims.len(), 4, "q must be (B, T, n_head, head_dim)");
        let (batch, seq_len, n_head, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);

        let k_dims = k.dims();
        let n_kv_head = k_dims[2];

        // Allocate output on the same device
        let out = Tensor::zeros((batch, seq_len, n_head, head_dim), DType::BF16, q.device())?;

        let (q_buf, q_off) = extract_metal_buffer(q)?;
        let (k_buf, k_off) = extract_metal_buffer(k)?;
        let (v_buf, v_off) = extract_metal_buffer(v)?;
        let (o_buf, o_off) = extract_metal_buffer(&out)?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let kernel = self.get_kernel("flash_attention_forward")?;
        encoder.set_compute_pipeline_state(kernel);

        encoder.set_buffer(0, Some(q_buf), q_off as u64);
        encoder.set_buffer(1, Some(k_buf), k_off as u64);
        encoder.set_buffer(2, Some(v_buf), v_off as u64);
        encoder.set_buffer(3, Some(o_buf), o_off as u64);

        #[repr(C)]
        struct AttentionParams {
            batch: u32,
            seq_len: u32,
            n_head: u32,
            n_kv_head: u32,
            head_dim: u32,
            window_size: u32,
        }
        let params = AttentionParams {
            batch: batch as u32,
            seq_len: seq_len as u32,
            n_head: n_head as u32,
            n_kv_head: n_kv_head as u32,
            head_dim: head_dim as u32,
            window_size: window_size as u32,
        };
        encoder.set_bytes(
            4,
            mem::size_of::<AttentionParams>() as u64,
            &params as *const _ as *const _,
        );

        // 2D tile dispatch: (32, 32) threadgroups over (n_head * batch) x ceil(seq_len / 32)
        let threads_per_group = MTLSize::new(32, 32, 1);
        let num_tiles_seq = (seq_len + 31) / 32;
        let thread_groups = MTLSize::new((n_head * batch) as u64, num_tiles_seq as u64, 1);
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(out)
    }
}

// ============================================================
// Elementwise Ops (RMS norm, ReLU^2, softcap)
// ============================================================

impl MetalKernels {
    /// Fused RMS normalization. x: arbitrary shape, bf16.
    /// Normalizes over the last dimension.
    pub fn rms_norm(&self, x: &Tensor, eps: f32) -> Result<Tensor> {
        let numel = x.elem_count();
        let shape = x.dims().to_vec();
        let last_dim = *shape.last().unwrap();
        let rows = numel / last_dim;

        let out = Tensor::zeros(&shape[..], DType::BF16, x.device())?;

        let (x_buf, x_off) = extract_metal_buffer(x)?;
        let (o_buf, o_off) = extract_metal_buffer(&out)?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let kernel = self.get_kernel("rms_norm_bf16")?;
        encoder.set_compute_pipeline_state(kernel);

        encoder.set_buffer(0, Some(x_buf), x_off as u64);
        encoder.set_buffer(1, Some(o_buf), o_off as u64);
        encoder.set_bytes(
            2,
            mem::size_of::<f32>() as u64,
            &eps as *const _ as *const _,
        );
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &(last_dim as u32) as *const _ as *const _,
        );

        // One threadgroup per row, 256 threads per group
        let threads_per_group = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(rows as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(out)
    }

    /// Fused ReLU squared: max(0, x)^2
    pub fn relu_squared(&self, x: &Tensor) -> Result<Tensor> {
        let numel = x.elem_count();
        let out = Tensor::zeros(x.dims(), DType::BF16, x.device())?;

        let (x_buf, x_off) = extract_metal_buffer(x)?;
        let (o_buf, o_off) = extract_metal_buffer(&out)?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let kernel = self.get_kernel("relu_squared_bf16")?;
        encoder.set_compute_pipeline_state(kernel);

        encoder.set_buffer(0, Some(x_buf), x_off as u64);
        encoder.set_buffer(1, Some(o_buf), o_off as u64);
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &(numel as u32) as *const _ as *const _,
        );

        let threads_per_group = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(((numel + 255) / 256) as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(out)
    }

    /// Fused softcap: cap * tanh(x / cap)
    pub fn softcap(&self, x: &Tensor, cap: f32) -> Result<Tensor> {
        let numel = x.elem_count();
        let out = Tensor::zeros(x.dims(), DType::BF16, x.device())?;

        let (x_buf, x_off) = extract_metal_buffer(x)?;
        let (o_buf, o_off) = extract_metal_buffer(&out)?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let kernel = self.get_kernel("softcap_bf16")?;
        encoder.set_compute_pipeline_state(kernel);

        encoder.set_buffer(0, Some(x_buf), x_off as u64);
        encoder.set_buffer(1, Some(o_buf), o_off as u64);
        encoder.set_bytes(
            2,
            mem::size_of::<f32>() as u64,
            &cap as *const _ as *const _,
        );
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &(numel as u32) as *const _ as *const _,
        );

        let threads_per_group = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(((numel + 255) / 256) as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(out)
    }
}

// ============================================================
// Cross Entropy
// ============================================================

impl MetalKernels {
    /// Fused cross entropy loss. Returns per-token losses.
    /// logits: (N, vocab_size) bf16
    /// targets: (N,) u32
    pub fn cross_entropy(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let logits_dims = logits.dims();
        assert_eq!(logits_dims.len(), 2);
        let n = logits_dims[0];
        let vocab_size = logits_dims[1];

        // Output: per-token scalar losses in f32
        let out = Tensor::zeros(n, DType::F32, logits.device())?;

        let (logits_buf, logits_off) = extract_metal_buffer(logits)?;
        let (targets_buf, targets_off) = extract_metal_buffer(targets)?;
        let (out_buf, out_off) = extract_metal_buffer(&out)?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let kernel = self.get_kernel("cross_entropy_bf16")?;
        encoder.set_compute_pipeline_state(kernel);

        encoder.set_buffer(0, Some(logits_buf), logits_off as u64);
        encoder.set_buffer(1, Some(targets_buf), targets_off as u64);
        encoder.set_buffer(2, Some(out_buf), out_off as u64);
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &(vocab_size as u32) as *const _ as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &(n as u32) as *const _ as *const _,
        );

        // One threadgroup per token, 256 threads for the reduction over vocab
        let threads_per_group = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(out)
    }
}

// ============================================================
// AdamW
// ============================================================

impl MetalKernels {
    /// Fused AdamW optimizer step on Metal.
    /// All tensors are bf16 and must be on the Metal device.
    /// Returns updated (param, exp_avg, exp_avg_sq).
    pub fn adamw_step(
        &self,
        param: &Tensor,
        grad: &Tensor,
        exp_avg: &Tensor,
        exp_avg_sq: &Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bias1_corr: f32,
        bias2_corr: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let numel = param.elem_count();

        // Output tensors (updated in-place on GPU, returned as new candle tensors)
        let param_out = Tensor::zeros(param.dims(), DType::BF16, param.device())?;
        let m_out = Tensor::zeros(param.dims(), DType::BF16, param.device())?;
        let v_out = Tensor::zeros(param.dims(), DType::BF16, param.device())?;

        let (param_buf, param_off) = extract_metal_buffer(param)?;
        let (grad_buf, grad_off) = extract_metal_buffer(grad)?;
        let (m_buf, m_off) = extract_metal_buffer(exp_avg)?;
        let (v_buf, v_off) = extract_metal_buffer(exp_avg_sq)?;
        let (param_out_buf, po_off) = extract_metal_buffer(&param_out)?;
        let (m_out_buf, mo_off) = extract_metal_buffer(&m_out)?;
        let (v_out_buf, vo_off) = extract_metal_buffer(&v_out)?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let kernel = self.get_kernel("adamw_step_bf16")?;
        encoder.set_compute_pipeline_state(kernel);

        encoder.set_buffer(0, Some(param_buf), param_off as u64);
        encoder.set_buffer(1, Some(grad_buf), grad_off as u64);
        encoder.set_buffer(2, Some(m_buf), m_off as u64);
        encoder.set_buffer(3, Some(v_buf), v_off as u64);
        encoder.set_buffer(4, Some(param_out_buf), po_off as u64);
        encoder.set_buffer(5, Some(m_out_buf), mo_off as u64);
        encoder.set_buffer(6, Some(v_out_buf), vo_off as u64);

        #[repr(C)]
        struct AdamWParams {
            lr: f32,
            beta1: f32,
            beta2: f32,
            eps: f32,
            wd: f32,
            bias1_corr: f32,
            bias2_corr: f32,
            numel: u32,
        }
        let params = AdamWParams {
            lr,
            beta1,
            beta2,
            eps,
            wd,
            bias1_corr,
            bias2_corr,
            numel: numel as u32,
        };
        encoder.set_bytes(
            7,
            mem::size_of::<AdamWParams>() as u64,
            &params as *const _ as *const _,
        );

        let threads_per_group = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(((numel + 255) / 256) as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((param_out, m_out, v_out))
    }
}

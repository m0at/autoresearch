//! Pipeline-parallel training across multiple GPUs.
//!
//! Splits N_LAYER across N_GPU devices. Each stage holds a contiguous chunk of
//! layers plus its own BufferManager, GemmRunner, and CUDA stream. Activations
//! flow between stages via P2P memcpy (NVLink on multi-GPU nodes).
//!
//! MVP: sequential pipeline (no micro-batch overlap). Each stage processes the
//! full batch before passing to the next. 1F1B micro-batch scheduling can be
//! added later for better GPU utilization.

use std::sync::Arc;
use anyhow::{ensure, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::backward::{backward_staged_ex, embedding_backward};
use crate::buffer::BufferManager;
use crate::config::*;
use crate::ffi;
use crate::forward::forward_staged;
use crate::gemm::GemmRunner;
use crate::optim::{optimizer_step, ScheduleConfig};

/// Which layers and resources a single GPU stage owns.
pub struct PipelineStage {
    pub device: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub bufs: BufferManager,
    pub gemm: GemmRunner,
    /// Inclusive layer range: layers[start..end]
    pub layer_start: usize,
    pub layer_end: usize, // exclusive
    /// True if this stage holds the embedding layer (first stage).
    pub has_embedding: bool,
    /// True if this stage holds the lm_head + loss (last stage).
    pub has_head: bool,
}

/// Multi-GPU pipeline for training.
pub struct Pipeline {
    pub stages: Vec<PipelineStage>,
    pub n_gpu: usize,
}

impl Pipeline {
    /// Create a pipeline across `n_gpu` devices.
    /// Layers are split as evenly as possible, with remainder layers going to
    /// earlier stages (they tend to be cheaper — no lm_head).
    pub fn new(n_gpu: usize, batch_size: usize) -> Result<Self> {
        ensure!(n_gpu >= 1 && n_gpu <= 8, "n_gpu must be 1..8, got {n_gpu}");
        ensure!(N_LAYER >= n_gpu, "N_LAYER ({N_LAYER}) must be >= n_gpu ({n_gpu})");

        let base = N_LAYER / n_gpu;
        let remainder = N_LAYER % n_gpu;

        let mut stages = Vec::with_capacity(n_gpu);
        let mut layer_offset = 0;

        for gpu_idx in 0..n_gpu {
            let n_layers = base + if gpu_idx < remainder { 1 } else { 0 };
            let layer_start = layer_offset;
            let layer_end = layer_offset + n_layers;
            layer_offset = layer_end;

            let device = CudaContext::new(gpu_idx)?;
            let stream = device.new_stream()?;

            // Each stage allocates its own buffers for its layer range.
            // BufferManager::new_staged() allocates only the layers in [start, end).
            let bufs = BufferManager::new_staged(
                Arc::clone(&stream),
                batch_size,
                layer_start,
                layer_end,
                gpu_idx == 0,           // has_embedding
                gpu_idx == n_gpu - 1,   // has_head
            )?;

            let gemm = GemmRunner::new(Arc::clone(&stream));

            stages.push(PipelineStage {
                device,
                stream,
                bufs,
                gemm,
                layer_start,
                layer_end,
                has_embedding: gpu_idx == 0,
                has_head: gpu_idx == n_gpu - 1,
            });
        }

        // Enable P2P access between all GPU pairs
        for i in 0..n_gpu {
            for j in 0..n_gpu {
                if i == j { continue; }
                unsafe {
                    // Bind source context, then enable peer access to destination
                    stages[i].device.bind_to_thread()?;
                    let _ = cudarc::driver::sys::cuCtxEnablePeerAccess(
                        stages[j].device.cu_ctx(), 0,
                    );
                }
            }
        }
        // Rebind stage 0 context
        stages[0].device.bind_to_thread()?;

        for i in 0..n_gpu - 1 {
            let dev_a = stages[i].device.cu_device();
            let dev_b = stages[i + 1].device.cu_device();
            eprintln!("Pipeline stage {i} (GPU {dev_a}, layers {}-{}) -> stage {} (GPU {dev_b}, layers {}-{})",
                stages[i].layer_start, stages[i].layer_end,
                i + 1,
                stages[i + 1].layer_start, stages[i + 1].layer_end,
            );
        }

        println!("Pipeline: {n_gpu} stages, {N_LAYER} layers");
        for (i, s) in stages.iter().enumerate() {
            println!("  Stage {i}: GPU {}, layers {}..{} ({} layers){}{}",
                s.device.cu_device(),
                s.layer_start, s.layer_end,
                s.layer_end - s.layer_start,
                if s.has_embedding { " [embed]" } else { "" },
                if s.has_head { " [head]" } else { "" },
            );
        }

        Ok(Pipeline { stages, n_gpu })
    }

    /// Transfer activation tensor `x` from stage `src` to stage `dst`.
    /// Uses peer-to-peer memcpy via the destination stream.
    pub fn send_activation(
        &self,
        src_stage: usize,
        dst_stage: usize,
        n_elements: usize,
    ) -> Result<()> {
        let src = &self.stages[src_stage];
        let dst = &self.stages[dst_stage];

        let src_ptr = src.bufs.x_device_ptr();
        let dst_ptr = dst.bufs.x_device_ptr_mut();
        let nbytes = n_elements * std::mem::size_of::<bf16>();

        // Sync source stream before copy (ensure forward pass is complete)
        src.stream.synchronize()?;

        // P2P copy on destination stream
        unsafe {
            cudarc::driver::sys::cuMemcpyPeerAsync(
                    dst_ptr,
                    dst.device.cu_ctx(),
                    src_ptr,
                    src.device.cu_ctx(),
                    nbytes,
                    dst.stream.cu_stream(),
                );
        }

        Ok(())
    }

    /// Transfer gradient tensor `d_x` from stage `src` back to stage `dst` (backward).
    /// Called as send_gradient(i, i-1): copies d_x FROM later stage i TO earlier stage i-1.
    pub fn send_gradient(
        &self,
        src_stage: usize,
        dst_stage: usize,
        n_elements: usize,
    ) -> Result<()> {
        let src = &self.stages[src_stage];
        let dst = &self.stages[dst_stage];

        let src_ptr = src.bufs.dx_device_ptr();
        let dst_ptr = dst.bufs.dx_device_ptr_mut();
        let nbytes = n_elements * std::mem::size_of::<bf16>();

        // Sync source stream before copy (ensure backward pass is complete)
        src.stream.synchronize()?;

        // P2P copy on destination stream
        unsafe {
            cudarc::driver::sys::cuMemcpyPeerAsync(
                    dst_ptr,
                    dst.device.cu_ctx(),
                    src_ptr,
                    src.device.cu_ctx(),
                    nbytes,
                    dst.stream.cu_stream(),
                );
        }

        Ok(())
    }

    /// Transfer `x0` (residual stream) from stage `src` to stage `dst`.
    /// Every stage needs x0 for the residual_scale connections.
    pub fn send_x0(
        &self,
        src_stage: usize,
        dst_stage: usize,
        n_elements: usize,
    ) -> Result<()> {
        let src = &self.stages[src_stage];
        let dst = &self.stages[dst_stage];

        let src_ptr = src.bufs.x0_device_ptr();
        let dst_ptr = dst.bufs.x0_device_ptr_mut();
        let nbytes = n_elements * std::mem::size_of::<bf16>();

        src.stream.synchronize()?;

        unsafe {
            cudarc::driver::sys::cuMemcpyPeerAsync(
                    dst_ptr,
                    dst.device.cu_ctx(),
                    src_ptr,
                    src.device.cu_ctx(),
                    nbytes,
                    dst.stream.cu_stream(),
                );
        }

        Ok(())
    }

    /// Transfer `targets` from stage 0 to the last stage for CE loss computation.
    pub fn send_targets(
        &self,
        n_elements: usize,
    ) -> Result<()> {
        let last = self.n_gpu - 1;
        if last == 0 { return Ok(()); }

        let src = &self.stages[0];
        let dst = &self.stages[last];

        let src_ptr = src.bufs.targets_device_ptr();
        let dst_ptr = dst.bufs.targets_device_ptr();
        let nbytes = n_elements * std::mem::size_of::<u32>();

        src.stream.synchronize()?;

        unsafe {
            cudarc::driver::sys::cuMemcpyPeerAsync(
                    dst_ptr,
                    dst.device.cu_ctx(),
                    src_ptr,
                    src.device.cu_ctx(),
                    nbytes,
                    dst.stream.cu_stream(),
                );
        }

        Ok(())
    }

    // =====================================================================
    //  Pipeline orchestration
    // =====================================================================

    /// Load `input_ids` and `targets` onto stage 0. Also copies `targets` to
    /// the last stage (needed for CE loss computation on the head stage).
    ///
    /// `input_ids` and `targets` are host-side slices of u32 with length BT.
    pub fn load_data(
        &mut self,
        input_ids: &[u32],
        targets: &[u32],
    ) -> Result<()> {
        let bt = self.stages[0].bufs.batch_size * SEQ;
        assert_eq!(input_ids.len(), bt);
        assert_eq!(targets.len(), bt);

        // Upload input_ids and targets to stage 0
        {
            let s0 = &mut self.stages[0];
            s0.stream.memcpy_htod(input_ids, &mut s0.bufs.input_ids)?;
            s0.stream.memcpy_htod(targets, &mut s0.bufs.targets)?;
        }

        // Copy targets to last stage for CE loss (P2P if multi-GPU)
        self.send_targets(bt)?;

        Ok(())
    }

    /// Read the scalar loss from the last stage (synchronizes that stage's stream).
    /// Returns the mean loss (loss_sum / total_tokens).
    pub fn read_loss(&self) -> Result<f32> {
        let last = self.n_gpu - 1;
        let stage = &self.stages[last];

        // Sync the last stage's stream so the loss value is ready
        stage.stream.synchronize()?;

        let mut loss_val = [0.0f32; 1];
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                loss_val.as_mut_ptr() as *mut std::ffi::c_void,
                stage.bufs.loss_device_ptr(),
                std::mem::size_of::<f32>(),
            );
        }

        let bt = stage.bufs.batch_size * SEQ;
        Ok(loss_val[0] / bt as f32)
    }

    /// Execute one full training step across all pipeline stages.
    ///
    /// Sequential pipeline (MVP, no micro-batch overlap):
    ///   Forward:  stage 0 -> stage 1 -> ... -> stage N-1
    ///   Backward: stage N-1 -> ... -> stage 0
    ///   Optimize: each stage updates its own parameters
    ///
    /// d_x0 handling: Each stage accumulates d_x0 from its own layers during
    /// backward. For the embedding backward on stage 0 to be correct, it needs
    /// the SUM of d_x0 across all stages. We run backward on stages N-1..1
    /// first, gather their d_x0 into a scratch buffer on stage 0, then run
    /// backward on stage 0, and finally add the gathered d_x0 contributions
    /// before the embedding backward (which is handled by a separate call to
    /// `embedding_backward_fixup`).
    pub fn train_step(
        &mut self,
        step: usize,
        progress: f64,
        cfg: &ScheduleConfig,
        grad_accum_steps: usize,
    ) -> Result<()> {
        let n = self.n_gpu;
        let bt = self.stages[0].bufs.batch_size * SEQ;
        let n_act = bt * D_MODEL; // bf16 elements for x and x0 transfers
        let n_btd = (bt * D_MODEL) as i32;

        // Zero loss on last stage before forward
        {
            let last = &mut self.stages[n - 1];
            last.stream.memset_zeros(&mut last.bufs.loss)?;
        }

        // =================================================================
        //  Forward pass: stage 0 -> 1 -> ... -> N-1
        // =================================================================

        for i in 0..n {
            {
                unsafe { crate::ffi::cuda_set_device(i as i32); }
                let stage = &mut self.stages[i];
                forward_staged(&mut stage.bufs, &stage.gemm);
            }

            // P2P transfer x and x0 to next stage
            if i + 1 < n {
                self.send_activation(i, i + 1, n_act)?;
                self.send_x0(i, i + 1, n_act)?;

                // Sync destination stream to ensure P2P copies land before
                // next stage's forward begins
                self.stages[i + 1].stream.synchronize()?;
            }
        }

        // =================================================================
        //  Backward pass: stage N-1 -> ... -> 0
        // =================================================================
        //
        // d_x0 correctness: each stage zeros d_x0, then accumulates from
        // its own layers. Stage 0's embedding backward needs the TOTAL d_x0
        // (sum across all stages).
        //
        // Strategy (multi-GPU):
        //   1. Run backward_staged_ex on all stages, with skip_embedding_bwd
        //      on stage 0 so the embedding backward is deferred.
        //   2. P2P copy + residual_add d_x0 from stages 1..N-1 into stage 0.
        //   3. Run embedding_backward on stage 0 with the complete d_x0.

        if n == 1 {
            let stage = &mut self.stages[0];
            backward_staged_ex(&mut stage.bufs, &stage.gemm, grad_accum_steps, false);
        } else {
            // Phase 1: backward on all stages (stage 0 skips embedding bwd)
            for i in (0..n).rev() {
                let skip_emb = i == 0;
                {
                    let stage = &mut self.stages[i];
                    unsafe { crate::ffi::cuda_set_device(i as i32); }
                    backward_staged_ex(
                        &mut stage.bufs,
                        &stage.gemm,
                        grad_accum_steps,
                        skip_emb,
                    );
                }

                // Transfer d_x to previous stage
                if i > 0 {
                    self.send_gradient(i, i - 1, n_act)?;
                    self.stages[i - 1].stream.synchronize()?;
                }
            }

            // Phase 2: accumulate d_x0 from remote stages into stage 0.
            // Each remote stage's d_x0 buffer holds its partial contribution
            // from its own layers' residual_scale_bwd. We P2P copy each into
            // stage 0's d_xn scratch, then residual_add into stage 0's d_x0.
            for i in 1..n {
                // P2P: stage[i].d_x0 -> stage[0].d_xn (scratch)
                {
                    let src = &self.stages[i];
                    let dst = &self.stages[0];
                    let src_ptr = {
                        let (ptr, _) = src.bufs.d_x0.device_ptr(src.bufs.d_x0.stream());
                        ptr
                    };
                    let dst_ptr = {
                        let (ptr, _) = dst.bufs.d_xn.device_ptr(dst.bufs.d_xn.stream());
                        ptr
                    };
                    let nbytes = n_act * std::mem::size_of::<bf16>();

                    src.stream.synchronize()?;

                    unsafe {
                        cudarc::driver::sys::cuMemcpyPeerAsync(
                                dst_ptr,
                                dst.device.cu_ctx(),
                                src_ptr,
                                src.device.cu_ctx(),
                                nbytes,
                                dst.stream.cu_stream(),
                        );
                    }
                }

                self.stages[0].stream.synchronize()?;

                // d_x0 += d_xn (accumulate remote contribution)
                {
                    let s0 = &self.stages[0];
                    let stream = s0.stream.cu_stream() as ffi::CudaStream;
                    let d_x0_ptr = {
                        let (ptr, _) = s0.bufs.d_x0.device_ptr(s0.bufs.d_x0.stream());
                        ptr
                    };
                    let d_xn_ptr = {
                        let (ptr, _) = s0.bufs.d_xn.device_ptr(s0.bufs.d_xn.stream());
                        ptr
                    };
                    unsafe {
                        ffi::residual_add(
                            d_x0_ptr as *mut std::ffi::c_void,
                            d_xn_ptr as *const std::ffi::c_void,
                            n_btd,
                            stream,
                        );
                    }
                }
            }

            // Phase 3: embedding backward on stage 0 with complete d_x0
            {
                unsafe { crate::ffi::cuda_set_device(0); }
                let stage = &mut self.stages[0];
                embedding_backward(&mut stage.bufs);
            }
        }

        // =================================================================
        //  Optimizer step: each stage updates its own parameters
        // =================================================================

        for i in 0..n {
            unsafe { crate::ffi::cuda_set_device(i as i32); }
            let stage = &mut self.stages[i];
            optimizer_step(
                &mut stage.bufs,
                &stage.gemm,
                step,
                progress,
                cfg,
            );
            stage.stream.synchronize()?;
        }

        Ok(())
    }

    /// Transfer `d_x0` gradient backward: accumulate from the stage that computed
    /// backward for [start..end] back toward stage 0 which holds the embedding.
    pub fn send_dx0(
        &self,
        src_stage: usize,
        dst_stage: usize,
        n_elements: usize,
    ) -> Result<()> {
        let src = &self.stages[src_stage];
        let dst = &self.stages[dst_stage];

        let src_ptr = {
            let (ptr, _) = src.bufs.d_x0.device_ptr(src.bufs.d_x0.stream());
            ptr
        };
        let dst_ptr = {
            let (ptr, _) = dst.bufs.d_x0.device_ptr(dst.bufs.d_x0.stream());
            ptr
        };
        let nbytes = n_elements * std::mem::size_of::<bf16>();

        src.stream.synchronize()?;

        unsafe {
            cudarc::driver::sys::cuMemcpyPeerAsync(
                    dst_ptr,
                    dst.device.cu_ctx(),
                    src_ptr,
                    src.device.cu_ctx(),
                    nbytes,
                    dst.stream.cu_stream(),
                );
        }

        Ok(())
    }
}

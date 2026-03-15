use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use cudarc::driver::sys as cuda_sys;
use half::bf16;

use crate::buffer::BufferManager;
use crate::config::*;
use crate::gemm::GemmRunner;

#[inline(always)]
fn dptr<T>(buf: &cudarc::driver::CudaSlice<T>) -> cudarc::driver::sys::CUdeviceptr {
    let (ptr, _sync) = buf.device_ptr(buf.stream());
    ptr
}

#[inline(always)]
fn vptr<T>(buf: &cudarc::driver::CudaSlice<T>) -> *const std::ffi::c_void {
    dptr(buf) as *const std::ffi::c_void
}

// ---------------------------------------------------------------------------
// Pinned (page-locked) host memory for async H2D copies
// ---------------------------------------------------------------------------

/// RAII wrapper around CUDA pinned host memory allocated via `cuMemAllocHost_v2`.
/// Pinned memory enables truly async H2D transfers — the GPU DMA engine copies
/// while kernels continue executing, instead of blocking the stream.
struct PinnedHostBuffer {
    ptr: *mut std::ffi::c_void,
    len: usize, // number of u32 elements
}

impl PinnedHostBuffer {
    /// Allocate `count` u32 elements of page-locked host memory.
    fn new(count: usize) -> Result<Self> {
        let bytesize = count * std::mem::size_of::<u32>();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cuda_sys::cuMemAllocHost_v2(&mut ptr, bytesize) };
        ensure!(
            err == cuda_sys::cudaError_enum::CUDA_SUCCESS,
            "cuMemAllocHost_v2 failed: {err:?}"
        );
        Ok(Self { ptr, len: count })
    }

    /// Get a mutable slice view of the pinned buffer as `&mut [u32]`.
    fn as_mut_slice(&mut self) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u32, self.len) }
    }

    /// Raw pointer for passing to `cuMemcpyHtoDAsync_v2`.
    fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr
    }

    /// Size in bytes.
    fn bytesize(&self) -> usize {
        self.len * std::mem::size_of::<u32>()
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuda_sys::cuMemFreeHost(self.ptr) };
        }
    }
}

// SAFETY: The pinned host buffer is just a host-side allocation, safe to send across threads.
unsafe impl Send for PinnedHostBuffer {}

/// Async H2D copy from pinned host memory to a device CudaSlice<u32>.
/// The copy runs on `stream` and returns immediately — the GPU DMA engine
/// handles the transfer in the background.
fn memcpy_htod_async(
    pinned: &PinnedHostBuffer,
    dst: &mut CudaSlice<u32>,
    stream: &Arc<CudaStream>,
) {
    let dst_ptr = dptr(dst);
    let err = unsafe {
        cuda_sys::cuMemcpyHtoDAsync_v2(
            dst_ptr,
            pinned.as_ptr(),
            pinned.bytesize(),
            stream.cu_stream(),
        )
    };
    assert_eq!(err, cuda_sys::cudaError_enum::CUDA_SUCCESS, "cuMemcpyHtoDAsync_v2 failed");
}

/// Read back the device-side loss sum from `bufs.loss[0]` and return the mean.
/// Must be called after a stream sync so the value is ready on the host side.
fn read_mean_loss(
    _stream: &Arc<CudaStream>,
    bufs: &BufferManager,
    total_tokens: usize,
) -> f64 {
    let mut loss_sum = [0.0f32; 1];
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoH_v2(
            loss_sum.as_mut_ptr() as *mut std::ffi::c_void,
            dptr(&bufs.loss),
            std::mem::size_of::<f32>(),
        );
    }
    loss_sum[0] as f64 / total_tokens as f64
}

fn read_f32_buf(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let mut host = vec![0.0f32; buf.len()];
    stream.memcpy_dtoh(buf, &mut host).expect("dtoh layer stats");
    host
}

/// How often (in optimizer steps) to sync the GPU and log metrics.
/// Between log steps, the GPU pipeline runs without any host sync.
const LOG_INTERVAL: usize = 5;

// ---------------------------------------------------------------------------
// Training hyperparameters (matching Python / candle reference)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
const EMBEDDING_LR: f64 = 0.9;
#[allow(dead_code)]
const UNEMBEDDING_LR: f64 = 0.005;
#[allow(dead_code)]
const MATRIX_LR: f64 = 0.04;
#[allow(dead_code)]
const SCALAR_LR: f64 = 0.5;
#[allow(dead_code)]
const WEIGHT_DECAY: f64 = 0.2;
#[allow(dead_code)]
const ADAM_BETA1: f64 = 0.8;
#[allow(dead_code)]
const ADAM_BETA2: f64 = 0.95;
#[allow(dead_code)]
const ADAM_EPS: f64 = 1e-10;
const WARMUP_RATIO: f64 = 0.0;
const WARMDOWN_RATIO: f64 = 0.75;
const FINAL_LR_FRAC: f64 = 0.05;

/// Estimated FLOPs per token (forward + backward).
fn estimate_flops_per_token() -> usize {
    let vocab = VOCAB;
    let d = D_MODEL;
    let n_head = N_HEAD;
    let head_dim = HEAD_DIM;
    let n_layer = N_LAYER;

    // Total params excluding embeddings, VE, and scalars
    let wte_numel = vocab * d;
    let lm_head_numel = vocab * d;
    let ve_numel = VE_LAYERS.len() * vocab * d;
    let scalar_numel = n_layer * 2;
    let ve_gate_numel: usize = VE_LAYERS.len() * N_KV_HEAD * VE_GATE_CH;

    // Block weights: per layer: 4*d*d (q,k,v,o) + MLP (dense or MoE active params)
    let mut block_params = 0;
    for i in 0..n_layer {
        block_params += 4 * d * d;
        if is_moe_layer(i) {
            // Active per token: router + TOP_K experts × 2 matrices
            block_params += N_EXPERTS * d + TOP_K * 2 * MLP_DIM_E * d;
        } else {
            block_params += 2 * MLP_DIM * d;
        }
    }
    let total_params = wte_numel + lm_head_numel + ve_numel + scalar_numel + ve_gate_numel + block_params;
    let nparams_exclude = wte_numel + ve_numel + scalar_numel;

    // Attention FLOPs per token
    let mut attn_flops: usize = 0;
    for &ws in &WINDOW_SIZES {
        let effective_seq = ws.min(SEQ);
        attn_flops += 12 * n_head * head_dim * effective_seq;
    }

    6 * (total_params - nparams_exclude) + attn_flops
}

/// Total number of model parameters.
fn num_params() -> usize {
    let d = D_MODEL;
    let wte = VOCAB * d;
    let lm_head = VOCAB * d;
    let ve = VE_LAYERS.len() * VOCAB * d;
    let ve_gate = VE_LAYERS.len() * N_KV_HEAD * VE_GATE_CH;
    let scalars = N_LAYER * 2;
    let mut block = 0;
    for i in 0..N_LAYER {
        // Attention: q, k, v, o
        block += 4 * d * d;
        if is_moe_layer(i) {
            // Router: [N_EXPERTS, D_MODEL] + N_EXPERTS experts × 2 matrices [MLP_DIM_E, D_MODEL]
            block += N_EXPERTS * d + N_EXPERTS * 2 * MLP_DIM_E * d;
        } else {
            block += 2 * MLP_DIM * d;
        }
    }
    wte + lm_head + ve + ve_gate + scalars + block
}

// ---------------------------------------------------------------------------
// LR / momentum / weight-decay schedules
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn get_lr_multiplier(progress: f64) -> f64 {
    if progress < WARMUP_RATIO {
        if WARMUP_RATIO > 0.0 {
            progress / WARMUP_RATIO
        } else {
            1.0
        }
    } else if progress < 1.0 - WARMDOWN_RATIO {
        1.0
    } else {
        let cooldown = (1.0 - progress) / WARMDOWN_RATIO;
        cooldown + (1.0 - cooldown) * FINAL_LR_FRAC
    }
}

#[allow(dead_code)]
fn get_muon_momentum(step: usize) -> f64 {
    let frac = (step as f64 / 200.0).min(1.0);
    (1.0 - frac) * 0.85 + frac * 0.95
}

#[allow(dead_code)]
fn get_weight_decay(progress: f64) -> f64 {
    WEIGHT_DECAY * (1.0 - progress)
}

// ---------------------------------------------------------------------------
// Shard reader (standalone, no candle dependency)
// ---------------------------------------------------------------------------

const SHARD_MAGIC: &[u8; 4] = b"TKNS";
const SHARD_HEADER_SIZE: usize = 20;
const SHARD_VERSION: u32 = 1;

struct ShardHeader {
    seq_len: u32,
    num_rows: u32,
}

fn read_shard_header(data: &[u8]) -> Result<ShardHeader> {
    ensure!(data.len() >= SHARD_HEADER_SIZE, "shard too small for header");
    ensure!(&data[0..4] == SHARD_MAGIC, "bad magic bytes");
    let version = u32::from_le_bytes(data[4..8].try_into()?);
    ensure!(version == SHARD_VERSION, "unsupported shard version {version}");
    let _vocab_size = u32::from_le_bytes(data[8..12].try_into()?);
    let seq_len = u32::from_le_bytes(data[12..16].try_into()?);
    let num_rows = u32::from_le_bytes(data[16..20].try_into()?);
    let expected = SHARD_HEADER_SIZE + (num_rows as usize) * (seq_len as usize) * 2;
    ensure!(
        data.len() >= expected,
        "shard truncated: expected {expected} bytes, got {}",
        data.len()
    );
    Ok(ShardHeader { seq_len, num_rows })
}

struct LoadedShard {
    data: Vec<u8>,
    header: ShardHeader,
}

impl LoadedShard {
    fn open(path: &Path) -> Result<Self> {
        let data = fs::read(path)?;
        let header = read_shard_header(&data)?;
        Ok(Self { data, header })
    }

    fn num_rows(&self) -> usize {
        self.header.num_rows as usize
    }

    fn seq_len(&self) -> usize {
        self.header.seq_len as usize
    }

    /// Get row `idx` as a slice of u16 tokens.
    fn row(&self, idx: usize) -> &[u16] {
        let sl = self.header.seq_len as usize;
        let byte_offset = SHARD_HEADER_SIZE + idx * sl * 2;
        let ptr = self.data[byte_offset..].as_ptr() as *const u16;
        unsafe { std::slice::from_raw_parts(ptr, sl) }
    }
}

/// List binary shard files matching `shard_*.bin`, sorted by name.
fn list_shard_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .expect("cannot read shard dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "bin")
                && p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .starts_with("shard_")
        })
        .collect();
    files.sort();
    files
}

// ---------------------------------------------------------------------------
// Deterministic shard shuffling (xorshift64 + Fisher-Yates)
// ---------------------------------------------------------------------------

fn shuffle_shard_order(order: &mut [usize], seed: u64) {
    let n = order.len();
    if n <= 1 {
        return;
    }
    // Reset to identity so shuffle is purely a function of seed
    for i in 0..n {
        order[i] = i;
    }
    let mut rng = seed;
    // xorshift64 step
    let mut next = || -> u64 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

/// Stdin-based streaming data loader. Reads packed rows from a pipe (feeder.py).
/// Each row is ROW_LEN u16 values (ROW_LEN * 2 bytes), no framing.
const ROW_LEN: usize = SEQ + 1; // 2049 tokens per row
const ROW_BYTES: usize = ROW_LEN * 2;

// How many parsed batches to keep buffered ahead of the GPU.
// At B=8, T=32768, each batch is ~512KB. 16 batches ≈ 8MB of look-ahead,
// enough to absorb feeder jitter.
const STDIN_PREFETCH_BATCHES: usize = 16;

/// Stdin loader with a background reader thread.
///
/// The background thread parses rows from feeder.py and fills a bounded channel
/// with fully-decoded (u32) batches.  The training thread calls `next_batch_into`
/// which is an instant channel recv if prefetch is keeping up — eliminating the
/// wall_gap stall that degraded MFU from 23.6% (synthetic) down to ~19-20%.
struct StdinDataLoader {
    rx: std::sync::mpsc::Receiver<(Vec<u32>, Vec<u32>)>,
    batch_size: usize,
}

impl StdinDataLoader {
    fn new(batch_size: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel(STDIN_PREFETCH_BATCHES);
        std::thread::spawn(move || {
            use std::io::Read;
            let stdin = std::io::stdin();
            // Large kernel buffer so the OS pipe never backs up waiting for us.
            let mut reader = std::io::BufReader::with_capacity(
                ROW_BYTES * batch_size * (STDIN_PREFETCH_BATCHES + 4),
                stdin,
            );
            let mut row_buf = vec![0u8; ROW_BYTES];
            loop {
                let mut inp = vec![0u32; batch_size * SEQ];
                let mut tgt = vec![0u32; batch_size * SEQ];
                for row in 0..batch_size {
                    if reader.read_exact(&mut row_buf).is_err() {
                        return; // feeder exited — training loop will handle
                    }
                    let off = row * SEQ;
                    for j in 0..SEQ {
                        inp[off + j] = u16::from_le_bytes([row_buf[j * 2], row_buf[j * 2 + 1]]) as u32;
                    }
                    for j in 0..SEQ {
                        tgt[off + j] = u16::from_le_bytes([row_buf[(j + 1) * 2], row_buf[(j + 1) * 2 + 1]]) as u32;
                    }
                }
                if tx.send((inp, tgt)).is_err() {
                    return; // training loop dropped the receiver — clean exit
                }
            }
        });
        Self { rx, batch_size }
    }

    fn next_batch_into(&mut self, input_ids: &mut [u32], targets: &mut [u32]) {
        let (inp, tgt) = self.rx.recv().expect("feeder thread died unexpectedly");
        input_ids.copy_from_slice(&inp);
        targets.copy_from_slice(&tgt);
    }
}

/// Synthetic data loader — LCG pseudo-random token IDs, no I/O.
/// Used to measure MFU without feeder.py pipe or shard read overhead.
struct SyntheticDataLoader {
    buf: Vec<u32>,
    pos: usize,
    batch_size: usize,
}

impl SyntheticDataLoader {
    fn new(batch_size: usize) -> Self {
        let t = SEQ + 1;
        let n = batch_size * t * 8; // 8 batches pre-allocated, cycled
        let mut buf = vec![0u32; n];
        let mut state: u64 = 0xdeadbeef12345678;
        for x in buf.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = ((state >> 33) as u32 % 8191) + 1; // [1, 8191]
        }
        Self { buf, pos: 0, batch_size }
    }

    fn next_batch_into(&mut self, input_ids: &mut [u32], targets: &mut [u32]) {
        let b = self.batch_size;
        let t = SEQ;
        let rows_total = self.buf.len() / (SEQ + 1);
        for row in 0..b {
            let src = ((self.pos + row) % rows_total) * (SEQ + 1);
            let dst = row * t;
            input_ids[dst..dst + t].copy_from_slice(&self.buf[src..src + t]);
            targets[dst..dst + t].copy_from_slice(&self.buf[src + 1..src + t + 1]);
        }
        self.pos = (self.pos + b) % rows_total;
    }
}

/// Minimal shard-based data loader for training (no candle dependency).
struct ShardDataLoader {
    shards: Vec<LoadedShard>,
    shard_order: Vec<usize>,
    row_order: Vec<usize>,
    batch_size: usize,
    shard_idx: usize,
    row_idx: usize,
    epoch: usize,
}

impl ShardDataLoader {
    /// Open train shards. If `num_train` is set, use shards [0..num_train).
    /// Otherwise falls back to all-but-last.
    fn new_train(dir: &Path, batch_size: usize, num_train: Option<usize>) -> Result<Self> {
        let all_paths = list_shard_files(dir);
        ensure!(!all_paths.is_empty(), "no shard files found in {}", dir.display());

        let split = num_train.unwrap_or_else(|| {
            if all_paths.len() > 1 { all_paths.len() - 1 } else { all_paths.len() }
        });
        ensure!(split <= all_paths.len(), "num_train_shards ({split}) > total shards ({})", all_paths.len());
        let paths: Vec<PathBuf> = all_paths[..split].to_vec();

        let shards: Vec<LoadedShard> = paths
            .iter()
            .map(|p| LoadedShard::open(p))
            .collect::<Result<_>>()?;

        let mut shard_order: Vec<usize> = (0..shards.len()).collect();
        // Don't shuffle epoch 1: pre-packed shards are already in the correct
        // shuffled order from prepack.py (random.seed(42)). Re-shuffling would
        // change the data ordering and hurt convergence vs streaming baseline.

        let first_rows = shards[shard_order[0]].num_rows();
        let row_order: Vec<usize> = (0..first_rows).collect();

        Ok(Self {
            shards,
            shard_order,
            row_order,
            batch_size,
            shard_idx: 0,
            row_idx: 0,
            epoch: 1,
        })
    }

    /// Open val shards. If `num_train` is set, val = shards [num_train..).
    /// Otherwise val = last shard only.
    fn new_val(dir: &Path, batch_size: usize, num_train: Option<usize>) -> Result<Self> {
        let all_paths = list_shard_files(dir);
        ensure!(!all_paths.is_empty(), "no shard files found in {}", dir.display());

        let split = num_train.unwrap_or(all_paths.len() - 1);
        let val_paths = &all_paths[split..];
        ensure!(!val_paths.is_empty(), "no val shards (split={split}, total={})", all_paths.len());

        let shards: Vec<LoadedShard> = val_paths
            .iter()
            .map(|p| LoadedShard::open(p))
            .collect::<Result<_>>()?;
        let shard_order: Vec<usize> = (0..shards.len()).collect();
        let first_rows = shards[shard_order[0]].num_rows();
        let row_order: Vec<usize> = (0..first_rows).collect();

        Ok(Self {
            shards,
            shard_order,
            row_order,
            batch_size,
            shard_idx: 0,
            row_idx: 0,
            epoch: 1,
        })
    }

    fn advance_shard(&mut self) {
        self.shard_idx += 1;
        self.row_idx = 0;
        if self.shard_idx >= self.shard_order.len() {
            self.shard_idx = 0;
            self.epoch += 1;
            shuffle_shard_order(&mut self.shard_order, self.epoch as u64);
        }
        let si = self.shard_order[self.shard_idx];
        let n = self.shards[si].num_rows();
        self.row_order.resize(n, 0);
        let seed = (self.epoch as u64).wrapping_mul(6364136223846793005)
            .wrapping_add(si as u64 + 1);
        shuffle_shard_order(&mut self.row_order, seed);
    }

    /// Produce the next batch as (input_ids, targets), each Vec<u32> of length B*T.
    ///
    /// Each shard row has `shard_seq_len` tokens. We take the first SEQ tokens as
    /// inputs and the last SEQ tokens as targets (standard causal LM shift).
    /// The shard seq_len should be SEQ + 1.
    fn next_batch(&mut self) -> (Vec<u32>, Vec<u32>) {
        let b = self.batch_size;
        let t = SEQ;
        let mut input_ids = vec![0u32; b * t];
        let mut targets = vec![0u32; b * t];
        self.next_batch_into(&mut input_ids, &mut targets);
        (input_ids, targets)
    }

    /// Write the next batch directly into pre-allocated pinned buffers.
    /// Avoids Vec allocation on every micro-step.
    ///
    /// Handles shard rows shorter than SEQ+1 by concatenating multiple rows.
    fn next_batch_into(&mut self, input_ids: &mut [u32], targets: &mut [u32]) {
        let b = self.batch_size;
        let t = SEQ;
        debug_assert_eq!(input_ids.len(), b * t);
        debug_assert_eq!(targets.len(), b * t);

        let shard_sl = self.shards[self.shard_order[0]].header.seq_len as usize;
        let needed = t + 1; // SEQ + 1 tokens to get SEQ inputs + SEQ targets

        if shard_sl >= needed {
            // Fast path: shard rows are long enough
            for row_in_batch in 0..b {
                let si = self.shard_order[self.shard_idx];
                let shard = &self.shards[si];
                let row = shard.row(self.row_order[self.row_idx]);
                let off = row_in_batch * t;

                for j in 0..t {
                    input_ids[off + j] = row[j] as u32;
                }
                for j in 0..t {
                    targets[off + j] = row[j + 1] as u32;
                }

                self.row_idx += 1;
                if self.row_idx >= shard.num_rows() {
                    self.advance_shard();
                }
            }
        } else {
            // Concat path: stitch multiple short shard rows into one long sequence.
            // Each shard row is `shard_sl` tokens. We concatenate enough rows to
            // fill SEQ+1 tokens, then take [0..SEQ] as input and [1..SEQ+1] as target.
            let rows_per_seq = (needed + shard_sl - 1) / shard_sl;
            let mut concat_buf = vec![0u16; rows_per_seq * shard_sl];

            for row_in_batch in 0..b {
                // Fill concat buffer from consecutive shard rows
                for r in 0..rows_per_seq {
                    let si = self.shard_order[self.shard_idx];
                    let shard = &self.shards[si];
                    let row = shard.row(self.row_order[self.row_idx]);
                    concat_buf[r * shard_sl..(r + 1) * shard_sl].copy_from_slice(row);

                    self.row_idx += 1;
                    if self.row_idx >= shard.num_rows() {
                        self.advance_shard();
                    }
                }

                let off = row_in_batch * t;
                for j in 0..t {
                    input_ids[off + j] = concat_buf[j] as u32;
                }
                for j in 0..t {
                    targets[off + j] = concat_buf[j + 1] as u32;
                }
            }
        }
    }

    /// Number of available rows across all shards.
    fn total_rows(&self) -> usize {
        self.shards.iter().map(|s| s.num_rows()).sum()
    }
}

// ---------------------------------------------------------------------------
// Training configuration
// ---------------------------------------------------------------------------

/// Training configuration.
pub struct TrainConfig {
    pub device_batch_size: usize,
    pub total_batch_size: usize,
    pub time_budget_s: f64,
    /// Total steps before stopping. None = run indefinitely until cooldown completes
    /// after a file trigger (`touch /tmp/autoresearch_cooldown`).
    pub max_steps: Option<usize>,
    /// Absolute number of steps for the LR decay phase (WSD schedule).
    /// Replaces the fractional warmdown_ratio for long/infinite runs.
    /// Cooldown fires automatically at max_steps - cooldown_steps, or on file trigger.
    pub cooldown_steps: usize,
    pub data_dir: String,
    pub tokenizer_dir: String,
    pub checkpoint_dir: String,
    pub eval_interval: usize,
    pub checkpoint_interval: usize,
    pub load_checkpoint: Option<String>,
    pub diagnostic_steps: Option<usize>,
    /// Number of shards reserved for training. Val = remaining shards.
    /// If None, falls back to old behavior (all-but-last = train, last = val).
    pub num_train_shards: Option<usize>,
    /// Read training data from stdin (piped from feeder.py) instead of shard files.
    pub stream_input: bool,
    /// Generate synthetic random token IDs instead of reading real data.
    /// Eliminates all I/O to isolate GPU compute time. Skips eval/checkpoint.
    pub synthetic_data: bool,
    pub schedule_cfg: crate::optim::ScheduleConfig,
    pub n_pipeline_stages: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            device_batch_size: 8,
            total_batch_size: 524288,
            load_checkpoint: None,
            diagnostic_steps: None,
            time_budget_s: 86400.0,
            max_steps: Some(700),
            cooldown_steps: 350,
            data_dir: String::new(),
            tokenizer_dir: String::new(),
            checkpoint_dir: String::new(),
            eval_interval: 25,
            checkpoint_interval: 50,
            num_train_shards: None,
            stream_input: false,
            synthetic_data: false,
            schedule_cfg: crate::optim::ScheduleConfig::default(),
            n_pipeline_stages: 1,
        }
    }
}

/// Compute synthetic training progress for the WSD (Warmup-Stable-Decay) schedule.
///
/// The key insight: progress is no longer step/max_steps. Instead:
/// - Stable phase: progress = 0.0 → LR stays at peak, WD stays at max
/// - Cooldown phase: progress ramps from (1-warmdown_ratio) to 1.0 over cooldown_steps
///
/// This means the model trains at full LR indefinitely during the stable phase,
/// and the optimizer schedule only activates during the final cooldown_steps steps,
/// regardless of how long the stable phase lasted.
pub fn wsd_progress(step: usize, cooldown_start: Option<usize>, cooldown_steps: usize, warmdown_ratio: f64) -> f64 {
    match cooldown_start {
        None => 0.0,
        Some(cs) => {
            let cd_frac = ((step.saturating_sub(cs)) as f64 / cooldown_steps as f64).min(1.0);
            (1.0 - warmdown_ratio) + warmdown_ratio * cd_frac
        }
    }
}

// ---------------------------------------------------------------------------
// Main training entry point
// ---------------------------------------------------------------------------

/// Main training entry point.
pub fn train(config: TrainConfig) -> Result<()> {
    let device_peak_flops: f64 = 990.0e12; // H100 SXM5

    // 1. Initialize CUDA
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    // Init CUDA runtime AFTER driver context is set (so they share the primary context)
    ctx.bind_to_thread()?;
    unsafe { crate::ffi::cuda_runtime_init(0); }

    // 2. Create BufferManager (allocates all GPU memory)
    let mut bufs = BufferManager::new(stream.clone(), config.device_batch_size)?;
    let gemm = GemmRunner::new(stream.clone());

    let total_mb = bufs.total_bytes() as f64 / (1024.0 * 1024.0);
    println!("Allocated {total_mb:.1} MB GPU memory (B={})", config.device_batch_size);
    println!("Model: {:.1}M total params ({:.1}M active), depth={N_LAYER}, d_model={D_MODEL}", num_params() as f64 / 1e6, estimate_flops_per_token() as f64 / 6.0 / 1e6);

    // 3. Initialize weights
    //    Priority: --load-checkpoint flag > INIT_WEIGHTS_PATH env var > random init
    if let Some(ref ckpt_path) = config.load_checkpoint {
        // Load from engine checkpoint (exact buffer layout match)
        load_checkpoint(&stream, &mut bufs, ckpt_path)?;
    } else if let Ok(init_path) = std::env::var("INIT_WEIGHTS_PATH") {
        // Load from external safetensors (Python or engine format)
        crate::init_weights::load_weights_from_safetensors(&init_path, &mut bufs, &stream)?;
    } else {
        init_weights(&stream, &mut bufs)?;
    }

    // 3b. Pack wq/wk/wv into wqkv for batched QKV GEMM (all weight-load paths)
    bufs.pack_wqkv();

    // 3c. Initialize f32 master weights from bf16 (mixed-precision Muon)
    init_f32_masters(&mut bufs)?;

    // 4. Precompute RoPE cos/sin tables
    precompute_rope(&stream, &mut bufs)?;

    // 5. Calculate grad accumulation steps
    let tokens_per_micro = config.device_batch_size * SEQ;
    let grad_accum_steps = config.total_batch_size / tokens_per_micro;
    let total_batch_size = grad_accum_steps * tokens_per_micro;
    println!(
        "Batch: device={}, total={total_batch_size} ({grad_accum_steps} accum steps)",
        config.device_batch_size,
    );

    // 6. Open data source
    let mut synth_loader: Option<SyntheticDataLoader> = None;
    let mut stdin_loader: Option<StdinDataLoader> = None;
    let mut shard_loader: Option<ShardDataLoader> = None;
    if config.synthetic_data {
        synth_loader = Some(SyntheticDataLoader::new(config.device_batch_size));
        println!("Data: synthetic (LCG random tokens — no I/O, MFU baseline mode)");
    } else {
        ensure!(!config.data_dir.is_empty(), "data_dir must be set");
        let data_path = Path::new(&config.data_dir);
        if config.stream_input {
            stdin_loader = Some(StdinDataLoader::new(config.device_batch_size));
            println!("Data: streaming from stdin (feeder, {} batches prefetch)", STDIN_PREFETCH_BATCHES);
        } else {
            let loader = ShardDataLoader::new_train(data_path, config.device_batch_size, config.num_train_shards)?;
            println!(
                "Data: {} train shards, {} rows total",
                loader.shards.len(),
                loader.total_rows(),
            );
            shard_loader = Some(loader);
        }
    }

    let num_flops_per_token = estimate_flops_per_token();

    // 6b. Allocate pinned host buffers for async H2D copies.
    // Pinned (page-locked) memory lets the GPU DMA engine copy data while
    // compute kernels run, instead of blocking the stream on each memcpy.
    let bt = config.device_batch_size * SEQ;
    let mut pinned_inp = PinnedHostBuffer::new(bt)?;
    let mut pinned_tgt = PinnedHostBuffer::new(bt)?;
    println!("Allocated {:.1} KB pinned host memory (2x B*T u32)",
        2.0 * bt as f64 * 4.0 / 1024.0);

    // Helper: load next batch from whichever data source is active
    macro_rules! load_batch {
        ($inp:expr, $tgt:expr $(,)?) => {
            if let Some(ref mut sl) = synth_loader {
                sl.next_batch_into($inp, $tgt);
            } else if let Some(ref mut sl) = stdin_loader {
                sl.next_batch_into($inp, $tgt);
            } else {
                shard_loader.as_mut().unwrap().next_batch_into($inp, $tgt);
            }
        };
    }

    // 7. Warmup step (un-timed) to let cuBLAS auto-tune + estimate total steps
    println!("Running warmup step...");
    let estimated_total_steps: usize;
    {
        let warmup_start = Instant::now();
        load_batch!(pinned_inp.as_mut_slice(), pinned_tgt.as_mut_slice());
        memcpy_htod_async(&pinned_inp, &mut bufs.input_ids, &stream);
        memcpy_htod_async(&pinned_tgt, &mut bufs.targets, &stream);

        bufs.zero_gradients()?;
        crate::forward::forward(&mut bufs, &gemm);
        crate::backward::backward(&mut bufs, &gemm, grad_accum_steps);
        // (warmup — loss value not needed)
        stream.synchronize()?;

        let warmup_dt = warmup_start.elapsed().as_secs_f64();
        // Estimate: each training step does grad_accum_steps micro-steps + optimizer
        let step_time_est = warmup_dt * grad_accum_steps as f64 * 1.1; // 10% overhead for optimizer
        estimated_total_steps = config.max_steps.unwrap_or(0);
        let steps_str = if estimated_total_steps > 0 { estimated_total_steps.to_string() } else { "∞".into() };
        println!("Warmup complete ({warmup_dt:.2}s). Estimated ~{steps_str} training steps.");
    }

    // 8. Training loop
    //
    // CUDA graph capture is NOT used: Flash Attention 3 uses TMA (Tensor
    // Memory Accelerator) which requires host-side descriptor creation,
    // making it incompatible with graph capture. The kernel launch overhead
    // (~400 launches * ~5μs ≈ 2ms) is small relative to compute time.
    //
    // Throughput strategy: never sync the GPU unless we need to log, eval, or
    // checkpoint. Between those points the GPU pipeline stays full — the host
    // just enqueues micro-step data + kernel launches without waiting.

    let t_start = Instant::now();
    let mut step: usize = 0;
    let mut total_training_time: f64 = 0.0;
    let mut smooth_loss: f64 = 0.0;
    let ema_beta: f64 = 0.9;

    // NOTE: Neuron-level rinsing is disabled. Muon's Newton-Schulz orthogonalization
    // continuously redistributes gradient energy across all neurons/layers, preventing
    // neuron death entirely. RINSE_THRESHOLD experiments at 0%, 2%, 5%, 10% all showed
    // zero reinit events through step 400+ — there are no dead neurons to rinse.
    //
    // // Dynamic layer importance: EMA of (act_norm × grad_norm) per layer.
    // // Updated at log steps (when we're already synced). Scale written back to GPU.
    // let mut neuron_scores: Vec<f64> = vec![1.0; N_LAYER * MLP_DIM];
    // const LAYER_SCORE_EMA: f64 = 0.99;
    // // Val gradient norms — updated at eval steps. Between evals, holds the last
    // // measured value. Initialized to 1.0 so the generalization ratio starts neutral.
    // let mut layer_val_grad_norms_host: Vec<f32> = vec![1.0; N_LAYER];
    // // Track when each neuron was last reinitialized (0 = never).
    // // Used to apply a post-reinit gradient boost: reinitialized layers get 2× scale
    // // decaying linearly to 1× over REINIT_BOOST_STEPS steps, giving fresh weights
    // // enough signal to learn before cooldown cuts LR.
    // let mut neuron_reinit_step: Vec<usize> = vec![0; N_LAYER * MLP_DIM];
    // const REINIT_BOOST_STEPS: usize = 50;

    // Match Python: skip the first 11 steps (0..=10) for timing, so cuBLAS
    // auto-tune and initial JIT overhead don't eat into the time budget.
    const TIMING_WARMUP_STEPS: usize = 10;

    // CUDA events for per-step GPU timing (must enable timing — None defaults to DISABLE_TIMING).
    let ev_step_start = ctx.new_event(Some(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("create CUDA event");
    let ev_step_end   = ctx.new_event(Some(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("create CUDA event");
    let ev_bwd_end    = ctx.new_event(Some(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("create CUDA event");
    let ev_opt_end    = ctx.new_event(Some(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)).expect("create CUDA event");
    let mut last_dt: f64 = 0.0; // last measured step time (for non-log steps)

    // WSD schedule state: the step at which cooldown began (None = stable phase).
    // Cooldown fires when:
    //   (a) MAX_STEPS is set and step reaches max_steps - cooldown_steps
    //   (b) file trigger: `touch /tmp/autoresearch_cooldown`
    let mut cooldown_start: Option<usize> = config.max_steps.map(|max| {
        max.saturating_sub(config.cooldown_steps)
    });
    let trigger_path = std::path::Path::new("/tmp/autoresearch_cooldown");

    loop {
        // Record GPU-side start of this step (no host sync).
        ev_step_start.record(&stream).expect("record step start event");

        // File-based cooldown trigger for infinite runs:
        //   touch /tmp/autoresearch_cooldown  →  starts cooldown from current step
        if cooldown_start.is_none() && trigger_path.exists() {
            println!("[schedule] cooldown triggered via file at step {step} — will stop in {} steps", config.cooldown_steps);
            cooldown_start = Some(step);
            let _ = std::fs::remove_file(trigger_path);
        }

        // Stop when cooldown is complete.
        if let Some(cs) = cooldown_start {
            if step >= cs + config.cooldown_steps {
                break;
            }
        }

        // Zero gradients for this optimizer step
        bufs.zero_gradients()?;

        // Zero the device-side loss accumulator. forward() atomically adds
        // each token's CE loss into bufs.loss[0], so after all micro-steps
        // it holds the total loss sum across all tokens and micro-steps.
        stream.memset_zeros(&mut bufs.loss)?;

        // ── Gradient accumulation micro-steps ──
        for micro in 0..grad_accum_steps {
            load_batch!(
                pinned_inp.as_mut_slice(),
                pinned_tgt.as_mut_slice(),
            );
            if micro % 2 == 0 {
                memcpy_htod_async(&pinned_inp, &mut bufs.input_ids, &stream);
                memcpy_htod_async(&pinned_tgt, &mut bufs.targets, &stream);
                crate::forward::forward(&mut bufs, &gemm);
                crate::backward::backward(&mut bufs, &gemm, grad_accum_steps);
            } else {
                memcpy_htod_async(&pinned_inp, &mut bufs.input_ids_b, &stream);
                memcpy_htod_async(&pinned_tgt, &mut bufs.targets_b, &stream);
                // Swap B into canonical slots for forward/backward.
                std::mem::swap(&mut bufs.input_ids, &mut bufs.input_ids_b);
                std::mem::swap(&mut bufs.targets, &mut bufs.targets_b);
                crate::forward::forward(&mut bufs, &gemm);
                crate::backward::backward(&mut bufs, &gemm, grad_accum_steps);
                // Swap back so A is canonical for diagnostics/eval/optimizer.
                std::mem::swap(&mut bufs.input_ids, &mut bufs.input_ids_b);
                std::mem::swap(&mut bufs.targets, &mut bufs.targets_b);
            }
        }

        // ── Diagnostic mode: dump per-layer gradient norms ──
        if let Some(diag_steps) = config.diagnostic_steps {
            if step < diag_steps {
                stream.synchronize()?;
                let train_loss = read_mean_loss(&stream, &bufs, config.device_batch_size * SEQ * grad_accum_steps);
                print_gradient_norms(&stream, &bufs, step, train_loss)?;
            }
            if step >= diag_steps {
                println!("[diagnostic] done after {diag_steps} steps (no optimizer applied)");
                break;
            }
        }

        // Record end of fwd+bwd accumulation (before optimizer).
        ev_bwd_end.record(&stream).expect("record bwd_end");

        let progress = wsd_progress(step, cooldown_start, config.cooldown_steps, config.schedule_cfg.warmdown_ratio);
        crate::optim::optimizer_step(&mut bufs, &gemm, step + 1, progress, &config.schedule_cfg);

        // Record end of optimizer step.
        ev_opt_end.record(&stream).expect("record opt_end");

        // Record GPU-side end of this step (no host sync).
        ev_step_end.record(&stream).expect("record step end event");

        // Decide whether we need to sync this step:
        //  - logging (every LOG_INTERVAL steps, plus first few and last few)
        //  - eval / checkpoint
        let needs_eval  = !config.synthetic_data && step > 0 && step % config.eval_interval == 0;
        let needs_ckpt  = false; // only save final checkpoint
        let needs_log   = step % LOG_INTERVAL == 0
                          || step <= TIMING_WARMUP_STEPS
                          || needs_eval
                          || needs_ckpt;

        if needs_log {
            // Sync via CUDA events — waits for GPU to reach ev_step_end,
            // then reads GPU-clock elapsed time. This is the ONLY sync per
            // step, and only on log steps.
            let dt = ev_step_start.elapsed_ms(&ev_step_end)
                .expect("CUDA event elapsed") as f64 / 1000.0;
            last_dt = dt;

            // // [DISABLED] Neuron score EMA update — Muon prevents neuron death, so this never triggers reinit.
            // // Update dynamic neuron importance scores from GPU norms
            // {
            //     let neuron_act = read_f32_buf(&stream, &bufs.layer_neuron_act_norms);
            //     let grad_norms = read_f32_buf(&stream, &bufs.layer_grad_norms);
            //     for i in 0..N_LAYER {
            //         let gen_ratio = (layer_val_grad_norms_host[i] as f64
            //             / (grad_norms[i] as f64 + 1e-8)).clamp(0.1, 10.0);
            //         for j in 0..MLP_DIM {
            //             let idx = i * MLP_DIM + j;
            //             let signal = neuron_act[idx] as f64 * grad_norms[i] as f64 * gen_ratio;
            //             neuron_scores[idx] = LAYER_SCORE_EMA * neuron_scores[idx] + (1.0 - LAYER_SCORE_EMA) * signal;
            //         }
            //     }
            //     let mean_score: f64 = neuron_scores.iter().sum::<f64>() / (N_LAYER * MLP_DIM) as f64;
            //     if mean_score > 1e-12 && step >= 200 {
            //         let scale_host: Vec<f32> = (0..N_LAYER).map(|i| {
            //             let layer_mean = neuron_scores[i*MLP_DIM..(i+1)*MLP_DIM].iter().sum::<f64>() / MLP_DIM as f64;
            //             let base = (layer_mean / mean_score) as f32;
            //             let min_reinit_step = neuron_reinit_step[i*MLP_DIM..(i+1)*MLP_DIM].iter().copied().min().unwrap_or(0);
            //             let steps_since_reinit = step.saturating_sub(min_reinit_step);
            //             if min_reinit_step > 0 && steps_since_reinit < REINIT_BOOST_STEPS {
            //                 let decay = 1.0 - steps_since_reinit as f32 / REINIT_BOOST_STEPS as f32;
            //                 base * (1.0 + decay)
            //             } else { base }
            //         }).collect();
            //         stream.memcpy_htod(&scale_host, &mut bufs.layer_dynamic_scale).expect("htod layer_dynamic_scale");
            //     }
            // }

            // Read back loss (1 f32). The stream is synced so this is free.
            let train_loss = read_mean_loss(&stream, &bufs, config.device_batch_size * SEQ * grad_accum_steps);

            if step > TIMING_WARMUP_STEPS {
                total_training_time += dt;
            }

            // EMA smoothed loss
            if !train_loss.is_nan() {
                smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * train_loss;
            }
            let debiased_loss = if smooth_loss == 0.0 {
                f64::NAN
            } else {
                smooth_loss / (1.0 - ema_beta.powi((step + 1) as i32))
            };

            let tok_per_sec = total_batch_size as f64 / dt;
            let mfu = 100.0 * num_flops_per_token as f64 * total_batch_size as f64 / dt / device_peak_flops;
            let remaining = match cooldown_start {
                Some(cs) => (cs + config.cooldown_steps).saturating_sub(step),
                None => match config.max_steps {
                    Some(max) => max.saturating_sub(step),
                    None => usize::MAX,
                },
            };
            let vram_mb = query_vram_usage_mb();
            let lr_mult = crate::optim::lr_multiplier(progress, &config.schedule_cfg);

            println!(
                "step {step:>5} | loss {debiased_loss:.4} | lr {lr_mult:.3} | dt {dt:.3}s | \
                 tok/s {tok_per_sec:.0} | mfu {mfu:.1}% | vram {vram_mb:.0}MB | steps_left {remaining}",
            );

            // Per-phase GPU timing: fwd+bwd / optimizer / wall gap (pipe+launch overhead)
            if step > TIMING_WARMUP_STEPS {
                if let (Ok(bwd_ms), Ok(opt_ms), Ok(gpu_ms)) = (
                    ev_step_start.elapsed_ms(&ev_bwd_end),
                    ev_bwd_end.elapsed_ms(&ev_opt_end),
                    ev_step_start.elapsed_ms(&ev_opt_end),
                ) {
                    let wall_gap_ms = (dt * 1000.0) as f32 - gpu_ms;
                    println!(
                        "timing | fwd+bwd {bwd_ms:.1}ms  opt {opt_ms:.1}ms  gpu {gpu_ms:.1}ms  wall_gap {wall_gap_ms:.1}ms"
                    );
                }
            }
            let _ = std::io::stdout().flush();

            // ── Periodic eval ──
            if needs_eval {
                match eval_bpb(&stream, &mut bufs, &gemm, &config, PROXY_EVAL_BATCHES) {
                    Ok(bpb) => println!("[eval] step {step} | val_bpb {bpb:.4}"),
                    Err(e) => eprintln!("[eval] step {step} | error: {e}"),
                }

                // // [DISABLED] Val gradient norm capture — only used for neuron rinsing gen_ratio.
                // if let Ok(()) = capture_val_grad_norms(&stream, &mut bufs, &gemm, &config) {
                //     layer_val_grad_norms_host = read_f32_buf(&stream, &bufs.layer_val_grad_norms);
                // }
            }

            // ── Periodic checkpoint ──
            if needs_ckpt {
                save_checkpoint(&stream, &bufs, step, false, &config.checkpoint_dir)?;
            }
        } else {
            // No sync — GPU pipeline stays full. Estimate timing from last
            // measured step for the time budget.
            if step > TIMING_WARMUP_STEPS {
                total_training_time += last_dt;
            }
        }

        // // [DISABLED] Neuron rinsing — Muon prevents neuron death; this never fired across
        // // all threshold experiments (0%, 2%, 5%, 10%). Leaving code for reference.
        // if step > 0 && step % 200 == 0 && step < config.max_steps.unwrap_or(usize::MAX).saturating_sub(config.cooldown_steps) {
        //     let rinse_frac: f64 = std::env::var("RINSE_THRESHOLD")
        //         .ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
        //     let d = D_MODEL;
        //     let s = INIT_SCALE * 3.0_f64.sqrt() * (d as f64).powf(-0.5);
        //     let mut total_reinit = 0usize;
        //     for layer in 0..N_LAYER {
        //         let layer_slice = &neuron_scores[layer * MLP_DIM..(layer + 1) * MLP_DIM];
        //         let layer_mean = layer_slice.iter().sum::<f64>() / MLP_DIM as f64;
        //         if layer_mean < 1e-12 { continue; }
        //         let threshold = rinse_frac * layer_mean;
        //         let dead: Vec<usize> = (0..MLP_DIM)
        //             .filter(|&j| neuron_scores[layer * MLP_DIM + j] < threshold)
        //             .collect();
        //         if dead.is_empty() { continue; }
        //         let mut wfc_host = vec![0.0f32; MLP_DIM * d];
        //         let mut wdn_host = vec![0.0f32; d * MLP_DIM];
        //         stream.memcpy_dtoh(&bufs.layer_weights[layer].wfc_f32, &mut wfc_host).expect("dtoh wfc");
        //         stream.memcpy_dtoh(&bufs.layer_weights[layer].wdn_f32, &mut wdn_host).expect("dtoh wdn");
        //         let mut wfc_mom = vec![0.0f32; MLP_DIM * d];
        //         let mut wdn_mom = vec![0.0f32; d * MLP_DIM];
        //         let mut wfc_smom = vec![0.0f32; MLP_DIM * d];
        //         let mut wdn_smom = vec![0.0f32; d * MLP_DIM];
        //         stream.memcpy_dtoh(&bufs.muon.momentum[layer * 6 + 4], &mut wfc_mom).expect("dtoh wfc_mom");
        //         stream.memcpy_dtoh(&bufs.muon.momentum[layer * 6 + 5], &mut wdn_mom).expect("dtoh wdn_mom");
        //         stream.memcpy_dtoh(&bufs.muon.second_momentum[layer * 6 + 4], &mut wfc_smom).expect("dtoh wfc_smom");
        //         stream.memcpy_dtoh(&bufs.muon.second_momentum[layer * 6 + 5], &mut wdn_smom).expect("dtoh wdn_smom");
        //         for &j in &dead {
        //             let row = uniform_f32(d, s);
        //             wfc_host[j * d..(j + 1) * d].copy_from_slice(&row);
        //             wfc_mom[j * d..(j + 1) * d].fill(0.0);
        //             wfc_smom[j * d..(j + 1) * d].fill(0.0);
        //             for row_idx in 0..d {
        //                 wdn_host[row_idx * MLP_DIM + j] = 0.0;
        //                 wdn_mom[row_idx * MLP_DIM + j] = 0.0;
        //                 wdn_smom[row_idx * MLP_DIM + j] = 0.0;
        //             }
        //             neuron_scores[layer * MLP_DIM + j] = layer_mean;
        //             neuron_reinit_step[layer * MLP_DIM + j] = step;
        //         }
        //         stream.memcpy_htod(&wfc_host, &mut bufs.layer_weights[layer].wfc_f32).expect("htod wfc");
        //         stream.memcpy_htod(&wdn_host, &mut bufs.layer_weights[layer].wdn_f32).expect("htod wdn");
        //         stream.memcpy_htod(&wfc_mom, &mut bufs.muon.momentum[layer * 6 + 4]).expect("htod wfc_mom");
        //         stream.memcpy_htod(&wdn_mom, &mut bufs.muon.momentum[layer * 6 + 5]).expect("htod wdn_mom");
        //         stream.memcpy_htod(&wfc_smom, &mut bufs.muon.second_momentum[layer * 6 + 4]).expect("htod wfc_smom");
        //         stream.memcpy_htod(&wdn_smom, &mut bufs.muon.second_momentum[layer * 6 + 5]).expect("htod wdn_smom");
        //         let wfc_bf16: Vec<half::bf16> = wfc_host.iter().map(|&x| half::bf16::from_f32(x)).collect();
        //         let wdn_bf16: Vec<half::bf16> = wdn_host.iter().map(|&x| half::bf16::from_f32(x)).collect();
        //         stream.memcpy_htod(&wfc_bf16, &mut bufs.layer_weights[layer].wfc).expect("htod wfc bf16");
        //         stream.memcpy_htod(&wdn_bf16, &mut bufs.layer_weights[layer].wdn).expect("htod wdn bf16");
        //         total_reinit += dead.len();
        //         println!("[neuron_reinit] step {step} layer {layer}: {} neurons reinitialized", dead.len());
        //     }
        //     if total_reinit > 0 {
        //         println!("[neuron_rinse] step {step}: total {total_reinit} neurons across all layers");
        //     }
        // }

        step += 1;
    }

    // ── Final eval and checkpoint ──
    stream.synchronize()?;
    println!("Training complete after {step} steps.");

    save_checkpoint(&stream, &bufs, step, true, &config.checkpoint_dir)?;

    if !config.synthetic_data {
        match eval_bpb(&stream, &mut bufs, &gemm, &config, usize::MAX) {
            Ok(bpb) => println!("[eval] final | val_bpb {bpb:.4}"),
            Err(e) => eprintln!("[eval] final eval failed: {e}"),
        }
    }

    let total_elapsed = t_start.elapsed().as_secs_f64();
    let total_tokens = step * total_batch_size;
    let steady_mfu = if total_training_time > 0.0 {
        100.0
            * num_flops_per_token as f64
            * total_batch_size as f64
            * step.saturating_sub(TIMING_WARMUP_STEPS + 1) as f64
            / total_training_time
            / device_peak_flops
    } else {
        0.0
    };

    println!("---");
    println!("training_seconds: {total_training_time:.1}");
    println!("total_seconds:    {total_elapsed:.1}");
    println!("mfu_percent:      {steady_mfu:.2}");
    println!("total_tokens_M:   {:.1}", total_tokens as f64 / 1e6);
    println!("num_steps:        {step}");
    println!("num_params_M:     {:.1}", num_params() as f64 / 1e6);
    println!("depth:            {N_LAYER}");

    Ok(())
}

// ---------------------------------------------------------------------------
// Pipeline-parallel training
// ---------------------------------------------------------------------------

/// Pipeline-parallel training across multiple GPUs.
///
/// Splits the model across `config.n_pipeline_stages` GPUs, each holding a
/// contiguous chunk of transformer layers. Sequential pipeline (no micro-batch
/// overlap): forward flows stage 0 -> N-1, backward flows N-1 -> 0.
pub fn train_pipeline(config: TrainConfig) -> Result<()> {
    use crate::pipeline::Pipeline;

    let n_gpu = config.n_pipeline_stages;
    let device_peak_flops: f64 = n_gpu as f64 * 990.0e12;

    // 1. Create pipeline (allocates buffers on each GPU — this sets up cudarc contexts)
    let mut pipeline = Pipeline::new(n_gpu, config.device_batch_size)?;

    // Skip per-GPU kernel registration — CUDA runtime lazy-loads on each device
    // The cudaSetDevice in forward_staged/backward_staged handles device switching

    let total_mb: f64 = pipeline.stages.iter()
        .map(|s| s.bufs.total_bytes() as f64)
        .sum::<f64>() / (1024.0 * 1024.0);
    println!("Allocated {total_mb:.1} MB total GPU memory across {n_gpu} GPUs (B={})", config.device_batch_size);
    println!("Model: {:.1}M total params ({:.1}M active), depth={N_LAYER}, d_model={D_MODEL}",
        num_params() as f64 / 1e6, estimate_flops_per_token() as f64 / 6.0 / 1e6);

    // 2. Initialize weights on each stage (only loading owned layers)
    for (gpu_idx, stage) in pipeline.stages.iter_mut().enumerate() {
        stage.stream.context().bind_to_thread().expect("bind CUDA context");
        if let Some(ref ckpt_path) = config.load_checkpoint {
            load_checkpoint_staged(&stage.stream, &mut stage.bufs, ckpt_path)?;
        } else if let Ok(init_path) = std::env::var("INIT_WEIGHTS_PATH") {
            crate::init_weights::load_weights_from_safetensors(&init_path, &mut stage.bufs, &stage.stream)?;
        } else {
            init_weights_staged(&stage.stream, &mut stage.bufs)?;
        }
        stage.bufs.pack_wqkv();
        init_f32_masters(&mut stage.bufs)?;
        println!("Stage {gpu_idx}: weights initialized");
    }

    // 3. Precompute RoPE cos/sin on each stage
    for stage in pipeline.stages.iter_mut() {
        stage.stream.context().bind_to_thread().expect("bind CUDA context");
        precompute_rope(&stage.stream, &mut stage.bufs)?;
    }

    // 4. Calculate grad accumulation steps
    let tokens_per_micro = config.device_batch_size * SEQ;
    let grad_accum_steps = config.total_batch_size / tokens_per_micro;
    let total_batch_size = grad_accum_steps * tokens_per_micro;
    println!(
        "Batch: device={}, total={total_batch_size} ({grad_accum_steps} accum steps)",
        config.device_batch_size,
    );

    // 5. Open data source (stage 0 receives data)
    let mut stdin_loader: Option<StdinDataLoader> = None;
    let mut synth_loader: Option<SyntheticDataLoader> = None;
    let mut shard_loader: Option<ShardDataLoader> = None;
    if config.synthetic_data {
        synth_loader = Some(SyntheticDataLoader::new(config.device_batch_size));
        println!("Data: synthetic (MFU baseline mode)");
    } else if config.stream_input {
        stdin_loader = Some(StdinDataLoader::new(config.device_batch_size));
        println!("Data: streaming from stdin (feeder, {} batches prefetch)", STDIN_PREFETCH_BATCHES);
    } else {
        ensure!(!config.data_dir.is_empty(), "data_dir must be set");
        let data_path = Path::new(&config.data_dir);
        let loader = ShardDataLoader::new_train(data_path, config.device_batch_size, config.num_train_shards)?;
        println!("Data: {} train shards, {} rows total", loader.shards.len(), loader.total_rows());
        shard_loader = Some(loader);
    }

    let num_flops_per_token = estimate_flops_per_token();

    // 5b. Allocate pinned host buffers for async H2D (stage 0)
    let bt = config.device_batch_size * SEQ;
    let mut pinned_inp = PinnedHostBuffer::new(bt)?;
    let mut pinned_tgt = PinnedHostBuffer::new(bt)?;

    macro_rules! load_batch {
        ($inp:expr, $tgt:expr $(,)?) => {
            if let Some(ref mut sl) = synth_loader {
                sl.next_batch_into($inp, $tgt);
            } else if let Some(ref mut sl) = stdin_loader {
                sl.next_batch_into($inp, $tgt);
            } else {
                shard_loader.as_mut().unwrap().next_batch_into($inp, $tgt);
            }
        };
    }

    // 6. Warmup step (un-timed, lets cuBLAS auto-tune on each GPU)
    println!("Running warmup step...");
    {
        load_batch!(pinned_inp.as_mut_slice(), pinned_tgt.as_mut_slice());
        pipeline.load_data(pinned_inp.as_mut_slice(), pinned_tgt.as_mut_slice())?;

        for stage in pipeline.stages.iter_mut() {
            stage.bufs.zero_gradients()?;
        }

        pipeline_forward_backward(&mut pipeline, grad_accum_steps)?;
        pipeline_optimizer_step(&mut pipeline, 1, 0.0, &config.schedule_cfg)?;

        for stage in pipeline.stages.iter() {
            stage.stream.synchronize()?;
        }
        println!("Warmup complete.");
    }

    // 7. Training loop
    //
    // Each optimizer step:
    //   1. Zero gradients on all stages
    //   2. For each micro-step in grad_accum_steps:
    //      a. Load data onto stage 0 + P2P copy targets to last stage
    //      b. Forward: stage 0 -> N-1 (P2P send x, x0 between stages)
    //      c. Backward: stage N-1 -> 0 (P2P send d_x, d_x0 between stages)
    //      Gradients accumulate across micro-steps.
    //   3. Optimizer step on all stages
    //   4. Pack wqkv on all stages

    let t_start = Instant::now();
    let mut step: usize = 0;
    let mut total_training_time: f64 = 0.0;
    let mut smooth_loss: f64 = 0.0;
    let ema_beta: f64 = 0.9;

    const TIMING_WARMUP_STEPS: usize = 10;

    let mut cooldown_start: Option<usize> = config.max_steps.map(|max| {
        max.saturating_sub(config.cooldown_steps)
    });
    let trigger_path = std::path::Path::new("/tmp/autoresearch_cooldown");
    let mut last_dt: f64 = 0.0;

    loop {
        let step_start = Instant::now();

        // File-based cooldown trigger
        if cooldown_start.is_none() && trigger_path.exists() {
            println!("[schedule] cooldown triggered via file at step {step}");
            cooldown_start = Some(step);
            let _ = std::fs::remove_file(trigger_path);
        }

        // Stop when cooldown complete
        if let Some(cs) = cooldown_start {
            if step >= cs + config.cooldown_steps {
                break;
            }
        }

        // Zero gradients on all stages
        for stage in pipeline.stages.iter_mut() {
            stage.bufs.zero_gradients()?;
        }

        // Zero loss accumulator on last stage
        {
            let last = &mut pipeline.stages[n_gpu - 1];
            last.stream.memset_zeros(&mut last.bufs.loss)?;
        }

        // ── Gradient accumulation micro-steps ──
        for _micro in 0..grad_accum_steps {
            // Load data onto stage 0 + P2P targets to last stage
            load_batch!(pinned_inp.as_mut_slice(), pinned_tgt.as_mut_slice());
            pipeline.load_data(pinned_inp.as_mut_slice(), pinned_tgt.as_mut_slice())?;

            // Forward + backward across all stages (gradients accumulate)
            pipeline_forward_backward(&mut pipeline, grad_accum_steps)?;
        }

        // ── Optimizer step on all stages ──
        let progress = wsd_progress(step, cooldown_start, config.cooldown_steps, config.schedule_cfg.warmdown_ratio);
        pipeline_optimizer_step(&mut pipeline, step + 1, progress, &config.schedule_cfg)?;

        // Pack wqkv on all stages (after optimizer updates w_q/w_k/w_v)
        for stage in pipeline.stages.iter() {
            stage.bufs.pack_wqkv();
        }

        // ── Logging ──
        let needs_eval = !config.synthetic_data && step > 0 && step % config.eval_interval == 0;
        let needs_log = step % LOG_INTERVAL == 0 || step <= TIMING_WARMUP_STEPS || needs_eval;

        if needs_log {
            for stage in pipeline.stages.iter() {
                stage.stream.synchronize()?;
            }
            let dt = step_start.elapsed().as_secs_f64();
            last_dt = dt;

            // Read loss from last stage
            let train_loss = pipeline.read_loss(grad_accum_steps)? as f64;

            if step > TIMING_WARMUP_STEPS {
                total_training_time += dt;
            }

            if !train_loss.is_nan() {
                smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * train_loss;
            }
            let debiased_loss = if smooth_loss == 0.0 {
                f64::NAN
            } else {
                smooth_loss / (1.0 - ema_beta.powi((step + 1) as i32))
            };

            let tok_per_sec = total_batch_size as f64 / dt;
            let mfu = 100.0 * num_flops_per_token as f64 * total_batch_size as f64 / dt / device_peak_flops;
            let remaining = match cooldown_start {
                Some(cs) => (cs + config.cooldown_steps).saturating_sub(step),
                None => config.max_steps.map_or(usize::MAX, |max| max.saturating_sub(step)),
            };
            let lr_mult = crate::optim::lr_multiplier(progress, &config.schedule_cfg);

            println!(
                "step {step:>5} | loss {debiased_loss:.4} | lr {lr_mult:.3} | dt {dt:.3}s | \
                 tok/s {tok_per_sec:.0} | mfu {mfu:.1}% | {n_gpu}xGPU | steps_left {remaining}",
            );
            let _ = std::io::stdout().flush();

            // TODO: pipeline-parallel eval requires gathering weights or running
            // a single-GPU eval pass. Skipped for now.
        } else {
            if step > TIMING_WARMUP_STEPS {
                total_training_time += last_dt;
            }
        }

        step += 1;
    }

    // ── Final ──
    for stage in pipeline.stages.iter() {
        stage.stream.synchronize()?;
    }
    println!("Training complete after {step} steps ({n_gpu}-GPU pipeline).");

    // Save per-stage checkpoints (full checkpoint requires weight gathering)
    if !config.checkpoint_dir.is_empty() {
        for (i, stage) in pipeline.stages.iter().enumerate() {
            let dir = Path::new(&config.checkpoint_dir).join(format!("stage_{i}"));
            fs::create_dir_all(&dir)?;
            save_checkpoint(&stage.stream, &stage.bufs, step, true, dir.to_str().unwrap())?;
        }
    }

    let total_elapsed = t_start.elapsed().as_secs_f64();
    let total_tokens = step * total_batch_size;
    let steady_mfu = if total_training_time > 0.0 {
        100.0
            * num_flops_per_token as f64
            * total_batch_size as f64
            * step.saturating_sub(TIMING_WARMUP_STEPS + 1) as f64
            / total_training_time
            / device_peak_flops
    } else {
        0.0
    };

    println!("---");
    println!("training_seconds: {total_training_time:.1}");
    println!("total_seconds:    {total_elapsed:.1}");
    println!("mfu_percent:      {steady_mfu:.2}");
    println!("total_tokens_M:   {:.1}", total_tokens as f64 / 1e6);
    println!("num_steps:        {step}");
    println!("num_params_M:     {:.1}", num_params() as f64 / 1e6);
    println!("pipeline_stages:  {n_gpu}");

    Ok(())
}

/// Run forward + backward across all pipeline stages (one micro-step).
///
/// Forward: stage 0 -> N-1, with P2P transfer of x and x0 between stages.
/// Backward: stage N-1 -> 0, with P2P transfer of d_x between stages.
///
/// For d_x0 accumulation: each stage zeros its d_x0 and accumulates from its
/// own layers during backward. For the embedding backward on stage 0 to be
/// correct, it needs the SUM of d_x0 across all stages. We:
///   1. Run backward_staged_ex on all stages (stage 0 skips embedding bwd)
///   2. P2P copy each remote stage's d_x0 into stage 0's d_xn scratch
///   3. residual_add d_xn into stage 0's d_x0
///   4. Run embedding_backward on stage 0 with the complete d_x0
///
/// Gradients accumulate into each stage's grad buffers. Caller must zero grads
/// before the first micro-step and call the optimizer after the last.
fn pipeline_forward_backward(
    pipeline: &mut crate::pipeline::Pipeline,
    grad_accum_steps: usize,
) -> Result<()> {
    use cudarc::driver::DevicePtr;

    let n = pipeline.n_gpu;
    let bt = pipeline.stages[0].bufs.batch_size * SEQ;
    let n_act = bt * D_MODEL;
    let n_btd = (bt * D_MODEL) as i32;

    // =================================================================
    //  Forward: stage 0 -> 1 -> ... -> N-1
    // =================================================================
    for i in 0..n {
        {
            let stage = &mut pipeline.stages[i];
            crate::forward::forward_staged(&mut stage.bufs, &stage.gemm);
        }

        if i + 1 < n {
            pipeline.send_activation(i, i + 1, n_act)?;
            pipeline.send_x0(i, i + 1, n_act)?;
            pipeline.stages[i + 1].stream.synchronize()?;
        }
    }

    // =================================================================
    //  Backward: stage N-1 -> ... -> 0
    // =================================================================
    if n == 1 {
        // Single GPU: no d_x0 accumulation needed
        let stage = &mut pipeline.stages[0];
        crate::backward::backward_staged_ex(&mut stage.bufs, &stage.gemm, grad_accum_steps, false);
    } else {
        // Phase 1: backward on all stages (stage 0 skips embedding bwd)
        for i in (0..n).rev() {
            let skip_emb = i == 0;
            {
                let stage = &mut pipeline.stages[i];
                crate::backward::backward_staged_ex(
                    &mut stage.bufs, &stage.gemm, grad_accum_steps, skip_emb,
                );
            }

            // Transfer d_x to previous stage
            if i > 0 {
                pipeline.send_gradient(i, i - 1, n_act)?;
                pipeline.stages[i - 1].stream.synchronize()?;
            }
        }

        // Phase 2: accumulate d_x0 from remote stages into stage 0.
        // P2P copy each stage's d_x0 into stage 0's d_xn scratch, then add.
        for i in 1..n {
            // P2P: stage[i].d_x0 -> stage[0].d_xn
            {
                let src = &pipeline.stages[i];
                let dst = &pipeline.stages[0];
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

            pipeline.stages[0].stream.synchronize()?;

            // d_x0 += d_xn (accumulate remote contribution)
            {
                let s0 = &pipeline.stages[0];
                let stream = s0.stream.cu_stream() as crate::ffi::CudaStream;
                let d_x0_ptr = {
                    let (ptr, _) = s0.bufs.d_x0.device_ptr(s0.bufs.d_x0.stream());
                    ptr
                };
                let d_xn_ptr = {
                    let (ptr, _) = s0.bufs.d_xn.device_ptr(s0.bufs.d_xn.stream());
                    ptr
                };
                unsafe {
                    crate::ffi::residual_add(
                        d_x0_ptr as *mut std::ffi::c_void,
                        d_xn_ptr as *const std::ffi::c_void,
                        n_btd,
                        stream,
                    );
                }
            }
        }

        // Phase 3: embedding backward on stage 0 with complete d_x0
        crate::backward::embedding_backward(&mut pipeline.stages[0].bufs);
    }

    Ok(())
}

/// Run optimizer step on all pipeline stages.
fn pipeline_optimizer_step(
    pipeline: &mut crate::pipeline::Pipeline,
    step: usize,
    progress: f64,
    cfg: &crate::optim::ScheduleConfig,
) -> Result<()> {
    for stage in pipeline.stages.iter_mut() {
        crate::optim::optimizer_step(&mut stage.bufs, &stage.gemm, step, progress, cfg);
        stage.stream.synchronize()?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Multi-process distributed pipeline-parallel training (NCCL)
// ---------------------------------------------------------------------------

/// Multi-process pipeline-parallel training via NCCL.
///
/// Each process runs on a single GPU (CUDA_VISIBLE_DEVICES restricts visibility
/// to one device, so all code sees device 0). Env vars:
///   PIPELINE_RANK       — this process's rank (0..world_size-1)
///   PIPELINE_WORLD_SIZE — total number of processes / GPUs
///   NCCL_UNIQUE_ID_FILE — path to shared file for NCCL bootstrap
///
/// Layer assignment: layers are split evenly across ranks, with remainder going
/// to earlier ranks. Activations flow rank 0 -> N-1 (forward) and N-1 -> 0
/// (backward) via NCCL send/recv.
pub fn train_distributed(config: TrainConfig) -> Result<()> {
    use crate::nccl::NcclPipeline;

    let rank: i32 = std::env::var("PIPELINE_RANK")
        .expect("PIPELINE_RANK not set")
        .parse()
        .expect("PIPELINE_RANK must be an integer");
    let world_size: i32 = std::env::var("PIPELINE_WORLD_SIZE")
        .expect("PIPELINE_WORLD_SIZE not set")
        .parse()
        .expect("PIPELINE_WORLD_SIZE must be an integer");
    let nccl_id_file = std::env::var("NCCL_UNIQUE_ID_FILE")
        .expect("NCCL_UNIQUE_ID_FILE not set");

    ensure!(rank >= 0 && rank < world_size, "PIPELINE_RANK {rank} out of range [0, {world_size})");
    ensure!(world_size >= 2, "PIPELINE_WORLD_SIZE must be >= 2 for distributed training");
    ensure!(N_LAYER >= world_size as usize, "N_LAYER ({N_LAYER}) must be >= world_size ({world_size})");

    let ws = world_size as usize;
    let rk = rank as usize;

    // 1. Initialize CUDA (device 0 — CUDA_VISIBLE_DEVICES restricts to 1 GPU)
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    ctx.bind_to_thread()?;
    unsafe { crate::ffi::cuda_runtime_init(0); }

    // 2. Initialize NCCL communicator
    let nccl = NcclPipeline::new(rank, world_size, Path::new(&nccl_id_file));
    eprintln!("[rank {rank}/{world_size}] NCCL communicator initialized");

    // 3. Compute layer range for this rank
    let base = N_LAYER / ws;
    let remainder = N_LAYER % ws;
    let layer_start = rk * base + rk.min(remainder);
    let layer_end = layer_start + base + if rk < remainder { 1 } else { 0 };
    let has_embedding = rank == 0;
    let has_head = rank == world_size - 1;

    eprintln!("[rank {rank}] layers {layer_start}..{layer_end} (embed={has_embedding}, head={has_head})");

    // 4. Create BufferManager for this rank's layers
    let mut bufs = BufferManager::new_staged(
        stream.clone(),
        config.device_batch_size,
        layer_start,
        layer_end,
        has_embedding,
        has_head,
    )?;
    let gemm = GemmRunner::new(stream.clone());

    let total_mb = bufs.total_bytes() as f64 / (1024.0 * 1024.0);
    eprintln!("[rank {rank}] Allocated {total_mb:.1} MB GPU memory (B={})", config.device_batch_size);

    // 5. Initialize weights (only for owned layers)
    if let Some(ref ckpt_path) = config.load_checkpoint {
        load_checkpoint_staged(&stream, &mut bufs, ckpt_path)?;
    } else if let Ok(init_path) = std::env::var("INIT_WEIGHTS_PATH") {
        crate::init_weights::load_weights_from_safetensors(&init_path, &mut bufs, &stream)?;
    } else {
        init_weights_staged(&stream, &mut bufs)?;
    }
    bufs.pack_wqkv();
    init_f32_masters(&mut bufs)?;
    precompute_rope(&stream, &mut bufs)?;

    // 6. Calculate grad accumulation steps
    let tokens_per_micro = config.device_batch_size * SEQ;
    let grad_accum_steps = config.total_batch_size / tokens_per_micro;
    let total_batch_size = grad_accum_steps * tokens_per_micro;
    if rank == 0 {
        println!("Distributed pipeline: {world_size} ranks, {N_LAYER} layers");
        println!("Batch: device={}, total={total_batch_size} ({grad_accum_steps} accum steps)", config.device_batch_size);
    }

    // 7. Data source (rank 0 reads from stdin, other ranks don't load data)
    let mut stdin_loader: Option<StdinDataLoader> = None;
    let mut synth_loader: Option<SyntheticDataLoader> = None;
    let mut shard_loader: Option<ShardDataLoader> = None;
    if has_embedding {
        if config.synthetic_data {
            synth_loader = Some(SyntheticDataLoader::new(config.device_batch_size));
        } else if config.stream_input {
            stdin_loader = Some(StdinDataLoader::new(config.device_batch_size));
        } else {
            ensure!(!config.data_dir.is_empty(), "data_dir must be set");
            let data_path = Path::new(&config.data_dir);
            shard_loader = Some(ShardDataLoader::new_train(data_path, config.device_batch_size, config.num_train_shards)?);
        }
    }

    let bt = config.device_batch_size * SEQ;
    let n_act = bt * D_MODEL; // bf16 elements for x, x0 transfers
    let n_btd = (bt * D_MODEL) as i32;
    let cu_stream = stream.cu_stream() as crate::nccl::CUstream;

    // Pinned host buffers for async H2D (rank 0 only, but allocating on all is cheap)
    let mut pinned_inp = PinnedHostBuffer::new(bt)?;
    let mut pinned_tgt = PinnedHostBuffer::new(bt)?;

    macro_rules! load_batch {
        ($inp:expr, $tgt:expr $(,)?) => {
            if let Some(ref mut sl) = synth_loader {
                sl.next_batch_into($inp, $tgt);
            } else if let Some(ref mut sl) = stdin_loader {
                sl.next_batch_into($inp, $tgt);
            } else {
                shard_loader.as_mut().unwrap().next_batch_into($inp, $tgt);
            }
        };
    }

    let num_flops_per_token = estimate_flops_per_token();
    let device_peak_flops: f64 = ws as f64 * 990.0e12;

    // 8. Warmup step
    if rank == 0 { eprintln!("Running warmup step..."); }
    {
        bufs.zero_gradients()?;
        stream.memset_zeros(&mut bufs.loss)?;

        if has_embedding {
            load_batch!(pinned_inp.as_mut_slice(), pinned_tgt.as_mut_slice());
            memcpy_htod_async(&pinned_inp, &mut bufs.input_ids, &stream);
            memcpy_htod_async(&pinned_tgt, &mut bufs.targets, &stream);
        }

        // Forward
        if !has_embedding {
            // Recv x and x0 from rank-1
            nccl.group_start();
            nccl.recv_bf16(bufs.x_device_ptr_mut() as *mut std::ffi::c_void, n_act, rank - 1, cu_stream);
            nccl.recv_bf16(bufs.x0_device_ptr_mut() as *mut std::ffi::c_void, n_act, rank - 1, cu_stream);
            nccl.group_end();
        }
        crate::forward::forward_staged(&mut bufs, &gemm);
        if !has_head {
            // Send x and x0 to rank+1
            nccl.group_start();
            nccl.send_bf16(bufs.x_device_ptr() as *const std::ffi::c_void, n_act, rank + 1, cu_stream);
            nccl.send_bf16(bufs.x0_device_ptr() as *const std::ffi::c_void, n_act, rank + 1, cu_stream);
            nccl.group_end();
        }

        // Targets: rank 0 sends directly to last rank for CE loss.
        // u32 is 4 bytes, bf16 is 2 bytes, so send bt*2 bf16 elements = bt*4 bytes = bt u32 values.
        if has_embedding && !has_head {
            nccl.send_bf16(bufs.targets_device_ptr() as *const std::ffi::c_void, bt * 2, world_size - 1, cu_stream);
        }
        if has_head && !has_embedding {
            nccl.recv_bf16(bufs.targets_device_ptr() as *mut std::ffi::c_void, bt * 2, 0, cu_stream);
        }

        // Backward
        if has_head {
            // Last rank starts backward — d_x comes from CE loss, no recv needed
            crate::backward::backward_staged_ex(&mut bufs, &gemm, grad_accum_steps, false);
        } else if has_embedding {
            // Recv d_x from rank+1, then backward (skip embedding bwd for now)
            nccl.recv_bf16(bufs.dx_device_ptr_mut() as *mut std::ffi::c_void, n_act, rank + 1, cu_stream);
            crate::backward::backward_staged_ex(&mut bufs, &gemm, grad_accum_steps, true);
            crate::backward::embedding_backward(&mut bufs);
        } else {
            // Middle rank: recv d_x from rank+1, backward, send d_x to rank-1
            nccl.recv_bf16(bufs.dx_device_ptr_mut() as *mut std::ffi::c_void, n_act, rank + 1, cu_stream);
            crate::backward::backward_staged_ex(&mut bufs, &gemm, grad_accum_steps, false);
            nccl.send_bf16(bufs.dx_device_ptr() as *const std::ffi::c_void, n_act, rank - 1, cu_stream);
        }

        // Send d_x backward if last rank
        if has_head && world_size > 1 {
            nccl.send_bf16(bufs.dx_device_ptr() as *const std::ffi::c_void, n_act, rank - 1, cu_stream);
        }

        let progress = wsd_progress(0, None, config.cooldown_steps, config.schedule_cfg.warmdown_ratio);
        crate::optim::optimizer_step(&mut bufs, &gemm, 1, progress, &config.schedule_cfg);
        bufs.pack_wqkv();
        stream.synchronize()?;
    }
    if rank == 0 { eprintln!("Warmup complete."); }

    // 9. Training loop
    let t_start = Instant::now();
    let mut step: usize = 0;
    let mut total_training_time: f64 = 0.0;
    let mut smooth_loss: f64 = 0.0;
    let ema_beta: f64 = 0.9;

    const TIMING_WARMUP_STEPS: usize = 10;

    let mut cooldown_start: Option<usize> = config.max_steps.map(|max| {
        max.saturating_sub(config.cooldown_steps)
    });
    let trigger_path = std::path::Path::new("/tmp/autoresearch_cooldown");
    let mut last_dt: f64 = 0.0;

    loop {
        let step_start = Instant::now();

        // File-based cooldown trigger (rank 0 checks, but all ranks need to stop together)
        if rank == 0 {
            if cooldown_start.is_none() && trigger_path.exists() {
                eprintln!("[schedule] cooldown triggered via file at step {step}");
                cooldown_start = Some(step);
                let _ = std::fs::remove_file(trigger_path);
            }
        }

        // Stop when cooldown complete
        if let Some(cs) = cooldown_start {
            if step >= cs + config.cooldown_steps {
                break;
            }
        }

        // Zero gradients + loss
        bufs.zero_gradients()?;
        if has_head {
            stream.memset_zeros(&mut bufs.loss)?;
        }

        // ── Gradient accumulation micro-steps ──
        for _micro in 0..grad_accum_steps {
            // Load data on rank 0
            if has_embedding {
                load_batch!(pinned_inp.as_mut_slice(), pinned_tgt.as_mut_slice());
                memcpy_htod_async(&pinned_inp, &mut bufs.input_ids, &stream);
                memcpy_htod_async(&pinned_tgt, &mut bufs.targets, &stream);
            }

            // ── Forward pass ──
            // Recv activations from previous rank
            if !has_embedding {
                nccl.group_start();
                nccl.recv_bf16(bufs.x_device_ptr_mut() as *mut std::ffi::c_void, n_act, rank - 1, cu_stream);
                nccl.recv_bf16(bufs.x0_device_ptr_mut() as *mut std::ffi::c_void, n_act, rank - 1, cu_stream);
                nccl.group_end();
            }

            crate::forward::forward_staged(&mut bufs, &gemm);

            // Send activations to next rank
            if !has_head {
                nccl.group_start();
                nccl.send_bf16(bufs.x_device_ptr() as *const std::ffi::c_void, n_act, rank + 1, cu_stream);
                nccl.send_bf16(bufs.x0_device_ptr() as *const std::ffi::c_void, n_act, rank + 1, cu_stream);
                nccl.group_end();
            }

            // Send targets from rank 0 directly to last rank for CE loss
            // (For intermediate ranks in > 2 GPU setups, rank 0 sends to last rank via NCCL
            //  which handles routing transparently.)
            if has_embedding && !has_head {
                // Targets are u32 (4 bytes each). NCCL bf16 is 2 bytes each.
                // Send bt*2 bf16 elements = bt*4 bytes = bt u32 values.
                nccl.send_bf16(bufs.targets_device_ptr() as *const std::ffi::c_void, bt * 2, world_size - 1, cu_stream);
            }
            if has_head && !has_embedding {
                nccl.recv_bf16(bufs.targets_device_ptr() as *mut std::ffi::c_void, bt * 2, 0, cu_stream);
            }

            // ── Backward pass ──
            // Last rank: backward starts from CE loss (d_x is set by backward_staged_ex)
            // Middle/first rank: recv d_x from rank+1 before backward
            if !has_head {
                nccl.recv_bf16(bufs.dx_device_ptr_mut() as *mut std::ffi::c_void, n_act, rank + 1, cu_stream);
            }

            let skip_emb = has_embedding && world_size > 1;
            crate::backward::backward_staged_ex(&mut bufs, &gemm, grad_accum_steps, skip_emb);

            // Send d_x to previous rank (all ranks except rank 0)
            if !has_embedding {
                nccl.send_bf16(bufs.dx_device_ptr() as *const std::ffi::c_void, n_act, rank - 1, cu_stream);
            }

            // d_x0 accumulation: for correctness, rank 0 needs the sum of d_x0 from all ranks.
            // Each rank sends its d_x0 to rank 0, which accumulates them.
            if !has_embedding {
                nccl.send_bf16(
                    {
                        let (ptr, _) = bufs.d_x0.device_ptr(bufs.d_x0.stream());
                        ptr as *const std::ffi::c_void
                    },
                    n_act, 0, cu_stream,
                );
            }
            if has_embedding {
                // Receive and accumulate d_x0 from each remote rank
                for src in 1..world_size {
                    nccl.recv_bf16(
                        {
                            let (ptr, _) = bufs.d_xn.device_ptr(bufs.d_xn.stream());
                            ptr as *mut std::ffi::c_void
                        },
                        n_act, src, cu_stream,
                    );
                    // d_x0 += d_xn
                    unsafe {
                        crate::ffi::residual_add(
                            {
                                let (ptr, _) = bufs.d_x0.device_ptr(bufs.d_x0.stream());
                                ptr as *mut std::ffi::c_void
                            },
                            {
                                let (ptr, _) = bufs.d_xn.device_ptr(bufs.d_xn.stream());
                                ptr as *const std::ffi::c_void
                            },
                            n_btd,
                            cu_stream as crate::ffi::CudaStream,
                        );
                    }
                }
                // Now run embedding backward with complete d_x0
                crate::backward::embedding_backward(&mut bufs);
            }
        }

        // ── Optimizer step (each rank updates its own weights) ──
        let progress = wsd_progress(step, cooldown_start, config.cooldown_steps, config.schedule_cfg.warmdown_ratio);
        crate::optim::optimizer_step(&mut bufs, &gemm, step + 1, progress, &config.schedule_cfg);
        bufs.pack_wqkv();

        // ── Logging (last rank logs loss) ──
        let needs_log = step % LOG_INTERVAL == 0 || step <= TIMING_WARMUP_STEPS;

        if needs_log {
            stream.synchronize()?;
            let dt = step_start.elapsed().as_secs_f64();
            last_dt = dt;

            if step > TIMING_WARMUP_STEPS {
                total_training_time += dt;
            }

            if has_head {
                let train_loss = read_mean_loss(&stream, &bufs, config.device_batch_size * SEQ * grad_accum_steps);
                if !train_loss.is_nan() {
                    smooth_loss = ema_beta * smooth_loss + (1.0 - ema_beta) * train_loss;
                }
                let debiased_loss = if smooth_loss == 0.0 {
                    f64::NAN
                } else {
                    smooth_loss / (1.0 - ema_beta.powi((step + 1) as i32))
                };

                let tok_per_sec = total_batch_size as f64 / dt;
                let mfu = 100.0 * num_flops_per_token as f64 * total_batch_size as f64 / dt / device_peak_flops;
                let remaining = match cooldown_start {
                    Some(cs) => (cs + config.cooldown_steps).saturating_sub(step),
                    None => config.max_steps.map_or(usize::MAX, |max| max.saturating_sub(step)),
                };
                let lr_mult = crate::optim::lr_multiplier(progress, &config.schedule_cfg);

                println!(
                    "step {step:>5} | loss {debiased_loss:.4} | lr {lr_mult:.3} | dt {dt:.3}s | \
                     tok/s {tok_per_sec:.0} | mfu {mfu:.1}% | {ws}xGPU(nccl) | steps_left {remaining}",
                );
                let _ = std::io::stdout().flush();
            }
        } else {
            if step > TIMING_WARMUP_STEPS {
                total_training_time += last_dt;
            }
        }

        step += 1;
    }

    // ── Final ──
    stream.synchronize()?;

    if has_head {
        println!("Training complete after {step} steps ({ws}-GPU distributed pipeline).");
    }

    // Save per-rank checkpoint
    if !config.checkpoint_dir.is_empty() {
        let dir = Path::new(&config.checkpoint_dir).join(format!("rank_{rank}"));
        fs::create_dir_all(&dir)?;
        save_checkpoint(&stream, &bufs, step, true, dir.to_str().unwrap())?;
    }

    if rank == 0 {
        let total_elapsed = t_start.elapsed().as_secs_f64();
        let total_tokens = step * total_batch_size;
        let steady_mfu = if total_training_time > 0.0 {
            100.0
                * num_flops_per_token as f64
                * total_batch_size as f64
                * step.saturating_sub(TIMING_WARMUP_STEPS + 1) as f64
                / total_training_time
                / device_peak_flops
        } else {
            0.0
        };

        println!("---");
        println!("training_seconds: {total_training_time:.1}");
        println!("total_seconds:    {total_elapsed:.1}");
        println!("mfu_percent:      {steady_mfu:.2}");
        println!("total_tokens_M:   {:.1}", total_tokens as f64 / 1e6);
        println!("num_steps:        {step}");
        println!("num_params_M:     {:.1}", num_params() as f64 / 1e6);
        println!("pipeline_ranks:   {ws}");
    }

    Ok(())
}

/// Initialize weights for a staged buffer manager (only initializes owned layers).
fn init_weights_staged(stream: &Arc<CudaStream>, bufs: &mut BufferManager) -> Result<()> {
    let d = D_MODEL;
    let s = INIT_SCALE * 3.0_f64.sqrt() * (d as f64).powf(-0.5);

    if bufs.has_embedding {
        let wte_data = randn_bf16(VOCAB * d);
        stream.memcpy_htod(&wte_data, &mut bufs.wte)?;
    }

    if bufs.has_head {
        let lm_data: Vec<bf16> = randn_f32(VOCAB * d)
            .iter()
            .map(|&x| bf16::from_f32(x * 0.001))
            .collect();
        stream.memcpy_htod(&lm_data, &mut bufs.lm_head)?;
    }

    // Lambdas are always full-sized (small, N_LAYER entries)
    let resid_lambdas = vec![bf16::from_f32(1.0); N_LAYER];
    stream.memcpy_htod(&resid_lambdas, &mut bufs.resid_lambdas)?;

    let x0_lambdas = vec![bf16::from_f32(0.1); N_LAYER];
    stream.memcpy_htod(&x0_lambdas, &mut bufs.x0_lambdas)?;

    // Per-layer weights (only for owned layers)
    let mut moe_idx = 0usize;
    for i in bufs.layer_start..bufs.layer_end {
        let lw = &mut bufs.layer_weights[i];

        let wq_data = uniform_bf16(d * d, s);
        stream.memcpy_htod(&wq_data, &mut lw.wq)?;
        let wk_data = uniform_bf16(d * d, s);
        stream.memcpy_htod(&wk_data, &mut lw.wk)?;
        let wv_data = uniform_bf16(d * d, s);
        stream.memcpy_htod(&wv_data, &mut lw.wv)?;

        if let Some(ref mut wfc) = lw.wfc {
            let wfc_data = uniform_bf16(MLP_DIM * d, s);
            stream.memcpy_htod(&wfc_data, wfc)?;
        }

        if let Some(ref mut ve_w) = lw.ve_weight {
            let ve_data = uniform_bf16(VOCAB * d, s);
            stream.memcpy_htod(&ve_data, ve_w)?;
        }

        // MoE weights (use compact moe_idx)
        if is_moe_layer(i) && moe_idx < bufs.moe_weights.len() {
            let mw = &mut bufs.moe_weights[moe_idx];

            let router_data = vec![bf16::ZERO; N_EXPERTS * d];
            stream.memcpy_htod(&router_data, &mut mw.w_router)?;

            let wfc_data = uniform_bf16(N_EXPERTS * MLP_DIM_E * d, s);
            stream.memcpy_htod(&wfc_data, &mut mw.expert_wfc)?;

            let wdn_data = vec![bf16::ZERO; N_EXPERTS * d * MLP_DIM_E];
            stream.memcpy_htod(&wdn_data, &mut mw.expert_wdn)?;
            moe_idx += 1;
        }
    }

    println!("Weights initialized (staged, layers {}..{})", bufs.layer_start, bufs.layer_end);
    Ok(())
}

/// Load checkpoint into staged buffers (only loads owned layers).
fn load_checkpoint_staged(
    stream: &Arc<CudaStream>,
    bufs: &mut BufferManager,
    path: &str,
) -> Result<()> {
    use safetensors::SafeTensors;

    let data = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    let upload = |name: &str, buf: &mut CudaSlice<bf16>| -> Result<()> {
        let t = tensors.tensor(name)?;
        let bytes = t.data();
        let host: &[bf16] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const bf16, bytes.len() / 2)
        };
        ensure!(host.len() == buf.len(), "checkpoint tensor {name} size mismatch");
        stream.memcpy_htod(host, buf)?;
        Ok(())
    };

    if bufs.has_embedding {
        upload("wte.weight", &mut bufs.wte)?;
    }
    if bufs.has_head {
        upload("lm_head.weight", &mut bufs.lm_head)?;
    }
    upload("resid_lambdas", &mut bufs.resid_lambdas)?;
    upload("x0_lambdas", &mut bufs.x0_lambdas)?;

    for i in bufs.layer_start..bufs.layer_end {
        let lw = &mut bufs.layer_weights[i];
        let prefix = format!("h.{i}");

        upload(&format!("{prefix}.attn.c_q.weight"), &mut lw.wq)?;
        upload(&format!("{prefix}.attn.c_k.weight"), &mut lw.wk)?;
        upload(&format!("{prefix}.attn.c_v.weight"), &mut lw.wv)?;
        upload(&format!("{prefix}.attn.c_proj.weight"), &mut lw.wo)?;

        if let Some(ref mut wfc) = lw.wfc {
            upload(&format!("{prefix}.mlp.c_fc.weight"), wfc)?;
        }
        if let Some(ref mut wdn) = lw.wdn {
            upload(&format!("{prefix}.mlp.c_proj.weight"), wdn)?;
        }
        if let Some(ref mut ve_w) = lw.ve_weight {
            upload(&format!("ve.{i}.weight"), ve_w)?;
        }
        if let Some(ref mut ve_g) = lw.ve_gate {
            upload(&format!("{prefix}.attn.ve_gate.weight"), ve_g)?;
        }
    }

    // MoE weights (compact moe_idx)
    let mut moe_idx = 0usize;
    for i in bufs.layer_start..bufs.layer_end {
        if is_moe_layer(i) && moe_idx < bufs.moe_weights.len() {
            let prefix = format!("h.{i}");
            upload(&format!("{prefix}.moe.router.weight"), &mut bufs.moe_weights[moe_idx].w_router)?;

            // Expert weights are packed — load per-expert and upload
            let expert_numel = MLP_DIM_E * D_MODEL;
            for e in 0..N_EXPERTS {
                let wfc_name = format!("{prefix}.moe.expert.{e}.fc.weight");
                let wdn_name = format!("{prefix}.moe.expert.{e}.proj.weight");
                if let (Ok(wfc_t), Ok(wdn_t)) = (tensors.tensor(&wfc_name), tensors.tensor(&wdn_name)) {
                    let wfc_host: &[bf16] = unsafe {
                        std::slice::from_raw_parts(wfc_t.data().as_ptr() as *const bf16, expert_numel)
                    };
                    let wdn_host: &[bf16] = unsafe {
                        std::slice::from_raw_parts(wdn_t.data().as_ptr() as *const bf16, expert_numel)
                    };
                    let offset = e * expert_numel;
                    let nbytes = expert_numel * std::mem::size_of::<bf16>();
                    // Upload into packed buffer at the right offset
                    unsafe {
                        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                            dptr(&bufs.moe_weights[moe_idx].expert_wfc) + (offset * std::mem::size_of::<bf16>()) as u64,
                            wfc_host.as_ptr() as *const std::ffi::c_void,
                            nbytes,
                            stream.cu_stream(),
                        );
                        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                            dptr(&bufs.moe_weights[moe_idx].expert_wdn) + (offset * std::mem::size_of::<bf16>()) as u64,
                            wdn_host.as_ptr() as *const std::ffi::c_void,
                            nbytes,
                            stream.cu_stream(),
                        );
                    }
                }
            }
            moe_idx += 1;
        }
    }

    stream.synchronize()?;
    println!("[checkpoint] loaded {path} (staged, layers {}..{})", bufs.layer_start, bufs.layer_end);
    Ok(())
}

// ---------------------------------------------------------------------------
// Diagnostic: print per-layer gradient norms (for oracle comparison)
// ---------------------------------------------------------------------------

fn bf16_norm(stream: &Arc<CudaStream>, buf: &CudaSlice<bf16>) -> f32 {
    let n = buf.len();
    let mut host = vec![bf16::ZERO; n];
    stream.memcpy_dtoh(buf, &mut host).expect("dtoh");
    let sum: f64 = host.iter().map(|x| { let v = x.to_f32() as f64; v * v }).sum();
    sum.sqrt() as f32
}

fn f32_norm(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> f32 {
    let n = buf.len();
    let mut host = vec![0.0f32; n];
    stream.memcpy_dtoh(buf, &mut host).expect("dtoh");
    let sum: f64 = host.iter().map(|x| { let v = *x as f64; v * v }).sum();
    sum.sqrt() as f32
}

fn print_gradient_norms(
    stream: &Arc<CudaStream>,
    bufs: &BufferManager,
    step: usize,
    loss: f64,
) -> Result<()> {
    println!("[diag] step {step} | loss {loss:.6}");

    for i in 0..N_LAYER {
        let g = &bufs.layer_grads[i];
        println!(
            "[diag]   layer {i}: wq={:.6} wk={:.6} wv={:.6} wo={:.6} wfc={:.6} wdn={:.6}",
            bf16_norm(stream, &g.wq),
            bf16_norm(stream, &g.wk),
            bf16_norm(stream, &g.wv),
            bf16_norm(stream, &g.wo),
            g.wfc.as_ref().map_or(0.0, |b| bf16_norm(stream, b)),
            g.wdn.as_ref().map_or(0.0, |b| bf16_norm(stream, b)),
        );
    }

    println!(
        "[diag]   lm_head={:.6} wte={:.6}",
        bf16_norm(stream, &bufs.lm_head_grad),
        bf16_norm(stream, &bufs.wte_grad),
    );
    println!(
        "[diag]   resid_lambdas_grad={:.6} x0_lambdas_grad={:.6}",
        f32_norm(stream, &bufs.resid_lambdas_grad),
        f32_norm(stream, &bufs.x0_lambdas_grad),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Weight initialization (matching Python / candle reference)
// ---------------------------------------------------------------------------

/// Initialize weights with the same scheme as Python/candle.
///
/// - wte: Normal(0, 1)
/// - lm_head: Normal(0, 0.001)
/// - resid_lambdas: 1.0
/// - x0_lambdas: 0.1
/// - Wq, Wk, Wv, Wfc: Uniform(-s, s) where s = sqrt(3) * d_model^(-0.5)
/// - Wo, Wdn: zeros (residual projections start as identity)
/// - ve_weight: Uniform(-s, s)
/// - ve_gate: zeros (sigmoid(0)=0.5, *2=1.0 = neutral gate)
fn init_weights(stream: &Arc<CudaStream>, bufs: &mut BufferManager) -> Result<()> {
    let d = D_MODEL;
    let s = INIT_SCALE * 3.0_f64.sqrt() * (d as f64).powf(-0.5);

    // -- wte: Normal(0, 1) --
    let wte_data = randn_bf16(VOCAB * d);
    stream.memcpy_htod(&wte_data, &mut bufs.wte)?;

    // -- lm_head: Normal(0, 0.001) --
    let lm_data: Vec<bf16> = randn_f32(VOCAB * d)
        .iter()
        .map(|&x| bf16::from_f32(x * 0.001))
        .collect();
    stream.memcpy_htod(&lm_data, &mut bufs.lm_head)?;

    // -- resid_lambdas: 1.0 --
    let resid_lambdas = vec![bf16::from_f32(1.0); N_LAYER];
    stream.memcpy_htod(&resid_lambdas, &mut bufs.resid_lambdas)?;

    // -- x0_lambdas: 0.1 --
    let x0_lambdas = vec![bf16::from_f32(0.1); N_LAYER];
    stream.memcpy_htod(&x0_lambdas, &mut bufs.x0_lambdas)?;

    // -- Per-layer weights --
    for i in 0..N_LAYER {
        let lw = &mut bufs.layer_weights[i];

        // Wq, Wk, Wv: Uniform(-s, s)
        let wq_data = uniform_bf16(d * d, s);
        stream.memcpy_htod(&wq_data, &mut lw.wq)?;

        let wk_data = uniform_bf16(d * d, s);
        stream.memcpy_htod(&wk_data, &mut lw.wk)?;

        let wv_data = uniform_bf16(d * d, s);
        stream.memcpy_htod(&wv_data, &mut lw.wv)?;

        // Wo: zeros (already zeroed by alloc_zeros, but explicit for clarity)
        // lw.wo is already zero from BufferManager::new

        // Wfc: Uniform(-s, s) — only for dense layers
        if let Some(ref mut wfc) = lw.wfc {
            let wfc_data = uniform_bf16(MLP_DIM * d, s);
            stream.memcpy_htod(&wfc_data, wfc)?;
        }

        // Wdn: zeros (already zeroed)
        // lw.wdn is already zero from BufferManager::new

        // VE weight: Uniform(-s, s) (layers 1,3,5,7)
        if let Some(ref mut ve_w) = lw.ve_weight {
            let ve_data = uniform_bf16(VOCAB * d, s);
            stream.memcpy_htod(&ve_data, ve_w)?;
        }

        // VE gate: zeros (already zeroed)
        // lw.ve_gate is already zero from BufferManager::new

        // MoE weights
        if is_moe_layer(i) && i < bufs.moe_weights.len() {
            let mw = &mut bufs.moe_weights[i];

            // w_router: zeros (uniform routing at start)
            let router_data = vec![bf16::ZERO; N_EXPERTS * d];
            stream.memcpy_htod(&router_data, &mut mw.w_router)?;

            // expert_wfc: Uniform(-s, s)
            let wfc_data = uniform_bf16(N_EXPERTS * MLP_DIM_E * d, s);
            stream.memcpy_htod(&wfc_data, &mut mw.expert_wfc)?;

            // expert_wdn: zeros
            let wdn_data = vec![bf16::ZERO; N_EXPERTS * d * MLP_DIM_E];
            stream.memcpy_htod(&wdn_data, &mut mw.expert_wdn)?;
        }
    }

    println!("Weights initialized (matching Python scheme)");
    Ok(())
}

/// Copy bf16 weights to f32 master copies for mixed-precision optimizer.
fn init_f32_masters(bufs: &mut BufferManager) -> Result<()> {
    let stream = bufs.stream.cu_stream() as crate::ffi::CudaStream;
    // lm_head
    let n = bufs.lm_head.len() as i32;
    unsafe {
        crate::ffi::copy_bf16_to_f32(
            vptr(&bufs.lm_head), dptr(&bufs.lm_head_f32) as *mut f32, n, stream,
        );
    }

    // Block matrix weights (only owned layers for pipeline stages)
    let mut moe_idx = 0usize;
    for i in bufs.layer_start..bufs.layer_end {
        let lw = &bufs.layer_weights[i];
        let dd = lw.wq.len() as i32;
        unsafe {
            crate::ffi::copy_bf16_to_f32(vptr(&lw.wq), dptr(&lw.wq_f32) as *mut f32, dd, stream);
            crate::ffi::copy_bf16_to_f32(vptr(&lw.wk), dptr(&lw.wk_f32) as *mut f32, dd, stream);
            crate::ffi::copy_bf16_to_f32(vptr(&lw.wv), dptr(&lw.wv_f32) as *mut f32, dd, stream);
            crate::ffi::copy_bf16_to_f32(vptr(&lw.wo), dptr(&lw.wo_f32) as *mut f32, dd, stream);
        }
        if let (Some(wfc), Some(wfc_f32)) = (&lw.wfc, &lw.wfc_f32) {
            unsafe { crate::ffi::copy_bf16_to_f32(vptr(wfc), dptr(wfc_f32) as *mut f32, wfc.len() as i32, stream); }
        }
        if let (Some(wdn), Some(wdn_f32)) = (&lw.wdn, &lw.wdn_f32) {
            unsafe { crate::ffi::copy_bf16_to_f32(vptr(wdn), dptr(wdn_f32) as *mut f32, wdn.len() as i32, stream); }
        }
        if let (Some(g), Some(g32)) = (&lw.ve_gate, &lw.ve_gate_f32) {
            unsafe { crate::ffi::copy_bf16_to_f32(vptr(g), dptr(g32) as *mut f32, g.len() as i32, stream); }
        }

        // MoE weights -> f32 masters (use compact moe_idx)
        if is_moe_layer(i) {
            let mw = &bufs.moe_weights[moe_idx];
            unsafe {
                crate::ffi::copy_bf16_to_f32(vptr(&mw.w_router), dptr(&mw.w_router_f32) as *mut f32, mw.w_router.len() as i32, stream);
                crate::ffi::copy_bf16_to_f32(vptr(&mw.expert_wfc), dptr(&mw.expert_wfc_f32) as *mut f32, mw.expert_wfc.len() as i32, stream);
                crate::ffi::copy_bf16_to_f32(vptr(&mw.expert_wdn), dptr(&mw.expert_wdn_f32) as *mut f32, mw.expert_wdn.len() as i32, stream);
            }
            moe_idx += 1;
        }
    }

    Ok(())
}

/// Precompute RoPE cos/sin tables on CPU, transfer to GPU.
///
/// cos[t, i] = cos(t * theta_i)
/// sin[t, i] = sin(t * theta_i)
/// where theta_i = 1 / ROPE_BASE^(2i / HEAD_DIM) for i in 0..HEAD_DIM/2
///
/// Buffer layout: [SEQ, HEAD_DIM/2] contiguous row-major.
fn precompute_rope(stream: &Arc<CudaStream>, bufs: &mut BufferManager) -> Result<()> {
    let half = HEAD_DIM / 2;
    let base: f64 = ROPE_BASE;

    let mut cos_data = Vec::with_capacity(SEQ * half);
    let mut sin_data = Vec::with_capacity(SEQ * half);

    for t in 0..SEQ {
        for i in 0..half {
            let theta = 1.0 / base.powf(2.0 * i as f64 / HEAD_DIM as f64);
            let angle = t as f64 * theta;
            cos_data.push(bf16::from_f64(angle.cos()));
            sin_data.push(bf16::from_f64(angle.sin()));
        }
    }

    stream.memcpy_htod(&cos_data, &mut bufs.cos)?;
    stream.memcpy_htod(&sin_data, &mut bufs.sin)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Random number generation helpers (CPU-side, for weight init)
// ---------------------------------------------------------------------------

/// Global RNG state — persists across calls so each invocation gets unique values.
static RNG_STATE: std::sync::Mutex<u64> = std::sync::Mutex::new(42);

/// Advance the xorshift64 state and return the new value.
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Generate `n` f32 samples from Normal(0, 1) using simple Box-Muller.
fn randn_f32(n: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    let mut out = Vec::with_capacity(n);
    let mut state = RNG_STATE.lock().unwrap();
    let pairs = (n + 1) / 2;
    for _ in 0..pairs {
        let u1 = (xorshift64(&mut state) as f32 / u64::MAX as f32).max(f32::MIN_POSITIVE);
        let u2 = xorshift64(&mut state) as f32 / u64::MAX as f32;

        let r = (-2.0 * u1.ln()).sqrt();
        out.push(r * (2.0 * PI * u2).cos());
        out.push(r * (2.0 * PI * u2).sin());
    }
    out.truncate(n);
    out
}

/// Generate `n` bf16 samples from Normal(0, 1).
fn randn_bf16(n: usize) -> Vec<bf16> {
    randn_f32(n).iter().map(|&x| bf16::from_f32(x)).collect()
}

/// Generate `n` bf16 samples from Uniform(-bound, bound).
fn uniform_bf16(n: usize, bound: f64) -> Vec<bf16> {
    let mut out = Vec::with_capacity(n);
    let mut state = RNG_STATE.lock().unwrap();
    for _ in 0..n {
        let u = xorshift64(&mut state) as f64 / u64::MAX as f64; // [0, 1)
        let val = (2.0 * bound * u - bound) as f32;
        out.push(bf16::from_f32(val));
    }
    out
}

/// Generate `n` f32 samples from Uniform(-bound, bound).
fn uniform_f32(n: usize, bound: f64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut state = RNG_STATE.lock().unwrap();
    for _ in 0..n {
        let u = xorshift64(&mut state) as f64 / u64::MAX as f64; // [0, 1)
        out.push((2.0 * bound * u - bound) as f32);
    }
    out
}

// ---------------------------------------------------------------------------
// Evaluation (BPB = bits per byte)
// ---------------------------------------------------------------------------

/// Number of tokens to evaluate — matches Python baseline (40 * 524288).
const EVAL_TOKENS: usize = 40 * 524288;
/// Number of batches for the quick proxy eval run during training.
/// Full eval (end of run / checkpoints) uses EVAL_TOKENS / batch_tokens batches.
const PROXY_EVAL_BATCHES: usize = 4;

/// Load token_bytes table from tokenizer_dir/token_bytes.json.
///
/// Returns a Vec<u32> of length VOCAB where entry i is the number of UTF-8
/// bytes that token i decodes to. Special tokens have value 0.
fn load_token_bytes(tokenizer_dir: &str) -> Result<Vec<u32>> {
    let path = Path::new(tokenizer_dir).join("token_bytes.json");
    let text = fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
    let trimmed = text.trim();
    ensure!(
        trimmed.starts_with('[') && trimmed.ends_with(']'),
        "token_bytes.json must be a JSON array"
    );
    let inner = &trimmed[1..trimmed.len() - 1];
    let values: Vec<u32> = inner
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("failed to parse token_bytes.json: {e}"))?;
    ensure!(
        values.len() == VOCAB,
        "token_bytes.json has {} entries, expected {VOCAB}",
        values.len()
    );
    Ok(values)
}

/// Evaluate validation BPB on val shards.
///
/// Matches Python's evaluate_bpb exactly:
/// - Per-token cross-entropy losses (nats), no mean reduction
/// - Weighted by token_bytes: special tokens (byte count 0) excluded
/// - BPB = total_nats / (ln(2) * total_bytes)
fn eval_bpb(
    stream: &Arc<CudaStream>,
    bufs: &mut BufferManager,
    gemm: &GemmRunner,
    config: &TrainConfig,
    max_batches: usize,
) -> Result<f64> {
    let token_bytes = load_token_bytes(&config.tokenizer_dir)?;
    let data_path = Path::new(&config.data_dir);
    let mut val_loader = ShardDataLoader::new_val(data_path, config.device_batch_size, if config.stream_input { None } else { config.num_train_shards })?;

    let bt = config.device_batch_size * SEQ;
    let full_steps = (EVAL_TOKENS / bt).max(1);
    let steps = full_steps.min(max_batches);

    let mut total_nats: f64 = 0.0;
    let mut total_bytes: u64 = 0;

    for _ in 0..steps {
        let (inp, tgt) = val_loader.next_batch();
        stream.memcpy_htod(&inp, &mut bufs.input_ids)?;
        stream.memcpy_htod(&tgt, &mut bufs.targets)?;

        // Eval-only forward: skips all activation saves (no backward needed).
        crate::forward::forward_eval(bufs, gemm);

        // Sync and read per-token losses (f32) from the start of h_act.
        stream.synchronize()?;
        let mut per_token_losses = vec![0.0f32; bt];
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                per_token_losses.as_mut_ptr() as *mut std::ffi::c_void,
                dptr(&bufs.h_act),
                bt * std::mem::size_of::<f32>(),
            );
        }

        // Accumulate nats and bytes, excluding special tokens (byte count 0).
        for (i, &loss) in per_token_losses.iter().enumerate() {
            let target_id = tgt[i] as usize;
            let nbytes = token_bytes[target_id];
            if nbytes > 0 {
                total_nats += loss as f64;
                total_bytes += nbytes as u64;
            }
        }
    }

    ensure!(total_bytes > 0, "eval produced zero bytes — val shard may be empty or all special tokens");
    Ok(total_nats / (2.0_f64.ln() * total_bytes as f64))
}


// ---------------------------------------------------------------------------
// Checkpoint save / load (safetensors format)
// ---------------------------------------------------------------------------

/// Save all weight buffers from GPU to CPU and write as safetensors.
fn save_checkpoint(
    stream: &Arc<CudaStream>,
    bufs: &BufferManager,
    step: usize,
    is_final: bool,
    checkpoint_dir: &str,
) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;

    let dir = if checkpoint_dir.is_empty() {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".cache/autoresearch/checkpoints")
    } else {
        PathBuf::from(checkpoint_dir)
    };
    fs::create_dir_all(&dir)?;

    let name = if is_final {
        "model.safetensors".to_string()
    } else {
        format!("model_step{step}.safetensors")
    };
    let path = dir.join(&name);

    // Helper: download GPU buffer to CPU Vec<bf16>
    let download_bf16 = |buf: &CudaSlice<bf16>| -> Result<Vec<bf16>> {
        let mut host = vec![bf16::ZERO; buf.len()];
        stream.memcpy_dtoh(buf, &mut host)?;
        Ok(host)
    };

    let download_f32_from_bf16 = |buf: &CudaSlice<bf16>| -> Result<Vec<u8>> {
        let host = download_bf16(buf)?;
        // Store as raw bf16 bytes
        let bytes: Vec<u8> = host
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        Ok(bytes)
    };

    stream.synchronize()?;

    let mut tensors: HashMap<String, (Vec<u8>, Vec<usize>, Dtype)> = HashMap::new();

    // wte
    let wte_bytes = download_f32_from_bf16(&bufs.wte)?;
    tensors.insert("wte.weight".to_string(), (wte_bytes, vec![VOCAB, D_MODEL], Dtype::BF16));

    // lm_head
    let lm_bytes = download_f32_from_bf16(&bufs.lm_head)?;
    tensors.insert("lm_head.weight".to_string(), (lm_bytes, vec![VOCAB, D_MODEL], Dtype::BF16));

    // resid_lambdas
    let rl_bytes = download_f32_from_bf16(&bufs.resid_lambdas)?;
    tensors.insert("resid_lambdas".to_string(), (rl_bytes, vec![N_LAYER], Dtype::BF16));

    // x0_lambdas
    let xl_bytes = download_f32_from_bf16(&bufs.x0_lambdas)?;
    tensors.insert("x0_lambdas".to_string(), (xl_bytes, vec![N_LAYER], Dtype::BF16));

    // Per-layer weights
    for i in 0..N_LAYER {
        let lw = &bufs.layer_weights[i];
        let prefix = format!("h.{i}");

        // Attention weights (always present)
        let attn_bufs: Vec<(&str, &CudaSlice<bf16>, Vec<usize>)> = vec![
            ("attn.c_q.weight", &lw.wq, vec![D_MODEL, D_MODEL]),
            ("attn.c_k.weight", &lw.wk, vec![D_MODEL, D_MODEL]),
            ("attn.c_v.weight", &lw.wv, vec![D_MODEL, D_MODEL]),
            ("attn.c_proj.weight", &lw.wo, vec![D_MODEL, D_MODEL]),
        ];

        for (suffix, buf, shape) in attn_bufs {
            let bytes = download_f32_from_bf16(buf)?;
            tensors.insert(format!("{prefix}.{suffix}"), (bytes, shape, Dtype::BF16));
        }

        // Dense MLP weights (only for non-MoE layers)
        if let Some(ref wfc) = lw.wfc {
            let bytes = download_f32_from_bf16(wfc)?;
            tensors.insert(format!("{prefix}.mlp.c_fc.weight"), (bytes, vec![MLP_DIM, D_MODEL], Dtype::BF16));
        }
        if let Some(ref wdn) = lw.wdn {
            let bytes = download_f32_from_bf16(wdn)?;
            tensors.insert(format!("{prefix}.mlp.c_proj.weight"), (bytes, vec![D_MODEL, MLP_DIM], Dtype::BF16));
        }

        // MoE weights
        if is_moe_layer(i) && i < bufs.moe_weights.len() {
            let mw = &bufs.moe_weights[i];

            // Router
            let router_bytes = download_f32_from_bf16(&mw.w_router)?;
            tensors.insert(
                format!("{prefix}.moe.router.weight"),
                (router_bytes, vec![N_EXPERTS, D_MODEL], Dtype::BF16),
            );

            // Per-expert weights — download the packed buffer, then split
            let expert_numel = MLP_DIM_E * D_MODEL;
            let all_wfc = download_bf16(&mw.expert_wfc)?;
            let all_wdn = download_bf16(&mw.expert_wdn)?;

            for e in 0..N_EXPERTS {
                let offset = e * expert_numel;
                let wfc_slice = &all_wfc[offset..offset + expert_numel];
                let wdn_slice = &all_wdn[offset..offset + expert_numel];

                let wfc_bytes: Vec<u8> = wfc_slice.iter().flat_map(|x| x.to_le_bytes()).collect();
                let wdn_bytes: Vec<u8> = wdn_slice.iter().flat_map(|x| x.to_le_bytes()).collect();

                tensors.insert(
                    format!("{prefix}.moe.expert.{e}.fc.weight"),
                    (wfc_bytes, vec![MLP_DIM_E, D_MODEL], Dtype::BF16),
                );
                tensors.insert(
                    format!("{prefix}.moe.expert.{e}.proj.weight"),
                    (wdn_bytes, vec![D_MODEL, MLP_DIM_E], Dtype::BF16),
                );
            }
        }

        if let Some(ref ve_w) = lw.ve_weight {
            let bytes = download_f32_from_bf16(ve_w)?;
            tensors.insert(format!("ve.{i}.weight"), (bytes, vec![VOCAB, D_MODEL], Dtype::BF16));
        }

        if let Some(ref ve_g) = lw.ve_gate {
            let bytes = download_f32_from_bf16(ve_g)?;
            tensors.insert(
                format!("{prefix}.attn.ve_gate.weight"),
                (bytes, vec![N_KV_HEAD, VE_GATE_CH], Dtype::BF16),
            );
        }
    }

    // Build TensorViews and serialize
    let views: HashMap<String, TensorView<'_>> = tensors
        .iter()
        .map(|(name, (data, shape, dtype))| {
            (name.clone(), TensorView::new(*dtype, shape.clone(), data).unwrap())
        })
        .collect();

    let sorted_names: Vec<&String> = {
        let mut names: Vec<&String> = views.keys().collect();
        names.sort();
        names
    };
    let sorted_views: Vec<(String, TensorView<'_>)> = sorted_names
        .iter()
        .map(|&name| (name.clone(), views[name].clone()))
        .collect();

    safetensors::tensor::serialize_to_file(sorted_views, &None, &path)?;
    println!("[checkpoint] saved {}", path.display());

    Ok(())
}

/// Load checkpoint from safetensors into GPU buffers.
fn load_checkpoint(
    stream: &Arc<CudaStream>,
    bufs: &mut BufferManager,
    path: &str,
) -> Result<()> {
    use safetensors::SafeTensors;

    let data = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    // Helper: upload bf16 bytes to GPU
    let upload = |name: &str, buf: &mut CudaSlice<bf16>| -> Result<()> {
        let t = tensors.tensor(name)?;
        let bytes = t.data();
        // Interpret as bf16 (2 bytes per element)
        let host: &[bf16] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const bf16, bytes.len() / 2)
        };
        ensure!(
            host.len() == buf.len(),
            "checkpoint tensor {name} has {} elements, buffer has {}",
            host.len(),
            buf.len()
        );
        stream.memcpy_htod(host, buf)?;
        Ok(())
    };

    upload("wte.weight", &mut bufs.wte)?;
    upload("lm_head.weight", &mut bufs.lm_head)?;
    upload("resid_lambdas", &mut bufs.resid_lambdas)?;
    upload("x0_lambdas", &mut bufs.x0_lambdas)?;

    for i in 0..N_LAYER {
        let lw = &mut bufs.layer_weights[i];
        let prefix = format!("h.{i}");

        upload(&format!("{prefix}.attn.c_q.weight"), &mut lw.wq)?;
        upload(&format!("{prefix}.attn.c_k.weight"), &mut lw.wk)?;
        upload(&format!("{prefix}.attn.c_v.weight"), &mut lw.wv)?;
        upload(&format!("{prefix}.attn.c_proj.weight"), &mut lw.wo)?;
        if let Some(ref mut wfc) = lw.wfc {
            upload(&format!("{prefix}.mlp.c_fc.weight"), wfc)?;
        }
        if let Some(ref mut wdn) = lw.wdn {
            upload(&format!("{prefix}.mlp.c_proj.weight"), wdn)?;
        }

        if let Some(ref mut ve_w) = lw.ve_weight {
            upload(&format!("ve.{i}.weight"), ve_w)?;
        }
        if let Some(ref mut ve_g) = lw.ve_gate {
            upload(&format!("{prefix}.attn.ve_gate.weight"), ve_g)?;
        }
    }

    stream.synchronize()?;
    println!("[checkpoint] loaded {path}");
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA Graphs — capture lives in train() initialization (step 8).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Query current GPU VRAM usage via CUDA driver API.
fn query_vram_usage_mb() -> f64 {
    use cudarc::driver::result;
    match result::mem_get_info() {
        Ok((free, total)) => (total.saturating_sub(free)) as f64 / (1024.0 * 1024.0),
        Err(_) => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn lr_multiplier_at_zero() {
        assert!((get_lr_multiplier(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn lr_multiplier_mid_training() {
        assert!((get_lr_multiplier(0.3) - 1.0).abs() < EPS);
    }

    #[test]
    fn lr_multiplier_at_warmdown_start() {
        // 1.0 - WARMDOWN_RATIO = 0.9
        assert!((get_lr_multiplier(0.9) - 1.0).abs() < EPS);
    }

    #[test]
    fn lr_multiplier_mid_warmdown() {
        // progress=0.95: cooldown = (1.0 - 0.95) / 0.1 = 0.5
        // lr = 0.5 + 0.5 * 0.15 = 0.575
        assert!((get_lr_multiplier(0.95) - 0.575).abs() < EPS);
    }

    #[test]
    fn lr_multiplier_at_end() {
        assert!((get_lr_multiplier(1.0) - FINAL_LR_FRAC).abs() < EPS);
    }

    #[test]
    fn muon_momentum_at_zero() {
        assert!((get_muon_momentum(0) - 0.85).abs() < EPS);
    }

    #[test]
    fn muon_momentum_at_300() {
        assert!((get_muon_momentum(300) - 0.95).abs() < EPS);
    }

    #[test]
    fn muon_momentum_at_150() {
        assert!((get_muon_momentum(150) - 0.90).abs() < EPS);
    }

    #[test]
    fn muon_momentum_clamped() {
        assert!((get_muon_momentum(1000) - 0.95).abs() < EPS);
    }

    #[test]
    fn weight_decay_at_start() {
        assert!((get_weight_decay(0.0) - WEIGHT_DECAY).abs() < EPS);
    }

    #[test]
    fn weight_decay_at_half() {
        assert!((get_weight_decay(0.5) - 0.1).abs() < EPS);
    }

    #[test]
    fn weight_decay_at_end() {
        assert!((get_weight_decay(1.0) - 0.0).abs() < EPS);
    }

    #[test]
    fn randn_f32_correct_length() {
        let v = randn_f32(100);
        assert_eq!(v.len(), 100);
    }

    #[test]
    fn randn_f32_not_all_zero() {
        let v = randn_f32(100);
        assert!(v.iter().any(|&x| x.abs() > 0.01));
    }

    #[test]
    fn uniform_bf16_in_bounds() {
        let bound = 0.1;
        let v = uniform_bf16(1000, bound);
        for x in &v {
            let f = x.to_f32();
            assert!(f >= -bound as f32 - 0.01 && f <= bound as f32 + 0.01,
                "value {f} out of bounds");
        }
    }

    #[test]
    fn estimate_flops_sanity() {
        let flops = estimate_flops_per_token();
        // Scale with model: 6 * block_params + attn_flops.
        // For depth=14, D=896, MLP=3584: ~600M-1200M range.
        assert!(flops > 10_000_000, "flops too low: {flops}");
        assert!(flops < 5_000_000_000, "flops too high: {flops}");
    }

    #[test]
    fn num_params_sanity() {
        let p = num_params();
        // Scales with depth/dims; sanity-check it's reasonable.
        assert!(p > 1_000_000, "params too low: {p}");
        assert!(p < 3_000_000_000, "params too high: {p}");
    }

    #[test]
    fn train_config_default() {
        let c = TrainConfig::default();
        assert_eq!(c.device_batch_size, 128);
        assert_eq!(c.total_batch_size, 524288);
        assert!((c.time_budget_s - 300.0).abs() < 0.01);
    }

    #[test]
    fn precompute_rope_cos_sin_range() {
        // Verify cos/sin values are in [-1, 1]
        let half = HEAD_DIM / 2;
        let base: f64 = ROPE_BASE;
        for t in [0, 1, 100, SEQ - 1] {
            for i in [0, half / 2, half - 1] {
                let theta = 1.0 / base.powf(2.0 * i as f64 / HEAD_DIM as f64);
                let angle = t as f64 * theta;
                assert!(angle.cos().abs() <= 1.0);
                assert!(angle.sin().abs() <= 1.0);
            }
        }
    }
}

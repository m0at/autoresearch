# Rust/CUDA Training Backend

A high-performance Rust replacement for `train.py`, targeting H100 GPUs with Flash Attention 3.

## What this is

This is an alternative training backend for autoresearch that trades Python flexibility for raw throughput. It reimplements the full training loop in Rust with custom CUDA kernels, achieving ~2.3× higher MFU than the Python baseline on H100s.

The model architecture is a modified version of the baseline with several improvements validated through systematic ablations:

| Component | Baseline | This backend |
|-----------|----------|--------------|
| Implementation | Python / PyTorch | Rust / cuDNN + custom CUDA |
| Attention | FA2 | Flash Attention 3 (bf16, sm90) |
| Attention type | Full | Sliding window (SSSSL pattern) |
| Depth | 6 layers | 30 layers |
| MLP | Standard | + Value Enhancement (VE) gating |
| Position | NoPE | RoPE (base=200K) |
| val_bpb (1000 steps) | ~0.998 | **~0.862** |

## Key findings from ablations

- **Cooldown ratio**: 50% of total steps is optimal (tested 25%/50%/75%)
- **Embedding LR**: 0.9× peak LR outperforms default by −0.007 bpb
- **Layer rinsing**: Disabled — Muon's Newton-Schulz orthogonalization prevents neuron death entirely (tested 0%/2%/5%/10% thresholds, zero reinit events observed)
- **Depth**: 30 layers with 512 d_model outperforms shallower wider configs at this parameter budget (~53M params)

## Requirements

- CUDA 12.x, H100 GPU (sm90) for Flash Attention 3
- Rust nightly (edition 2024)
- Python 3.10+ for `feeder.py`

## Build

```bash
# Optional: build Flash Attention 3 first (H100 only)
bash brain/fa3/build_fa3.sh

# Build the binary
FLASH_ATTN_V3_BUILD_DIR=brain/fa3/build cargo build --release
```

## Run

```bash
# Stream training data from HuggingFace and pipe into the binary
NUM_TRAIN_SHARDS=794 python3 feeder.py --stream --prefetch 4 2>feeder.log \
  | MAX_STEPS=1000 COOLDOWN_STEPS=500 \
    ./target/release/autoresearch-brain train \
      --stream-input \
      --data-dir /path/to/val/shards \
      --seed 42 \
    > train.log 2>&1
```

## Model config (current)

```
N_LAYER=30, d_model=512, n_head=4, head_dim=128, vocab=8192, seq_len=2048
Attention window pattern: SSSSL (256,256,256,256,2048 repeating)
RoPE base=200K, softcap=15.0
Value Enhancement on every other layer, gate_ch=32
~53M parameters
Optimizer: Muon (matrix weights) + AdamW (embeddings/biases)
```

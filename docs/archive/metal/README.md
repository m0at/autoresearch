# Archived Metal/M5 Code

Archived 2026-03-08. Focus shifted exclusively to CUDA/H100 for training.
Metal is not relevant until Apple ships server-class hardware.

## Contents

- `shaders/` -- Metal shader source files (attention, fused_ops, ops)
- `flash_attn.rs` -- Metal flash attention CustomOp3 (forward + 2-kernel backward)
- `fused_ops.rs` -- Metal CustomOp implementations (RMSNorm, RoPE, QK-Norm, Softmax, Sigmoid, ReLU-squared, Residual Scale, Cross-Entropy)
- `metal_kernels.rs` -- Metal kernel loading and pipeline management
- `fused_metal.rs` -- Fused Metal optimizer kernels

## Reactivation

All call sites in `src/model/gpt.rs`, `src/train.rs`, `src/main.rs`, and
`src/optim/mod.rs` are behind `#[cfg(feature = "metal")]` guards. To restore:

1. Move `shaders/*.metal` back to `metal/`
2. Move `*.rs` files back to their original locations in `src/`
3. Build with `--features metal`

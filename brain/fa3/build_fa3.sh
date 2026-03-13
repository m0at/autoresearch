#!/bin/bash
# Build Flash Attention 3 (Hopper) for H100 (sm_90a).
# Run this ON the H100 machine. Produces libflashattention3.a in BUILD_DIR.
#
# Usage:
#   bash build_fa3.sh [BUILD_DIR]
#
# Requirements: nvcc (CUDA 12+), git, ar
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${1:-$SCRIPT_DIR/build}"

FA_REPO="https://github.com/Dao-AILab/flash-attention.git"

echo "=== FA3 Build ==="
echo "Build dir: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ── Clone flash-attention (includes CUTLASS as submodule) ────────────────────
if [ ! -d "flash-attention" ]; then
    echo "Cloning flash-attention..."
    git clone --depth=1 "$FA_REPO" flash-attention
    echo "Initializing CUTLASS submodule..."
    cd flash-attention && git submodule update --init csrc/cutlass && cd ..
else
    echo "flash-attention already cloned."
    # Ensure submodule is initialized
    if [ ! -d "flash-attention/csrc/cutlass/include" ]; then
        cd flash-attention && git submodule update --init csrc/cutlass && cd ..
    fi
fi

FA_DIR="$BUILD_DIR/flash-attention"
CUTLASS_DIR="$FA_DIR/csrc/cutlass"
HOPPER_DIR="$FA_DIR/hopper"

# ── Verify dirs exist ────────────────────────────────────────────────────────
[ -d "$HOPPER_DIR" ] || { echo "ERROR: $HOPPER_DIR not found"; exit 1; }
[ -d "$CUTLASS_DIR/include" ] || { echo "ERROR: CUTLASS include dir not found"; exit 1; }

# ── NVCC flags ───────────────────────────────────────────────────────────────
NVCC_FLAGS=(
    -std=c++17 -O3 --use_fast_math -lineinfo
    --threads 4
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
    -DCUTLASS_ENABLE_GDC_FOR_SM90
    -DCUTLASS_DEBUG_TRACE_LEVEL=0
    -DNDEBUG
    # Disable features we don't need to speed up compilation
    -DFLASHATTENTION_DISABLE_FP16
    -DFLASHATTENTION_DISABLE_FP8
    -DFLASHATTENTION_DISABLE_SPLIT
    -DFLASHATTENTION_DISABLE_PAGEDKV
    -DFLASHATTENTION_DISABLE_SOFTCAP
    -DFLASHATTENTION_DISABLE_PACKGQA
    -DFLASHATTENTION_DISABLE_HDIM64
    -DFLASHATTENTION_DISABLE_HDIM96
    -DFLASHATTENTION_DISABLE_HDIM192
    -DFLASHATTENTION_DISABLE_HDIM256
    -DFLASHATTENTION_DISABLE_SM80
    -gencode arch=compute_90a,code=sm_90a
    -I"$CUTLASS_DIR/include"
    -I"$HOPPER_DIR"
    -I"$FA_DIR/csrc/flash_attn/src"
)

# ── Source files ─────────────────────────────────────────────────────────────
# Only compile what we need: hdim128, bf16, sm90, no softcap
SRCS=(
    "$HOPPER_DIR/instantiations/flash_fwd_hdim128_bf16_sm90.cu"
    "$HOPPER_DIR/instantiations/flash_bwd_hdim128_bf16_sm90.cu"
    "$HOPPER_DIR/flash_prepare_scheduler.cu"
    "$SCRIPT_DIR/flash3_api.cu"
)

# ── Compile ──────────────────────────────────────────────────────────────────
OBJS=()
for src in "${SRCS[@]}"; do
    base="$(basename "$src" .cu)"
    obj="$BUILD_DIR/${base}.o"
    if [ "$src" -nt "$obj" ] 2>/dev/null; then
        echo "Compiling $base.cu ..."
        nvcc "${NVCC_FLAGS[@]}" -c "$src" -o "$obj"
    else
        echo "Skipping $base.cu (up to date)"
    fi
    OBJS+=("$obj")
done

# ── Link static library ─────────────────────────────────────────────────────
echo "Creating libflashattention3.a ..."
ar rcs "$BUILD_DIR/libflashattention3.a" "${OBJS[@]}"

echo ""
echo "=== Done ==="
echo "Library: $BUILD_DIR/libflashattention3.a"
echo "Size: $(du -h "$BUILD_DIR/libflashattention3.a" | cut -f1)"
echo ""
echo "To use: set FLASH_ATTN_V3_BUILD_DIR=$BUILD_DIR when building the engine."

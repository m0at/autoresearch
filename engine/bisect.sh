#!/bin/bash
set -e

# Kernel bisect: test each optimized kernel independently against baseline.
# Baseline = 6 vectorized + 4 original (val_bpb 1.1272, 665 steps, 27.6% MFU)

cd "$(dirname "$0")"
KERNELS=engine/kernels
RESULTS_DIR=bisect_results
mkdir -p $RESULTS_DIR

CANDIDATES=(cross_entropy rope rms_norm fused_norm_residual)

echo "=== KERNEL BISECT ==="
echo "Baseline reference: 665 steps, 1.14M tok/s, 27.6% MFU, val_bpb 1.1272"
echo ""

# Back up originals
for k in "${CANDIDATES[@]}"; do
    cp kernels/${k}.cu kernels/${k}.cu.orig
done

# Run each candidate
for k in "${CANDIDATES[@]}"; do
    echo "──────────────────────────────────────────────"
    echo "Testing: ${k}.cu (vectorized)"
    echo "──────────────────────────────────────────────"

    # Swap in vectorized version
    cp kernels/${k}.cu.vec kernels/${k}.cu

    # Build
    echo "Building..."
    source $HOME/.cargo/env
    FLASH_ATTN_V3_BUILD_DIR=fa3/build cargo build --release 2>&1 | tail -5

    # Run training
    echo "Running 5-min training..."
    ./target/release/autoresearch-engine 2>&1 | tee $RESULTS_DIR/${k}.log

    # Restore original
    cp kernels/${k}.cu.orig kernels/${k}.cu

    echo ""
done

# Restore originals and rebuild clean
echo "Restoring all originals and rebuilding..."
for k in "${CANDIDATES[@]}"; do
    cp kernels/${k}.cu.orig kernels/${k}.cu
done
FLASH_ATTN_V3_BUILD_DIR=fa3/build cargo build --release 2>&1 | tail -3

# Print summary
echo ""
echo "=== BISECT SUMMARY ==="
echo "kernel              | steps | tok/s    | MFU    | val_bpb | status"
echo "--------------------|-------|----------|--------|---------|-------"
echo "BASELINE (ref)      |   665 | 1.14M    | 27.6%  | 1.1272  | PASS"

for k in "${CANDIDATES[@]}"; do
    LOG=$RESULTS_DIR/${k}.log
    if [ -f "$LOG" ]; then
        FINAL_LINE=$(grep "final | val_bpb" "$LOG" 2>/dev/null || echo "")
        if [ -z "$FINAL_LINE" ]; then
            echo "${k}         |   ??? | ???      | ???    | ???     | CRASH"
            continue
        fi
        VAL_BPB=$(echo "$FINAL_LINE" | grep -oP 'val_bpb \K[0-9.]+')
        STEPS=$(grep "^num_steps:" "$LOG" | grep -oP '[0-9]+')
        MFU=$(grep "^mfu_percent:" "$LOG" | grep -oP '[0-9.]+')
        TOKENS=$(grep "^total_tokens_M:" "$LOG" | grep -oP '[0-9.]+')

        # Compute tok/s from tokens and time
        TIME=$(grep "^training_seconds:" "$LOG" | grep -oP '[0-9.]+')
        if [ -n "$TOKENS" ] && [ -n "$TIME" ]; then
            TOKS=$(echo "$TOKENS $TIME" | awk '{printf "%.2fM", $1/$2}')
        else
            TOKS="???"
        fi

        # Check gate
        STATUS="PASS"
        if [ -n "$VAL_BPB" ]; then
            REGRESS=$(echo "$VAL_BPB" | awk '{if ($1 > 1.1372) print "FAIL"; else print "PASS"}')
            STATUS=$REGRESS
        fi

        # Check for NaN
        if grep -q "NaN\|nan\|inf" "$LOG" 2>/dev/null; then
            STATUS="NaN"
        fi

        printf "%-19s | %5s | %-8s | %-6s | %-7s | %s\n" \
            "$k" "${STEPS:-???}" "${TOKS}" "${MFU:-???}%" "${VAL_BPB:-???}" "$STATUS"
    else
        echo "${k}         |   ??? | ???      | ???    | ???     | NO LOG"
    fi
done

echo ""
echo "Gate: val_bpb <= 1.1372 (baseline + 0.01), no NaN"
echo "Done."

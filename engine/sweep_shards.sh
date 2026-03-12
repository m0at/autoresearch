#!/bin/bash
set -euo pipefail

# Phase 1: Shard count sweep
# Tests how many train shards are needed for best val_bpb at 700 steps.
# shards_v2 has 251 shards total; NUM_TRAIN_SHARDS controls train/val split.

ENGINE_DIR="/root/autoresearch/engine"
BINARY="/root/autoresearch/target/release/autoresearch-engine"
DATA_DIR="/root/.cache/autoresearch/shards_v2"

shard_counts=(50 100 150 179 225)

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "================================================================"
echo " Shard Count Sweep — $(timestamp)"
echo " ${#shard_counts[@]} configs: ${shard_counts[*]}"
echo " 700 steps each"
echo "================================================================"
echo ""

# ── Build once ───────────────────────────────────────────────────────────────

export PATH="/root/.cargo/bin:$PATH"
cd "$ENGINE_DIR"

echo "[$(timestamp)] Building engine (release)..."
cargo build --release 2>&1 | tail -5
echo "[$(timestamp)] Build complete."
echo ""

# ── Run sweep ────────────────────────────────────────────────────────────────

total=${#shard_counts[@]}
declare -a results_n
declare -a results_bpb
declare -a results_time

for i in "${!shard_counts[@]}"; do
  n="${shard_counts[$i]}"
  run_num=$((i + 1))
  logfile="/root/sweep_shards_${n}.log"

  echo "────────────────────────────────────────────────────────────────"
  echo " [$run_num/$total] NUM_TRAIN_SHARDS=$n"
  echo " Log: $logfile"
  echo " Start: $(timestamp)"
  echo "────────────────────────────────────────────────────────────────"

  start_secs=$SECONDS
  NUM_TRAIN_SHARDS=$n "$BINARY" train \
    --data-dir "$DATA_DIR" \
    --max-steps 700 \
    > "$logfile" 2>&1 || true
  elapsed=$(( SECONDS - start_secs ))
  mins=$(( elapsed / 60 ))
  secs=$(( elapsed % 60 ))

  # Extract best val_bpb (look for lowest value reported)
  val_bpb=$(grep -oP 'val_bpb[= :]+\K[0-9]+\.[0-9]+' "$logfile" | sort -n | head -1 || echo "FAIL")
  final_bpb=$(grep -oP 'val_bpb[= :]+\K[0-9]+\.[0-9]+' "$logfile" | tail -1 || echo "FAIL")

  results_n+=("$n")
  results_bpb+=("$val_bpb")
  results_time+=("${mins}m${secs}s")

  echo " Done: $(timestamp) (${mins}m ${secs}s)"
  echo " Best val_bpb=$val_bpb  Final val_bpb=$final_bpb"
  echo ""
done

# ── Summary table ─────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo " SHARD SWEEP COMPLETE — $(timestamp)"
echo "================================================================"
echo ""
printf "%-20s  %-15s  %s\n" "NUM_TRAIN_SHARDS" "BEST_VAL_BPB" "TIME"
printf "%-20s  %-15s  %s\n" "----------------" "------------" "----"

# Sort by best val_bpb
tmpfile=$(mktemp)
for i in "${!results_n[@]}"; do
  printf "%s\t%s\t%s\n" "${results_bpb[$i]}" "${results_n[$i]}" "${results_time[$i]}" >> "$tmpfile"
done
sort -t$'\t' -k1 -n "$tmpfile" | while IFS=$'\t' read -r bpb n t; do
  printf "%-20s  %-15s  %s\n" "$n" "$bpb" "$t"
done
rm -f "$tmpfile"

echo ""
echo "Logs: /root/sweep_shards_*.log"

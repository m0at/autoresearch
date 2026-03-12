#!/bin/bash
set -euo pipefail

# Hyperparameter sweep for the Rust training engine on H100.
# Runs locally on the GPU machine. SCP + execute.
#
# Defaults (winning recipe from exp/H100/mar8):
#   PEAK_LR=0.04, WARMDOWN_RATIO=0.75, WEIGHT_DECAY=0.2,
#   FINAL_LR_FRAC=0.05, SCHEDULE=linear, EPS=1e-5
#   Also baked into code: depth=9, SSSSL windows (256/2048), RoPE base=200K,
#   init_scale=0.68, emb_lr=0.9, unemb_lr=0.005, momentum warmup=200 steps,
#   WD on wte=0.001, lm_head=0.01, VE=0.003, x0_lambda_init=0.05

ENGINE_DIR="/root/autoresearch/engine"
RESULTS_DIR="/root/sweep_results"
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
BINARY="$ENGINE_DIR/target/release/autoresearch-engine"

# Fixed parameters
export MAX_STEPS=700
export NUM_TRAIN_SHARDS=794
export BATCH_SIZE=128
export FLASH_ATTN_V3_BUILD_DIR=fa3/build

# ── Configurations ────────────────────────────────────────────────────────────
# Research-backed: modded-nanogpt leaderboard uses WD=1.2, WR=0.6, FINAL=0.15
configs=(
  # 1. New baseline (eps fix + data fix, defaults)
  "PEAK_LR=0.04 WEIGHT_DECAY=0.2 WARMDOWN_RATIO=0.5 FINAL_LR_FRAC=0.0 SCHEDULE=linear"
  # 2. Don't decay to zero (biggest free win)
  "PEAK_LR=0.04 WEIGHT_DECAY=0.2 WARMDOWN_RATIO=0.5 FINAL_LR_FRAC=0.15 SCHEDULE=linear"
  # 3. Higher weight decay (modded-nanogpt insight)
  "PEAK_LR=0.04 WEIGHT_DECAY=1.0 WARMDOWN_RATIO=0.5 FINAL_LR_FRAC=0.0 SCHEDULE=linear"
  # 4. Longer warmdown (modded-nanogpt uses 0.6)
  "PEAK_LR=0.04 WEIGHT_DECAY=0.2 WARMDOWN_RATIO=0.6 FINAL_LR_FRAC=0.0 SCHEDULE=linear"
  # 5. Combined: WD+WR+FINAL (the full modded-nanogpt recipe)
  "PEAK_LR=0.04 WEIGHT_DECAY=1.0 WARMDOWN_RATIO=0.6 FINAL_LR_FRAC=0.15 SCHEDULE=linear"
  # 6. Same as 5 but cosine
  "PEAK_LR=0.04 WEIGHT_DECAY=1.0 WARMDOWN_RATIO=0.6 FINAL_LR_FRAC=0.15 SCHEDULE=cosine"
  # 7. Lower LR (closer to Muon default 0.02) + full recipe
  "PEAK_LR=0.03 WEIGHT_DECAY=1.0 WARMDOWN_RATIO=0.6 FINAL_LR_FRAC=0.15 SCHEDULE=linear"
  # 8. Even lower LR + full recipe
  "PEAK_LR=0.023 WEIGHT_DECAY=1.0 WARMDOWN_RATIO=0.6 FINAL_LR_FRAC=0.15 SCHEDULE=linear"
)

# ── Helpers ──────────────────────────────────────────────────────────────────

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

config_tag() {
  local cfg="$1"
  local lr wd wr fl sched
  lr=$(echo "$cfg" | grep -oP 'PEAK_LR=\K[^ ]+')
  wd=$(echo "$cfg" | grep -oP 'WEIGHT_DECAY=\K[^ ]+')
  wr=$(echo "$cfg" | grep -oP 'WARMDOWN_RATIO=\K[^ ]+')
  fl=$(echo "$cfg" | grep -oP 'FINAL_LR_FRAC=\K[^ ]+')
  sched=$(echo "$cfg" | grep -oP 'SCHEDULE=\K[^ ]+')
  echo "lr${lr}_wd${wd}_wr${wr}_fl${fl}_${sched}"
}

extract_val_bpb() {
  grep -oP 'val_bpb[= :]+\K[0-9]+\.[0-9]+' "$1" | tail -1
}

# ── Build once ───────────────────────────────────────────────────────────────

echo "================================================================"
echo " Hyperparameter Sweep — $(timestamp)"
echo " ${#configs[@]} configurations × 700 steps each"
echo "================================================================"
echo ""

export PATH="/root/.cargo/bin:$PATH"
cd "$ENGINE_DIR"

echo "[$(timestamp)] Building engine (release)..."
cargo build --release 2>&1 | tail -5
echo "[$(timestamp)] Build complete."
echo ""

mkdir -p "$RESULTS_DIR"
> "$SUMMARY_FILE"

# ── Run sweep ────────────────────────────────────────────────────────────────

total=${#configs[@]}
declare -a tags
declare -a bpbs

for i in "${!configs[@]}"; do
  cfg="${configs[$i]}"
  tag=$(config_tag "$cfg")
  tags+=("$tag")
  run_num=$((i + 1))
  logfile="$RESULTS_DIR/sweep_${tag}.log"

  echo "────────────────────────────────────────────────────────────────"
  echo " [$run_num/$total] $tag"
  echo " Config: $cfg"
  echo " Start:  $(timestamp)"
  echo "────────────────────────────────────────────────────────────────"

  # Export this run's config
  export $cfg

  # Run
  start_secs=$SECONDS
  "$BINARY" train > "$logfile" 2>&1 || true
  elapsed=$(( SECONDS - start_secs ))
  mins=$(( elapsed / 60 ))
  secs=$(( elapsed % 60 ))

  val_bpb=$(extract_val_bpb "$logfile" || echo "FAIL")
  bpbs+=("$val_bpb")

  echo " Done:   $(timestamp)  (${mins}m ${secs}s)  val_bpb=$val_bpb"
  echo ""
  printf "%-50s  val_bpb=%s  (%dm%ds)\n" "$tag" "$val_bpb" "$mins" "$secs" >> "$SUMMARY_FILE"
done

# ── Sorted summary ───────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo " SWEEP COMPLETE — $(timestamp)"
echo "================================================================"
echo ""
printf "%-50s  %s\n" "CONFIG" "VAL_BPB"
printf "%-50s  %s\n" "------" "-------"

tmpfile=$(mktemp)
for i in "${!tags[@]}"; do
  printf "%s\t%s\n" "${bpbs[$i]}" "${tags[$i]}" >> "$tmpfile"
done
sort -t$'\t' -k1 -n "$tmpfile" | while IFS=$'\t' read -r bpb tag; do
  printf "%-50s  %s\n" "$tag" "$bpb"
done
rm -f "$tmpfile"

echo ""
echo "Logs: $RESULTS_DIR/"

#!/usr/bin/env bash
set -euo pipefail

# Step-generalization benchmark runner:
#   - Python parity model: engine/compare_train.py
#   - Rust engine:        engine/src/main.rs (MAX_STEPS)
#
# Runs both models at fixed step budgets and writes a TSV summary.
#
# Usage:
#   ./engine/run_step_benchmark.sh
#   ./engine/run_step_benchmark.sh 700 1000 1500

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
  STEPS=("$@")
else
  STEPS=(700 1000 1500)
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT/results/bench_step_generalization_${STAMP}"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"

cat > "$SUMMARY" <<'EOF'
steps	python_val_bpb	engine_val_bpb
EOF

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

require_cmd cargo
require_cmd python3
require_cmd nvidia-smi
require_cmd nvcc

echo "[bench] output dir: $OUT_DIR"
echo "[bench] steps: ${STEPS[*]}"

for S in "${STEPS[@]}"; do
  echo ""
  echo "[bench] === steps=$S ==="

  PY_LOG="$OUT_DIR/python_${S}.log"
  EN_LOG="$OUT_DIR/engine_${S}.log"

  echo "[bench] python compare_train.py --max-steps $S"
  (
    cd "$ROOT/original"
    PYTHONPATH="$ROOT/original:${PYTHONPATH:-}" python3 "$ROOT/engine/compare_train.py" --max-steps "$S"
  ) >"$PY_LOG" 2>&1
  PY_BPB="$(grep '^val_bpb:' "$PY_LOG" | tail -1 | awk '{print $2}')"
  if [[ -z "${PY_BPB:-}" ]]; then
    echo "ERROR: failed to parse python val_bpb for steps=$S (see $PY_LOG)" >&2
    exit 1
  fi

  VAL_DIR="${VAL_SHARD_DIR:-$HOME/.cache/autoresearch/shards_packed}"
  echo "[bench] rust engine MAX_STEPS=$S (stream-input from feeder.py, val=$VAL_DIR)"
  (
    python3 "$ROOT/engine/feeder.py" --stream \
      | NUM_TRAIN_SHARDS=0 MAX_STEPS="$S" \
        cargo run --release --manifest-path "$ROOT/engine/Cargo.toml" --bin autoresearch-engine \
        -- --stream-input --data-dir "$VAL_DIR"
  ) >"$EN_LOG" 2>&1 || true  # feeder exits with BrokenPipe after engine finishes — expected
  EN_BPB="$(grep -E '^\[eval\] final \| val_bpb|^val_bpb:' "$EN_LOG" | tail -1 | awk '{print $NF}')"
  if [[ -z "${EN_BPB:-}" ]]; then
    echo "ERROR: failed to parse engine val_bpb for steps=$S (see $EN_LOG)" >&2
    exit 1
  fi

  printf "%s\t%s\t%s\n" "$S" "$PY_BPB" "$EN_BPB" >> "$SUMMARY"
  echo "[bench] steps=$S python=$PY_BPB engine=$EN_BPB"
done

echo ""
echo "[bench] complete"
echo "[bench] summary: $SUMMARY"

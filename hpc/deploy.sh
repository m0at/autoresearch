#!/bin/bash
# deploy.sh — Find cheapest H100 on Vast.ai, build, train, download results, destroy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Config ──────────────────────────────────────────────────────────────────
GPU_NAME="${1:-H100_SXM}"
NUM_GPUS="${2:-1}"
SAFETY_TIMEOUT_MIN=60
NUM_SHARDS=100
DOWNLOAD_WORKERS=16
REMOTE_DIR="autoresearch"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10"

# Map GPU name → cost/hr estimate and hardware profile
case "$GPU_NAME" in
    H100_SXM)  COST_PER_HOUR=1.54; HW_PROFILE=h100 ;;
    H100*)     COST_PER_HOUR=1.67; HW_PROFILE=h100 ;;
    A100*)     COST_PER_HOUR=0.80; HW_PROFILE=a100 ;;
    *)         COST_PER_HOUR=2.00; HW_PROFILE=h100; echo "WARNING: unknown GPU $GPU_NAME, assuming h100 profile" ;;
esac

# ── Load env ────────────────────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "ERROR: $SCRIPT_DIR/.env not found. Need VASTAI API key."
    exit 1
fi
set -a; source "$SCRIPT_DIR/.env"; set +a

if [ -z "${VASTAI:-}" ]; then echo "ERROR: VASTAI API key not set in .env"; exit 1; fi

# Ensure vastai CLI is available
if ! command -v vastai &>/dev/null; then
    echo "ERROR: vastai CLI not found. Install with: pip install vastai"
    exit 1
fi

# Set API key for vastai CLI
vastai set api-key "$VASTAI" >/dev/null 2>&1

# ── Notifications ───────────────────────────────────────────────────────────
ntfy() {
    local msg="$1"
    local priority="${2:-default}"
    echo "[ntfy] $msg"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -s -d "$msg" -H "Priority: $priority" -H "Tags: robot" \
            "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    fi
}

# ── State ───────────────────────────────────────────────────────────────────
INSTANCE_ID=""
SSH_HOST=""
SSH_PORT=""
LAUNCH_TIME=""

# ── Cleanup trap ────────────────────────────────────────────────────────────
DEPLOY_SUCCESS=""

cleanup() {
    local exit_code=$?
    if [ -z "$INSTANCE_ID" ]; then
        exit $exit_code
    fi

    local elapsed=""
    local cost=""
    if [ -n "$LAUNCH_TIME" ]; then
        elapsed=$(( $(date +%s) - LAUNCH_TIME ))
        cost=$(echo "scale=2; $elapsed / 3600 * $COST_PER_HOUR" | bc)
    fi

    if [ "$DEPLOY_SUCCESS" = "1" ]; then
        echo ""
        echo "=== CLEANUP: Destroying instance $INSTANCE_ID ==="
        vastai destroy instance "$INSTANCE_ID" || true
        ntfy "Done! Destroyed. Wall: ${elapsed}s, cost: ~\$$cost" "default"
    else
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║  DEPLOY FAILED — instance LEFT RUNNING for recovery        ║"
        echo "╠══════════════════════════════════════════════════════════════╣"
        echo "║  Instance:  $INSTANCE_ID"
        echo "║  SSH:       ssh -p ${SSH_PORT:-???} root@${SSH_HOST:-???}"
        echo "║  Cost:      ~\$$COST_PER_HOUR/hr (running!)"
        [ -n "$elapsed" ] && echo "║  Elapsed:   ${elapsed}s (~\$$cost so far)"
        echo "║                                                            ║"
        echo "║  Kill: vastai destroy instance $INSTANCE_ID                ║"
        echo "╚══════════════════════════════════════════════════════════════╝"

        # Try to salvage data before giving up
        if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ]; then
            echo ""
            echo "Attempting to salvage data from instance..."
            mkdir -p "$REPO_DIR/results/checkpoints"

            rsync -az -e "ssh -p $SSH_PORT $SSH_OPTS" \
                "root@$SSH_HOST:.cache/autoresearch/tokenizer/" \
                "$REPO_DIR/results/tokenizer/" 2>/dev/null && \
                echo "  tokenizer: saved" || echo "  tokenizer: not found"

            rsync -az -e "ssh -p $SSH_PORT $SSH_OPTS" \
                "root@$SSH_HOST:.cache/autoresearch/checkpoints/" \
                "$REPO_DIR/results/checkpoints/" 2>/dev/null && \
                echo "  checkpoints: saved" || echo "  checkpoints: not found"

            for f in diagnostics_*.jsonl profile_*.jsonl; do
                scp -P "$SSH_PORT" $SSH_OPTS "root@$SSH_HOST:$REMOTE_DIR/$f" \
                    "$REPO_DIR/results/" 2>/dev/null && \
                    echo "  $f: saved" || true
            done
        fi

        cat > "$SCRIPT_DIR/.deploy_state" <<STATE
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=${SSH_HOST:-}
SSH_PORT=${SSH_PORT:-}
LAUNCH_EPOCH=${LAUNCH_TIME:-}
STATE
        echo ""
        echo "State saved to $SCRIPT_DIR/.deploy_state"
        ntfy "DEPLOY FAILED (exit $exit_code). Instance STILL RUNNING. Run: vastai destroy instance $INSTANCE_ID" "urgent"
    fi
    exit $exit_code
}
trap cleanup EXIT INT TERM

# ── Helper: remote SSH ──────────────────────────────────────────────────────
remote() {
    ssh -p "$SSH_PORT" $SSH_OPTS "root@$SSH_HOST" "$@"
}

# ── Helper: wait for SSH ────────────────────────────────────────────────────
wait_for_ssh() {
    local host="$1"
    local port="$2"
    local max_attempts=60
    local attempt=0
    echo "Waiting for SSH on $host:$port..."
    while [ $attempt -lt $max_attempts ]; do
        if ssh -p "$port" $SSH_OPTS "root@$host" "true" 2>/dev/null; then
            echo "SSH ready."
            return 0
        fi
        attempt=$((attempt + 1))
        printf "  attempt %d/%d\r" "$attempt" "$max_attempts"
        sleep 5
    done
    echo "ERROR: SSH not available after $((max_attempts * 5))s"
    return 1
}

# ── Helper: wait for instance to be running ─────────────────────────────────
wait_for_running() {
    local id="$1"
    local max_attempts=60
    local attempt=0
    echo "Waiting for instance $id to start..."
    while [ $attempt -lt $max_attempts ]; do
        local status
        status=$(vastai show instance "$id" --raw 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(d.get('actual_status','') or d.get('status_msg',''))" 2>/dev/null || echo "")
        if [ "$status" = "running" ]; then
            echo "Instance running."
            return 0
        fi
        attempt=$((attempt + 1))
        printf "  attempt %d/%d (status: %s)\r" "$attempt" "$max_attempts" "$status"
        sleep 5
    done
    echo "ERROR: Instance never reached running state after $((max_attempts * 5))s"
    return 1
}

# ── Pre-flight: cost estimate ───────────────────────────────────────────────
MAX_COST=$(echo "scale=2; $SAFETY_TIMEOUT_MIN / 60 * $COST_PER_HOUR" | bc)
EST_COST=$(echo "scale=2; 15 / 60 * $COST_PER_HOUR" | bc)
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  autoresearch — Vast.ai GPU deployment                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  GPU:       $GPU_NAME (×$NUM_GPUS)"
echo "║  Profile:   $HW_PROFILE"
echo "║  Cost:      ~\$$COST_PER_HOUR/hr"
echo "║  Estimate:  ~15 min total → ~\$$EST_COST"
echo "║  Safety:    auto-terminate after ${SAFETY_TIMEOUT_MIN} min → \$$MAX_COST max"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
# Skip confirmation if --yes flag or non-interactive
if [[ "${3:-}" == "--yes" ]] || [[ ! -t 0 ]]; then
    echo "Auto-confirmed (non-interactive mode)"
else
    read -p "Proceed? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        INSTANCE_ID=""
        echo "Aborted."
        exit 0
    fi
fi

# ── 1. Find cheapest offer ─────────────────────────────────────────────────
echo ""
echo "=== Searching for cheapest ${GPU_NAME} (×${NUM_GPUS}) ==="
OFFER_ID=$(vastai search offers "gpu_name=${GPU_NAME} num_gpus=${NUM_GPUS} reliability>0.95 inet_down>500" \
    -o 'dph_total' --limit 1 --raw 2>/dev/null | python3 -c "
import json,sys
offers=json.load(sys.stdin)
if not offers:
    print('')
else:
    o=offers[0]
    print(o['id'])
    import sys as s
    s.stderr.write(f\"  Best offer: id={o['id']}, \${o.get('dph_total',0):.2f}/hr, {o.get('gpu_name','?')}, {o.get('geolocation','?')}\n\")
" 2>&1)

# Parse: first line is ID, rest is info
OFFER_INFO=$(echo "$OFFER_ID" | tail -n +2)
OFFER_ID=$(echo "$OFFER_ID" | head -1)

if [ -z "$OFFER_ID" ]; then
    echo "ERROR: No ${GPU_NAME} offers available."
    ntfy "${GPU_NAME} not available on Vast.ai" "high"
    exit 1
fi
echo "$OFFER_INFO"

# ── 2. Create instance ─────────────────────────────────────────────────────
echo ""
echo "=== Creating instance ==="
CREATE_RESPONSE=$(vastai create instance "$OFFER_ID" \
    --image nvidia/cuda:12.6.3-devel-ubuntu22.04 \
    --disk 100 --ssh --direct --raw 2>/dev/null || echo "{}")

INSTANCE_ID=$(echo "$CREATE_RESPONSE" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(d.get('new_contract',''))" 2>/dev/null || echo "")

if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: Failed to create instance."
    echo "$CREATE_RESPONSE"
    INSTANCE_ID=""
    exit 1
fi
echo "Instance ID: $INSTANCE_ID"
LAUNCH_TIME=$(date +%s)
ntfy "Created Vast.ai instance $INSTANCE_ID (${GPU_NAME})"

# ── 3. Wait for running + get SSH details ───────────────────────────────────
wait_for_running "$INSTANCE_ID"

echo "Getting SSH connection details..."
SSH_CMD=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || echo "")
if [ -z "$SSH_CMD" ]; then
    # Fallback: parse from instance info
    INSTANCE_JSON=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null)
    SSH_HOST=$(echo "$INSTANCE_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh_host',''))" 2>/dev/null)
    SSH_PORT=$(echo "$INSTANCE_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh_port',''))" 2>/dev/null)
else
    # Parse ssh -p PORT root@HOST from ssh-url output
    SSH_PORT=$(echo "$SSH_CMD" | grep -oE '\-p [0-9]+' | awk '{print $2}')
    SSH_HOST=$(echo "$SSH_CMD" | grep -oE 'root@[^ ]+' | sed 's/root@//')
fi

if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "ERROR: Could not determine SSH connection details."
    echo "SSH URL output: $SSH_CMD"
    exit 1
fi
echo "SSH: ssh -p $SSH_PORT root@$SSH_HOST"

wait_for_ssh "$SSH_HOST" "$SSH_PORT"

# ── 4. Safety timeout (background watchdog) ─────────────────────────────────
(
    sleep $((SAFETY_TIMEOUT_MIN * 60))
    echo ""
    echo "=== REMINDER: ${SAFETY_TIMEOUT_MIN} min elapsed (~\$$COST_PER_HOUR/hr). Instance still running. ==="
    ntfy "REMINDER: ${SAFETY_TIMEOUT_MIN} min elapsed, instance still running (~\$$COST_PER_HOUR/hr)" "high"
) &
WATCHDOG_PID=$!

# ── 5. Rsync repo to remote ────────────────────────────────────────────────
echo ""
echo "=== Syncing repo ==="
rsync -az --delete \
    --exclude target/ \
    --exclude .git/ \
    --exclude results/ \
    --exclude 'hpc/.env' \
    -e "ssh -p $SSH_PORT $SSH_OPTS" \
    "$REPO_DIR/" "root@$SSH_HOST:$REMOTE_DIR/"
echo "Sync complete."

# ── 6. Install Rust + build ─────────────────────────────────────────────────
echo ""
echo "=== Setting up remote environment ==="
remote bash -l <<'SETUP'
set -euo pipefail

# System deps
apt-get update -qq
apt-get install -y -qq pkg-config libssl-dev > /dev/null 2>&1

# Rust (skip if already present)
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

cd autoresearch
export FLASH_ATTN_BUILD_DIR="$HOME/.cache/flash-attn-build"
mkdir -p "$FLASH_ATTN_BUILD_DIR"

HAD_PREBUILT=0
if [ -f prebuilt/libflashattention.a ]; then
    cp prebuilt/libflashattention.a "$FLASH_ATTN_BUILD_DIR/libflashattention.a"
    echo "Copied prebuilt flash-attn .a → instant CUDA build"
    HAD_PREBUILT=1
fi

echo "Building with CUDA support..."
cargo build --release --features cuda --no-default-features 2>&1 | tail -10
echo "Build complete."

if [ "$HAD_PREBUILT" -eq 0 ]; then
    mkdir -p prebuilt
    cp "$FLASH_ATTN_BUILD_DIR/libflashattention.a" prebuilt/libflashattention.a 2>/dev/null || true
    echo "Saved flash-attn .a for future deploys"
fi
SETUP
ntfy "Build complete, starting data prep"

# ── 7. Prepare data ────────────────────────────────────────────────────────
echo ""
echo "=== Preparing data ==="
PREP_START=$(date +%s)
remote bash -l <<PREPARE
set -euo pipefail
source "\$HOME/.cargo/env"
cd autoresearch
cargo run --release --features cuda --no-default-features -- prepare \
    --num-shards $NUM_SHARDS --download-workers $DOWNLOAD_WORKERS
PREPARE
PREP_ELAPSED=$(( $(date +%s) - PREP_START ))
echo "Data prep took ${PREP_ELAPSED}s"
ntfy "Data prep done (${PREP_ELAPSED}s), starting training"

# ── 8. Train ────────────────────────────────────────────────────────────────
echo ""
echo "=== Training ==="
TRAIN_START=$(date +%s)

mkdir -p "$REPO_DIR/results"

remote bash -l <<TRAIN | tee "$REPO_DIR/results/train.log"
set -euo pipefail
source "\$HOME/.cargo/env"
cd autoresearch
cargo run --release --features cuda --no-default-features -- train --profile $HW_PROFILE --diagnostics
TRAIN
TRAIN_ELAPSED=$(( $(date +%s) - TRAIN_START ))
echo ""
echo "Training took ${TRAIN_ELAPSED}s"
ntfy "Training complete (${TRAIN_ELAPSED}s), downloading results" "high"

# ── 9. Download results ────────────────────────────────────────────────────
echo ""
echo "=== Downloading results ==="
mkdir -p "$REPO_DIR/results"

rsync -avz \
    -e "ssh -p $SSH_PORT $SSH_OPTS" \
    "root@$SSH_HOST:.cache/autoresearch/checkpoints/" "$REPO_DIR/results/checkpoints/" 2>/dev/null || echo "  no checkpoints found"

for f in diagnostics_${HW_PROFILE}.jsonl profile_${HW_PROFILE}.jsonl; do
    scp -P "$SSH_PORT" $SSH_OPTS "root@$SSH_HOST:$REMOTE_DIR/$f" "$REPO_DIR/results/" 2>/dev/null && \
        echo "  $f downloaded" || echo "  $f not found (ok)"
done

echo "Downloading tokenizer..."
rsync -avz \
    -e "ssh -p $SSH_PORT $SSH_OPTS" \
    "root@$SSH_HOST:.cache/autoresearch/tokenizer/" "$REPO_DIR/results/tokenizer/" 2>/dev/null || echo "  no tokenizer found"

echo "Downloading binary shards..."
rsync -avz \
    -e "ssh -p $SSH_PORT $SSH_OPTS" \
    "root@$SSH_HOST:.cache/autoresearch/shards/" "$REPO_DIR/results/shards/" 2>/dev/null || echo "  no shards found"

echo "Results downloaded to $REPO_DIR/results/"

# ── 10. Kill watchdog ───────────────────────────────────────────────────────
kill $WATCHDOG_PID 2>/dev/null || true

# ── 11. Cost report ─────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - LAUNCH_TIME ))
TOTAL_HOURS=$(echo "scale=4; $TOTAL_ELAPSED / 3600" | bc)
TOTAL_COST=$(echo "scale=2; $TOTAL_HOURS * $COST_PER_HOUR" | bc)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  COMPLETE                                                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Total wall time:  ${TOTAL_ELAPSED}s"
echo "║  Data prep:        ${PREP_ELAPSED}s"
echo "║  Training:         ${TRAIN_ELAPSED}s"
echo "║  Estimated cost:   \$$TOTAL_COST"
echo "║  Results:          $REPO_DIR/results/"
echo "╚══════════════════════════════════════════════════════════════╝"

ntfy "DONE! Wall: ${TOTAL_ELAPSED}s, cost: ~\$$TOTAL_COST" "high"

# Mark success so cleanup trap destroys the instance
DEPLOY_SUCCESS=1

#!/bin/bash
# monitor.sh — Monitor autoresearch training on a Vast.ai GPU instance.
# Usage: ./monitor.sh <instance_id>
#    or: ./monitor.sh <ssh_host> <ssh_port>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NTFY_TOPIC="${NTFY_TOPIC:-autoresearch}"
COST_PER_HOUR=1.54
SAFETY_TIMEOUT=1800  # 30 min
POLL_INTERVAL=10
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o ServerAliveInterval=15 -o LogLevel=ERROR"

# Load .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi
if [ -n "${VASTAI:-}" ]; then
    vastai set api-key "$VASTAI" >/dev/null 2>&1 || true
fi

# --- resolve SSH connection ---
if [ $# -ge 2 ]; then
    # Direct: host port
    SSH_HOST="$1"
    SSH_PORT="$2"
elif [ $# -eq 1 ]; then
    # Instance ID — get SSH details from vastai
    INSTANCE_ID="$1"
    echo "Looking up SSH details for instance $INSTANCE_ID..."
    SSH_CMD=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || echo "")
    if [ -n "$SSH_CMD" ]; then
        SSH_PORT=$(echo "$SSH_CMD" | grep -oE '\-p [0-9]+' | awk '{print $2}')
        SSH_HOST=$(echo "$SSH_CMD" | grep -oE 'root@[^ ]+' | sed 's/root@//')
    fi
    if [ -z "${SSH_HOST:-}" ] || [ -z "${SSH_PORT:-}" ]; then
        INSTANCE_JSON=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null)
        SSH_HOST=$(echo "$INSTANCE_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh_host',''))" 2>/dev/null)
        SSH_PORT=$(echo "$INSTANCE_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh_port',''))" 2>/dev/null)
    fi
elif [ -f "$SCRIPT_DIR/.deploy_state" ]; then
    SSH_HOST=$(grep "^SSH_HOST=" "$SCRIPT_DIR/.deploy_state" | cut -d= -f2)
    SSH_PORT=$(grep "^SSH_PORT=" "$SCRIPT_DIR/.deploy_state" | cut -d= -f2)
else
    echo "Usage: $0 <instance_id>"
    echo "   or: $0 <ssh_host> <ssh_port>"
    exit 1
fi

if [ -z "${SSH_HOST:-}" ] || [ -z "${SSH_PORT:-}" ]; then
    echo "ERROR: Could not determine SSH connection details."
    exit 1
fi

ts() { date "+%H:%M:%S"; }

notify() {
    local title="$1" msg="$2" priority="${3:-default}"
    curl -s -o /dev/null \
        -H "Title: $title" \
        -H "Priority: $priority" \
        -d "$msg" \
        "https://ntfy.sh/$NTFY_TOPIC" 2>/dev/null || true
}

# --- GPU snapshot ---
gpu_snapshot() {
    ssh -p "$SSH_PORT" $SSH_OPTS root@"$SSH_HOST" \
        "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits" 2>/dev/null
}

print_gpu() {
    local csv
    csv="$(gpu_snapshot)" || return
    IFS=',' read -r util mem_util mem_used mem_total temp power <<< "$csv"
    printf "  GPU: %s%% util | %s/%s MiB | %s°C | %sW\n" \
        "$(echo "$util" | xargs)" \
        "$(echo "$mem_used" | xargs)" "$(echo "$mem_total" | xargs)" \
        "$(echo "$temp" | xargs)" "$(echo "$power" | xargs)"
}

# --- main loop ---
echo "=== autoresearch monitor ==="
echo "  SSH: ssh -p $SSH_PORT root@$SSH_HOST"
echo "  ntfy topic: $NTFY_TOPIC"
echo "  Safety timeout: ${SAFETY_TIMEOUT}s"
echo ""

# Wait for SSH to be reachable
echo "[$(ts)] Waiting for SSH..."
for i in $(seq 1 60); do
    if ssh -p "$SSH_PORT" $SSH_OPTS root@"$SSH_HOST" true 2>/dev/null; then
        echo "[$(ts)] SSH connected."
        break
    fi
    [ "$i" -eq 60 ] && { echo "SSH timeout after 60 attempts"; notify "autoresearch" "SSH connection failed to $SSH_HOST:$SSH_PORT" "high"; exit 1; }
    sleep 5
done

REMOTE_LOG="autoresearch/train.log"
MONITOR_START=$(date +%s)
NOTIFIED_START=0
NOTIFIED_HALFWAY=0
NOTIFIED_COMPLETE=0
LAST_STEP=0
LAST_LOSS=""
FIRST_LOSS=""

echo "[$(ts)] Monitoring remote training..."
echo ""

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - MONITOR_START))

    # Safety timeout
    if [ "$ELAPSED" -ge "$SAFETY_TIMEOUT" ]; then
        echo ""
        echo "[$(ts)] SAFETY TIMEOUT (${SAFETY_TIMEOUT}s). Check instance manually."
        notify "autoresearch TIMEOUT" "Monitor safety timeout after ${ELAPSED}s" "urgent"
        exit 2
    fi

    # Check if training process is running
    PROC_RUNNING=$(ssh -p "$SSH_PORT" $SSH_OPTS root@"$SSH_HOST" "pgrep -f 'cargo.*train\|autoresearch.*train' >/dev/null 2>&1 && echo 1 || echo 0" 2>/dev/null || echo "-1")

    # Try to get latest log lines
    LINES=$(ssh -p "$SSH_PORT" $SSH_OPTS root@"$SSH_HOST" "tail -20 $REMOTE_LOG 2>/dev/null || echo '__NO_LOG__'" 2>/dev/null || echo "__SSH_ERR__")

    if [ "$LINES" = "__SSH_ERR__" ]; then
        echo "[$(ts)] SSH error, retrying..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    if [ "$LINES" = "__NO_LOG__" ]; then
        echo "[$(ts)] Waiting for training to start (no log yet)..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Parse the latest step line
    STEP_LINE=$(echo "$LINES" | grep -E '^step\s+[0-9]+' | tail -1 || true)

    if [ -n "$STEP_LINE" ]; then
        if [ "$NOTIFIED_START" -eq 0 ]; then
            NOTIFIED_START=1
            notify "autoresearch started" "Training started on Vast.ai" "default"
            echo "[$(ts)] Training started notification sent."
        fi

        STEP=$(echo "$STEP_LINE" | grep -oE 'step\s+[0-9]+' | grep -oE '[0-9]+')
        LOSS=$(echo "$STEP_LINE" | grep -oE 'loss [0-9.]+' | grep -oE '[0-9.]+')
        TOKS=$(echo "$STEP_LINE" | grep -oE 'tok/s [0-9]+' | grep -oE '[0-9]+')
        MFU=$(echo "$STEP_LINE" | grep -oE 'mfu [0-9.]+' | grep -oE '[0-9.]+')
        REMAINING=$(echo "$STEP_LINE" | grep -oE 'remaining [0-9]+' | grep -oE '[0-9]+')
        DT=$(echo "$STEP_LINE" | grep -oE 'dt [0-9.]+' | grep -oE '[0-9.]+')

        if [ -z "$FIRST_LOSS" ] && [ -n "$LOSS" ]; then
            FIRST_LOSS="$LOSS"
        fi
        LAST_LOSS="$LOSS"
        LAST_STEP="$STEP"

        printf "\r\033[K[$(ts)] step %s | loss %s | tok/s %s | mfu %s%% | remaining %ss | dt %ss" \
            "$STEP" "$LOSS" "$TOKS" "$MFU" "$REMAINING" "$DT"

        if [ $((STEP % 20)) -eq 0 ] && [ "$STEP" -gt 0 ]; then
            echo ""
            print_gpu
            HOURS=$(echo "scale=4; $ELAPSED / 3600" | bc)
            COST=$(echo "scale=2; $HOURS * $COST_PER_HOUR" | bc)
            printf "  Cost so far: \$%s (%.0fs elapsed)\n" "$COST" "$ELAPSED"
            if [ -n "$FIRST_LOSS" ] && [ -n "$LAST_LOSS" ]; then
                printf "  Loss trend: %s -> %s\n" "$FIRST_LOSS" "$LAST_LOSS"
            fi
            echo ""
        fi

        if [ -n "$REMAINING" ] && [ "$REMAINING" -lt 150 ] && [ "$NOTIFIED_HALFWAY" -eq 0 ]; then
            NOTIFIED_HALFWAY=1
            notify "autoresearch halfway" "Step $STEP, loss $LOSS, ~${REMAINING}s left" "default"
        fi
    fi

    # Check for completion
    if echo "$LINES" | grep -q "^val_bpb:"; then
        echo ""
        echo ""
        echo "[$(ts)] === Training Complete ==="
        echo "$LINES" | grep -A100 "^---" || echo "$LINES" | tail -10
        echo ""

        VAL_BPB=$(echo "$LINES" | grep -oE 'val_bpb:\s+[0-9.]+' | grep -oE '[0-9.]+' || echo "?")
        HOURS=$(echo "scale=4; $ELAPSED / 3600" | bc)
        COST=$(echo "scale=2; $HOURS * $COST_PER_HOUR" | bc)
        echo "[$(ts)] Wall time: ${ELAPSED}s, Estimated cost: \$$COST"

        if [ "$NOTIFIED_COMPLETE" -eq 0 ]; then
            NOTIFIED_COMPLETE=1
            notify "autoresearch DONE" "val_bpb=$VAL_BPB, loss=$LAST_LOSS, step=$LAST_STEP, cost=\$$COST" "high"
        fi
        exit 0
    fi

    # Check if process died without completing
    if [ "$PROC_RUNNING" = "0" ] && [ "$NOTIFIED_START" -eq 1 ]; then
        LAST_LINES=$(ssh -p "$SSH_PORT" $SSH_OPTS root@"$SSH_HOST" "tail -5 $REMOTE_LOG 2>/dev/null" || echo "")
        if echo "$LAST_LINES" | grep -qi "error\|panic\|nan"; then
            echo ""
            echo "[$(ts)] ERROR: Training crashed!"
            echo "$LAST_LINES"
            notify "autoresearch ERROR" "Training crashed. Last: $(echo "$LAST_LINES" | tail -1)" "urgent"
            exit 3
        fi
        echo ""
        echo "[$(ts)] Training process stopped, waiting for final output..."
    fi

    sleep "$POLL_INTERVAL"
done

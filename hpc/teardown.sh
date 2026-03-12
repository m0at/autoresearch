#!/bin/bash
# teardown.sh — Destroy a Vast.ai instance and download any remaining artifacts.
# Usage: ./teardown.sh [instance_id]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_FILE="$SCRIPT_DIR/.deploy_state"
RESULTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/results"
COST_PER_HOUR=1.54
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10"

# Load .env for NTFY_TOPIC
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi

# Ensure vastai CLI is available
if ! command -v vastai &>/dev/null; then
    echo "ERROR: vastai CLI not found. Install with: pip install vastai"
    exit 1
fi
if [ -n "${VASTAI:-}" ]; then
    vastai set api-key "$VASTAI" >/dev/null 2>&1
fi

# --- resolve instance ID ---
INSTANCE_ID="${1:-}"
if [ -z "$INSTANCE_ID" ] && [ -f "$STATE_FILE" ]; then
    INSTANCE_ID=$(grep "^INSTANCE_ID=" "$STATE_FILE" | cut -d= -f2)
    echo "Read instance ID from state file: $INSTANCE_ID"
fi
if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: $0 [instance_id]"
    echo "No instance ID provided and no state file found at $STATE_FILE"
    exit 1
fi

# --- read SSH details from state file (if available) ---
LAUNCH_EPOCH=""
SSH_HOST=""
SSH_PORT=""
if [ -f "$STATE_FILE" ]; then
    LAUNCH_EPOCH=$(grep "^LAUNCH_EPOCH=" "$STATE_FILE" 2>/dev/null | cut -d= -f2 || true)
    SSH_HOST=$(grep "^SSH_HOST=" "$STATE_FILE" 2>/dev/null | cut -d= -f2 || true)
    SSH_PORT=$(grep "^SSH_PORT=" "$STATE_FILE" 2>/dev/null | cut -d= -f2 || true)
fi

# --- check instance status ---
echo "Checking instance $INSTANCE_ID ..."
INSTANCE_JSON=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || echo "{}")
INSTANCE_STATUS=$(echo "$INSTANCE_JSON" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(d.get('actual_status','') or d.get('status_msg','unknown'))
" 2>/dev/null || echo "unknown")

if [ "$INSTANCE_STATUS" = "unknown" ] || [ "$INSTANCE_STATUS" = "" ]; then
    echo "Instance $INSTANCE_ID not found or already destroyed."
    exit 0
fi

# Get SSH details from API if not in state file
if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    SSH_HOST=$(echo "$INSTANCE_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh_host',''))" 2>/dev/null || true)
    SSH_PORT=$(echo "$INSTANCE_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh_port',''))" 2>/dev/null || true)
fi

# Get actual cost from API
ACTUAL_COST=$(echo "$INSTANCE_JSON" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f\"{d.get('total_cost', 0):.2f}\")
" 2>/dev/null || echo "?")

echo "Instance $INSTANCE_ID  status=$INSTANCE_STATUS  ssh=$SSH_HOST:$SSH_PORT"

# --- download remaining artifacts ---
if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && [ "$INSTANCE_STATUS" = "running" ]; then
    echo ""
    echo "Downloading artifacts from $SSH_HOST:$SSH_PORT ..."
    mkdir -p "$RESULTS_DIR"
    scp -P "$SSH_PORT" $SSH_OPTS \
        "root@$SSH_HOST:autoresearch/model.safetensors" "$RESULTS_DIR/" 2>/dev/null && \
        echo "  model.safetensors downloaded" || echo "  model.safetensors not found (ok)"
    scp -P "$SSH_PORT" $SSH_OPTS \
        "root@$SSH_HOST:autoresearch/train.log" "$RESULTS_DIR/" 2>/dev/null && \
        echo "  train.log downloaded" || echo "  train.log not found (ok)"
fi

# --- confirm destruction ---
echo ""
echo "=== CONFIRM DESTRUCTION ==="
echo "Instance:  $INSTANCE_ID"
echo "Status:    $INSTANCE_STATUS"
echo "SSH:       ssh -p ${SSH_PORT:-?} root@${SSH_HOST:-?}"
echo "Cost:      ~\$$COST_PER_HOUR/hr (total so far: \$$ACTUAL_COST)"
echo ""
read -r -p "Destroy this instance? [y/N] " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

# --- destroy ---
echo "Destroying instance $INSTANCE_ID ..."
vastai destroy instance "$INSTANCE_ID"
echo "Destroyed."

# --- cleanup state file ---
if [ -f "$STATE_FILE" ]; then
    mv "$STATE_FILE" "$STATE_FILE.done"
    echo ""
    echo "State file moved to $STATE_FILE.done"
fi

# --- ntfy notification ---
if [ -n "${NTFY_TOPIC:-}" ]; then
    BODY="Instance $INSTANCE_ID destroyed. Total cost: \$$ACTUAL_COST"
    curl -s -d "$BODY" -H "Title: autoresearch: instance destroyed" -H "Tags: white_check_mark" \
        "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 && \
        echo "Notification sent to ntfy.sh/$NTFY_TOPIC" || \
        echo "Failed to send ntfy notification (non-fatal)"
fi

echo ""
echo "Done."

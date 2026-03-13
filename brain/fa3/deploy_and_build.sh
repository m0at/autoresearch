#!/bin/bash
# Deploy FA3 wrapper to H100 and build libflashattention3.a
# Usage: bash deploy_and_build.sh <ssh_host> [ssh_port]
# Example: bash deploy_and_build.sh root@ssh2.vast.ai 36516
set -euo pipefail

SSH_HOST="${1:?Usage: $0 <ssh_host> [ssh_port]}"
SSH_PORT="${2:-22}"
SSH="ssh -p $SSH_PORT $SSH_HOST"
SCP="scp -P $SSH_PORT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/root/autoresearch/brain/fa3"

echo "=== Deploying FA3 to $SSH_HOST:$SSH_PORT ==="

# Copy wrapper and build script
$SSH "mkdir -p $REMOTE_DIR"
$SCP "$SCRIPT_DIR/flash3_api.cu" "$SCRIPT_DIR/build_fa3.sh" "$SSH_HOST:$REMOTE_DIR/"

# Build on remote
echo "=== Building FA3 on H100 (this takes a few minutes) ==="
$SSH "cd $REMOTE_DIR && bash build_fa3.sh"

# Copy result back
echo "=== Fetching libflashattention3.a ==="
mkdir -p "$SCRIPT_DIR/build"
$SCP "$SSH_HOST:$REMOTE_DIR/build/libflashattention3.a" "$SCRIPT_DIR/build/"

echo ""
echo "=== Done ==="
echo "Library at: $SCRIPT_DIR/build/libflashattention3.a"
echo "Size: $(du -h "$SCRIPT_DIR/build/libflashattention3.a" | cut -f1)"
echo ""
echo "Now build engine with:"
echo "  FLASH_ATTN_V3_BUILD_DIR=$SCRIPT_DIR/build cargo build --release"

#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="$SCRIPT_DIR/overnight.log"
mkdir -p "$SCRIPT_DIR"
exec 9>"$SCRIPT_DIR/overnight.lock"
if ! flock -n 9; then
  echo "another one-clock LIBERO-4 worker already holds overnight.lock"
  exit 0
fi
trap '' HUP
exec > >(tee -a "$LOG_PATH") 2>&1
echo "one-clock LIBERO-4 overnight worker starting"
echo "worker_script=$SCRIPT_DIR/overnight_worker.py"
echo "progress_path=$SCRIPT_DIR/progress.json"
exec /home/thor/projects/upstreams/lerobot-env/bin/python "$SCRIPT_DIR/overnight_worker.py"

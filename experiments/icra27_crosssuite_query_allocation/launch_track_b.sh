#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime="/home/wjq/workspace/venvs/libero_act/bin/python"
mkdir -p "$root/track_b/logs" "$root/track_b/pids" "$root/track_b/markers"
pid_file="$root/track_b/pids/worker_0.pid"
if [[ -f "$pid_file" ]]; then
  old_pid="$(<"$pid_file")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Track-B worker already alive as PID $old_pid" >&2
    exit 1
  fi
fi
setsid nohup "$runtime" "$root/run_track_b.py" \
  --manifest "$root/track_b_manifest.json" --gpu 0 --worker-index 0 --num-workers 1 \
  >>"$root/track_b/logs/worker_0.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pid_file"
date --iso-8601=seconds > "$root/track_b/markers/queue_launched"
printf 'launched Track-B PID: %s\n' "$pid"

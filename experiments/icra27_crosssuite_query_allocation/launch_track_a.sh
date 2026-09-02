#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime="/home/wjq/workspace/venvs/libero_act/bin/python"
mkdir -p "$root/track_a/logs" "$root/track_a/pids" "$root/track_a/markers"

for worker in 0 1 2; do
  pid_file="$root/track_a/pids/worker_${worker}.pid"
  if [[ -f "$pid_file" ]]; then
    old_pid="$(<"$pid_file")"
    if kill -0 "$old_pid" 2>/dev/null; then
      echo "worker $worker already alive as PID $old_pid" >&2
      exit 1
    fi
  fi
done

for worker in 0 1 2; do
  log="$root/track_a/logs/worker_${worker}.log"
  setsid nohup "$runtime" "$root/run_track_a.py" \
    --manifest "$root/track_a_manifest.json" \
    --gpu "$worker" --worker-index "$worker" --num-workers 3 \
    >>"$log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$root/track_a/pids/worker_${worker}.pid"
done

date --iso-8601=seconds > "$root/track_a/markers/queue_launched"
printf 'launched PIDs: %s %s %s\n' "$(<"$root/track_a/pids/worker_0.pid")" "$(<"$root/track_a/pids/worker_1.pid")" "$(<"$root/track_a/pids/worker_2.pid")"

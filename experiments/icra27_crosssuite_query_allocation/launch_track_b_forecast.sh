#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime="/home/wjq/workspace/venvs/libero_act/bin/python"
device="${B3_DEVICE:-cpu}"
mkdir -p "$root/track_b/forecast/logs" "$root/track_b/forecast/pids"
pid_file="$root/track_b/forecast/pids/worker_0.pid"
if [[ -f "$pid_file" ]]; then
  old_pid="$(<"$pid_file")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "B3 forecast worker already alive as PID $old_pid" >&2
    exit 1
  fi
fi
setsid nohup "$runtime" "$root/run_track_b_forecast.py" --gpu "$device" \
  >>"$root/track_b/forecast/logs/worker_0.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pid_file"
printf 'launched B3 forecast PID: %s\n' "$pid"

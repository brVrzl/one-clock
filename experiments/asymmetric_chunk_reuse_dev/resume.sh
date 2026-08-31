#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

launch() {
  local tasks="$1"
  local gpu="$2"
  local slug="gpu${gpu}_tasks${tasks//,/}"
  local log="$ROOT/logs/${slug}.log"
  mkdir -p "$ROOT/logs" "$ROOT/progress" "$ROOT/markers"
  local complete=1
  IFS=',' read -r -a task_array <<< "$tasks"
  for task_id in "${task_array[@]}"; do
    if [[ ! -f "$ROOT/markers/${slug}_task${task_id}.complete" ]]; then
      complete=0
    fi
  done
  if [[ "$complete" -eq 1 ]]; then return 0; fi
  if pgrep -f "run_executor.py --tasks ${tasks} --gpu ${gpu} " >/dev/null; then
    return 0
  fi
  setsid nohup "$ROOT/run_shard.sh" "$tasks" "$gpu" >"$log" 2>&1 < /dev/null &
}

if [[ $# -eq 2 ]]; then
  launch "$1" "$2"
  exit 0
fi

launch "1,2,3" 0
launch "4,5,6" 1
launch "7,8,9" 2

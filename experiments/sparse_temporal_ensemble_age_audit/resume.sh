#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wjq/workspace/one-clock/experiments/sparse_temporal_ensemble_age_audit"

launch() {
  local task="$1"
  local gpu="$2"
  local slug="${task/:task/_task}"
  local marker="$ROOT/act_h16/markers/${slug}.complete"
  local log="$ROOT/act_h16/logs/${slug}.log"
  mkdir -p "$ROOT/act_h16/logs" "$ROOT/act_h16/markers"
  if [[ -f "$marker" ]]; then
    return 0
  fi
  if pgrep -f "run_repaired_h16.py --task ${task} " >/dev/null; then
    return 0
  fi
  setsid nohup "$ROOT/act_h16/run_task.sh" "$task" "$gpu" >"$log" 2>&1 < /dev/null &
}

if [[ $# -eq 2 ]]; then
  launch "$1" "$2"
  exit 0
fi

launch "libero_10:task3" 0
launch "libero_object:task3" 1
launch "libero_spatial:task0" 2

#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
while :; do
  active=0
  for pid in $(jq -r '.gpu_queues[].controller_pid' "$root/launch_manifest.json"); do
    if kill -0 "$pid" 2>/dev/null; then
      active=1
    fi
  done
  if [ "$active" -eq 0 ]; then
    break
  fi
  sleep 30
done

/home/wjq/workspace/venvs/libero_act/bin/python "$root/analyze_fixed_horizon_blind.py"

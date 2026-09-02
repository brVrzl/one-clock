#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python=/home/wjq/workspace/venvs/libero_act/bin/python
mkdir -p "$root/logs" "$root/pids" "$root/progress"

for worker in 0 1 2; do
  pattern="icra27_care_final_gate/run_queue.py.*--worker-index ${worker}"
  if pgrep -af "$pattern" >/dev/null; then
    echo "worker $worker already running"
    continue
  fi
  log="$root/logs/worker_${worker}.log"
  setsid nohup "$python" "$root/run_queue.py" \
    --manifest "$root/queue_manifest.json" \
    --gpu "$worker" --worker-index "$worker" --num-workers 3 \
    >>"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$root/pids/worker_${worker}.pid"
  echo "launched worker $worker pid $pid log $log"
done


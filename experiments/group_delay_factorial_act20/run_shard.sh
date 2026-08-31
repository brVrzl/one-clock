#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/wjq/workspace/venvs/libero_act/bin/python"
TASKS="${1:?comma-separated task IDs required}"
GPU="${2:?GPU index required}"
SLUG="gpu${GPU}_tasks${TASKS//,/}"
mkdir -p "$ROOT/logs" "$ROOT/progress"
"$PYTHON" "$ROOT/run_factorial.py" \
  --tasks "$TASKS" \
  --gpu "$GPU" \
  --output-root "$ROOT" \
  --progress "$ROOT/progress/${SLUG}.json"

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
for task_id in "${TASK_ARRAY[@]}"; do
  "$PYTHON" "$ROOT/validate_shard.py" \
    --result "$ROOT/results/task_$(printf '%02d' "$task_id").json" \
    --protocol "$ROOT/protocol.json" \
    --marker "$ROOT/markers/${SLUG}_task${task_id}.complete"
done

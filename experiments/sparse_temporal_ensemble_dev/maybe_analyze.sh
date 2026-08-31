#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/wjq/workspace/venvs/libero_act/bin/python"
tasks=(libero_object_task3 libero_spatial_task0 libero_goal_task2 libero_10_task3)

for slug in "${tasks[@]}"; do
  [[ -s "${ROOT_DIR}/act/markers/${slug}.complete" ]] || exit 0
  [[ -s "${ROOT_DIR}/smolvla/markers/${slug}.complete" ]] || exit 0
done

mkdir -p "${ROOT_DIR}/run_state"
if ! mkdir "${ROOT_DIR}/run_state/analysis.lock" 2>/dev/null; then
  exit 0
fi
trap 'rmdir "${ROOT_DIR}/run_state/analysis.lock" 2>/dev/null || true' EXIT
if [[ -s "${ROOT_DIR}/run_state/analysis.complete" ]]; then
  exit 0
fi
"${PYTHON}" "${ROOT_DIR}/analyze.py"
printf 'complete\n' >"${ROOT_DIR}/run_state/analysis.complete"

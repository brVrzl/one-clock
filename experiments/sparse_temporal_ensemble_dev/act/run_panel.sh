#!/usr/bin/env bash
set -euo pipefail

# One ACT worker owns GPU 0 and processes task shards serially.  Each task has
# its own output, progress sidecar, log, and completion marker, so a resumed
# invocation skips only shards that finished cleanly.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/wjq/workspace/venvs/libero_act/bin/python"
GPU="${1:-0}"
ACT_DIR="${ROOT_DIR}/act"
mkdir -p "${ACT_DIR}/results" "${ACT_DIR}/progress" "${ACT_DIR}/logs" \
  "${ACT_DIR}/markers" "${ACT_DIR}/invalidated"

tasks=(
  "libero_object:task3"
  "libero_spatial:task0"
  "libero_goal:task2"
  "libero_10:task3"
)

for task in "${tasks[@]}"; do
  slug="${task//:/_}"
  output="${ACT_DIR}/results/${slug}.json"
  progress="${ACT_DIR}/progress/${slug}.json"
  log="${ACT_DIR}/logs/${slug}.log"
  marker="${ACT_DIR}/markers/${slug}.complete"
  if [[ -s "${marker}" && -s "${output}" ]] && \
     "${PYTHON}" "${ROOT_DIR}/validate_result.py" \
       --result "${output}" --policy act --episodes 10 >/dev/null 2>&1; then
    continue
  fi
  if [[ -e "${output}" || -e "${progress}" || -e "${log}" || -e "${marker}" ]]; then
    archive="${ACT_DIR}/invalidated/${slug}/$(date +%Y%m%dT%H%M%S)-$$"
    mkdir -p "${archive}"
    for path in "${output}" "${progress}" "${log}" "${marker}"; do
      if [[ -e "${path}" ]]; then
        mv "${path}" "${archive}/"
      fi
    done
  fi
  "${PYTHON}" "${ACT_DIR}/run_act_sparse_te.py" \
    --protocol "${ROOT_DIR}/protocol.json" \
    --task "${task}" --gpu "${GPU}" \
    --output "${output}" --progress "${progress}" \
    >"${log}" 2>&1
  "${PYTHON}" "${ROOT_DIR}/validate_result.py" \
    --result "${output}" --policy act --episodes 10 --marker "${marker}" \
    >>"${log}" 2>&1
done

printf 'completed ACT panel\n' >"${ACT_DIR}/markers/panel.complete"

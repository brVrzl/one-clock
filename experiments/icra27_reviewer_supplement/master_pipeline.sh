#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/../.." && pwd)"
TRACK_DIR="${REPO_DIR}/experiments/icra27_crosssuite_query_allocation"
PYTHON="/home/wjq/workspace/venvs/libero_act/bin/python"
ORCH="${ROOT_DIR}/orchestration"
mkdir -p "${ORCH}/logs" "${ORCH}/pids"
date +%s > "${ORCH}/pipeline_start_epoch"

fail() {
  printf '%s\n' "FAILED stage=${1} time=$(date --iso-8601=seconds)" > "${ORCH}/PIPELINE_FAILED"
  "${PYTHON}" "${ROOT_DIR}/update_handoff.py" || true
  exit 1
}
trap 'fail unexpected_line_${LINENO}' ERR

wait_track_a() {
  while true; do
    complete=$(find "${TRACK_DIR}/track_a/markers" -maxdepth 1 -type f -name '*.complete' | wc -l)
    failed=$(find "${TRACK_DIR}/track_a/markers" -maxdepth 1 -type f -name '*.technical_failed' | wc -l)
    if [[ "${failed}" -ne 0 ]]; then fail track_a_technical_failure; fi
    if [[ "${complete}" -eq 2700 ]]; then break; fi
    sleep 60
  done
  while pgrep -f "${TRACK_DIR}/run_track_a.py" >/dev/null; do sleep 10; done
}

launch_phase() {
  local phase="$1" expected="$2"
  mkdir -p "${ROOT_DIR}/markers/${phase}"
  if mkdir "${ORCH}/${phase}_LAUNCH_LOCK" 2>/dev/null; then
    for gpu in 0 1 2; do
      setsid nohup "${PYTHON}" "${ROOT_DIR}/run_queue.py" --phase "${phase}" --gpu "${gpu}" --worker-index "${gpu}" --num-workers 3 \
        >"${ORCH}/logs/${phase}_worker_${gpu}.log" 2>&1 < /dev/null &
      echo "$!" > "${ORCH}/pids/${phase}_worker_${gpu}.pid"
    done
    printf '%s\n' "$(date --iso-8601=seconds)" > "${ORCH}/${phase}_LAUNCHED"
  fi
  while true; do
    complete=$(find "${ROOT_DIR}/markers/${phase}" -maxdepth 1 -type f -name '*.complete' 2>/dev/null | wc -l)
    failed=$(find "${ROOT_DIR}/markers/${phase}" -maxdepth 1 -type f -name '*.technical_failed' 2>/dev/null | wc -l)
    if [[ "${failed}" -ne 0 ]]; then fail "${phase}_technical_failure"; fi
    if [[ "${complete}" -eq "${expected}" ]]; then break; fi
    live=0
    for pidfile in "${ORCH}/pids/${phase}_worker_"*.pid; do
      [[ -f "${pidfile}" ]] && kill -0 "$(<"${pidfile}")" 2>/dev/null && live=$((live+1))
    done
    if [[ "${live}" -eq 0 ]]; then fail "${phase}_workers_exited_early"; fi
    sleep 60
  done
  "${PYTHON}" "${ROOT_DIR}/validate_supplement.py" --phase "${phase}" > "${ORCH}/logs/${phase}_integrity.log" 2>&1 || fail "${phase}_integrity"
  printf 'COMPLETE\n' > "${ORCH}/${phase^^}_COMPLETE"
  "${PYTHON}" "${ROOT_DIR}/update_handoff.py" || true
}

wait_track_a
"${PYTHON}" "${ROOT_DIR}/track_a_gate.py" > "${ORCH}/logs/track_a_integrity.log" 2>&1 || fail track_a_integrity
"${PYTHON}" "${ROOT_DIR}/track_a_finalize.py" > "${ORCH}/logs/track_a_analysis.log" 2>&1 || fail track_a_analysis
"${PYTHON}" "${ROOT_DIR}/validate_supplement.py" --static > "${ORCH}/logs/supplement_static.log" 2>&1 || fail supplement_static
"${PYTHON}" "${ROOT_DIR}/run_canaries.py" --gpu 0 > "${ORCH}/logs/r1_canaries.log" 2>&1 || fail r1_canaries

launch_phase r1a 1512
launch_phase r1b 252
launch_phase r1c 280
launch_phase r1d 100

start_epoch=$(<"${ORCH}/pipeline_start_epoch")
elapsed=$(( $(date +%s) - start_epoch ))
if [[ "${elapsed}" -le 57600 ]]; then
  "${PYTHON}" "${ROOT_DIR}/run_canaries.py" --gpu 0 --r2 > "${ORCH}/logs/r2_preflight.log" 2>&1 || fail r2_preflight
  launch_phase r2 160
else
  printf 'SKIPPED_RUNTIME_GATE elapsed_seconds=%s required_remaining_seconds=28800\n' "${elapsed}" > "${ORCH}/R2_SKIPPED_RUNTIME_GATE"
fi

"${PYTHON}" "${ROOT_DIR}/analyze_supplement.py" > "${ORCH}/logs/supplement_analysis.log" 2>&1 || fail supplement_analysis
printf 'COMPLETE\n' > "${ORCH}/PIPELINE_COMPLETE"
"${PYTHON}" "${ROOT_DIR}/update_handoff.py"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/wjq/workspace/venvs/libero_act/bin/python"
ORCH="${ROOT_DIR}/orchestration"
LOG_DIR="${ORCH}/logs"
PID_DIR="${ORCH}/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}"
test "$(<"${ORCH}/pipeline_start_epoch")" = 1788354953
test "$(find "${ROOT_DIR}/results/r1d" -maxdepth 1 -type f 2>/dev/null | wc -l)" -eq 0
test "$(find "${ROOT_DIR}/markers/r1d" -maxdepth 1 -type f 2>/dev/null | wc -l)" -eq 0
test "$(find "${ROOT_DIR}/attempts/r1d" -maxdepth 1 -type f 2>/dev/null | wc -l)" -eq 0

PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" "${ROOT_DIR}/r1d_repair_canary.py" \
  >"${LOG_DIR}/r1d_repair_canary.log" 2>&1

for worker in 0 1 2; do
  pid_file="${PID_DIR}/r1d_repair_worker_${worker}.pid"
  if [[ -s "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
    echo "R1D repair worker ${worker} is already active" >&2
    exit 2
  fi
done

rmdir "${ORCH}/r1d_LAUNCH_LOCK"
for worker in 0 1 2; do
  setsid nohup "${PYTHON}" "${ROOT_DIR}/r1d_runtime_repair.py" \
    --phase r1d --gpu "${worker}" --worker-index "${worker}" --num-workers 3 \
    >"${LOG_DIR}/r1d_repair_worker_${worker}.log" 2>&1 < /dev/null &
  echo "$!" >"${PID_DIR}/r1d_repair_worker_${worker}.pid"
done
date --iso-8601=seconds >"${ORCH}/r1d_REPAIR_LAUNCHED"

while true; do
  complete="$(find "${ROOT_DIR}/markers/r1d" -maxdepth 1 -type f -name '*.complete' 2>/dev/null | wc -l)"
  failed="$(find "${ROOT_DIR}/markers/r1d" -maxdepth 1 -type f -name '*.technical_failed' 2>/dev/null | wc -l)"
  [[ "${failed}" -eq 0 ]] || { echo "R1D technical failure" >&2; exit 3; }
  [[ "${complete}" -eq 100 ]] && break
  live=0
  for worker in 0 1 2; do
    pid_file="${PID_DIR}/r1d_repair_worker_${worker}.pid"
    [[ -s "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null && live=$((live + 1))
  done
  [[ "${live}" -gt 0 ]] || { echo "R1D repair workers exited early" >&2; exit 4; }
  sleep 30
done

for worker in 0 1 2; do
  pid_file="${PID_DIR}/r1d_repair_worker_${worker}.pid"
  if [[ -s "${pid_file}" ]]; then
    while kill -0 "$(<"${pid_file}")" 2>/dev/null; do sleep 2; done
  fi
done

PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" "${ROOT_DIR}/validate_supplement.py" --phase r1d \
  >"${LOG_DIR}/r1d_repair_integrity.log" 2>&1
test "$(find "${ROOT_DIR}/attempts/r1d" -maxdepth 1 -type f 2>/dev/null | wc -l)" -eq 0
printf 'COMPLETE\n' >"${ORCH}/R1D_COMPLETE"
printf 'COMPLETE scientific_retries=0 cells=100\n' >"${ORCH}/R1D_REPAIR_COMPLETE"

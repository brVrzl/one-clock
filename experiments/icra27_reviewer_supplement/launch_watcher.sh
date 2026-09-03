#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="${ROOT_DIR}/orchestration"
PYTHON="/home/wjq/workspace/venvs/libero_act/bin/python"
mkdir -p "${ORCH}/logs" "${ORCH}/pids"
[[ -x "${PYTHON}" ]] || { echo "configured Python is not executable: ${PYTHON}" >&2; exit 2; }
[[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == "--resume" ) ]] || {
  echo "usage: $0 [--resume]" >&2; exit 2
}

master_is_live() {
  local pid="$1" command_line
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null || return 1
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${command_line}" == *"master_pipeline.sh"* || "${command_line}" == *"resume_master.sh"* ]]
}

if [[ -f "${ORCH}/PIPELINE_COMPLETE" ]]; then
  echo "pipeline already complete"; exit 0
fi
if ! mkdir "${ORCH}/MASTER_LAUNCH_LOCK" 2>/dev/null; then
  if [[ -s "${ORCH}/pids/master.pid" ]] && master_is_live "$(<"${ORCH}/pids/master.pid")"; then
    echo "master already active pid=$(<"${ORCH}/pids/master.pid")"; exit 3
  fi
  if [[ "${1:-}" != "--resume" ]]; then
    echo "stale launch lock; inspect state, then use --resume" >&2; exit 4
  fi
  rmdir "${ORCH}/MASTER_LAUNCH_LOCK"
  mkdir "${ORCH}/MASTER_LAUNCH_LOCK"
fi
mode="${1:-}"
if [[ "${mode}" == "--resume" ]]; then
  [[ -s "${ORCH}/pipeline_start_epoch" ]] || { echo "resume requires original pipeline_start_epoch" >&2; exit 5; }
  [[ -s "${ORCH}/PIPELINE_FAILED" ]] || { echo "resume requires PIPELINE_FAILED" >&2; exit 6; }
  mkdir -p "${ORCH}/failure_history"
  failure_stamp="$(date +%Y%m%dT%H%M%S%z)"
  mv "${ORCH}/PIPELINE_FAILED" "${ORCH}/failure_history/PIPELINE_FAILED.${failure_stamp}"
  runner=(bash "${ROOT_DIR}/resume_master.sh" --resume)
else
  runner=(bash "${ROOT_DIR}/master_pipeline.sh")
fi
setsid nohup "${runner[@]}" >>"${ORCH}/logs/master.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${ORCH}/pids/master.pid"
printf '%s\n' "$(date --iso-8601=seconds)" > "${ORCH}/MASTER_LAUNCHED"
echo "master launched pid=${pid} log=${ORCH}/logs/master.log"

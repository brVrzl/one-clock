#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="${ROOT_DIR}/orchestration"
mkdir -p "${ORCH}/logs" "${ORCH}/pids"
if ! mkdir "${ORCH}/MASTER_LAUNCH_LOCK" 2>/dev/null; then
  if [[ -s "${ORCH}/pids/master.pid" ]] && kill -0 "$(<"${ORCH}/pids/master.pid")" 2>/dev/null; then
    echo "master already active pid=$(<"${ORCH}/pids/master.pid")"; exit 3
  fi
  if [[ "${1:-}" != "--resume" ]]; then
    echo "stale launch lock; inspect state, then use --resume" >&2; exit 4
  fi
  rmdir "${ORCH}/MASTER_LAUNCH_LOCK"
  mkdir "${ORCH}/MASTER_LAUNCH_LOCK"
fi
setsid nohup bash "${ROOT_DIR}/master_pipeline.sh" >"${ORCH}/logs/master.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${ORCH}/pids/master.pid"
printf '%s\n' "$(date --iso-8601=seconds)" > "${ORCH}/MASTER_LAUNCHED"
echo "master launched pid=${pid} log=${ORCH}/logs/master.log"

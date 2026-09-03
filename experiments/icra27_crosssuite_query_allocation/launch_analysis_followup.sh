#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="${ROOT}/orchestration"
mkdir -p "${ORCH}"
if ! mkdir "${ORCH}/ANALYSIS_FOLLOWUP_LOCK" 2>/dev/null; then
  if [[ -s "${ORCH}/analysis_followup.pid" ]]; then
    pid="$(<"${ORCH}/analysis_followup.pid")"
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    if kill -0 "${pid}" 2>/dev/null && [[ "${cmdline}" == *"analysis_followup.sh"* ]]; then
      echo "analysis follower already active pid=${pid}"; exit 3
    fi
  fi
  echo "stale analysis-follower lock; inspect before relaunch" >&2
  exit 4
fi
setsid nohup bash "${ROOT}/analysis_followup.sh" >> "${ORCH}/analysis_followup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${ORCH}/analysis_followup.pid"
printf 'analysis follower launched pid=%s log=%s\n' "${pid}" "${ORCH}/analysis_followup.log"

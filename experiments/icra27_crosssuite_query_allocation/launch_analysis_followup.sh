#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="${ROOT}/orchestration"
mkdir -p "${ORCH}"

resume=0
if [[ "${1:-}" == "--resume" ]]; then
  resume=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--resume]" >&2
  exit 2
fi

if ! mkdir "${ORCH}/ANALYSIS_FOLLOWUP_LOCK" 2>/dev/null; then
  if [[ -s "${ORCH}/analysis_followup.pid" ]]; then
    pid="$(<"${ORCH}/analysis_followup.pid")"
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    if kill -0 "${pid}" 2>/dev/null && [[ "${cmdline}" == *"analysis_followup.sh"* ]]; then
      echo "analysis follower already active pid=${pid}"; exit 3
    fi
  fi
  if [[ "${resume}" -ne 1 ]]; then
    echo "stale analysis-follower lock; use --resume after inspection" >&2
    exit 4
  fi
  if [[ ! -f "${ORCH}/ANALYSIS_FOLLOWUP_FAILED" ]]; then
    echo "refusing --resume without ANALYSIS_FOLLOWUP_FAILED" >&2
    exit 5
  fi
  mkdir -p "${ORCH}/failure_history"
  stamp="$(date +%Y%m%dT%H%M%S%z)"
  mv "${ORCH}/ANALYSIS_FOLLOWUP_FAILED" \
    "${ORCH}/failure_history/ANALYSIS_FOLLOWUP_FAILED.${stamp}"
  rmdir "${ORCH}/ANALYSIS_FOLLOWUP_LOCK"
  mkdir "${ORCH}/ANALYSIS_FOLLOWUP_LOCK"
fi
setsid nohup bash "${ROOT}/analysis_followup.sh" >> "${ORCH}/analysis_followup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${ORCH}/analysis_followup.pid"
printf 'analysis follower launched pid=%s log=%s\n' "${pid}" "${ORCH}/analysis_followup.log"

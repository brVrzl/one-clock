#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
reservation="${ROOT_DIR}/smolvla/claims/libero_10_task3.claim"
mkdir -p "${ROOT_DIR}/smolvla/claims"
if mkdir "${reservation}" 2>/dev/null; then
  printf '%s\n' "$$" >"${reservation}/owner.pid"
fi
while [[ ! -s "${ROOT_DIR}/act/markers/panel.complete" ]]; do
  if [[ -s "${ROOT_DIR}/act/panel.pid" ]]; then
    panel_pid="$(cat "${ROOT_DIR}/act/panel.pid")"
    if ! kill -0 "${panel_pid}" 2>/dev/null; then
      printf 'ACT panel exited without a completion marker; run resume.sh\n' >&2
      exit 1
    fi
  fi
  sleep 15
done
if [[ -d "${reservation}" ]] && [[ "$(cat "${reservation}/owner.pid" 2>/dev/null || true)" == "$$" ]]; then
  rm -f "${reservation}/owner.pid"
  rmdir "${reservation}"
fi
exec "${ROOT_DIR}/smolvla/queue_worker.sh" 0

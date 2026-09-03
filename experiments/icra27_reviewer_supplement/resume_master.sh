#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="${ROOT_DIR}/orchestration"
SYSTEM_DATE="/usr/bin/date"
ORIGINAL_EPOCH_FILE="${ORCH}/pipeline_start_epoch"

[[ "${1:-}" == "--resume" || "${1:-}" == "--canary" ]] || {
  echo "usage: $0 --resume|--canary" >&2
  exit 2
}
[[ -s "${ORIGINAL_EPOCH_FILE}" ]] || {
  echo "missing original pipeline start epoch: ${ORIGINAL_EPOCH_FILE}" >&2
  exit 3
}
ORIGINAL_EPOCH="$(<"${ORIGINAL_EPOCH_FILE}")"
[[ "${ORIGINAL_EPOCH}" =~ ^[0-9]+$ ]] || {
  echo "invalid original pipeline start epoch: ${ORIGINAL_EPOCH}" >&2
  exit 4
}

DATE_INTERCEPT_MARKER="${ORCH}/.resume_date_intercept.$$"
export ORIGINAL_EPOCH DATE_INTERCEPT_MARKER SYSTEM_DATE
date() {
  if [[ "$#" -eq 1 && "$1" == "+%s" && ! -e "${DATE_INTERCEPT_MARKER}" ]]; then
    : > "${DATE_INTERCEPT_MARKER}"
    printf '%s\n' "${ORIGINAL_EPOCH}"
  else
    "${SYSTEM_DATE}" "$@"
  fi
}
export -f date

cleanup() { rm -f "${DATE_INTERCEPT_MARKER}"; }
trap cleanup EXIT

if [[ "$1" == "--canary" ]]; then
  first="$(date +%s)"
  second="$(date +%s)"
  [[ "${first}" == "${ORIGINAL_EPOCH}" && "${second}" -ge "${ORIGINAL_EPOCH}" ]] || exit 5
  [[ "$(<"${ORIGINAL_EPOCH_FILE}")" == "${ORIGINAL_EPOCH}" ]] || exit 6
  echo "resume epoch canary PASS original=${ORIGINAL_EPOCH} subsequent=${second}"
  exit 0
fi

set +e
bash "${ROOT_DIR}/master_pipeline.sh"
status=$?
set -e
if [[ "${status}" -ne 0 ]]; then
  echo "master pipeline exited status=${status}" >&2
  [[ -s "${ORCH}/PIPELINE_FAILED" ]] && cat "${ORCH}/PIPELINE_FAILED" >&2
  while IFS= read -r log; do
    echo "last lines: ${log}" >&2
    tail -n 30 "${log}" >&2
  done < <(find "${ORCH}/logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' | sort -nr | head -n 3 | cut -d' ' -f2-)
fi
exit "${status}"

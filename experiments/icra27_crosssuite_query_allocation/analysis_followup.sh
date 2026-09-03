#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPLEMENT="${ROOT}/../icra27_reviewer_supplement"
PYTHON="/home/wjq/workspace/venvs/libero_act/bin/python"
ORCH="${ROOT}/orchestration"
mkdir -p "${ORCH}"

fail() {
  printf 'FAILED stage=%s time=%s\n' "$1" "$(date --iso-8601=seconds)" > "${ORCH}/ANALYSIS_FOLLOWUP_FAILED"
  "${PYTHON}" "${SUPPLEMENT}/update_handoff.py" || true
  exit 1
}
trap 'fail unexpected_line_${LINENO}' ERR

while true; do
  b3_complete=$(find "${ROOT}/track_b/forecast/markers" -maxdepth 1 -type f -name '*.complete' 2>/dev/null | wc -l)
  if [[ "${b3_complete}" -eq 8 ]]; then break; fi
  if [[ -s "${ROOT}/track_b/forecast/pids/worker_0.pid" ]]; then
    b3_pid="$(<"${ROOT}/track_b/forecast/pids/worker_0.pid")"
    kill -0 "${b3_pid}" 2>/dev/null || fail b3_worker_exited_with_${b3_complete}_of_8
  else
    fail b3_pid_missing
  fi
  sleep 60
done

"${PYTHON}" "${ROOT}/analyze_track_b_forecast.py" > "${ORCH}/b3_analysis.log" 2>&1 || fail b3_analysis
printf 'COMPLETE\n' > "${ORCH}/B3_ANALYSIS_COMPLETE"
"${PYTHON}" "${SUPPLEMENT}/update_handoff.py" || true

while [[ ! -f "${SUPPLEMENT}/orchestration/PIPELINE_COMPLETE" ]]; do
  [[ ! -f "${SUPPLEMENT}/orchestration/PIPELINE_FAILED" ]] || fail reviewer_pipeline_failed
  sleep 60
done

"${PYTHON}" "${ROOT}/analyze_final_mechanism_relationships.py" > "${ORCH}/final_mechanism_analysis.log" 2>&1 || fail final_mechanism_analysis
printf 'COMPLETE\n' > "${ORCH}/ANALYSIS_FOLLOWUP_COMPLETE"
"${PYTHON}" "${SUPPLEMENT}/update_handoff.py"

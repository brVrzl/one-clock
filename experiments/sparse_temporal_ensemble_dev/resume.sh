#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ROOT_DIR}/run_state" "${ROOT_DIR}/smolvla/logs"
if ! mkdir "${ROOT_DIR}/run_state/resume.lock" 2>/dev/null; then
  printf 'another resume invocation is active\n' >&2
  exit 1
fi
trap 'rmdir "${ROOT_DIR}/run_state/resume.lock" 2>/dev/null || true' EXIT

alive() {
  local pid_file="$1"
  [[ -s "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}")"
  kill -0 "${pid}" 2>/dev/null
}

for claim in "${ROOT_DIR}"/smolvla/claims/*.claim; do
  [[ -d "${claim}" ]] || continue
  owner_file="${claim}/owner.pid"
  if [[ ! -s "${owner_file}" ]]; then
    rm -rf "${claim}"
    continue
  fi
  owner="$(cat "${owner_file}")"
  owner_command="$(ps -p "${owner}" -o args= 2>/dev/null || true)"
  if [[ "${owner_command}" != *"smolvla/queue_worker.sh"* && \
        "${owner_command}" != *"after_act_join_smol.sh"* ]]; then
    rm -rf "${claim}"
  fi
done

if [[ ! -s "${ROOT_DIR}/act/markers/panel.complete" ]] && ! alive "${ROOT_DIR}/act/panel.pid"; then
  setsid nohup bash "${ROOT_DIR}/act/run_panel.sh" 0 \
    >"${ROOT_DIR}/act/logs/panel_launcher.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"${ROOT_DIR}/act/panel.pid"
fi

smol_complete=true
for slug in libero_object_task3 libero_spatial_task0 libero_goal_task2 libero_10_task3; do
  if [[ ! -s "${ROOT_DIR}/smolvla/markers/${slug}.complete" ]]; then
    smol_complete=false
  fi
done

if [[ "${smol_complete}" == false ]]; then
  for gpu in 1 2; do
    pid_file="${ROOT_DIR}/smolvla/worker_gpu${gpu}.pid"
    if ! alive "${pid_file}"; then
      setsid nohup bash "${ROOT_DIR}/smolvla/queue_worker.sh" "${gpu}" \
        >"${ROOT_DIR}/smolvla/logs/worker_gpu${gpu}.log" 2>&1 < /dev/null &
      printf '%s\n' "$!" >"${pid_file}"
    fi
  done
  if ! alive "${ROOT_DIR}/run_state/after_act.pid"; then
    setsid nohup bash "${ROOT_DIR}/after_act_join_smol.sh" \
      >"${ROOT_DIR}/smolvla/logs/after_act_gpu0.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" >"${ROOT_DIR}/run_state/after_act.pid"
  fi
fi

"${ROOT_DIR}/maybe_analyze.sh"
printf 'resume complete; inspect STATUS.md and per-worker progress files\n'

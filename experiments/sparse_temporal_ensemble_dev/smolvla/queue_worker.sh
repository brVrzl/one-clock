#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: queue_worker.sh GPU}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOL_DIR="${ROOT_DIR}/smolvla"
PYTHON="/home/wjq/workspace/venvs/libero_act/bin/python"
RUNNER="${SMOL_DIR}/run_smolvla_sparse_te.py"
VALIDATOR="${ROOT_DIR}/validate_result.py"
mkdir -p "${SMOL_DIR}/results" "${SMOL_DIR}/progress" "${SMOL_DIR}/logs" \
  "${SMOL_DIR}/markers" "${SMOL_DIR}/claims" "${SMOL_DIR}/invalidated"

tasks=(
  "libero_object:task3"
  "libero_spatial:task0"
  "libero_goal:task2"
  "libero_10:task3"
)

active_claim=""
cleanup_claim() {
  if [[ -n "${active_claim}" && -d "${active_claim}" ]]; then
    rm -f "${active_claim}/owner.pid"
    rmdir "${active_claim}" 2>/dev/null || true
  fi
}
trap cleanup_claim EXIT INT TERM

archive_unvalidated() {
  local slug="$1"
  local output="$2"
  local progress="$3"
  local log="$4"
  local marker="$5"
  if [[ -e "${output}" || -e "${progress}" || -e "${log}" || -e "${marker}" ]]; then
    local archive="${SMOL_DIR}/invalidated/${slug}/$(date +%Y%m%dT%H%M%S)-$$"
    mkdir -p "${archive}"
    for path in "${output}" "${progress}" "${log}" "${marker}"; do
      if [[ -e "${path}" ]]; then
        mv "${path}" "${archive}/"
      fi
    done
  fi
}

for task in "${tasks[@]}"; do
  slug="${task//:/_}"
  output="${SMOL_DIR}/results/${slug}.json"
  progress="${SMOL_DIR}/progress/${slug}.json"
  log="${SMOL_DIR}/logs/${slug}.log"
  marker="${SMOL_DIR}/markers/${slug}.complete"
  claim="${SMOL_DIR}/claims/${slug}.claim"

  if [[ -s "${marker}" && -s "${output}" ]] && \
     "${PYTHON}" "${VALIDATOR}" --result "${output}" --policy smolvla --episodes 10 >/dev/null 2>&1; then
    continue
  fi
  if ! mkdir "${claim}" 2>/dev/null; then
    continue
  fi
  active_claim="${claim}"
  printf '%s\n' "$$" >"${claim}/owner.pid"

  if [[ -s "${marker}" && -s "${output}" ]] && \
     "${PYTHON}" "${VALIDATOR}" --result "${output}" --policy smolvla --episodes 10 >/dev/null 2>&1; then
    cleanup_claim
    active_claim=""
    continue
  fi
  archive_unvalidated "${slug}" "${output}" "${progress}" "${log}" "${marker}"
  "${PYTHON}" "${RUNNER}" --protocol "${ROOT_DIR}/protocol.json" \
    --task "${task}" --gpu "${GPU}" --output "${output}" --progress-file "${progress}" \
    >"${log}" 2>&1
  "${PYTHON}" "${VALIDATOR}" --result "${output}" --policy smolvla --episodes 10 \
    --marker "${marker}" >>"${log}" 2>&1
  cleanup_claim
  active_claim=""
done

"${ROOT_DIR}/maybe_analyze.sh"

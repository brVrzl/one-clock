#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SUITE:taskID GPU" >&2
  exit 2
fi

task_key="$1"
gpu="$2"
slug="${task_key//:/_}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/wjq/workspace/venvs/libero_act/bin/python"
methods="${GROUP_MEMORY_ACT_METHODS:-M0_h16,M1_shared_te_h16,M2_shared_cogact_h16,M3_group_cogact_h16}"

mkdir -p "${script_dir}/act/pairing" "${script_dir}/act/results" "${script_dir}/act/progress"
"${python_bin}" "${script_dir}/run_act_group_memory.py" \
  --task "${task_key}" --methods "${methods}" --gpu "${gpu}" \
  --pairing-smoke --output "${script_dir}/act/pairing/${slug}.json"
"${python_bin}" "${script_dir}/run_act_group_memory.py" \
  --task "${task_key}" --methods "${methods}" --gpu "${gpu}" \
  --pairing-audit "${script_dir}/act/pairing/${slug}.json" \
  --output "${script_dir}/act/results/${slug}.json" \
  --progress-file "${script_dir}/act/progress/${slug}.json"


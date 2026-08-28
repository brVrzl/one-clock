#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/wjq/workspace/one-clock"
run_dir="$repo_root/research/overnight_pppr_20260828"
python_bin="/home/wjq/workspace/venvs/libero_act/bin/python"

cd "$repo_root"

if [[ ! -f "$run_dir/phase0_features.complete" ]]; then
  "$python_bin" -m pytest -q "$run_dir/test_pppr_phase0.py"
  "$python_bin" "$run_dir/build_phase0.py"
fi

if [[ -f "$run_dir/analyze_control_relevance.py" && ! -f "$run_dir/phase0_analysis.complete" ]]; then
  "$python_bin" "$run_dir/analyze_control_relevance.py"
fi

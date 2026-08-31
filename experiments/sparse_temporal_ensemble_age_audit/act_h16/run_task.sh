#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wjq/workspace/one-clock/experiments/sparse_temporal_ensemble_age_audit"
PYTHON="/home/wjq/workspace/venvs/libero_act/bin/python"
TASK="${1:?task key required}"
GPU="${2:?GPU index required}"
SLUG="${TASK/:task/_task}"
SLUG="${SLUG/libero_/libero_}"
RESULT="$ROOT/act_h16/results/${SLUG}.json"
PROGRESS="$ROOT/act_h16/progress/${SLUG}.json"
MARKER="$ROOT/act_h16/markers/${SLUG}.complete"

mkdir -p "$ROOT/act_h16/results" "$ROOT/act_h16/progress" "$ROOT/act_h16/markers"

if [[ -f "$MARKER" ]] && "$PYTHON" "$ROOT/act_h16/validate_repaired_h16.py" --result "$RESULT" --episodes 10 >/dev/null; then
  exit 0
fi

"$PYTHON" "$ROOT/act_h16/run_repaired_h16.py" \
  --task "$TASK" \
  --gpu "$GPU" \
  --output "$RESULT" \
  --progress "$PROGRESS"

"$PYTHON" "$ROOT/act_h16/validate_repaired_h16.py" \
  --result "$RESULT" \
  --episodes 10 \
  --marker "$MARKER"

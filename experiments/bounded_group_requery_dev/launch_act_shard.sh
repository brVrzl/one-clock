#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <task-key> <gpu> [method]" >&2
  exit 2
fi

TASK_KEY="$1"
GPU="$2"
METHOD="${3:-M1_arm_phase}"
TASK_SLUG="${TASK_KEY//:/_}"
ACT_PY="${ACT_PY:-/home/wjq/workspace/venvs/libero_act/bin/python}"
OUT="act/results/${TASK_SLUG}.json"
PROGRESS="act/progress/${TASK_SLUG}.json"
LOG="act/logs/${TASK_SLUG}_${METHOD}.log"

mkdir -p act/results act/progress act/logs
exec "$ACT_PY" run_bounded_group_requery.py \
  --task "$TASK_KEY" \
  --methods "$METHOD" \
  --gpu "$GPU" \
  --output "$OUT" \
  --progress-file "$PROGRESS" \
  2>&1 | tee "$LOG"

#!/usr/bin/env bash
set -euo pipefail

# Run the preregistered Stage-A cells as isolated one-seed processes.  The
# pinned headless SAPIEN build can stall while resetting a second episode in a
# long-lived process; isolating seeds preserves the fixed paired seed set and
# lets the remaining trials continue.  A timeout is recorded as an anomaly by
# the analysis script and is never converted into a success.

ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
RW=${RW:-$ROOT/../upstreams/RoboTwin}
PY=${PY:-$ROOT/../venvs/robotwin/bin/python}
SITE=${SITE:-$ROOT/../venvs/robotwin/lib/python3.10/site-packages}
SEEDS=${SEEDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-120}
CHECKPOINT=${CHECKPOINT:-$ROOT/../../checkpoints/robotwin/act-place_can_basket/demo_clean-50}
RUN_ROOT=${RUN_ROOT:-$ROOT/experiments/runs/robotwin_static/place_can_basket}

LIBS=$(find "$SITE/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | paste -sd:)
COMMON=(
  --config "$ROOT/configs/gate0_place_can_basket.yaml"
  --robotwin-root "$RW"
  --checkpoint "$CHECKPOINT"
  --planner mplib_RRT
  --skip-expert-check
)

run_cell() {
  local label=$1 strategy=$2 arm=$3 gripper=$4 seed=$5
  local out="$RUN_ROOT/$label/seed_$seed"
  local log="/tmp/robotwin_static_${label}_seed${seed}.log"
  mkdir -p "$RUN_ROOT/$label"
  if [[ -f "$out/summary.json" ]]; then
    echo "SKIP $label seed=$seed"
    return 0
  fi
  if [[ -d "$out" ]]; then
    mv "$out" "${out}.partial.$(date +%s)"
  fi
  echo "START $label seed=$seed $(date -Is)"
  set +e
  if [[ "$strategy" == global_fixed ]]; then
    (cd "$RW" && env PYTHONUNBUFFERED=1 EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="$SITE/nvidia/cusparselt/lib:$LIBS" CUDA_VISIBLE_DEVICES=0 \
      PYTHONPATH="$ROOT/src:$ROOT/scripts:$RW:$RW/XPolicyLab" ROBOTWIN_SUPPRESS_EVAL_CONFIG=1 \
      timeout "${TIMEOUT_SECONDS}s" "$PY" "$ROOT/scripts/run_gate0.py" "${COMMON[@]}" \
      --strategy global_fixed --horizon "$arm" --seeds "$seed" --output-dir "$out") >"$log" 2>&1
  else
    (cd "$RW" && env PYTHONUNBUFFERED=1 EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="$SITE/nvidia/cusparselt/lib:$LIBS" CUDA_VISIBLE_DEVICES=0 \
      PYTHONPATH="$ROOT/src:$ROOT/scripts:$RW:$RW/XPolicyLab" ROBOTWIN_SUPPRESS_EVAL_CONFIG=1 \
      timeout "${TIMEOUT_SECONDS}s" "$PY" "$ROOT/scripts/run_gate0.py" "${COMMON[@]}" \
      --strategy groupwise_fixed \
      --group-horizons "left_arm=$arm,left_gripper=$gripper,right_arm=$arm,right_gripper=$gripper" \
      --seeds "$seed" --output-dir "$out") >"$log" 2>&1
  fi
  local code=$?
  set -e
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$seed" "$code" "$out" "$log" >>"$RUN_ROOT/isolation_status.tsv"
  if [[ $code -eq 0 ]]; then
    echo "DONE $label seed=$seed $(date -Is)"
  else
    echo "ANOMALY $label seed=$seed exit=$code $(date -Is)"
  fi
}

for seed in ${SEEDS//,/ }; do
  run_cell G2 global_fixed 2 2 "$seed"
  run_cell G4 global_fixed 4 4 "$seed"
  run_cell G8 global_fixed 8 8 "$seed"
  run_cell G16 global_fixed 16 16 "$seed"
  run_cell A2G8 groupwise_fixed 2 8 "$seed"
  run_cell A2G16 groupwise_fixed 2 16 "$seed"
  run_cell A4G16 groupwise_fixed 4 16 "$seed"
  run_cell A8G16 groupwise_fixed 8 16 "$seed"
  run_cell A8G2 groupwise_fixed 8 2 "$seed"
  run_cell A16G2 groupwise_fixed 16 2 "$seed"
  run_cell A16G4 groupwise_fixed 16 4 "$seed"
  run_cell A16G8 groupwise_fixed 16 8 "$seed"
done

echo "STAGE_A_ISOLATED_COMPLETE $(date -Is)"

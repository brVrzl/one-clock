#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible detached launcher.  It intentionally stops at the preflight
# when the pinned checkpoint and dataset contracts are incompatible.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ENV_ROOT="/home/xdl/miniforge3/envs/env_isaaclab"
CONDA_ROOT="/home/xdl/miniforge3"
DATASET_ROOT="/home/wjq/datasets/robotwin_unified"
DATASET_REVISION="1287871839fae2296bc27b88a5457c3e1eba8e1f"
POLICY_REVISION="967623a0f38c7e1236c66b3893c830398d793ff7"
POLICY_ROOT="/home/wjq/checkpoints/lerobot_smolvla_robotwin/${POLICY_REVISION}"
CACHE_ROOT="/home/wjq/robotwin_reliability_cache"
RUN_ROOT="${CACHE_ROOT}/${POLICY_REVISION}"
PROGRESS_PATH="${RUN_ROOT}/progress.json"
RUN_LOG="${RUN_ROOT}/run.log"
DEVICE="cuda:0"
SEED="20260820"

mkdir -p "$RUN_ROOT"
exec >> "$RUN_LOG" 2>&1

echo "one-clock RoboTwin detached run started at $(date -Is)"
echo "repo=${REPO_ROOT}"
echo "dataset_root=${DATASET_ROOT} dataset_revision=${DATASET_REVISION}"
echo "policy_root=${POLICY_ROOT} policy_revision=${POLICY_REVISION}"
echo "cache_root=${CACHE_ROOT} device=${DEVICE} seed=${SEED}"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "$ENV_ROOT"
else
  export PATH="${ENV_ROOT}/bin:${PATH}"
fi

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:/home/wjq/workspace/upstreams/XPolicyLab/policy/SmolVLA/smovla/src:${PYTHONPATH:-}"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "blocked: local dataset metadata is absent: ${DATASET_ROOT}/meta/info.json"
  exit 2
fi

SMOKE_OUTPUT="${RUN_ROOT}/smoke_result.json"
SMOKE_ARGS=()
if [[ -f "${POLICY_ROOT}/config.json" ]]; then
  SMOKE_ARGS+=(--checkpoint "$POLICY_ROOT")
fi
set +e
python -m experiments.dynamic_reliability_horizon.robotwin50_dataset.smoke_check \
  --dataset-root "$DATASET_ROOT" \
  --dataset-info "${DATASET_ROOT}/meta/info.json" \
  --dataset-revision "$DATASET_REVISION" \
  --checkpoint-revision "$POLICY_REVISION" \
  "${SMOKE_ARGS[@]}" \
  --output "$SMOKE_OUTPUT" \
  --progress "$PROGRESS_PATH"
SMOKE_RC=$?
set -e

if (( SMOKE_RC != 0 )); then
  echo "blocked: smoke/preflight returned ${SMOKE_RC}; no full cache was started"
  exit "$SMOKE_RC"
fi

if [[ ! -f "${DATASET_ROOT}/meta/stats.json" ]] \
  || [[ -z "$(find "${DATASET_ROOT}/data" -type f -name '*.parquet' -print -quit 2>/dev/null)" ]] \
  || [[ -z "$(find "${DATASET_ROOT}/videos" -type f -name '*.mp4' -print -quit 2>/dev/null)" ]]; then
  echo "blocked: policy-compatible smoke passed but the full local dataset (stats/data/videos) is incomplete"
  exit 2
fi

if [[ ! -f "${POLICY_ROOT}/config.json" || ! -f "${POLICY_ROOT}/model.safetensors" ]]; then
  echo "blocked: compatible local checkpoint weights are absent: ${POLICY_ROOT}"
  exit 2
fi

python -m experiments.dynamic_reliability_horizon.robotwin50_dataset.cache_builder \
  --dataset-root "$DATASET_ROOT" \
  --dataset-repo-id lerobot/robotwin_unified \
  --checkpoint "$POLICY_ROOT" \
  --lerobot-source /home/wjq/workspace/upstreams/XPolicyLab/policy/SmolVLA/smovla \
  --cache-root "$CACHE_ROOT" \
  --progress "$PROGRESS_PATH" \
  --checkpoint-revision "$POLICY_REVISION" \
  --dataset-revision "$DATASET_REVISION" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --threshold 0.05 \
  --retries 2 \
  --retry-seconds 5 \
  --minimum-free-bytes 100000000000

echo "one-clock RoboTwin run ended at $(date -Is)"

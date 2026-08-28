#!/usr/bin/env bash
set -u
cd /home/wjq/workspace/one-clock
root=experiments/component_temporal_reuse/act_confirmation
ckpt=experiments/standard_libero_baselines/act_final/libero_10_task3/checkpoints/100000/pretrained_model
out="$root/libero10_task3.json"
progress="$root/progress/libero10_task3.json"
log="$root/logs/libero10_task3.log"
while [[ ! -f "$ckpt/config.json" || ! -f "$ckpt/model.safetensors" ]]; do
  sleep 60
done
if [[ -f "$out" ]]; then
  exit 0
fi
# The long-task confirmation starts only after the fixed SmolVLA aggregation
# workers have released their research GPU allocation.
while pgrep -f 'run_temporal_aggregation.py --gpu [012]' >/dev/null; do
  sleep 60
done
mkdir -p "$root/logs" "$root/progress"
setsid env CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl HF_HUB_OFFLINE=1 \
  /home/wjq/workspace/venvs/libero_act/bin/python experiments/component_temporal_reuse/run_act_source_confirmation.py \
  --protocol experiments/component_temporal_reuse/act_confirmation_protocol.json \
  --checkpoint "$ckpt" --output "$out" --progress-file "$progress" --gpu 2 --task libero_10:3 \
  >> "$log" 2>&1 < /dev/null

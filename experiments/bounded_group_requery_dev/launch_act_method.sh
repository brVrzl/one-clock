#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <M1_arm_phase|M2_gripper_event|M3_group_event_joint> <gpu>" >&2
  exit 2
fi

METHOD="$1"
GPU="$2"
case "$METHOD" in
  M1_arm_phase|M2_gripper_event|M3_group_event_joint) ;;
  *) echo "unsupported adaptive method: $METHOD" >&2; exit 2 ;;
esac

for task in \
  libero_object:task3 \
  libero_spatial:task0 \
  libero_goal:task2 \
  libero_10:task3
do
  ./launch_act_shard.sh "$task" "$GPU" "$METHOD"
done

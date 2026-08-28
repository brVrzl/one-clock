#!/usr/bin/env bash
set -u
cd /home/wjq/workspace/one-clock
root=experiments/component_temporal_reuse/act_confirmation
while [[ ! -f "$root/object6.json" || ! -f "$root/spatial2.json" || ! -f "$root/goal1.json" ]]; do
  sleep 60
done
inputs=(--input "$root/object6.json" --input "$root/spatial2.json" --input "$root/goal1.json")
run_analysis() {
  /home/wjq/workspace/venvs/libero_act/bin/python experiments/component_temporal_reuse/analyze_act_confirmation.py \
    --protocol "$root/../act_confirmation_protocol.json" "${inputs[@]}" \
    --output-json "$root/analysis.json" --output-report "$root/report.md"
}
run_analysis
while [[ ! -f "$root/libero10_task3.json" ]]; do
  sleep 60
done
inputs+=(--input "$root/libero10_task3.json")
run_analysis

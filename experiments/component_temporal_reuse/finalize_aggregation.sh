#!/usr/bin/env bash
set -u
cd /home/wjq/workspace/one-clock
root=experiments/component_temporal_reuse/aggregation
while [[ ! -f "$root/aggregation_gpu0.json" || ! -f "$root/aggregation_gpu1.json" || ! -f "$root/aggregation_gpu2.json" ]]; do
  sleep 60
done
/home/wjq/workspace/venvs/libero_act/bin/python experiments/component_temporal_reuse/analyze_aggregation.py \
  --protocol experiments/component_temporal_reuse/protocol.json \
  --input "$root/aggregation_gpu0.json" --input "$root/aggregation_gpu1.json" --input "$root/aggregation_gpu2.json" \
  --output-json "$root/analysis.json" --output-report "$root/report.md"

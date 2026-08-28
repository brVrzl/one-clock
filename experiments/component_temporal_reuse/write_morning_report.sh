#!/usr/bin/env bash
set -u
cd /home/wjq/workspace/one-clock
root=experiments/component_temporal_reuse
while [[ ! -f "$root/aggregation/analysis.json" || ! -f "$root/act_confirmation/analysis.json" ]]; do
  sleep 60
done
/home/wjq/workspace/venvs/libero_act/bin/python research/write_overnight_morning_report.py
while [[ ! -f "$root/act_confirmation/libero10_task3.json" ]]; do
  sleep 60
done
/home/wjq/workspace/venvs/libero_act/bin/python research/write_overnight_morning_report.py

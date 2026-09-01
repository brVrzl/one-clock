# ICRA 2027 overnight SmolVLA cross-policy handoff

## Repository state

- Branch: `exp/icra27-overnight-smolvla-crosspolicy`
- Frozen fallback base: `7ea83e1c0bea4367cc722a3d7b72ac0ca827e009`
- Frozen queue commit: `aa23175`
- Worktree: `/home/wjq/workspace/one-clock-icra27-overnight`
- Manifest: `experiments/icra27_overnight_smolvla_crosspolicy/queue_manifest.json`
- Current implementation/audit HEAD at handoff creation: `29ba07c`

The fallback manuscript and fallback branch were not modified.

## Detached workers

Three static shards were launched with `setsid nohup`, one per independent RTX 5080:

- GPU/worker 0: PID/session `2578531`
- GPU/worker 1: PID/session `2578596`
- GPU/worker 2: PID/session `2578606`

PID files: `experiments/icra27_overnight_smolvla_crosspolicy/pids/worker_{0,1,2}.pid`.

The runner uses fresh environments per method/state, durable per-cell JSON, marker validation, and at most two retries after the initial episode attempt. Valid scientific failures are complete and are never retried. Capacity h16 waits on a marker-only barrier requiring all 320 primary cells to be complete or `TECHNICAL_FAILED`; it does not read outcomes.

## Durable paths

- Results: `experiments/icra27_overnight_smolvla_crosspolicy/results/<phase>/*.json`
- Markers: `experiments/icra27_overnight_smolvla_crosspolicy/markers/<phase>/*.{complete,technical_failed}`
- Progress: `experiments/icra27_overnight_smolvla_crosspolicy/progress/worker_{0,1,2}.json`
- Logs: `experiments/icra27_overnight_smolvla_crosspolicy/logs/worker_{0,1,2}.log`
- Attempts: `experiments/icra27_overnight_smolvla_crosspolicy/attempts/<phase>/*.json`
- Preserved no-outcome preflight history: `experiments/icra27_overnight_smolvla_crosspolicy/preflight_failures/`
- SmolVLA smoke: `experiments/icra27_overnight_smolvla_crosspolicy/smoke/smolvla_smoke.json`
- Analysis: `experiments/icra27_overnight_smolvla_crosspolicy/{analysis.json,report.md}`

## Monitor without starting anything

```bash
cd /home/wjq/workspace/one-clock-icra27-overnight
/home/wjq/workspace/venvs/libero_act/bin/python \
  experiments/icra27_overnight_smolvla_crosspolicy/status.py
for f in experiments/icra27_overnight_smolvla_crosspolicy/progress/worker_*.json; do jq -c . "$f"; done
for f in experiments/icra27_overnight_smolvla_crosspolicy/pids/*.pid; do
  pid=$(tr -d '\n' < "$f")
  ps -o pid,ppid,sid,stat,etime,cmd -p "$pid"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

## Exact safe resume

```bash
cd /home/wjq/workspace/one-clock-icra27-overnight
bash experiments/icra27_overnight_smolvla_crosspolicy/resume.sh
```

The launcher refuses a duplicate matching worker. The runner skips only a result that passes identity, status, action-count, source-age-count, query-count, and exact query-schedule validation, or a terminal `TECHNICAL_FAILED` marker.

## Next analysis command

```bash
cd /home/wjq/workspace/one-clock-icra27-overnight
/home/wjq/workspace/venvs/libero_act/bin/python \
  experiments/icra27_overnight_smolvla_crosspolicy/analyze.py
```

Then inspect `analysis.json`, update the exposure-inventory SmolVLA row from `PROTOCOL_ONLY` to `OUTCOME_EXPOSED` for completed cells, commit scientific results/markers/progress/logs/analysis/handoff, and push this branch. Do not edit the manuscript and do not launch any experiment outside the frozen manifest.

## Frozen stop conditions

After all 160 `smolvla_capacity_h16` cells reach a terminal marker, stop. Do not launch any additional coherent horizon, groupwise cell, RoboTwin rollout, model, training, or adaptive method.

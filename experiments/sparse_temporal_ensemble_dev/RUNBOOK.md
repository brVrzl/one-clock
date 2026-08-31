# Sparse temporal ensemble development runbook

This experiment compares hard newest-chunk execution with temporal ensembling at identical sparse query schedules for ACT and SmolVLA. It uses only the four exposed development tasks and state IDs 10--19.

## Required order

1. Run the CPU semantic tests.
2. Run one task/state through all four methods for ACT.
3. Run the paired SmolVLA raw-chunk check, then one task/state through all four methods.
4. Launch full resumable shards only after the affected smoke passes.
5. Run analysis only after all completion markers exist.

## Shards and paths

ACT runs the four task shards serially on GPU 0 with:

```bash
bash act/run_panel.sh 0
```

For each task slug `<suite>_task<N>`, the log is `act/logs/<slug>.log`, the output is `act/results/<slug>.json`, and the completion marker is `act/markers/<slug>.complete`.

SmolVLA uses the same atomic per-task queue from each available GPU:

```bash
bash smolvla/queue_worker.sh 1
bash smolvla/queue_worker.sh 2
bash smolvla/queue_worker.sh 0  # launched after ACT if unclaimed work remains
```

For each task slug, the log is `smolvla/logs/<slug>.log`, the output is `smolvla/results/<slug>.json`, and the completion marker is `smolvla/markers/<slug>.complete`. The actual GPU is also stored in the result. Atomic claim directories prevent duplicate task execution.

A completion marker is written only after a runner exits successfully and `validate_result.py` verifies all ten episodes per method, schedules, provenance, weights, action dimensions, and paired common prefixes.

Launch or resume all work with:

```bash
bash resume.sh
```

`resume.sh` is idempotent: completed valid shards are skipped, stale queue claims are released, missing or failed workers are restarted, GPU 0 is assigned unfinished Smol work after ACT, and analysis runs once every required shard is complete. Unvalidated partial artifacts are moved under the policy's `invalidated/` directory before a restart.

## Protocol failures

Stop only affected jobs, invalidate only their artifacts, rerun the affected unit test and one-state smoke, then restart those shards. Do not combine artifacts produced under different query schedules, RNG rules, task/state/seed mappings, runtimes, or executor semantics.

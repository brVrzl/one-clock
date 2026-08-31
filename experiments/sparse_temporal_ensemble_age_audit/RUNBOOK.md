# Repaired ACT h16 runbook

The panel contains exactly three methods at policy-query cadence 16: `hard_h16`, `candidate_index_te_h16`, and `dense_equivalent_te_h16`. Every method/state episode constructs a fresh environment under the same method-independent environment seed. No h8, SmolVLA, or blind task is included.

Initial detached launch:

```bash
experiments/sparse_temporal_ensemble_age_audit/resume.sh
```

This assigns task10 to GPU 0, object to GPU 1, and spatial to GPU 2. After the first object/spatial shard completes, launch goal on that freed GPU:

```bash
experiments/sparse_temporal_ensemble_age_audit/resume.sh libero_goal:task2 GPU_INDEX
```

Each task has one log, result JSON, progress JSON, and validated completion marker under `act_h16/`. `resume.sh` skips a completed marker and an active matching runner. `run_task.sh` validates the full same-target/query/weight semantics and exact t=0--15 cross-method prefix before writing a completion marker.

Failed diagnostics and the historical candidate-index experiment remain separate. Do not combine historical aggregate outcomes with this repaired trio.

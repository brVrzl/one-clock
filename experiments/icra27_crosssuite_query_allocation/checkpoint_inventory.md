# Local ACT checkpoint bank inventory

Audit time (UTC): `2026-09-02T07:58:34.840481+00:00`

The local filesystem contains all 40 expected task-specific ACT 100k exports. The audit constructed each policy and its saved preprocessing/postprocessing pipeline on CPU; no environment was initialized and no rollout outcome was generated.

No checkpoint hashes were computed. Exact task-specific paths, step, byte size, and file mtime disambiguate these local exports.

## Summary

- Expected task policies: 40
- Technically valid task policies: 40
- Expected non-Object confirmation policies: 30
- Valid non-Object confirmation policies: 30
- Valid Object development/reference policies: 10
- Track-A checkpoint-bank gate: **PROCEED_ALL_30**

All selected policies use seven-dimensional actions, 100-step chunks, checkpoint-frozen `MEAN_STD` action statistics, and native temporal ensembling disabled in the saved config.

## Per-policy inventory

| Suite | Task | Role | Step | Action dim | Chunk | Export bytes | Standard baseline | CPU load smoke | Valid | Exact local path |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
| `libero_spatial` | 0 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740413 | 5/10 (50.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task0/checkpoints/100000/pretrained_model` |
| `libero_spatial` | 1 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740413 | 6/10 (60.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task1/checkpoints/100000/pretrained_model` |
| `libero_spatial` | 2 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740431 | 9/10 (90.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task2/checkpoints/100000/pretrained_model` |
| `libero_spatial` | 3 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740377 | 6/10 (60.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task3/checkpoints/100000/pretrained_model` |
| `libero_spatial` | 4 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740359 | 3/10 (30.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task4/checkpoints/100000/pretrained_model` |
| `libero_spatial` | 5 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740305 | 8/10 (80.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task5/checkpoints/100000/pretrained_model` |
| `libero_spatial` | 6 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740431 | 8/10 (80.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task6/checkpoints/100000/pretrained_model` |
| `libero_spatial` | 7 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740233 | 7/10 (70.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task7/checkpoints/100000/pretrained_model` |
| `libero_spatial` | 8 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740449 | 5/10 (50.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task8/checkpoints/100000/pretrained_model` |
| `libero_spatial` | 9 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740395 | 7/10 (70.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task9/checkpoints/100000/pretrained_model` |
| `libero_object` | 0 | development_reference | 100000 | 7 | 100 | 206740374 | 2/10 (20.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task0/checkpoints/100000/pretrained_model` |
| `libero_object` | 1 | development_reference | 100000 | 7 | 100 | 206740391 | 5/10 (50.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task1/checkpoints/100000/pretrained_model` |
| `libero_object` | 2 | development_reference | 100000 | 7 | 100 | 206740426 | 5/10 (50.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task2/checkpoints/100000/pretrained_model` |
| `libero_object` | 3 | development_reference | 100000 | 7 | 100 | 206740406 | 2/10 (20.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task3/checkpoints/100000/pretrained_model` |
| `libero_object` | 4 | development_reference | 100000 | 7 | 100 | 206740390 | 6/10 (60.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task4/checkpoints/100000/pretrained_model` |
| `libero_object` | 5 | development_reference | 100000 | 7 | 100 | 206740342 | 5/10 (50.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task5/checkpoints/100000/pretrained_model` |
| `libero_object` | 6 | development_reference | 100000 | 7 | 100 | 206740388 | 8/10 (80.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task6/checkpoints/100000/pretrained_model` |
| `libero_object` | 7 | development_reference | 100000 | 7 | 100 | 206740390 | 5/10 (50.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task7/checkpoints/100000/pretrained_model` |
| `libero_object` | 8 | development_reference | 100000 | 7 | 100 | 206740476 | 7/10 (70.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task8/checkpoints/100000/pretrained_model` |
| `libero_object` | 9 | development_reference | 100000 | 7 | 100 | 206740396 | 3/10 (30.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task9/checkpoints/100000/pretrained_model` |
| `libero_goal` | 0 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740325 | 5/10 (50.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task0/checkpoints/100000/pretrained_model` |
| `libero_goal` | 1 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740410 | 10/10 (100.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task1/checkpoints/100000/pretrained_model` |
| `libero_goal` | 2 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740393 | 10/10 (100.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task2/checkpoints/100000/pretrained_model` |
| `libero_goal` | 3 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740206 | 5/10 (50.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task3/checkpoints/100000/pretrained_model` |
| `libero_goal` | 4 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740376 | 9/10 (90.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task4/checkpoints/100000/pretrained_model` |
| `libero_goal` | 5 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740155 | 3/10 (30.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task5/checkpoints/100000/pretrained_model` |
| `libero_goal` | 6 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740274 | 3/10 (30.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task6/checkpoints/100000/pretrained_model` |
| `libero_goal` | 7 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740444 | 9/10 (90.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task7/checkpoints/100000/pretrained_model` |
| `libero_goal` | 8 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740427 | 9/10 (90.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task8/checkpoints/100000/pretrained_model` |
| `libero_goal` | 9 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740206 | 7/10 (70.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task9/checkpoints/100000/pretrained_model` |
| `libero_10` | 0 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740139 | 0/10 (0.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task0/checkpoints/100000/pretrained_model` |
| `libero_10` | 1 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740408 | 7/10 (70.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task1/checkpoints/100000/pretrained_model` |
| `libero_10` | 2 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740272 | 5/10 (50.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task2/checkpoints/100000/pretrained_model` |
| `libero_10` | 3 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740173 | 3/10 (30.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task3/checkpoints/100000/pretrained_model` |
| `libero_10` | 4 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740226 | 2/10 (20.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task4/checkpoints/100000/pretrained_model` |
| `libero_10` | 5 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740278 | 9/10 (90.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task5/checkpoints/100000/pretrained_model` |
| `libero_10` | 6 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740187 | 4/10 (40.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task6/checkpoints/100000/pretrained_model` |
| `libero_10` | 7 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740305 | 3/10 (30.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task7/checkpoints/100000/pretrained_model` |
| `libero_10` | 8 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740070 | 1/10 (10.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task8/checkpoints/100000/pretrained_model` |
| `libero_10` | 9 | primary_confirmation_candidate | 100000 | 7 | 100 | 206740155 | 7/10 (70.0%) | PASS | yes | `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task9/checkpoints/100000/pretrained_model` |

## Contract and provenance notes

- The selected export for every task is `checkpoints/100000/pretrained_model`; the earlier 20k, 40k, 60k, and 80k snapshots are also present but are not selected.
- `config.json` and `train_config.json` independently record ACT, seven action dimensions, chunk size 100, and training step 100,000.
- The preprocessor applies the saved normalizer after batching/device placement. The postprocessor applies the saved action unnormalizer before moving results to CPU.
- Both saved processor state files open successfully and expose seven-element `action.mean` and `action.std` tensors for every policy.
- The prior standard-baseline figures are transcribed from each task directory's existing `eval10/eval_info.json`; they are not executor-variant results and were not rerun by this audit.
- Object policies are recorded as development/reference only. They are excluded from the Track-A primary confirmation bank.

## Audit environment

- Python: `3.12.3`
- PyTorch: `2.11.0+cu130`
- LeRobot: `0.6.2`
- Load device: `cpu` (CUDA hidden for the audit command)

# Object executor decomposition

This is a zero-rollout analysis of the exact existing Object development outcomes. The two protocols match on the listed compatibility checks, so the 126 paired task-state blocks are combined.

## Paired contrasts

| Contrast | First successes | Second successes | First-only | Second-only | Net | Delta (pp) | Exact McNemar p | Paired 95% CI | Cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 H16Arm+FreshGrip vs Fresh | 42 | 56 | 8 | 22 | -14 | -11.11 | 0.0161248 | [-0.198, -0.032] | [-0.190, -0.040] |
| hard h16 vs Fresh | 88 | 56 | 36 | 4 | 32 | 25.40 | 1.85702e-07 | [0.167, 0.341] | [0.159, 0.349] |
| C1 PreviousChunkGrip vs Fresh | 64 | 56 | 19 | 11 | 8 | 6.35 | 0.200488 | [-0.024, 0.151] | [-0.032, 0.183] |

## Mean-effect decomposition

HARD_H16 − FRESH = 0.253968 = (HARD_H16 − C2) +0.365079 + (C2 − FRESH) -0.111111.

This is an arithmetic decomposition of paired mean success differences, not a mediation or percentage attribution analysis.

## Per-task counts and leave-one-task-out deltas

| Task | Fresh | C2 | hard h16 | C1 | C2−Fresh | hard−Fresh | C1−Fresh |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2/14 | 2/14 | 8/14 | 4/14 | 0 | 6 | 2 |
| 2 | 10/14 | 7/14 | 13/14 | 10/14 | -3 | 3 | 0 |
| 3 | 6/14 | 6/14 | 11/14 | 9/14 | 0 | 5 | 3 |
| 4 | 4/14 | 4/14 | 10/14 | 10/14 | 0 | 6 | 6 |
| 5 | 7/14 | 3/14 | 11/14 | 6/14 | -4 | 4 | -1 |
| 6 | 7/14 | 5/14 | 8/14 | 6/14 | -2 | 1 | -1 |
| 7 | 3/14 | 3/14 | 8/14 | 3/14 | 0 | 5 | 0 |
| 8 | 6/14 | 2/14 | 7/14 | 4/14 | -4 | 1 | -2 |
| 9 | 11/14 | 10/14 | 12/14 | 12/14 | -1 | 1 | 1 |

| Omitted task | C2−Fresh | hard−Fresh | C1−Fresh |
|---:|---:|---:|---:|
| 1 | -0.1250 | 0.2321 | 0.0536 |
| 2 | -0.0982 | 0.2589 | 0.0714 |
| 3 | -0.1250 | 0.2411 | 0.0446 |
| 4 | -0.1250 | 0.2321 | 0.0179 |
| 5 | -0.0893 | 0.2500 | 0.0804 |
| 6 | -0.1071 | 0.2768 | 0.0804 |
| 7 | -0.1250 | 0.2411 | 0.0714 |
| 8 | -0.0893 | 0.2768 | 0.0893 |
| 9 | -0.1161 | 0.2768 | 0.0625 |

## Compatibility checks

```json
{
  "factorial_protocol": "/home/wjq/workspace/one-clock/experiments/group_delay_factorial_act20/protocol.json",
  "asym_protocol": "/home/wjq/workspace/one-clock/experiments/asymmetric_chunk_reuse_dev/protocol.json",
  "factorial_commit": "7ab52cbc6360ae8436cfe5a04f8d200130d3f7a4",
  "asym_commit": "4cf1cbf97411e0cd7face0974c26adc1b25de37d",
  "checks": {
    "tasks": true,
    "state_ids": true,
    "environment_seed_rule": true,
    "full_per_task_seed_list": true,
    "act_checkpoint": true,
    "checkpoint_chunk_size": true,
    "observation_preprocessing": true,
    "control_mode": true,
    "control_frequency_hz": true,
    "max_episode_steps": true,
    "success_criterion": true,
    "fresh_environment_reset_protocol": true,
    "policy_deterministic_inference_settings": true,
    "source_executor": true,
    "existing_results_are_fresh_env": true
  },
  "compatible": true
}
```

Interpretation is conditional on these paired-result comparisons. No new episodes were run.

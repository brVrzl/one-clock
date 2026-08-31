# Gate-4A2 Spatial ACT reanalysis

## Results

This is a zero-rollout reanalysis of existing logs from the preregistered, completed LIBERO Spatial rollout. The dataset was preregistered and completed before the final cross-suite confirmation.

| Method | Success | Success % |
|---|---:|---:|
| Fresh | 40/100 | 40.0% |
| FO20 | 40/100 | 40.0% |
| FullOld20 | 30/100 | 30.0% |
| AGE_EXP_B003 | 42/100 | 42.0% |
| COGACT_A03 | 18/100 | 18.0% |

## Paired contrasts

| Contrast | First-only | Second-only | Net | Delta (pp) | Exact two-sided McNemar p | Paired 95% CI | Task-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| FO20 vs Fresh | 5 | 5 | 0 | 0.00 | 1 | [-0.060, 0.060] | [-0.050, 0.060] |
| FO20 vs FullOld20 | 17 | 7 | 10 | 10.00 | 0.0639147 | [0.010, 0.190] | [0.020, 0.190] |

## Per-task success

| Task | Fresh | FO20 | FullOld20 | AGE_EXP_B003 | COGACT_A03 | FO20−Fresh | FO20−FullOld20 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8/10 | 7/10 | 7/10 | 6/10 | 1/10 | -1 | 0 |
| 1 | 4/10 | 4/10 | 3/10 | 4/10 | 0/10 | 0 | 1 |
| 2 | 9/10 | 9/10 | 5/10 | 7/10 | 4/10 | 0 | 4 |
| 3 | 7/10 | 8/10 | 7/10 | 8/10 | 6/10 | 1 | 1 |
| 4 | 3/10 | 2/10 | 1/10 | 1/10 | 1/10 | -1 | 1 |
| 5 | 0/10 | 0/10 | 1/10 | 0/10 | 0/10 | 0 | -1 |
| 6 | 7/10 | 6/10 | 5/10 | 9/10 | 3/10 | -1 | 1 |
| 7 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0 | 0 |
| 8 | 2/10 | 4/10 | 1/10 | 5/10 | 2/10 | 2 | 3 |
| 9 | 0/10 | 0/10 | 0/10 | 2/10 | 1/10 | 0 | 0 |

## Leave-one-task-out

| Omitted task | FO20−Fresh | FO20−FullOld20 |
|---:|---:|---:|
| 0 | 0.0111 | 0.1111 |
| 1 | 0.0000 | 0.1000 |
| 2 | 0.0000 | 0.0667 |
| 3 | -0.0111 | 0.1000 |
| 4 | 0.0111 | 0.1000 |
| 5 | 0.0000 | 0.1222 |
| 6 | 0.0111 | 0.1000 |
| 7 | 0.0000 | 0.1111 |
| 8 | -0.0222 | 0.0778 |
| 9 | 0.0000 | 0.1111 |

## Scope limitations

This dataset was preregistered and completed before the final cross-suite confirmation.

It contains no Reverse20 and therefore cannot establish the full arm-vs-gripper factorial asymmetry.

It contains no hard-h16 practical baseline.

Its suite-level checkpoint differs from the final confirmation checkpoint family, so absolute success rates are not compared across experiments.

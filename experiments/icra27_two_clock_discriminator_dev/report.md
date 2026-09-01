# ICRA 2027 two-clock discriminator development result

This is a development-only comparison on the already exposed 126-block LIBERO Object cohort. H16 and C1 are exact historical reuses; H32 and true arm16/grip32 are the only new rollout conditions.

Descriptive interpretation: **H16_REMAINS_BEST**. Coherent H16 matched or exceeded both new conditions.

## Main results

| Method | Status | Success /126 | Success % | Queries | Env steps | Query rate | Mean arm age | Mean grip age | Arm age range | Grip age range |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H16_COHERENT | historical reuse | 88/126 | 69.8% | 1490 | 22869 | 0.06515 | 7.358 | 7.358 | 0–15 | 0–15 |
| H32_COHERENT | new rollout | 76/126 | 60.3% | 843 | 25581 | 0.03295 | 15.042 | 15.042 | 0–31 | 0–31 |
| TWO_CLOCK_ARM16_GRIP32 | new rollout | 78/126 | 61.9% | 1566 | 24110 | 0.06495 | 7.364 | 14.999 | 0–15 | 0–31 |
| C1_PREVIOUS_CHUNK_GRIP | historical reuse | 64/126 | 50.8% | 1652 | 25555 | 0.06464 | 7.370 | 22.108 | 0–15 | 0–31 |

## Primary paired contrasts

| Contrast | First-only | Second-only | Net | Delta (pp) | Exact McNemar p | Paired 95% CI | Task-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| TWO_CLOCK_ARM16_GRIP32 vs H16_COHERENT | 4 | 14 | -10 | -7.94 | 0.0308838 | [-0.143, -0.016] | [-0.151, -0.016] |
| TWO_CLOCK_ARM16_GRIP32 vs H32_COHERENT | 18 | 16 | 2 | 1.59 | 0.864166 | [-0.079, 0.103] | [-0.103, 0.127] |
| H32_COHERENT vs H16_COHERENT | 13 | 25 | -12 | -9.52 | 0.0729514 | [-0.190, 0.000] | [-0.190, 0.000] |

McNemar p-values are descriptive because this is development.

## Rescue and regression

- Against historical H16: 4 H16 failures rescued; 14 H16 successes regressed.
- Against coherent H32: 18 H32 failures rescued; 16 H32 successes regressed.

## Per-task success counts

| Task | H16 | H32 | TWO_CLOCK | C1 context |
|---:|---:|---:|---:|---:|
| 1 | 8/14 | 10/14 | 7/14 | 4/14 |
| 2 | 13/14 | 10/14 | 13/14 | 10/14 |
| 3 | 11/14 | 10/14 | 11/14 | 9/14 |
| 4 | 10/14 | 6/14 | 10/14 | 10/14 |
| 5 | 11/14 | 12/14 | 8/14 | 6/14 |
| 6 | 8/14 | 7/14 | 6/14 | 6/14 |
| 7 | 8/14 | 4/14 | 4/14 | 3/14 |
| 8 | 7/14 | 5/14 | 6/14 | 4/14 |
| 9 | 12/14 | 12/14 | 13/14 | 12/14 |

## Leave-one-task-out deltas

| Omitted task | TWO_CLOCK−H16 | TWO_CLOCK−H32 | H32−H16 |
|---:|---:|---:|---:|
| 1 | -0.0804 | 0.0446 | -0.1250 |
| 2 | -0.0893 | -0.0089 | -0.0804 |
| 3 | -0.0893 | 0.0089 | -0.0982 |
| 4 | -0.0893 | -0.0179 | -0.0714 |
| 5 | -0.0625 | 0.0536 | -0.1161 |
| 6 | -0.0714 | 0.0268 | -0.0982 |
| 7 | -0.0536 | 0.0179 | -0.0714 |
| 8 | -0.0804 | 0.0089 | -0.0893 |
| 9 | -0.0982 | 0.0089 | -0.1071 |

The semantic smoke passed before the full rollout. No additional horizon, adaptive gate, confirmation task, RoboTwin, pi0/pi0.5, SmolVLA, or real-robot experiment was launched.

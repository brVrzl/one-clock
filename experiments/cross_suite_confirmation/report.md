# Branch K cross-suite confirmation

## Scope and interpretation

The confirmation tasks are unseen to executor development, not unseen to policy training: each per-task ACT checkpoint was trained for its corresponding task. Absolute success rates are therefore interpreted only within the confirmation experiment, and scientific inference uses paired executor contrasts.

States 0..13 were selected by the deterministic outcome-independent rule `first 14 initialization states`; this does not claim that these numerical state IDs were globally unused.

These are five frozen conditions: Fresh, FO20, Reverse20, FullOld20, and hard h16. The Branch K confirmation used 910 episodes: 140 primary blocks per method (Goal + LIBERO-10) and 42 exposed Object bridge blocks per method.

## Checkpoint preflight and cohort

The per-task 100k ACT checkpoint family was loadable for all 13 selected tasks. No missing-checkpoint contingency was invoked. The actual primary cohort is Goal tasks 4, 6, 7, 8, 9 and LIBERO-10 tasks 0, 2, 4, 6, 7, each with states 0..13 and the frozen seed rule `340000 + 1000*suite_index + 100*task_id + state_id`. The Object bridge cohort is tasks 1, 5, 9.

## Semantic and rollout validation

The CPU analyzer checked 910 episodes and every persisted step. It verified `source_q + offset = target_t`, the four fixed-source definitions, fixed-source Fresh prefixes through t=19, hard-h16 query steps 0,16,32,... with newest-chunk offsets, sequential targets, finite 7D actions, exact frozen seeds/caps, and fresh-environment metadata.

The pre-outcome semantic suite passed 3/3 tests and the required pairing smoke passed for one Goal task x3 states and one LIBERO-10 task x3 states before rollout. All three outcome shards completed without interruption; no C1/C2 or HARD_H16 rerun was performed.

## Primary unseen-to-executor-development aggregate

| Method | Success | Success % | Observed query rate | Observed mean gripper age |
|---|---:|---:|---:|---:|
| Fresh | 77/140 | 55.0% | 1.00000 | 0.000 |
| FO20 | 83/140 | 59.3% | 1.00000 | 18.549 |
| Reverse20 | 38/140 | 27.1% | 1.00000 | 0.000 |
| FullOld20 | 66/140 | 47.1% | 1.00000 | 18.690 |
| hard h16 | 93/140 | 66.4% | 0.06430 | 7.399 |

The observed query rate is total policy calls divided by total environment steps within the indicated 140-block primary aggregate. Gripper age is step-weighted over the realized episode trajectories.

## Primary paired contrasts

| Contrast | First success | Second success | First-only | Second-only | Net | Delta (pp) | Exact two-sided McNemar p | Paired 95% CI | Task-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FO20 vs Reverse20 | 83/140 | 38/140 | 48 | 3 | 45 | 32.14 | 1.96749e-11 | [0.236, 0.407] | [0.214, 0.443] |
| FO20 vs Fresh | 83/140 | 77/140 | 12 | 6 | 6 | 4.29 | 0.237885 | [-0.014, 0.100] | [-0.014, 0.107] |
| FO20 vs FullOld20 | 83/140 | 66/140 | 28 | 11 | 17 | 12.14 | 0.0094753 | [0.036, 0.207] | [-0.021, 0.286] |

McNemar p-values are exact two-sided binomial tests on discordant paired task-state blocks. Bootstrap intervals use the preregistered 20,000 draws and seeds. Task-cluster intervals resample task-level mean differences; leave-one-task-out values are reported below.

### Primary per-task results

| Task | Fresh | FO20 | Reverse20 | FullOld20 | hard h16 | FO20−Reverse20 | FO20−Fresh | FO20−FullOld20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| libero_goal:task4 | 13/14 | 12/14 | 8/14 | 12/14 | 11/14 | 4 | -1 | 0 |
| libero_goal:task6 | 10/14 | 10/14 | 3/14 | 2/14 | 8/14 | 7 | 0 | 8 |
| libero_goal:task7 | 14/14 | 14/14 | 13/14 | 13/14 | 14/14 | 1 | 0 | 1 |
| libero_goal:task8 | 13/14 | 14/14 | 9/14 | 6/14 | 14/14 | 5 | 1 | 8 |
| libero_goal:task9 | 8/14 | 9/14 | 3/14 | 12/14 | 11/14 | 6 | 1 | -3 |
| libero_10:task0 | 2/14 | 3/14 | 0/14 | 1/14 | 4/14 | 3 | 1 | 2 |
| libero_10:task2 | 8/14 | 12/14 | 2/14 | 10/14 | 13/14 | 10 | 4 | 2 |
| libero_10:task4 | 3/14 | 2/14 | 0/14 | 1/14 | 9/14 | 2 | -1 | 1 |
| libero_10:task6 | 5/14 | 5/14 | 0/14 | 6/14 | 4/14 | 5 | 0 | -1 |
| libero_10:task7 | 1/14 | 2/14 | 0/14 | 3/14 | 5/14 | 2 | 1 | -1 |

### Primary leave-one-task-out deltas

| Omitted task | FO20_VS_REVERSE20 | FO20_VS_FRESH | FO20_VS_FULL_OLD20 |
|---|---:|---:|---:|
| libero_goal:task4 | 0.3254 | 0.0556 | 0.1349 |
| libero_goal:task6 | 0.3016 | 0.0476 | 0.0714 |
| libero_goal:task7 | 0.3492 | 0.0476 | 0.1270 |
| libero_goal:task8 | 0.3175 | 0.0397 | 0.0714 |
| libero_goal:task9 | 0.3095 | 0.0397 | 0.1587 |
| libero_10:task0 | 0.3333 | 0.0397 | 0.1190 |
| libero_10:task2 | 0.2778 | 0.0159 | 0.1190 |
| libero_10:task4 | 0.3413 | 0.0556 | 0.1270 |
| libero_10:task6 | 0.3175 | 0.0476 | 0.1429 |
| libero_10:task7 | 0.3413 | 0.0397 | 0.1429 |

## Object bridge context

Object tasks 1, 5, and 9 are exposed bridge tasks only and are not pooled into the primary inference set.

| Method | Success | Success % | Observed query rate | Observed mean gripper age |
|---|---:|---:|---:|---:|
| Fresh | 20/42 | 47.6% | 1.00000 | 0.000 |
| FO20 | 21/42 | 50.0% | 1.00000 | 18.067 |
| Reverse20 | 8/42 | 19.0% | 1.00000 | 0.000 |
| FullOld20 | 20/42 | 47.6% | 1.00000 | 18.087 |
| hard h16 | 33/42 | 78.6% | 0.06524 | 7.373 |

### Bridge paired contrasts (descriptive)

| Contrast | First success | Second success | First-only | Second-only | Net | Delta (pp) | Exact two-sided McNemar p | Paired 95% CI | Task-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FO20 vs Reverse20 | 21/42 | 8/42 | 15 | 2 | 13 | 30.95 | 0.00234985 | [0.143, 0.476] | [0.214, 0.429] |
| FO20 vs Fresh | 21/42 | 20/42 | 4 | 3 | 1 | 2.38 | 1 | [-0.095, 0.143] | [0.000, 0.071] |
| FO20 vs FullOld20 | 21/42 | 20/42 | 7 | 6 | 1 | 2.38 | 1 | [-0.143, 0.190] | [-0.143, 0.143] |

### Bridge per-task results

| Task | Fresh | FO20 | Reverse20 | FullOld20 | hard h16 | FO20−Reverse20 | FO20−Fresh | FO20−FullOld20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| libero_object:task1 | 5/14 | 6/14 | 0/14 | 5/14 | 11/14 | 6 | 1 | 1 |
| libero_object:task5 | 5/14 | 5/14 | 2/14 | 7/14 | 13/14 | 3 | 0 | -2 |
| libero_object:task9 | 10/14 | 10/14 | 6/14 | 8/14 | 9/14 | 4 | 0 | 2 |

### Bridge leave-one-task-out deltas

| Omitted task | FO20_VS_REVERSE20 | FO20_VS_FRESH | FO20_VS_FULL_OLD20 |
|---|---:|---:|---:|
| libero_object:task1 | 0.2500 | 0.0000 | 0.0000 |
| libero_object:task5 | 0.3571 | 0.0357 | 0.1071 |
| libero_object:task9 | 0.3214 | 0.0357 | -0.0357 |

## Spatial context

The completed preregistered Gate-4A2 Spatial reanalysis is reported in `experiments/gate4a2_spatial_analysis/report.md`. It is independent second-suite context for FO20, but it has no Reverse20 and no hard-h16 baseline, so it cannot establish the full arm-versus-gripper factorial asymmetry. Its suite-level checkpoint also differs from this confirmation checkpoint family, so absolute success rates are not compared across experiments.

## Paper artifact registration

The `{C2, hard h16, C1}` three-point gripper-age table from the completed development gate remains the executor-decomposition artifact. This Branch K table is the frozen confirmation artifact for the five-condition same-target comparison, with primary inference restricted to the unseen-to-executor-development Goal + LIBERO-10 cohort and Object reported separately as bridge context.

Method development is closed after this confirmation. Negative or heterogeneous results are retained as paper results; no rescue executor or additional rollout is authorized by this protocol.

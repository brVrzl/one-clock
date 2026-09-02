# ICRA 2027 overnight fixed-clock results harvest

## Completion

Completion count first: **800/800 frozen overnight cells complete**. Including the already-completed 126-cell ACT discriminator phase, the full manifest is **926/926 complete**, with **0 TECHNICAL_FAILED**, **0 pending**, and **0 running**.

## A. ACT-B: post-hoc 140-block H8 audit

This is a **post-hoc coherent-baseline audit**, not new confirmation. H8 succeeded on **100/140**; historical H16 succeeded on **93/140**.
- H8 vs historical H16 (H8-only:H16-only): first-only `15`, second-only `8`, ties `117`, delta `+5.00 pp`, exact two-sided McNemar `p=0.21004`, paired bootstrap 95% CI `[-1.43, +12.14] pp`, task-cluster bootstrap 95% CI `[-3.57, +12.86] pp`.
- Interpretation: `COHERENT_OPTIMUM_IS_NOT_H16`. The label `COHERENT_OPTIMUM_IS_NOT_H16` applies.

| Task | H8 | H16 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_10:task0` | 8 | 4 | 14 | +28.57 | +2.38 |
| `libero_10:task2` | 13 | 13 | 14 | +0.00 | +5.56 |
| `libero_10:task4` | 6 | 9 | 14 | -21.43 | +7.94 |
| `libero_10:task6` | 6 | 4 | 14 | +14.29 | +3.97 |
| `libero_10:task7` | 5 | 5 | 14 | +0.00 | +5.56 |
| `libero_goal:task4` | 14 | 11 | 14 | +21.43 | +3.17 |
| `libero_goal:task6` | 8 | 8 | 14 | +0.00 | +5.56 |
| `libero_goal:task7` | 14 | 14 | 14 | +0.00 | +5.56 |
| `libero_goal:task8` | 14 | 14 | 14 | +0.00 | +5.56 |
| `libero_goal:task9` | 12 | 11 | 14 | +7.14 | +4.76 |

## B. ACT-C: ARM4_GRIP32

ARM4_GRIP32 succeeded on **131/180**; historical references are ARM4_GRIP16 **128/180** and ARM4_GRIP4 **112/180**.
- ARM4_GRIP32 vs ARM4_GRIP16: first-only `20`, second-only `17`, ties `143`, delta `+1.67 pp`, exact two-sided McNemar `p=0.742829`, paired bootstrap 95% CI `[-5.00, +8.33] pp`, task-cluster bootstrap 95% CI `[-7.78, +11.11] pp`.

| Task | GRIP32 | GRIP16 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_object:task1` | 17 | 12 | 20 | +25.00 | -1.25 |
| `libero_object:task2` | 18 | 16 | 20 | +10.00 | +0.62 |
| `libero_object:task3` | 17 | 16 | 20 | +5.00 | +1.25 |
| `libero_object:task4` | 16 | 16 | 20 | +0.00 | +1.88 |
| `libero_object:task5` | 15 | 17 | 20 | -10.00 | +3.12 |
| `libero_object:task6` | 10 | 11 | 20 | -5.00 | +2.50 |
| `libero_object:task7` | 8 | 13 | 20 | -25.00 | +5.00 |
| `libero_object:task8` | 11 | 7 | 20 | +20.00 | -0.62 |
| `libero_object:task9` | 19 | 20 | 20 | -5.00 | +2.50 |

- ARM4_GRIP32 vs ARM4_GRIP4: first-only `32`, second-only `13`, ties `135`, delta `+10.56 pp`, exact two-sided McNemar `p=0.00660882`, paired bootstrap 95% CI `[+3.33, +17.78] pp`, task-cluster bootstrap 95% CI `[+2.22, +19.44] pp`.

| Task | GRIP32 | GRIP4 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_object:task1` | 17 | 10 | 20 | +35.00 | +7.50 |
| `libero_object:task2` | 18 | 15 | 20 | +15.00 | +10.00 |
| `libero_object:task3` | 17 | 15 | 20 | +10.00 | +10.62 |
| `libero_object:task4` | 16 | 16 | 20 | +0.00 | +11.88 |
| `libero_object:task5` | 15 | 10 | 20 | +25.00 | +8.75 |
| `libero_object:task6` | 10 | 10 | 20 | +0.00 | +11.88 |
| `libero_object:task7` | 8 | 10 | 20 | -10.00 | +13.12 |
| `libero_object:task8` | 11 | 8 | 20 | +15.00 | +10.00 |
| `libero_object:task9` | 19 | 18 | 20 | +5.00 | +11.25 |

Execution totals and source ages (mean/p95/max):

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP32 | 131/180 | 72.78% | 31695 | 7972 | 25.152% | 8.692 | 1.495/3.0/3 | 14.979/30.0/31 |

Plateau criterion `abs(successes_ARM4_GRIP32 - 128) <= 3`: **met**. Label: `GRIPPER_PLATEAU_AT_16`. No grip64 or other grid cell was run.

## C. SmolVLA primary

All **320/320 primary episodes** (160 paired blocks) completed.

### Spatial

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COHERENT_H8 | 23/40 | 57.50% | 7104 | 900 | 12.669% | 13.792 | 3.482/7.0/7 | 3.482/7.0/7 |
| ARM8_GRIP16 | 24/40 | 60.00% | 6997 | 886 | 12.663% | 13.669 | 3.482/7.0/7 | 7.367/15.0/15 |

- ARM8_GRIP16 vs COHERENT_H8: first-only `1`, second-only `0`, ties `39`, delta `+2.50 pp`, exact two-sided McNemar `p=1`, paired bootstrap 95% CI `[+0.00, +7.50] pp`, task-cluster bootstrap 95% CI `[+0.00, +7.50] pp`.

| Task | ARM8_GRIP16 | H8 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_spatial:task0` | 1 | 1 | 4 | +0.00 | +2.78 |
| `libero_spatial:task1` | 3 | 3 | 4 | +0.00 | +2.78 |
| `libero_spatial:task2` | 3 | 3 | 4 | +0.00 | +2.78 |
| `libero_spatial:task3` | 2 | 2 | 4 | +0.00 | +2.78 |
| `libero_spatial:task4` | 2 | 2 | 4 | +0.00 | +2.78 |
| `libero_spatial:task5` | 3 | 3 | 4 | +0.00 | +2.78 |
| `libero_spatial:task6` | 2 | 1 | 4 | +25.00 | +0.00 |
| `libero_spatial:task7` | 3 | 3 | 4 | +0.00 | +2.78 |
| `libero_spatial:task8` | 3 | 3 | 4 | +0.00 | +2.78 |
| `libero_spatial:task9` | 2 | 2 | 4 | +0.00 | +2.78 |

### Goal

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COHERENT_H8 | 26/40 | 65.00% | 6877 | 877 | 12.753% | 12.317 | 3.464/7.0/7 | 3.464/7.0/7 |
| ARM8_GRIP16 | 29/40 | 72.50% | 6613 | 847 | 12.808% | 11.935 | 3.462/7.0/7 | 7.355/15.0/15 |

- ARM8_GRIP16 vs COHERENT_H8: first-only `5`, second-only `2`, ties `33`, delta `+7.50 pp`, exact two-sided McNemar `p=0.453125`, paired bootstrap 95% CI `[-5.00, +20.00] pp`, task-cluster bootstrap 95% CI `[-2.50, +17.50] pp`.

| Task | ARM8_GRIP16 | H8 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_goal:task0` | 2 | 2 | 4 | +0.00 | +8.33 |
| `libero_goal:task1` | 3 | 3 | 4 | +0.00 | +8.33 |
| `libero_goal:task2` | 3 | 4 | 4 | -25.00 | +11.11 |
| `libero_goal:task3` | 3 | 2 | 4 | +25.00 | +5.56 |
| `libero_goal:task4` | 4 | 3 | 4 | +25.00 | +5.56 |
| `libero_goal:task5` | 2 | 2 | 4 | +0.00 | +8.33 |
| `libero_goal:task6` | 3 | 2 | 4 | +25.00 | +5.56 |
| `libero_goal:task7` | 4 | 4 | 4 | +0.00 | +8.33 |
| `libero_goal:task8` | 4 | 3 | 4 | +25.00 | +5.56 |
| `libero_goal:task9` | 1 | 1 | 4 | +0.00 | +8.33 |

### Object

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COHERENT_H8 | 36/40 | 90.00% | 6121 | 780 | 12.743% | 10.259 | 3.472/7.0/7 | 3.472/7.0/7 |
| ARM8_GRIP16 | 35/40 | 87.50% | 6419 | 818 | 12.743% | 10.751 | 3.475/7.0/7 | 7.356/15.0/15 |

- ARM8_GRIP16 vs COHERENT_H8: first-only `3`, second-only `4`, ties `33`, delta `-2.50 pp`, exact two-sided McNemar `p=1`, paired bootstrap 95% CI `[-15.00, +10.00] pp`, task-cluster bootstrap 95% CI `[-10.00, +5.00] pp`.

| Task | ARM8_GRIP16 | H8 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_object:task0` | 4 | 4 | 4 | +0.00 | -2.78 |
| `libero_object:task1` | 4 | 4 | 4 | +0.00 | -2.78 |
| `libero_object:task2` | 4 | 4 | 4 | +0.00 | -2.78 |
| `libero_object:task3` | 4 | 4 | 4 | +0.00 | -2.78 |
| `libero_object:task4` | 4 | 4 | 4 | +0.00 | -2.78 |
| `libero_object:task5` | 2 | 1 | 4 | +25.00 | -5.56 |
| `libero_object:task6` | 3 | 4 | 4 | -25.00 | +0.00 |
| `libero_object:task7` | 3 | 4 | 4 | -25.00 | +0.00 |
| `libero_object:task8` | 4 | 4 | 4 | +0.00 | -2.78 |
| `libero_object:task9` | 3 | 3 | 4 | +0.00 | -2.78 |

### Long/LIBERO-10

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COHERENT_H8 | 21/40 | 52.50% | 15911 | 1996 | 12.545% | 25.441 | 3.494/7.0/7 | 3.494/7.0/7 |
| ARM8_GRIP16 | 18/40 | 45.00% | 16297 | 2043 | 12.536% | 25.726 | 3.496/7.0/7 | 7.428/15.0/15 |

- ARM8_GRIP16 vs COHERENT_H8: first-only `3`, second-only `6`, ties `31`, delta `-7.50 pp`, exact two-sided McNemar `p=0.507812`, paired bootstrap 95% CI `[-22.50, +7.50] pp`, task-cluster bootstrap 95% CI `[-22.50, +5.00] pp`.

| Task | ARM8_GRIP16 | H8 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_10:task0` | 0 | 1 | 4 | -25.00 | -5.56 |
| `libero_10:task1` | 3 | 3 | 4 | +0.00 | -8.33 |
| `libero_10:task2` | 1 | 2 | 4 | -25.00 | -5.56 |
| `libero_10:task3` | 2 | 3 | 4 | -25.00 | -5.56 |
| `libero_10:task4` | 1 | 0 | 4 | +25.00 | -11.11 |
| `libero_10:task5` | 4 | 4 | 4 | +0.00 | -8.33 |
| `libero_10:task6` | 2 | 4 | 4 | -50.00 | -2.78 |
| `libero_10:task7` | 2 | 2 | 4 | +0.00 | -8.33 |
| `libero_10:task8` | 2 | 1 | 4 | +25.00 | -11.11 |
| `libero_10:task9` | 1 | 1 | 4 | +0.00 | -8.33 |

### Pooled across all suites

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COHERENT_H8 | 106/160 | 66.25% | 36013 | 4553 | 12.643% | 15.453 | 3.482/7.0/7 | 3.482/7.0/7 |
| ARM8_GRIP16 | 106/160 | 66.25% | 36326 | 4594 | 12.647% | 15.520 | 3.483/7.0/7 | 7.390/15.0/15 |

- ARM8_GRIP16 vs COHERENT_H8: first-only `12`, second-only `12`, ties `136`, delta `+0.00 pp`, exact two-sided McNemar `p=1`, paired bootstrap 95% CI `[-6.25, +6.25] pp`, task-cluster bootstrap 95% CI `[-5.62, +5.00] pp`.

| Task | ARM8_GRIP16 | H8 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_10:task0` | 0 | 1 | 4 | -25.00 | +0.64 |
| `libero_10:task1` | 3 | 3 | 4 | +0.00 | +0.00 |
| `libero_10:task2` | 1 | 2 | 4 | -25.00 | +0.64 |
| `libero_10:task3` | 2 | 3 | 4 | -25.00 | +0.64 |
| `libero_10:task4` | 1 | 0 | 4 | +25.00 | -0.64 |
| `libero_10:task5` | 4 | 4 | 4 | +0.00 | +0.00 |
| `libero_10:task6` | 2 | 4 | 4 | -50.00 | +1.28 |
| `libero_10:task7` | 2 | 2 | 4 | +0.00 | +0.00 |
| `libero_10:task8` | 2 | 1 | 4 | +25.00 | -0.64 |
| `libero_10:task9` | 1 | 1 | 4 | +0.00 | +0.00 |
| `libero_goal:task0` | 2 | 2 | 4 | +0.00 | +0.00 |
| `libero_goal:task1` | 3 | 3 | 4 | +0.00 | +0.00 |
| `libero_goal:task2` | 3 | 4 | 4 | -25.00 | +0.64 |
| `libero_goal:task3` | 3 | 2 | 4 | +25.00 | -0.64 |
| `libero_goal:task4` | 4 | 3 | 4 | +25.00 | -0.64 |
| `libero_goal:task5` | 2 | 2 | 4 | +0.00 | +0.00 |
| `libero_goal:task6` | 3 | 2 | 4 | +25.00 | -0.64 |
| `libero_goal:task7` | 4 | 4 | 4 | +0.00 | +0.00 |
| `libero_goal:task8` | 4 | 3 | 4 | +25.00 | -0.64 |
| `libero_goal:task9` | 1 | 1 | 4 | +0.00 | +0.00 |
| `libero_object:task0` | 4 | 4 | 4 | +0.00 | +0.00 |
| `libero_object:task1` | 4 | 4 | 4 | +0.00 | +0.00 |
| `libero_object:task2` | 4 | 4 | 4 | +0.00 | +0.00 |
| `libero_object:task3` | 4 | 4 | 4 | +0.00 | +0.00 |
| `libero_object:task4` | 4 | 4 | 4 | +0.00 | +0.00 |
| `libero_object:task5` | 2 | 1 | 4 | +25.00 | -0.64 |
| `libero_object:task6` | 3 | 4 | 4 | -25.00 | +0.64 |
| `libero_object:task7` | 3 | 4 | 4 | -25.00 | +0.64 |
| `libero_object:task8` | 4 | 4 | 4 | +0.00 | +0.00 |
| `libero_object:task9` | 3 | 3 | 4 | +0.00 | +0.00 |
| `libero_spatial:task0` | 1 | 1 | 4 | +0.00 | +0.00 |
| `libero_spatial:task1` | 3 | 3 | 4 | +0.00 | +0.00 |
| `libero_spatial:task2` | 3 | 3 | 4 | +0.00 | +0.00 |
| `libero_spatial:task3` | 2 | 2 | 4 | +0.00 | +0.00 |
| `libero_spatial:task4` | 2 | 2 | 4 | +0.00 | +0.00 |
| `libero_spatial:task5` | 3 | 3 | 4 | +0.00 | +0.00 |
| `libero_spatial:task6` | 2 | 1 | 4 | +25.00 | -0.64 |
| `libero_spatial:task7` | 3 | 3 | 4 | +0.00 | +0.00 |
| `libero_spatial:task8` | 3 | 3 | 4 | +0.00 | +0.00 |
| `libero_spatial:task9` | 2 | 2 | 4 | +0.00 | +0.00 |

Query-schedule audit: `{'methods': {'SMOLVLA_COHERENT_H8': {'episodes': 160, 'query_periods': [8], 'all_exact_periodic_schedules': True, 'invalid_cell_ids': [], 'all_arm_driven': True}, 'SMOLVLA_ARM8_GRIP16': {'episodes': 160, 'query_periods': [8], 'all_exact_periodic_schedules': True, 'invalid_cell_ids': [], 'all_arm_driven': True}}, 'matched_query_periods': True, 'intended_matched_arm_driven_schedule': True}`. The intended matched arm-driven query schedule is present: both methods query at steps 0, 8, 16, ... until their own terminal step. Aggregate query rates can differ slightly because episode lengths differ.

## D. SmolVLA COHERENT_H16 capacity condition

The capacity condition **ran and completed all 160/160 episodes** after the primary barrier.
Barrier audit: `{'latest_primary_finished_at': 1788278134.3636134, 'earliest_capacity_started_at': 1788278136.736722, 'capacity_started_after_all_primary_finished': True}`.

### Spatial

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COHERENT_H16 | 27/40 | 67.50% | 6551 | 429 | 6.549% | 9.994 | 7.348/15.0/15 | 7.348/15.0/15 |
| COHERENT_H8 | 23/40 | 57.50% | 7104 | 900 | 12.669% | 13.792 | 3.482/7.0/7 | 3.482/7.0/7 |
| ARM8_GRIP16 | 24/40 | 60.00% | 6997 | 886 | 12.663% | 13.669 | 3.482/7.0/7 | 7.367/15.0/15 |

- COHERENT_H16 vs COHERENT_H8: first-only `11`, second-only `7`, ties `22`, delta `+10.00 pp`, exact two-sided McNemar `p=0.480682`, paired bootstrap 95% CI `[-10.00, +30.00] pp`, task-cluster bootstrap 95% CI `[-7.50, +27.50] pp`.

| Task | H16 | H8 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_spatial:task0` | 3 | 1 | 4 | +50.00 | +5.56 |
| `libero_spatial:task1` | 4 | 3 | 4 | +25.00 | +8.33 |
| `libero_spatial:task2` | 3 | 3 | 4 | +0.00 | +11.11 |
| `libero_spatial:task3` | 1 | 2 | 4 | -25.00 | +13.89 |
| `libero_spatial:task4` | 3 | 2 | 4 | +25.00 | +8.33 |
| `libero_spatial:task5` | 2 | 3 | 4 | -25.00 | +13.89 |
| `libero_spatial:task6` | 3 | 1 | 4 | +50.00 | +5.56 |
| `libero_spatial:task7` | 2 | 3 | 4 | -25.00 | +13.89 |
| `libero_spatial:task8` | 4 | 3 | 4 | +25.00 | +8.33 |
| `libero_spatial:task9` | 2 | 2 | 4 | +0.00 | +11.11 |

- ARM8_GRIP16 vs COHERENT_H16: first-only `7`, second-only `10`, ties `23`, delta `-7.50 pp`, exact two-sided McNemar `p=0.629059`, paired bootstrap 95% CI `[-27.50, +12.50] pp`, task-cluster bootstrap 95% CI `[-22.50, +7.50] pp`.

| Task | ARM8_GRIP16 | H16 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_spatial:task0` | 1 | 3 | 4 | -50.00 | -2.78 |
| `libero_spatial:task1` | 3 | 4 | 4 | -25.00 | -5.56 |
| `libero_spatial:task2` | 3 | 3 | 4 | +0.00 | -8.33 |
| `libero_spatial:task3` | 2 | 1 | 4 | +25.00 | -11.11 |
| `libero_spatial:task4` | 2 | 3 | 4 | -25.00 | -5.56 |
| `libero_spatial:task5` | 3 | 2 | 4 | +25.00 | -11.11 |
| `libero_spatial:task6` | 2 | 3 | 4 | -25.00 | -5.56 |
| `libero_spatial:task7` | 3 | 2 | 4 | +25.00 | -11.11 |
| `libero_spatial:task8` | 3 | 4 | 4 | -25.00 | -5.56 |
| `libero_spatial:task9` | 2 | 2 | 4 | +0.00 | -8.33 |

### Goal

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COHERENT_H16 | 27/40 | 67.50% | 6882 | 448 | 6.510% | 8.933 | 7.369/15.0/15 | 7.369/15.0/15 |
| COHERENT_H8 | 26/40 | 65.00% | 6877 | 877 | 12.753% | 12.317 | 3.464/7.0/7 | 3.464/7.0/7 |
| ARM8_GRIP16 | 29/40 | 72.50% | 6613 | 847 | 12.808% | 11.935 | 3.462/7.0/7 | 7.355/15.0/15 |

- COHERENT_H16 vs COHERENT_H8: first-only `6`, second-only `5`, ties `29`, delta `+2.50 pp`, exact two-sided McNemar `p=1`, paired bootstrap 95% CI `[-12.50, +17.50] pp`, task-cluster bootstrap 95% CI `[-10.00, +15.00] pp`.

| Task | H16 | H8 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_goal:task0` | 1 | 2 | 4 | -25.00 | +5.56 |
| `libero_goal:task1` | 3 | 3 | 4 | +0.00 | +2.78 |
| `libero_goal:task2` | 3 | 4 | 4 | -25.00 | +5.56 |
| `libero_goal:task3` | 2 | 2 | 4 | +0.00 | +2.78 |
| `libero_goal:task4` | 4 | 3 | 4 | +25.00 | +0.00 |
| `libero_goal:task5` | 3 | 2 | 4 | +25.00 | +0.00 |
| `libero_goal:task6` | 3 | 2 | 4 | +25.00 | +0.00 |
| `libero_goal:task7` | 4 | 4 | 4 | +0.00 | +2.78 |
| `libero_goal:task8` | 2 | 3 | 4 | -25.00 | +5.56 |
| `libero_goal:task9` | 2 | 1 | 4 | +25.00 | +0.00 |

- ARM8_GRIP16 vs COHERENT_H16: first-only `7`, second-only `5`, ties `28`, delta `+5.00 pp`, exact two-sided McNemar `p=0.774414`, paired bootstrap 95% CI `[-12.50, +22.50] pp`, task-cluster bootstrap 95% CI `[-7.50, +20.00] pp`.

| Task | ARM8_GRIP16 | H16 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_goal:task0` | 2 | 1 | 4 | +25.00 | +2.78 |
| `libero_goal:task1` | 3 | 3 | 4 | +0.00 | +5.56 |
| `libero_goal:task2` | 3 | 3 | 4 | +0.00 | +5.56 |
| `libero_goal:task3` | 3 | 2 | 4 | +25.00 | +2.78 |
| `libero_goal:task4` | 4 | 4 | 4 | +0.00 | +5.56 |
| `libero_goal:task5` | 2 | 3 | 4 | -25.00 | +8.33 |
| `libero_goal:task6` | 3 | 3 | 4 | +0.00 | +5.56 |
| `libero_goal:task7` | 4 | 4 | 4 | +0.00 | +5.56 |
| `libero_goal:task8` | 4 | 2 | 4 | +50.00 | +0.00 |
| `libero_goal:task9` | 1 | 2 | 4 | -25.00 | +8.33 |

### Object

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COHERENT_H16 | 38/40 | 95.00% | 5884 | 384 | 6.526% | 6.978 | 7.352/15.0/15 | 7.352/15.0/15 |
| COHERENT_H8 | 36/40 | 90.00% | 6121 | 780 | 12.743% | 10.259 | 3.472/7.0/7 | 3.472/7.0/7 |
| ARM8_GRIP16 | 35/40 | 87.50% | 6419 | 818 | 12.743% | 10.751 | 3.475/7.0/7 | 7.356/15.0/15 |

- COHERENT_H16 vs COHERENT_H8: first-only `3`, second-only `1`, ties `36`, delta `+5.00 pp`, exact two-sided McNemar `p=0.625`, paired bootstrap 95% CI `[-5.00, +15.00] pp`, task-cluster bootstrap 95% CI `[-5.00, +17.50] pp`.

| Task | H16 | H8 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_object:task0` | 4 | 4 | 4 | +0.00 | +5.56 |
| `libero_object:task1` | 4 | 4 | 4 | +0.00 | +5.56 |
| `libero_object:task2` | 4 | 4 | 4 | +0.00 | +5.56 |
| `libero_object:task3` | 4 | 4 | 4 | +0.00 | +5.56 |
| `libero_object:task4` | 4 | 4 | 4 | +0.00 | +5.56 |
| `libero_object:task5` | 3 | 1 | 4 | +50.00 | +0.00 |
| `libero_object:task6` | 4 | 4 | 4 | +0.00 | +5.56 |
| `libero_object:task7` | 3 | 4 | 4 | -25.00 | +8.33 |
| `libero_object:task8` | 4 | 4 | 4 | +0.00 | +5.56 |
| `libero_object:task9` | 4 | 3 | 4 | +25.00 | +2.78 |

- ARM8_GRIP16 vs COHERENT_H16: first-only `2`, second-only `5`, ties `33`, delta `-7.50 pp`, exact two-sided McNemar `p=0.453125`, paired bootstrap 95% CI `[-20.00, +5.00] pp`, task-cluster bootstrap 95% CI `[-15.00, +0.00] pp`.

| Task | ARM8_GRIP16 | H16 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_object:task0` | 4 | 4 | 4 | +0.00 | -8.33 |
| `libero_object:task1` | 4 | 4 | 4 | +0.00 | -8.33 |
| `libero_object:task2` | 4 | 4 | 4 | +0.00 | -8.33 |
| `libero_object:task3` | 4 | 4 | 4 | +0.00 | -8.33 |
| `libero_object:task4` | 4 | 4 | 4 | +0.00 | -8.33 |
| `libero_object:task5` | 2 | 3 | 4 | -25.00 | -5.56 |
| `libero_object:task6` | 3 | 4 | 4 | -25.00 | -5.56 |
| `libero_object:task7` | 3 | 3 | 4 | +0.00 | -8.33 |
| `libero_object:task8` | 4 | 4 | 4 | +0.00 | -8.33 |
| `libero_object:task9` | 3 | 4 | 4 | -25.00 | -5.56 |

### Long/LIBERO-10

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COHERENT_H16 | 20/40 | 50.00% | 15932 | 1015 | 6.371% | 16.604 | 7.429/15.0/15 | 7.429/15.0/15 |
| COHERENT_H8 | 21/40 | 52.50% | 15911 | 1996 | 12.545% | 25.441 | 3.494/7.0/7 | 3.494/7.0/7 |
| ARM8_GRIP16 | 18/40 | 45.00% | 16297 | 2043 | 12.536% | 25.726 | 3.496/7.0/7 | 7.428/15.0/15 |

- COHERENT_H16 vs COHERENT_H8: first-only `6`, second-only `7`, ties `27`, delta `-2.50 pp`, exact two-sided McNemar `p=1`, paired bootstrap 95% CI `[-20.00, +15.00] pp`, task-cluster bootstrap 95% CI `[-25.00, +17.50] pp`.

| Task | H16 | H8 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_10:task0` | 0 | 1 | 4 | -25.00 | +0.00 |
| `libero_10:task1` | 3 | 3 | 4 | +0.00 | -2.78 |
| `libero_10:task2` | 3 | 2 | 4 | +25.00 | -5.56 |
| `libero_10:task3` | 1 | 3 | 4 | -50.00 | +2.78 |
| `libero_10:task4` | 1 | 0 | 4 | +25.00 | -5.56 |
| `libero_10:task5` | 4 | 4 | 4 | +0.00 | -2.78 |
| `libero_10:task6` | 1 | 4 | 4 | -75.00 | +5.56 |
| `libero_10:task7` | 2 | 2 | 4 | +0.00 | -2.78 |
| `libero_10:task8` | 2 | 1 | 4 | +25.00 | -5.56 |
| `libero_10:task9` | 3 | 1 | 4 | +50.00 | -8.33 |

- ARM8_GRIP16 vs COHERENT_H16: first-only `7`, second-only `9`, ties `24`, delta `-5.00 pp`, exact two-sided McNemar `p=0.803619`, paired bootstrap 95% CI `[-25.00, +15.00] pp`, task-cluster bootstrap 95% CI `[-20.00, +7.50] pp`.

| Task | ARM8_GRIP16 | H16 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_10:task0` | 0 | 0 | 4 | +0.00 | -5.56 |
| `libero_10:task1` | 3 | 3 | 4 | +0.00 | -5.56 |
| `libero_10:task2` | 1 | 3 | 4 | -50.00 | +0.00 |
| `libero_10:task3` | 2 | 1 | 4 | +25.00 | -8.33 |
| `libero_10:task4` | 1 | 1 | 4 | +0.00 | -5.56 |
| `libero_10:task5` | 4 | 4 | 4 | +0.00 | -5.56 |
| `libero_10:task6` | 2 | 1 | 4 | +25.00 | -8.33 |
| `libero_10:task7` | 2 | 2 | 4 | +0.00 | -5.56 |
| `libero_10:task8` | 2 | 2 | 4 | +0.00 | -5.56 |
| `libero_10:task9` | 1 | 3 | 4 | -50.00 | +0.00 |

### Pooled across all suites

| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COHERENT_H16 | 112/160 | 70.00% | 35249 | 2276 | 6.457% | 10.627 | 7.389/15.0/15 | 7.389/15.0/15 |
| COHERENT_H8 | 106/160 | 66.25% | 36013 | 4553 | 12.643% | 15.453 | 3.482/7.0/7 | 3.482/7.0/7 |
| ARM8_GRIP16 | 106/160 | 66.25% | 36326 | 4594 | 12.647% | 15.520 | 3.483/7.0/7 | 7.390/15.0/15 |

- COHERENT_H16 vs COHERENT_H8: first-only `26`, second-only `20`, ties `114`, delta `+3.75 pp`, exact two-sided McNemar `p=0.461391`, paired bootstrap 95% CI `[-4.38, +11.88] pp`, task-cluster bootstrap 95% CI `[-5.00, +12.50] pp`.

| Task | H16 | H8 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_10:task0` | 0 | 1 | 4 | -25.00 | +4.49 |
| `libero_10:task1` | 3 | 3 | 4 | +0.00 | +3.85 |
| `libero_10:task2` | 3 | 2 | 4 | +25.00 | +3.21 |
| `libero_10:task3` | 1 | 3 | 4 | -50.00 | +5.13 |
| `libero_10:task4` | 1 | 0 | 4 | +25.00 | +3.21 |
| `libero_10:task5` | 4 | 4 | 4 | +0.00 | +3.85 |
| `libero_10:task6` | 1 | 4 | 4 | -75.00 | +5.77 |
| `libero_10:task7` | 2 | 2 | 4 | +0.00 | +3.85 |
| `libero_10:task8` | 2 | 1 | 4 | +25.00 | +3.21 |
| `libero_10:task9` | 3 | 1 | 4 | +50.00 | +2.56 |
| `libero_goal:task0` | 1 | 2 | 4 | -25.00 | +4.49 |
| `libero_goal:task1` | 3 | 3 | 4 | +0.00 | +3.85 |
| `libero_goal:task2` | 3 | 4 | 4 | -25.00 | +4.49 |
| `libero_goal:task3` | 2 | 2 | 4 | +0.00 | +3.85 |
| `libero_goal:task4` | 4 | 3 | 4 | +25.00 | +3.21 |
| `libero_goal:task5` | 3 | 2 | 4 | +25.00 | +3.21 |
| `libero_goal:task6` | 3 | 2 | 4 | +25.00 | +3.21 |
| `libero_goal:task7` | 4 | 4 | 4 | +0.00 | +3.85 |
| `libero_goal:task8` | 2 | 3 | 4 | -25.00 | +4.49 |
| `libero_goal:task9` | 2 | 1 | 4 | +25.00 | +3.21 |
| `libero_object:task0` | 4 | 4 | 4 | +0.00 | +3.85 |
| `libero_object:task1` | 4 | 4 | 4 | +0.00 | +3.85 |
| `libero_object:task2` | 4 | 4 | 4 | +0.00 | +3.85 |
| `libero_object:task3` | 4 | 4 | 4 | +0.00 | +3.85 |
| `libero_object:task4` | 4 | 4 | 4 | +0.00 | +3.85 |
| `libero_object:task5` | 3 | 1 | 4 | +50.00 | +2.56 |
| `libero_object:task6` | 4 | 4 | 4 | +0.00 | +3.85 |
| `libero_object:task7` | 3 | 4 | 4 | -25.00 | +4.49 |
| `libero_object:task8` | 4 | 4 | 4 | +0.00 | +3.85 |
| `libero_object:task9` | 4 | 3 | 4 | +25.00 | +3.21 |
| `libero_spatial:task0` | 3 | 1 | 4 | +50.00 | +2.56 |
| `libero_spatial:task1` | 4 | 3 | 4 | +25.00 | +3.21 |
| `libero_spatial:task2` | 3 | 3 | 4 | +0.00 | +3.85 |
| `libero_spatial:task3` | 1 | 2 | 4 | -25.00 | +4.49 |
| `libero_spatial:task4` | 3 | 2 | 4 | +25.00 | +3.21 |
| `libero_spatial:task5` | 2 | 3 | 4 | -25.00 | +4.49 |
| `libero_spatial:task6` | 3 | 1 | 4 | +50.00 | +2.56 |
| `libero_spatial:task7` | 2 | 3 | 4 | -25.00 | +4.49 |
| `libero_spatial:task8` | 4 | 3 | 4 | +25.00 | +3.21 |
| `libero_spatial:task9` | 2 | 2 | 4 | +0.00 | +3.85 |

- ARM8_GRIP16 vs COHERENT_H16: first-only `23`, second-only `29`, ties `108`, delta `-3.75 pp`, exact two-sided McNemar `p=0.488456`, paired bootstrap 95% CI `[-12.50, +5.00] pp`, task-cluster bootstrap 95% CI `[-10.62, +3.12] pp`.

| Task | ARM8_GRIP16 | H16 | Blocks | Delta (pp) | LOTO pooled delta (pp) |
|---|---:|---:|---:|---:|---:|
| `libero_10:task0` | 0 | 0 | 4 | +0.00 | -3.85 |
| `libero_10:task1` | 3 | 3 | 4 | +0.00 | -3.85 |
| `libero_10:task2` | 1 | 3 | 4 | -50.00 | -2.56 |
| `libero_10:task3` | 2 | 1 | 4 | +25.00 | -4.49 |
| `libero_10:task4` | 1 | 1 | 4 | +0.00 | -3.85 |
| `libero_10:task5` | 4 | 4 | 4 | +0.00 | -3.85 |
| `libero_10:task6` | 2 | 1 | 4 | +25.00 | -4.49 |
| `libero_10:task7` | 2 | 2 | 4 | +0.00 | -3.85 |
| `libero_10:task8` | 2 | 2 | 4 | +0.00 | -3.85 |
| `libero_10:task9` | 1 | 3 | 4 | -50.00 | -2.56 |
| `libero_goal:task0` | 2 | 1 | 4 | +25.00 | -4.49 |
| `libero_goal:task1` | 3 | 3 | 4 | +0.00 | -3.85 |
| `libero_goal:task2` | 3 | 3 | 4 | +0.00 | -3.85 |
| `libero_goal:task3` | 3 | 2 | 4 | +25.00 | -4.49 |
| `libero_goal:task4` | 4 | 4 | 4 | +0.00 | -3.85 |
| `libero_goal:task5` | 2 | 3 | 4 | -25.00 | -3.21 |
| `libero_goal:task6` | 3 | 3 | 4 | +0.00 | -3.85 |
| `libero_goal:task7` | 4 | 4 | 4 | +0.00 | -3.85 |
| `libero_goal:task8` | 4 | 2 | 4 | +50.00 | -5.13 |
| `libero_goal:task9` | 1 | 2 | 4 | -25.00 | -3.21 |
| `libero_object:task0` | 4 | 4 | 4 | +0.00 | -3.85 |
| `libero_object:task1` | 4 | 4 | 4 | +0.00 | -3.85 |
| `libero_object:task2` | 4 | 4 | 4 | +0.00 | -3.85 |
| `libero_object:task3` | 4 | 4 | 4 | +0.00 | -3.85 |
| `libero_object:task4` | 4 | 4 | 4 | +0.00 | -3.85 |
| `libero_object:task5` | 2 | 3 | 4 | -25.00 | -3.21 |
| `libero_object:task6` | 3 | 4 | 4 | -25.00 | -3.21 |
| `libero_object:task7` | 3 | 3 | 4 | +0.00 | -3.85 |
| `libero_object:task8` | 4 | 4 | 4 | +0.00 | -3.85 |
| `libero_object:task9` | 3 | 4 | 4 | -25.00 | -3.21 |
| `libero_spatial:task0` | 1 | 3 | 4 | -50.00 | -2.56 |
| `libero_spatial:task1` | 3 | 4 | 4 | -25.00 | -3.21 |
| `libero_spatial:task2` | 3 | 3 | 4 | +0.00 | -3.85 |
| `libero_spatial:task3` | 2 | 1 | 4 | +25.00 | -4.49 |
| `libero_spatial:task4` | 2 | 3 | 4 | -25.00 | -3.21 |
| `libero_spatial:task5` | 3 | 2 | 4 | +25.00 | -4.49 |
| `libero_spatial:task6` | 2 | 3 | 4 | -25.00 | -3.21 |
| `libero_spatial:task7` | 3 | 2 | 4 | +25.00 | -4.49 |
| `libero_spatial:task8` | 3 | 4 | 4 | -25.00 | -3.21 |
| `libero_spatial:task9` | 2 | 2 | 4 | +0.00 | -3.85 |

Capacity interpretation: coherent H16 was numerically higher than H8 (112/160 vs 106/160; +3.75 pp), while ARM8_GRIP16 did not exceed H16. Together with the exactly null pooled primary comparison, these data do not establish a component-specific SmolVLA advantage.

## E. Execution integrity

- Completed: **800/800 requested overnight cells**; **926/926 full-manifest cells**.
- Pending/running: **0 / 0**.
- Scientific execution retries: **0**; TECHNICAL_FAILED: **0**.
- Pre-scientific integration failures: `{'act_object_h8_126': {'cells_with_exception_history': 126, 'exception_attempts': 378, 'provisional_technical_failed_markers': 126, 'scientific_result_files': 0}, 'act_object_h8_126_second_mapping': {'cells_with_exception_history': 56, 'exception_attempts': 163, 'provisional_technical_failed_markers': 53, 'scientific_result_files': 0}}`. These archived camera-key/configuration construction failures produced no scientific result files.
- Valid scientific failure rerun: **False**.
- Prohibited experiment launched: **False**. Observed result methods were exactly `['ARM4_GRIP32', 'COHERENT_H8', 'SMOLVLA_ARM8_GRIP16', 'SMOLVLA_COHERENT_H16', 'SMOLVLA_COHERENT_H8']`.

## F. Branch state

This harvest updates results, complete markers, worker progress/logs, aggregate analysis, the exposure inventory, handoff, and this report. The fallback manuscript is unchanged.

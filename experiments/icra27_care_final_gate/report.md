# ICRA 2027 final CARE method gate

Final label: **METHOD_NULL**.

The held-out cohort contains 130 paired Object blocks. Four raw historical outcomes omitted by the prior inventory were removed before preregistration: task 6 states 25, 26, 28, and 29. No replacement states were added.

## Gate M methods

| Method | Success | Rate (%) | Environment steps | Policy queries | Query rate | Mean execution horizon | Mean successful completion length |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0_HARD16 | 92/130 | 70.77 | 23082 | 1503 | 0.065116 | 16.000 | 135.2391304347826 |
| M2_GRIPPER_EVENT | 86/130 | 66.15 | 23606 | 2160 | 0.091502 | 11.328 | 131.2325581395349 |
| FIXED_H13 | 95/130 | 73.08 | 22345 | 1780 | 0.079660 | 13.000 | 132.05263157894737 |
| SHUFFLED_TRIGGER | 90/130 | 69.23 | 23389 | 1829 | 0.078199 | 13.220 | 135.43333333333334 |

Execution-horizon histograms:

- `M0_HARD16`: 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:1503
- `M2_GRIPPER_EVENT`: 4:273, 5:182, 6:171, 7:103, 8:97, 9:80, 10:56, 11:35, 12:43, 13:20, 14:32, 15:33, 16:1035
- `FIXED_H13`: 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:1780, 14:0, 15:0, 16:0
- `SHUFFLED_TRIGGER`: 4:103, 5:88, 6:36, 7:58, 8:79, 9:70, 10:60, 11:32, 12:33, 13:40, 14:36, 15:32, 16:1162

## Gate M primary contrasts

| Contrast | First | Second | Discordance | Delta (pp) | McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Positive LOTO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2_VS_M0 | 86/130 | 92/130 | 7:13 | -4.62 | 0.263176 | [-11.54, +2.31] | [-11.81, +1.68] | 0/9 |
| M2_VS_FIXED_H13 | 86/130 | 95/130 | 3:12 | -6.92 | 0.0351562 | [-13.08, -1.54] | [-11.94, -1.59] | 0/9 |
| M2_VS_SHUFFLED | 86/130 | 90/130 | 9:13 | -3.08 | 0.523467 | [-10.00, +3.85] | [-12.50, +4.80] | 0/9 |

Per-task deltas and leave-one-task-out pooled deltas:

| Task | Blocks | M2−M0 task / LOTO (pp) | M2−H13 task / LOTO (pp) | M2−SHUFFLED task / LOTO (pp) |
|---|---:|---:|---:|---:|
| `libero_object:task1` | 11 | +0.00 / -5.04 | -9.09 / -6.72 | +0.00 / -3.36 |
| `libero_object:task2` | 16 | -12.50 / -3.51 | -12.50 / -6.14 | +0.00 / -3.51 |
| `libero_object:task3` | 16 | +6.25 / -6.14 | +0.00 / -7.89 | +6.25 / -4.39 |
| `libero_object:task4` | 11 | +0.00 / -5.04 | -9.09 / -6.72 | +0.00 / -3.36 |
| `libero_object:task5` | 16 | -25.00 / -1.75 | -18.75 / -5.26 | -25.00 / +0.00 |
| `libero_object:task6` | 12 | +0.00 / -5.08 | +0.00 / -7.63 | +8.33 / -4.24 |
| `libero_object:task7` | 16 | -12.50 / -3.51 | -12.50 / -6.14 | -25.00 / +0.00 |
| `libero_object:task8` | 16 | +0.00 / -5.26 | -6.25 / -7.02 | +6.25 / -4.39 |
| `libero_object:task9` | 16 | +6.25 / -6.14 | +6.25 / -8.77 | +6.25 / -4.39 |

## Gate M secondary descriptive contrasts

| Contrast | First | Second | Discordance | Delta (pp) | McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Positive LOTO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIXED_H13_VS_M0 | 95/130 | 92/130 | 8:5 | +2.31 | 0.581055 | [-3.08, +7.69] | [-0.75, +5.43] | 9/9 |
| SHUFFLED_VS_M0 | 90/130 | 92/130 | 9:11 | -1.54 | 0.823803 | [-8.46, +5.38] | [-5.97, +3.10] | 0/9 |
| FIXED_H13_VS_SHUFFLED | 95/130 | 90/130 | 13:8 | +3.85 | 0.38331 | [-3.08, +10.77] | [-2.22, +9.23] | 9/9 |

## Query-budget sanity

M2 versus SHUFFLED absolute query-rate difference: **0.013303** (criterion <= 0.005: **MISS**). M2 minus FIXED_H13: **+0.011842**.

## SmolVLA cross-policy robustness

This cohort is outcome-exposed and is labeled **CROSS_POLICY_ROBUSTNESS**, not independent confirmation.

### libero_spatial

| Method | Success | Environment steps | Policy queries | Query rate | Wall-clock (s) | Arm age mean/max | Gripper age mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP4 | 29/40 | 6140 | 1548 | 0.252117 | 708.3 | 1.494/3 | 1.494/3 |
| ARM4_GRIP32 | 31/40 | 6049 | 1525 | 0.252108 | 706.6 | 1.494/3 | 14.983/31 |

| Contrast | First | Second | Discordance | Delta (pp) | McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Positive LOTO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP32 vs ARM4_GRIP4 | 31/40 | 29/40 | 5:3 | +5.00 | 0.726562 | [-7.50, +20.00] | [-5.00, +15.00] | 10/10 |

### libero_object

| Method | Success | Environment steps | Policy queries | Query rate | Wall-clock (s) | Arm age mean/max | Gripper age mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP4 | 33/40 | 6697 | 1686 | 0.251755 | 718.3 | 1.494/3 | 1.494/3 |
| ARM4_GRIP32 | 35/40 | 6733 | 1699 | 0.252339 | 721.5 | 1.492/3 | 14.943/31 |

| Contrast | First | Second | Discordance | Delta (pp) | McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Positive LOTO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP32 vs ARM4_GRIP4 | 35/40 | 33/40 | 5:3 | +5.00 | 0.726562 | [-7.50, +20.00] | [-7.50, +20.00] | 9/10 |

### libero_goal

| Method | Success | Environment steps | Policy queries | Query rate | Wall-clock (s) | Arm age mean/max | Gripper age mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP4 | 27/40 | 6762 | 1704 | 0.251996 | 745.3 | 1.495/3 | 1.495/3 |
| ARM4_GRIP32 | 25/40 | 6957 | 1750 | 0.251545 | 760.0 | 1.496/3 | 14.974/31 |

| Contrast | First | Second | Discordance | Delta (pp) | McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Positive LOTO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP32 vs ARM4_GRIP4 | 25/40 | 27/40 | 2:4 | -5.00 | 0.6875 | [-17.50, +7.50] | [-15.00, +5.00] | 0/10 |

### libero_10

| Method | Success | Environment steps | Policy queries | Query rate | Wall-clock (s) | Arm age mean/max | Gripper age mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP4 | 19/40 | 15755 | 3944 | 0.250333 | 1728.3 | 1.499/3 | 1.499/3 |
| ARM4_GRIP32 | 19/40 | 15898 | 3982 | 0.250472 | 1732.5 | 1.499/3 | 15.265/31 |

| Contrast | First | Second | Discordance | Delta (pp) | McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Positive LOTO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP32 vs ARM4_GRIP4 | 19/40 | 19/40 | 5:5 | +0.00 | 1 | [-15.00, +15.00] | [-17.50, +17.50] | 3/10 |

### pooled

| Method | Success | Environment steps | Policy queries | Query rate | Wall-clock (s) | Arm age mean/max | Gripper age mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP4 | 108/160 | 35354 | 8882 | 0.251230 | 3900.2 | 1.496/3 | 1.496/3 |
| ARM4_GRIP32 | 110/160 | 35637 | 8956 | 0.251312 | 3920.4 | 1.496/3 | 15.100/31 |

| Contrast | First | Second | Discordance | Delta (pp) | McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Positive LOTO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARM4_GRIP32 vs ARM4_GRIP4 | 110/160 | 108/160 | 17:15 | +1.25 | 0.86005 | [-5.62, +8.12] | [-5.00, +8.12] | 38/40 |

## Integrity

Completed Gate M: **520/520**. Completed SmolVLA robustness: **320/320**. Technical failure attempts: **0**; terminal technical failures: **0**.

Observed methods were exactly: `['ARM4_GRIP32', 'ARM4_GRIP4', 'FIXED_H13', 'M0_HARD16', 'M2_GRIPPER_EVENT', 'SHUFFLED_TRIGGER']`. No follow-up condition is present in the frozen manifest or completed results.

# Asymmetric Temporal Reuse development gate

## Results

These are development results on the exposed Object tasks 1-9 cohort, used only to decide whether to freeze the executor. The inferential result for the paper will come from the subsequent frozen cross-suite / unseen confirmation run, not from this table.

Decision branch reached: **ASYM_REUSE_MECHANISM_ONLY**. No paper title is chosen in this run.

### C1/C2/hard-h16 table

| Method | Arm source | Grip source | Success /126 | Success % | Observed query rate | Observed mean grip age | Age range |
|---|---|---|---:|---:|---:|---:|---:|
| C2_H16_ARM_FRESH_GRIP | current h16 chunk | fresh q=t | 42/126 | 33.3% | 1.00000 | 0.000 | 0–0 |
| HARD_H16 | current h16 chunk | current h16 chunk | 88/126 | 69.8% | 0.06515 | 7.358 | 0–15 |
| C1_PREVIOUS_CHUNK_GRIP | current h16 chunk | previous h16 chunk | 64/126 | 50.8% | 0.06464 | 22.108 | 0–31 |

C1 has the same structural h16 query schedule as hard h16. C2 queries densely for fresh gripper values, but its arm always comes from the scheduled h16 chunk.

### Timing (secondary)

| Method | Total policy queries | Total environment steps | Mean wall-clock s/episode | Mean policy-call latency (s) |
|---|---:|---:|---:|---:|
| C2_H16_ARM_FRESH_GRIP | 29413 | 29413 | 26.360 | 0.083171 |
| HARD_H16 | 1490 | 22869 | n/a (reused baseline) | n/a (reused baseline) |
| C1_PREVIOUS_CHUNK_GRIP | 1652 | 25555 | 6.975 | 0.051432 |

### Primary and secondary paired contrasts

| Contrast | First-only | Second-only | Net | Success delta (pp) | Exact two-sided McNemar p | Paired 95% CI | Task-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1_VS_HARD_H16: C1_PREVIOUS_CHUNK_GRIP vs HARD_H16 | 0 | 24 | -24 | -19.05 | 1.19209e-07 | [-0.262, -0.127] | [-0.270, -0.103] |
| C1_VS_C2: C1_PREVIOUS_CHUNK_GRIP vs C2_H16_ARM_FRESH_GRIP | 27 | 5 | 22 | 17.46 | 0.000113074 | [0.095, 0.262] | [0.103, 0.254] |
| HARD_H16_VS_C2: HARD_H16 vs C2_H16_ARM_FRESH_GRIP | 48 | 2 | 46 | 36.51 | 2.26663e-12 | [0.278, 0.452] | [0.286, 0.437] |

### Per-task success counts

| Task | C2 | hard h16 | C1 | C1−hard | C1−C2 | hard−C2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2/14 | 8/14 | 4/14 | -4 | 2 | 6 |
| 2 | 7/14 | 13/14 | 10/14 | -3 | 3 | 6 |
| 3 | 6/14 | 11/14 | 9/14 | -2 | 3 | 5 |
| 4 | 4/14 | 10/14 | 10/14 | 0 | 6 | 6 |
| 5 | 3/14 | 11/14 | 6/14 | -5 | 3 | 8 |
| 6 | 5/14 | 8/14 | 6/14 | -2 | 1 | 3 |
| 7 | 3/14 | 8/14 | 3/14 | -5 | 0 | 5 |
| 8 | 2/14 | 7/14 | 4/14 | -3 | 2 | 5 |
| 9 | 10/14 | 12/14 | 12/14 | 0 | 2 | 2 |

### Leave-one-task-out stability

| Omitted task | C1−hard h16 | C1−C2 | hard h16−C2 |
|---:|---:|---:|---:|
| 1 | -0.1786 | 0.1786 | 0.3571 |
| 2 | -0.1875 | 0.1696 | 0.3571 |
| 3 | -0.1964 | 0.1696 | 0.3661 |
| 4 | -0.2143 | 0.1429 | 0.3571 |
| 5 | -0.1696 | 0.1696 | 0.3393 |
| 6 | -0.1964 | 0.1875 | 0.3839 |
| 7 | -0.1696 | 0.1964 | 0.3661 |
| 8 | -0.1875 | 0.1786 | 0.3661 |
| 9 | -0.2143 | 0.1786 | 0.3929 |

### Outcome-stratified descriptive step-log analyses

These are descriptive only and are not used to choose a lag, offset bound, horizon, or source-selection rule.

#### C2_H16_ARM_FRESH_GRIP

| Outcome | Episodes | Steps | |a[6]|<0.5 | Sign flips/100 steps | First a[6]>0 step (mean; median; no-positive) | a[6]>0 fraction | Negative recurrence after first positive | Mean ||a[0:3]|| | Mean ||a[3:6]|| |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| success | 42 | 5893 | 0.02833870694043781 | 1.5611742745630408 | 59.357142857142854; 50.0; 0 | 0.5418292889869336 | 0.8095238095238095 | 0.6893928154211109 | 0.04470111820035309 |
| failure | 84 | 23520 | 0.025467687074829933 | 0.6590136054421769 | 137.71052631578948; 130.5; 46 | 0.14306972789115646 | 0.32142857142857145 | 0.5067109162256578 | 0.04352463392886116 |

Gripper source-age histograms (age: step count, all bins 0–31):
- success: 0:5893
- failure: 0:23520

#### HARD_H16

| Outcome | Episodes | Steps | |a[6]|<0.5 | Sign flips/100 steps | First a[6]>0 step (mean; median; no-positive) | a[6]>0 fraction | Negative recurrence after first positive | Mean ||a[0:3]|| | Mean ||a[3:6]|| |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| success | 88 | 12229 | 0.0206067544361763 | 2.0361435930983727 | 48.0; 47.0; 0 | 0.5700384332324802 | 0.7840909090909091 | 0.6975750792224878 | 0.047866295937513154 |
| failure | 38 | 10640 | 0.14163533834586467 | 6.212406015037594 | 53.78378378378378; 52.0; 1 | 0.45460526315789473 | 0.9736842105263158 | 0.49366367584549775 | 0.04223404766776264 |

Gripper source-age histograms (age: step count, all bins 0–31):
- success: 0:806, 1:803, 2:798, 3:794, 4:789, 5:783, 6:775, 7:764, 8:760, 9:754, 10:746, 11:741, 12:734, 13:731, 14:728, 15:723
- failure: 0:684, 1:684, 2:684, 3:684, 4:684, 5:684, 6:684, 7:684, 8:646, 9:646, 10:646, 11:646, 12:646, 13:646, 14:646, 15:646

#### C1_PREVIOUS_CHUNK_GRIP

| Outcome | Episodes | Steps | |a[6]|<0.5 | Sign flips/100 steps | First a[6]>0 step (mean; median; no-positive) | a[6]>0 fraction | Negative recurrence after first positive | Mean ||a[0:3]|| | Mean ||a[3:6]|| |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| success | 64 | 8195 | 0.012080536912751677 | 1.403294691885296 | 47.9375; 48.0; 0 | 0.5802318486882245 | 0.71875 | 0.7068195183475472 | 0.04530340282656311 |
| failure | 62 | 17360 | 0.07131336405529953 | 4.988479262672811 | 49.54838709677419; 49.0; 0 | 0.7036290322580645 | 0.9193548387096774 | 0.4742553432994561 | 0.04067251338597201 |

Gripper source-age histograms (age: step count, all bins 0–31):
- success: 0:64, 1:64, 2:64, 3:64, 4:64, 5:64, 6:64, 7:64, 8:64, 9:64, 10:64, 11:64, 12:64, 13:64, 14:64, 15:64, 16:472, 17:470, 18:469, 19:467, 20:464, 21:460, 22:456, 23:454, 24:450, 25:446, 26:440, 27:434, 28:430, 29:427, 30:419, 31:413
- failure: 0:62, 1:62, 2:62, 3:62, 4:62, 5:62, 6:62, 7:62, 8:62, 9:62, 10:62, 11:62, 12:62, 13:62, 14:62, 15:62, 16:1054, 17:1054, 18:1054, 19:1054, 20:1054, 21:1054, 22:1054, 23:1054, 24:992, 25:992, 26:992, 27:992, 28:992, 29:992, 30:992, 31:992

### Pre-registered paper artifact mapping

The `{C2, hard h16, C1} × {success/126, observed query rate, observed mean gripper age}` table is the three-point gripper-age series at fixed h16 arm semantics.

- If `C1 > hard h16`: this table is the method table; working title recorded for later consideration: `Asymmetric Temporal Reuse for Action-Chunked Robot Policies`.
- If `C1 ~= hard h16` and `C1 > C2`: this table is the executor-decomposition result; working title recorded for later consideration: `Component-Dependent Effects of Delayed Prediction in Action-Chunked Robot Policies`.
- If `C1 <= hard h16` and `C1 ~= C2`: this table is a negative/scope result and the paper centres on the repaired group-delay factorial.

Reached branch recorded for this run: `C1 <= hard h16 and C1 > C2: executor-decomposition result`. The title remains undecided.

Method development stops after this gate. Any subsequent run requires explicit approval and is limited to the single frozen cross-suite / unseen confirmation.

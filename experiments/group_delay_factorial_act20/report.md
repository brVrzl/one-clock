# Repaired ACT group-delay factorial

Decision: **GROUP_DELAY_METHOD_STRONG**

The primary aggregate contains only the new repaired outcomes for Object tasks 1–9, 14 states per task, and 126 paired blocks per method. Historical Gate-3C outcomes are context only and are not spliced into this table.

## Primary table

| Method | d_arm | d_grip | Success /126 | Success % | Query rate |
|---|---:|---:|---:|---:|---:|
| FRESH | 0 | 0 | 56/126 | 44.4% | 1.00000 |
| FO20 | 0 | 20 | 81/126 | 64.3% | 1.00000 |
| REVERSE20 | 20 | 0 | 12/126 | 9.5% | 1.00000 |
| FULL_OLD20 | 20 | 20 | 47/126 | 37.3% | 1.00000 |
| HARD_H16 | joint h16 | joint h16 | 88/126 | 69.8% | 0.06515 |

## Primary contrasts

| Contrast | First-only | Second-only | Net | Exact McNemar p | Delta (pp) | Paired 95% CI | Cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1_REPLICATION: FO20 vs FRESH | 26 | 1 | 25 | 4.17233e-07 | 19.8 | [0.127, 0.270] | [0.095, 0.310] |
| P2_GROUP_ASYMMETRY: FO20 vs REVERSE20 | 70 | 1 | 69 | 6.09864e-20 | 54.8 | [0.460, 0.635] | [0.444, 0.659] |
| P3_GROUP_STRUCTURE: FO20 vs FULL_OLD20 | 42 | 8 | 34 | 1.16356e-06 | 27.0 | [0.175, 0.365] | [0.175, 0.365] |
| P4_PRACTICAL_BASELINE: FO20 vs HARD_H16 | 9 | 16 | -7 | 0.229523 | -5.6 | [-0.135, 0.024] | [-0.159, 0.032] |

## Per-task primary results

| Task | Fresh | FO20 | Reverse20 | FullOld20 | hard h16 | Interaction I |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2/14 | 9/14 | 0/14 | 2/14 | 8/14 | 5 (0.357) |
| 2 | 10/14 | 13/14 | 1/14 | 6/14 | 13/14 | -2 (-0.143) |
| 3 | 6/14 | 9/14 | 1/14 | 8/14 | 11/14 | -4 (-0.286) |
| 4 | 4/14 | 10/14 | 0/14 | 6/14 | 10/14 | 0 (0.000) |
| 5 | 7/14 | 6/14 | 0/14 | 4/14 | 11/14 | -5 (-0.357) |
| 6 | 7/14 | 8/14 | 1/14 | 4/14 | 8/14 | -2 (-0.143) |
| 7 | 3/14 | 5/14 | 1/14 | 1/14 | 8/14 | 2 (0.143) |
| 8 | 6/14 | 8/14 | 1/14 | 4/14 | 7/14 | -1 (-0.071) |
| 9 | 11/14 | 13/14 | 7/14 | 12/14 | 12/14 | -3 (-0.214) |

## Descriptive 2×2 interaction

I = FO20 − FRESH − FULL_OLD20 + REVERSE20 = -10 successes, or -0.079 per primary block.

## Leave-one-task-out deltas

| Omitted task | FO20−Fresh | FO20−Reverse20 | FO20−hard h16 |
|---:|---:|---:|---:|
| 1 | 0.161 | 0.536 | -0.071 |
| 2 | 0.196 | 0.509 | -0.062 |
| 3 | 0.196 | 0.545 | -0.045 |
| 4 | 0.170 | 0.527 | -0.062 |
| 5 | 0.232 | 0.562 | -0.018 |
| 6 | 0.214 | 0.554 | -0.062 |
| 7 | 0.205 | 0.580 | -0.036 |
| 8 | 0.205 | 0.554 | -0.071 |
| 9 | 0.205 | 0.562 | -0.071 |

## Protocol interpretation

The four dense conditions query ACT at every controller step. HARD_H16 queries only at q=0,16,32,… and executes A_q[t−q] from the newest query; its query rate is reported from observed policy queries divided by environment steps. No task0 or newly claimed held-out task is included in the primary aggregate.

FO20 clearly beat FRESH, REVERSE20, and FULL_OLD20 and was not clearly below HARD_H16.

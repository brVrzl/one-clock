# Sparse temporal ensemble development experiment

At the same fixed sparse-query cadence, this experiment compares executing the newest chunk with canonical oldest-to-newest temporal ensembling over all still-valid same-target predictions. Policies are analyzed separately on four exposed development tasks; no blind tasks are included.

## ACT

| method | success | object3 | spatial0 | goal2 | libero10-3 | queries | query rate | queries/ep | mean candidates | weighted age | completion steps | latency/query (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_h8 | 34/40 | 7/10 | 8/10 | 9/10 | 10/10 | 801 | 0.12793 | 20.02 | 8.87 | 3.47 | 134.1 | 0.0708 |
| sparse_te_h8 | 23/40 | 5/10 | 5/10 | 9/10 | 4/10 | 1126 | 0.12637 | 28.15 | 9.93 | 39.95 | 117.0 | 0.0637 |
| hard_h16 | 33/40 | 8/10 | 7/10 | 10/10 | 8/10 | 441 | 0.06510 | 11.03 | 4.71 | 7.36 | 131.3 | 0.0542 |
| sparse_te_h16 | 20/40 | 4/10 | 6/10 | 9/10 | 1/10 | 642 | 0.06437 | 16.05 | 5.20 | 41.39 | 109.7 | 0.0556 |

| contrast | candidate-only | reference-only | net wins | task nets (obj/sp/goal/10) | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| sparse_te_h8 vs hard_h8 | 2 | 13 | -11 | -2/-3/+0/-6 | 0.007385 |
| sparse_te_h16 vs hard_h16 | 2 | 15 | -13 | -4/-1/-1/-7 | 0.002350 |
| sparse_te_h16 vs sparse_te_h8 | 3 | 6 | -3 | -1/+1/+0/-3 | 0.507812 |

## SmolVLA

| method | success | object3 | spatial0 | goal2 | libero10-3 | queries | query rate | queries/ep | mean candidates | weighted age | completion steps | latency/query (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_h8 | 30/40 | 10/10 | 5/10 | 9/10 | 6/10 | 983 | 0.12672 | 24.57 | 5.57 | 3.48 | 132.6 | 0.4269 |
| sparse_te_h8 | 28/40 | 9/10 | 6/10 | 10/10 | 3/10 | 1032 | 0.12659 | 25.80 | 5.61 | 22.12 | 111.1 | 0.3818 |
| hard_h16 | 30/40 | 9/10 | 8/10 | 10/10 | 3/10 | 510 | 0.06465 | 12.75 | 2.86 | 7.38 | 113.6 | 0.4218 |
| sparse_te_h16 | 29/40 | 6/10 | 8/10 | 10/10 | 5/10 | 507 | 0.06468 | 12.68 | 2.85 | 22.30 | 122.7 | 0.3889 |

| contrast | candidate-only | reference-only | net wins | task nets (obj/sp/goal/10) | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| sparse_te_h8 vs hard_h8 | 2 | 4 | -2 | -1/+1/+1/-3 | 0.687500 |
| sparse_te_h16 vs hard_h16 | 3 | 4 | -1 | -3/+0/+0/+2 | 1.000000 |
| sparse_te_h16 vs sparse_te_h8 | 5 | 4 | +1 | -3/+2/+0/+2 | 1.000000 |

Paired SmolVLA query RNG verification: **smolvla_paired_flow_real_smoke_pass**. The real postprocessed raw chunks had shape `[50, 7]` and maximum absolute difference `0.0` for key `smolvla|libero_object:task3|state=10|env_seed=2000|q=0`.

## Cross-policy summary

| policy | hard h8 | h8+TE | hard h16 | h16+TE |
|---|---:|---:|---:|---:|
| ACT | 34/40 (0.12793) | 23/40 (0.12637) | 33/40 (0.06510) | 20/40 (0.06437) |
| SmolVLA | 30/40 (0.12672) | 28/40 (0.12659) | 30/40 (0.06465) | 29/40 (0.06468) |

Query rates are shown in parentheses. Differences in query rate within a cadence arise only from different episode completion lengths; the scheduled policy-query times are identical over every common trajectory prefix.

## Decision

**SPARSE_TE_HARMFUL**

At essentially matched scheduled policy-query budgets, canonical sparse temporal ensembling clearly harmed ACT at both cadences and did not improve SmolVLA. The ACT loss was not confined to one task: task-level paired nets were nonpositive at h8 and negative for all four tasks at h16, although libero_10:task3 contributed the largest drop. SmolVLA was mixed by task but slightly negative in aggregate at both horizons, so its evidence is compatible with harm but individually inconclusive. On these four development tasks, the gain from moderate commitment does not come from lacking temporal smoothing; averaging historical modes may itself be harmful once query cadence is sparse.

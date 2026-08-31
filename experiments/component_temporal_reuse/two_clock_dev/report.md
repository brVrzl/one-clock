# Matched-query asymmetric component commitment

All methods query every physical step 0, 8, 16, ...; component source chunks are executed at the current target offset.

## Pooled and per-task results

| method | pooled success | object 6 | spatial 2 | goal 1 | libero_10 3 | query rate | mean arm age | mean gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_8_8 | 37/40 | 9/10 | 10/10 | 10/10 | 8/10 | 0.127185 | 3.476432 | 3.476432 |
| arm8_grip16 | 35/40 | 9/10 | 9/10 | 10/10 | 7/10 | 0.126839 | 3.474015 | 7.393332 |
| arm16_grip8 | 30/40 | 6/10 | 8/10 | 8/10 | 8/10 | 0.126819 | 7.396308 | 3.479696 |

## Paired comparisons

| comparison | candidate-only | reference-only | net paired wins | exact McNemar p |
|---|---:|---:|---:|---:|
| arm8_grip16_vs_global_8_8 | 1 | 3 | -2 | 0.625 |
| arm8_grip16_vs_arm16_grip8 | 8 | 3 | +5 | 0.2265625 |

Query schedules exact on every episode: **yes**; common-prefix schedules matched: **yes**; total query counts identical across methods: **no, only differing when episode termination lengths differ**.

Decision: **FAIL**.

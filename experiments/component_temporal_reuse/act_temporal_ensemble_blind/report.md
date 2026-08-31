# Canonical ACT temporal-ensemble blind baseline

LeRobot 0.6.2 `ACTPolicy.select_action` with `ACTTemporalEnsembler`, coefficient `0.01`, effective `n_action_steps=1`.

## Per-task success

| task | success | query rate |
|---|---:|---:|
| libero_10:task1 | 0/10 | 1.000000 |
| libero_10:task9 | 0/10 | 1.000000 |
| libero_goal:task0 | 3/10 | 1.000000 |
| libero_goal:task3 | 0/10 | 1.000000 |
| libero_object:task1 | 0/10 | 1.000000 |
| libero_object:task4 | 0/10 | 1.000000 |
| libero_spatial:task3 | 0/10 | 1.000000 |
| libero_spatial:task7 | 0/10 | 1.000000 |

## Per-suite pooled success

| suite | success | query rate |
|---|---:|---:|
| libero_10 | 0/20 | 1.000000 |
| libero_goal | 3/20 | 1.000000 |
| libero_object | 0/20 | 1.000000 |
| libero_spatial | 0/20 | 1.000000 |

Pooled: **3/80**; query rate **1.000000**; query every environment step: **yes**.

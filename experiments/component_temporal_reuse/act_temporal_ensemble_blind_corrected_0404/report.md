# Corrected ACT temporal-ensemble blind baseline

The previous LeRobot 0.6.2 result (`3/80`) is invalid and protocol-deviated. It is not included here.

## Parity

Runtime: `/home/wjq/workspace/venvs/libero_act/bin/python`, LeRobot 0.4.4, PyTorch 2.7.1+cu128, MuJoCo 3.3.1.

The deterministic 20-step comparison used coefficient `0.01`, chunk size `100`, and the repository's existing same-target ACT-style operator on the exact same chunks.

| check | max absolute error | result |
|---|---:|---|
| t=0 versus Fresh | 0.000000000000000 | pass |
| t=1 explicit formula | 0.000000017570309 | pass |
| t=2 explicit formula | 0.000000008721722 | pass |
| official versus custom, steps 0--19 | 0.000000037619502 | pass |

## Corrected frozen blind cohort

Official LeRobot ACT temporal ensemble, coefficient `0.01`, query every environment step, states 20--29, seeds 4000--4009, and the same 100k checkpoints.

| suite | task | success |
|---|---|---:|
| libero_object | task1 | 3/10 |
| libero_object | task4 | 10/10 |
| libero_spatial | task3 | 6/10 |
| libero_spatial | task7 | 7/10 |
| libero_goal | task0 | 6/10 |
| libero_goal | task3 | 5/10 |
| libero_10 | task1 | 3/10 |
| libero_10 | task9 | 8/10 |

Pooled success: **48/80**.

Total environment steps and policy queries: **18,886** each. Query rate: **1.000000**.

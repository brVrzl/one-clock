# Track-A cross-suite query-allocation confirmation

## Decision labels

- `PENALTY_4X_CONFIRMED`: **YES**
- `DOSE_RESPONSE_SUPPORTED`: **YES**
- `MECHANISM_PASS_A`: **YES**
- `METHOD_PASS_A`: **NO**
- `QUERY_EFFICIENT_TE_LEVEL_PERFORMANCE`: **YES**

## Conditions

| Condition | Success | Rate | Queries | Query rate | Env steps | Mean wall-clock/episode |
|---|---:|---:|---:|---:|---:|---:|
| H16 | 357/450 | 79.33% | 6010 | 0.06486 | 92656 | 5.70s |
| H4 | 314/450 | 69.78% | 25315 | 0.25116 | 100794 | 8.74s |
| ARM4_GRIP32 | 335/450 | 74.44% | 24177 | 0.25132 | 96201 | 8.40s |
| H2 | 295/450 | 65.56% | 53237 | 0.50072 | 106321 | 13.16s |
| ARM2_GRIP16 | 321/450 | 71.33% | 49724 | 0.50091 | 99268 | 12.36s |
| TE_DENSE | 288/450 | 64.00% | 105947 | 1.00000 | 105947 | 19.99s |

## Paired contrasts

| Contrast | Discordance | Delta pp | Exact p | Paired 95% CI | Task-cluster 95% CI | LOTO + | LOSO + |
|---|---:|---:|---:|---:|---:|---:|---:|
| H16-H4 | 60:17 | +9.56 | 8.90985e-07 | [+5.78,+13.33] | [+4.44,+15.33] | 30/30 | 3/3 |
| H4-H2 | 32:13 | +4.22 | 0.00660882 | [+1.33,+7.11] | [+1.33,+7.11] | 30/30 | 3/3 |
| ARM4_GRIP32-H4 | 32:11 | +4.67 | 0.00191396 | [+2.00,+7.56] | [+0.67,+9.11] | 30/30 | 3/3 |
| ARM2_GRIP16-H2 | 28:2 | +5.78 | 8.67993e-07 | [+3.56,+8.22] | [+2.67,+9.56] | 30/30 | 3/3 |
| ARM4_GRIP32-H16 | 24:46 | -4.89 | 0.0115264 | [-8.67,-1.33] | [-9.33,-0.22] | 0/30 | 0/3 |
| ARM2_GRIP16-H16 | 24:60 | -8.00 | 0.000107148 | [-12.00,-4.22] | [-12.22,-3.78] | 0/30 | 0/3 |
| TE_DENSE-H16 | 14:83 | -15.33 | 4.22241e-13 | [-19.33,-11.33] | [-21.56,-10.00] | 0/30 | 0/3 |
| TE_DENSE-ARM4_GRIP32 | 31:78 | -10.44 | 7.73392e-06 | [-14.89,-6.00] | [-17.78,-3.78] | 0/30 | 0/3 |

## All task-level deltas

### H16-H4

| Task | Delta pp |
|---|---:|
| libero_10:task0 | +20.00 |
| libero_10:task1 | +40.00 |
| libero_10:task2 | +13.33 |
| libero_10:task3 | +0.00 |
| libero_10:task4 | +0.00 |
| libero_10:task5 | +6.67 |
| libero_10:task6 | -20.00 |
| libero_10:task7 | +13.33 |
| libero_10:task8 | +6.67 |
| libero_10:task9 | +60.00 |
| libero_goal:task0 | +20.00 |
| libero_goal:task1 | +0.00 |
| libero_goal:task2 | +0.00 |
| libero_goal:task3 | +26.67 |
| libero_goal:task4 | +0.00 |
| libero_goal:task5 | +6.67 |
| libero_goal:task6 | +13.33 |
| libero_goal:task7 | +0.00 |
| libero_goal:task8 | +0.00 |
| libero_goal:task9 | +13.33 |
| libero_spatial:task0 | -6.67 |
| libero_spatial:task1 | +13.33 |
| libero_spatial:task2 | +0.00 |
| libero_spatial:task3 | -13.33 |
| libero_spatial:task4 | +13.33 |
| libero_spatial:task5 | +13.33 |
| libero_spatial:task6 | +0.00 |
| libero_spatial:task7 | +26.67 |
| libero_spatial:task8 | +20.00 |
| libero_spatial:task9 | +0.00 |

### H4-H2

| Task | Delta pp |
|---|---:|
| libero_10:task0 | -6.67 |
| libero_10:task1 | +13.33 |
| libero_10:task2 | +6.67 |
| libero_10:task3 | +0.00 |
| libero_10:task4 | +20.00 |
| libero_10:task5 | +6.67 |
| libero_10:task6 | +20.00 |
| libero_10:task7 | +6.67 |
| libero_10:task8 | +0.00 |
| libero_10:task9 | -13.33 |
| libero_goal:task0 | +20.00 |
| libero_goal:task1 | +6.67 |
| libero_goal:task2 | +0.00 |
| libero_goal:task3 | +6.67 |
| libero_goal:task4 | +0.00 |
| libero_goal:task5 | +6.67 |
| libero_goal:task6 | +0.00 |
| libero_goal:task7 | +6.67 |
| libero_goal:task8 | +0.00 |
| libero_goal:task9 | +6.67 |
| libero_spatial:task0 | +6.67 |
| libero_spatial:task1 | -6.67 |
| libero_spatial:task2 | +13.33 |
| libero_spatial:task3 | +6.67 |
| libero_spatial:task4 | -6.67 |
| libero_spatial:task5 | +6.67 |
| libero_spatial:task6 | +13.33 |
| libero_spatial:task7 | -13.33 |
| libero_spatial:task8 | +0.00 |
| libero_spatial:task9 | +0.00 |

### ARM4_GRIP32-H4

| Task | Delta pp |
|---|---:|
| libero_10:task0 | +33.33 |
| libero_10:task1 | +6.67 |
| libero_10:task2 | +20.00 |
| libero_10:task3 | +0.00 |
| libero_10:task4 | +6.67 |
| libero_10:task5 | +0.00 |
| libero_10:task6 | -13.33 |
| libero_10:task7 | +13.33 |
| libero_10:task8 | +33.33 |
| libero_10:task9 | +33.33 |
| libero_goal:task0 | +0.00 |
| libero_goal:task1 | +0.00 |
| libero_goal:task2 | +0.00 |
| libero_goal:task3 | +6.67 |
| libero_goal:task4 | +0.00 |
| libero_goal:task5 | +0.00 |
| libero_goal:task6 | +0.00 |
| libero_goal:task7 | +0.00 |
| libero_goal:task8 | +0.00 |
| libero_goal:task9 | -6.67 |
| libero_spatial:task0 | +0.00 |
| libero_spatial:task1 | +6.67 |
| libero_spatial:task2 | +0.00 |
| libero_spatial:task3 | -6.67 |
| libero_spatial:task4 | +6.67 |
| libero_spatial:task5 | -13.33 |
| libero_spatial:task6 | -6.67 |
| libero_spatial:task7 | +20.00 |
| libero_spatial:task8 | +0.00 |
| libero_spatial:task9 | +0.00 |

### ARM2_GRIP16-H2

| Task | Delta pp |
|---|---:|
| libero_10:task0 | +6.67 |
| libero_10:task1 | +26.67 |
| libero_10:task2 | +13.33 |
| libero_10:task3 | +6.67 |
| libero_10:task4 | +40.00 |
| libero_10:task5 | +0.00 |
| libero_10:task6 | +6.67 |
| libero_10:task7 | +20.00 |
| libero_10:task8 | +0.00 |
| libero_10:task9 | +20.00 |
| libero_goal:task0 | +0.00 |
| libero_goal:task1 | +0.00 |
| libero_goal:task2 | +0.00 |
| libero_goal:task3 | +6.67 |
| libero_goal:task4 | +0.00 |
| libero_goal:task5 | +0.00 |
| libero_goal:task6 | +0.00 |
| libero_goal:task7 | +0.00 |
| libero_goal:task8 | +0.00 |
| libero_goal:task9 | +13.33 |
| libero_spatial:task0 | +0.00 |
| libero_spatial:task1 | +0.00 |
| libero_spatial:task2 | +6.67 |
| libero_spatial:task3 | +0.00 |
| libero_spatial:task4 | +0.00 |
| libero_spatial:task5 | +0.00 |
| libero_spatial:task6 | +6.67 |
| libero_spatial:task7 | +0.00 |
| libero_spatial:task8 | +6.67 |
| libero_spatial:task9 | -6.67 |

### ARM4_GRIP32-H16

| Task | Delta pp |
|---|---:|
| libero_10:task0 | +13.33 |
| libero_10:task1 | -33.33 |
| libero_10:task2 | +6.67 |
| libero_10:task3 | +0.00 |
| libero_10:task4 | +6.67 |
| libero_10:task5 | -6.67 |
| libero_10:task6 | +6.67 |
| libero_10:task7 | +0.00 |
| libero_10:task8 | +26.67 |
| libero_10:task9 | -26.67 |
| libero_goal:task0 | -20.00 |
| libero_goal:task1 | +0.00 |
| libero_goal:task2 | +0.00 |
| libero_goal:task3 | -20.00 |
| libero_goal:task4 | +0.00 |
| libero_goal:task5 | -6.67 |
| libero_goal:task6 | -13.33 |
| libero_goal:task7 | +0.00 |
| libero_goal:task8 | +0.00 |
| libero_goal:task9 | -20.00 |
| libero_spatial:task0 | +6.67 |
| libero_spatial:task1 | -6.67 |
| libero_spatial:task2 | +0.00 |
| libero_spatial:task3 | +6.67 |
| libero_spatial:task4 | -6.67 |
| libero_spatial:task5 | -26.67 |
| libero_spatial:task6 | -6.67 |
| libero_spatial:task7 | -6.67 |
| libero_spatial:task8 | -20.00 |
| libero_spatial:task9 | +0.00 |

### ARM2_GRIP16-H16

| Task | Delta pp |
|---|---:|
| libero_10:task0 | -6.67 |
| libero_10:task1 | -26.67 |
| libero_10:task2 | -6.67 |
| libero_10:task3 | +6.67 |
| libero_10:task4 | +20.00 |
| libero_10:task5 | -13.33 |
| libero_10:task6 | +6.67 |
| libero_10:task7 | +0.00 |
| libero_10:task8 | -6.67 |
| libero_10:task9 | -26.67 |
| libero_goal:task0 | -40.00 |
| libero_goal:task1 | -6.67 |
| libero_goal:task2 | +0.00 |
| libero_goal:task3 | -26.67 |
| libero_goal:task4 | +0.00 |
| libero_goal:task5 | -13.33 |
| libero_goal:task6 | -13.33 |
| libero_goal:task7 | -6.67 |
| libero_goal:task8 | +0.00 |
| libero_goal:task9 | -6.67 |
| libero_spatial:task0 | +0.00 |
| libero_spatial:task1 | -6.67 |
| libero_spatial:task2 | -6.67 |
| libero_spatial:task3 | +6.67 |
| libero_spatial:task4 | -6.67 |
| libero_spatial:task5 | -20.00 |
| libero_spatial:task6 | -6.67 |
| libero_spatial:task7 | -13.33 |
| libero_spatial:task8 | -13.33 |
| libero_spatial:task9 | -6.67 |

### TE_DENSE-H16

| Task | Delta pp |
|---|---:|
| libero_10:task0 | -33.33 |
| libero_10:task1 | -6.67 |
| libero_10:task2 | -13.33 |
| libero_10:task3 | -66.67 |
| libero_10:task4 | -33.33 |
| libero_10:task5 | -13.33 |
| libero_10:task6 | +6.67 |
| libero_10:task7 | -6.67 |
| libero_10:task8 | -6.67 |
| libero_10:task9 | -6.67 |
| libero_goal:task0 | -26.67 |
| libero_goal:task1 | +0.00 |
| libero_goal:task2 | +6.67 |
| libero_goal:task3 | -46.67 |
| libero_goal:task4 | +0.00 |
| libero_goal:task5 | -6.67 |
| libero_goal:task6 | -26.67 |
| libero_goal:task7 | +0.00 |
| libero_goal:task8 | +0.00 |
| libero_goal:task9 | -13.33 |
| libero_spatial:task0 | -20.00 |
| libero_spatial:task1 | -40.00 |
| libero_spatial:task2 | -13.33 |
| libero_spatial:task3 | -6.67 |
| libero_spatial:task4 | -20.00 |
| libero_spatial:task5 | -20.00 |
| libero_spatial:task6 | -6.67 |
| libero_spatial:task7 | -6.67 |
| libero_spatial:task8 | -26.67 |
| libero_spatial:task9 | -6.67 |

### TE_DENSE-ARM4_GRIP32

| Task | Delta pp |
|---|---:|
| libero_10:task0 | -46.67 |
| libero_10:task1 | +26.67 |
| libero_10:task2 | -20.00 |
| libero_10:task3 | -66.67 |
| libero_10:task4 | -40.00 |
| libero_10:task5 | -6.67 |
| libero_10:task6 | +0.00 |
| libero_10:task7 | -6.67 |
| libero_10:task8 | -33.33 |
| libero_10:task9 | +20.00 |
| libero_goal:task0 | -6.67 |
| libero_goal:task1 | +0.00 |
| libero_goal:task2 | +6.67 |
| libero_goal:task3 | -26.67 |
| libero_goal:task4 | +0.00 |
| libero_goal:task5 | +0.00 |
| libero_goal:task6 | -13.33 |
| libero_goal:task7 | +0.00 |
| libero_goal:task8 | +0.00 |
| libero_goal:task9 | +6.67 |
| libero_spatial:task0 | -26.67 |
| libero_spatial:task1 | -33.33 |
| libero_spatial:task2 | -13.33 |
| libero_spatial:task3 | -13.33 |
| libero_spatial:task4 | -13.33 |
| libero_spatial:task5 | +6.67 |
| libero_spatial:task6 | +0.00 |
| libero_spatial:task7 | +0.00 |
| libero_spatial:task8 | -6.67 |
| libero_spatial:task9 | -6.67 |

## Frozen moderator

Spearman rho `0.1922`, two-sided p `0.308853`, all 30 tasks.

No forbidden post-result method development was launched by this analysis.

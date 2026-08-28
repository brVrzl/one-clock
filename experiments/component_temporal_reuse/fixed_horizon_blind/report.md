# Fixed-horizon ACT blind result

Frozen eight-task, 320-episode confirmatory panel on initial-state IDs 20--29. These tasks were recorded before the custom method results and received no intervention tuning.

| task | Fresh h1 | fixed h8 | fixed h16 | native h100 | h16-only / h100-only |
|---|---:|---:|---:|---:|---:|
| libero_10:task1 | 1/10 | 3/10 | 5/10 | 3/10 | 4/2 |
| libero_10:task9 | 5/10 | 9/10 | 6/10 | 6/10 | 3/3 |
| libero_goal:task0 | 3/10 | 6/10 | 8/10 | 6/10 | 2/0 |
| libero_goal:task3 | 4/10 | 5/10 | 8/10 | 4/10 | 4/0 |
| libero_object:task1 | 4/10 | 8/10 | 6/10 | 4/10 | 3/1 |
| libero_object:task4 | 9/10 | 9/10 | 9/10 | 8/10 | 1/0 |
| libero_spatial:task3 | 10/10 | 9/10 | 9/10 | 4/10 | 5/0 |
| libero_spatial:task7 | 9/10 | 9/10 | 9/10 | 8/10 | 1/0 |
| **pooled** | **45/80** | **58/80** | **60/80** | **43/80** | **23/6 (net +17)** |

## Query efficiency

| method | successes | pooled query rate | mean executed source age |
|---|---:|---:|---:|
| Fresh h1 | 45/80 | 1.0000 | 0.000 |
| fixed h8 | 58/80 | 0.1266 | 3.481 |
| fixed h16 | 60/80 | 0.0647 | 7.395 |
| native h100 | 43/80 | 0.0119 | 46.526 |

## Frozen gate

| criterion | pass |
|---|:---:|
| h16_vs_h100_paired_net_at_least_8 | yes |
| h16_at_least_fresh_minus_4 | yes |
| h16_at_least_h8_minus_2 | yes |
| taskwise_fresh_paired_net_loss_below_2 | yes |
| h16_nonworse_than_h8_on_at_least_6_tasks | yes |
| h16_query_rate_at_most_0.075 | yes |

Advance to SmolVLA cross-policy confirmation: **YES**.

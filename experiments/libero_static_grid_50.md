# LIBERO static horizon landscape — 50 paired states

This combines the existing states 0–19 with the controlled extension states 20–49. It is a diagnostic execution result, not a statistical claim.

States: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]`; total episodes: **1500**; pairing valid: **True**.

## Global fixed

| Global horizon | Successes | Success rate (95% Wilson CI) | Mean success steps | Query rate |
|---:|---:|---:|---:|---:|
| 1 | 29 | 0.580 [0.442, 0.706] | 179.72 | 1.000 |
| 2 | 31 | 0.620 [0.482, 0.741] | 163.81 | 0.501 |
| 4 | 42 | 0.840 [0.715, 0.917] | 153.17 | 0.252 |
| 8 | 45 | 0.900 [0.786, 0.957] | 139.76 | 0.128 |
| 16 | 42 | 0.840 [0.715, 0.917] | 139.55 | 0.065 |

### Group-wise success rate

| arm \ gripper | 1 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.580 | 0.600 | 0.680 | 0.800 | 0.800 |
| 2 | 0.520 | 0.620 | 0.840 | 0.840 | 0.880 |
| 4 | 0.560 | 0.620 | 0.840 | 0.900 | 0.940 |
| 8 | 0.520 | 0.660 | 0.820 | 0.900 | 0.900 |
| 16 | 0.580 | 0.600 | 0.740 | 0.820 | 0.840 |

### Group-wise mean successful completion steps

| arm \ gripper | 1 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 179.72 | 160.00 | 156.85 | 156.65 | 151.65 |
| 2 | 171.23 | 163.81 | 159.76 | 150.50 | 148.64 |
| 4 | 178.61 | 166.81 | 153.17 | 147.64 | 147.32 |
| 8 | 150.08 | 156.30 | 151.95 | 139.76 | 142.02 |
| 16 | 168.79 | 151.83 | 147.89 | 144.68 | 139.55 |

### Group-wise mean policy query rate

| arm \ gripper | 1 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | 0.501 | 0.502 | 0.502 | 0.501 |
| 4 | 1.000 | 0.501 | 0.252 | 0.252 | 0.252 |
| 8 | 1.000 | 0.501 | 0.252 | 0.128 | 0.128 |
| 16 | 1.000 | 0.501 | 0.252 | 0.127 | 0.066 |

## Best configurations

- **best_global:** global_h8
- **best_groupwise:** group_arm4_grip16
- **best_off_diagonal:** group_arm4_grip16

## Diagonal controls

- `global_h1 vs group_arm1_grip1`: all equal = **True**; differences `{'success': [], 'environment_steps': [], 'policy_queries': [], 'policy_query_rate': []}`
- `global_h2 vs group_arm2_grip2`: all equal = **True**; differences `{'success': [], 'environment_steps': [], 'policy_queries': [], 'policy_query_rate': []}`
- `global_h4 vs group_arm4_grip4`: all equal = **True**; differences `{'success': [], 'environment_steps': [], 'policy_queries': [], 'policy_query_rate': []}`
- `global_h8 vs group_arm8_grip8`: all equal = **True**; differences `{'success': [], 'environment_steps': [], 'policy_queries': [], 'policy_query_rate': []}`
- `global_h16 vs group_arm16_grip16`: all equal = **True**; differences `{'success': [], 'environment_steps': [], 'policy_queries': [], 'policy_query_rate': []}`

## Key paired comparisons

- **best_global_vs_best_off_diagonal:** `global_h8` vs `group_arm4_grip16`, counts `{'both_succeed': 45, 'a_only_succeeds': 0, 'b_only_succeeds': 2, 'both_fail': 3}`, exact p=0.5000, difference b−a=0.040
- **closest_query_rate_global_vs_best_off_diagonal:** `global_h4` vs `group_arm4_grip16`, counts `{'both_succeed': 42, 'a_only_succeeds': 0, 'b_only_succeeds': 5, 'both_fail': 3}`, exact p=0.0625, difference b−a=0.100

## Directionality

| Pair | A success | B success | A query rate | B query rate | A-only | B-only | Both fail | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| group_arm1_grip2 vs group_arm2_grip1 | 0.600 | 0.520 | 1.000 | 1.000 | 10 | 6 | 14 | 0.4545 |
| group_arm1_grip4 vs group_arm4_grip1 | 0.680 | 0.560 | 1.000 | 1.000 | 11 | 5 | 11 | 0.2101 |
| group_arm1_grip8 vs group_arm8_grip1 | 0.800 | 0.520 | 1.000 | 1.000 | 17 | 3 | 7 | 0.0026 |
| group_arm1_grip16 vs group_arm16_grip1 | 0.800 | 0.580 | 1.000 | 1.000 | 16 | 5 | 5 | 0.0266 |
| group_arm2_grip4 vs group_arm4_grip2 | 0.840 | 0.620 | 0.502 | 0.501 | 13 | 2 | 6 | 0.0074 |
| group_arm2_grip8 vs group_arm8_grip2 | 0.840 | 0.660 | 0.501 | 0.501 | 12 | 3 | 5 | 0.0352 |
| group_arm2_grip16 vs group_arm16_grip2 | 0.880 | 0.600 | 0.501 | 0.501 | 16 | 2 | 4 | 0.0013 |
| group_arm4_grip8 vs group_arm8_grip4 | 0.900 | 0.820 | 0.252 | 0.252 | 5 | 1 | 4 | 0.2188 |
| group_arm4_grip16 vs group_arm16_grip4 | 0.940 | 0.740 | 0.252 | 0.252 | 10 | 0 | 3 | 0.0020 |
| group_arm8_grip16 vs group_arm16_grip8 | 0.900 | 0.820 | 0.128 | 0.127 | 6 | 2 | 3 | 0.2891 |

## Pareto frontier

- global_h16
- group_arm4_grip16
- group_arm8_grip16
- group_arm16_grip16

## 20-state vs 50-state comparison

- Best global: `global_h8` → `['global_h8']`.
- Best off-diagonal: `group_arm4_grip16` → `['group_arm4_grip16']`.
- Pareto frontier: `['global_h16', 'group_arm16_grip16', 'group_arm4_grip16', 'group_arm8_grip16']` → `['global_h16', 'group_arm4_grip16', 'group_arm8_grip16', 'group_arm16_grip16']`.

## Exploratory trace comparisons

LIBERO/robosuite PandaGripper source semantics were verified: gripper command −1 means open, +1 means closed, and zero produces no sign change/holds the current gripper action. Trace metrics are post-hoc execution-pattern diagnostics only.

| Comparison | Side | Gripper TV | Mean gripper Δ | Gripper sign changes | Arm mean L2 Δ | Arm total L2 |
|---|---|---:|---:|---:|---:|---:|
| best_global_vs_best_off_diagonal | a `global_h8` | 8.805 | 0.053 | 3.84 | 0.077 | 11.687 |
| best_global_vs_best_off_diagonal | b `group_arm4_grip16` | 8.073 | 0.048 | 3.64 | 0.082 | 12.847 |
| best_off_diagonal_vs_reversed | a `group_arm4_grip16` | 8.073 | 0.048 | 3.64 | 0.082 | 12.847 |
| best_off_diagonal_vs_reversed | b `group_arm16_grip4` | 13.681 | 0.067 | 5.74 | 0.085 | 16.589 |
| closest_query_rate_global_vs_best_off_diagonal | a `global_h4` | 17.519 | 0.090 | 8.12 | 0.085 | 15.177 |
| closest_query_rate_global_vs_best_off_diagonal | b `group_arm4_grip16` | 8.073 | 0.048 | 3.64 | 0.082 | 12.847 |

The JSON artifact contains per-configuration Wilson intervals, exact success vectors, paired diagnostics, Pareto dominance, and full trace aggregates.

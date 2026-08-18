# LIBERO static horizon landscape

This is the complete paired static sweep on the frozen LIBERO ACT checkpoint. It is a diagnostic execution result, not a statistical claim.

Pairing valid: **True**; official states available: **50**; states used: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; total episodes: **600**.

## Global fixed

| Global horizon | Successes | Success rate | Mean success steps | Query rate |
|---:|---:|---:|---:|---:|
| 1 | 14 | 0.700 | 168.500 | 1.000 |
| 2 | 13 | 0.650 | 173.000 | 0.501 |
| 4 | 17 | 0.850 | 152.235 | 0.252 |
| 8 | 18 | 0.900 | 141.611 | 0.128 |
| 16 | 17 | 0.850 | 143.235 | 0.066 |

### Group-wise success rate

| arm \ gripper | 1 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.700 | 0.650 | 0.700 | 0.800 | 0.800 |
| 2 | 0.500 | 0.650 | 0.900 | 0.900 | 0.900 |
| 4 | 0.600 | 0.650 | 0.850 | 0.850 | 0.950 |
| 8 | 0.550 | 0.650 | 0.750 | 0.900 | 0.900 |
| 16 | 0.650 | 0.650 | 0.750 | 0.800 | 0.850 |

### Group-wise mean successful completion steps

| arm \ gripper | 1 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 168.50 | 156.46 | 158.29 | 156.69 | 148.94 |
| 2 | 166.40 | 173.00 | 167.56 | 154.39 | 154.61 |
| 4 | 189.75 | 176.54 | 152.24 | 146.06 | 149.21 |
| 8 | 155.27 | 156.15 | 163.93 | 141.61 | 145.22 |
| 16 | 166.54 | 151.54 | 144.53 | 153.75 | 143.24 |

### Group-wise mean policy query rate

| arm \ gripper | 1 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | 0.501 | 0.502 | 0.501 | 0.501 |
| 4 | 1.000 | 0.501 | 0.252 | 0.252 | 0.253 |
| 8 | 1.000 | 0.501 | 0.252 | 0.128 | 0.127 |
| 16 | 1.000 | 0.501 | 0.252 | 0.127 | 0.066 |

## Best configurations

- **best_global:** global_h8
- **best_groupwise:** group_arm4_grip16
- **best_off_diagonal:** group_arm4_grip16

## Diagonal controls

- `global_h1 vs group_arm1_grip1`: all compared per-state fields equal = **True**. Differences: `{'success': [], 'environment_steps': [], 'policy_queries': [], 'policy_query_rate': []}`
- `global_h2 vs group_arm2_grip2`: all compared per-state fields equal = **True**. Differences: `{'success': [], 'environment_steps': [], 'policy_queries': [], 'policy_query_rate': []}`
- `global_h4 vs group_arm4_grip4`: all compared per-state fields equal = **True**. Differences: `{'success': [], 'environment_steps': [], 'policy_queries': [], 'policy_query_rate': []}`
- `global_h8 vs group_arm8_grip8`: all compared per-state fields equal = **True**. Differences: `{'success': [], 'environment_steps': [], 'policy_queries': [], 'policy_query_rate': []}`
- `global_h16 vs group_arm16_grip16`: all compared per-state fields equal = **True**. Differences: `{'success': [], 'environment_steps': [], 'policy_queries': [], 'policy_query_rate': []}`

## Best-global vs best-off-diagonal paired contingencies

- `global_h8` vs `group_arm4_grip16`: `{'both_succeed': 18, 'groupwise_only_succeeds': 1, 'global_only_succeeds': 0, 'both_fail': 1}`

## Budget-controlled best off-diagonal diagnostic

- `global_h4` (17/20, query rate 0.252) vs `group_arm4_grip16` (19/20, query rate 0.252): `{'both_succeed': 17, 'groupwise_only_succeeds': 2, 'global_only_succeeds': 0, 'both_fail': 1}`

## Directional paired diagnostics

- `group_arm8_grip2` (13/20) vs `group_arm2_grip8` (18/20): `{'both_succeed': 13, 'groupwise_only_succeeds': 5, 'global_only_succeeds': 0, 'both_fail': 2}`
- `group_arm16_grip2` (13/20) vs `group_arm2_grip16` (18/20): `{'both_succeed': 13, 'groupwise_only_succeeds': 5, 'global_only_succeeds': 0, 'both_fail': 2}`
- `group_arm8_grip4` (15/20) vs `group_arm4_grip8` (17/20): `{'both_succeed': 14, 'groupwise_only_succeeds': 3, 'global_only_succeeds': 1, 'both_fail': 2}`

## Configuration details

The JSON artifact contains environment-step means/medians, successful-step means/medians, query budgets, source ages, and success/failure state IDs for every cell.

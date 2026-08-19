# LIBERO Object task 5: pick_up_the_tomato_sauce_and_place_it_in_the_basket

Paired states: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; configurations: 12 executed cells plus a diagonal `(4,4)` alias where the duplicate raw run was omitted.
Standard global `h=8` sanity: 5/5; mean successful steps=125.6.

| Configuration | Successes | Rate | 95% CI | Mean success steps | Query rate |
|---|---:|---:|---|---:|---:|
| global_h4 | 10 | 0.500 | [0.299, 0.701] | 155.3 | 0.251 |
| group_arm2_grip2 | 6 | 0.300 | [0.145, 0.519] | 192.0 | 0.500 |
| group_arm2_grip8 | 11 | 0.550 | [0.342, 0.742] | 146.1 | 0.501 |
| group_arm2_grip16 | 15 | 0.750 | [0.531, 0.888] | 163.3 | 0.501 |
| group_arm4_grip4 | 10 | 0.500 | [0.299, 0.701] | 155.3 | 0.251 |
| group_arm4_grip16 | 17 | 0.850 | [0.640, 0.948] | 163.8 | 0.252 |
| group_arm8_grip2 | 7 | 0.350 | [0.181, 0.567] | 160.3 | 0.500 |
| group_arm8_grip8 | 15 | 0.750 | [0.531, 0.888] | 131.4 | 0.128 |
| group_arm8_grip16 | 17 | 0.850 | [0.640, 0.948] | 145.1 | 0.128 |
| group_arm16_grip2 | 9 | 0.450 | [0.258, 0.658] | 143.4 | 0.501 |
| group_arm16_grip4 | 15 | 0.750 | [0.531, 0.888] | 141.4 | 0.252 |
| group_arm16_grip8 | 18 | 0.900 | [0.699, 0.972] | 131.6 | 0.129 |
| group_arm16_grip16 | 17 | 0.850 | [0.640, 0.948] | 124.8 | 0.065 |

## Selected results

- Best global: global_h16
- Best group-wise: group_arm16_grip8
- Best off-diagonal: group_arm16_grip8
- Best-global vs best-off-diagonal paired comparisons: 1
- Best group-wise class: **off-diagonal**
- Off-diagonal Pareto points: group_arm16_grip8
- Budget-matched comparison: `global_h8` vs `group_arm16_grip8`, counts `{'both_succeed': 13, 'a_only_succeeds': 2, 'b_only_succeeds': 5, 'both_fail': 0}`, exact paired p=0.4531.

Success vectors and full per-configuration records are in the JSON artifact.

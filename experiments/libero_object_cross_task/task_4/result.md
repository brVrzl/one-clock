# LIBERO Object task 4: pick_up_the_ketchup_and_place_it_in_the_basket

Paired states: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; configurations: 12 executed cells plus a diagonal `(4,4)` alias where the duplicate raw run was omitted.
Standard global `h=8` sanity: 3/5; mean successful steps=139.3.

| Configuration | Successes | Rate | 95% CI | Mean success steps | Query rate |
|---|---:|---:|---|---:|---:|
| global_h4 | 16 | 0.800 | [0.584, 0.919] | 146.8 | 0.252 |
| group_arm2_grip2 | 10 | 0.500 | [0.299, 0.701] | 145.5 | 0.501 |
| group_arm2_grip8 | 16 | 0.800 | [0.584, 0.919] | 139.2 | 0.501 |
| group_arm2_grip16 | 17 | 0.850 | [0.640, 0.948] | 145.9 | 0.502 |
| group_arm4_grip4 | 16 | 0.800 | [0.584, 0.919] | 146.8 | 0.252 |
| group_arm4_grip16 | 16 | 0.800 | [0.584, 0.919] | 139.4 | 0.252 |
| group_arm8_grip2 | 13 | 0.650 | [0.433, 0.819] | 173.5 | 0.501 |
| group_arm8_grip8 | 15 | 0.750 | [0.531, 0.888] | 137.1 | 0.126 |
| group_arm8_grip16 | 15 | 0.750 | [0.531, 0.888] | 138.9 | 0.127 |
| group_arm16_grip2 | 12 | 0.600 | [0.387, 0.781] | 160.4 | 0.501 |
| group_arm16_grip4 | 15 | 0.750 | [0.531, 0.888] | 148.7 | 0.251 |
| group_arm16_grip8 | 16 | 0.800 | [0.584, 0.919] | 139.6 | 0.127 |
| group_arm16_grip16 | 16 | 0.800 | [0.584, 0.919] | 138.4 | 0.065 |

## Selected results

- Best global: global_h4, global_h16
- Best group-wise: group_arm2_grip16
- Best off-diagonal: group_arm2_grip16
- Best-global vs best-off-diagonal paired comparisons: 2
- Best group-wise class: **off-diagonal**
- Off-diagonal Pareto points: group_arm2_grip16
- Budget-matched comparison: `global_h2` vs `group_arm2_grip16`, counts `{'both_succeed': 10, 'a_only_succeeds': 0, 'b_only_succeeds': 7, 'both_fail': 3}`, exact paired p=0.0156.

Success vectors and full per-configuration records are in the JSON artifact.

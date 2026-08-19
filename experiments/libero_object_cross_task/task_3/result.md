# LIBERO Object task 3: pick_up_the_bbq_sauce_and_place_it_in_the_basket

Paired states: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; configurations: 12 executed cells plus a diagonal `(4,4)` alias where the duplicate raw run was omitted.
Standard global `h=8` sanity: 5/5; mean successful steps=128.8.

| Configuration | Successes | Rate | 95% CI | Mean success steps | Query rate |
|---|---:|---:|---|---:|---:|
| global_h4 | 15 | 0.750 | [0.531, 0.888] | 128.7 | 0.252 |
| group_arm2_grip2 | 16 | 0.800 | [0.584, 0.919] | 129.2 | 0.501 |
| group_arm2_grip8 | 17 | 0.850 | [0.640, 0.948] | 128.8 | 0.501 |
| group_arm2_grip16 | 16 | 0.800 | [0.584, 0.919] | 128.8 | 0.501 |
| group_arm4_grip4 | 15 | 0.750 | [0.531, 0.888] | 128.7 | 0.252 |
| group_arm4_grip16 | 16 | 0.800 | [0.584, 0.919] | 129.9 | 0.252 |
| group_arm8_grip2 | 12 | 0.600 | [0.387, 0.781] | 130.6 | 0.501 |
| group_arm8_grip8 | 16 | 0.800 | [0.584, 0.919] | 132.1 | 0.127 |
| group_arm8_grip16 | 16 | 0.800 | [0.584, 0.919] | 132.0 | 0.127 |
| group_arm16_grip2 | 13 | 0.650 | [0.433, 0.819] | 133.8 | 0.501 |
| group_arm16_grip4 | 15 | 0.750 | [0.531, 0.888] | 129.7 | 0.252 |
| group_arm16_grip8 | 18 | 0.900 | [0.699, 0.972] | 137.7 | 0.127 |
| group_arm16_grip16 | 18 | 0.900 | [0.699, 0.972] | 131.1 | 0.065 |

## Selected results

- Best global: global_h16
- Best group-wise: group_arm16_grip16, group_arm16_grip8
- Best off-diagonal: group_arm16_grip8
- Best-global vs best-off-diagonal paired comparisons: 1
- Best group-wise class: **tied**
- Off-diagonal Pareto points: none
- Budget-matched comparison: `global_h8` vs `group_arm16_grip8`, counts `{'both_succeed': 16, 'a_only_succeeds': 0, 'b_only_succeeds': 2, 'both_fail': 2}`, exact paired p=0.5000.

Success vectors and full per-configuration records are in the JSON artifact.

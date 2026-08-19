# LIBERO Object task 2: pick_up_the_salad_dressing_and_place_it_in_the_basket

Paired states: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; configurations: 12 executed cells plus a diagonal `(4,4)` alias where the duplicate raw run was omitted.
Standard global `h=8` sanity: 4/5; mean successful steps=107.5.

| Configuration | Successes | Rate | 95% CI | Mean success steps | Query rate |
|---|---:|---:|---|---:|---:|
| global_h4 | 15 | 0.750 | [0.531, 0.888] | 109.5 | 0.252 |
| group_arm2_grip2 | 14 | 0.700 | [0.481, 0.855] | 113.3 | 0.501 |
| group_arm2_grip8 | 15 | 0.750 | [0.531, 0.888] | 111.1 | 0.501 |
| group_arm2_grip16 | 16 | 0.800 | [0.584, 0.919] | 114.8 | 0.501 |
| group_arm4_grip4 | 15 | 0.750 | [0.531, 0.888] | 109.5 | 0.252 |
| group_arm4_grip16 | 16 | 0.800 | [0.584, 0.919] | 116.9 | 0.252 |
| group_arm8_grip2 | 12 | 0.600 | [0.387, 0.781] | 112.9 | 0.501 |
| group_arm8_grip8 | 14 | 0.700 | [0.481, 0.855] | 109.6 | 0.126 |
| group_arm8_grip16 | 14 | 0.700 | [0.481, 0.855] | 109.9 | 0.127 |
| group_arm16_grip2 | 13 | 0.650 | [0.433, 0.819] | 120.8 | 0.501 |
| group_arm16_grip4 | 13 | 0.650 | [0.433, 0.819] | 119.2 | 0.252 |
| group_arm16_grip8 | 14 | 0.700 | [0.481, 0.855] | 120.4 | 0.127 |
| group_arm16_grip16 | 14 | 0.700 | [0.481, 0.855] | 113.1 | 0.065 |

## Selected results

- Best global: global_h4
- Best group-wise: group_arm2_grip16, group_arm4_grip16
- Best off-diagonal: group_arm2_grip16, group_arm4_grip16
- Best-global vs best-off-diagonal paired comparisons: 2
- Best group-wise class: **off-diagonal**
- Off-diagonal Pareto points: group_arm4_grip16
- Budget-matched comparison: `global_h2` vs `group_arm2_grip16`, counts `{'both_succeed': 14, 'a_only_succeeds': 0, 'b_only_succeeds': 2, 'both_fail': 4}`, exact paired p=0.5000.

Success vectors and full per-configuration records are in the JSON artifact.

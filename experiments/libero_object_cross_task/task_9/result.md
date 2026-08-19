# LIBERO Object task 9: pick_up_the_orange_juice_and_place_it_in_the_basket

Paired states: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; configurations: 12 executed cells plus a diagonal `(4,4)` alias where the duplicate raw run was omitted.
Standard global `h=8` sanity: 4/5; mean successful steps=112.5.

| Configuration | Successes | Rate | 95% CI | Mean success steps | Query rate |
|---|---:|---:|---|---:|---:|
| global_h4 | 18 | 0.900 | [0.699, 0.972] | 115.2 | 0.252 |
| group_arm2_grip2 | 18 | 0.900 | [0.699, 0.972] | 115.9 | 0.502 |
| group_arm2_grip8 | 19 | 0.950 | [0.764, 0.991] | 119.7 | 0.502 |
| group_arm2_grip16 | 18 | 0.900 | [0.699, 0.972] | 116.0 | 0.502 |
| group_arm4_grip4 | 18 | 0.900 | [0.699, 0.972] | 115.2 | 0.252 |
| group_arm4_grip16 | 20 | 1.000 | [0.839, 1.000] | 122.0 | 0.254 |
| group_arm8_grip2 | 17 | 0.850 | [0.640, 0.948] | 126.5 | 0.502 |
| group_arm8_grip8 | 18 | 0.900 | [0.699, 0.972] | 119.6 | 0.128 |
| group_arm8_grip16 | 18 | 0.900 | [0.699, 0.972] | 120.0 | 0.128 |
| group_arm16_grip2 | 19 | 0.950 | [0.764, 0.991] | 127.4 | 0.501 |
| group_arm16_grip4 | 19 | 0.950 | [0.764, 0.991] | 126.6 | 0.252 |
| group_arm16_grip8 | 20 | 1.000 | [0.839, 1.000] | 130.9 | 0.128 |
| group_arm16_grip16 | 17 | 0.850 | [0.640, 0.948] | 117.2 | 0.065 |

## Selected results

- Best global: global_h2, global_h4, global_h8
- Best group-wise: group_arm16_grip8, group_arm4_grip16
- Best off-diagonal: group_arm16_grip8, group_arm4_grip16
- Best-global vs best-off-diagonal paired comparisons: 6
- Best group-wise class: **off-diagonal**
- Off-diagonal Pareto points: group_arm8_grip16, group_arm16_grip8
- Budget-matched comparison: `global_h8` vs `group_arm16_grip8`, counts `{'both_succeed': 18, 'a_only_succeeds': 0, 'b_only_succeeds': 2, 'both_fail': 0}`, exact paired p=0.5000.

Success vectors and full per-configuration records are in the JSON artifact.

# LIBERO Object task 1: pick_up_the_cream_cheese_and_place_it_in_the_basket

Paired states: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; configurations: 13 executed cells plus a diagonal `(4,4)` alias where the duplicate raw run was omitted.
Standard global `h=8` sanity: 3/5; mean successful steps=142.3.

| Configuration | Successes | Rate | 95% CI | Mean success steps | Query rate |
|---|---:|---:|---|---:|---:|
| global_h4 | 10 | 0.500 | [0.299, 0.701] | 146.5 | 0.251 |
| group_arm2_grip2 | 6 | 0.300 | [0.145, 0.519] | 144.7 | 0.500 |
| group_arm2_grip8 | 9 | 0.450 | [0.258, 0.658] | 139.9 | 0.500 |
| group_arm2_grip16 | 14 | 0.700 | [0.481, 0.855] | 158.9 | 0.501 |
| group_arm4_grip4 | 10 | 0.500 | [0.299, 0.701] | 146.5 | 0.251 |
| group_arm4_grip16 | 12 | 0.600 | [0.387, 0.781] | 143.0 | 0.251 |
| group_arm8_grip2 | 6 | 0.300 | [0.145, 0.519] | 153.8 | 0.501 |
| group_arm8_grip8 | 9 | 0.450 | [0.258, 0.658] | 150.1 | 0.126 |
| group_arm8_grip16 | 13 | 0.650 | [0.433, 0.819] | 155.2 | 0.126 |
| group_arm16_grip2 | 4 | 0.200 | [0.081, 0.416] | 167.2 | 0.500 |
| group_arm16_grip4 | 6 | 0.300 | [0.145, 0.519] | 128.7 | 0.250 |
| group_arm16_grip8 | 7 | 0.350 | [0.181, 0.567] | 154.7 | 0.125 |
| group_arm16_grip16 | 10 | 0.500 | [0.299, 0.701] | 166.2 | 0.065 |

## Selected results

- Best global: global_h4, global_h16
- Best group-wise: group_arm2_grip16
- Best off-diagonal: group_arm2_grip16
- Best-global vs best-off-diagonal paired comparisons: 2
- Best group-wise class: **off-diagonal**
- Off-diagonal Pareto points: group_arm8_grip16, group_arm2_grip16
- Budget-matched comparison: `global_h2` vs `group_arm2_grip16`, counts `{'both_succeed': 6, 'a_only_succeeds': 0, 'b_only_succeeds': 8, 'both_fail': 6}`, exact paired p=0.0078.

Success vectors and full per-configuration records are in the JSON artifact.

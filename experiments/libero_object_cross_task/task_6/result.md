# LIBERO Object task 6: pick_up_the_butter_and_place_it_in_the_basket

Paired states: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; configurations: 12 executed cells plus a diagonal `(4,4)` alias where the duplicate raw run was omitted.
Standard global `h=8` sanity: 3/5; mean successful steps=137.3.

| Configuration | Successes | Rate | 95% CI | Mean success steps | Query rate |
|---|---:|---:|---|---:|---:|
| global_h4 | 10 | 0.500 | [0.299, 0.701] | 170.8 | 0.251 |
| group_arm2_grip2 | 8 | 0.400 | [0.219, 0.613] | 170.8 | 0.500 |
| group_arm2_grip8 | 9 | 0.450 | [0.258, 0.658] | 160.4 | 0.501 |
| group_arm2_grip16 | 9 | 0.450 | [0.258, 0.658] | 157.0 | 0.501 |
| group_arm4_grip4 | 10 | 0.500 | [0.299, 0.701] | 170.8 | 0.251 |
| group_arm4_grip16 | 11 | 0.550 | [0.342, 0.742] | 173.0 | 0.251 |
| group_arm8_grip2 | 7 | 0.350 | [0.181, 0.567] | 140.9 | 0.500 |
| group_arm8_grip8 | 9 | 0.450 | [0.258, 0.658] | 140.7 | 0.125 |
| group_arm8_grip16 | 13 | 0.650 | [0.433, 0.819] | 160.8 | 0.126 |
| group_arm16_grip2 | 10 | 0.500 | [0.299, 0.701] | 145.5 | 0.500 |
| group_arm16_grip4 | 11 | 0.550 | [0.342, 0.742] | 149.2 | 0.251 |
| group_arm16_grip8 | 11 | 0.550 | [0.342, 0.742] | 145.0 | 0.126 |
| group_arm16_grip16 | 13 | 0.650 | [0.433, 0.819] | 157.7 | 0.064 |

## Selected results

- Best global: global_h16
- Best group-wise: group_arm16_grip16, group_arm8_grip16
- Best off-diagonal: group_arm8_grip16
- Best-global vs best-off-diagonal paired comparisons: 1
- Best group-wise class: **tied**
- Off-diagonal Pareto points: none
- Budget-matched comparison: `global_h8` vs `group_arm8_grip16`, counts `{'both_succeed': 9, 'a_only_succeeds': 0, 'b_only_succeeds': 4, 'both_fail': 7}`, exact paired p=0.1250.

Success vectors and full per-configuration records are in the JSON artifact.

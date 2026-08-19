# LIBERO Object task 7: pick_up_the_milk_and_place_it_in_the_basket

Paired states: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; configurations: 12 executed cells plus a diagonal `(4,4)` alias where the duplicate raw run was omitted.
Standard global `h=8` sanity: 2/5; mean successful steps=119.0.

| Configuration | Successes | Rate | 95% CI | Mean success steps | Query rate |
|---|---:|---:|---|---:|---:|
| global_h4 | 10 | 0.500 | [0.299, 0.701] | 118.0 | 0.251 |
| group_arm2_grip2 | 10 | 0.500 | [0.299, 0.701] | 138.6 | 0.500 |
| group_arm2_grip8 | 12 | 0.600 | [0.387, 0.781] | 127.8 | 0.501 |
| group_arm2_grip16 | 10 | 0.500 | [0.299, 0.701] | 125.8 | 0.501 |
| group_arm4_grip4 | 10 | 0.500 | [0.299, 0.701] | 118.0 | 0.251 |
| group_arm4_grip16 | 13 | 0.650 | [0.433, 0.819] | 130.8 | 0.251 |
| group_arm8_grip2 | 11 | 0.550 | [0.342, 0.742] | 122.0 | 0.501 |
| group_arm8_grip8 | 11 | 0.550 | [0.342, 0.742] | 125.7 | 0.126 |
| group_arm8_grip16 | 11 | 0.550 | [0.342, 0.742] | 124.5 | 0.126 |
| group_arm16_grip2 | 9 | 0.450 | [0.258, 0.658] | 129.3 | 0.501 |
| group_arm16_grip4 | 9 | 0.450 | [0.258, 0.658] | 121.9 | 0.251 |
| group_arm16_grip8 | 11 | 0.550 | [0.342, 0.742] | 126.2 | 0.127 |
| group_arm16_grip16 | 11 | 0.550 | [0.342, 0.742] | 130.1 | 0.065 |

## Selected results

- Best global: global_h8, global_h16
- Best group-wise: group_arm4_grip16
- Best off-diagonal: group_arm4_grip16
- Best-global vs best-off-diagonal paired comparisons: 2
- Best group-wise class: **off-diagonal**
- Off-diagonal Pareto points: group_arm4_grip16
- Budget-matched comparison: `global_h4` vs `group_arm4_grip16`, counts `{'both_succeed': 10, 'a_only_succeeds': 0, 'b_only_succeeds': 3, 'both_fail': 7}`, exact paired p=0.2500.

Success vectors and full per-configuration records are in the JSON artifact.

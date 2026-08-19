# LIBERO Object task 8: pick_up_the_chocolate_pudding_and_place_it_in_the_basket

Paired states: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; configurations: 12 executed cells plus a diagonal `(4,4)` alias where the duplicate raw run was omitted.
Standard global `h=8` sanity: 1/5; mean successful steps=165.0.

| Configuration | Successes | Rate | 95% CI | Mean success steps | Query rate |
|---|---:|---:|---|---:|---:|
| global_h4 | 8 | 0.400 | [0.219, 0.613] | 152.6 | 0.251 |
| group_arm2_grip2 | 8 | 0.400 | [0.219, 0.613] | 175.8 | 0.500 |
| group_arm2_grip8 | 8 | 0.400 | [0.219, 0.613] | 143.1 | 0.501 |
| group_arm2_grip16 | 7 | 0.350 | [0.181, 0.567] | 142.9 | 0.500 |
| group_arm4_grip4 | 8 | 0.400 | [0.219, 0.613] | 152.6 | 0.251 |
| group_arm4_grip16 | 7 | 0.350 | [0.181, 0.567] | 141.0 | 0.250 |
| group_arm8_grip2 | 8 | 0.400 | [0.219, 0.613] | 164.9 | 0.501 |
| group_arm8_grip8 | 7 | 0.350 | [0.181, 0.567] | 149.4 | 0.125 |
| group_arm8_grip16 | 6 | 0.300 | [0.145, 0.519] | 146.2 | 0.125 |
| group_arm16_grip2 | 7 | 0.350 | [0.181, 0.567] | 180.6 | 0.501 |
| group_arm16_grip4 | 7 | 0.350 | [0.181, 0.567] | 169.0 | 0.251 |
| group_arm16_grip8 | 5 | 0.250 | [0.112, 0.469] | 144.8 | 0.126 |
| group_arm16_grip16 | 7 | 0.350 | [0.181, 0.567] | 153.9 | 0.065 |

## Selected results

- Best global: global_h2, global_h4
- Best group-wise: group_arm2_grip2, group_arm2_grip8, group_arm8_grip2, group_arm4_grip4
- Best off-diagonal: group_arm2_grip8, group_arm8_grip2
- Best-global vs best-off-diagonal paired comparisons: 4
- Best group-wise class: **tied**
- Off-diagonal Pareto points: none
- Budget-matched comparison: `global_h2` vs `group_arm2_grip8`, counts `{'both_succeed': 8, 'a_only_succeeds': 0, 'b_only_succeeds': 0, 'both_fail': 12}`, exact paired p=1.0000.

Success vectors and full per-configuration records are in the JSON artifact.

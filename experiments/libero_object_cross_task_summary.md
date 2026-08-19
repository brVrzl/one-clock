# LIBERO Object cross-task coarse horizon diagnostic

This is a paired, 20-state-per-task diagnostic using the frozen ACT checkpoint. Task 0 was not rerun. Diagonal group-wise entries `(2,2)`, `(8,8)`, and `(16,16)` are the verified global-equivalent controls; `(4,4)` is a raw control on task 1 and a documented alias of global `h=4` elsewhere.

Tasks evaluated: **9**; controlled episodes: **2180**; macro best-global success: **0.700**; macro best-off-diagonal success: **0.761**.

## Per-task summary

| ID | Task | Sanity h8 | Mean sanity success steps | Best global | Best off-diagonal | Difference | Offdiag frontier |
|---:|---|---:|---:|---|---|---:|---|
| 1 | pick_up_the_cream_cheese_and_place_it_in_the_basket | 3/5 | 142.3 | h=4, h=16 (0.500) | (2,16) (0.700) | 0.200 | yes |
| 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket | 4/5 | 107.5 | h=4 (0.750) | (2,16), (4,16) (0.800) | 0.050 | yes |
| 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket | 5/5 | 128.8 | h=16 (0.900) | (16,8) (0.900) | 0.000 | no |
| 4 | pick_up_the_ketchup_and_place_it_in_the_basket | 3/5 | 139.3 | h=4, h=16 (0.800) | (2,16) (0.850) | 0.050 | yes |
| 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket | 5/5 | 125.6 | h=16 (0.850) | (16,8) (0.900) | 0.050 | yes |
| 6 | pick_up_the_butter_and_place_it_in_the_basket | 3/5 | 137.3 | h=16 (0.650) | (8,16) (0.650) | 0.000 | no |
| 7 | pick_up_the_milk_and_place_it_in_the_basket | 2/5 | 119.0 | h=8, h=16 (0.550) | (4,16) (0.650) | 0.100 | yes |
| 8 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | 1/5 | 165.0 | h=2, h=4 (0.400) | (2,8), (8,2) (0.400) | 0.000 | no |
| 9 | pick_up_the_orange_juice_and_place_it_in_the_basket | 4/5 | 112.5 | h=2, h=4, h=8 (0.900) | (16,8), (4,16) (1.000) | 0.100 | yes |

## Hypothesis diagnostics

- Best group-wise class counts: diagonal=0, off-diagonal=6, tied=3
- Tasks with at least one off-diagonal Pareto point: 6 / 9
- Tasks with an off-diagonal point strictly improving a global point at no higher query rate: 6 / 9

## Symmetric directionality

| Task | Pair | Left rate | Right rate | Query rates | Winner | Paired counts | Exact p |
|---:|---|---:|---:|---|---|---|---:|
| 1 | (2,8) vs (8,2) | 0.450 | 0.300 | 0.500 / 0.501 | left | {'both_succeed': 6, 'a_only_succeeds': 3, 'b_only_succeeds': 0, 'both_fail': 11} | 0.2500 |
| 1 | (2,16) vs (16,2) | 0.700 | 0.200 | 0.501 / 0.500 | left | {'both_succeed': 4, 'a_only_succeeds': 10, 'b_only_succeeds': 0, 'both_fail': 6} | 0.0020 |
| 1 | (8,16) vs (16,8) | 0.650 | 0.350 | 0.126 / 0.125 | left | {'both_succeed': 7, 'a_only_succeeds': 6, 'b_only_succeeds': 0, 'both_fail': 7} | 0.0312 |
| 1 | (4,16) vs (16,4) | 0.600 | 0.300 | 0.251 / 0.250 | left | {'both_succeed': 6, 'a_only_succeeds': 6, 'b_only_succeeds': 0, 'both_fail': 8} | 0.0312 |
| 2 | (2,8) vs (8,2) | 0.750 | 0.600 | 0.501 / 0.501 | left | {'both_succeed': 12, 'a_only_succeeds': 3, 'b_only_succeeds': 0, 'both_fail': 5} | 0.2500 |
| 2 | (2,16) vs (16,2) | 0.800 | 0.650 | 0.501 / 0.501 | left | {'both_succeed': 13, 'a_only_succeeds': 3, 'b_only_succeeds': 0, 'both_fail': 4} | 0.2500 |
| 2 | (8,16) vs (16,8) | 0.700 | 0.700 | 0.127 / 0.127 | tie | {'both_succeed': 13, 'a_only_succeeds': 1, 'b_only_succeeds': 1, 'both_fail': 5} | 1.0000 |
| 2 | (4,16) vs (16,4) | 0.800 | 0.650 | 0.252 / 0.252 | left | {'both_succeed': 13, 'a_only_succeeds': 3, 'b_only_succeeds': 0, 'both_fail': 4} | 0.2500 |
| 3 | (2,8) vs (8,2) | 0.850 | 0.600 | 0.501 / 0.501 | left | {'both_succeed': 12, 'a_only_succeeds': 5, 'b_only_succeeds': 0, 'both_fail': 3} | 0.0625 |
| 3 | (2,16) vs (16,2) | 0.800 | 0.650 | 0.501 / 0.501 | left | {'both_succeed': 13, 'a_only_succeeds': 3, 'b_only_succeeds': 0, 'both_fail': 4} | 0.2500 |
| 3 | (8,16) vs (16,8) | 0.800 | 0.900 | 0.127 / 0.127 | right | {'both_succeed': 16, 'a_only_succeeds': 0, 'b_only_succeeds': 2, 'both_fail': 2} | 0.5000 |
| 3 | (4,16) vs (16,4) | 0.800 | 0.750 | 0.252 / 0.252 | left | {'both_succeed': 13, 'a_only_succeeds': 3, 'b_only_succeeds': 2, 'both_fail': 2} | 1.0000 |
| 4 | (2,8) vs (8,2) | 0.800 | 0.650 | 0.501 / 0.501 | left | {'both_succeed': 13, 'a_only_succeeds': 3, 'b_only_succeeds': 0, 'both_fail': 4} | 0.2500 |
| 4 | (2,16) vs (16,2) | 0.850 | 0.600 | 0.502 / 0.501 | left | {'both_succeed': 11, 'a_only_succeeds': 6, 'b_only_succeeds': 1, 'both_fail': 2} | 0.1250 |
| 4 | (8,16) vs (16,8) | 0.750 | 0.800 | 0.127 / 0.127 | right | {'both_succeed': 15, 'a_only_succeeds': 0, 'b_only_succeeds': 1, 'both_fail': 4} | 1.0000 |
| 4 | (4,16) vs (16,4) | 0.800 | 0.750 | 0.252 / 0.251 | left | {'both_succeed': 14, 'a_only_succeeds': 2, 'b_only_succeeds': 1, 'both_fail': 3} | 1.0000 |
| 5 | (2,8) vs (8,2) | 0.550 | 0.350 | 0.501 / 0.500 | left | {'both_succeed': 6, 'a_only_succeeds': 5, 'b_only_succeeds': 1, 'both_fail': 8} | 0.2188 |
| 5 | (2,16) vs (16,2) | 0.750 | 0.450 | 0.501 / 0.501 | left | {'both_succeed': 7, 'a_only_succeeds': 8, 'b_only_succeeds': 2, 'both_fail': 3} | 0.1094 |
| 5 | (8,16) vs (16,8) | 0.850 | 0.900 | 0.128 / 0.129 | right | {'both_succeed': 15, 'a_only_succeeds': 2, 'b_only_succeeds': 3, 'both_fail': 0} | 1.0000 |
| 5 | (4,16) vs (16,4) | 0.850 | 0.750 | 0.252 / 0.252 | left | {'both_succeed': 14, 'a_only_succeeds': 3, 'b_only_succeeds': 1, 'both_fail': 2} | 0.6250 |
| 6 | (2,8) vs (8,2) | 0.450 | 0.350 | 0.501 / 0.500 | left | {'both_succeed': 7, 'a_only_succeeds': 2, 'b_only_succeeds': 0, 'both_fail': 11} | 0.5000 |
| 6 | (2,16) vs (16,2) | 0.450 | 0.500 | 0.501 / 0.500 | right | {'both_succeed': 8, 'a_only_succeeds': 1, 'b_only_succeeds': 2, 'both_fail': 9} | 1.0000 |
| 6 | (8,16) vs (16,8) | 0.650 | 0.550 | 0.126 / 0.126 | left | {'both_succeed': 10, 'a_only_succeeds': 3, 'b_only_succeeds': 1, 'both_fail': 6} | 0.6250 |
| 6 | (4,16) vs (16,4) | 0.550 | 0.550 | 0.251 / 0.251 | tie | {'both_succeed': 8, 'a_only_succeeds': 3, 'b_only_succeeds': 3, 'both_fail': 6} | 1.0000 |
| 7 | (2,8) vs (8,2) | 0.600 | 0.550 | 0.501 / 0.501 | left | {'both_succeed': 9, 'a_only_succeeds': 3, 'b_only_succeeds': 2, 'both_fail': 6} | 1.0000 |
| 7 | (2,16) vs (16,2) | 0.500 | 0.450 | 0.501 / 0.501 | left | {'both_succeed': 7, 'a_only_succeeds': 3, 'b_only_succeeds': 2, 'both_fail': 8} | 1.0000 |
| 7 | (8,16) vs (16,8) | 0.550 | 0.550 | 0.126 / 0.127 | tie | {'both_succeed': 10, 'a_only_succeeds': 1, 'b_only_succeeds': 1, 'both_fail': 8} | 1.0000 |
| 7 | (4,16) vs (16,4) | 0.650 | 0.450 | 0.251 / 0.251 | left | {'both_succeed': 9, 'a_only_succeeds': 4, 'b_only_succeeds': 0, 'both_fail': 7} | 0.1250 |
| 8 | (2,8) vs (8,2) | 0.400 | 0.400 | 0.501 / 0.501 | tie | {'both_succeed': 7, 'a_only_succeeds': 1, 'b_only_succeeds': 1, 'both_fail': 11} | 1.0000 |
| 8 | (2,16) vs (16,2) | 0.350 | 0.350 | 0.500 / 0.501 | tie | {'both_succeed': 4, 'a_only_succeeds': 3, 'b_only_succeeds': 3, 'both_fail': 10} | 1.0000 |
| 8 | (8,16) vs (16,8) | 0.300 | 0.250 | 0.125 / 0.126 | left | {'both_succeed': 5, 'a_only_succeeds': 1, 'b_only_succeeds': 0, 'both_fail': 14} | 1.0000 |
| 8 | (4,16) vs (16,4) | 0.350 | 0.350 | 0.250 / 0.251 | tie | {'both_succeed': 5, 'a_only_succeeds': 2, 'b_only_succeeds': 2, 'both_fail': 11} | 1.0000 |
| 9 | (2,8) vs (8,2) | 0.950 | 0.850 | 0.502 / 0.502 | left | {'both_succeed': 17, 'a_only_succeeds': 2, 'b_only_succeeds': 0, 'both_fail': 1} | 0.5000 |
| 9 | (2,16) vs (16,2) | 0.900 | 0.950 | 0.502 / 0.501 | right | {'both_succeed': 17, 'a_only_succeeds': 1, 'b_only_succeeds': 2, 'both_fail': 0} | 1.0000 |
| 9 | (8,16) vs (16,8) | 0.900 | 1.000 | 0.128 / 0.128 | right | {'both_succeed': 18, 'a_only_succeeds': 0, 'b_only_succeeds': 2, 'both_fail': 0} | 0.5000 |
| 9 | (4,16) vs (16,4) | 1.000 | 0.950 | 0.254 / 0.252 | left | {'both_succeed': 19, 'a_only_succeeds': 1, 'b_only_succeeds': 0, 'both_fail': 0} | 1.0000 |

## Macro comparison

- Mean per-task best-global success rate: 0.700
- Mean per-task best-group-wise success rate: 0.761
- Mean per-task best-off-diagonal success rate: 0.761
- Best-global vs best-off-diagonal mean difference: 0.061

Per-task JSON artifacts contain success vectors, Wilson intervals, Pareto frontiers, budget-matched paired comparisons, and all configuration summaries.

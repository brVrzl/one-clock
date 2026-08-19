# LIBERO Object dynamic-readiness analysis

This is a post-hoc analysis of completed task-0 50-state and tasks-1..9 20-state static rollout artifacts. No rollouts were run.

- Tasks: 10; each task has equal macro weight.
- Common global horizons: [2, 4, 8, 16].
- Common group-wise pairs: (2,2), (2,8), (2,16), (4,4), (4,16), (8,2), (8,8), (8,16), (16,2), (16,4), (16,8), (16,16).

## Common configuration set

Common global fixed configurations are G2, G4, G8, and G16. Diagonal group-wise aliases are (2,2), (4,4), (8,8), and (16,16). The common off-diagonal set is (2,8), (2,16), (4,16), (8,2), (8,16), (16,2), (16,4), and (16,8). Task 0's h=1/full-grid-only cells are excluded.

## Universal global horizons

| Horizon | Macro success | Worst | Median | Mean query rate | Per-task rates |
|---|---:|---:|---:|---:|---|
| G2 | 0.542 | 0.300 | 0.500 | 0.501 | 0.620, 0.300, 0.700, 0.800, 0.500, 0.300, 0.400, 0.500, 0.400, 0.900 |
| G4 | 0.644 | 0.400 | 0.625 | 0.251 | 0.840, 0.500, 0.750, 0.750, 0.800, 0.500, 0.500, 0.500, 0.400, 0.900 |
| G8 | 0.660 | 0.350 | 0.725 | 0.127 | 0.900, 0.450, 0.700, 0.800, 0.750, 0.750, 0.450, 0.550, 0.350, 0.900 |
| G16 | 0.699 | 0.350 | 0.750 | 0.065 | 0.840, 0.500, 0.700, 0.900, 0.800, 0.850, 0.650, 0.550, 0.350, 0.850 |

Best universal global: **G16**, macro success 0.699.

## Universal group-wise pairs

| Pair | Macro success | Median | Worst | Mean query rate | Best/tied-best tasks | Pareto tasks | Per-task rates |
|---|---:|---:|---:|---:|---:|---:|---|
| (2,2) | 0.542 | 0.500 | 0.300 | 0.501 | 1 | 0 | 0.620, 0.300, 0.700, 0.800, 0.500, 0.300, 0.400, 0.500, 0.400, 0.900 |
| (2,8) | 0.664 | 0.675 | 0.400 | 0.501 | 1 | 0 | 0.840, 0.450, 0.750, 0.850, 0.800, 0.550, 0.450, 0.600, 0.400, 0.950 |
| (2,16) | 0.698 | 0.775 | 0.350 | 0.501 | 3 | 2 | 0.880, 0.700, 0.800, 0.800, 0.850, 0.750, 0.450, 0.500, 0.350, 0.900 |
| (4,4) | 0.644 | 0.625 | 0.400 | 0.251 | 1 | 1 | 0.840, 0.500, 0.750, 0.750, 0.800, 0.500, 0.500, 0.500, 0.400, 0.900 |
| (4,16) | 0.734 | 0.800 | 0.350 | 0.252 | 4 | 3 | 0.940, 0.600, 0.800, 0.800, 0.800, 0.850, 0.550, 0.650, 0.350, 1.000 |
| (8,2) | 0.531 | 0.575 | 0.300 | 0.501 | 1 | 0 | 0.660, 0.300, 0.600, 0.600, 0.650, 0.350, 0.350, 0.550, 0.400, 0.850 |
| (8,8) | 0.660 | 0.725 | 0.350 | 0.127 | 0 | 0 | 0.900, 0.450, 0.700, 0.800, 0.750, 0.750, 0.450, 0.550, 0.350, 0.900 |
| (8,16) | 0.705 | 0.725 | 0.300 | 0.127 | 1 | 3 | 0.900, 0.650, 0.700, 0.800, 0.750, 0.850, 0.650, 0.550, 0.300, 0.900 |
| (16,2) | 0.540 | 0.550 | 0.200 | 0.501 | 0 | 0 | 0.600, 0.200, 0.650, 0.650, 0.600, 0.450, 0.500, 0.450, 0.350, 0.950 |
| (16,4) | 0.624 | 0.695 | 0.300 | 0.251 | 0 | 0 | 0.740, 0.300, 0.650, 0.750, 0.750, 0.750, 0.550, 0.450, 0.350, 0.950 |
| (16,8) | 0.682 | 0.750 | 0.250 | 0.127 | 3 | 2 | 0.820, 0.350, 0.700, 0.900, 0.800, 0.900, 0.550, 0.550, 0.250, 1.000 |
| (16,16) | 0.699 | 0.750 | 0.350 | 0.065 | 2 | 10 | 0.840, 0.500, 0.700, 0.900, 0.800, 0.850, 0.650, 0.550, 0.350, 0.850 |

Best universal group-wise pair: **(4,16)**, macro success 0.734.

### Per-task Wilson 95% intervals

The cells below are success rate followed by its per-task binomial Wilson interval; task 0 uses 50 episodes and tasks 1..9 use 20.

| Config | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|---|---|---|---|---|---|---|---|---|---|---|
| G2 | 0.620 [0.482,0.741] | 0.300 [0.145,0.519] | 0.700 [0.481,0.855] | 0.800 [0.584,0.919] | 0.500 [0.299,0.701] | 0.300 [0.145,0.519] | 0.400 [0.219,0.613] | 0.500 [0.299,0.701] | 0.400 [0.219,0.613] | 0.900 [0.699,0.972] |
| G4 | 0.840 [0.715,0.917] | 0.500 [0.299,0.701] | 0.750 [0.531,0.888] | 0.750 [0.531,0.888] | 0.800 [0.584,0.919] | 0.500 [0.299,0.701] | 0.500 [0.299,0.701] | 0.500 [0.299,0.701] | 0.400 [0.219,0.613] | 0.900 [0.699,0.972] |
| G8 | 0.900 [0.786,0.957] | 0.450 [0.258,0.658] | 0.700 [0.481,0.855] | 0.800 [0.584,0.919] | 0.750 [0.531,0.888] | 0.750 [0.531,0.888] | 0.450 [0.258,0.658] | 0.550 [0.342,0.742] | 0.350 [0.181,0.567] | 0.900 [0.699,0.972] |
| G16 | 0.840 [0.715,0.917] | 0.500 [0.299,0.701] | 0.700 [0.481,0.855] | 0.900 [0.699,0.972] | 0.800 [0.584,0.919] | 0.850 [0.640,0.948] | 0.650 [0.433,0.819] | 0.550 [0.342,0.742] | 0.350 [0.181,0.567] | 0.850 [0.640,0.948] |
| (2,2) | 0.620 [0.482,0.741] | 0.300 [0.145,0.519] | 0.700 [0.481,0.855] | 0.800 [0.584,0.919] | 0.500 [0.299,0.701] | 0.300 [0.145,0.519] | 0.400 [0.219,0.613] | 0.500 [0.299,0.701] | 0.400 [0.219,0.613] | 0.900 [0.699,0.972] |
| (2,8) | 0.840 [0.715,0.917] | 0.450 [0.258,0.658] | 0.750 [0.531,0.888] | 0.850 [0.640,0.948] | 0.800 [0.584,0.919] | 0.550 [0.342,0.742] | 0.450 [0.258,0.658] | 0.600 [0.387,0.781] | 0.400 [0.219,0.613] | 0.950 [0.764,0.991] |
| (2,16) | 0.880 [0.762,0.944] | 0.700 [0.481,0.855] | 0.800 [0.584,0.919] | 0.800 [0.584,0.919] | 0.850 [0.640,0.948] | 0.750 [0.531,0.888] | 0.450 [0.258,0.658] | 0.500 [0.299,0.701] | 0.350 [0.181,0.567] | 0.900 [0.699,0.972] |
| (4,4) | 0.840 [0.715,0.917] | 0.500 [0.299,0.701] | 0.750 [0.531,0.888] | 0.750 [0.531,0.888] | 0.800 [0.584,0.919] | 0.500 [0.299,0.701] | 0.500 [0.299,0.701] | 0.500 [0.299,0.701] | 0.400 [0.219,0.613] | 0.900 [0.699,0.972] |
| (4,16) | 0.940 [0.838,0.979] | 0.600 [0.387,0.781] | 0.800 [0.584,0.919] | 0.800 [0.584,0.919] | 0.800 [0.584,0.919] | 0.850 [0.640,0.948] | 0.550 [0.342,0.742] | 0.650 [0.433,0.819] | 0.350 [0.181,0.567] | 1.000 [0.839,1.000] |
| (8,2) | 0.660 [0.522,0.776] | 0.300 [0.145,0.519] | 0.600 [0.387,0.781] | 0.600 [0.387,0.781] | 0.650 [0.433,0.819] | 0.350 [0.181,0.567] | 0.350 [0.181,0.567] | 0.550 [0.342,0.742] | 0.400 [0.219,0.613] | 0.850 [0.640,0.948] |
| (8,8) | 0.900 [0.786,0.957] | 0.450 [0.258,0.658] | 0.700 [0.481,0.855] | 0.800 [0.584,0.919] | 0.750 [0.531,0.888] | 0.750 [0.531,0.888] | 0.450 [0.258,0.658] | 0.550 [0.342,0.742] | 0.350 [0.181,0.567] | 0.900 [0.699,0.972] |
| (8,16) | 0.900 [0.786,0.957] | 0.650 [0.433,0.819] | 0.700 [0.481,0.855] | 0.800 [0.584,0.919] | 0.750 [0.531,0.888] | 0.850 [0.640,0.948] | 0.650 [0.433,0.819] | 0.550 [0.342,0.742] | 0.300 [0.145,0.519] | 0.900 [0.699,0.972] |
| (16,2) | 0.600 [0.462,0.724] | 0.200 [0.081,0.416] | 0.650 [0.433,0.819] | 0.650 [0.433,0.819] | 0.600 [0.387,0.781] | 0.450 [0.258,0.658] | 0.500 [0.299,0.701] | 0.450 [0.258,0.658] | 0.350 [0.181,0.567] | 0.950 [0.764,0.991] |
| (16,4) | 0.740 [0.604,0.841] | 0.300 [0.145,0.519] | 0.650 [0.433,0.819] | 0.750 [0.531,0.888] | 0.750 [0.531,0.888] | 0.750 [0.531,0.888] | 0.550 [0.342,0.742] | 0.450 [0.258,0.658] | 0.350 [0.181,0.567] | 0.950 [0.764,0.991] |
| (16,8) | 0.820 [0.692,0.902] | 0.350 [0.181,0.567] | 0.700 [0.481,0.855] | 0.900 [0.699,0.972] | 0.800 [0.584,0.919] | 0.900 [0.699,0.972] | 0.550 [0.342,0.742] | 0.550 [0.342,0.742] | 0.250 [0.112,0.469] | 1.000 [0.839,1.000] |
| (16,16) | 0.840 [0.715,0.917] | 0.500 [0.299,0.701] | 0.700 [0.481,0.855] | 0.900 [0.699,0.972] | 0.800 [0.584,0.919] | 0.850 [0.640,0.948] | 0.650 [0.433,0.819] | 0.550 [0.342,0.742] | 0.350 [0.181,0.567] | 0.850 [0.640,0.948] |

## Per-task static oracle

| Task | Name | Best global | Best group-wise | Static oracle | Oracle gap over universal group |
|---:|---|---:|---:|---:|---:|
| 0 | pick_up_the_alphabet_soup_and_place_it_in_the_basket | G8 (0.900) | (4,16) (0.940) | (4,16) (0.940) | 0.000 |
| 1 | pick_up_the_cream_cheese_and_place_it_in_the_basket | G4, G16 (0.500) | (2,16) (0.700) | (2,16) (0.700) | 0.100 |
| 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket | G4 (0.750) | (2,16), (4,16) (0.800) | (2,16), (4,16) (0.800) | 0.000 |
| 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket | G16 (0.900) | (16,8), (16,16) (0.900) | (16,8), (16,16) (0.900) | 0.100 |
| 4 | pick_up_the_ketchup_and_place_it_in_the_basket | G4, G16 (0.800) | (2,16) (0.850) | (2,16) (0.850) | 0.050 |
| 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket | G16 (0.850) | (16,8) (0.900) | (16,8) (0.900) | 0.050 |
| 6 | pick_up_the_butter_and_place_it_in_the_basket | G16 (0.650) | (8,16), (16,16) (0.650) | (8,16), (16,16) (0.650) | 0.100 |
| 7 | pick_up_the_milk_and_place_it_in_the_basket | G8, G16 (0.550) | (4,16) (0.650) | (4,16) (0.650) | 0.000 |
| 8 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | G2, G4 (0.400) | (2,2), (2,8), (4,4), (8,2) (0.400) | (2,2), (2,8), (4,4), (8,2) (0.400) | 0.050 |
| 9 | pick_up_the_orange_juice_and_place_it_in_the_basket | G2, G4, G8 (0.900) | (4,16), (16,8) (1.000) | (4,16), (16,8) (1.000) | 0.000 |

Macro per-task best global: **0.720**. Macro per-task best group-wise: **0.779**. Group-wise minus global: **0.059**; group-wise strictly better on 7, tied on 3, global better on 0 tasks.

## Leave-one-task-out selection

| Held-out task | Selected pair(s) | Held-out selected success | Held-out oracle | Regret | Best global |
|---:|---|---:|---:|---:|---:|
| 0 | (4,16) | 0.940 | 0.940 | 0.000 | 0.900 |
| 1 | (4,16) | 0.600 | 0.700 | 0.100 | 0.500 |
| 2 | (4,16) | 0.800 | 0.800 | 0.000 | 0.750 |
| 3 | (4,16) | 0.800 | 0.900 | 0.100 | 0.900 |
| 4 | (4,16) | 0.800 | 0.850 | 0.050 | 0.800 |
| 5 | (4,16) | 0.850 | 0.900 | 0.050 | 0.850 |
| 6 | (4,16) | 0.550 | 0.650 | 0.100 | 0.650 |
| 7 | (4,16) | 0.650 | 0.650 | 0.000 | 0.550 |
| 8 | (4,16) | 0.350 | 0.400 | 0.050 | 0.400 |
| 9 | (4,16) | 1.000 | 1.000 | 0.000 | 0.900 |

Mean leave-one-task-out regret: **0.045**. Tied selections are retained.

## Group-wise rank stability

| Pair | Mean rank | Rank SD | Best rank | Worst rank |
|---|---:|---:|---:|---:|
| (2,2) | 8.85 | 2.88 | 2.5 | 12.0 |
| (2,8) | 4.85 | 2.27 | 2.0 | 9.0 |
| (2,16) | 5.30 | 3.06 | 1.0 | 9.0 |
| (4,4) | 6.25 | 2.42 | 2.5 | 9.5 |
| (4,16) | 3.25 | 2.08 | 1.0 | 7.5 |
| (8,2) | 9.60 | 3.08 | 2.5 | 12.0 |
| (8,8) | 6.55 | 1.75 | 2.5 | 9.0 |
| (8,16) | 5.40 | 2.98 | 1.5 | 11.0 |
| (16,2) | 9.60 | 2.56 | 4.0 | 12.0 |
| (16,4) | 8.00 | 2.49 | 4.0 | 11.5 |
| (16,8) | 5.20 | 3.36 | 1.0 | 12.0 |
| (16,16) | 5.15 | 2.88 | 1.5 | 11.5 |

## Horizon preferences among task-optimal group-wise pairs

Arm horizon frequencies over tied optima: {'2': 5, '4': 5, '8': 2, '16': 5}.
Gripper horizon frequencies over tied optima: {'2': 2, '4': 1, '8': 4, '16': 10}.
Pair frequencies: {'(4,16)': 4, '(2,16)': 3, '(16,8)': 3, '(16,16)': 2, '(8,16)': 1, '(2,2)': 1, '(2,8)': 1, '(4,4)': 1, '(8,2)': 1}.
Relation counts: gripper > arm 9, equal 4, gripper < arm 4.

## Success/query Pareto analysis

| Configuration | Kind | Macro success | Mean query rate | Dominated |
|---|---|---:|---:|---|
| G2 | global | 0.542 | 0.501 | yes |
| G4 | global | 0.644 | 0.251 | yes |
| G8 | global | 0.660 | 0.127 | yes |
| G16 | global | 0.699 | 0.065 | no |
| (2,2) | diagonal_groupwise | 0.542 | 0.501 | yes |
| (2,8) | off_diagonal_groupwise | 0.664 | 0.501 | yes |
| (2,16) | off_diagonal_groupwise | 0.698 | 0.501 | yes |
| (4,4) | diagonal_groupwise | 0.644 | 0.251 | yes |
| (4,16) | off_diagonal_groupwise | 0.734 | 0.252 | no |
| (8,2) | off_diagonal_groupwise | 0.531 | 0.501 | yes |
| (8,8) | diagonal_groupwise | 0.660 | 0.127 | yes |
| (8,16) | off_diagonal_groupwise | 0.705 | 0.127 | no |
| (16,2) | off_diagonal_groupwise | 0.540 | 0.501 | yes |
| (16,4) | off_diagonal_groupwise | 0.624 | 0.251 | yes |
| (16,8) | off_diagonal_groupwise | 0.682 | 0.127 | yes |
| (16,16) | diagonal_groupwise | 0.699 | 0.065 | no |

Empirical cross-task Pareto frontier: **G16, (4,16), (8,16), (16,16)**.
Universal group-wise versus global: macro success difference 0.035; query-rate difference 0.187 (group-wise minus global).

## Confidence intervals and bootstrap

The JSON artifact contains per-task Wilson 95% intervals for every common configuration. Macro comparisons use a deterministic task-level bootstrap, not pooled episodes.
Bootstrap seed 20260819, draws 20000. Universal group-wise minus universal global: 0.035, CI [-0.020, 0.085].
Per-task static oracle minus universal group-wise: 0.045, CI [0.020, 0.070].

## Dynamic-readiness decision

Classification: **B**.
A single universal off-diagonal pair, (4,16), is selected in every leave-one-task-out split and reaches 0.734 macro success versus a 0.779 per-task group-wise static oracle. Its 0.045 oracle gap is real in the task bootstrap diagnostic, but it is small relative to the one-task-per-task sample and the task-optimal pairs vary. The evidence supports a useful static heterogeneous baseline, while the universal pair captures most observed oracle performance; dynamic scheduling is therefore not yet justified.

## PACE source audit and deferred baseline list

PACE is a scalar/global test-time execution rule: from each full predicted chunk it builds a joint- or Cartesian-space arm speed profile, suppresses short fluctuations, identifies low-speed transition regions, and selects one prefix boundary; in multi-arm settings it uses the earliest accepted arm boundary. It uses fixed selection parameters calibrated from demonstrations, does not use evaluation rollouts or policy internals, and discards the unexecuted suffix after each query. The current LIBERO ACT path supplies a full unnormalized (100,7) chunk with six relative end-effector controls and one gripper control, so the chunk boundary input exists. A source-faithful PACE comparison would still need an explicit, verified mapping from the six relative controls to the paper's motion-speed profile and calibration procedure. PACE is global/scalar and is not a group-wise method; it was not implemented.

Deferred comparator list if a later dynamic-method task is authorized: best universal global G16, best universal static group-wise (4,16), per-task static group-wise oracle, PACE as a separately audited global dynamic baseline if later authorized. No item in this list was implemented here.

No dynamic method was implemented and no rollouts were run for this analysis.

## Figures

- experiments/figures/libero_object_universal_vs_oracle.png
- experiments/figures/libero_object_config_rank_heatmap.png
- experiments/figures/libero_object_universal_success_query.png

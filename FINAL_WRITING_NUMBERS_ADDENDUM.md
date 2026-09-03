# Final writing-numbers addendum

Freeze date: 2026-09-03 (Asia/Shanghai)

Purpose: resolve the remaining exact numerical and comparability questions for writing. This document reports persisted canonical results only. It adds no experiment, rollout, statistical test, or scientific analysis.

Source basis: `FINAL_CANONICAL_NUMBERS.md`, `FINAL_CLAIM_FREEZE_20260903.md`, `FINAL_FIGURE_INFORMATION_HIERARCHY.md`, and `FINAL_EXPERIMENT_HANDOFF.md`. Where those files explicitly route details to canonical records, this addendum also extracts the persisted `experiments/icra27_reviewer_supplement/canonical_report.json`, `experiments/icra27_crosssuite_query_allocation/factorial_interaction_convention.json`, and `experiments/icra27_crosssuite_query_allocation/interaction_robustness/analysis.json` values. No raw outcomes were reanalyzed.

## 1. Exact Object-126 Fresh anchor

The exact Fresh result is **56/126 (44.44%)**.

Its cohort identity is:

- policy and suite: ACT on `libero_object`;
- tasks: `1,2,3,4,5,6,7,8,9`;
- initial states for every task: `20,21,22,23,27,31,34,35,38,39,44,45,47,48`;
- total: 9 tasks x 14 states = 126 paired task-state blocks;
- environment seed: `330000 + 100 * task_id + state_id`;
- policy seed: `424242`;
- checkpoint: `/home/wjq/checkpoints/zeromidnight_act_libero_object` (`zeromidnight/act_libero_object`);
- evaluator: installed LeRobot 0.4.4 ACT at 20 Hz, 280-step cap, 256-pixel observations, and the frozen success criterion;
- action contract: seven dimensions, with translation `0:3`, rotation `3:6`, and gripper `6`;
- execution contract: one whole-policy query per executed environment step, same-target identity `q+k=t`, and an all-Fresh prefix for `t<d`.

Fresh was historically reused from `experiments/group_delay_factorial_act20`; it was not rerun in R1A or R1B. The frozen reuse audit establishes exact matching, not approximate matching.

Fresh is exactly paired on the same block identities with all four requested d=20 conditions:

| Condition | Exactly paired with Fresh? | Basis |
|---|---|---|
| `A20G0` at d=20 | Yes | Audited historical Object-126 anchor |
| `A0G20` at d=20 | Yes | Audited historical Object-126 anchor |
| R1B translation-stale `T20_R0_G0` | Yes | New R1B condition on the same 126 blocks |
| R1B rotation-stale `T0_R20_G0` | Yes | New R1B condition on the same 126 blocks |

## 2. R1B comparability and complete statistics

Under the authoritative 20 Hz ACT evaluator, **d=20 is exactly 1.00 s**.

R1B uses exactly the same 126 task-state blocks, Object checkpoint, initial states, environment seeds and block identities, policy seed, evaluator, seven-dimensional action grouping, Fresh-prefix rule, same-target `q+k=t` contract, dense whole-policy query schedule, 280-step cap, and success criterion as the R1A historical d=20 anchors. R1B is therefore block-paired with Fresh, `A20G0`, and `A0G20`. Only the intended component assignment differs.

### Absolute results

| Condition | Success/N | Success |
|---|---:|---:|
| Translation stale, `T20_R0_G0` | 11/126 | 8.73% |
| Rotation stale, `T0_R20_G0` | 53/126 | 42.06% |
| Fresh, `A0_G0` | 56/126 | 44.44% |
| Whole arm stale, `A20_G0` | 12/126 | 9.52% |
| Gripper stale, `A0_G20` | 81/126 | 64.29% |

### Canonical R1B contrasts

Discordance is `first condition only:second condition only`.

| Contrast and sign | Effect (pp) | Discordance | Exact McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| `T20_R0_G0 - T0_R20_G0` | -33.33 | 3:45 | 1.31259e-10 | [-42.06, -24.60] | [-44.44, -22.22] |
| `T20_R0_G0 - Fresh` | -35.71 | 3:48 | 1.96749e-11 | [-45.24, -26.19] | [-46.03, -25.40] |
| `T0_R20_G0 - Fresh` | -2.38 | 6:9 | 0.607239 | [-8.73, +3.97] | [-6.35, +1.59] |

The primary sign convention is `S(translation-stale) - S(rotation-stale)`. Its frozen leave-one-task-out estimates range from **-36.61 to -30.36 pp**, and all nine are negative. Thus the primary ordering is stable under every frozen leave-one-task-out omission.

The rotation-stale comparison with Fresh is canonically estimated at -2.38 pp, but both canonical intervals include zero. The appropriate wording is that the canonical comparison detects little cost relative to Fresh on this cohort, not that rotation staleness is free.

### Canonical historical-anchor comparisons with Fresh

The authoritative tables use Fresh first, so their original sign convention is retained here.

| Canonical contrast | Successes | Effect (pp) | Discordance | Exact McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| `Fresh - A20_G0` | 56 vs 12 / 126 | +34.92 | 45:1 | 1.33582e-12 | [+26.19, +43.65] | [+25.40, +45.24] |
| `Fresh - A0_G20` | 56 vs 81 / 126 | -19.84 | 1:26 | 4.17233e-07 | [-27.78, -12.70] | [-30.95, -9.52] |

These are existing canonical paired comparisons, not newly created tests. The Object-cohort ordering of `A0G20` above Fresh is strong, but its effect magnitude is not robust across cohorts: the separate frozen 140-block confirmation found only a small and uncertain `A0G20`-versus-Fresh simple effect. It must not be described as a generic benefit of gripper staleness.

## 3. Five-point d=20 comparability gate

**`STAIRCASE_EXACTLY_COMPARABLE`**

All five points share the Object cohort, the same 126 task-state block identities, checkpoint, initial states, seeds, 20 Hz evaluator, action contract, dense one-query-per-step schedule, Fresh-prefix semantics, current-target alignment, episode cap, and success definition. Fresh is the paired zero-age reference; the four stale-component interventions use d=20, or 1.00 s. The intended component assignment is the only intervention difference.

| Reader-facing point | Canonical condition | Success/N | Success |
|---|---|---:|---:|
| Translation stale | `T20_R0_G0` | 11/126 | 8.73% |
| Full arm stale | `A20_G0` | 12/126 | 9.52% |
| Rotation stale | `T0_R20_G0` | 53/126 | 42.06% |
| Fresh | `A0_G0` | 56/126 | 44.44% |
| Gripper stale | `A0_G20` | 81/126 | 64.29% |

Permitted interpretation: **staling translation alone nearly reproduces the degradation observed when the entire arm is stale**. This does not establish that all whole-arm degradation is caused by translation. The rotation result has little detectable cost relative to Fresh under the existing paired comparison, but it is not “free.” The gripper-above-Fresh ordering is exact on this Object cohort, whereas the corresponding simple-effect magnitude was small and uncertain on the primary 140-block cohort. Ordering robustness and effect-magnitude robustness must remain distinct.

## 4. R1C complete frozen statistics

R1C is a frozen reviewer-directed query-matched extension. Its design, including the risk-difference interaction formula, was frozen before R1C outcomes were inspected. It is not a retrospective arithmetic interaction.

All four cells query the whole policy once per executed environment step:

| Condition | Meaning | Success/N | Success |
|---|---|---:|---:|
| `C00` | dense-query Fresh arm + Fresh gripper | 77/140 | 55.00% |
| `C10` | dense-query scheduled-H16 arm + Fresh gripper | 76/140 | 54.29% |
| `C01` | dense-query Fresh arm + scheduled-H16 gripper | 81/140 | 57.86% |
| `C11` | dense-query scheduled-H16 arm + scheduled-H16 gripper | 93/140 | 66.43% |

For every row below, the sign is `first condition - second condition`, and discordance is `first only:second only`.

| Frozen contrast | Scientific meaning | Effect (pp) | Discordance | Exact McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) |
|---|---|---:|---:|---:|---:|---:|
| `C10-C00` | Scheduled-arm effect with gripper Fresh | -0.71 | 18:19 | 1 | [-9.29, +7.86] | [-12.14, +10.71] |
| `C11-C01` | Scheduled-arm effect with gripper scheduled | +8.57 | 26:14 | 0.0806905 | [+0.00, +17.14] | [-2.14, +21.43] |
| `C01-C00` | Scheduled-gripper effect with arm Fresh | +2.86 | 10:6 | 0.454498 | [-2.86, +8.57] | [-4.29, +9.29] |
| `C11-C10` | Scheduled-gripper effect with arm scheduled | +12.14 | 20:3 | 0.000488281 | [+5.71, +18.57] | [+3.57, +21.43] |
| `C11-C00` | Both scheduled versus both Fresh diagonal | +11.43 | 28:12 | 0.016589 | [+2.86, +20.00] | [-0.71, +23.57] |
| `C10-C01` | Opposite diagonal | -3.57 | 12:17 | 0.458258 | [-11.43, +4.29] | [-14.29, +6.43] |

### Frozen sensitivity summaries

The persisted canonical report contains the following leave-one-task-out and leave-one-suite-out descriptions:

| Contrast | Frozen LOTO range/status (pp) | Frozen LOSO values/status (pp) |
|---|---|---|
| `C10-C00` | -4.76 to +3.17; mixed | -4.29, +2.86; mixed |
| `C11-C01` | +3.17 to +11.11; all positive | -1.43, +18.57; mixed |
| `C01-C00` | +0.79 to +4.76; all positive | +1.43, +4.29; both positive |
| `C11-C10` | +9.52 to +14.29; all positive | +4.29, +20.00; both positive |
| `C11-C00` | +7.94 to +14.29; all positive | +0.00, +22.86; nonnegative, including zero |
| `C10-C01` | -6.35 to +0.00; nonpositive, including zero | -5.71, -1.43; both negative |

The frozen interaction is

`C11 - C10 - C01 + C00 = +9.29 pp`.

Its sign convention is positive when the joint scheduled-arm/scheduled-gripper result exceeds the additive risk-difference prediction from the two simple effects at `C00`. The canonical R1C artifacts freeze only the point estimate. They provide **no interaction-specific p-value, paired interval, task-cluster interval, LOTO vector, or LOSO vector**, so none is added here.

## 5. Existing interaction evidence inventory

No estimates are pooled.

| Evidence | Canonical signed point estimate | Governance | Existing canonical uncertainty | Query schedule matched? |
|---|---:|---|---|---|
| Original frozen 140-block same-target factorial | `+15.71 pp`, using `p(A20G20)-p(A20G0)-p(A0G20)+p(A0G0)` | Post-hoc supporting interaction on frozen outcomes; the earlier Object preregistration does not transfer | Paired-block bootstrap 95% CI `[+6.43,+25.00]`; task-cluster bootstrap 95% CI `[+0.71,+30.71]`; task-t CI `[-2.76,+34.18]`; exact task sign-flip `p=0.101562` | Yes, within the dense same-target factorial |
| R1C dense-query factorial | `+9.29 pp`, using `C11-C10-C01+C00` | Formula frozen before R1C outcomes in the reviewer-directed query-matched extension | No interaction-specific interval or p-value is canonical | Yes; all four cells query every step |
| R1D Spatial completion | No named interaction estimate exists in the canonical artifacts | Later completion of the missing Spatial full-arm-stale cell | No interaction-specific uncertainty is canonical | Yes; the reconstructed same-target factorial uses dense queries |

The first two canonical point estimates share a positive sign, but the R1D interaction was not canonically reported and the cohorts and interventions differ. This inventory therefore does not warrant a replication claim or a three-estimate directional-consistency claim.

## 6. R1A single-branch writing summary

For stale arm with fresh gripper, success starts at 38.89% at both 0.10 and 0.20 s, falls to 28.57% at 0.40 s and 16.67% at 0.60 s, rises slightly to 18.25% at 0.80 s, reaches 9.52% at 1.00 s, and then recovers to 25.40% at 1.60 s. The observed curve is non-monotone.

For fresh arm with stale gripper, success rises from 60.32% at 0.10 s to 65.08% at 0.20 s, 69.84% at 0.40 and 0.60 s, and 71.43% at 0.80 s, then declines to 64.29% at 1.00 s and 62.70% at 1.60 s. It remains above the 44.44% Fresh anchor at every frozen lag and is also non-monotone.

The largest observed between-branch separations occur from 0.60 to 1.00 s. At d=16, d=20, and d=32, stale-arm success is 18.25%, 9.52%, and 25.40%; stale-gripper success is 71.43%, 64.29%, and 62.70%; and separation is +53.17, +54.76, and +37.30 pp. From d=20 to d=32, the separation narrows by 17.46 pp because stale-arm success improves by 15.87 pp while stale-gripper success declines by 1.59 pp. This is a descriptive decomposition, not a causal attribution. d=20 was frozen before R1A and was not selected as an optimum.

## 7. R1A per-lag inferential status

At every frozen lag d=`2,4,8,12,16,20,32`, canonical exact McNemar p-values, paired 95% CIs, and task-cluster 95% CIs exist for all three paired contrasts:

- `Fresh - A_dG0`;
- `Fresh - A0G_d`;
- `A0G_d - A_dG0`.

The absolute curve shapes, non-monotonicity, locations of observed minima or maxima, changes between lags, the statement that the largest observed separations occur from 0.60 to 1.00 s, and the d=20-to-d=32 decomposition are descriptive only. No canonical lag-versus-lag test establishes a plateau, optimum, tolerance threshold, change point, or statistically distinct peak.

## 8. Same-target query-rate scope

Fresh, `A0G20`, and `A20G0` same-target probe conditions make one whole-policy query per executed environment step, so their **policy-query rate is 1.0**. The R1A and R1B component probes use the same dense diagnostic contract. These conditions isolate source-age and component effects while holding the executed physical target step fixed; they are not deployment executors and do not establish that a qrate=1 executor is practically preferable.

Coherent H16 queries only once every 16 executed steps, which is descriptively much less frequent. Query rate is not converted into FLOPs, latency, or a claim of equal compute.

Because same-target alignment uses `k=d`, an older source observation is coupled to a longer prediction horizon. The manuscript must state early: **“We therefore measure the joint effect of using an older observation and a longer-lookahead prediction; this experiment does not separate those two factors.”**

## 9. Placement decisions carried into writing

- Five-point Object comparison: exactly comparable, but development-cohort characterization. If shown, place it in Fig. 2 with explicit historical-reuse versus new-probe labeling, not in the confirmatory Fig. 1 and not as five levels of one preregistered factorial.
- Interaction: **`INTERACTION_SUPPLEMENT_ONLY`**. The R1C formula was frozen, but its canonical record has no interaction-specific uncertainty; the original 140-block interaction is post-hoc supporting and small-cluster sensitivity crosses zero; R1D has no named canonical interaction estimate. Interaction is not needed to establish R1C's query-schedule result, which rests on the query-matched factorial, conditional contrasts, and identity canary.

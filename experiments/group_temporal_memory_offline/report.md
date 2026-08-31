# Group-specific temporal memory: offline audit

Decision: `GROUP_TEMPORAL_HETEROGENEITY_PARTIAL`

The outcome-blind profiles differ materially across the cohort and persist descriptively on held-out tasks, but the existing FO-versus-Reverse outcome association is not directionally consistent enough for a strong label. The heterogeneity signal remains identifiable without outcomes.

This is an offline/development analysis. It does not establish causal control benefit. Same-target SmolVLA prediction differences combine observation-delay effects with stochastic flow sampling variation that was not keyed by physical step.

## 1. Data/protocol

The primary source is the Fresh dense-query SmolVLA cache at `../component_temporal_reuse/query_cache`. Each episode has shape `[T, 50, 7]`, and the same-target alignment is `a_{t|t-d} = cache[t-d, d, :]`, with source query `q=t-d`. The separate ACT dense cache was inspected but not pooled because it has a different checkpoint/cohort and 100-step chunks. The exact frozen protocol is in [protocol.json](protocol.json).

The cohort contains 8 tasks, 80 episodes, and 15586 current target steps. The requested delay set is `d=0, 4, 8, 16, 20, 32` steps, or approximately `0.000s / 0.133s / 0.267s / 0.533s / 0.667s / 1.067s` at 30 Hz. Candidate counts are support counts, not inferential N. The development split is `object-3, spatial-0, goal-2, 10-3`; held-out descriptive tasks are `object-5, spatial-4, goal-5, 10-5`.

The arm is dimensions 0–5, with translation and rotation treated separately and equally combined using the validated PPPR-style IQR normalization fit from development Fresh actions only. The gripper is dimension 6; its continuous difference uses the fixed postprocessed range 2.0 and its sign disagreement is reported separately. B2 demonstration agreement was skipped because no correctly aligned expert-action table exists in the current SmolVLA artifacts, and labels were not reconstructed.

Total materialized feature rows, including masked cells, are 93516; valid rows are 87116. Per-delay support by task is in `delay_profiles.csv` and the cached feature table.

## 2. Arm delay profiles

`U_arm(d)=1-R_arm(d)`, where higher means more same-target agreement with Fresh, not better control. The columns below list values in the frozen delay order. The revision component columns are exported in `delay_profiles.csv`; the table gives the combined revision explicitly and keeps translation/rotation diagnostics available in the cached table.

| task | split | valid target counts by d=0,4,8,16,20,32 | U_arm by d | translation revision by d | rotation revision by d | best positive d (U) | slope | AUC | threshold crossing |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| object-3 | development | 1501/1461/1421/1341/1301/1181 | 1.000 / 0.700 / 0.704 / 0.695 / 0.688 / 0.671 | 0.000 / 0.321 / 0.323 / 0.351 / 0.363 / 0.418 | 0.000 / 0.602 / 0.586 / 0.600 / 0.619 / 0.658 | 8 (0.704) | -0.0067 | 0.710 | NA |
| object-5 | held_out | 1882/1842/1802/1722/1682/1562 | 1.000 / 0.673 / 0.669 / 0.648 / 0.640 / 0.616 | 0.000 / 0.387 / 0.411 / 0.485 / 0.523 / 0.679 | 0.000 / 0.669 / 0.669 / 0.717 / 0.723 / 0.731 | 4 (0.673) | -0.0082 | 0.669 | 32 |
| spatial-0 | development | 1568/1528/1488/1408/1368/1248 | 1.000 / 0.562 / 0.550 / 0.525 / 0.521 / 0.517 | 0.000 / 0.604 / 0.648 / 0.728 / 0.745 / 0.728 | 0.000 / 1.175 / 1.225 / 1.327 / 1.367 / 1.387 | 4 (0.562) | -0.0101 | 0.561 | 4 |
| spatial-4 | held_out | 1625/1585/1545/1465/1425/1305 | 1.000 / 0.633 / 0.634 / 0.629 / 0.621 / 0.611 | 0.000 / 0.406 / 0.412 / 0.428 / 0.440 / 0.457 | 0.000 / 0.854 / 0.842 / 0.856 / 0.888 / 0.932 | 8 (0.634) | -0.0078 | 0.648 | 32 |
| goal-2 | development | 1212/1172/1132/1052/1012/892 | 1.000 / 0.635 / 0.624 / 0.614 / 0.615 / 0.583 | 0.000 / 0.382 / 0.406 / 0.431 / 0.455 / 0.532 | 0.000 / 0.871 / 0.918 / 0.959 / 0.939 / 1.077 | 4 (0.635) | -0.0086 | 0.637 | 16 |
| goal-5 | held_out | 1592/1552/1512/1432/1392/1272 | 1.000 / 0.662 / 0.653 / 0.638 / 0.631 / 0.614 | 0.000 / 0.484 / 0.524 / 0.587 / 0.605 / 0.677 | 0.000 / 0.674 / 0.683 / 0.712 / 0.738 / 0.789 | 4 (0.662) | -0.0081 | 0.660 | 32 |
| 10-3 | development | 2852/2812/2772/2692/2652/2532 | 1.000 / 0.641 / 0.639 / 0.621 / 0.606 / 0.584 | 0.000 / 0.361 / 0.383 / 0.442 / 0.479 / 0.548 | 0.000 / 0.866 / 0.865 / 0.943 / 1.020 / 1.164 | 4 (0.641) | -0.0088 | 0.640 | 20 |
| 10-5 | held_out | 3354/3314/3274/3194/3154/3034 | 1.000 / 0.668 / 0.663 / 0.651 / 0.646 / 0.633 | 0.000 / 0.411 / 0.430 / 0.459 / 0.478 / 0.534 | 0.000 / 0.675 / 0.681 / 0.720 / 0.728 / 0.750 | 4 (0.668) | -0.0076 | 0.673 | NA |

## 3. Gripper delay profiles

`U_grip(d)=1-clip(mean(abs(g_old-g_fresh))/2, 0, 1)`. Sign disagreement is the fraction of valid target positions with different `np.sign` commands. Delay maxima, slopes, AUCs, and the development-only threshold crossing are descriptive summaries, not exact optimal horizons.

| task | split | valid target counts by d=0,4,8,16,20,32 | U_grip by d | abs difference by d | sign disagreement by d | best positive d (U) | slope | AUC | threshold crossing |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| object-3 | development | 1501/1461/1421/1341/1301/1181 | 1.000 / 0.907 / 0.900 / 0.886 / 0.888 / 0.857 | 0.000 / 0.187 / 0.200 / 0.228 / 0.225 / 0.287 | 0.000 / 0.089 / 0.095 / 0.109 / 0.108 / 0.138 | 4 (0.907) | -0.0033 | 0.893 | 16 |
| object-5 | held_out | 1882/1842/1802/1722/1682/1562 | 1.000 / 0.831 / 0.799 / 0.759 / 0.732 / 0.698 | 0.000 / 0.338 / 0.403 / 0.482 / 0.536 / 0.605 | 0.000 / 0.165 / 0.198 / 0.237 / 0.265 / 0.299 | 4 (0.831) | -0.0078 | 0.773 | 4 |
| spatial-0 | development | 1568/1528/1488/1408/1368/1248 | 1.000 / 0.838 / 0.843 / 0.792 / 0.776 / 0.714 | 0.000 / 0.325 / 0.314 / 0.418 / 0.449 / 0.574 | 0.000 / 0.157 / 0.150 / 0.202 / 0.219 / 0.280 | 8 (0.843) | -0.0073 | 0.802 | 4 |
| spatial-4 | held_out | 1625/1585/1545/1465/1425/1305 | 1.000 / 0.939 / 0.932 / 0.920 / 0.921 / 0.901 | 0.000 / 0.123 / 0.136 / 0.160 / 0.158 / 0.199 | 0.000 / 0.056 / 0.062 / 0.075 / 0.074 / 0.096 | 4 (0.939) | -0.0024 | 0.926 | NA |
| goal-2 | development | 1212/1172/1132/1052/1012/892 | 1.000 / 0.944 / 0.945 / 0.921 / 0.897 / 0.842 | 0.000 / 0.112 / 0.111 / 0.159 / 0.207 / 0.318 | 0.000 / 0.049 / 0.048 / 0.072 / 0.096 / 0.153 | 8 (0.945) | -0.0044 | 0.913 | 32 |
| goal-5 | held_out | 1592/1552/1512/1432/1392/1272 | 1.000 / 0.994 / 0.993 / 0.992 / 0.993 / 0.992 | 0.000 / 0.013 / 0.014 / 0.016 / 0.014 / 0.016 | 0.000 / 0.000 / 0.000 / 0.001 / 0.000 / 0.000 | 4 (0.994) | -0.0002 | 0.993 | NA |
| 10-3 | development | 2852/2812/2772/2692/2652/2532 | 1.000 / 0.924 / 0.905 / 0.891 / 0.888 / 0.881 | 0.000 / 0.153 / 0.191 / 0.218 / 0.225 / 0.239 | 0.000 / 0.071 / 0.090 / 0.102 / 0.106 / 0.112 | 4 (0.924) | -0.0029 | 0.902 | 20 |
| 10-5 | held_out | 3354/3314/3274/3194/3154/3034 | 1.000 / 0.985 / 0.983 / 0.981 / 0.982 / 0.979 | 0.000 / 0.030 / 0.034 / 0.037 / 0.035 / 0.042 | 0.000 / 0.010 / 0.011 / 0.013 / 0.011 / 0.015 | 4 (0.985) | -0.0005 | 0.983 | NA |

The exact episode-level profiles and per-delay valid support are in [episode_delay_profiles.csv](episode_delay_profiles.csv), and all row-level aligned features are in [cached_same_target_features.npz](cached_same_target_features.npz). Because d=0 is the Fresh identity, the report shows the best positive-delay summary in addition to the full sampled profile; the d=0 maximum is a reference identity, not evidence of an optimal executor horizon.

## 4. Non-Markovian evidence

Under the legitimate offline revision metric, historical sources do not strictly outperform Fresh: Fresh at d=0 is the identical target prediction, so its revision distance is exactly zero and its bounded utility is exactly one. Exact historical matches can occur and are reported below. This does not show that historical observations are useless; it shows that this particular same-target consistency metric is anchored to Fresh and cannot demonstrate historical superiority. B2 expert agreement was unavailable, and no closed-loop outcomes are merged into this section.

### arm

| d | task utility better than Fresh | candidate rows better than Fresh | exact candidate matches |
|---:|---:|---:|---:|
| 4 | 0/8 | 0/15266 (0.000) | 0/15266 (0.000) |
| 8 | 0/8 | 0/14946 (0.000) | 0/14946 (0.000) |
| 16 | 0/8 | 0/14306 (0.000) | 0/14306 (0.000) |
| 20 | 0/8 | 0/13986 (0.000) | 0/13986 (0.000) |
| 32 | 0/8 | 0/13026 (0.000) | 0/13026 (0.000) |

The revision metric is a distance to the identical Fresh prediction at d=0, so strict improvement is impossible by construction; exact matches are repeat/persistence cases.

### gripper

| d | task utility better than Fresh | candidate rows better than Fresh | exact candidate matches |
|---:|---:|---:|---:|
| 4 | 0/8 | 0/15266 (0.000) | 0/15266 (0.000) |
| 8 | 0/8 | 0/14946 (0.000) | 1/14946 (0.000) |
| 16 | 0/8 | 0/14306 (0.000) | 0/14306 (0.000) |
| 20 | 0/8 | 0/13986 (0.000) | 0/13986 (0.000) |
| 32 | 0/8 | 0/13026 (0.000) | 0/13026 (0.000) |

The revision metric is a distance to the identical Fresh prediction at d=0, so strict improvement is impossible by construction; exact matches are repeat/persistence cases.


## 5. Temporal heterogeneity

The primary task descriptor was frozen before reading the intervention outcome file:

`H_temp(task) = mean_d |U_arm(d) - U_grip(d)|` over the six requested delays, with equal delay weight and equal task weight. It uses the normalized same-target revision utility only. This definition, normalization, split, and freeze status are recorded in [protocol.json](protocol.json) and [h_temp_frozen.json](h_temp_frozen.json). The outcome-blind builder explicitly loaded no success/intervention artifact.

| rank | task | split | H_temp | episode bootstrap 95% CI |
|---:|---|---|---:|---:|
| 1 | goal-5 | held_out | 0.294 | [0.275, 0.322] |
| 2 | 10-5 | held_out | 0.275 | [0.257, 0.293] |
| 3 | spatial-4 | held_out | 0.247 | [0.220, 0.269] |
| 4 | goal-2 | development | 0.246 | [0.231, 0.263] |
| 5 | 10-3 | development | 0.233 | [0.196, 0.257] |
| 6 | spatial-0 | development | 0.215 | [0.159, 0.269] |
| 7 | object-3 | development | 0.163 | [0.145, 0.184] |
| 8 | object-5 | held_out | 0.095 | [0.052, 0.134] |

Development macro H_temp is 0.214 (SD 0.036); held-out macro H_temp is 0.228 (SD 0.090). The score is not dominated by a single task if its rank and bootstrap interval are read together, but the eight-task sample remains descriptive. Figure A shows the profiles and Figure B shows the frozen task score.

## 6. Existing closed-loop relation

Only after [h_temp_frozen.json](h_temp_frozen.json) was written, the existing source-intervention results were opened. For each task and d in 4, 8, and 16, `A_task(d)=success(FO_d)-success(Reverse_d)`, with FullOld reported separately.

| d | mean FO−Reverse | mean abs FO−Reverse | positive / negative / zero tasks | Spearman H vs signed A | Spearman H vs |A| |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.038 | 0.113 | 4 / 2 / 2 | -0.566 | -0.085 |
| 8 | 0.012 | 0.063 | 3 / 1 / 4 | 0.140 | -0.370 |
| 16 | 0.138 | 0.312 | 4 / 4 / 0 | -0.970 | -0.699 |

| task | split | H_temp | mean FO−Reverse | mean abs FO−Reverse | mean FullOld−Fresh |
|---|---|---:|---:|---:|---:|
| object-3 | development | 0.163 | 0.100 | 0.233 | 0.000 |
| object-5 | held_out | 0.095 | 0.300 | 0.300 | -0.067 |
| spatial-0 | development | 0.215 | 0.100 | 0.100 | 0.133 |
| spatial-4 | held_out | 0.247 | 0.033 | 0.100 | -0.067 |
| goal-2 | development | 0.246 | -0.067 | 0.067 | 0.067 |
| goal-5 | held_out | 0.294 | -0.067 | 0.133 | 0.033 |
| 10-3 | development | 0.233 | 0.233 | 0.233 | -0.433 |
| 10-5 | held_out | 0.275 | -0.133 | 0.133 | 0.200 |

Across delay-specific rows, the descriptive Spearman correlations of H_temp with signed FO−Reverse are `d=4: -0.566`, `d=8: 0.140`, and `d=16: -0.970`. The across-delay task-mean correlation is `-0.826` for signed A and `-0.539` for |A|. With eight tasks these are descriptive, not significance tests. Obvious counterexamples under the fixed high-H/negative-mean-A rule are: goal-5 (H=0.294, mean A=-0.067), 10-5 (H=0.275, mean A=-0.133), goal-2 (H=0.246, mean A=-0.067). FullOld is not used to define H_temp and is only reported as a separate whole-action comparison. Figure C is the task scatter.

## 7. Reliability reinterpretation

The current branch does not contain a directly comparable SmolVLA source-context reliability table. The validated historical ACT artifact at `historical git artifact at 6ed5d06: experiments/dynamic_reliability_horizon/oracle_group_horizon_metrics.json and source_context_ablation/*` defines reliability as prefix survival of an old cached action against the corresponding Fresh action at the future target: arm is reliable when both normalized translation and rotation discrepancies remain at most 1.0; gripper is reliable when normalized absolute error is at most 1.0 and signs agree. It is semantically compatible with source-age persistence but comes from a different cohort and is not pooled into H_temp.

The historical oracle curves show group-specific decay:

| offset | arm pointwise | arm prefix survival | gripper pointwise | gripper prefix survival |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 0.931 | 0.931 |
| 4 | 0.996 | 0.995 | 0.893 | 0.818 |
| 8 | 0.980 | 0.971 | 0.871 | 0.722 |
| 16 | 0.945 | 0.877 | 0.788 | 0.581 |
| 32 | 0.933 | 0.715 | 0.609 | 0.291 |
| 64 | 0.910 | 0.395 | 0.687 | 0.111 |

The same artifact reports that the group horizons differ in 95.9% of rows where both are uncensored, with gripper expiring first in 64.9% and arm first in 31.0%. In the frozen chunk-only source-context ablation, fixed-cohort AUROC was 0.838 for arm versus 0.964 for gripper, while horizon MAE was 24.06 versus 7.51 actions, respectively. Thus reliability classification was easier for gripper, whereas converting reliability into a hard predicted horizon was especially poor for arm. Together, these results support treating reliability as a continuous/soft temporal signal candidate, not as evidence for a hard per-group horizon. They do not establish that soft weighting improves control.

## 8. Decision

`GROUP_TEMPORAL_HETEROGENEITY_PARTIAL`

The outcome-blind profiles differ materially across the cohort and persist descriptively on held-out tasks, but the existing FO-versus-Reverse outcome association is not directionally consistent enough for a strong label. The heterogeneity signal remains identifiable without outcomes.

## 9. Recommended next experiment

The result supports preparation of the following later, rollout-based ladder, without executing it here: M0 hard sparse h16; M1 shared sparse temporal ensemble; M2 whole-action sparse CogACT-style similarity weighting; M3 group-wise CogACT similarity weighting; M4 group-specific delay prior plus group-wise CogACT; M5 anchored group-wise temporal memory (newest joint action as anchor plus group-specific historical residual/fusion); and M6 optional reliability-weighted anchored group memory. For SmolVLA, an AutoHorizon eligibility mask can be noted as an implementation option. The intended distinction is soft group-specific historical correction around a common newest anchor, not independent hard group replacement.

## Files and reproducibility

- [outcome_blind.json](outcome_blind.json) is the frozen primary result and records `closed_loop_files_loaded: false`.
- [h_temp_frozen.json](h_temp_frozen.json) is the frozen task score artifact.
- [closed_loop_relation.json](closed_loop_relation.json) contains the secondary association only.
- [tests/test_offline_semantics.py](tests/test_offline_semantics.py) covers alignment, slicing, availability, normalization, d=0 identity, and outcome-blind independence.
- Figures are [Figure A](figures/figure_A_delay_profiles.png), [Figure B](figures/figure_B_task_heterogeneity.png), and [Figure C](figures/figure_C_h_temp_vs_fo_reverse.png).

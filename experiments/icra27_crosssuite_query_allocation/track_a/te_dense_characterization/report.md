# TE_DENSE effective-age and gripper characterization

Label: `POST_HOC_TE_EFFECTIVE_AGE_CHARACTERIZATION`

Status: **COMPLETE**

This is explanatory post-hoc characterization after observing the frozen TE_DENSE result. It uses completed Track-A artifacts only and does not tune the coefficient.

## Verified implementation

The frozen implementation is **not** equivalent to `exp(-0.01*age)`. LeRobot assigns rank `i=0` to the oldest candidate and weights rank as `exp(-0.01*i)`. In source-age coordinates this is `exp(+0.01*age)`, normalized over the candidates available at that step. Aggregation occurs in checkpoint-normalized action space before inverse normalization.

## Candidate availability

| Segment | Steps | Mean count | p50 | p95 | Maximum |
|---|---:|---:|---:|---:|---:|
| All executed TE steps | 105947 | 79.127 | 100 | 100 | 100 |
| Startup (`count<100`) | 42835 | 48.374 | 48 | 93 | 99 |
| Steady state (`count=100`) | 63112 | 100.000 | 100 | 100 | 100 |

## Effective age

| Distribution | Mean | p50 | p95 | Maximum support |
|---|---:|---:|---:|---:|
| Theoretical steady-state unweighted candidates | 49.500 steps / 2.475 s | 49 / 2.45 s | 94 / 4.70 s | 99 / 4.95 s |
| Empirical pooled unweighted candidates | 45.110 steps / 2.255 s | 43 / 2.15 s | 93 / 4.65 s | 99 / 4.95 s |
| Theoretical steady state | 57.697 steps / 2.885 s | 62 / 3.10 s | 96 / 4.80 s | 99 / 4.95 s |
| Empirical realized | 44.987 steps / 2.249 s | 43 / 2.15 s | 94 / 4.70 s | 99 / 4.95 s |

The pooled unweighted row counts every candidate occurrence once. The normalized-weight row gives every executed step total mass one before pooling. Maximum support is not the weighted mean effective age.

## Normalized weight assigned to old predictions

Thresholds are strict: older than 0.50 s means age >10 control steps.

| Age threshold | Theoretical steady state | Empirical realized |
|---|---:|---:|
| >0.50 s | 0.932329 | 0.827438 |
| >1.00 s | 0.864005 | 0.715163 |
| >2.00 s | 0.705044 | 0.522690 |

## Executed gripper diagnostics

The native gripper command is continuous. These are pooled executed-step summaries without new inferential tests.

| Condition | Steps | mean(g) | mean(abs(g)) | p50(abs(g)) | abs(g)<0.25 | abs(g)<0.50 | sign/state-switch rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| H16 | 92656 | -0.230380 | 0.999286 | 1.035536 | 0.017991 | 0.039253 | 0.024445 |
| H4 | 100794 | -0.228869 | 1.004481 | 1.037593 | 0.015318 | 0.034149 | 0.027316 |
| ARM4_GRIP32 | 96201 | -0.086092 | 0.996363 | 1.034032 | 0.018940 | 0.040457 | 0.020052 |
| H2 | 106321 | -0.200106 | 1.001660 | 1.035869 | 0.016018 | 0.034631 | 0.020043 |
| ARM2_GRIP16 | 99268 | -0.141349 | 0.996343 | 1.034801 | 0.017770 | 0.040325 | 0.023983 |
| TE_DENSE | 105947 | -0.153195 | 0.797359 | 1.018770 | 0.142703 | 0.244094 | 0.010247 |

## Candidate-level gripper disagreement

`NOT_IDENTIFIABLE_FROM_EXISTING_TRACK_A_ARTIFACTS`: candidate chunks and pre-aggregation gripper values were not persisted. Computing sign disagreement or minority-sign mass would require a rerollout, which is not authorized.

## Scope

Any interpretation is limited to the frozen upstream coefficient and chunk length in this ACT/LIBERO evaluation. These results do not show that canonical temporal ensembling is intrinsically harmful.

Canonical values are in `analysis.json` and the accompanying CSV files.

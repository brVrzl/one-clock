# Frozen 140-block factorial interaction robustness

Status: **POST_HOC_SUPPORTING_INTERACTION**. The earlier 126-block Object preregistration does not transfer to this analysis.

Canonical signed formula:

`I_RD = p(A20G20) - p(A20G0) - p(A0G20) + p(A0G0)`

The log-odds sensitivity uses the identical orientation. An unsigned value must be called the interaction magnitude.

| Scale | Estimate | Paired-block bootstrap 95% CI | Task-cluster bootstrap 95% CI | Small-cluster sensitivity |
|---|---:|---:|---:|---:|
| Risk difference | 15.71 pp | [6.43, 25.00] pp | [0.71, 30.71] pp | task-t CI [-2.76, 34.18] pp; exact task sign-flip p=0.101562 |
| Log odds | 0.6979 (interaction OR 2.009) | [0.3034, 1.1142] | [0.0007, 1.5980] | delete-one-task jackknife-t CI [-0.2089, 1.6046] |

The exact sign-flip calculation enumerates all 1,024 sign assignments of the ten task-level risk-difference interactions and assumes task effects are sign-exchangeable under the null. The bootstrap intervals are descriptive sensitivity analyses; inference is not based solely on percentile cluster-bootstrap exclusion of zero.

These interactions quantify departure from additivity on a chosen scale. They do not identify a unique arm or gripper contribution, and they do not justify path-dependent contribution percentages.

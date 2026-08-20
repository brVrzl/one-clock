# Oracle group-horizon analysis

## Decision-relevant result

The cached `Y_refresh` targets provide an offline, right-censored oracle for group-specific temporal persistence. The result is descriptive only: it is not a learned horizon label, rollout-success supervision, or a closed-loop execution measurement.

The action-count convention is `h* = max { h >= 1 : Y_refresh(h-1) remains true }`. Offset `k=0` supplies only the minimum one-action convention; positive evidence begins at `k=1`. Rows valid through the last observed action are right-censored.

Total windows: 3740; episodes: 454; both-uncensored windows: 1918 (0.513).

## Group distributions

| group | mean | median | q10 | q25 | q75 | q90 | censoring |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm | 45.05 | 42.0 | 10.0 | 21.0 | 61.0 | 100.0 | 0.484 |
| gripper | 22.36 | 18.0 | 2.0 | 6.0 | 32.0 | 52.0 | 0.059 |

These are lower-bound distributions because censored rows are displayed at their last observed action count. The report and JSON also include uncensored-only summaries.

## Heterogeneity and global-clock waste

| comparison population | arm expires first | gripper expires first | equal | different |
|---|---:|---:|---:|---:|
| all rows (lower-bound) | 0.162 | 0.761 | 0.077 | 0.923 |
| both uncensored | 0.310 | 0.649 | 0.041 | 0.959 |

On the both-uncensored subset, forcing both groups to the shorter oracle clock discards a mean of 23.06 valid action positions per window (median 17.0); the discarded fraction is 0.409 of the two groups' observed oracle commitment. This is commitment discarded by the global clock, not a success gain.

### Episode-bootstrap uncertainty

The following 95% intervals resample whole episodes (2,000 draws; seed 20260820) and use only windows uncensored for both groups:

| estimand | point estimate | episode-bootstrap 95% CI |
|---|---:|---:|
| P(h*_arm != h*_gripper) | 0.959 | [0.950, 0.967] |
| mean discarded valid positions | 23.06 | [22.11, 24.03] |
| discarded commitment fraction | 0.409 | [0.396, 0.423] |

## Offset prevalence

The complete per-offset pointwise and prefix-survival arrays are in `oracle_group_horizon_metrics.json`. Selected refresh prefix-survival values are:

| offset k | arm | gripper | observed rows |
|---:|---:|---:|---:|
| 1 | 1.000 | 0.931 | 3718 |
| 2 | 0.999 | 0.881 | 3696 |
| 4 | 0.995 | 0.818 | 3652 |
| 8 | 0.971 | 0.722 | 3576 |
| 16 | 0.877 | 0.581 | 3441 |
| 32 | 0.715 | 0.291 | 3143 |
| 64 | 0.395 | 0.111 | 2153 |
| 99 | 0.292 | 0.000 | 1299 |

## Fixed threshold sensitivity

This predeclared audit rescored the cached old/refreshed action pairs at tolerance multipliers 0.75, 1.0, and 1.25. It performed no new frozen-policy inference, did not use rollout success, and did not tune the threshold toward a desired result. `k=0` remains excluded from horizon evidence.

| tolerance multiplier | both-uncensored fraction | P(different horizons) | mean discarded positions | discarded fraction | arm censoring | gripper censoring |
|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.813 | 0.952 | 16.16 | 0.374 | 0.182 | 0.053 |
| 1.0 | 0.513 | 0.959 | 23.06 | 0.409 | 0.484 | 0.059 |
| 1.25 | 0.212 | 0.955 | 28.58 | 0.465 | 0.787 | 0.059 |

## Task and offline phase variation

Task and normalized-episode-phase summaries are retrospective analyses. Progress, phase, and terminal episode length were not used to select a PACE horizon and must not be estimator inputs.

### Task-conditioned lower-bound distributions

| task | rows | difference rate | discarded positions | discarded fraction | arm mean | gripper mean |
|---|---:|---:|---:|---:|---:|---:|
| pick up the alphabet soup and place it in the basket | 376 | 0.981 | 23.34 | 0.467 | 41.02 | 15.92 |
| pick up the bbq sauce and place it in the basket | 379 | 0.992 | 20.66 | 0.325 | 42.59 | 28.30 |
| pick up the butter and place it in the basket | 389 | 0.974 | 27.40 | 0.460 | 45.97 | 29.94 |
| pick up the chocolate pudding and place it in the basket | 427 | 0.975 | 24.22 | 0.396 | 54.49 | 26.76 |
| pick up the cream cheese and place it in the basket | 365 | 0.933 | 24.44 | 0.464 | 38.75 | 21.48 |
| pick up the ketchup and place it in the basket | 375 | 0.923 | 19.32 | 0.350 | 42.97 | 18.52 |
| pick up the milk and place it in the basket | 361 | 0.988 | 27.42 | 0.433 | 53.47 | 18.66 |
| pick up the orange juice and place it in the basket | 362 | 0.986 | 22.87 | 0.451 | 37.29 | 17.48 |
| pick up the salad dressing and place it in the basket | 361 | 0.934 | 21.02 | 0.333 | 45.03 | 27.05 |
| pick up the tomato sauce and place it in the basket | 345 | 0.869 | 21.01 | 0.434 | 47.67 | 18.10 |

### Offline phase-conditioned lower-bound distributions

| phase | rows | arm mean | gripper mean | arm censoring | gripper censoring |
|---|---:|---:|---:|---:|---:|
| early | 1091 | 56.37 | 13.80 | 0.284 | 0.000 |
| late | 1314 | 27.99 | 15.38 | 0.752 | 0.162 |
| middle | 1335 | 52.59 | 36.24 | 0.384 | 0.005 |

Full quantiles, uncensored-only strata, and per-stratum heterogeneity are in `oracle_group_horizon_metrics.json`.

## Figures

![figure](figures/oracle_horizon_distributions.png)
![figure](figures/oracle_arm_gripper_scatter.png)
![figure](figures/refresh_survival_by_offset.png)

## Limitations

- `Y_refresh` queries the frozen policy on a demonstrated future observation; it does not execute the old action in an environment.
- Censoring and teacher forcing prevent interpreting `h*` as a physical safety or task-success horizon.
- Group comparison is exact only when both group prefixes fail before their observed suffix ends; all-row rates are explicitly lower-bound comparisons.

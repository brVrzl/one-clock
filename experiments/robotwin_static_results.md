# RoboTwin static horizon results

Task: `place_can_basket`, configuration: `demo_clean`. The 12 rows use the same ordered 20 evaluation seeds: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`.

**STATUS: blocked/incomplete.** Timeout or error cells are retained as anomalies; no scientific success claim is made from an incomplete sweep.

Pinned RoboTwin SHA: `266f3aadf505a4f7fe9af0faa41a20f5f47cd123`; XPolicyLab SHA: `c37109c500be67d0dea6b36bf7337bbd26e763cd`; ACT chunk size: `50`.

## Global horizons

| Configuration | Arm | Gripper | Success | Queries/episode | Query rate | Mean source age |
|---|---:|---:|---:|---:|---:|---|
| G2 | 2 | 2 | 0.0% | 35.0 | 0.5000 | left_arm=0.50, left_gripper=0.50, right_arm=0.50, right_gripper=0.50 |
| G4 | 4 | 4 | 0.0% | 26.2 | 0.2500 | left_arm=1.50, left_gripper=1.50, right_arm=1.50, right_gripper=1.50 |
| G8 | 8 | 8 | 0.0% | 13.2 | 0.1257 | left_arm=3.49, left_gripper=3.49, right_arm=3.49, right_gripper=3.49 |
| G16 | 16 | 16 | 0.0% | 6.6 | 0.0629 | left_arm=7.47, left_gripper=7.47, right_arm=7.47, right_gripper=7.47 |

## Group-specific configurations

| Configuration | Arm | Gripper | Success | Queries/episode | Query rate | Mean source age |
|---|---:|---:|---:|---:|---:|---|
| A2G8 | 2 | 8 | 0.0% | 52.5 | 0.5000 | left_arm=0.50, left_gripper=3.49, right_arm=0.50, right_gripper=3.49 |
| A2G16 | 2 | 16 | 0.0% | 35.0 | 0.5000 | left_arm=0.50, left_gripper=7.47, right_arm=0.50, right_gripper=7.47 |
| A4G16 | 4 | 16 | 0.0% | 26.2 | 0.2500 | left_arm=1.50, left_gripper=7.47, right_arm=1.50, right_gripper=7.47 |
| A8G16 | 8 | 16 | 0.0% | 13.2 | 0.1257 | left_arm=3.49, left_gripper=7.47, right_arm=3.49, right_gripper=7.47 |
| A8G2 | 8 | 2 | 0.0% | 35.0 | 0.5000 | left_arm=3.49, left_gripper=0.50, right_arm=3.49, right_gripper=0.50 |
| A16G2 | 16 | 2 | 0.0% | 0.0 | 0.0000 |  |
| A16G4 | 16 | 4 | 0.0% | 26.2 | 0.2500 | left_arm=7.47, left_gripper=1.50, right_arm=7.47, right_gripper=1.50 |
| A16G8 | 16 | 8 | 0.0% | 8.8 | 0.1257 | left_arm=7.47, left_gripper=3.49, right_arm=7.47, right_gripper=3.49 |

## Arm/gripper success matrix

Rows are arm horizons and columns are gripper horizons.

| Arm \ Gripper | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| 2 | 0.0% | — | 0.0% | 0.0% |
| 4 | — | 0.0% | — | 0.0% |
| 8 | 0.0% | — | 0.0% | 0.0% |
| 16 | 0.0% | 0.0% | 0.0% | 0.0% |

## Query-rate matrix

| Arm \ Gripper | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| 2 | 0.5000 | — | 0.5000 | 0.5000 |
| 4 | — | 0.2500 | — | 0.2500 |
| 8 | 0.5000 | — | 0.1257 | 0.1257 |
| 16 | 0.0000 | 0.2500 | 0.1257 | 0.0629 |

## Paired and symmetric comparisons

Counts use only seeds with terminal records for both configurations; incomplete cells are not assigned a success.

| Pair | First wins | Second wins | Ties | Valid paired seeds | Incomplete | Success difference |
|---|---:|---:|---:|---:|---:|---:|
| `A2G8 vs A8G2` | 0 | 0 | 2 | 2 | 18 | 0 |
| `A2G16 vs A16G2` | 0 | 0 | 0 | 0 | 20 | 0 |
| `A4G16 vs A16G4` | 0 | 0 | 3 | 3 | 17 | 0 |
| `A8G16 vs A16G8` | 0 | 0 | 2 | 2 | 18 | 0 |

## Interpretation

- Best global configuration(s): `G2, G4, G8, G16` at 0.0%.
- Best off-diagonal success rate: 0.0%; empirical success/query Pareto labels: ``.
- Query-budget matching is reported using the measured policy-query rate, not configured horizon alone.
- Classification is deliberately mechanical: A means success varies across complete evaluated configurations; C means no variation; BLOCKED means at least one required seed/configuration did not terminate.

- Anomalies: `{"A16G2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "A16G4": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "A16G8": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "A2G16": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "A2G8": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "A4G16": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "A8G16": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "A8G2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "G16": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "G2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "G4": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "G8": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}`.
- Execution blocker: Pinned headless SAPIEN/MPLIB qpos rollout stalled before terminal completion for some fixed-horizon cells. Timeout attempts recorded: `7`.
## Reproducibility

The JSON artifact contains per-episode seeds, success, environment steps, policy queries, configured horizons, and source-age summaries. Raw step traces remain under `experiments/runs/` and are ignored by git.

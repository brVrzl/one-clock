# ICRA 2027 final canonical numbers

Freeze date: 2026-09-03 (Asia/Shanghai)

Authoritative experiment commit: `2a2dc2ff8980c8c38097b8345f7b448ca7f1e90d`

This file is a scientific writing-support artifact. It extracts existing canonical outputs only. Scientific experimentation is closed.

## 1. R1A: complete frozen temporal-sensitivity curves

Status: `EXPOSED_DEVELOPMENT_CHARACTERIZATION`.

### 1.1 Exact cell accounting and anchor identity

- Cohort: ACT on `libero_object`, tasks `1,2,3,4,5,6,7,8,9`.
- Frozen states for every task: `20,21,22,23,27,31,34,35,38,39,44,45,47,48`.
- Blocks per condition: 9 tasks x 14 task-state blocks = 126 paired blocks.
- Environment-seed rule: `330000 + 100 * task_id + state_id`; policy seed: `424242`.
- Checkpoint: `/home/wjq/checkpoints/zeromidnight_act_libero_object` (`zeromidnight/act_libero_object`).
- Evaluator: 20 Hz, 280-step episode cap, 256-pixel observations, installed LeRobot 0.4.4 ACT, seven-dimensional action with translation `0:3`, rotation `3:6`, and gripper `6`.
- Same-target contract: `q+k=t`; for `t<d`, every dimension executes the Fresh `A_t[0]` prefix; one whole-policy query is made per executed environment step.
- Complete frozen analyzed condition list: `A0_G0`; `A2_G0`, `A0_G2`; `A4_G0`, `A0_G4`; `A8_G0`, `A0_G8`; `A12_G0`, `A0_G12`; `A16_G0`, `A0_G16`; `A20_G0`, `A0_G20`; `A32_G0`, `A0_G32`.
- Newly executed R1A conditions, exactly 12: `A2_G0`, `A0_G2`, `A4_G0`, `A0_G4`, `A8_G0`, `A0_G8`, `A12_G0`, `A0_G12`, `A16_G0`, `A0_G16`, `A32_G0`, `A0_G32`.
- Cells per newly executed condition: 126. Frozen product: 9 tasks x 14 states x 12 new conditions = `1512`, exactly matching manifest `cell_count=1512` and the terminal `1512/1512` PASS result. Retries and duplicates were both zero.

Fresh/d=0 was not newly executed inside R1A. `A0_G0` (Fresh), `A20_G0` (historical `REVERSE20`), and `A0_G20` (historical `FO20`) were reused from `experiments/group_delay_factorial_act20`. The frozen reuse audit establishes exact matching on the task cohort, suite checkpoint, initial-state identities, environment seeds and task-state block identities, policy seed, seven-dimensional action grouping, fixed-source/Fresh-prefix semantics, dense query schedule, 20 Hz evaluator, simulator settings, 280-step cap, and success criterion. Thus Fresh/d=0 and both d=20 branches are audited historical anchors, not new R1A data.

### 1.2 Absolute curves and between-branch contrasts

Semantic names used throughout:

- `A_d G0`: stale arm, fresh gripper.
- `A0 G_d`: fresh arm, stale gripper.

The canonical between-branch sign convention is `S(A0 G_d) - S(A_d G0)`.

| d | Time | `A_d G0` success/N (%) | `A0 G_d` success/N (%) | Difference (pp) | Discordance `A0Gd only:AdG0 only` | Exact McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.10 s | 49/126 (38.89%) | 76/126 (60.32%) | +21.43 | 29:2 | 4.62867e-07 | [+13.49, +29.37] | [+11.11, +30.95] |
| 4 | 0.20 s | 49/126 (38.89%) | 82/126 (65.08%) | +26.19 | 40:7 | 1.07091e-06 | [+16.67, +35.71] | [+13.49, +38.10] |
| 8 | 0.40 s | 36/126 (28.57%) | 88/126 (69.84%) | +41.27 | 53:1 | 6.10623e-15 | [+32.54, +50.00] | [+32.54, +50.79] |
| 12 | 0.60 s | 21/126 (16.67%) | 88/126 (69.84%) | +53.17 | 68:1 | 2.37169e-19 | [+44.44, +61.90] | [+45.24, +61.90] |
| 16 | 0.80 s | 23/126 (18.25%) | 90/126 (71.43%) | +53.17 | 67:0 | 1.35525e-20 | [+44.44, +61.90] | [+42.86, +62.70] |
| 20 | 1.00 s | 12/126 (9.52%) | 81/126 (64.29%) | +54.76 | 70:1 | 6.09864e-20 | [+46.03, +63.49] | [+44.44, +65.08] |
| 32 | 1.60 s | 32/126 (25.40%) | 79/126 (62.70%) | +37.30 | 49:2 | 1.17861e-12 | [+28.57, +46.03] | [+26.19, +46.03] |

### 1.3 Single-branch sensitivity relative to Fresh

Fresh/d=0 is 56/126 (44.44%). These Fresh-relative contrasts and uncertainty procedures were frozen, so they are canonical rather than newly constructed.

| d | Contrast | Successes | Delta (pp) | Discordance `first only:second only` | Exact McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | `Fresh - A2_G0` | 56 vs 49 / 126 | +5.56 | 13:6 | 0.167068 | [-0.79, +12.70] | [-1.59, +12.70] |
| 2 | `Fresh - A0_G2` | 56 vs 76 / 126 | -15.87 | 1:21 | 1.09673e-05 | [-23.02, -9.52] | [-21.43, -10.32] |
| 4 | `Fresh - A4_G0` | 56 vs 49 / 126 | +5.56 | 20:13 | 0.296206 | [-3.17, +14.29] | [-3.97, +15.87] |
| 4 | `Fresh - A0_G4` | 56 vs 82 / 126 | -20.63 | 0:26 | 2.98023e-08 | [-27.78, -13.49] | [-30.95, -11.11] |
| 8 | `Fresh - A8_G0` | 56 vs 36 / 126 | +15.87 | 27:7 | 0.000821395 | [+7.14, +24.60] | [+8.73, +23.81] |
| 8 | `Fresh - A0_G8` | 56 vs 88 / 126 | -25.40 | 1:33 | 4.07454e-09 | [-33.33, -17.46] | [-37.30, -15.87] |
| 12 | `Fresh - A12_G0` | 56 vs 21 / 126 | +27.78 | 38:3 | 1.04792e-08 | [+19.05, +36.51] | [+19.05, +36.51] |
| 12 | `Fresh - A0_G12` | 56 vs 88 / 126 | -25.40 | 2:34 | 1.94123e-08 | [-34.13, -17.46] | [-37.30, -15.08] |
| 16 | `Fresh - A16_G0` | 56 vs 23 / 126 | +26.19 | 36:3 | 3.60887e-08 | [+17.46, +34.92] | [+15.87, +36.51] |
| 16 | `Fresh - A0_G16` | 56 vs 90 / 126 | -26.98 | 1:35 | 1.07684e-09 | [-34.92, -19.05] | [-38.89, -16.67] |
| 20 | `Fresh - A20_G0` | 56 vs 12 / 126 | +34.92 | 45:1 | 1.33582e-12 | [+26.19, +43.65] | [+25.40, +45.24] |
| 20 | `Fresh - A0_G20` | 56 vs 81 / 126 | -19.84 | 1:26 | 4.17233e-07 | [-27.78, -12.70] | [-30.95, -9.52] |
| 32 | `Fresh - A32_G0` | 56 vs 32 / 126 | +19.05 | 33:9 | 0.000271539 | [+9.52, +28.57] | [+8.73, +28.57] |
| 32 | `Fresh - A0_G32` | 56 vs 79 / 126 | -18.25 | 1:24 | 1.54972e-06 | [-25.40, -11.11] | [-28.57, -8.73] |

### 1.4 Canonical per-task effect matrices

All entries are percentage points.

| Task | `Fresh-A2G0` | `Fresh-A4G0` | `Fresh-A8G0` | `Fresh-A12G0` | `Fresh-A16G0` | `Fresh-A20G0` | `Fresh-A32G0` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Object 1 | +7.14 | +7.14 | +7.14 | +7.14 | +7.14 | +14.29 | +7.14 |
| Object 2 | +14.29 | -7.14 | +21.43 | +42.86 | +42.86 | +64.29 | +21.43 |
| Object 3 | -7.14 | +7.14 | +7.14 | +28.57 | +35.71 | +35.71 | +21.43 |
| Object 4 | +0.00 | -14.29 | +0.00 | +21.43 | +21.43 | +28.57 | +14.29 |
| Object 5 | +14.29 | +28.57 | +35.71 | +42.86 | +50.00 | +50.00 | +42.86 |
| Object 6 | +14.29 | +14.29 | +28.57 | +35.71 | +28.57 | +42.86 | +21.43 |
| Object 7 | -7.14 | -7.14 | +7.14 | +14.29 | +14.29 | +14.29 | -14.29 |
| Object 8 | +21.43 | +28.57 | +28.57 | +42.86 | +35.71 | +35.71 | +21.43 |
| Object 9 | -7.14 | -7.14 | +7.14 | +14.29 | +0.00 | +28.57 | +35.71 |

| Task | `Fresh-A0G2` | `Fresh-A0G4` | `Fresh-A0G8` | `Fresh-A0G12` | `Fresh-A0G16` | `Fresh-A0G20` | `Fresh-A0G32` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Object 1 | -28.57 | -50.00 | -64.29 | -64.29 | -64.29 | -50.00 | -42.86 |
| Object 2 | -28.57 | -28.57 | -28.57 | -28.57 | -21.43 | -21.43 | -14.29 |
| Object 3 | -14.29 | -35.71 | -21.43 | -14.29 | -28.57 | -21.43 | -28.57 |
| Object 4 | -21.43 | -28.57 | -42.86 | -42.86 | -42.86 | -42.86 | -42.86 |
| Object 5 | -14.29 | -7.14 | -14.29 | -7.14 | -7.14 | +7.14 | +0.00 |
| Object 6 | -7.14 | -7.14 | -7.14 | -7.14 | -7.14 | -7.14 | -7.14 |
| Object 7 | -14.29 | -14.29 | -21.43 | -28.57 | -28.57 | -14.29 | -14.29 |
| Object 8 | -14.29 | -14.29 | -14.29 | -14.29 | -21.43 | -14.29 | -14.29 |
| Object 9 | +0.00 | +0.00 | -14.29 | -21.43 | -21.43 | -14.29 | +0.00 |

| Task | `A0G2-A2G0` | `A0G4-A4G0` | `A0G8-A8G0` | `A0G12-A12G0` | `A0G16-A16G0` | `A0G20-A20G0` | `A0G32-A32G0` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Object 1 | +35.71 | +57.14 | +71.43 | +71.43 | +71.43 | +64.29 | +50.00 |
| Object 2 | +42.86 | +21.43 | +50.00 | +71.43 | +64.29 | +85.71 | +35.71 |
| Object 3 | +7.14 | +42.86 | +28.57 | +42.86 | +64.29 | +57.14 | +50.00 |
| Object 4 | +21.43 | +14.29 | +42.86 | +64.29 | +64.29 | +71.43 | +57.14 |
| Object 5 | +28.57 | +35.71 | +50.00 | +50.00 | +57.14 | +42.86 | +42.86 |
| Object 6 | +21.43 | +21.43 | +35.71 | +42.86 | +35.71 | +50.00 | +28.57 |
| Object 7 | +7.14 | +7.14 | +28.57 | +42.86 | +42.86 | +28.57 | +0.00 |
| Object 8 | +35.71 | +42.86 | +42.86 | +57.14 | +57.14 | +50.00 | +35.71 |
| Object 9 | -7.14 | -7.14 | +21.43 | +35.71 | +21.43 | +42.86 | +35.71 |

### 1.5 Curve-shape freeze

`POST_HOC_DESCRIPTIVE`:

- Stale arm/fresh gripper (`A_d G0`) is not monotone. Success is 38.89%, 38.89%, 28.57%, 16.67%, 18.25%, 9.52%, and 25.40% across increasing d. Its largest observed drop from Fresh is 34.92 pp at d=20; its minimum is 12/126 (9.52%) at d=20. It degrades through d=12, rises slightly at d=16, reaches its minimum at d=20, and recovers at d=32. Exactly, d=16 -> d=20 -> d=32 is 23/126 (18.25%) -> 12/126 (9.52%) -> 32/126 (25.40%), changes of -8.73 pp and +15.87 pp.
- Fresh arm/stale gripper (`A0 G_d`) is not monotone. Success is 60.32%, 65.08%, 69.84%, 69.84%, 71.43%, 64.29%, and 62.70%. There is no observed drop below Fresh at any frozen lag; the lowest lagged value is 76/126 (60.32%) at d=2, still +15.87 pp relative to Fresh. The maximum is 90/126 (71.43%) at d=16. After d=16 it degrades over the two largest frozen lags but remains above Fresh. Exactly, d=16 -> d=20 -> d=32 is 90/126 (71.43%) -> 81/126 (64.29%) -> 79/126 (62.70%), changes of -7.14 pp and -1.59 pp.
- `S(A0 G_d) > S(A_d G0)` at every frozen d; no ordering reversal occurs. The observed separation range over 0.10-1.60 s is `+21.43 to +54.76 pp`. The minimum is +21.43 pp at d=2; the maximum is +54.76 pp at d=20. The exact d=16, d=20, and d=32 separations are +53.17, +54.76, and +37.30 pp.
- From d=20 to d=32 the separation narrows by 17.46 pp because both observed branch curves change: stale-arm success improves by 15.87 pp while stale-gripper success degrades by 1.59 pp. This is a descriptive decomposition, not a causal attribution.
- d=20 was frozen before R1A existed and was not selected from the sweep. It attained the largest observed separation on the frozen R1A grid. No preregistered lag-versus-lag inference establishes that d=20 is a statistically distinct peak.

The original confirmatory 140-block d=20 result is `+32.14 pp`; it is not the numerical maximum of the later 126-block R1A grid. The later grid's maximum is `+54.76 pp`, on its separate exposed-development cohort.

## 2. R1B: translation versus rotation

Status: frozen reviewer-directed `EXPOSED_DEVELOPMENT_CHARACTERIZATION`; not new confirmatory evidence.

### 2.1 Frozen intervention definitions

The exact frozen design contains two new conditions plus the reused Fresh anchor, not a three-way stale-component experiment:

> On the same 126 blocks at `d=20`, run `T20_R0_G0` (translation stale; rotation and gripper Fresh) and `T0_R20_G0` (rotation stale; translation and gripper Fresh).

- `T20_R0_G0`: at `t>=20`, translation dimensions 0-2 use the 20-step-old same-target source `A_(t-20)[20]`, while rotation dimensions 3-5 and gripper dimension 6 use Fresh `A_t[0]`; for `t<20`, all dimensions are Fresh.
- `T0_R20_G0`: at `t>=20`, rotation dimensions 3-5 use the 20-step-old same-target source `A_(t-20)[20]`, while translation dimensions 0-2 and gripper dimension 6 use Fresh `A_t[0]`; for `t<20`, all dimensions are Fresh.
- Both use `q+k=t`, one whole-policy query per executed 20 Hz environment step, and d=20 = 1.00 s.

Absolute results: translation-stale `T20_R0_G0` = 11/126 (8.73%); rotation-stale `T0_R20_G0` = 53/126 (42.06%); reused Fresh `A0_G0` = 56/126 (44.44%).

### 2.2 Canonical contrasts and explicit sign

The primary sign convention is `S(translation-stale) - S(rotation-stale)`.

| Contrast | Successes | Delta (pp) | Discordance `first only:second only` | Exact McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| `T20_R0_G0 - T0_R20_G0` | 11 vs 53 / 126 | -33.33 | 3:45 | 1.31259e-10 | [-42.06, -24.60] | [-44.44, -22.22] |
| `T20_R0_G0 - Fresh` | 11 vs 56 / 126 | -35.71 | 3:48 | 1.96749e-11 | [-45.24, -26.19] | [-46.03, -25.40] |
| `T0_R20_G0 - Fresh` | 53 vs 56 / 126 | -2.38 | 6:9 | 0.607239 | [-8.73, +3.97] | [-6.35, +1.59] |

Under `S(translation-stale) - S(rotation-stale)`, the estimate is `-33.33 pp`; the paired interval excludes zero and lies strictly below zero, and the task-cluster interval excludes zero and lies strictly below zero.

Primary per-task vector (Object tasks 1-9, pp): `[-7.14, -57.14, -28.57, -28.57, -50.00, -35.71, -14.29, -21.43, -57.14]`.

Primary leave-one-task-out vector in the same omitted-task order (pp): `[-36.61, -30.36, -33.93, -33.93, -31.25, -33.04, -35.71, -34.82, -30.36]`. Every frozen LOTO estimate is negative, so direction is stable under every reported frozen LOTO sensitivity.

### 2.3 B1-derived prospective prediction and scope gate

The exact prospective prediction was:

> B1 found the same-target normalized dispersion ordering `rotation > translation > gripper` for both ACT and SmolVLA. If same-target source disagreement predicts behavioral temporal sensitivity, then at d=20 stale rotation should be at least as damaging as stale translation: `success(T0_R20_G0) <= success(T20_R0_G0)`.

Canonical ACT normalized dispersion is rotation 0.1484, translation 0.1364, and gripper 0.0790, hence the descriptive B1 ordering is `rotation > translation > gripper`. R1B instead orders behavioral temporal sensitivity `translation > rotation`: translation staleness lowers success by 35.71 pp from Fresh, whereas rotation staleness lowers it by 2.38 pp. The behavioral ordering disagrees with the prospective B1-derived ordering.

Narrow classification: `CONTRADICTED_AS_AN_ORDERING_PREDICTOR_WITHIN_ARM`.

Within the arm, same-target dispersion ranks rotation above translation, whereas behavioral temporal sensitivity ranks translation above rotation. No new test links B1 to R1B, and the measured B1 dispersion remains valid.

Because both canonical primary R1B intervals are strictly below zero and every frozen LOTO estimate has the same negative direction, Scope A is supported: `component-dependent temporal sensitivity within ACT` is defensible. This scope remains ACT-focused and Object-development-specific.

## 3. R1C: exact dense-query factorial

Status: frozen reviewer-directed `POST_HOC_QUERY_MATCHED_EXTENSION`.

All four conditions make one whole-policy query every executed environment step:

| Condition | Reader-facing semantics | Success/N | Success |
|---|---|---:|---:|
| `C00` | dense-query Fresh arm + Fresh gripper | 77/140 | 55.00% |
| `C10` | dense-query scheduled-H16 arm + Fresh gripper | 76/140 | 54.29% |
| `C01` | dense-query Fresh arm + scheduled-H16 gripper | 81/140 | 57.86% |
| `C11` | dense-query scheduled-H16 arm + scheduled-H16 gripper | 93/140 | 66.43% |

`C00` and `C10` are audited historical reuse; `C01` and `C11` are the 280 new cells.

| Frozen contrast | Interpretation | Delta (pp) | Discordance | Exact McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) |
|---|---|---:|---:|---:|---:|---:|
| `C10-C00` | conditional scheduled-arm effect with gripper Fresh | -0.71 | 18:19 | 1 | [-9.29, +7.86] | [-12.14, +10.71] |
| `C01-C00` | conditional scheduled-gripper effect with arm Fresh | +2.86 | 10:6 | 0.454498 | [-2.86, +8.57] | [-4.29, +9.29] |
| `C11-C10` | conditional scheduled-gripper effect with arm scheduled | +12.14 | 20:3 | 0.000488281 | [+5.71, +18.57] | [+3.57, +21.43] |
| `C11-C01` | conditional scheduled-arm effect with gripper scheduled | +8.57 | 26:14 | 0.0806905 | [+0.00, +17.14] | [-2.14, +21.43] |
| `C11-C00` | diagonal: both scheduled versus both Fresh | +11.43 | 28:12 | 0.016589 | [+2.86, +20.00] | [-0.71, +23.57] |
| `C10-C01` | opposite diagonal | -3.57 | 12:17 | 0.458258 | [-11.43, +4.29] | [-14.29, +6.43] |

The frozen risk-difference interaction is `C11 - C10 - C01 + C00 = +9.29 pp`. The canonical artifact freezes only this observed interaction value; no new uncertainty is added here. No component contribution percentages are derived.

### 3.1 C11 identity canary

Frozen canary definition:

> Before bulk rollout a frozen deterministic canary must show that unused dense forward passes leave C11 executed actions, simulator trajectory, terminal result, and length identical to sparse HARD-H16. Canary failure stops R1C.

Canonical result: `PASS`, excluded from scientific analysis. The persisted checks were:

- exact executed action sequence: true;
- exact episode length: true;
- exact terminal success: true;
- exact completion step: true;
- `trajectory_identity_from_same_initial_state_and_exact_actions`: true.

The trajectory identity was the frozen deterministic definition based on the same initial state and exact executed actions; the canary did not persist a separate elementwise environment-state trajectory array comparison. Under that intended definition, additional discarded policy queries did not alter the executed frozen C11 behavior under the tested deterministic ACT evaluator.

The old confound was that the same-target diagnostic probes queried the policy every step whereas sparse coherent H16 queried only every 16 steps, leaving policy-query schedule/count as an alternative explanation. R1C makes a whole-policy query every step in all four cells and discards unused predictions where scheduled execution requires it. Together with the passed C11 identity canary, it removes that query-schedule confound for the tested ACT component comparison under this deterministic evaluator. It does not generalize to arbitrary stochastic policies, inference stacks, or randomized environments.

## 4. B3: complete canonical forecastability curves

Formal classification: `B3_NO_FROZEN_DISCRIMINATIVE_CRITERION`.

### 4.1 Frozen rule, metric, and aggregation

The frozen B3 analysis rule was:

> If and only if the temporal-contract audit establishes an unambiguous mapping, compare the complete ACT B3 future-action forecast curves with the complete R1A behavioral sensitivity curves on a seconds-first axis. The provenance-only timebase audit resolved the 20 Hz R1A ages `d={2,4,8,12,16,20,32}` to the exact 10 Hz B3 offsets `k={1,2,4,6,8,10,16}` at matched times `{0.1,0.2,0.4,0.6,0.8,1.0,1.6}` seconds, so no interpolation is used. The table must include B3 arm, translation, rotation, gripper-value, and gripper-sign forecast quantities and R1A `Fresh-A_d_G0` and `Fresh-A0_G_d`. This is cross-cohort descriptive characterization, not a significance-gated hypothesis. If time alignment is ambiguous, report the comparison as not identifiable and keep native step axes separate. The full B3 offset grid `0..32` and every frozen R1A age remain reported in their own canonical analyses; this matched-time table is not a lag-selection rule.

The historical 10 Hz wording above is retained verbatim for governance. The later authoritative corrective temporal audit established `time=k/20 s`; exact R1A matching is therefore `d={2,4,8,12,16,20,32}` to `k={2,4,8,12,16,20,32}`. No interpolation is used and the characterization rule is otherwise unchanged.

The frozen uncertainty rule was:

> Episode-cluster percentile bootstrap uses 20,000 draws, 95% intervals, seed 27401 for ACT and 27402 for SmolVLA. No performance outcome, task subset, lag, or offset is selected after results. These associations do not establish that persistence or forecastability causes executor sensitivity.

The frozen record contains no operational threshold, correlation, or ordering criterion for discriminative mechanism support. None is constructed now.

Metric: at each demonstration anchor t, compare the policy's frozen normalized predicted action at chunk offset k with the recorded demonstrated action at exact target row t+k. Report translation, rotation, and combined-arm normalized RMSE; gripper absolute normalized error (the one-dimensional gripper RMSE); and controller-native gripper sign-disagreement probability. Physical time is `k/20 s` for all k=0..32.

Aggregation: training-demonstration reference, not held-out. Each policy has 40 frozen demonstration episodes: the lowest ten numeric training episode IDs for Object 3, Spatial 0, Goal 2, and LIBERO-10 3. Centers pool squared errors or sign counts across all valid anchors in those 40 episodes within each policy. The canonical uncertainty resamples demonstration episodes as clusters across the combined 40-episode policy panel (20,000 draws). ACT and SmolVLA are reported separately in their own normalized spaces. Eight of eight task-policy shards completed.

### 4.2 ACT complete frozen curve

Each entry is center `[episode-cluster 95% CI]`.

| k | Time | Translation RMSE | Rotation RMSE | Arm RMSE | Gripper absolute normalized error | Gripper sign disagreement |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 s | 0.0992 [0.0893, 0.1103] | 0.1443 [0.1322, 0.1575] | 0.1238 [0.1137, 0.1349] | 0.0959 [0.0481, 0.1479] | 0.0021 [0.0000, 0.0069] |
| 1 | 0.05 s | 0.0916 [0.0849, 0.0992] | 0.1392 [0.1279, 0.1508] | 0.1178 [0.1096, 0.1267] | 0.1107 [0.0707, 0.1481] | 0.0021 [0.0000, 0.0072] |
| 2 | 0.10 s | 0.0891 [0.0833, 0.0956] | 0.1271 [0.1181, 0.1352] | 0.1098 [0.1036, 0.1158] | 0.1040 [0.0648, 0.1447] | 0.0021 [0.0000, 0.0070] |
| 3 | 0.15 s | 0.0909 [0.0856, 0.0964] | 0.1279 [0.1184, 0.1369] | 0.1110 [0.1044, 0.1173] | 0.1736 [0.1035, 0.2368] | 0.0105 [0.0022, 0.0205] |
| 4 | 0.20 s | 0.0907 [0.0809, 0.1026] | 0.1238 [0.1157, 0.1321] | 0.1085 [0.1009, 0.1165] | 0.0808 [0.0542, 0.1036] | 0.0000 [0.0000, 0.0000] |
| 5 | 0.25 s | 0.0956 [0.0838, 0.1093] | 0.1251 [0.1155, 0.1347] | 0.1114 [0.1021, 0.1205] | 0.0871 [0.0556, 0.1131] | 0.0021 [0.0000, 0.0066] |
| 6 | 0.30 s | 0.0966 [0.0843, 0.1130] | 0.1205 [0.1109, 0.1300] | 0.1092 [0.1005, 0.1183] | 0.1001 [0.0687, 0.1268] | 0.0000 [0.0000, 0.0000] |
| 7 | 0.35 s | 0.0992 [0.0840, 0.1182] | 0.1290 [0.1176, 0.1397] | 0.1151 [0.1054, 0.1248] | 0.1139 [0.0727, 0.1509] | 0.0021 [0.0000, 0.0066] |
| 8 | 0.40 s | 0.0949 [0.0798, 0.1162] | 0.1219 [0.1122, 0.1313] | 0.1092 [0.1000, 0.1192] | 0.1042 [0.0588, 0.1417] | 0.0042 [0.0000, 0.0106] |
| 9 | 0.45 s | 0.0885 [0.0791, 0.1007] | 0.1215 [0.1124, 0.1298] | 0.1063 [0.0988, 0.1136] | 0.0671 [0.0366, 0.1027] | 0.0021 [0.0000, 0.0071] |
| 10 | 0.50 s | 0.0880 [0.0808, 0.0961] | 0.1216 [0.1099, 0.1342] | 0.1062 [0.0975, 0.1154] | 0.0622 [0.0389, 0.0852] | 0.0000 [0.0000, 0.0000] |
| 11 | 0.55 s | 0.0817 [0.0767, 0.0872] | 0.1215 [0.1115, 0.1317] | 0.1035 [0.0964, 0.1111] | 0.0763 [0.0537, 0.0987] | 0.0000 [0.0000, 0.0000] |
| 12 | 0.60 s | 0.0821 [0.0779, 0.0865] | 0.1209 [0.1111, 0.1301] | 0.1034 [0.0970, 0.1094] | 0.0765 [0.0526, 0.0991] | 0.0000 [0.0000, 0.0000] |
| 13 | 0.65 s | 0.0868 [0.0809, 0.0930] | 0.1203 [0.1099, 0.1298] | 0.1049 [0.0974, 0.1118] | 0.0858 [0.0574, 0.1117] | 0.0000 [0.0000, 0.0000] |
| 14 | 0.70 s | 0.0870 [0.0792, 0.0952] | 0.1162 [0.1070, 0.1245] | 0.1026 [0.0950, 0.1100] | 0.1017 [0.0545, 0.1388] | 0.0042 [0.0000, 0.0106] |
| 15 | 0.75 s | 0.0854 [0.0782, 0.0932] | 0.1182 [0.1081, 0.1282] | 0.1031 [0.0948, 0.1112] | 0.1150 [0.0652, 0.1539] | 0.0063 [0.0000, 0.0134] |
| 16 | 0.80 s | 0.0863 [0.0793, 0.0941] | 0.1153 [0.1061, 0.1247] | 0.1018 [0.0949, 0.1088] | 0.1101 [0.0588, 0.1573] | 0.0021 [0.0000, 0.0066] |
| 17 | 0.85 s | 0.0905 [0.0807, 0.1015] | 0.1257 [0.1126, 0.1397] | 0.1095 [0.0999, 0.1198] | 0.1216 [0.0673, 0.1757] | 0.0021 [0.0000, 0.0067] |
| 18 | 0.90 s | 0.0865 [0.0781, 0.0961] | 0.1206 [0.1108, 0.1301] | 0.1049 [0.0972, 0.1125] | 0.1014 [0.0359, 0.1463] | 0.0063 [0.0000, 0.0140] |
| 19 | 0.95 s | 0.0843 [0.0779, 0.0911] | 0.1209 [0.1095, 0.1330] | 0.1042 [0.0963, 0.1126] | 0.0685 [0.0403, 0.0935] | 0.0000 [0.0000, 0.0000] |
| 20 | 1.00 s | 0.0892 [0.0810, 0.0981] | 0.1230 [0.1094, 0.1361] | 0.1074 [0.0972, 0.1177] | 0.1127 [0.0427, 0.1653] | 0.0042 [0.0000, 0.0102] |
| 21 | 1.05 s | 0.0850 [0.0786, 0.0920] | 0.1244 [0.1123, 0.1366] | 0.1066 [0.0978, 0.1158] | 0.1233 [0.0678, 0.1683] | 0.0042 [0.0000, 0.0104] |
| 22 | 1.10 s | 0.0867 [0.0807, 0.0932] | 0.1243 [0.1135, 0.1345] | 0.1072 [0.0995, 0.1148] | 0.1078 [0.0735, 0.1386] | 0.0000 [0.0000, 0.0000] |
| 23 | 1.15 s | 0.0886 [0.0817, 0.0959] | 0.1178 [0.1088, 0.1263] | 0.1042 [0.0974, 0.1109] | 0.1166 [0.0782, 0.1518] | 0.0042 [0.0000, 0.0112] |
| 24 | 1.20 s | 0.0903 [0.0818, 0.0991] | 0.1171 [0.1068, 0.1266] | 0.1045 [0.0962, 0.1127] | 0.1769 [0.0978, 0.2362] | 0.0127 [0.0022, 0.0248] |
| 25 | 1.25 s | 0.0862 [0.0781, 0.0944] | 0.1192 [0.1084, 0.1305] | 0.1040 [0.0954, 0.1127] | 0.1703 [0.0872, 0.2307] | 0.0063 [0.0000, 0.0131] |
| 26 | 1.30 s | 0.0847 [0.0762, 0.0928] | 0.1231 [0.1090, 0.1372] | 0.1057 [0.0958, 0.1151] | 0.1375 [0.0867, 0.1834] | 0.0042 [0.0000, 0.0106] |
| 27 | 1.35 s | 0.0911 [0.0787, 0.1042] | 0.1378 [0.1217, 0.1535] | 0.1168 [0.1044, 0.1291] | 0.1763 [0.1168, 0.2317] | 0.0063 [0.0000, 0.0140] |
| 28 | 1.40 s | 0.0888 [0.0785, 0.0993] | 0.1255 [0.1150, 0.1355] | 0.1087 [0.0996, 0.1171] | 0.1212 [0.0748, 0.1619] | 0.0042 [0.0000, 0.0106] |
| 29 | 1.45 s | 0.0905 [0.0818, 0.0991] | 0.1284 [0.1174, 0.1394] | 0.1111 [0.1023, 0.1194] | 0.0817 [0.0440, 0.1165] | 0.0021 [0.0000, 0.0066] |
| 30 | 1.50 s | 0.0866 [0.0790, 0.0946] | 0.1274 [0.1110, 0.1441] | 0.1089 [0.0975, 0.1204] | 0.0924 [0.0485, 0.1357] | 0.0021 [0.0000, 0.0066] |
| 31 | 1.55 s | 0.0834 [0.0759, 0.0914] | 0.1382 [0.1182, 0.1603] | 0.1142 [0.1003, 0.1290] | 0.1199 [0.0724, 0.1636] | 0.0063 [0.0000, 0.0141] |
| 32 | 1.60 s | 0.0888 [0.0782, 0.1013] | 0.1502 [0.1262, 0.1739] | 0.1234 [0.1055, 0.1417] | 0.1218 [0.0811, 0.1585] | 0.0042 [0.0000, 0.0106] |

### 4.3 SmolVLA complete frozen curve

Each entry is center `[episode-cluster 95% CI]`. Values are in SmolVLA's own frozen normalized space and are not numerically compared with ACT's normalized values.

| k | Time | Translation RMSE | Rotation RMSE | Arm RMSE | Gripper absolute normalized error | Gripper sign disagreement |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 s | 0.5143 [0.4550, 0.5871] | 0.8354 [0.7759, 0.8936] | 0.6937 [0.6427, 0.7476] | 0.4089 [0.3149, 0.5005] | 0.0422 [0.0251, 0.0629] |
| 1 | 0.05 s | 0.4879 [0.4254, 0.5642] | 0.7929 [0.7360, 0.8455] | 0.6583 [0.6082, 0.7102] | 0.4226 [0.3324, 0.5083] | 0.0443 [0.0274, 0.0640] |
| 2 | 0.10 s | 0.4805 [0.4159, 0.5630] | 0.7822 [0.7236, 0.8383] | 0.6491 [0.5974, 0.7026] | 0.4505 [0.3644, 0.5257] | 0.0506 [0.0331, 0.0690] |
| 3 | 0.15 s | 0.4764 [0.4166, 0.5505] | 0.7769 [0.7101, 0.8423] | 0.6444 [0.5894, 0.7004] | 0.4942 [0.4074, 0.5717] | 0.0612 [0.0416, 0.0819] |
| 4 | 0.20 s | 0.4680 [0.4087, 0.5436] | 0.7706 [0.7017, 0.8393] | 0.6375 [0.5819, 0.6944] | 0.5315 [0.4586, 0.5996] | 0.0717 [0.0539, 0.0905] |
| 5 | 0.25 s | 0.4693 [0.4061, 0.5476] | 0.7527 [0.6891, 0.8147] | 0.6272 [0.5745, 0.6820] | 0.5109 [0.4465, 0.5715] | 0.0654 [0.0500, 0.0819] |
| 6 | 0.30 s | 0.4478 [0.3986, 0.5063] | 0.7448 [0.6846, 0.8031] | 0.6145 [0.5681, 0.6613] | 0.4739 [0.4084, 0.5388] | 0.0570 [0.0424, 0.0734] |
| 7 | 0.35 s | 0.4398 [0.3940, 0.4940] | 0.7337 [0.6716, 0.7927] | 0.6048 [0.5591, 0.6506] | 0.4193 [0.3393, 0.4895] | 0.0443 [0.0292, 0.0603] |
| 8 | 0.40 s | 0.4329 [0.3884, 0.4862] | 0.7237 [0.6676, 0.7785] | 0.5963 [0.5539, 0.6383] | 0.4414 [0.3494, 0.5315] | 0.0485 [0.0302, 0.0704] |
| 9 | 0.45 s | 0.4333 [0.3883, 0.4861] | 0.7272 [0.6761, 0.7760] | 0.5986 [0.5579, 0.6375] | 0.4593 [0.3512, 0.5605] | 0.0527 [0.0308, 0.0786] |
| 10 | 0.50 s | 0.4342 [0.3905, 0.4857] | 0.7404 [0.6881, 0.7890] | 0.6069 [0.5650, 0.6479] | 0.4829 [0.3844, 0.5781] | 0.0591 [0.0377, 0.0842] |
| 11 | 0.55 s | 0.4359 [0.3909, 0.4896] | 0.7625 [0.7040, 0.8185] | 0.6210 [0.5741, 0.6688] | 0.4656 [0.3622, 0.5600] | 0.0549 [0.0333, 0.0790] |
| 12 | 0.60 s | 0.4451 [0.3891, 0.5116] | 0.7862 [0.7229, 0.8471] | 0.6389 [0.5862, 0.6933] | 0.4304 [0.3338, 0.5167] | 0.0464 [0.0280, 0.0668] |
| 13 | 0.65 s | 0.4513 [0.3939, 0.5202] | 0.7822 [0.7159, 0.8471] | 0.6386 [0.5841, 0.6944] | 0.5111 [0.4223, 0.5934] | 0.0654 [0.0446, 0.0882] |
| 14 | 0.70 s | 0.4493 [0.4006, 0.5042] | 0.7891 [0.7210, 0.8554] | 0.6421 [0.5886, 0.6956] | 0.5001 [0.4000, 0.5952] | 0.0633 [0.0408, 0.0894] |
| 15 | 0.75 s | 0.4519 [0.4059, 0.5004] | 0.7889 [0.7188, 0.8606] | 0.6429 [0.5890, 0.6963] | 0.4953 [0.3847, 0.5991] | 0.0612 [0.0367, 0.0897] |
| 16 | 0.80 s | 0.4882 [0.4313, 0.5503] | 0.7983 [0.7250, 0.8708] | 0.6616 [0.6036, 0.7204] | 0.4611 [0.3771, 0.5401] | 0.0527 [0.0352, 0.0725] |
| 17 | 0.85 s | 0.5100 [0.4424, 0.5875] | 0.7907 [0.7092, 0.8725] | 0.6653 [0.5976, 0.7364] | 0.5032 [0.4299, 0.5735] | 0.0633 [0.0461, 0.0822] |
| 18 | 0.90 s | 0.5235 [0.4497, 0.6083] | 0.7901 [0.7081, 0.8728] | 0.6702 [0.6004, 0.7444] | 0.5124 [0.4254, 0.5951] | 0.0654 [0.0448, 0.0885] |
| 19 | 0.95 s | 0.5329 [0.4480, 0.6342] | 0.8025 [0.7252, 0.8802] | 0.6812 [0.6090, 0.7590] | 0.5445 [0.4459, 0.6419] | 0.0738 [0.0495, 0.1026] |
| 20 | 1.00 s | 0.5459 [0.4591, 0.6466] | 0.8141 [0.7413, 0.8879] | 0.6931 [0.6234, 0.7690] | 0.5671 [0.4569, 0.6791] | 0.0802 [0.0518, 0.1152] |
| 21 | 1.05 s | 0.5605 [0.4683, 0.6680] | 0.8244 [0.7529, 0.8996] | 0.7049 [0.6339, 0.7852] | 0.5692 [0.4549, 0.6798] | 0.0823 [0.0526, 0.1174] |
| 22 | 1.10 s | 0.5684 [0.4842, 0.6646] | 0.8568 [0.7862, 0.9304] | 0.7270 [0.6593, 0.8017] | 0.6212 [0.5176, 0.7244] | 0.0970 [0.0676, 0.1320] |
| 23 | 1.15 s | 0.5629 [0.4803, 0.6565] | 0.8632 [0.7852, 0.9430] | 0.7287 [0.6572, 0.8053] | 0.6094 [0.4960, 0.7191] | 0.0928 [0.0615, 0.1293] |
| 24 | 1.20 s | 0.5722 [0.4850, 0.6707] | 0.8712 [0.7882, 0.9555] | 0.7370 [0.6631, 0.8165] | 0.5937 [0.5007, 0.6862] | 0.0886 [0.0633, 0.1182] |
| 25 | 1.25 s | 0.5678 [0.4892, 0.6579] | 0.8642 [0.7818, 0.9476] | 0.7312 [0.6594, 0.8083] | 0.5576 [0.4681, 0.6460] | 0.0781 [0.0551, 0.1047] |
| 26 | 1.30 s | 0.5742 [0.4952, 0.6648] | 0.8535 [0.7702, 0.9380] | 0.7274 [0.6573, 0.8027] | 0.5850 [0.4903, 0.6764] | 0.0865 [0.0608, 0.1156] |
| 27 | 1.35 s | 0.5746 [0.5023, 0.6576] | 0.8365 [0.7554, 0.9171] | 0.7176 [0.6508, 0.7890] | 0.5819 [0.4909, 0.6753] | 0.0844 [0.0601, 0.1137] |
| 28 | 1.40 s | 0.5743 [0.5005, 0.6591] | 0.8376 [0.7595, 0.9164] | 0.7181 [0.6501, 0.7917] | 0.5692 [0.4697, 0.6665] | 0.0823 [0.0561, 0.1127] |
| 29 | 1.45 s | 0.5855 [0.4975, 0.6880] | 0.8389 [0.7592, 0.9195] | 0.7234 [0.6480, 0.8066] | 0.6031 [0.4973, 0.7096] | 0.0907 [0.0615, 0.1256] |
| 30 | 1.50 s | 0.6016 [0.5052, 0.7140] | 0.8597 [0.7733, 0.9486] | 0.7419 [0.6582, 0.8343] | 0.6006 [0.5028, 0.7016] | 0.0928 [0.0654, 0.1259] |
| 31 | 1.55 s | 0.6106 [0.5212, 0.7144] | 0.8673 [0.7778, 0.9612] | 0.7500 [0.6674, 0.8425] | 0.6293 [0.5576, 0.7023] | 0.0992 [0.0778, 0.1235] |
| 32 | 1.60 s | 0.6199 [0.5335, 0.7217] | 0.8787 [0.7857, 0.9766] | 0.7604 [0.6768, 0.8549] | 0.6419 [0.5530, 0.7336] | 0.1034 [0.0768, 0.1351] |

### 4.4 Frozen descriptive conclusion

At the exact R1A-matched offsets `k=d`, ACT rotation forecast RMSE exceeds translation RMSE at all seven points, whereas R1B behavior shows translation staleness is much more damaging than rotation staleness. ACT combined-arm error is nearly flat over most of k=0..32, whereas R1A stale-arm harm grows and then recedes non-monotonically. ACT gripper sign disagreement remains near zero even where stale-gripper execution improves success relative to Fresh. The curves therefore fail to discriminate the R1A/R1B behavioral ordering under the frozen descriptive comparison. This is mixed/non-corresponding mechanism evidence, not a new significance gate. Status remains `B3_NO_FROZEN_DISCRIMINATIVE_CRITERION`, and mechanism search ends here.

## 5. R1D: exact Spatial factorial completion

Status: `POST_HOC_SPATIAL_FACTORIAL_COMPLETION`.

- Frozen suite/tasks: `libero_spatial`, tasks `0,1,2,3,4,5,6,7,8,9`.
- Frozen states for every task: `1,13,15,19,21,24,31,37,40,47`.
- Frozen seeds: `340000 + 100 * task_id + state_id`.
- Checkpoint: `/home/wjq/checkpoints/ishandotsh_act_libero_spatial_test`, immutable revision `8f04de1472975d62db214238b2fc07e78bde2474`.
- Reverse20 condition: `A20_G0`, stale arm at d=20 with fresh gripper, same-target `q+k=t`, all-Fresh prefix for `t<20`, dense whole-policy queries, 20 Hz, 280-step cap.
- Historical comparators: `A0_G0` (Fresh arm/Fresh gripper), `A0_G20` (Fresh arm/stale gripper), and `A20_G20` (stale arm/stale gripper).
- Previously missing cell completed: the Spatial `A20_G0`/Reverse20 condition, exactly 100 new task-state cells. Validator PASS; zero scientific retries; zero duplicates.

| Condition | Reader-facing semantics | Success/N | Success |
|---|---|---:|---:|
| `A0_G0` | Fresh arm + Fresh gripper | 40/100 | 40.00% |
| `A0_G20` | Fresh arm + 20-step-stale gripper | 40/100 | 40.00% |
| `A20_G0` | 20-step-stale arm + Fresh gripper | 12/100 | 12.00% |
| `A20_G20` | 20-step-stale arm + 20-step-stale gripper | 30/100 | 30.00% |

| Frozen contrast | Successes | Delta (pp) | Discordance | Exact McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| `A0_G20-A20_G0` | 40 vs 12 / 100 | +28.00 | 33:5 | 4.25596e-06 | [+17.00, +39.00] | [+13.00, +45.00] |
| `A20_G0-A0_G0` | 12 vs 40 / 100 | -28.00 | 3:31 | 7.66013e-07 | [-38.00, -18.00] | [-45.00, -13.00] |
| `A0_G20-A0_G0` | 40 vs 40 / 100 | +0.00 | 5:5 | 1 | [-6.00, +6.00] | [-5.00, +6.00] |
| `A20_G20-A0_G0` | 30 vs 40 / 100 | -10.00 | 7:17 | 0.0639147 | [-19.00, -1.00] | [-19.00, -2.00] |

Canonical per-task effects (Spatial tasks 0-9, pp):

| Task | `A0G20-A20G0` | `A20G0-Fresh` | `A0G20-Fresh` | `A20G20-Fresh` |
|---|---:|---:|---:|---:|
| Spatial 0 | +30.00 | -40.00 | -10.00 | -10.00 |
| Spatial 1 | +30.00 | -30.00 | +0.00 | -10.00 |
| Spatial 2 | +80.00 | -80.00 | +0.00 | -40.00 |
| Spatial 3 | +70.00 | -60.00 | +10.00 | +0.00 |
| Spatial 4 | +10.00 | -20.00 | -10.00 | -20.00 |
| Spatial 5 | +0.00 | +0.00 | +0.00 | +10.00 |
| Spatial 6 | +30.00 | -40.00 | -10.00 | -20.00 |
| Spatial 7 | +0.00 | +0.00 | +0.00 | +0.00 |
| Spatial 8 | +30.00 | -10.00 | +20.00 | -10.00 |
| Spatial 9 | +0.00 | +0.00 | +0.00 | +0.00 |

Against the original preregistered 140-block `A0G20-A20G0` asymmetry of +32.14 pp, R1D preserves the positive sign and has a 4.14 pp smaller point estimate (+28.00 pp), a broadly similar descriptive magnitude. It does not reverse or become approximately null. R1D remains separate and is not pooled with the 140-block inference.

## 6. Original 140-block same-target confirmation

Status: `PREREGISTERED_CONFIRMATORY`.

Primary cohort: Goal tasks 4, 6, 7, 8, 9 and LIBERO-10 tasks 0, 2, 4, 6, 7; states 0..13; 140 paired task-state blocks per condition; ten task-specific 100k ACT checkpoints.

Primary diagonal sign convention: `S(A0G20)-S(A20G0)`, where `A0G20` is fresh arm plus a 20-step-old gripper prediction for the same current physical target time, and `A20G0` is a 20-step-old arm prediction plus fresh gripper for that target.

Canonical result: 83/140 (59.29%) versus 38/140 (27.14%), `+32.14 pp`; discordance 48:3; exact two-sided McNemar `p=1.96749e-11`; paired 95% CI `[+23.6,+40.7] pp`; task-cluster 95% CI `[+21.4,+44.3] pp`. d=20 equals 1.00 s at 20 Hz and was frozen before R1A existed; it was not selected from the later sweep.

The two cohorts must not be conflated: +32.14 pp is the confirmatory 140-block headline, whereas +54.76 pp is the maximum observed separation on the later exposed-development 126-block R1A grid.

## 7. Track A: locked execution result

Status: frozen preregistered Track-A execution evidence; no recomputation.

The matching quantity is `policy queries per executed environment step`, not total query count and not identical compute.

- `H4 -> ARM4_GRIP32`: 314/450 to 335/450, `+4.667 pp`; paired 95% CI `[+2.00,+7.56]`, task-cluster 95% CI `[+0.67,+9.11]`; policy-query rate approximately 0.251 in both conditions.
- `H2 -> ARM2_GRIP16`: 295/450 to 321/450, `+5.778 pp`; paired 95% CI `[+3.56,+8.22]`, task-cluster 95% CI `[+2.67,+9.56]`; policy-query rate approximately 0.501 in both conditions.
- Coherent `H16` remains the strongest overall frozen operating point: 357/450 (79.33%). It exceeds `ARM4_GRIP32` by 4.889 pp and `ARM2_GRIP16` by 8.000 pp.

| Contrast | LIBERO-10 | Goal | Spatial |
|---|---:|---:|---:|
| `ARM4_GRIP32-H4` | +13.333 pp | +0.000 pp | +0.667 pp |
| `ARM2_GRIP16-H2` | +14.000 pp | +2.000 pp | +1.333 pp |
| `H16` absolute success | 54.7% | 91.3% | 92.0% |

Suite identity, baseline difficulty/ceiling, task semantics, and component-resolved gain covary. The source of the concentration is not identifiable, and no single moderator is assigned causally.

## 8. TE_DENSE: locked scoped negative result

Status: canonical Track-A condition with resolved runtime provenance; no recomputation.

- Runtime: site-packages LeRobot 0.4.4.
- Upstream class: canonical `ACTTemporalEnsembler`.
- Coefficient: 0.01; chunk length: 100.
- Under the upstream oldest-to-newest ordering, a positive coefficient intentionally gives greater weight to older predictions; an independent runtime canary confirmed this direction.
- Empirical weighted mean age: 44.99 steps = 2.249 s.
- Weighted p50: 43 steps = 2.15 s.
- Weighted p95: 94 steps = 4.70 s.
- Normalized weight older than 2.0 s: 52.27%.
- `abs(g)<0.50`: 24.41% of executed steps.
- Gripper sign/state-switch rate: 1.02%.

Allowed scientific scope: Under the canonical frozen LeRobot v0.4.4 coefficient and chunk length in this ACT/LIBERO evaluation, dense temporal aggregation places substantial weight on old predictions and produces substantially more near-boundary gripper commands. This is not an implementation bug, is not chatter, and does not show that near-boundary commands cause the full success loss.

## 9. Final mechanism accounting

| Diagnostic or interpretation | Exact evidence | Classification |
|---|---|---|
| B1 same-target dispersion measurement | ACT: rotation 0.1484, translation 0.1364, gripper 0.0790; cross-policy normalized ordering also places gripper below arm | `SUPPORTED` as descriptive measurement |
| Original ACT-specific gripper-localization story | ACT gripper/arm normalized-dispersion ratio `R_ACT=0.5396`, episode-cluster CI `[0.3975,0.7034]`, entirely below 1; `ACT_LOCALIZATION_PASS=no` | `CONTRADICTED` |
| Cross-policy ACT-specific localization difference | `R_ACT-R_SMOLVLA=0.1085`, CI `[-0.0422,0.2815]` crosses zero | `UNSUPPORTED` |
| B1 dispersion as translation-versus-rotation behavioral ordering predictor | B1: rotation > translation; R1B behavioral harm: translation > rotation, with both canonical primary intervals below zero | `CONTRADICTED_AS_AN_ORDERING_PREDICTOR_WITHIN_ARM` |
| B2 simple “gripper stays unchanged over the relevant window” explanation | At 1.00 s, `P(no gripper transition)=0.675018` | `INSUFFICIENT_AS_A_COMPLETE_EXPLANATION` |
| Preregistered occupancy moderator | n=30, Spearman rho=0.1922, p=0.3089 | `NULL / UNSUPPORTED` |
| Command discontinuity/coherence | Usable between-condition comparisons are confounded by state/trajectory composition and may depend on prediction offset | `NON_IDENTIFYING_POST_HOC_CHARACTERIZATION` |
| B3 forecastability as a discriminator | Complete curves are descriptively non-corresponding with R1A/R1B behavior; no threshold, correlation, or ordering rule was frozen | `NO_FROZEN_CRITERION` |
| Positive explanatory/causal mechanism | No tested diagnostic identifies one | `UNRESOLVED` |

The observed temporal sensitivity remains mechanistically unresolved by the diagnostics tested here.

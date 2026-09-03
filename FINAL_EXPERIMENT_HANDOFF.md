# Final experiment handoff

Status: `FINAL_SCIENTIFIC_CLAIM_FREEZE`

Branch: `exp/icra27-crosssuite-query-allocation`

Pre-unblinding technical disposition: `FROZEN_BEFORE_REVIEWER_SUPPLEMENT_OUTCOME_INSPECTION`.

No manuscript, LaTeX, `CLAIMS.md`, or paper-facing artwork was changed.

## 1. Runtime/source provenance

### Track A

Track A, including TE_DENSE, ran with `/home/wjq/workspace/venvs/libero_act/bin/python` and installed pip/site-packages LeRobot 0.4.4 at `/home/wjq/workspace/venvs/libero_act/lib/python3.12/site-packages/lerobot`. There was no LeRobot `PYTHONPATH`, editable install, or checkout shadow. `ACTTemporalEnsembler`, ACT configuration/loading, the policy factory, and LIBERO construction all resolved to that 0.4.4 package.

### R1A--R1C

R1A, R1B, and R1C used the same interpreter and installed LeRobot 0.4.4 paths. They imported the full 0.4.4 ACT/config/factory path, not `/home/wjq/workspace/upstreams/lerobot/src`. The launcher set no `PYTHONPATH`; the unrelated editable `verl-vla` path did not contain or shadow `lerobot`.

### R1D

The failed original R1D initialization selected `/home/wjq/workspace/upstreams/lerobot/src` at clean commit `f66e5128ecb2456e8c54a63d15404fa59c16aebc`. Its import chain was `run_queue.Runtime -> lerobot.policies.factory -> lerobot.policies.__init__ -> eo1.configuration_eo1 -> Qwen2_5_VLTextConfig`; Transformers 4.51.3 lacks that export. EO1/Qwen is unrelated to the frozen ACT evaluator.

The neutral repair preloaded only the installed LeRobot 0.4.4 package root, then invoked the unchanged frozen queue. Every ACT/LIBERO submodule consequently resolved to the same 0.4.4 files used by R1A--R1C. The source comparison found no ACT inference, chunk-indexing, or temporal-ensembler change. The checkout explicitly forwards `fps=20`; 0.4.4 omits it and resolves to the same LIBERO/robosuite default 20 Hz. No relevant package, checkout, environment setting, or source changed in the 0.307 s between R1C completion and the original R1D launch.

The corrective technical record is `experiments/icra27_reviewer_supplement/RUNTIME_SOURCE_PROVENANCE_AUDIT_20260903.md`; it preserves the failed launch's historical source identity and records the completed run's actual source separately.

### Temporal-contract consequence

The conclusions remain valid: ACT policy index = 0.05 s; R1A--R1D evaluator step = 0.05 s; `d=20` = 1.00 s; and `q+k=t` is physically same-target.

## 2. R1A fixed-source temporal sensitivity

> Report successes/rates, `Fresh-A_d_G0`, `Fresh-A0_G_d`, and `A0_G_d-A_d_G0`, with discordances, exact McNemar, paired and task-cluster bootstrap intervals, per-task effects, LOTO, queries, query rate, steps, and wall time for every `d`. No threshold or best `d` is selected.

Status: `EXPOSED_DEVELOPMENT_CHARACTERIZATION`. Fresh, A20G0, and A0G20 are audited historical reuse anchors; the other fixed-age conditions are new supplement cells.

| d | seconds | `S(A_d G0)` | `S(A0 G_d)` | `S(A0 G_d)-S(A_d G0)` (pp) |
|---:|---:|---:|---:|---:|
| 2 | 0.10 | 49/126 (38.89%) | 76/126 (60.32%) | +21.43 |
| 4 | 0.20 | 49/126 (38.89%) | 82/126 (65.08%) | +26.19 |
| 8 | 0.40 | 36/126 (28.57%) | 88/126 (69.84%) | +41.27 |
| 12 | 0.60 | 21/126 (16.67%) | 88/126 (69.84%) | +53.17 |
| 16 | 0.80 | 23/126 (18.25%) | 90/126 (71.43%) | +53.17 |
| 20 | 1.00 | 12/126 (9.52%) | 81/126 (64.29%) | +54.76 |
| 32 | 1.60 | 32/126 (25.40%) | 79/126 (62.70%) | +37.30 |

Common Fresh/d=0 anchor: 56/126 (44.44%).

### Fresh-relative and between-branch inference

| Contrast | Successes | Delta (pp) | Discordance | exact McNemar p | paired 95% CI (pp) | task-cluster 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| `A0_G0-A2_G0` | 56 vs 49 / 126 | +5.56 | 13:6 | 0.167068 | [-0.79, +12.70] | [-1.59, +12.70] |
| `A0_G0-A0_G2` | 56 vs 76 / 126 | -15.87 | 1:21 | 1.09673e-05 | [-23.02, -9.52] | [-21.43, -10.32] |
| `A0_G2-A2_G0` | 76 vs 49 / 126 | +21.43 | 29:2 | 4.62867e-07 | [+13.49, +29.37] | [+11.11, +30.95] |
| `A0_G0-A4_G0` | 56 vs 49 / 126 | +5.56 | 20:13 | 0.296206 | [-3.17, +14.29] | [-3.97, +15.87] |
| `A0_G0-A0_G4` | 56 vs 82 / 126 | -20.63 | 0:26 | 2.98023e-08 | [-27.78, -13.49] | [-30.95, -11.11] |
| `A0_G4-A4_G0` | 82 vs 49 / 126 | +26.19 | 40:7 | 1.07091e-06 | [+16.67, +35.71] | [+13.49, +38.10] |
| `A0_G0-A8_G0` | 56 vs 36 / 126 | +15.87 | 27:7 | 0.000821395 | [+7.14, +24.60] | [+8.73, +23.81] |
| `A0_G0-A0_G8` | 56 vs 88 / 126 | -25.40 | 1:33 | 4.07454e-09 | [-33.33, -17.46] | [-37.30, -15.87] |
| `A0_G8-A8_G0` | 88 vs 36 / 126 | +41.27 | 53:1 | 6.10623e-15 | [+32.54, +50.00] | [+32.54, +50.79] |
| `A0_G0-A12_G0` | 56 vs 21 / 126 | +27.78 | 38:3 | 1.04792e-08 | [+19.05, +36.51] | [+19.05, +36.51] |
| `A0_G0-A0_G12` | 56 vs 88 / 126 | -25.40 | 2:34 | 1.94123e-08 | [-34.13, -17.46] | [-37.30, -15.08] |
| `A0_G12-A12_G0` | 88 vs 21 / 126 | +53.17 | 68:1 | 2.37169e-19 | [+44.44, +61.90] | [+45.24, +61.90] |
| `A0_G0-A16_G0` | 56 vs 23 / 126 | +26.19 | 36:3 | 3.60887e-08 | [+17.46, +34.92] | [+15.87, +36.51] |
| `A0_G0-A0_G16` | 56 vs 90 / 126 | -26.98 | 1:35 | 1.07684e-09 | [-34.92, -19.05] | [-38.89, -16.67] |
| `A0_G16-A16_G0` | 90 vs 23 / 126 | +53.17 | 67:0 | 1.35525e-20 | [+44.44, +61.90] | [+42.86, +62.70] |
| `A0_G0-A20_G0` | 56 vs 12 / 126 | +34.92 | 45:1 | 1.33582e-12 | [+26.19, +43.65] | [+25.40, +45.24] |
| `A0_G0-A0_G20` | 56 vs 81 / 126 | -19.84 | 1:26 | 4.17233e-07 | [-27.78, -12.70] | [-30.95, -9.52] |
| `A0_G20-A20_G0` | 81 vs 12 / 126 | +54.76 | 70:1 | 6.09864e-20 | [+46.03, +63.49] | [+44.44, +65.08] |
| `A0_G0-A32_G0` | 56 vs 32 / 126 | +19.05 | 33:9 | 0.000271539 | [+9.52, +28.57] | [+8.73, +28.57] |
| `A0_G0-A0_G32` | 56 vs 79 / 126 | -18.25 | 1:24 | 1.54972e-06 | [-25.40, -11.11] | [-28.57, -8.73] |
| `A0_G32-A32_G0` | 79 vs 32 / 126 | +37.30 | 49:2 | 1.17861e-12 | [+28.57, +46.03] | [+26.19, +46.03] |

### R1A per-task Fresh minus arm-stale effects (pp)

| Task | `A0_G0-A2_G0` | `A0_G0-A4_G0` | `A0_G0-A8_G0` | `A0_G0-A12_G0` | `A0_G0-A16_G0` | `A0_G0-A20_G0` | `A0_G0-A32_G0` |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_object:task1 | +7.14 | +7.14 | +7.14 | +7.14 | +7.14 | +14.29 | +7.14 |
| libero_object:task2 | +14.29 | -7.14 | +21.43 | +42.86 | +42.86 | +64.29 | +21.43 |
| libero_object:task3 | -7.14 | +7.14 | +7.14 | +28.57 | +35.71 | +35.71 | +21.43 |
| libero_object:task4 | +0.00 | -14.29 | +0.00 | +21.43 | +21.43 | +28.57 | +14.29 |
| libero_object:task5 | +14.29 | +28.57 | +35.71 | +42.86 | +50.00 | +50.00 | +42.86 |
| libero_object:task6 | +14.29 | +14.29 | +28.57 | +35.71 | +28.57 | +42.86 | +21.43 |
| libero_object:task7 | -7.14 | -7.14 | +7.14 | +14.29 | +14.29 | +14.29 | -14.29 |
| libero_object:task8 | +21.43 | +28.57 | +28.57 | +42.86 | +35.71 | +35.71 | +21.43 |
| libero_object:task9 | -7.14 | -7.14 | +7.14 | +14.29 | +0.00 | +28.57 | +35.71 |

### R1A per-task Fresh minus gripper-stale effects (pp)

| Task | `A0_G0-A0_G2` | `A0_G0-A0_G4` | `A0_G0-A0_G8` | `A0_G0-A0_G12` | `A0_G0-A0_G16` | `A0_G0-A0_G20` | `A0_G0-A0_G32` |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_object:task1 | -28.57 | -50.00 | -64.29 | -64.29 | -64.29 | -50.00 | -42.86 |
| libero_object:task2 | -28.57 | -28.57 | -28.57 | -28.57 | -21.43 | -21.43 | -14.29 |
| libero_object:task3 | -14.29 | -35.71 | -21.43 | -14.29 | -28.57 | -21.43 | -28.57 |
| libero_object:task4 | -21.43 | -28.57 | -42.86 | -42.86 | -42.86 | -42.86 | -42.86 |
| libero_object:task5 | -14.29 | -7.14 | -14.29 | -7.14 | -7.14 | +7.14 | +0.00 |
| libero_object:task6 | -7.14 | -7.14 | -7.14 | -7.14 | -7.14 | -7.14 | -7.14 |
| libero_object:task7 | -14.29 | -14.29 | -21.43 | -28.57 | -28.57 | -14.29 | -14.29 |
| libero_object:task8 | -14.29 | -14.29 | -14.29 | -14.29 | -21.43 | -14.29 | -14.29 |
| libero_object:task9 | +0.00 | +0.00 | -14.29 | -21.43 | -21.43 | -14.29 | +0.00 |

### R1A per-task gripper-stale minus arm-stale asymmetry (pp)

| Task | `A0_G2-A2_G0` | `A0_G4-A4_G0` | `A0_G8-A8_G0` | `A0_G12-A12_G0` | `A0_G16-A16_G0` | `A0_G20-A20_G0` | `A0_G32-A32_G0` |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_object:task1 | +35.71 | +57.14 | +71.43 | +71.43 | +71.43 | +64.29 | +50.00 |
| libero_object:task2 | +42.86 | +21.43 | +50.00 | +71.43 | +64.29 | +85.71 | +35.71 |
| libero_object:task3 | +7.14 | +42.86 | +28.57 | +42.86 | +64.29 | +57.14 | +50.00 |
| libero_object:task4 | +21.43 | +14.29 | +42.86 | +64.29 | +64.29 | +71.43 | +57.14 |
| libero_object:task5 | +28.57 | +35.71 | +50.00 | +50.00 | +57.14 | +42.86 | +42.86 |
| libero_object:task6 | +21.43 | +21.43 | +35.71 | +42.86 | +35.71 | +50.00 | +28.57 |
| libero_object:task7 | +7.14 | +7.14 | +28.57 | +42.86 | +42.86 | +28.57 | +0.00 |
| libero_object:task8 | +35.71 | +42.86 | +42.86 | +57.14 | +57.14 | +50.00 | +35.71 |
| libero_object:task9 | -7.14 | -7.14 | +21.43 | +35.71 | +21.43 | +42.86 | +35.71 |

The arm-success curve is not monotone: it falls through d=12, rises at d=16, reaches its minimum at d=20, and rises again at d=32. The gripper-success curve is also not monotone: it rises through d=16 and then declines at d=20 and d=32. The branch ordering is preserved at every frozen d, with no reversal: `S(A0 G_d) > S(A_d G0)` throughout. The largest divergence is at d=20 (1.00 s), +54.76 pp. These are complete-grid characterizations, not lag selection.

## 3. R1B translation versus rotation

> On the same 126 blocks at `d=20`, run `T20_R0_G0` (translation stale; rotation and gripper Fresh) and `T0_R20_G0` (rotation stale; translation and gripper Fresh).

> B1 found the same-target normalized dispersion ordering `rotation > translation > gripper` for both ACT and SmolVLA. If same-target source disagreement predicts behavioral temporal sensitivity, then at d=20 stale rotation should be at least as damaging as stale translation: `success(T0_R20_G0) <= success(T20_R0_G0)`.

Translation-stale: 11/126 (8.73%); rotation-stale: 53/126 (42.06%); Fresh: 56/126 (44.44%).

| Contrast | Successes | Delta (pp) | Discordance | exact McNemar p | paired 95% CI (pp) | task-cluster 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| `T20_R0_G0-T0_R20_G0` | 11 vs 53 / 126 | -33.33 | 3:45 | 1.31259e-10 | [-42.06, -24.60] | [-44.44, -22.22] |
| `T20_R0_G0-A0_G0` | 11 vs 56 / 126 | -35.71 | 3:48 | 1.96749e-11 | [-45.24, -26.19] | [-46.03, -25.40] |
| `T0_R20_G0-A0_G0` | 53 vs 56 / 126 | -2.38 | 6:9 | 0.607239 | [-8.73, +3.97] | [-6.35, +1.59] |

### R1B per-task effects (pp)

| Task | `T20_R0_G0-T0_R20_G0` | `T20_R0_G0-A0_G0` | `T0_R20_G0-A0_G0` |
|---|---:|---:|---:|
| libero_object:task1 | -7.14 | -14.29 | -7.14 |
| libero_object:task2 | -57.14 | -57.14 | +0.00 |
| libero_object:task3 | -28.57 | -28.57 | +0.00 |
| libero_object:task4 | -28.57 | -21.43 | +7.14 |
| libero_object:task5 | -50.00 | -50.00 | +0.00 |
| libero_object:task6 | -35.71 | -42.86 | -7.14 |
| libero_object:task7 | -14.29 | -14.29 | +0.00 |
| libero_object:task8 | -21.43 | -35.71 | -14.29 |
| libero_object:task9 | -57.14 | -57.14 | +0.00 |

The prospective B1-derived prediction is `MECHANISM_DISSOCIATION`: the direct contrast is -33.33 pp, not non-negative. Translation staleness is strongly damaging; rotation staleness is close to Fresh. Because translation and rotation behave differently on every task in the primary contrast, R1B supports broader within-arm component dependence, while rejecting B1 dispersion ordering as its explanation. The paper need not narrow to only arm-versus-gripper, but must keep this evidence ACT- and Object-development-specific.

## 4. R1C dense-query matched H16 factorial

> All four conditions make a whole-policy query every step: `C00` Fresh/Fresh; `C10` scheduled-H16 arm/Fresh gripper; `C01` Fresh arm/scheduled-H16 gripper; `C11` scheduled-H16 arm/gripper.

> Before bulk rollout a frozen deterministic canary must show that unused dense forward passes leave C11 executed actions, simulator trajectory, terminal result, and length identical to sparse HARD-H16. Canary failure stops R1C.

The exact frozen identity gate passed: dense C11 and sparse HARD-H16 had exactly identical executed actions, simulator trajectory from the same initial state, terminal success, completion step, and episode length. Extra discarded policy queries therefore did not change the canary trajectory.

| Condition | Reader-facing semantics | Success/N |
|---|---|---:|
| C00 | dense-query Fresh arm + Fresh gripper | 77/140 (55.00%) |
| C10 | dense-query scheduled-H16 arm + Fresh gripper | 76/140 (54.29%) |
| C01 | dense-query Fresh arm + scheduled-H16 gripper | 81/140 (57.86%) |
| C11 | dense-query scheduled-H16 arm + scheduled-H16 gripper | 93/140 (66.43%) |

| Contrast | Successes | Delta (pp) | Discordance | exact McNemar p | paired 95% CI (pp) | task-cluster 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| `C10-C00` | 76 vs 77 / 140 | -0.71 | 18:19 | 1 | [-9.29, +7.86] | [-12.14, +10.71] |
| `C01-C00` | 81 vs 77 / 140 | +2.86 | 10:6 | 0.454498 | [-2.86, +8.57] | [-4.29, +9.29] |
| `C11-C10` | 93 vs 76 / 140 | +12.14 | 20:3 | 0.000488281 | [+5.71, +18.57] | [+3.57, +21.43] |
| `C11-C01` | 93 vs 81 / 140 | +8.57 | 26:14 | 0.0806905 | [+0.00, +17.14] | [-2.14, +21.43] |
| `C11-C00` | 93 vs 77 / 140 | +11.43 | 28:12 | 0.016589 | [+2.86, +20.00] | [-0.71, +23.57] |
| `C10-C01` | 76 vs 81 / 140 | -3.57 | 12:17 | 0.458258 | [-11.43, +4.29] | [-14.29, +6.43] |

### R1C per-task frozen effects (pp)

| Task | `C10-C00` | `C01-C00` | `C11-C10` | `C11-C01` | `C11-C00` | `C10-C01` |
|---|---:|---:|---:|---:|---:|---:|
| libero_10:task0 | +0.00 | +14.29 | +14.29 | +0.00 | +14.29 | -14.29 |
| libero_10:task2 | +35.71 | +21.43 | +0.00 | +14.29 | +35.71 | +14.29 |
| libero_10:task4 | +7.14 | -14.29 | +35.71 | +57.14 | +42.86 | +21.43 |
| libero_10:task6 | -21.43 | -14.29 | +14.29 | +7.14 | -7.14 | -7.14 |
| libero_10:task7 | -7.14 | +14.29 | +35.71 | +14.29 | +28.57 | -21.43 |
| libero_goal:task4 | -7.14 | +0.00 | -7.14 | -14.29 | -14.29 | -7.14 |
| libero_goal:task6 | -35.71 | +0.00 | +21.43 | -14.29 | -14.29 | -35.71 |
| libero_goal:task7 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| libero_goal:task8 | +7.14 | +0.00 | +0.00 | +7.14 | +7.14 | +7.14 |
| libero_goal:task9 | +14.29 | +7.14 | +7.14 | +14.29 | +21.43 | +7.14 |

Per-suite conditional effects are retained in `canonical_report.json`. In particular, C11-C10 is +20.00 pp on LIBERO-10 and +4.29 pp on Goal; C11-C01 is +18.57 pp on LIBERO-10 and -1.43 pp on Goal. The frozen risk-difference interaction `C11-C10-C01+C00` is +9.29 pp.

Conclusion: the H16 advantage survives whole-policy query-rate matching. Because C11 exactly matches sparse H16 on the preregistered identity canary, extra discarded forward passes are not an executed-behavior confound under that frozen rule. This does not authorize component percentage attribution.

## 5. B3 forecastability

> If and only if the temporal-contract audit establishes an unambiguous mapping, compare the complete ACT B3 future-action forecast curves with the complete R1A behavioral sensitivity curves on a seconds-first axis. The provenance-only timebase audit resolved the 20 Hz R1A ages `d={2,4,8,12,16,20,32}` to the exact 10 Hz B3 offsets `k={1,2,4,6,8,10,16}` at matched times `{0.1,0.2,0.4,0.6,0.8,1.0,1.6}` seconds, so no interpolation is used. The table must include B3 arm, translation, rotation, gripper-value, and gripper-sign forecast quantities and R1A `Fresh-A_d_G0` and `Fresh-A0_G_d`. This is cross-cohort descriptive characterization, not a significance-gated hypothesis. If time alignment is ambiguous, report the comparison as not identifiable and keep native step axes separate. The full B3 offset grid `0..32` and every frozen R1A age remain reported in their own canonical analyses; this matched-time table is not a lag-selection rule.

The quoted 10 Hz offset mapping is retained verbatim as historical governance. The later corrective temporal audit, which controls the present handoff, established the physical B3 mapping as `k/20` seconds; therefore the exact matched correspondence is `d={2,4,8,12,16,20,32}` to `k={2,4,8,12,16,20,32}`. The characterization rule itself is unchanged.

> Episode-cluster percentile bootstrap uses 20,000 draws, 95% intervals, seed 27401 for ACT and 27402 for SmolVLA. No performance outcome, task subset, lag, or offset is selected after results. These associations do not establish that persistence or forecastability causes executor sensitivity.

Classification: `B3_NO_FROZEN_DISCRIMINATIVE_CRITERION`. No threshold, correlation, or ordering rule for mechanism support was frozen, so none is constructed after unblinding.

Metric and target: at each demonstration anchor t, compare the frozen policy's normalized predicted action at chunk offset k with the recorded demonstrated action at exact target row t+k. Report per-dimension RMSE, translation/rotation/combined-arm normalized RMSE, gripper absolute normalized error, and controller-native gripper sign-disagreement probability. Physical offset is `k/20` seconds for every k=0..32.

Cohort: training-demonstration reference, not held-out; tasks Object 3, Spatial 0, Goal 2, and LIBERO-10 3. The selection rule is `lowest ten numeric training episode IDs per panel task`. Episode IDs: libero_object:task3=[811, 812, 824, 843, 846, 849, 853, 858, 867, 871]; libero_spatial:task0=[1272, 1273, 1275, 1282, 1300, 1327, 1330, 1344, 1347, 1352]; libero_goal:task2=[385, 389, 396, 397, 404, 417, 419, 423, 427, 430]; libero_10:task3=[14, 15, 16, 31, 32, 36, 75, 89, 90, 97].

ACT uses the four frozen task-specific ACT checkpoints; SmolVLA uses local revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, whose training-data relationship remains unknown. Policies are reported in separate normalized spaces. Uncertainty is the already-generated 20,000-draw demonstration-episode cluster percentile 95% interval (ACT seed 27401; SmolVLA seed 27402).

### ACT: complete frozen B3 curve

All entries are center `[episode-cluster 95% CI]`. RMSE quantities are in that policy's frozen normalized action space.

| k | seconds | translation RMSE | rotation RMSE | arm RMSE | gripper absolute normalized error | gripper sign disagreement |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.0992 [0.0893, 0.1103] | 0.1443 [0.1322, 0.1575] | 0.1238 [0.1137, 0.1349] | 0.0959 [0.0481, 0.1479] | 0.0021 [0.0000, 0.0069] |
| 1 | 0.05 | 0.0916 [0.0849, 0.0992] | 0.1392 [0.1279, 0.1508] | 0.1178 [0.1096, 0.1267] | 0.1107 [0.0707, 0.1481] | 0.0021 [0.0000, 0.0072] |
| 2 | 0.10 | 0.0891 [0.0833, 0.0956] | 0.1271 [0.1181, 0.1352] | 0.1098 [0.1036, 0.1158] | 0.1040 [0.0648, 0.1447] | 0.0021 [0.0000, 0.0070] |
| 3 | 0.15 | 0.0909 [0.0856, 0.0964] | 0.1279 [0.1184, 0.1369] | 0.1110 [0.1044, 0.1173] | 0.1736 [0.1035, 0.2368] | 0.0105 [0.0022, 0.0205] |
| 4 | 0.20 | 0.0907 [0.0809, 0.1026] | 0.1238 [0.1157, 0.1321] | 0.1085 [0.1009, 0.1165] | 0.0808 [0.0542, 0.1036] | 0.0000 [0.0000, 0.0000] |
| 5 | 0.25 | 0.0956 [0.0838, 0.1093] | 0.1251 [0.1155, 0.1347] | 0.1114 [0.1021, 0.1205] | 0.0871 [0.0556, 0.1131] | 0.0021 [0.0000, 0.0066] |
| 6 | 0.30 | 0.0966 [0.0843, 0.1130] | 0.1205 [0.1109, 0.1300] | 0.1092 [0.1005, 0.1183] | 0.1001 [0.0687, 0.1268] | 0.0000 [0.0000, 0.0000] |
| 7 | 0.35 | 0.0992 [0.0840, 0.1182] | 0.1290 [0.1176, 0.1397] | 0.1151 [0.1054, 0.1248] | 0.1139 [0.0727, 0.1509] | 0.0021 [0.0000, 0.0066] |
| 8 | 0.40 | 0.0949 [0.0798, 0.1162] | 0.1219 [0.1122, 0.1313] | 0.1092 [0.1000, 0.1192] | 0.1042 [0.0588, 0.1417] | 0.0042 [0.0000, 0.0106] |
| 9 | 0.45 | 0.0885 [0.0791, 0.1007] | 0.1215 [0.1124, 0.1298] | 0.1063 [0.0988, 0.1136] | 0.0671 [0.0366, 0.1027] | 0.0021 [0.0000, 0.0071] |
| 10 | 0.50 | 0.0880 [0.0808, 0.0961] | 0.1216 [0.1099, 0.1342] | 0.1062 [0.0975, 0.1154] | 0.0622 [0.0389, 0.0852] | 0.0000 [0.0000, 0.0000] |
| 11 | 0.55 | 0.0817 [0.0767, 0.0872] | 0.1215 [0.1115, 0.1317] | 0.1035 [0.0964, 0.1111] | 0.0763 [0.0537, 0.0987] | 0.0000 [0.0000, 0.0000] |
| 12 | 0.60 | 0.0821 [0.0779, 0.0865] | 0.1209 [0.1111, 0.1301] | 0.1034 [0.0970, 0.1094] | 0.0765 [0.0526, 0.0991] | 0.0000 [0.0000, 0.0000] |
| 13 | 0.65 | 0.0868 [0.0809, 0.0930] | 0.1203 [0.1099, 0.1298] | 0.1049 [0.0974, 0.1118] | 0.0858 [0.0574, 0.1117] | 0.0000 [0.0000, 0.0000] |
| 14 | 0.70 | 0.0870 [0.0792, 0.0952] | 0.1162 [0.1070, 0.1245] | 0.1026 [0.0950, 0.1100] | 0.1017 [0.0545, 0.1388] | 0.0042 [0.0000, 0.0106] |
| 15 | 0.75 | 0.0854 [0.0782, 0.0932] | 0.1182 [0.1081, 0.1282] | 0.1031 [0.0948, 0.1112] | 0.1150 [0.0652, 0.1539] | 0.0063 [0.0000, 0.0134] |
| 16 | 0.80 | 0.0863 [0.0793, 0.0941] | 0.1153 [0.1061, 0.1247] | 0.1018 [0.0949, 0.1088] | 0.1101 [0.0588, 0.1573] | 0.0021 [0.0000, 0.0066] |
| 17 | 0.85 | 0.0905 [0.0807, 0.1015] | 0.1257 [0.1126, 0.1397] | 0.1095 [0.0999, 0.1198] | 0.1216 [0.0673, 0.1757] | 0.0021 [0.0000, 0.0067] |
| 18 | 0.90 | 0.0865 [0.0781, 0.0961] | 0.1206 [0.1108, 0.1301] | 0.1049 [0.0972, 0.1125] | 0.1014 [0.0359, 0.1463] | 0.0063 [0.0000, 0.0140] |
| 19 | 0.95 | 0.0843 [0.0779, 0.0911] | 0.1209 [0.1095, 0.1330] | 0.1042 [0.0963, 0.1126] | 0.0685 [0.0403, 0.0935] | 0.0000 [0.0000, 0.0000] |
| 20 | 1.00 | 0.0892 [0.0810, 0.0981] | 0.1230 [0.1094, 0.1361] | 0.1074 [0.0972, 0.1177] | 0.1127 [0.0427, 0.1653] | 0.0042 [0.0000, 0.0102] |
| 21 | 1.05 | 0.0850 [0.0786, 0.0920] | 0.1244 [0.1123, 0.1366] | 0.1066 [0.0978, 0.1158] | 0.1233 [0.0678, 0.1683] | 0.0042 [0.0000, 0.0104] |
| 22 | 1.10 | 0.0867 [0.0807, 0.0932] | 0.1243 [0.1135, 0.1345] | 0.1072 [0.0995, 0.1148] | 0.1078 [0.0735, 0.1386] | 0.0000 [0.0000, 0.0000] |
| 23 | 1.15 | 0.0886 [0.0817, 0.0959] | 0.1178 [0.1088, 0.1263] | 0.1042 [0.0974, 0.1109] | 0.1166 [0.0782, 0.1518] | 0.0042 [0.0000, 0.0112] |
| 24 | 1.20 | 0.0903 [0.0818, 0.0991] | 0.1171 [0.1068, 0.1266] | 0.1045 [0.0962, 0.1127] | 0.1769 [0.0978, 0.2362] | 0.0127 [0.0022, 0.0248] |
| 25 | 1.25 | 0.0862 [0.0781, 0.0944] | 0.1192 [0.1084, 0.1305] | 0.1040 [0.0954, 0.1127] | 0.1703 [0.0872, 0.2307] | 0.0063 [0.0000, 0.0131] |
| 26 | 1.30 | 0.0847 [0.0762, 0.0928] | 0.1231 [0.1090, 0.1372] | 0.1057 [0.0958, 0.1151] | 0.1375 [0.0867, 0.1834] | 0.0042 [0.0000, 0.0106] |
| 27 | 1.35 | 0.0911 [0.0787, 0.1042] | 0.1378 [0.1217, 0.1535] | 0.1168 [0.1044, 0.1291] | 0.1763 [0.1168, 0.2317] | 0.0063 [0.0000, 0.0140] |
| 28 | 1.40 | 0.0888 [0.0785, 0.0993] | 0.1255 [0.1150, 0.1355] | 0.1087 [0.0996, 0.1171] | 0.1212 [0.0748, 0.1619] | 0.0042 [0.0000, 0.0106] |
| 29 | 1.45 | 0.0905 [0.0818, 0.0991] | 0.1284 [0.1174, 0.1394] | 0.1111 [0.1023, 0.1194] | 0.0817 [0.0440, 0.1165] | 0.0021 [0.0000, 0.0066] |
| 30 | 1.50 | 0.0866 [0.0790, 0.0946] | 0.1274 [0.1110, 0.1441] | 0.1089 [0.0975, 0.1204] | 0.0924 [0.0485, 0.1357] | 0.0021 [0.0000, 0.0066] |
| 31 | 1.55 | 0.0834 [0.0759, 0.0914] | 0.1382 [0.1182, 0.1603] | 0.1142 [0.1003, 0.1290] | 0.1199 [0.0724, 0.1636] | 0.0063 [0.0000, 0.0141] |
| 32 | 1.60 | 0.0888 [0.0782, 0.1013] | 0.1502 [0.1262, 0.1739] | 0.1234 [0.1055, 0.1417] | 0.1218 [0.0811, 0.1585] | 0.0042 [0.0000, 0.0106] |

### SmolVLA: complete frozen B3 curve

All entries are center `[episode-cluster 95% CI]`. RMSE quantities are in that policy's frozen normalized action space.

| k | seconds | translation RMSE | rotation RMSE | arm RMSE | gripper absolute normalized error | gripper sign disagreement |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.5143 [0.4550, 0.5871] | 0.8354 [0.7759, 0.8936] | 0.6937 [0.6427, 0.7476] | 0.4089 [0.3149, 0.5005] | 0.0422 [0.0251, 0.0629] |
| 1 | 0.05 | 0.4879 [0.4254, 0.5642] | 0.7929 [0.7360, 0.8455] | 0.6583 [0.6082, 0.7102] | 0.4226 [0.3324, 0.5083] | 0.0443 [0.0274, 0.0640] |
| 2 | 0.10 | 0.4805 [0.4159, 0.5630] | 0.7822 [0.7236, 0.8383] | 0.6491 [0.5974, 0.7026] | 0.4505 [0.3644, 0.5257] | 0.0506 [0.0331, 0.0690] |
| 3 | 0.15 | 0.4764 [0.4166, 0.5505] | 0.7769 [0.7101, 0.8423] | 0.6444 [0.5894, 0.7004] | 0.4942 [0.4074, 0.5717] | 0.0612 [0.0416, 0.0819] |
| 4 | 0.20 | 0.4680 [0.4087, 0.5436] | 0.7706 [0.7017, 0.8393] | 0.6375 [0.5819, 0.6944] | 0.5315 [0.4586, 0.5996] | 0.0717 [0.0539, 0.0905] |
| 5 | 0.25 | 0.4693 [0.4061, 0.5476] | 0.7527 [0.6891, 0.8147] | 0.6272 [0.5745, 0.6820] | 0.5109 [0.4465, 0.5715] | 0.0654 [0.0500, 0.0819] |
| 6 | 0.30 | 0.4478 [0.3986, 0.5063] | 0.7448 [0.6846, 0.8031] | 0.6145 [0.5681, 0.6613] | 0.4739 [0.4084, 0.5388] | 0.0570 [0.0424, 0.0734] |
| 7 | 0.35 | 0.4398 [0.3940, 0.4940] | 0.7337 [0.6716, 0.7927] | 0.6048 [0.5591, 0.6506] | 0.4193 [0.3393, 0.4895] | 0.0443 [0.0292, 0.0603] |
| 8 | 0.40 | 0.4329 [0.3884, 0.4862] | 0.7237 [0.6676, 0.7785] | 0.5963 [0.5539, 0.6383] | 0.4414 [0.3494, 0.5315] | 0.0485 [0.0302, 0.0704] |
| 9 | 0.45 | 0.4333 [0.3883, 0.4861] | 0.7272 [0.6761, 0.7760] | 0.5986 [0.5579, 0.6375] | 0.4593 [0.3512, 0.5605] | 0.0527 [0.0308, 0.0786] |
| 10 | 0.50 | 0.4342 [0.3905, 0.4857] | 0.7404 [0.6881, 0.7890] | 0.6069 [0.5650, 0.6479] | 0.4829 [0.3844, 0.5781] | 0.0591 [0.0377, 0.0842] |
| 11 | 0.55 | 0.4359 [0.3909, 0.4896] | 0.7625 [0.7040, 0.8185] | 0.6210 [0.5741, 0.6688] | 0.4656 [0.3622, 0.5600] | 0.0549 [0.0333, 0.0790] |
| 12 | 0.60 | 0.4451 [0.3891, 0.5116] | 0.7862 [0.7229, 0.8471] | 0.6389 [0.5862, 0.6933] | 0.4304 [0.3338, 0.5167] | 0.0464 [0.0280, 0.0668] |
| 13 | 0.65 | 0.4513 [0.3939, 0.5202] | 0.7822 [0.7159, 0.8471] | 0.6386 [0.5841, 0.6944] | 0.5111 [0.4223, 0.5934] | 0.0654 [0.0446, 0.0882] |
| 14 | 0.70 | 0.4493 [0.4006, 0.5042] | 0.7891 [0.7210, 0.8554] | 0.6421 [0.5886, 0.6956] | 0.5001 [0.4000, 0.5952] | 0.0633 [0.0408, 0.0894] |
| 15 | 0.75 | 0.4519 [0.4059, 0.5004] | 0.7889 [0.7188, 0.8606] | 0.6429 [0.5890, 0.6963] | 0.4953 [0.3847, 0.5991] | 0.0612 [0.0367, 0.0897] |
| 16 | 0.80 | 0.4882 [0.4313, 0.5503] | 0.7983 [0.7250, 0.8708] | 0.6616 [0.6036, 0.7204] | 0.4611 [0.3771, 0.5401] | 0.0527 [0.0352, 0.0725] |
| 17 | 0.85 | 0.5100 [0.4424, 0.5875] | 0.7907 [0.7092, 0.8725] | 0.6653 [0.5976, 0.7364] | 0.5032 [0.4299, 0.5735] | 0.0633 [0.0461, 0.0822] |
| 18 | 0.90 | 0.5235 [0.4497, 0.6083] | 0.7901 [0.7081, 0.8728] | 0.6702 [0.6004, 0.7444] | 0.5124 [0.4254, 0.5951] | 0.0654 [0.0448, 0.0885] |
| 19 | 0.95 | 0.5329 [0.4480, 0.6342] | 0.8025 [0.7252, 0.8802] | 0.6812 [0.6090, 0.7590] | 0.5445 [0.4459, 0.6419] | 0.0738 [0.0495, 0.1026] |
| 20 | 1.00 | 0.5459 [0.4591, 0.6466] | 0.8141 [0.7413, 0.8879] | 0.6931 [0.6234, 0.7690] | 0.5671 [0.4569, 0.6791] | 0.0802 [0.0518, 0.1152] |
| 21 | 1.05 | 0.5605 [0.4683, 0.6680] | 0.8244 [0.7529, 0.8996] | 0.7049 [0.6339, 0.7852] | 0.5692 [0.4549, 0.6798] | 0.0823 [0.0526, 0.1174] |
| 22 | 1.10 | 0.5684 [0.4842, 0.6646] | 0.8568 [0.7862, 0.9304] | 0.7270 [0.6593, 0.8017] | 0.6212 [0.5176, 0.7244] | 0.0970 [0.0676, 0.1320] |
| 23 | 1.15 | 0.5629 [0.4803, 0.6565] | 0.8632 [0.7852, 0.9430] | 0.7287 [0.6572, 0.8053] | 0.6094 [0.4960, 0.7191] | 0.0928 [0.0615, 0.1293] |
| 24 | 1.20 | 0.5722 [0.4850, 0.6707] | 0.8712 [0.7882, 0.9555] | 0.7370 [0.6631, 0.8165] | 0.5937 [0.5007, 0.6862] | 0.0886 [0.0633, 0.1182] |
| 25 | 1.25 | 0.5678 [0.4892, 0.6579] | 0.8642 [0.7818, 0.9476] | 0.7312 [0.6594, 0.8083] | 0.5576 [0.4681, 0.6460] | 0.0781 [0.0551, 0.1047] |
| 26 | 1.30 | 0.5742 [0.4952, 0.6648] | 0.8535 [0.7702, 0.9380] | 0.7274 [0.6573, 0.8027] | 0.5850 [0.4903, 0.6764] | 0.0865 [0.0608, 0.1156] |
| 27 | 1.35 | 0.5746 [0.5023, 0.6576] | 0.8365 [0.7554, 0.9171] | 0.7176 [0.6508, 0.7890] | 0.5819 [0.4909, 0.6753] | 0.0844 [0.0601, 0.1137] |
| 28 | 1.40 | 0.5743 [0.5005, 0.6591] | 0.8376 [0.7595, 0.9164] | 0.7181 [0.6501, 0.7917] | 0.5692 [0.4697, 0.6665] | 0.0823 [0.0561, 0.1127] |
| 29 | 1.45 | 0.5855 [0.4975, 0.6880] | 0.8389 [0.7592, 0.9195] | 0.7234 [0.6480, 0.8066] | 0.6031 [0.4973, 0.7096] | 0.0907 [0.0615, 0.1256] |
| 30 | 1.50 | 0.6016 [0.5052, 0.7140] | 0.8597 [0.7733, 0.9486] | 0.7419 [0.6582, 0.8343] | 0.6006 [0.5028, 0.7016] | 0.0928 [0.0654, 0.1259] |
| 31 | 1.55 | 0.6106 [0.5212, 0.7144] | 0.8673 [0.7778, 0.9612] | 0.7500 [0.6674, 0.8425] | 0.6293 [0.5576, 0.7023] | 0.0992 [0.0778, 0.1235] |
| 32 | 1.60 | 0.6199 [0.5335, 0.7217] | 0.8787 [0.7857, 0.9766] | 0.7604 [0.6768, 0.8549] | 0.6419 [0.5530, 0.7336] | 0.1034 [0.0768, 0.1351] |

At the exact R1A-matched offsets k=d, ACT rotation forecast RMSE exceeds translation RMSE at all seven points, while behavior shows translation staleness is much more damaging than rotation staleness. ACT combined-arm error is nearly flat over most of 0..32, whereas R1A arm harm grows and then recedes non-monotonically. ACT gripper sign disagreement remains near zero even where gripper staleness improves behavioral success. Under the frozen descriptive rule, simple forecastability therefore does not provide discriminative mechanism support. B3 is retained as null/mixed mechanism evidence and the mechanism search is closed.

## 6. R1D Spatial Reverse20 completion

Status: `POST_HOC_SPATIAL_FACTORIAL_COMPLETION`. The import-only repair and canary details are in Sections 1 and 6; `REFERENCE_SEQUENCE_UNAVAILABLE` applies to the optional stronger technical comparison.

Required canaries passed: exact ACT imports; exact checkpoint/config identities; policy load; Spatial environment construction/reset; actual 20 Hz clock; expected 256x256 two-camera plus 8D-state preprocessing; frozen normalization/denormalization processors; Reverse20 Fresh-prefix/source-age semantics; physical `q+k=t`; exact manifest; zero prelaunch results, markers, and attempts.

Exactly 100 new scientific cells executed; skipped completed cells = 0; retries = 0; duplicates = 0; frozen manifest identity = exact match; terminal validator = PASS.

| Condition | Success/N |
|---|---:|
| `A0_G0` | 40/100 (40.00%) |
| `A0_G20` | 40/100 (40.00%) |
| `A20_G0` | 12/100 (12.00%) |
| `A20_G20` | 30/100 (30.00%) |

### Frozen R1D contrasts

| Contrast | Successes | Delta (pp) | Discordance | exact McNemar p | paired 95% CI (pp) | task-cluster 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| `A0_G20-A20_G0` | 40 vs 12 / 100 | +28.00 | 33:5 | 4.25596e-06 | [+17.00, +39.00] | [+13.00, +45.00] |
| `A20_G0-A0_G0` | 12 vs 40 / 100 | -28.00 | 3:31 | 7.66013e-07 | [-38.00, -18.00] | [-45.00, -13.00] |
| `A0_G20-A0_G0` | 40 vs 40 / 100 | +0.00 | 5:5 | 1 | [-6.00, +6.00] | [-5.00, +6.00] |
| `A20_G20-A0_G0` | 30 vs 40 / 100 | -10.00 | 7:17 | 0.0639147 | [-19.00, -1.00] | [-19.00, -2.00] |

### R1D per-task effects (pp)

| Task | `A0_G20-A20_G0` | `A20_G0-A0_G0` | `A0_G20-A0_G0` | `A20_G20-A0_G0` |
|---|---:|---:|---:|---:|
| libero_spatial:task0 | +30.00 | -40.00 | -10.00 | -10.00 |
| libero_spatial:task1 | +30.00 | -30.00 | +0.00 | -10.00 |
| libero_spatial:task2 | +80.00 | -80.00 | +0.00 | -40.00 |
| libero_spatial:task3 | +70.00 | -60.00 | +10.00 | +0.00 |
| libero_spatial:task4 | +10.00 | -20.00 | -10.00 | -20.00 |
| libero_spatial:task5 | +0.00 | +0.00 | +0.00 | +10.00 |
| libero_spatial:task6 | +30.00 | -40.00 | -10.00 | -20.00 |
| libero_spatial:task7 | +0.00 | +0.00 | +0.00 | +0.00 |
| libero_spatial:task8 | +30.00 | -10.00 | +20.00 | -10.00 |
| libero_spatial:task9 | +0.00 | +0.00 | +0.00 | +0.00 |

Spatial completes the same qualitative arm–gripper asymmetry: stale arm reduces success by 28.00 pp from Fresh, stale gripper changes success by 0.00 pp, and stale-gripper minus stale-arm is +28.00 pp. This separate post-hoc 100-block panel is not pooled into the original 140-block confirmation.

## 7. R2A

`R2A_NOT_RUN_FROZEN_GATE_INELIGIBLE`. Original epoch 1788354953 = 2026-09-02T21:15:53+08:00; eligibility required elapsed <=57,600 s; observed elapsed was 63,630 s. R2A was correctly not launched. This is a permanent governance outcome, not a technical failure.

## 8. TE_DENSE

Actual runtime provenance is installed LeRobot 0.4.4. TE_DENSE uses canonical upstream `ACTTemporalEnsembler`, coefficient 0.01, chunk length 100. The oldest available prediction receives the first temporal weight; a positive coefficient intentionally weights older predictions more strongly. The runtime canary confirms this actual direction and seven-dimension normalized-space aggregation.

Observed characterization: empirical weighted mean age 44.99 steps = 2.249 s; weighted p50 43 steps = 2.15 s; p95 94 steps = 4.70 s; 52.27% normalized weight is older than 2.0 s. `abs(g)<0.50` occurs on 24.41% of executed steps; gripper sign/state-switch rate is 1.02%.

Scoped interpretation: under the frozen upstream/runtime coefficient and chunk length, TE_DENSE places substantial aggregate weight on older predictions and produces substantially more near-boundary gripper commands than the other frozen executors. This is not evidence that canonical temporal ensembling is intrinsically harmful, that near-boundary commands causally explain the full loss, or that gripper chatter is the failure mode. The low switch rate is incompatible with a chatter description. No TE tuning or further analysis is authorized.

## 9. Track A

Track A's main matched-query component-resolved contrasts are ARM4_GRIP32-H4 = +4.667 pp (335/450 vs 314/450; paired 95% CI [+2.00,+7.56], task-cluster CI [+0.67,+9.11]) and ARM2_GRIP16-H2 = +5.778 pp (321/450 vs 295/450; paired CI [+3.56,+8.22], task-cluster CI [+2.67,+9.56]). Policy-query rates are matched within each comparison: approximately 0.251 and 0.501, respectively.

H16 remains the boundary: 357/450 (79.33%), exceeding ARM4_GRIP32 by 4.889 pp and ARM2_GRIP16 by 8.000 pp. The measured path `H2 -> ARM2_GRIP16 -> H16` is +5.778 pp followed by an additional +8.000 pp. It is a measured path only, not unique or path-independent component attribution. No analogous symmetric decomposition is allowed for `H4 -> ARM4_GRIP32 -> H16`, because the second edge changes both arm 4->16 and gripper 32->16.

H16 absolute suite baselines are LIBERO-10 54.7%, Goal 91.3%, and Spatial 92.0%. ARM4_GRIP32-H4 is +13.333 pp, 0.000 pp, and +0.667 pp; ARM2_GRIP16-H2 is +14.000 pp, +2.000 pp, and +1.333 pp, respectively. The largest gain occurs on the hardest suite, while Goal and Spatial operate near the H16 ceiling. Because suite identity, baseline difficulty, and task semantics covary, the source of this concentration is not identifiable.

LOSO minimal-margin disclosure: the minimum leave-one-suite-out effect is +0.333 pp for ARM4_GRIP32-H4 and +1.667 pp for ARM2_GRIP16-H2; all three LOSO estimates are positive for both. This robustness summary is not independent evidence.

## 10. Mechanism accounting

- B1 same-target source disagreement: `ACT_LOCALIZATION_KILL`; no cross-policy mechanism support. Normalized dispersion orders rotation > translation > gripper for ACT and SmolVLA, but R1B behavior orders translation staleness as far more damaging than rotation staleness. Status: descriptive source disagreement plus mechanism dissociation.

- B2 training-demonstration persistence: gripper actions are persistent, but this is training-data characterization, not held-out evidence or a causal mechanism. Under the corrected 20 Hz physical mapping, S(5), S(10), and S(20) correspond to 0.25, 0.50, and 1.00 s; their survival estimates are 0.921765, 0.840878, and 0.675018. The complete-case 30.775-step mean is biased and not a population mean.

- Prospective conditional moderators: gripper transition density versus Delta_G has rho=-0.419 (descriptive p=0.229); arm variation versus Delta_A has rho=-0.030 (p=0.934). The first is directionally compatible but uncertain; the second is null.

- Frozen Track-A gripper-activity occupancy moderator: rho=0.192, descriptive p=0.309 across all 30 tasks. Status: unsupported.

- B3 forecastability: `B3_NO_FROZEN_DISCRIMINATIVE_CRITERION`; the complete curves are descriptively non-corresponding with R1A/R1B behavior. Status: no discriminative mechanism support.

- Command discontinuity: `NON_IDENTIFYING_POST_HOC_CHARACTERIZATION`. Between-condition D1 comparisons are confounded by state/trajectory composition and may depend on prediction offset. Coherence is neither supported, falsified, nor causal.

- Supported explanations/claims: ACT execution is temporally component-dependent; matched-query gripper commitment can improve success relative to matched global short horizons; translation and rotation staleness differ strongly on the exposed Object cohort; the matched-query R1C result rules out policy-query count alone under the frozen canary contract.

- Killed or unsupported explanations: ACT-specific gripper instability/chatter localization; B1 dispersion ordering as a behavioral-sensitivity predictor; frozen occupancy moderation; simple B3 forecastability as discriminative mechanism support; command-coherence causality; canonical TE as intrinsically harmful.

- Unresolved: the causal mechanism producing component differences; the source of suite concentration; SmolVLA's physical training timebase and cross-policy generality; whether TE near-boundary commands explain any success loss; generality beyond one checkpoint/training seed per task.

## 11. Final paper claim scope

Strongest defensible claim: for the evaluated ACT policies, temporal source/execution choices have component-dependent behavioral effects. Cross-suite matched-query allocation improves over matched global short-horizon baselines, while coherent H16 remains better; exposed fixed-age characterization further separates translation, rotation, and gripper responses.

The working broad title `Component-Dependent Temporal Effects in Action-Chunked Robot Policies` may remain, because R1B directly shows within-arm translation-versus-rotation dependence rather than only an arm-versus-gripper split. The paper text must narrow the empirical claim to evaluated ACT policies and must not imply universal cross-policy confirmation.

Remove or avoid: unique arm/gripper contribution; component percentages; per-dimension additive attribution; path-independent decomposition; a symmetric H4->ARM4_GRIP32->H16 decomposition; causal suite moderators; a forecastability/coherence mechanism claim; gripper chatter; intrinsic harm from canonical temporal ensembling; broad SmolVLA replication; pooling R1D into the original 140 blocks.

Principal limitations: exposed/development status of R1A/R1B; post-hoc status of R1C/R1D; task/suite/checkpoint covariation; one checkpoint/training seed per task; no frozen discriminative mechanism criterion for B3; unresolved causal mechanism; SmolVLA timebase provenance gap.

## 12. Experiment closure

Open-ended scientific search is `CLOSED`. R1A, R1B, R1C, R1D, and B3 are complete and canonically reported; R2A is permanently gate-ineligible. No new executor, rescue method, d sweep, horizon, mechanism analysis, seed, benchmark, TE coefficient, or SmolVLA repair is recommended or authorized. Only a genuine technical-integrity defect capable of invalidating an existing main claim may reopen execution.

Next state: `FINAL SCIENTIFIC CLAIM FREEZE -> PAPER WRITING -> FINAL FIGURES`.

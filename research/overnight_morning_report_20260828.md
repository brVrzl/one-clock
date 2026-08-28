# ICRA27 overnight LIBERO research status

Generated 2026-08-28T02:24:32.269023+00:00 (UTC). Research branch: `exp/libero-component-temporal-reuse`.

## Frozen 80-block experiment

The frozen cohort is complete with exactly 80 task-condition blocks. Task-macro rates are primary; pooled rates are descriptive.

| condition | task-macro | pooled |
|---|---:|---:|
| fresh | 82.5% | 66/80 (82.5%) |
| fo4 | 85.0% | 68/80 (85.0%) |
| full_old4 | 85.0% | 68/80 (85.0%) |
| reverse4 | 81.2% | 65/80 (81.2%) |
| fo8 | 81.2% | 65/80 (81.2%) |
| full_old8 | 85.0% | 68/80 (85.0%) |
| reverse8 | 80.0% | 64/80 (80.0%) |
| fo16 | 76.2% | 61/80 (76.2%) |
| full_old16 | 72.5% | 58/80 (72.5%) |
| reverse16 | 62.5% | 50/80 (62.5%) |

Classification: C (non-monotonic temporal-source utility) plus D (strong task/suite heterogeneity). The age-16 FO-versus-reverse direction is not universal, so A, stable component-specific asymmetry, is not established.

## Standard LIBERO baselines

These are the untouched native-policy baselines, separate from all research interventions.

| model | spatial | object | goal | long | average |
|---|---:|---:|---:|---:|---:|
| SmolVLA | 85.0% | 93.0% | 78.0% | 42.0% | 74.5% |
| ACT | 64.0% | 48.0% | 70.0% | 41.0% | 55.8% |

Completed native ACT total: 223/400 (55.8%), across 40 task checkpoints.

## Fixed temporal aggregation follow-up

| method | task-macro | pooled |
|---|---:|---:|
| fresh | 90.0% | 72/80 (90.0%) |
| official_act_m001 | 82.5% | 66/80 (82.5%) |
| physical_exp_beta003 | 88.8% | 71/80 (88.8%) |
| cogact_alpha03 | 91.2% | 73/80 (91.2%) |
| component_arm_fresh_gripper_act | 91.2% | 73/80 (91.2%) |

Official ACT temporal aggregation does not explain the historical-source benefit: it is below Fresh (66/80 vs 72/80). CogACT-style shared aggregation captures most of the descriptive gain (73/80), while component-aware aggregation ties it (73/80) and has no clear added advantage.

## Independent ACT source-age confirmation

| task | fresh | FO16 | full-old16 | reverse16 | FO16−reverse16 |
|---|---:|---:|---:|---:|---:|
| libero_10:task3 | 9/10 | 7/10 | 6/10 | 3/10 | +0.40 (4/0) |
| libero_goal:task1 | 10/10 | 9/10 | 10/10 | 9/10 | +0.00 (1/1) |
| libero_object:task6 | 8/10 | 7/10 | 2/10 | 1/10 | +0.60 (7/1) |
| libero_spatial:task2 | 10/10 | 10/10 | 10/10 | 4/10 | +0.60 (6/0) |

Aggregate confirmation: Fresh 37/40, FO16 33/40, full-old16 28/40, reverse16 17/40; FO16 vs reverse16 is 18 candidate-only vs 2 reference-only, exact McNemar p=0.00040.

These are matched-query interventions on independent initial states 10–19, not the native ACT baseline. The native baseline retains its installed `n_action_steps`.

## Cross-policy direction

At age 16, SmolVLA has FO > reverse on 4/8 frozen tasks; ACT has FO > reverse on 3/4 confirmation tasks. These cohorts use different tasks, so this is a suite-level directional comparison rather than a same-task replication. Both policies retain fresh as the strongest aggregate reference in the completed cohorts, while reverse is the most fragile condition; the evidence supports structured component sensitivity, not a universal FO improvement.

## Scientific interpretation

The current main framing is broader conditional temporal-source utility. Large source ages often hurt the arm more than the gripper across the tested SmolVLA and ACT tasks, but this is task-dependent rather than a universal component rule. Fresh is not reliably surpassed. The source-age response is non-monotonic in some tasks and strongly heterogeneous across suites, supporting classification C+D rather than a stable global component-asymmetry claim.

The official ACT temporal ensemble does not explain the effect. CogACT-style shared aggregation captures most of the descriptive aggregation gain, and component-aware aggregation currently has no clear advantage over CogACT.

## Standard ACT baseline

Supervisor PID: 3373410. It remains independent and untouched.

| suite | completed native ACT task results |
|---|---:|
| libero_spatial | 64/100 (64.0%) |
| libero_object | 48/100 (48.0%) |
| libero_goal | 70/100 (70.0%) |
| libero_10 | 41/100 (41.0%) |

Queue state: completed=40.

Baseline jobs currently running: none.

Research processes currently running: none.

## Closest-work and novelty boundary

Action chunking and temporal ensembling are established in [ACT](https://arxiv.org/abs/2304.13705). [Lazzati et al.](https://arxiv.org/abs/2608.02547) is the closest conceptual collision because it explains action-chunking gains through delayed observation-conditioned predictions and implicit temporal ensembling. [CogACT](https://arxiv.org/abs/2411.19650) is a full-action similarity-weighted comparator. [TAS](https://arxiv.org/abs/2511.04421), [AutoHorizon](https://arxiv.org/abs/2602.21445), [AAC](https://arxiv.org/abs/2604.04161), and [PACE](https://arxiv.org/abs/2606.00537) address selection or execution-horizon adaptation rather than the present same-target component assignment.

RoboTwin was correctly deferred while LIBERO aggregation and ACT confirmation used the GPUs. The official/current path is [RoboTwin-Platform/RoboTwin](https://github.com/robotwin-Platform/robotwin), documented at [robotwin-platform.github.io](https://robotwin-platform.github.io/doc/); no standard RoboTwin run is reported here.

## Recommended framing and next experiment

The strongest defensible framing is conditional, non-monotonic temporal-source utility with substantial task heterogeneity. Do not claim a universal gripper advantage or present an intervention as a native baseline. The single highest-value next experiment is an independently frozen confirmatory cohort testing the fixed aggregation/source-age comparison selected before outcomes, with task-level paired analysis.

Active research artifacts remain under `experiments/component_temporal_reuse/`; baseline artifacts and supervisor remain under `experiments/standard_libero_baselines/`.

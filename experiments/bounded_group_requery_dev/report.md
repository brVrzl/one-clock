# Bounded group-triggered joint re-query development

## Protocol

This ACT-only development panel used four exposed tasks, states 10--19, and environment seeds 2000--2009. Every method/state condition used a fresh identically seeded environment. M0 is the repaired authoritative hard16 result; M1--M3 are 120 new episodes. All methods query one new ACT chunk and execute only that newest chunk for a bounded joint horizon. No historical action averaging, temporal ensemble, CogACT aggregation, independent group action source, learned predictor, or H_temp control was used.

The arm rule uses the earliest local minimum in normalized six-dimensional arm speed with threshold 0.5. The gripper rule uses the earliest open/close intent transition, with nonnegative commands treated as open and negative commands as close. M3 takes the minimum of the two proposed horizons.

## ACT results

| Method | Success /40 | Object | Spatial | Goal | L10 | Query rate | Mean horizon | Median horizon | Mean successful completion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 hard16 | 32/40 | 8 | 7 | 10 | 7 | 0.06506 | 16.00 | 16.0 | 127.5 |
| M1 arm phase | 30/40 | 5 | 8 | 9 | 8 | 0.08834 | 11.75 | 14.0 | 127.0 |
| M2 grip event | 35/40 | 8 | 8 | 10 | 9 | 0.07773 | 13.30 | 16.0 | 148.7 |
| M3 combined | 31/40 | 5 | 8 | 9 | 9 | 0.10110 | 10.12 | 9.0 | 142.0 |

### Horizon and trigger statistics

Fractions below use noninitial queries as the denominator. Both-nomination proximity is defined a priori as boundaries within one action step.

| Method | Total queries | Env steps | Arm nominations | Grip nominations | Both nominations | Both nearby (count/fraction) | Horizon histogram 4..16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 arm phase | 628 | 7109 | 0.561 | 0.000 | 0 | 0 | 4:56, 5:41, 6:33, 7:13, 8:52, 9:27, 10:22, 11:21, 12:22, 13:25, 14:46, 15:0, 16:270 |
| M2 grip event | 532 | 6844 | 0.000 | 0.376 | 0 | 0 | 4:29, 5:25, 6:12, 7:18, 8:22, 9:16, 10:16, 11:10, 12:10, 13:9, 14:10, 15:8, 16:347 |
| M3 combined | 726 | 7181 | 0.558 | 0.429 | 201 | 32/201 (0.159) | 4:110, 5:65, 6:87, 7:43, 8:38, 9:30, 10:17, 11:21, 12:24, 13:24, 14:42, 15:3, 16:222 |
| M0 hard16 | 458 | 7040 | n/a | n/a | n/a | n/a | 16:458 |

M0's exact query count and environment-step denominator are retained in `analysis.json`; its repaired baseline query rate is approximately 0.065.
McNemar probabilities are exact paired descriptive values for this 40-episode development cohort, not confirmatory significance claims.

## Paired comparisons

| Contrast | Candidate-only | Reference-only | Paired net | Exact McNemar p |
|---|---:|---:|---:|---:|
| M1_arm_phase_vs_M0_hard16 | 2 | 4 | -2 | 0.6875 |
| M2_gripper_event_vs_M0_hard16 | 3 | 0 | +3 | 0.25 |
| M3_group_event_joint_vs_M0_hard16 | 3 | 4 | -1 | 1 |
| M3_group_event_joint_vs_M1_arm_phase | 2 | 1 | +1 | 1 |
| M3_group_event_joint_vs_M2_gripper_event | 0 | 4 | -4 | 0.125 |

## M0 to M3 transition mechanism

The lists below contain every paired outcome transition between M0 and M3. The first dynamic re-query is logged from M3's newly predicted chunk; the first action divergence is compared only as a diagnostic, not as an additional inferential sample.

| Outcome transition | Task/state | First re-query (q, h_arm, h_grip, h_exec, reason) | First action divergence |
|---|---|---|---|
| M0 success → M3 failure | object3 / 10 | 12, 4, 16, 4, arm_phase | t=12 (q=12, h=4, arm_phase, Δ=0.0742) |
| M0 success → M3 failure | object3 / 14 | 16, 16, 16, 16, joint_fallback | t=55 (q=55, h=11, gripper_event, Δ=0.312) |
| M0 success → M3 failure | object3 / 16 | 13, 16, 16, 16, joint_fallback | t=13 (q=13, h=16, joint_fallback, Δ=0.0389) |
| M0 success → M3 failure | goal2 / 16 | 13, 16, 16, 16, joint_fallback | t=13 (q=13, h=16, joint_fallback, Δ=0.0573) |
| M0 failure → M3 success | spatial0 / 11 | 16, 13, 16, 13, arm_phase | t=29 (q=29, h=8, gripper_event, Δ=0.138) |
| M0 failure → M3 success | L10-3 / 11 | 16, 16, 16, 16, joint_fallback | t=68 (q=68, h=16, joint_fallback, Δ=1.26) |
| M0 failure → M3 success | L10-3 / 15 | 16, 16, 16, 16, joint_fallback | t=68 (q=68, h=5, arm_phase, Δ=0.475) |

## H_temp post-hoc analysis

H_temp was loaded only after adaptive outcomes were frozen and was never read by the executor. It is descriptive only.
- M1_arm_phase: H_temp versus arm-trigger frequency Spearman=-0.39999999999999997; versus gripper-trigger frequency Spearman=None; versus success gain Spearman=0.316227766016838.
- M2_gripper_event: H_temp versus arm-trigger frequency Spearman=None; versus gripper-trigger frequency Spearman=-0.7999999999999999; versus success gain Spearman=0.10540925533894598.
- M3_group_event_joint: H_temp versus arm-trigger frequency Spearman=-0.19999999999999998; versus gripper-trigger frequency Spearman=0.19999999999999998; versus success gain Spearman=0.39999999999999997.

## SmolVLA

SmolVLA was not launched. Because ACT selected SINGLE_TRIGGER_BETTER, a minimal M2-only confirmation is prepared in `protocol.json` and remains unrun; any execution requires separate approval and the same method-independent keyed flow-sampling protocol.

## Decision

**SINGLE_TRIGGER_BETTER**

The gripper-event joint re-query trigger is the smallest development winner: M2 reaches 35/40, exceeding repaired hard16 at 32/40, while M1 reaches 30/40 and M3 reaches 31/40. Select M2 for any separately approved follow-up; do not carry M1 or M3 forward.

See `protocol.json`, `analysis.json`, and the per-method result shards for the frozen definitions and complete episode-level logs.

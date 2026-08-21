# Matched-query group-wise selective commitment

Final verdict: **NO-GO**.

This is the completed 1,200-rollout LIBERO-Object mechanism gate: 10 tasks, 20 fixed initial states per task, two methods, and q in {4, 8, 16}. Both methods use the same frozen ACT policy and query the full joint chunk at exactly t % q == 0. The comparison makes no compute-saving claim.

## Matched-query success

| q | Global Replace | Selective Commit | Selective − Global | Paired bootstrap 95% CI |
|---:|---:|---:|---:|---:|
| 4 | 0.645 | 0.385 | -0.260 | [-0.350, -0.165] |
| 8 | 0.660 | 0.460 | -0.200 | [-0.285, -0.115] |
| 16 | 0.700 | 0.410 | -0.290 | [-0.375, -0.205] |

All three pooled paired intervals are below zero. Selective commitment does not improve task success.

## Acceptance and continuity

Selective Commit makes different arm/gripper decisions on 26.9%, 26.4%, and 28.2% of fresh queries for q=4, 8, and 16 respectively. Its both/arm-only/gripper-only/neither fractions are recorded in `acceptance_statistics.csv`.

| q | Method | Queries/step | Arm switches | Gripper switches | Overall discontinuity | Arm discontinuity | Gripper discontinuity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | global_replace | 0.25139 | 46.25 | 46.25 | 0.241 | 0.219 | 0.206 |
| 4 | selective_commit | 0.25064 | 10.67 | 14.74 | 0.309 | 0.251 | 0.333 |
| 8 | global_replace | 0.12643 | 22.05 | 22.05 | 0.339 | 0.313 | 0.296 |
| 8 | selective_commit | 0.12578 | 7.12 | 6.74 | 0.381 | 0.335 | 0.341 |
| 16 | global_replace | 0.06494 | 10.60 | 10.60 | 0.389 | 0.376 | 0.289 |
| 16 | selective_commit | 0.06466 | 4.95 | 3.29 | 0.452 | 0.434 | 0.285 |

Global Replace has lower normalized overall and arm query-boundary discontinuity at every q. Selective Commit reduces generation switches, but this does not translate into better execution; it also produces retained-generation exhaustion steps at q=8 and q=16, which are explicitly logged and do not trigger extra queries.

## Task consistency

Across the 30 task-by-cadence cells, Selective Commit is lower in 24, higher in 4, and tied in 2. The gains are therefore not a consistent task-level mechanism effect.

## Answers to the gate questions

1. Query schedules are matched exactly: every rollout satisfies t % q == 0, and paired methods share the same schedule through their common executed prefix.
2. Different group decisions occur in 26.9%/26.4%/28.2% of fresh queries for q=4/8/16.
3. Selective commitment reduces success at all three q values.
4. It does not reduce overall or arm action discontinuity; gripper discontinuity is lower only marginally at q=16.
5. The effect is not consistent across q: the direction is consistently harmful for success, while continuity is also consistently worse overall and for arm.
6. The effect is not consistent across tasks; 24/30 task-q cells favor Global Replace.
7. Verdict: **NO-GO** under the predeclared interpretation.
8. Recommendation: **D — stop/reframe group-wise commitment**. Do not automatically proceed to learned cheap verification, another current-query verifier, or soft acceptance.

No ACT retraining, reliability estimator, Y_refresh label, future observation, rollout oracle, soft blending, SmolVLA, RoboTwin, or paper change was used.

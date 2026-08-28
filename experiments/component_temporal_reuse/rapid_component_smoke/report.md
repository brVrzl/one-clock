# Rapid ACT component-wise aggregation smoke

New paired initial states and fixed policy RNG seed; task-macro values are primary and pooled values are descriptive.

## Per-task success

| task | fresh | official_act_temporal_ensemble | cogact_shared_full_action | component_arm_fresh_gripper_act | groupwise_similarity | groupwise_similarity_age | groupwise_similarity_age_gripper_vote |
|---|---|---|---|---|---|---|---|
| libero_10:task3 | 5/5 | 2/5 | 2/5 | 2/5 | 2/5 | 2/5 | 2/5 |
| libero_goal:task1 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| libero_object:task6 | 4/5 | 2/5 | 2/5 | 4/5 | 2/5 | 3/5 | 3/5 |
| libero_spatial:task2 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

## Paired variants versus CogACT shared full-action aggregation

| variant | task-macro delta | pooled delta | candidate-only/reference-only | exact McNemar p | Goal |
|---|---:|---:|---:|---:|---:|
| groupwise_similarity | +0.000 | +0.000 | 0/0 | None | 5/5 |
| groupwise_similarity_age | +0.050 | +0.050 | 2/1 | 1.0 | 5/5 |
| groupwise_similarity_age_gripper_vote | +0.050 | +0.050 | 2/1 | 1.0 | 5/5 |

Leading variant under the predeclared rule: `none`.

No ACT variant met the predeclared smoke criterion; run the same smoke on the existing SmolVLA development tasks.

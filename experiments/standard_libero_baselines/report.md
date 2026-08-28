# Standard LIBERO baselines

Standard native-policy evaluation only. No one-clock, DCTA, FO, or adaptive-horizon logic was used.

Environment: LeRobot 0.4.4, LIBERO 0.1.1, PyTorch 2.7.1+cu128, MuJoCo 3.3.1, `MUJOCO_GL=egl`.

| model | spatial | object | goal | long | average |
|---|---:|---:|---:|---:|---:|
| SmolVLA | 85.0% | 93.0% | 78.0% | 42.0% | 74.5% |
| ACT | 64.0% | 48.0% | 70.0% | 41.0% | 55.8% |

Rates are successes / 10 episodes per task, aggregated over the ten tasks in each suite. `average` is the macro-average of the four completed suite rates.

SmolVLA: public `HuggingFaceVLA/smolvla_libero`, revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, native `chunk_size=50`, `n_action_steps=1`.

ACT final baseline: official LeRobot ACT, one newly trained task-specific checkpoint per task, common recipe, native `chunk_size=100`, `n_action_steps=100`, no temporal ensembling.

Corrected Object gate: the 100,000-step task-specific model trained on 34 of 44 alphabet-soup episodes; the remaining 10 episodes were held out. Held-out one-step arm RMSE was 0.116 and gripper RMSE 0.474; valid chunk arm RMSE was 0.115 and gripper RMSE 0.349. Native standard rollout was 5/10. The full ACT queue is therefore unlocked; the corrected Object artifact is a gate diagnostic and is not substituted for the final all-data model.

## ACT diagnostic

Selected task: `libero_object` task 0, `pick up the alphabet soup and place it in the basket` (dataset task index 24, 44 episodes, 6,867 frames). The matched native protocol used initial-state IDs 0-9, seeds 1000-1009, relative control, the two standard cameras, and 10 episodes.

Root cause: the four 1,000-step ACT pilots selected data by dataset-global index, but LIBERO task IDs are suite-local and the dataset order differs. The Object task-0 pilot trained on dataset task 20 (`orange juice`) while the evaluator ran benchmark task 0 (`alphabet soup`); the same order mismatch affected Goal, LIBERO-10, and Spatial. The corrected queue manifest now matches exact task language to dataset metadata.

Matched native results: new official pilot `0/10`; historical checkpoint `0/10`. The historical checkpoint is diagnostic only and is not used in the final baseline. With the exact same evaluator and task, the separate historical `n_action_steps=8` execution diagnostic scored `8/10`, showing that native 100-step open-loop execution is an additional runtime sensitivity; native `n_action_steps=100` remains the standard reported ACT protocol.

Offline sanity for the failed new pilot is in `act_diagnostic_object_task0_offline.json`; it compares the actual training task and the intended alphabet-soup task. The model receives `observation.images.image` and `observation.images.image2` (3x256x256, [0,1]), an 8-D state (EEF position xyz, EEF axis-angle xyz, two gripper positions), and a 7-D relative action (translation delta xyz, rotation delta xyz, gripper command). Saved training/evaluation MEAN-STD statistics match the dataset metadata. The intended-task one-step RMSE was 0.390 versus 0.253 on the actual training-task audit, consistent with the wrong task selection.

The dataset image tensors are 3x256x256, while the untouched native LIBERO evaluator supplies 360x360 camera frames and the native processor performs no resize. This train/eval resolution difference was retained as part of the standard evaluator; the corrected model nevertheless achieved 5/10 natively.

All four original pilots are smoke models, not failed final models: each stopped at 1,000 of the intended 100,000 steps. Their loss trajectories fell from roughly 8.7-9.2 at step 100 to 1.61-1.73 at step 1,000, with no plateau. Representative failed videos show the new Object pilot making small/incorrect workspace motions without a grasp; the historical native-100 rollout moves substantially but misses under the long open-loop chunk, while its h8 diagnostic produces directed successful rollouts.

Detached ACT supervisor: `/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/overnight_state.json`. It records pending/running/completed/failed state, per-stage PID and logs under `logs/`, adopts live jobs after restart, and judges completion from checkpoint/evaluation artifacts. Current queue status: {'pending': 0, 'running': 0, 'completed': 40, 'failed': 0}.

## ACT inventory

- `/home/wjq/checkpoints/zeromidnight_act_libero_object`: found and accepted only as a historical Object sanity/reference checkpoint; rejected as a final model because it is language-blind/multi-task.
- `/home/wjq/checkpoints/ishandotsh_act_libero_spatial_test`: found and accepted only as a historical Spatial sanity/reference checkpoint; rejected as a final model because it is language-blind/multi-task.
- `/home/thor/projects/checkpoints/zeromidnight_act_libero_object`: absent on this machine; `/home/wjq/checkpoints/zeromidnight_act_libero_object` is the local equivalent.
- `upstreams/verl-vla/assets/hf_models/act_libero`: rejected because it contains config metadata but no model weights. No prior valid Goal or Long checkpoints were found. The final ACT queue trains any missing task-specific models with one common recipe.

## Per-task results

| model | suite | task | checkpoint | successes / episodes | rate | chunk | n_action | origin | status |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| SmolVLA | libero_spatial | 0 | HuggingFaceVLA/smolvla_libero | 8 / 10 | 80.0% | 50 | 1 | public | complete |
| SmolVLA | libero_spatial | 1 | HuggingFaceVLA/smolvla_libero | 10 / 10 | 100.0% | 50 | 1 | public | complete |
| SmolVLA | libero_spatial | 2 | HuggingFaceVLA/smolvla_libero | 8 / 10 | 80.0% | 50 | 1 | public | complete |
| SmolVLA | libero_spatial | 3 | HuggingFaceVLA/smolvla_libero | 5 / 10 | 50.0% | 50 | 1 | public | complete |
| SmolVLA | libero_spatial | 4 | HuggingFaceVLA/smolvla_libero | 9 / 10 | 90.0% | 50 | 1 | public | complete |
| SmolVLA | libero_spatial | 5 | HuggingFaceVLA/smolvla_libero | 10 / 10 | 100.0% | 50 | 1 | public | complete |
| SmolVLA | libero_spatial | 6 | HuggingFaceVLA/smolvla_libero | 10 / 10 | 100.0% | 50 | 1 | public | complete |
| SmolVLA | libero_spatial | 7 | HuggingFaceVLA/smolvla_libero | 8 / 10 | 80.0% | 50 | 1 | public | complete |
| SmolVLA | libero_spatial | 8 | HuggingFaceVLA/smolvla_libero | 9 / 10 | 90.0% | 50 | 1 | public | complete |
| SmolVLA | libero_spatial | 9 | HuggingFaceVLA/smolvla_libero | 8 / 10 | 80.0% | 50 | 1 | public | complete |
| SmolVLA | libero_object | 0 | HuggingFaceVLA/smolvla_libero | 10 / 10 | 100.0% | 50 | 1 | public | complete |
| SmolVLA | libero_object | 1 | HuggingFaceVLA/smolvla_libero | 10 / 10 | 100.0% | 50 | 1 | public | complete |
| SmolVLA | libero_object | 2 | HuggingFaceVLA/smolvla_libero | 10 / 10 | 100.0% | 50 | 1 | public | complete |
| SmolVLA | libero_object | 3 | HuggingFaceVLA/smolvla_libero | 9 / 10 | 90.0% | 50 | 1 | public | complete |
| SmolVLA | libero_object | 4 | HuggingFaceVLA/smolvla_libero | 9 / 10 | 90.0% | 50 | 1 | public | complete |
| SmolVLA | libero_object | 5 | HuggingFaceVLA/smolvla_libero | 7 / 10 | 70.0% | 50 | 1 | public | complete |
| SmolVLA | libero_object | 6 | HuggingFaceVLA/smolvla_libero | 10 / 10 | 100.0% | 50 | 1 | public | complete |
| SmolVLA | libero_object | 7 | HuggingFaceVLA/smolvla_libero | 9 / 10 | 90.0% | 50 | 1 | public | complete |
| SmolVLA | libero_object | 8 | HuggingFaceVLA/smolvla_libero | 10 / 10 | 100.0% | 50 | 1 | public | complete |
| SmolVLA | libero_object | 9 | HuggingFaceVLA/smolvla_libero | 9 / 10 | 90.0% | 50 | 1 | public | complete |
| SmolVLA | libero_goal | 0 | HuggingFaceVLA/smolvla_libero | 8 / 10 | 80.0% | 50 | 1 | public | complete |
| SmolVLA | libero_goal | 1 | HuggingFaceVLA/smolvla_libero | 10 / 10 | 100.0% | 50 | 1 | public | complete |
| SmolVLA | libero_goal | 2 | HuggingFaceVLA/smolvla_libero | 7 / 10 | 70.0% | 50 | 1 | public | complete |
| SmolVLA | libero_goal | 3 | HuggingFaceVLA/smolvla_libero | 5 / 10 | 50.0% | 50 | 1 | public | complete |
| SmolVLA | libero_goal | 4 | HuggingFaceVLA/smolvla_libero | 9 / 10 | 90.0% | 50 | 1 | public | complete |
| SmolVLA | libero_goal | 5 | HuggingFaceVLA/smolvla_libero | 8 / 10 | 80.0% | 50 | 1 | public | complete |
| SmolVLA | libero_goal | 6 | HuggingFaceVLA/smolvla_libero | 6 / 10 | 60.0% | 50 | 1 | public | complete |
| SmolVLA | libero_goal | 7 | HuggingFaceVLA/smolvla_libero | 10 / 10 | 100.0% | 50 | 1 | public | complete |
| SmolVLA | libero_goal | 8 | HuggingFaceVLA/smolvla_libero | 8 / 10 | 80.0% | 50 | 1 | public | complete |
| SmolVLA | libero_goal | 9 | HuggingFaceVLA/smolvla_libero | 7 / 10 | 70.0% | 50 | 1 | public | complete |
| SmolVLA | libero_10 | 0 | HuggingFaceVLA/smolvla_libero | 2 / 10 | 20.0% | 50 | 1 | public | complete |
| SmolVLA | libero_10 | 1 | HuggingFaceVLA/smolvla_libero | 5 / 10 | 50.0% | 50 | 1 | public | complete |
| SmolVLA | libero_10 | 2 | HuggingFaceVLA/smolvla_libero | 3 / 10 | 30.0% | 50 | 1 | public | complete |
| SmolVLA | libero_10 | 3 | HuggingFaceVLA/smolvla_libero | 9 / 10 | 90.0% | 50 | 1 | public | complete |
| SmolVLA | libero_10 | 4 | HuggingFaceVLA/smolvla_libero | 2 / 10 | 20.0% | 50 | 1 | public | complete |
| SmolVLA | libero_10 | 5 | HuggingFaceVLA/smolvla_libero | 8 / 10 | 80.0% | 50 | 1 | public | complete |
| SmolVLA | libero_10 | 6 | HuggingFaceVLA/smolvla_libero | 5 / 10 | 50.0% | 50 | 1 | public | complete |
| SmolVLA | libero_10 | 7 | HuggingFaceVLA/smolvla_libero | 4 / 10 | 40.0% | 50 | 1 | public | complete |
| SmolVLA | libero_10 | 8 | HuggingFaceVLA/smolvla_libero | 0 / 10 | 0.0% | 50 | 1 | public | complete |
| SmolVLA | libero_10 | 9 | HuggingFaceVLA/smolvla_libero | 4 / 10 | 40.0% | 50 | 1 | public | complete |
| ACT | libero_spatial | 0 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task0/checkpoints/100000/pretrained_model | 5 / 10 | 50.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_spatial | 1 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task1/checkpoints/100000/pretrained_model | 6 / 10 | 60.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_spatial | 2 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task2/checkpoints/100000/pretrained_model | 9 / 10 | 90.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_spatial | 3 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task3/checkpoints/100000/pretrained_model | 6 / 10 | 60.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_spatial | 4 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task4/checkpoints/100000/pretrained_model | 3 / 10 | 30.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_spatial | 5 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task5/checkpoints/100000/pretrained_model | 8 / 10 | 80.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_spatial | 6 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task6/checkpoints/100000/pretrained_model | 8 / 10 | 80.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_spatial | 7 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task7/checkpoints/100000/pretrained_model | 7 / 10 | 70.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_spatial | 8 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task8/checkpoints/100000/pretrained_model | 5 / 10 | 50.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_spatial | 9 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_spatial_task9/checkpoints/100000/pretrained_model | 7 / 10 | 70.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_object | 0 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task0/checkpoints/100000/pretrained_model | 2 / 10 | 20.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_object | 1 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task1/checkpoints/100000/pretrained_model | 5 / 10 | 50.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_object | 2 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task2/checkpoints/100000/pretrained_model | 5 / 10 | 50.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_object | 3 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task3/checkpoints/100000/pretrained_model | 2 / 10 | 20.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_object | 4 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task4/checkpoints/100000/pretrained_model | 6 / 10 | 60.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_object | 5 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task5/checkpoints/100000/pretrained_model | 5 / 10 | 50.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_object | 6 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task6/checkpoints/100000/pretrained_model | 8 / 10 | 80.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_object | 7 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task7/checkpoints/100000/pretrained_model | 5 / 10 | 50.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_object | 8 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task8/checkpoints/100000/pretrained_model | 7 / 10 | 70.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_object | 9 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_object_task9/checkpoints/100000/pretrained_model | 3 / 10 | 30.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_goal | 0 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task0/checkpoints/100000/pretrained_model | 5 / 10 | 50.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_goal | 1 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task1/checkpoints/100000/pretrained_model | 10 / 10 | 100.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_goal | 2 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task2/checkpoints/100000/pretrained_model | 10 / 10 | 100.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_goal | 3 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task3/checkpoints/100000/pretrained_model | 5 / 10 | 50.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_goal | 4 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task4/checkpoints/100000/pretrained_model | 9 / 10 | 90.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_goal | 5 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task5/checkpoints/100000/pretrained_model | 3 / 10 | 30.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_goal | 6 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task6/checkpoints/100000/pretrained_model | 3 / 10 | 30.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_goal | 7 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task7/checkpoints/100000/pretrained_model | 9 / 10 | 90.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_goal | 8 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task8/checkpoints/100000/pretrained_model | 9 / 10 | 90.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_goal | 9 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_goal_task9/checkpoints/100000/pretrained_model | 7 / 10 | 70.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_10 | 0 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task0/checkpoints/100000/pretrained_model | 0 / 10 | 0.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_10 | 1 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task1/checkpoints/100000/pretrained_model | 7 / 10 | 70.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_10 | 2 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task2/checkpoints/100000/pretrained_model | 5 / 10 | 50.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_10 | 3 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task3/checkpoints/100000/pretrained_model | 3 / 10 | 30.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_10 | 4 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task4/checkpoints/100000/pretrained_model | 2 / 10 | 20.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_10 | 5 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task5/checkpoints/100000/pretrained_model | 9 / 10 | 90.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_10 | 6 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task6/checkpoints/100000/pretrained_model | 4 / 10 | 40.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_10 | 7 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task7/checkpoints/100000/pretrained_model | 3 / 10 | 30.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_10 | 8 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task8/checkpoints/100000/pretrained_model | 1 / 10 | 10.0% | 100 | 100 | newly_trained | complete |
| ACT | libero_10 | 9 | /home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/libero_10_task9/checkpoints/100000/pretrained_model | 7 / 10 | 70.0% | 100 | 100 | newly_trained | complete |

## Pilot and queue status

ACT pilot evaluations completed natively for one task in each suite (all four loaded, rolled out, and produced videos; pilot success counts were Goal 0/10, Object 0/10, LIBERO-10 0/10, Spatial 0/10). The pilot models are validation artifacts and are excluded from the final ACT table above.

Incomplete or failed final records at report-generation time: none.

The complete per-task machine-readable records, pilot results, queue status, checkpoint origin, and artifact paths are in `results.json`.

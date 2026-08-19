# Cross-task video replay provenance

These are small post-hoc replays, not additional sweep datapoints. The video
frames use the existing agent-view/wrist-view side-by-side recording path.

## Off-diagonal success / global failure and directionality

| init_state_id | task_id | configuration | success | environment steps | video |
|---:|---:|---|---|---:|---|
| 0 | 1 | global h=4 | false | 280 | `experiments/runs/libero_object_cross_task/videos/4.mp4` |
| 0 | 1 | arm=2, gripper=16 | true | 169 | `experiments/runs/libero_object_cross_task/videos/arm=2,gripper=16.mp4` |
| 0 | 1 | arm=16, gripper=2 | false | 280 | `experiments/runs/libero_object_cross_task/videos/arm=16,gripper=2.mp4` |

State 0 was selected because the preferred `(2,16)` assignment succeeded while
the selected tied-best global `h=4` and reversed `(16,2)` assignment failed.

## No-benefit paired case

| init_state_id | task_id | configuration | success | environment steps | video |
|---:|---:|---|---|---:|---|
| 11 | 3 | global h=16 | false | 280 | `experiments/runs/libero_object_cross_task/videos/task3_init11_global_h16.mp4` |
| 11 | 3 | arm=16, gripper=8 | false | 280 | `experiments/runs/libero_object_cross_task/videos/arm=16,gripper=8.mp4` |

The selected state has the same failure outcome for the task-3 global and
off-diagonal configurations. No qualitative mechanism claim is made from
these videos.

## Replay anomaly

The first replay shell command created one additional unselected task-3 replay
at global `h=4` under
`experiments/runs/libero_object_cross_task/videos/unused.mp4` because of a
local naming/argument mistake. It was not used as the task-3 no-benefit case;
the correctly configured `task3_init11_global_h16.mp4` was run separately.

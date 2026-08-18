# LIBERO 50-state static-grid video cases

These are small post-hoc replays selected from new official states 20–49.
They record objective provenance only; no qualitative mechanism claim is made.

| Init state | Configuration | Result | Environment steps | Video |
|---:|---|---|---:|---|
| 41 | global h=8 | failure | 280 | `experiments/runs/libero_static_grid_50/videos/state41_global_h8.mp4` |
| 41 | arm=4, gripper=16 | success | 161 | `experiments/runs/libero_static_grid_50/videos/state41_arm4_grip16.mp4` |
| 20 | arm=4, gripper=16 | success | 182 | `experiments/runs/libero_static_grid_50/videos/state20_arm4_grip16.mp4` |
| 20 | arm=16, gripper=4 | failure | 280 | `experiments/runs/libero_static_grid_50/videos/state20_arm16_grip4.mp4` |

State 41 was selected because the best off-diagonal configuration succeeded
where the best global configuration failed. State 20 was selected for the
directional `(4,16)` versus `(16,4)` disagreement.

# LIBERO static-grid video cases

These videos are objective replays selected after the complete 600-episode
static sweep. They are for human inspection; no qualitative behavior claim is
recorded here.

| Init state | Configuration | Result | Environment steps | Video |
|---:|---|---|---:|---|
| 19 | global h=8 | failure | 280 | `experiments/runs/libero_static_grid_20/videos/init19_global_h8.mp4` |
| 19 | arm=4, gripper=16 | success | 208 | `experiments/runs/libero_static_grid_20/videos/init19_arm4_grip16.mp4` |
| 5 | arm=8, gripper=2 | failure | 280 | `experiments/runs/libero_static_grid_20/videos/init05_arm8_grip2.mp4` |
| 5 | arm=2, gripper=8 | success | 191 | `experiments/runs/libero_static_grid_20/videos/init05_arm2_grip8.mp4` |

The first pair is the only best-off-diagonal versus best-global
groupwise-only success in the paired 20-state comparison. The second pair is
the selected directional comparison because the two arm/gripper assignments
disagree on init state 5.

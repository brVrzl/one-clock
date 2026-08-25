# RoboTwin sealed exploratory analysis

Classification: **NO_SIGNAL**

## Success by task and method

| Task | Method | Success | Rate |
|---|---|---:|---:|
| `beat_block_hammer` | `NATIVE_ACT` | 9/20 | 45.0% |
| `click_alarmclock` | `NATIVE_ACT` | 1/20 | 5.0% |
| `dump_bin_bigbin` | `NATIVE_ACT` | 1/20 | 5.0% |
| `handover_block` | `NATIVE_ACT` | 0/20 | 0.0% |
| `open_laptop` | `NATIVE_ACT` | 8/20 | 40.0% |
| `beat_block_hammer` | `NEWEST` | 0/20 | 0.0% |
| `click_alarmclock` | `NEWEST` | 4/20 | 20.0% |
| `dump_bin_bigbin` | `NEWEST` | 0/20 | 0.0% |
| `handover_block` | `NEWEST` | 0/20 | 0.0% |
| `open_laptop` | `NEWEST` | 7/20 | 35.0% |
| `beat_block_hammer` | `FULL_OLD_1S` | 1/20 | 5.0% |
| `click_alarmclock` | `FULL_OLD_1S` | 2/20 | 10.0% |
| `dump_bin_bigbin` | `FULL_OLD_1S` | 0/20 | 0.0% |
| `handover_block` | `FULL_OLD_1S` | 0/20 | 0.0% |
| `open_laptop` | `FULL_OLD_1S` | 6/20 | 30.0% |
| `beat_block_hammer` | `FO_1S` | 0/20 | 0.0% |
| `click_alarmclock` | `FO_1S` | 3/20 | 15.0% |
| `dump_bin_bigbin` | `FO_1S` | 0/20 | 0.0% |
| `handover_block` | `FO_1S` | 0/20 | 0.0% |
| `open_laptop` | `FO_1S` | 8/20 | 40.0% |
| `beat_block_hammer` | `GRIPPER_HOLD` | 0/20 | 0.0% |
| `click_alarmclock` | `GRIPPER_HOLD` | 0/20 | 0.0% |
| `dump_bin_bigbin` | `GRIPPER_HOLD` | 0/20 | 0.0% |
| `handover_block` | `GRIPPER_HOLD` | 0/20 | 0.0% |
| `open_laptop` | `GRIPPER_HOLD` | 5/20 | 25.0% |
| `beat_block_hammer` | `GRIPPER_EMA_1S` | 0/20 | 0.0% |
| `click_alarmclock` | `GRIPPER_EMA_1S` | 3/20 | 15.0% |
| `dump_bin_bigbin` | `GRIPPER_EMA_1S` | 0/20 | 0.0% |
| `handover_block` | `GRIPPER_EMA_1S` | 0/20 | 0.0% |
| `open_laptop` | `GRIPPER_EMA_1S` | 7/20 | 35.0% |

## Paired contrasts

- `FO_1S - NEWEST`: +0.000; wins/losses/ties 2/2/96; task-cluster 95% interval [-0.030, +0.030].
- `FO_1S - NATIVE_ACT`: -0.080; wins/losses/ties 3/11/86; task-cluster 95% interval [-0.270, +0.050].
- `FO_1S - FULL_OLD_1S`: +0.020; wins/losses/ties 3/1/96; task-cluster 95% interval [-0.020, +0.070].
- `FO_1S - GRIPPER_HOLD`: +0.060; wins/losses/ties 6/0/94; task-cluster 95% interval [+0.000, +0.120].
- `FO_1S - GRIPPER_EMA_1S`: +0.010; wins/losses/ties 2/1/97; task-cluster 95% interval [+0.000, +0.030].

This is the preregistered exploratory analysis; no confirmatory p-value is claimed.

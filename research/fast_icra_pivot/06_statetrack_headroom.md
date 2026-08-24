# StateTrack headroom-first causal gate

Date: 2026-08-24  
Starting commit: `39cc9f76ea0997349d67bc52fa7ca27e1a20a983`  
Checkpoint: `/home/wjq/checkpoints/zeromidnight_act_libero_object`  
Tasks and states: LIBERO-Object tasks 1, 6, and 8; official initial-state IDs 0--9, paired by `(task_id, init_state_id, seed=1000+init_state_id)`.

## Action/state semantics audit

The active LeRobot `LiberoEnv` is configured with `control_mode: relative`. On reset, its source code sets `robot.controller.use_delta = True`. LIBERO's environment wrapper selects the default `OSC_POSE` robosuite controller. That controller clips normalized arm inputs to `[-1, 1]`, scales the first three dimensions to `+/-0.05 m` and the next three to `+/-0.5 rad`, and applies the relative position / axis-angle goal from the current EEF pose. Dimension 6 is the gripper command. Therefore these are relative EEF pose commands, not absolute joint targets, and StateTrack reconstructs a nominal EEF trajectory from the query-time EEF position/quaternion by integrating the controller scales. No re-anchoring or numerical arm correction is applied.

The online observation contains `robot_state.eef.pos`, `robot_state.eef.quat`, `robot_state.eef.mat`, `robot_state.joints.pos/vel`, and `robot_state.gripper.qpos/qvel`. The policy state processor uses `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]`.

## Implementation

`src/one_clock/state_track.py` implements a monotonic nearest-state index with maximum one forward progress step per control tick. It permits repeats when measured state remains behind the nominal trajectory and selects `action[progress + lookahead]` with lookahead 1 or 2. The selected row is copied verbatim, including its gripper value. `scripts/run_libero_statetrack.py` keeps the existing 8-step policy query horizon and 100-step ACT output chunk, logs raw per-step diagnostics, and optionally repeats one prior action every fifth control tick as an interpretable dropped-update stress test.

## Paired nominal results

| Task | State | ACT | StateTrack L1 | StateTrack L2 | ACT hold-5 | StateTrack L1 hold-5 |
|---:|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0 | ✓ | ✗ | ✗ | ✓ | ✗ |
| 1 | 1 | ✓ | ✗ | ✗ | ✗ | ✗ |
| 1 | 2 | ✓ | ✓ | ✓ | ✓ | ✗ |
| 1 | 3 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1 | 4 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1 | 5 | ✓ | ✗ | ✗ | ✗ | ✗ |
| 1 | 6 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1 | 7 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1 | 8 | ✓ | ✓ | ✗ | ✓ | ✗ |
| 1 | 9 | ✓ | ✗ | ✓ | ✓ | ✗ |
| 6 | 0 | ✓ | ✗ | ✗ | ✗ | ✗ |
| 6 | 1 | ✓ | ✗ | ✗ | ✓ | ✗ |
| 6 | 2 | ✓ | ✗ | ✗ | ✓ | ✗ |
| 6 | 3 | ✓ | ✗ | ✓ | ✓ | ✗ |
| 6 | 4 | ✓ | ✗ | ✗ | ✓ | ✗ |
| 6 | 5 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 6 | 6 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 6 | 7 | ✓ | ✗ | ✗ | ✓ | ✗ |
| 6 | 8 | ✓ | ✗ | ✓ | ✓ | ✗ |
| 6 | 9 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | 0 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | 1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | 2 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | 3 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | 4 | ✓ | ✗ | ✗ | ✗ | ✗ |
| 8 | 5 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | 6 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | 7 | ✓ | ✗ | ✓ | ✓ | ✗ |
| 8 | 8 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | 9 | ✗ | ✗ | ✓ | ✗ | ✗ |

### Pooled gate metrics

| Method | Successes / 30 | Success rate | Baseline failures rescued | Baseline successes broken |
|---|---:|---:|---:|---:|
| Frozen ACT, nominal | 15/30 | 50.0% | — | — |
| StateTrack lookahead=1, nominal | 2/30 | 6.7% | 0/15 | 13/15 |
| StateTrack lookahead=2, nominal | 6/30 | 20.0% | 1/15 | 10/15 |
| Timing oracle over {ACT,L1,L2}, nominal | 16/30 | 53.3% | 1/15 (6.7%) | — |
| Frozen ACT, hold-5 | 11/30 | 36.7% | — | — |
| StateTrack lookahead=1, hold-5 | 0/30 | 0.0% | 0/19 | 11/11 |

StateTrack's observed progress remained near the start of each 8-step active prefix (mean progress index 0.06--0.19) and repeated approximately 204--237 rows per rollout. This indicates the prescribed tracker mostly held the first lookahead target rather than following the nominal trajectory. Query cadence remained 8 control ticks in every method; no extra policy calls were made. The tracker adds only a 6-D nearest-distance calculation per control tick and no learned model. The current runner records policy-query latency but not a separate wall-clock timer around this NumPy calculation; the overhead claim is therefore architectural, not a separately measured latency result.

## Decision

The nominal oracle headroom is only one extra state (15/30 to 16/30), while both deployable fixed lookaheads regress heavily. Under the one-step-hold mismatch, StateTrack is 0/30 versus frozen ACT 11/30. This fails both continuation criteria. StateTrack is killed immediately; no threshold tuning, selector training, continuous residual, or larger sweep is justified.

Raw episode logs are retained under `artifacts/statetrack_nominal/` and `artifacts/statetrack_perturbed/`. The stress perturbation is a one-step repeat of the previous action every fifth control tick, applied identically by seed across methods.


# System contract re-audit

Audit date: 2026-08-21.

## Decision

The **LIBERO execution contract is internally consistent enough to audit the available single-arm results**. The claimed arm/gripper slices are correct for this runner, ACT is frozen and deterministic at inference, and diagonal group schedules reproduce the global execution rule.

The **RoboTwin experimental contract is not established**. Current upstream source confirms the reported 14-D ordering, but no valid historical project rollout is tied to an exact upstream commit, checkpoint, and verified policy output semantics. RoboTwin evidence must not be used for scientific conclusions until that linkage is rebuilt.

## Policy contract: LIBERO ACT

| Item | Re-audited fact | Primary evidence | Status |
|---|---|---|---|
| Policy | LeRobot ACT, checkpoint path `/home/thor/projects/checkpoints/zeromidnight_act_libero_object` | Run metadata; checkpoint `config.json` | VERIFIED at current bytes; historical byte identity partial |
| Current checkpoint hash | `model.safetensors` SHA-256 `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410` | Direct hash | VERIFIED now |
| Chunk size | 100 actions, seven dimensions | Checkpoint config and every canonical run’s observed chunk shape | VERIFIED |
| Inference | `torch.inference_mode()`, policy in eval mode, full chunk returned then postprocessed | `scripts/run_libero_gate0.py` and LeRobot policy path at commit `f66e512...` | VERIFIED |
| Stochasticity | ACT CVAE prior is not sampled during inference; prediction is deterministic for a fixed observation/checkpoint/runtime | ACT policy code | VERIFIED at code level |
| Temporal ensemble | Disabled (`temporal_ensemble_coeff = null`) in all audited fixed-horizon LIBERO metadata; the runner rejects incompatible setup | Run metadata and runner | VERIFIED |
| Policy reset | Called at every episode boundary | Runner | VERIFIED |
| Query rule | Query at t=0 and when the executor’s relevant buffer expires; one full ACT chunk per query | Runner and `src/one_clock/executor.py` | VERIFIED |
| Weight updates | None during rollout | Frozen runner path; no optimizer/training call | VERIFIED |

Historical early run metadata records the checkpoint path but not its hash and does not record the one-clock Git commit. “Same frozen checkpoint” is therefore supported by path/config consistency, not cryptographic proof across every early run.

## Action contract: LIBERO

The valid rollout action is seven-dimensional:

\[
a_t = [\Delta x,\Delta y,\Delta z,\Delta r_x,\Delta r_y,\Delta r_z,g].
\]

| Property | Re-audited fact | Consequence |
|---|---|---|
| Ordering | indices 0:6 are end-effector translation plus axis-angle rotation deltas; index 6 is gripper | The project’s `arm=[0..5]`, `gripper=[6]` partition is correct for LIBERO. |
| Absolute/relative | Arm is relative Cartesian pose control (`use_delta=True`) | Offline numerical error is in controller-normalized delta coordinates, not joint positions. |
| Nominal range | LeRobot declares action space `[-1,1]^7` | Declaration is not the whole runtime contract. |
| Arm clipping | Robosuite’s arm controller clips normalized inputs before scaling | Occasional arm values beyond one are behaviorally saturated. |
| Gripper semantics | Environment uses only `sign(g)` and clips the internal gripper state | Continuous gripper magnitude MSE is not aligned with environment behavior. |
| Observed range | Across 709,241 saved historical actions, counts with `abs(a)>1` are `[95,874,8628,0,0,0,551635]` | 77.78% of saved gripper scalars exceed magnitude one, but magnitude beyond sign has no execution effect. |
| Project-side clipping | The runner does not clip after ACT postprocessing | Environment/controller transformations determine saturation/sign behavior. |
| Interpolation | LIBERO/robosuite controller processes each 20 Hz control action through its low-level controller | One high-level action is one control tick for the audit’s execution-horizon accounting. |

This contract directly weakens any claim based on continuous gripper MSE. A control-aligned gripper loss should at minimum treat open/close sign separately and report transition timing.

## Observation contract: LIBERO

| Input | Exact treatment | Status / limitation |
|---|---|---|
| Third-person image | `agentview_image`, 256×256 RGB | Mapped to `observation.images.image` |
| Wrist image | `robot0_eye_in_hand_image`, 256×256 RGB | Mapped to `observation.images.wrist_image` |
| Orientation | Both images are rotated 180° by the LeRobot LIBERO processor (height and width flips) | Matches the training adapter |
| Pixel normalization | uint8 HWC → float CHW in `[0,1]` | Verified in preprocessing path |
| Proprioception | 8-D: end-effector position (3), quaternion converted to axis-angle (3), gripper qpos (2) | Matches checkpoint feature shape |
| Camera ordering | Determined by checkpoint feature names, not positional concatenation | Verified for the two named inputs |
| Language | Task description is recorded by the runner but is **not** an ACT policy input | No language-conditioned inference claim is valid for this checkpoint |
| Episode state | Fresh environment observation is used on every control tick; the policy is queried only on the execution schedule | No observation leakage found in the runner |

## Reset, timing, and success contract

- The runner loads an official LIBERO initial state, then performs 10 no-op settling actions `[0,0,0,0,0,0,-1]` before recording the evaluated episode.
- The control frequency is 20 Hz and the hard episode limit is 280 evaluated control steps.
- Success is read from the environment’s task success checker; the runner terminates on success. The wrapper’s `truncated` value is not used as an independent condition, but the for-loop enforces timeout.
- Seeds are `1000 + init_state_id`. There is exactly one seed for each official initial state in the static grids. State effects and seed effects therefore cannot be decomposed.
- The demonstration dataset is 10 Hz. Thus an offline temporal offset of 16 spans 1.6 s, whereas a 16-step rollout commitment spans 0.8 s. Step-index equality is not physical-time equality.

## Fixed executor trace

The audited path is:

```text
environment observation
  -> LIBERO processor (images/state)
  -> ACT inference (100 x 7 normalized/postprocessed chunk)
  -> FixedChunkExecutor
  -> select group-local chunk row(s)
  -> concatenate one 7-D action
  -> LIBERO take_action / robosuite controller
```

For a global horizon (h), the executor installs one fresh chunk and executes rows (0,\ldots,h-1), then re-queries.

For group horizons \((h_a,h_g)\):

1. If either group has expired, ACT is queried once for a fresh full chunk.
2. Only expired groups install their slice of the fresh chunk.
3. Nonexpired groups retain their old slice and advance their own row cursor.
4. The composed action may mix arm and gripper predictions made from different observations and at different positions inside their source chunks.

Consequences:

- A diagonal schedule `(h,h)` is mathematically equivalent to the global schedule (h). Raw traces validate the diagonal controls.
- Query cadence is the union of group expirations. For `(4,16)`, the four-step arm usually determines ACT call frequency, so it is query-matched to global 4, not global 16.
- Group-wise execution changes more than a scalar horizon: it introduces cross-generation action composition. Any gain or loss can arise from that composition, not only “group timescales.”
- A phase change does not force a fresh query in Gate-2B. The old active chunk can cross the phase boundary until its current commitment expires.

No off-by-one, noncontiguous step index, incorrect row cursor, hidden reset, query-schedule, or diagonal-equivalence failure was found in the canonical complete LIBERO traces. The initial pilot summaries without raw logs remain non-reproducible.

## Selective-commitment executor differences

The later matched-query experiment queries ACT at fixed (q\in\{4,8,16\}). At each query it compares the current stale action with the fresh chunk’s first action per group, accepts the fresh group if normalized distance is greater than one, and otherwise retains the old group source. This rule:

- is not the earlier binary reliability estimator;
- can mix source generations;
- clamps to a source chunk’s last row after exhaustion rather than forcing a query; and
- is exactly query-matched to the global-replace control.

The raw negative result therefore applies to this specific rule and execution behavior.

## RoboTwin action contract

Current upstream `_base_task.take_action` slices a 14-D `qpos` command as:

```text
left_arm[0:6] | left_gripper[6] | right_arm[7:13] | right_gripper[13]
```

The action is an absolute joint/gripper target. Environment-side TOPP/interpolation can turn a single high-level target into a variable number of physics steps. This differs fundamentally from LIBERO’s relative 20 Hz Cartesian delta contract.

What remains missing:

- exact upstream commit used by each historical attempt;
- exact policy checkpoint and output convention;
- normalization/postprocessing contract;
- proof that the policy emits absolute qpos in this order;
- completed rollout traces linked to that complete contract.

Accordingly, the reported ordering is **verified only for the current upstream environment code**, while the project’s historical RoboTwin policy-to-environment contract is **NOT REPRODUCIBLE**. This is a stop condition for RoboTwin-based claims, not for the separately verified LIBERO audit.

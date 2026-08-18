# LIBERO / current LeRobot execution audit

This audit records the first native LIBERO runtime used for the One Clock
Gate-0 path. The external LeRobot checkout is
`/home/thor/projects/embodied_lab/third_party/lerobot`, commit
`f66e5128ecb2456e8c54a63d15404fa59c16aebc`. The environment uses LeRobot's
current `LiberoEnv` integration and the `hf-libero` package; no LIBERO source
was copied into this repository.

## Verified task and environment

The selected task is `libero_object` task 0,
`pick_up_the_alphabet_soup_and_place_it_in_the_basket`. It is a standard
Object-suite manipulation task with approach, grasp, transport, and placement
phases, and has the official LIBERO initial states used by the environment.

The verified host/runtime was Linux `aarch64` on NVIDIA Jetson Thor, Python
3.12.3, PyTorch `2.11.0+cu130` with CUDA `13.0`, MuJoCo `3.8.1`, LeRobot
`0.6.2`, and `hf-libero` `0.1.4`. The policy probe and rollouts used
`MUJOCO_GL=egl` and the NVIDIA Thor GPU.

`lerobot.envs.libero.LiberoEnv` constructs LIBERO's
`OffScreenRenderEnv`. With `MUJOCO_GL=egl`, the environment was instantiated,
reset, stepped with the official dummy action, and closed on the Jetson Thor.
The runtime action space was `Box(-1, 1, (7,), float32)`. The selected control
mode is `relative`; current source sets each robot controller's `use_delta` to
`True` for this mode.

The direct environment reset returned:

```text
pixels.image       (256, 256, 3) uint8
pixels.wrist_image (256, 256, 3) uint8
robot_state.eef.pos       (3,)
robot_state.eef.quat      (4,)
robot_state.eef.mat       (3, 3)
robot_state.gripper.qpos (2,)
robot_state.joints.pos    (7,)
```

The actual raw state also contains gripper velocity, joint velocity, and the
other fields shown by `LiberoEnv.observation_space`; the current LeRobot
`LiberoProcessorStep` uses eef position, quaternion, and gripper qpos.

## Policy contract

The public checkpoint used for the runtime probe is
`zeromidnight/act_libero_object`, downloaded locally at
`/home/thor/projects/checkpoints/zeromidnight_act_libero_object`. Its saved
LeRobot processor files were loaded with the checkpoint. The checkpoint has
two 256x256 visual inputs and an 8-D state input. The environment camera names
are mapped to the checkpoint's `image` and `wrist_image` keys.

The checkpoint is a public pretrained ACT artifact whose recorded training
dataset is `DorayakiLin/libero_object_25_08_23_lerobotv2.1`; no training was
run for this Gate-0 task.

The current LeRobot path is:

```text
raw LiberoEnv observation
  -> lerobot.envs.utils.preprocess_observation
  -> LiberoProcessorStep
  -> saved ACT policy preprocessor
  -> ACTPolicy.predict_action_chunk
  -> saved ACT policy postprocessor
  -> LiberoEnv.step
```

The direct non-vector observation helper does not batch nested `robot_state`
arrays, so the runner adds a batch dimension to those arrays before calling
the official helper. This is the only LIBERO-specific observation adaptation.
The verified policy-facing tensors are two `(1, 3, 256, 256)` images and
`(1, 8)` state. The direct ACT call returned `(1, 100, 7)`; postprocessing
returned a finite `(1, 100, 7)` chunk, and its first 7-D action was accepted by
`LiberoEnv.step`.

The action semantics are the current LIBERO contract:

```text
action[0:6]  end-effector control
action[6]    gripper control
```

The first Gate-0 groups therefore are exactly `arm: [0..5]` and
`gripper: [6]`. No finer grouping is used. The current policy checkpoint has
temporal ensembling disabled. The runner calls `predict_action_chunk` directly
and never calls LeRobot's queued `select_action`, so `FixedChunkExecutor` owns
all commitment decisions.

## Integration point and logging

The narrow integration point is immediately after the policy postprocessor
has produced a full environment-action chunk and immediately before the
executor composes the action passed to `LiberoEnv.step`. `global_fixed` and
`groupwise_fixed` are the existing `FixedChunkExecutor` strategies; the
runner does not add another action queue.

Each output `steps.jsonl` record is the executor's existing decision log. It
contains environment step, query flag, chunk/source IDs, source position and
age for both groups, refreshed groups, configured horizons, and the composed
7-D action. `summary.json` includes environment steps, policy queries,
queries per episode, query rate, success, and mean source age by group.

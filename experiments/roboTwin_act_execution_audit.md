# RoboTwin 2.0 / XPolicyLab ACT execution audit

Audit date: 2026-08-18

Upstream source used:

- RoboTwin `266f3aadf505a4f7fe9af0faa41a20f5f47cd123`
- XPolicyLab submodule `c37109c500be67d0dea6b36bf7337bbd26e763cd`
- Local checkout: `/home/thor/projects/upstreams/RoboTwin`

The audit is source-verified. A live simulator/model inspection was not
available in this session because the workspace has no ACT checkpoint and the
current Python environment lacks PyTorch, SAPIEN, mplib, and websocket
dependencies.

## Verified path

1. `scripts/eval_policy_xpolicylab.py::eval_remote_policy` obtains an
   observation with `task_env.get_obs()`, converts it with
   `robotwin_obs_to_xpolicylab`, calls the policy client's `update_obs`, calls
   `get_action`, and passes each returned action to
   `task_env.take_action(...)`. The loop updates the observation between
   returned actions.
2. `XPolicyLab/policy/ACT/model.py::Model.update_obs` converts each RoboTwin
   observation into ACT's camera tensors and packed `qpos` using
   `pack_robot_state`. `Model.get_action` calls the ACT adapter and unpacks the
   selected action using `unpack_robot_state`.
3. `XPolicyLab/policy/ACT/detr/act_policy.py::ACT.get_action` runs the loaded
   ACT model at its query frequency and stores the complete model output in
   `self.all_actions`. It normally returns only the selected current row after
   post-processing. With `chunk_size: 50` and the joint setting, the internal
   full output is `(1, 50, 14)` and one selected action is `(1, 14)`. With
   aggregation disabled, the ordinary upstream cursor consumes all 50 rows
   before its next policy query.
4. The current `XPolicyLab/policy/ACT/deploy.yml` sets `temporal_agg: true`.
   That changes `query_frequency` to one control step and combines available
   predictions through `all_time_actions` with exponential weights. This is
   temporal ensembling, so the Gate-0 runner explicitly sets it to `false` for
   both strategies. With aggregation disabled, the upstream ACT cursor would
   normally consume one sequential action per query chunk.

## Verified action semantics

The selected Gate-0 setting is `env_cfg/aloha_agilex.yml`, whose robot is
`aloha_agilex`. `env_cfg/robot/_robot_info.json` declares two arms with six
arm dimensions and one end-effector dimension each. The default clean task
configuration selects the `aloha-agilex` embodiment.

The semantics and ordering come from packing and environment code, not from
the width alone:

```text
packed joint action: [left arm joints 0..5,
                      left gripper 6,
                      right arm joints 7..12,
                      right gripper 13]
```

`XPolicyLab/utils/process_data.py::pack_robot_state` constructs the dual-arm
joint state in left-arm/left-end-effector then right-arm/right-end-effector
order. `scripts/eval_policy_xpolicylab.py::xpolicylab_action_to_robotwin`
constructs the same flat order for dictionary actions. Finally,
`envs/_base_task.py::take_action` splits a qpos action into the left arm,
left gripper, right arm, and right gripper paths before sending them to the
two robot arms. Thus this setting is joint-space absolute target execution,
not end-effector-pose execution.

## Buffering, commitment, and integration point

There is no separate RoboTwin environment action buffer. ACT owns its
prediction buffer (`all_actions`, or `all_time_actions` when temporal
aggregation is enabled), and the official client receives the selected row.
The narrow external integration point is immediately before
`task_env.take_action`: the Gate-0 runner calls the official ACT object once
to obtain its already-produced full chunk, then `FixedChunkExecutor` selects
or composes the row that is sent to RoboTwin. Observation conversion and ACT
pre/post-processing remain upstream code.

The runner deliberately reads `ACT.all_actions` after the official
`ACT.get_action` call with its cursor reset to zero for an explicit query. It
does not reimplement the ACT network or temporal aggregation. This is needed
because the current XPolicyLab adapter exposes only one selected row to its
client.

## Gate-0 grouping

The first grouping follows the verified physical schema:

- `left_arm`: indices `[0, 1, 2, 3, 4, 5]`;
- `left_gripper`: index `[6]`;
- `right_arm`: indices `[7, 8, 9, 10, 11, 12]`;
- `right_gripper`: index `[13]`.

`global_fixed` assigns one horizon to all four groups. `groupwise_fixed`
refreshes only groups whose configured commitment has expired; each group keeps
its source chunk ID and position until then. Per-step JSONL records include the
query, new generation, source generation, age, source position, and remaining
commitment for every group, refreshed groups, horizons, and the composed action.

## Current resource status

No official demonstrations, ACT checkpoint, or runnable simulator stack was
available in this session. Therefore no RoboTwin rollout or fixed-vs-groupwise
pilot result is claimed. After installing the official RoboTwin/XPolicyLab
dependencies and supplying a real checkpoint, run the README command first for
one global horizon, then repeat it for `1`, `2`, `4`, `8`, and `16` (only when
the checkpoint chunk length supports the value), followed by the small
group-wise diagnostic in the config.

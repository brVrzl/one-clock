# RoboTwin frozen-policy provenance and contract audit

Status: `blocked_before_cache_generation`.

## Primary candidate

- Repository: `lerobot/smolvla_robotwin`
- Exact revision: `967623a0f38c7e1236c66b3893c830398d793ff7`
- Pinned `config.json` SHA256: `d7da5f3281abb825fb917e64c028a536279a9cc6e9eb64c28224ff899e6f11c1`
- Pinned `policy_preprocessor.json` SHA256: `7f8edfe4b9477706facccb44748abc2dc58a35285daa9c62f7348f66001d3a75`
- Model type: `smolvla`
- Public config: 0.5B-class VLA, `output_features.action.shape=[14]`
- `chunk_size=50`, `n_action_steps=50`, `n_obs_steps=1`
- `num_steps=10`, `use_cache=true`, `normalization_mapping=VISUAL:IDENTITY,
  STATE:MEAN_STD, ACTION:MEAN_STD`
- Camera mapping in the pinned preprocessor:
  `cam_high -> camera1`, `cam_left_wrist -> camera2`,
  `cam_right_wrist -> camera3`
- Exact state declaration in the pinned config/preprocessor: shape `[6]`.

## Target dataset contract

At `lerobot/robotwin_unified@1287871839fae2296bc27b88a5457c3e1eba8e1f`,
`meta/info.json` declares state and action shape `[14]` with the motor ordering
stored in `group_schema.json`. The three camera keys and 480x640 AV1 video
features are present.

## Structural result

The checkpoint's saved state normalizer also declares a six-element state. The
LeRobot adapter has no state selection/projection step in the pinned
preprocessor; `SmolVLAPolicy.prepare_state` only pads its input to width 32.
Therefore passing the target dataset's 14-D state through the checkpoint's
normalizer is a shape/semantic contract mismatch. The cache builder refuses to
truncate, reorder, or invent a six-dimensional state. The model weights were
not downloaded and no invalid cache was generated.

This is a structural integration failure, not a low smoke success result.

## Fallback audit

The available fallback `lerobot/hy_vla_robotwin` is pinned at
`6f33e74a97a6e1197d370e9a8c59a0b5e33bf39b`. Its released conversion documents
an action representation of `relative_absolute`, a token horizon of 40, a
physical horizon of 20, 7 executed actions, and six image-history frames at
interval 5. It is not interchangeable with SmolVLA. Its public conversion
also targets a released 16-D dual-arm pose/gripper representation, whereas the
preferred target dataset is a 14-D joint representation. It is therefore
audited but not silently selected.

The Tencent author checkpoint is `tencent/Hy-Embodied-0.5-VLA-RoboTwin@bd7bba6f5934ad62293a2a34f74760c6a3ef2ff8`; it has the same family-specific
representation caveat. No fallback cache was generated.

## What remains required

Provide a checkpoint revision whose declared input state and action contract
match the 14-D target dataset, or provide the documented checkpoint adapter
that maps 14-D dataset state to the model's six-D state. Until then, a full
50-task policy-response cache would be invalid.

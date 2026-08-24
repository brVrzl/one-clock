# Gate-4A Spatial asset audit

Audit time: 2026-08-24T20:01:59+08:00

Status: **STOP — THE NAMED SPATIAL CHECKPOINT HAS NO CHECKPOINT FILES**

No scientific rollout, task-success inspection, state selection, schedule,
preregistration, or method outcome was generated.

## Checkpoint repository

- Hugging Face repository: `zeromidnight/act_libero_spatial`
- Requested revision: `main`
- Immutable repository revision: `73644a66b54689e4602e5d4f6a4d62a04c3f406f`
- Repository history: one commit (`initial commit`, 2025-12-16), one branch
  (`main`), no tags, and no conversion refs.
- Complete file inventory at the immutable revision: `.gitattributes` only
  (1,519 bytes; Git blob `a6344aac8c09253b3b630fb776ae94478aa0275b`).

The repository contains no `model.safetensors`, policy `config.json`,
preprocessor, postprocessor, normalization/statistics, training config, or
model card. Consequently, no checkpoint/model file can be used and no model,
policy-config, preprocessor, postprocessor, or normalization SHA256 exists for
this candidate revision.

## Critical compatibility audit

All checkpoint-dependent fields are unresolvable: ACT implementation/version,
action dimension and ordering, chunk size, `n_action_steps`, observation and
image keys, state dimension, output/action keys, normalization modes,
checkpoint dtype, and stored temporal-ensemble configuration. The policy
cannot be instantiated, complete action chunks cannot be retrieved, and the
ability to disable policy-internal temporal ensembling or issue deterministic
per-step queries cannot be established.

This fails the zero-trust checkpoint gate. In particular, the required facts
`action_dim == 7`, arm `action[0:6]`, gripper `action[6]`, and
`chunk_size >= 21` cannot be verified. The action contract is not reinterpreted
and no substitute checkpoint is authorized.

## Dataset repository resolved before the stop

- Hugging Face repository: `zeromidnight/libero_spatial_lerobot_v3.0`
- Requested revision: `main`
- Immutable repository revision: `38927e939de5d2bfd40effcf27d16710aea6f864`
- Remote inventory: one data Parquet, one episode-metadata Parquet,
  `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, and two MP4 camera
  streams.

The dataset was not downloaded or promoted into an evaluation contract because
the checkpoint gate had already failed. Dataset metadata is not being used to
infer controller timing.

## Local software context (not an approved rollout contract)

- LeRobot checkout: `f66e5128ecb2456e8c54a63d15404fa59c16aebc`, clean; package reports `0.6.2`
- Python 3.12.3; PyTorch 2.11.0+cu130; CUDA runtime 13.0
- MuJoCo 3.8.1; robosuite 1.4.0; Gymnasium 1.3.0
- NVIDIA driver 595.84; three visible NVIDIA GeForce RTX 5080 devices
- OS: Linux 6.8.0-136-generic x86_64, glibc 2.35
- Hugging Face Hub client 1.28.0
- Model device/dtype: not applicable because no model could be instantiated
- Rendering backend and LIBERO commit/version: not promoted or frozen because
  execution stopped at the checkpoint gate

The machine-readable counterpart is
[`audit_outputs/gate4a_spatial_asset_audit.json`](audit_outputs/gate4a_spatial_asset_audit.json).


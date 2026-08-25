# Overnight RTX 5080 preparation — 2026-08-24

Operational-only preparation record. No Gate-4A2/Gate-4A2 scientific rollout,
task-success evaluation, manuscript edit, Gate-3C edit, or preregistered
experiment change was performed.

## Protected state

- Frozen base: `f7cb0559594a15a7ab4ab92758fad766d841af04`
- Ops branch: `ops/overnight-5080-prep-20260824`
- Protected scientific branches were not checked out for work.
- Large assets and logs are outside the Git repository under
  `/home/wjq/research-assets/overnight_20260824/` and sibling roots.

## Machine

- Host: `xdl-MS73-HB1-000`, Ubuntu 22.04.5, dual Xeon Platinum 8581C,
  188 GiB RAM.
- Three RTX 5080 GPUs, 16,303 MiB each. GPU 0 was reserved for an existing
  scientific `maniskill_act` process; GPU 1 ran the beat reference and GPU 2
  ran the adjust reference. Neither preparation run contended with GPU 0.
- Root filesystem started at 8%, peaked at the observed 17% used, and stayed
  below the 85% ceiling during the recorded work.

## Completed preparation

- Spatial checkpoint and dataset snapshots were downloaded and hashed; see
  `overnight_5080_asset_manifest.json` and external manifests.
- Checkpoint provenance metadata was mirrored. Its config names
  `HuggingFaceVLA/libero`; metadata reports 40 tasks, so the operational label
  is `MULTI_SUITE`.
- Official RoboTwin main checkout and pinned XPolicyLab submodule were cloned;
  source provenance is external at
  `overnight_20260824/manifests/robotwin_source_provenance.json`.
- Official selected `demo_clean` data for `beat_block_hammer` and
  `adjust_bottle` was downloaded, verified, and preprocessed through the
  pinned ACT scripts, 50 trajectories per task.
- XPolicyLab scheduler dry-run passed for the two-task, GPU 1/2 pool. ACT
  adapter and policy-server scripts are present.
- The `beat_block_hammer` ACT reference completed the official 6000-step
  protocol on GPU 1 and passed an offline one-observation contract audit.
- The independent `adjust_bottle` ACT reference completed the same official
  6000-step protocol on GPU 2 and passed an offline one-observation contract
  audit. Both runs are infrastructure-only and not paper evidence.
- The pinned source catalog lists 50 official `demo_clean` tasks. All 50
  official archives and extracted task directories are now present at immutable
  revision `a967b852afa21a9cbf19a198f7e653109042e87c`; the 7,650-file hashed
  manifest is external. This raw-data extension remains separate from the
  two-task ACT preprocessing/training path.

## Blockers

The external `STATE.json` is authoritative. Current blockers are the missing
RoboTwin `objects`/background asset archives after repeated transient HF TLS
failures, CuRobo compilation requiring an unavailable CUDA toolkit, and the
optional full 34.9 GB `HuggingFaceVLA/libero` mirror stopping after a preserved
partial download. The first and third blockers are retryable network blockers;
the CuRobo toolkit blocker requires human/system provisioning.

## Labels

Any ACT checkpoint produced by this queue is `INFRASTRUCTURE-ONLY` and
`NOT-PAPER-EVIDENCE`.

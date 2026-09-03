# Runtime/source provenance audit and R1D corrective note

Audit date: 2026-09-03 (Asia/Shanghai)

Status: `FROZEN_TECHNICAL_AUDIT_BEFORE_R1A_R1B_R1C_R1D_B3_UNBLINDING`

This note supplements, and does not erase, the historical records in
`TEMPORAL_CONTRACT_AUDIT_SUPERSEDING_20260903.md`,
`REPRODUCIBILITY_TIMEBASE_NOTE.md`, and
`PRE_UNBLINDING_TECHNICAL_DISPOSITION_20260903.md`. In particular, the original
R1D launch selected the source checkout described below and failed before any
scientific cell. Any completed repaired R1D cell instead uses the installed
LeRobot 0.4.4 package described below.

No R1, R2, or B3 scientific outcome payload was opened for this audit.

## Track A, including TE_DENSE

- Interpreter: `/home/wjq/workspace/venvs/libero_act/bin/python` (Python 3.12.3).
- Actual LeRobot package: installed LeRobot 0.4.4 at
  `/home/wjq/workspace/venvs/libero_act/lib/python3.12/site-packages/lerobot`.
- `lerobot.__file__`:
  `/home/wjq/workspace/venvs/libero_act/lib/python3.12/site-packages/lerobot/__init__.py`.
- `ACTTemporalEnsembler` and ACT inference:
  `.../site-packages/lerobot/policies/act/modeling_act.py`.
- ACT configuration/loading:
  `.../site-packages/lerobot/policies/act/configuration_act.py`,
  `.../site-packages/lerobot/configs/policies.py`, and
  `.../site-packages/lerobot/policies/factory.py`.
- LIBERO construction:
  `.../site-packages/lerobot/envs/configs.py`,
  `.../site-packages/lerobot/envs/factory.py`, and
  `.../site-packages/lerobot/envs/libero.py`.

`launch_track_a.sh` invokes that interpreter without setting `PYTHONPATH`.
`run_track_a.py` prepends only the project `src` directory and its own
experiment directory; neither contains a `lerobot` package. The environment has
no LeRobot editable-install `.pth` or `direct_url.json`. Its only unrelated
editable path is `/home/wjq/workspace/upstreams/verl-vla/src`, which does not
contain or shadow `lerobot`. Thus Track A used pinned pip/site-packages LeRobot
0.4.4, not `/home/wjq/workspace/upstreams/lerobot/src`.

The actual 0.4.4 `ACTTemporalEnsembler` implements coefficient 0.01 with chunk
length 100 and assigns temporal weight index zero to the oldest available
prediction. A positive coefficient intentionally weights older predictions
more strongly. The existing runtime canary in `te_dense_audit.json` passes this
direction and the all-seven-dimension normalized-space aggregation contract.

## R1A, R1B, and R1C

All three phases used the same interpreter and installed LeRobot 0.4.4 paths as
Track A. `master_pipeline.sh` launches `run_queue.py` without `PYTHONPATH` or a
source-checkout argument. The only conditional checkout insertion in
`run_queue.py` is guarded by `args.phase == "r1d"`; it is not reached for R1A,
R1B, or R1C.

R1A--R1C did not use only a hand-imported ACT leaf. They imported
`lerobot.configs.policies`, `lerobot.envs.configs`, `lerobot.envs.factory`, and
`lerobot.policies.factory` from the installed 0.4.4 package. That package's
policy imports are compatible with Transformers 4.51.3. The asymmetry is that
R1D redirected the `lerobot` package root to the newer checkout before
constructing `Runtime`.

## Original R1D launch and import failure

- Interpreter: `/home/wjq/workspace/venvs/libero_act/bin/python`.
- Selected package root:
  `/home/wjq/workspace/upstreams/lerobot/src/lerobot`.
- Source checkout commit at launch:
  `f66e5128ecb2456e8c54a63d15404fa59c16aebc`.
- Transformers: installed version 4.51.3.

The exact failing chain was:

`run_queue.Runtime` -> `lerobot.policies.factory` -> parent package
`lerobot.policies.__init__` -> `lerobot.policies.eo1.configuration_eo1` ->
`transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLTextConfig`.

Transformers 4.51.3 does not export that name. EO1 and its Qwen VLM
configuration are not used by the frozen ACT evaluator. The failure was caused
by eager package-registry initialization, not by ACT checkpoint loading,
inference, preprocessing, or execution.

## Relevant source comparison

The installed 0.4.4 package and `f66e5128...` checkout differ materially in one
project-facing runtime area: the checkout explicitly forwards
`LiberoEnv.fps` as `OffScreenRenderEnv(control_freq=...)`, while 0.4.4 omits
that argument. For the frozen value 20, both resolve to exactly 20 Hz because
LIBERO 0.1.1 / robosuite 1.4.0 defaults to 20 Hz. The checkout also defers
environment construction and reorganizes factories/registries; those changes
do not alter the single synchronous R1D action contract.

ACT configuration fields, action delta indices, `select_action`,
`predict_action_chunk`, and `ACTTemporalEnsembler` inference are unchanged on
the relevant path. The `modeling_act.py` differences are import organization
and training-loss handling, which is unreachable during evaluation. Both
factory paths instantiate `ACTPolicy` from the same frozen checkpoint and load
its serialized normalization processors. The 0.4.4 path is also the path that
successfully completed R1A--R1C.

## Mutation check

R1C completed at `2026-09-03T14:32:54.280459740+08:00`; R1D launched at
`2026-09-03T14:32:54.587455483+08:00`. No relevant package, package metadata,
checkout file, environment `.pth`, or project execution source changed in that
interval. LeRobot 0.4.4 and Transformers 4.51.3 were installed on 2026-08-26.
The source checkout has remained clean at `f66e5128...` since 2026-08-24. The
launchers set no `PYTHONPATH`; no LeRobot editable install is present.

## Temporal-contract consequence

The authoritative physical-time conclusions remain valid:

- ACT policy index = 0.05 s;
- R1A--R1D evaluator step = 0.05 s;
- `d=20` = 1.00 s;
- `q+k=t` is physically same-target.

No material source difference threatens these conclusions.

## R1D repair disposition

Disposition: `SCIENTIFICALLY_NEUTRAL_IMPORT_PATH_REPAIR_AVAILABLE`.

`r1d_runtime_repair.py` preloads only the installed LeRobot 0.4.4 package root
before invoking the unchanged frozen `run_queue.py`. Python consequently
resolves the ACT and LIBERO submodules from the same package used by R1A--R1C,
and the frozen runner's later checkout path insertion cannot redirect them.
This avoids the unused EO1/VLM registry import. It changes no manifest,
checkpoint, executor, source mapping, policy code, processor, seed, condition,
state, action contract, control frequency, or governed execution source.

The stronger step-level deterministic comparison is
`REFERENCE_SEQUENCE_UNAVAILABLE`: the existing R1D portability canary is a
static provenance/source-map canary and does not persist a technical-canary
reference action trajectory. No new rollout will be created to manufacture
one.

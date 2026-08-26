# LIBERO baseline closeout and handoff

Date: 2026-08-26  
Scope: LIBERO only. This handoff closes the current research-development
session; it does not authorize another rollout, DCTA training run, or temporal
aggregation intervention.

## A. Current LIBERO environment

### Repository and software

- Repository: `/home/wjq/workspace/one-clock`.
- Upstream ACT recipe checkout: `/home/wjq/workspace/upstreams/verl-vla`,
  commit `856c1d747c19695caaf66f55431c57b27e3c9d8e`.
- Isolated Python environment: `/home/wjq/workspace/venvs/libero_act`.
  Installed versions include LeRobot 0.4.4, LIBERO 0.1.1, PyTorch
  2.7.1+cu128, torchvision 0.22.1+cu128, torchcodec 0.4.0, and
  transformers 4.51.3.
- LIBERO simulator assets and init files are the isolated LIBERO package assets
  under `/home/wjq/workspace/venvs/libero_act/lib/python3.12/site-packages/`.
  The track-specific paths are in
  `configs/libero_dcta/libero/config.yaml`.
- Hardware assumption: the machine has three RTX 5080 GPUs (16,303 MiB each).
  LIBERO jobs must check occupancy immediately before launch and use only a
  GPU that remains free. At closeout no LIBERO GPU process is running.

### Rendering, observations, and actions

- Evaluation uses LIBERO `OffScreenRenderEnv` with MuJoCo EGL rendering,
  256x256 `agentview` and `robot0_eye_in_hand` RGB observations, and no depth.
  The verl-vla adapter rotates each image by 180 degrees.
- The policy state is 8-D: end-effector xyz, end-effector axis-angle, and the
  two gripper positions. Actions are 7-D relative OSC-POSE commands: six arm
  dimensions followed by one gripper scalar.
- The simulator control rate is 20 Hz. The selected LeRobot image datasets are
  stored at 10 Hz and use gripper values in `[-1,+1]`; the rejected NVIDIA
  conversion uses a different gripper/rate convention and is not a baseline
  source.
- ACT normalization is the checkpoint-native mean/std preprocessor and affine
  postprocessor. Aggregation is performed in normalized action space before
  the native postprocessor.
- The frozen evaluator contract is a 512-decision episode, 10 zero-action
  stabilization steps after reset, and the official LIBERO reset states. The
  common evaluator/debug implementation is
  `scripts/run_libero_dcta_rollouts.py`; no rollout was completed in this
  session.

## Public ACT reference search

| Source | Suite | Checkpoint | Public weights | Published success | N episodes | Temporal aggregation | Protocol match | Authority level |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| [verl-vla official ACT recipe](https://verl-vla.readthedocs.io/en/latest/fine-tuning/act/official-libero-spatial.html) | Spatial | Recipe export at step 8,275 | No. The checked-in `assets/hf_models/act_libero` contains an architecture/ImageNet-backbone initializer, not the trained export. | 85/100; per-task 10, 9, 9, 9, 9, 10, 10, 6, 8, 5 | 100 | Disabled in the documented export (`chunk_size=10`, `n_action_steps=10`, `temporal_ensemble_coeff=null`) | Environment, data, task identities, training, and horizon match; native temporal-ensemble contract does not | Official framework release |
| [LeRobot ACT implementation](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act) | No suite-specific result | N/A | Implementation is public; no authoritative standard-LIBERO ACT checkpoint was found | N/A | N/A | Native online exponential temporal ensemble, coefficient 0.01, queried each step | Defines the canonical aggregation semantics used here | Official framework release |
| [LeRobot LIBERO integration](https://huggingface.co/docs/lerobot/libero) | Spatial, Object, Goal, Long | No ACT checkpoint listed | No ACT weights/results | N/A | Recommends 10 episodes/task | Policy-dependent | Environment/evaluation reference only | Official framework release |
| `ishandotsh/act_libero_spatial_test`, `zeromidnight/act_libero_object`, `Harrysunshine/act-libero-goal` | Spatial, Object, one Goal task | Community checkpoints | Yes | No complete authoritative protocol/result satisfying this baseline | Not established | Metadata/protocol mismatch or single-task scope | Diagnostic only |

Search conclusion: no public trained ACT checkpoint was found that provides
weights, suite identity, closed-loop success, and a reproducible native ACT
temporal-aggregation protocol at primary-evidence quality. No released copy of
the verl-vla step-8,275 Spatial export was found in its repository, releases,
linked storage, or the public Hugging Face ACT/LIBERO index. Object and Goal
also have no authoritative public ACT weights/results satisfying these fields.
The correct next step is therefore a normal public baseline evaluation or a
reproduction of the authoritative training recipe, not adoption of an
unverified community checkpoint.

## B. Standard ACT status

### Canonical temporal aggregation

`STANDARD_ACT` is defined to preserve native ACT temporal ensembling. The
policy is queried at every simulator decision and predicts a normalized chunk
of 10 actions. At decision `t`, valid same-target candidates are the chunk
elements from query sources `q=max(0,t-9),...,t`; candidates are ordered oldest
to newest and receive native logits `-0.01*i`, with `i=0` for the oldest valid
candidate. A softmax over the valid candidates produces the executed action.
The history is reset on every environment reset, so warmup has one candidate at
decision zero and grows to ten. This is the LeRobot `ACTTemporalEnsembler`
semantics, not NEWEST-only, hard source selection, hold, or EMA.

The numerical equivalence tests against the upstream ensembler passed during
development (8 local DCTA/contract tests passed). The implementation and
contract are retained in `src/one_clock/libero_dcta.py`,
`tests/test_libero_dcta.py`, and
`research/libero_dcta/standard_act_contract.md`.

### Checkpoint and reproduction status

- No trained ACT checkpoint is currently available for a trustworthy local
  Standard ACT success claim.
- The initial one-step Spatial smoke was launched while a GPU was free, but it
  terminated before training because the isolated PyTorch CUDA 12.6 build did
  not support the RTX 5080 `sm_120` capability. The isolated environment was
  corrected to PyTorch CUDA 12.8; compatibility was verified, but the guarded
  follow-up smoke never launched because the GPUs remained occupied by another
  session and was stopped cleanly at closeout.
- Spatial data are complete at the pinned revision
  `d86c0b94922572b3b657e1d1a3d01f0952ddeb46` (432 episodes, 52,970 frames).
  Object (`e1e080d7df1d0a359dff5c86c222e047549f447f`) and Goal
  (`91a97115558b5b611200a432d9c82e4f30991b60`) downloads are partial and must
  not be treated as training inputs until completed and checked. At closeout,
  the local partial directories contained approximately 4.4 GiB/50 parquet
  files for Object and 5.9 GiB/68 parquet files for Goal.
- No suite has a local Standard ACT success rate, per-task success table, or
  public-reference delta from this session. The documented 85/100 Spatial
  number is an open-loop recipe diagnostic, not a native temporal-ensemble
  result and is not directly comparable to the requested Standard ACT result.
- No DCTA candidate extraction, gate training, or rollout-success evaluation
  was completed. DCTA, SHARED_DYNAMIC_AGG, and STANDARD_ACT code/tests and
  evaluator/debug scripts are preserved for a later session; no held-out DCTA
  error or rollout result exists.

### Job state at closeout

- LIBERO GPU jobs: none running. The failed CUDA 12.6 smoke is completed with
  no checkpoint. The CUDA 12.8 guarded smoke was waiting and was stopped
  cleanly without launching. Object/Goal downloads were stopped cleanly after
  their current files were written; partial files remain.
- No Python worker from this LIBERO session remains. No active DCTA or temporal
  experiment is pending.
- The separate RoboTwin workers were not killed or preempted. At the final
  process/GPU check they were no longer visible and `nvidia-smi` reported no
  compute processes; this handoff makes no claim about their later state.

## C. SmolVLA readiness

LeRobot 0.4.4 is installed in `/home/wjq/workspace/venvs/libero_act`. SmolVLA
dependencies and a working SmolVLA LIBERO evaluation were not separately
verified, and nothing was installed or evaluated during closeout. The next
session should treat the public checkpoint
[`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero) as a
fast baseline-fidelity candidate, using its native evaluation contract.

## D. Known LIBERO pitfalls

Only observed or already documented issues are listed here:

1. Historical Gate-4A2 task 4 showed reset-state nondeterminism. Gate-4A2 was
   permanently invalidated before outcome analysis and is not being rerun or
   rescued.
2. The prior LIBERO Spatial attempt showed task 3 observation/render
   nondeterminism and was permanently invalidated. Its sealed success outcomes
   are not reused.
3. The first ACT smoke used a CUDA 12.6 PyTorch build without RTX 5080 `sm_120`
   support and failed before training. The isolated environment now uses the
   CUDA 12.8 build.
4. The official verl-vla Spatial 85/100 recipe evaluation disables native
   temporal aggregation, whereas this track's Standard ACT contract enables
   it. Mixing those protocols would create a false fidelity claim.
5. The selected official image datasets are 10 Hz and use `[-1,+1]` gripper
   actions. The downloaded NVIDIA conversion labels Object/Goal as 20 Hz with
   `[0,1]` gripper actions and is rejected for this baseline.
6. Dataset `task_index` order is a permutation of LIBERO benchmark task-id
   order; the frozen mappings are recorded in
   `research/libero_dcta/standard_act_contract.md` and must be respected in
   per-task reports.

The valid historical Gate-3C frozen diagnostic remains unchanged and is not an
input to this new baseline. Its old hard-source intervention is not the final
DCTA method.

## E. Next clean step

The separate follow-up session should:

1. Run a public standard SmolVLA or other authoritative baseline using its
   native evaluation contract; do not install or evaluate it as part of this
   closeout.
2. Verify reset, rendering, action dimensionality, gripper convention, and
   success detection on a small canary.
3. Obtain a full-suite baseline in a normal public performance range before
   comparing methods. Keep paired reset states/seeds and report per-task and
   per-suite success.
4. Only after that baseline is trustworthy, return to frozen research
   interventions such as Standard ACT versus SHARED_DYNAMIC_AGG versus DCTA.

The next session must perform its own GPU-occupancy check and must not use the
partial Object/Goal downloads until they are complete and verified. No paper
manuscript was modified in this session.

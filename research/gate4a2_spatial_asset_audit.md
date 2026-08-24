# Gate-4A2 Spatial checkpoint, provenance, dataset, and environment audit

Audit time: 2026-08-24T21:03:14+08:00

Status: **PASS — ELIGIBLE FOR PREREGISTRATION, NOT YET FOR OUTCOME GENERATION**

The originally proposed `zeromidnight/act_libero_spatial` candidate failed the
asset gate before any outcome. Its audit at
[`gate4a_spatial_asset_audit.md`](gate4a_spatial_asset_audit.md) and
[`audit_outputs/gate4a_spatial_asset_audit.json`](audit_outputs/gate4a_spatial_asset_audit.json)
is preserved unchanged. No scientific Spatial episode or task-success outcome
was generated during either asset audit.

## Replacement checkpoint identity

- Repository: `ishandotsh/act_libero_spatial_test`
- Immutable revision: `8f04de1472975d62db214238b2fc07e78bde2474`
- Model SHA256: `912f41808962d80ca9084435aa01eccccdd97b7eae3a841c9f4ac71caaf9f8b0`
- Config SHA256: `0e783369890d33a714cef603185c10dff4215328a9862b181eb7f511f3f1a93c`
- Preprocessor SHA256: `8a5df04ea1f67ab515898ba211bc64b6c38020e259bc0bd520ddd7b38a660128`
- Postprocessor SHA256: `c27cf6f42b42352f9b8f9c40da155fd4459e0ee9b85b9f23072941eb52b3ffb5`
- Normalizer and unnormalizer SHA256:
  `a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9`
- Training config SHA256:
  `551dd7bdb8b4ffb109f3ebc40a26856b72953188a74b4a02d597ba2989528b5f`

The complete immutable repository inventory contains `.gitattributes`,
`README.md`, the seven checkpoint/configuration assets above, and
`model.safetensors`; the machine-readable audit lists every file and hash.

Pinned LeRobot commit `f66e5128ecb2456e8c54a63d15404fa59c16aebc`
instantiated the checkpoint as `lerobot.policies.act.modeling_act.ACTPolicy`.
The verified checkpoint contract is:

- action shape `(7,)`, with the six-dimensional OSC arm command at `[0:6]`
  and the one-dimensional Panda gripper command at `[6]`;
- `chunk_size=100`, `n_action_steps=100`, and a complete finite `100×7`
  postprocessed chunk from every fresh query;
- input keys `observation.images.image`, `observation.images.image2`, and
  `observation.state`, with two `3×256×256` images and an 8-D state;
- output key `action`; visual, state, and action normalization mode
  `MEAN_STD`;
- `temporal_ensemble_coeff=null`, no action smoothing, CUDA float32;
- repeated fixed-seed queries on one valid official reset were bit-exact;
- a valid observation-plus-action preprocess/postprocess round trip had maximum
  absolute error `2.98e-8`.

One audited chunk ranged from `-1.09665` to `1.09123`. This is compatible with
the unchanged vanilla LIBERO relative-control stack: the six arm elements are
passed to the OSC controller whose `scale_action` clips to its normalized input
range, while the Panda gripper consumes the sign of element 6. Gate-4A2 adds no
new clipper, smoothing, or postprocessing rule.

## Training provenance: MULTI-SUITE

`train_config.json` names `HuggingFaceVLA/libero`, uses `episodes=null`, and
contains no task filter. Its dataset revision field is null, so the training
command did not store an immutable revision. The audited repository tip is
`86958911c0f959db2bbbdb107eb3e17c5f9c798e`; that tip predates the checkpoint
and is also tag `v3.0`.

The checkpoint normalizer contains count `273,465` for action, state, task
index, and episode index, with task-index range `0..39`. These values and the
action/state statistics match the audited 1,693-episode, 273,465-frame,
40-task dataset (apart from float32 serialization roundoff and the configured
ImageNet visual statistics). Its task strings exactly partition into ten
vanilla tasks from each of LIBERO Spatial, Object, Goal, and LIBERO-10.

The frozen category is therefore **MULTI-SUITE**. The checkpoint must not be
called Spatial-only. The missing revision pin in the training command is
retained as a provenance limitation, but it does not make the suite category
unknown because the checkpoint normalization state independently identifies
the full 40-task corpus.

## Independence from the Object checkpoint

The replacement is a genuinely different checkpoint identity:

- replacement model SHA256 `912f…f8b0` versus Object `3400…9410`;
- replacement normalization SHA256 `a002…17a9` versus Object `3cb9…e941`;
- different config/preprocessor hashes and camera key (`image2` versus
  `wrist_image`);
- different training corpus (unfiltered four-suite corpus versus
  `DorayakiLin/libero_object_25_08_23_lerobotv2.1`), training length (5,000
  versus 100,000 configured steps), batch size, resume status, and output path;
- independent repositories and histories: replacement revision
  `8f04de1…` was uploaded 2025-11-03, while Object revision
  `9cb23a1…` was uploaded 2025-12-16.

Shared ACT architecture and a byte-identical generic postprocessor JSON do not
make the learned weights or normalization assets identical.

## Evaluation dataset audit

- Repository: `zeromidnight/libero_spatial_lerobot_v3.0`
- Immutable revision: `38927e939de5d2bfd40effcf27d16710aea6f864`
- 432 episodes, 52,970 frames, ten task strings exactly matching vanilla
  LIBERO Spatial tasks 0–9
- 8-D state; 7-D relative action ordered as translation `(x,y,z)`, axis-angle
  rotation `(3)`, and gripper `(1)`
- observed action ranges by dimension:
  `[-.9375,-.9375,-.9375,-.1875,-.3675,-.3600,-1]` to
  `[.9375,.9375,.9375,.19714,.33643,.375,1]`; gripper values are exactly
  `{-1,+1}`
- agent and wrist cameras are 256×256 RGB, named
  `observation.images.image` and `observation.images.wrist_image`
- dataset metadata and timestamps are 10 Hz (`dt≈0.1 s`)

The dataset clock is not promoted to the execution contract.

## Vanilla evaluation environment and time contract

The audited runtime uses `libero.libero.benchmark.LIBERO_SPATIAL` and
`libero.libero.envs.problems.libero_tabletop_manipulation.Libero_Tabletop_Manipulation`,
not LIBERO-plus. All ten tasks expose 50 official initial states. The runtime
uses official BDDL/state files, `hard_reset=true`, relative control, two mapped
256×256 cameras (`image`, `image2`), 8-D model state, a 7-D action space, and a
280-step Spatial horizon.

The wrapper and underlying simulator both report `control_freq=20`; the
underlying controller timestep is `0.05 s` and MuJoCo model timestep is
`0.002 s`. The OSC controller has dimension 6, the gripper has one action
dimension, and `controller.use_delta=true`. Therefore the primary frozen
intervention remains `d=20` controller ticks and equals exactly `1.0 s`.

The official vanilla asset snapshot is `lerobot/libero-assets` revision
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`. Rendering is EGL.

## Software contract

- LeRobot `0.6.2`, Git commit
  `f66e5128ecb2456e8c54a63d15404fa59c16aebc`, clean
- `hf_libero 0.1.4`; robosuite `1.4.0`; MuJoCo `3.8.1`
- Python `3.12.3`; PyTorch `2.11.0+cu130`; CUDA runtime `13.0`
- NumPy `2.2.6`; SciPy `1.18.0`; Gymnasium `1.3.0`
- EGL rendering; NVIDIA driver `595.84`
- three visible NVIDIA GeForce RTX 5080 devices; official execution uses
  CUDA device 0
- Linux `6.8.0-136-generic` x86_64, glibc 2.35

The machine-readable counterpart is
[`audit_outputs/gate4a2_spatial_asset_audit.json`](audit_outputs/gate4a2_spatial_asset_audit.json).

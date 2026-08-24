# Repository and Gate-0 audit

Date: 2026-08-24 (Asia/Shanghai)

## Starting state

- Requested working directory `/home/wjq/workspace/icra27_fast_pivot` was initially empty and was not a Git repository.
- Active repository found at `/home/wjq/workspace/one-clock`.
- That shared checkout was clean when inspected, on `exp/fast5080-cross-generation-offline` at `a8d4983`. Another active process subsequently changed its branch, so this sprint uses a dedicated worktree at `/home/wjq/workspace/icra27_fast_pivot`.
- Pivot branch: `exp/icra27-chunkfix-fast`.
- Starting SHA: `597f55ab6ed311a7075032acab47ecddca5eb767`, the requested synchronization point on `exp/fast5080-adaptive-recency`.
- No historical research branch was merged. Later code was inspected only to locate reusable assets and runners.

Recent commits at the branch point were `597f55a` (remote-branch audit), `1ce9bf0` (dense temporal audit), `07bfc40` (validation selection), and `d163f5a` (dense temporal protocol).

## Upstreams and preserved dirty state

- RoboTwin: `/home/wjq/workspace/upstreams/RoboTwin`, SHA `266f3aadf505a4f7fe9af0faa41a20f5f47cd123`.
- XPolicyLab sub-repository: SHA `c37109c500be67d0dea6b36bf7337bbd26e763cd`.
- LeRobot: `/home/wjq/workspace/upstreams/lerobot`, SHA `f66e5128ecb2456e8c54a63d15404fa59c16aebc`.
- RoboTwin had pre-existing tracked edits in `envs/_base_task.py`, `envs/robot/planner.py`, and `envs/robot/robot.py`. Their starting tracked-diff SHA-256 was `1a781d7050dd6c26836e0c5d04ed80920aaed51d327171e3e76106d5ea9fdbe3`. They implement the known 5080 headless renderer, MPLIB fallback, and safe-qpos path. They were used but not modified or committed by this sprint.

## Available policies and data

- RoboTwin ACT checkpoint: `/home/wjq/checkpoints/robotwin/act-place_can_basket/demo_clean-50/policy_last.ckpt`, SHA-256 `a8a1b61614788a068c9b266b209e845034050281a0df737965406b3aec3ef1b0`.
- RoboTwin stats SHA-256: `269981a2f99456f8c64f9237442b4612085d2f73b63824d79ff4b175db365ee6`.
- This is the only local RoboTwin task checkpoint. Historical completed evaluations of it were at floor success, so a 3–5-task RoboTwin sweep was not possible without fetching new checkpoints.
- LIBERO ACT checkpoint: `/home/wjq/checkpoints/zeromidnight_act_libero_object`, a 100-step, 7-D ACT policy for all ten LIBERO-Object tasks. `model.safetensors` SHA-256 is `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`.
- LIBERO dataset: `/home/wjq/datasets/libero_object_25_08_23_lerobotv2.1`, 454 episodes and 66,984 frames. `meta/info.json` SHA-256 is `59bdeaa37775fa3289734286e00726334cea4037eafefd23426c4326a2d707fe`.
- Existing aligned dense ACT cache: `/home/wjq/workspace/one-clock/experiments/gate3a1_dense_temporal_cache`, 82 episodes split 41 validation / 41 test, with 12,294 full `(100, 7)` ACT chunks.
- Hardware: three NVIDIA GeForce RTX 5080 GPUs, 16 GB each, driver 595.84.

## Existing execution path

- `scripts/run_libero_gate0.py`: LeRobot/LIBERO runner, official initial states, raw step and episode JSONL, fixed/global or groupwise action-chunk execution.
- `scripts/run_robotwin_gate0.py`: bounded single-seed RoboTwin runner brought onto this branch from already-audited engineering history.
- `src/one_clock/executor.py`: fixed action-chunk executor used without changing policy inference.
- Existing configs: `configs/gate0_libero_object.yaml` and `configs/gate0_place_can_basket.yaml`.

The fastest useful batch is LIBERO global horizon 8, with one method per GPU. Model and environment setup occur once per process; five episodes take roughly 29–46 seconds of rollout phase depending on success termination.

## Gate 0

RoboTwin `place_can_basket`, `demo_clean`, seed 0, global horizon 8 completed one real episode:

- 1 episode, 0/1 success, 700 steps, 88 ACT queries.
- Model init 3.696 s; environment setup 1.194 s; observation 9.674 s; inference 3.112 s; action execution 23.122 s. Sum of measured phases: 40.80 s.
- Failure reason: `task_not_successful_at_step_limit`.
- Raw artifact: `artifacts/gate0_robotwin/global8_seed0.json`.

The official temporal-aggregation mode entered the real path but did not complete within a 120-second hard bound. This is recorded as a timeout, not an episode result. Curobo JIT import also warned about missing `ninja`; the preserved upstream patch fell back to MPLIB and the global-horizon run completed.

LIBERO task 8 then completed identical-seed smoke episodes for frozen ACT, EMA, and affine correction. All were 0/1 success, confirming that the correction hook, diagnostics, environment, assets, and checkpoint execute end to end. Several earlier smoke directories contain failed setup attempts (missing LIBERO config/assets, proxy `socksio`, and EGL mapping); they are retained rather than erased.

Runtime setup required installing `socksio==1.0.0` into the pre-existing `/home/wjq/workspace/one-clock/.venv` so Hugging Face downloads could use the configured SOCKS proxy. No repository dependency manifest was changed. Required LIBERO assets were cached under `/home/wjq/.cache/libero/assets`.

Gate-0 conclusion: infrastructure works. Because only one floor-performing RoboTwin checkpoint exists, the fast multi-task science path is the available 10-task LIBERO ACT stack.

# RoboTwin 2.0 environment audit — 2026-08-24

## Source

- Official repository: `https://github.com/RoboTwin-Platform/RoboTwin.git`
- Commit: `30954692d06ba7e89f7a6b76064f4062c488fa81`
- Pinned XPolicyLab submodule: `c37109c500be67d0dea6b36bf7337bbd26e763cd`
- Environment: `/home/wjq/research-assets/robotwin/robotwin2_overnight`

## Installation

The dedicated Python 3.10 environment installed the official RoboTwin
requirements and pinned XPolicyLab. PyTorch 2.4.1 from the repository
requirements could not execute on RTX 5080 `sm_120`; only this isolated
environment was updated to the official CUDA 12.8 Torch 2.8.0 / torchvision
0.23.0 compatibility pair. The official SAPIEN and mplib documented patches
were applied. CuRobo source was pinned to v0.7.8 but its build is blocked by
the absence of `CUDA_HOME`/`nvcc`. PyTorch3D was not retained because the
official document states it is unnecessary for non-3D functionality.

## Infrastructure checks

- `torch`, `sapien`, `gymnasium`, `XPolicyLab`: import checks passed.
- Installed versions recorded: Torch `2.8.0+cu128`, torchvision `0.23.0+cu128`,
  SAPIEN `3.0.0b1`, mplib `0.2.1`, Warp `1.12.0`, and XPolicyLab `0.0.1`.
  CuRobo and the optional MuJoCo Python package were not installed.
- `scripts/eval_policy.sh`: present.
- ACT adapter and `setup_eval_policy_server.sh`: present.
- Two-task scheduler dry-run passed using GPUs 1 and 2, with separate policy
  and simulator environment fields. No policy server or simulator episode was
  launched.
- The official `demo_clean` catalog contained 50 tasks; all 50 archive and
  extracted task directories are present under the external data root at
  revision `a967b852afa21a9cbf19a198f7e653109042e87c`. The external manifest
  hashes 7,650 files totaling 58,551,419,446 bytes.
- Simulator/task reset smoke test is blocked by missing official
  `assets/objects/objaverse/list.json`; `embodiments.zip` was extracted, while
  `background_texture.zip` and `objects.zip` repeatedly hit transient HF TLS
  failures. A bounded official downloader retry on 2026-08-25 resumed one of
  three archives but timed out at 900 seconds; the required object index was
  still absent, so the smoke test remains blocked.

## Reference training compatibility

Both ACT reference runs completed with the pinned `train.sh` defaults, seed 0,
6000 steps, and action dimension 14. The repository Torch 2.4.1 build was
incompatible with the host's RTX 5080 `sm_120`; the isolated environment's
Torch 2.8.0/cu128 replacement passed a synthetic CUDA convolution smoke test.
Both checkpoints passed CPU-only offline contract audits with chunk length 50,
action dimension 14, finite outputs, and no closed-loop action or success
evaluation. Checkpoint hashes and logs are in the external ACT reference
manifest.

## Safety

No complete episode, policy evaluation, task-success inspection, or scientific
rollout was performed.

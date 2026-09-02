# Track-C strong-policy checkpoint audit (C-0)

Status: **candidate selected, locally complete, and policy runtime prepared**. This audit is CPU/network/disk only. No SAPIEN, Vulkan, or RoboTwin environment has been initialized while LIBERO work is active.

## Selected feasibility candidate

`SidneyXie/pi05_robotwin` at immutable Hugging Face revision `e49e2ab6c11f07511573b67261bd129e88d0a416` is the selected C-1 canary candidate.

- Exact weights: `model.safetensors`, 9,354,050,752 bytes, plus the checkpoint's config and frozen pre/postprocessor buffers.
- License: Apache-2.0 in the model card.
- Architecture: LeRobot PyTorch π0.5, `gemma_2b` PaliGemma trunk plus `gemma_300m` action expert, bfloat16, 10 flow-matching inference steps.
- Training: final step 50,000 from `lerobot/pi05_base`; batch 16; seed 1000; full model fine-tuning rather than PEFT.
- Dataset/task coverage: `lerobot/robotwin_unified`, 27,500 episodes and 6,075,103 frames over the 50-task RoboTwin domain. The checkpoint card reports 32 public Easy task results.
- Embodiment/action: Aloha-AgileX bimanual, 14-D absolute joint-space action, two six-DoF arms plus two grippers, 50-action chunks; one 14-D state and head/left-wrist/right-wrist RGB inputs.
- Normalization: checkpoint-frozen MEAN_STD state/action buffers are included.
- Local compatibility: the embodiment, joint order, and three camera roles match the local RoboTwin Aloha-AgileX assets. The pinned local LeRobot source contains native π0.5 support. The 9.35 GB checkpoint may fit a 16 GB RTX 5080 in bfloat16, but that remains a technical canary question, not an assumed pass.
- Local identity: all nine files at the pinned revision were downloaded to `/home/wjq/research-assets/robotwin/checkpoints/SidneyXie_pi05_robotwin`; the local `model.safetensors` size is exactly 9,354,050,752 bytes.
- Isolated policy environment: `/home/wjq/research-assets/robotwin/pi05_policy_venv` (Python 3.12.3, LeRobot 0.6.2 at local source commit `f66e5128ecb2456e8c54a63d15404fa59c16aebc`, Torch 2.11.0+cu128, Transformers 5.5.4, NumPy 2.2.6). A config-only import resolved π0.5, chunk 50, 14-D absolute action, and left CUDA uninitialized.
- Isolated environment client: the pre-existing dedicated `/home/wjq/research-assets/robotwin/robotwin2_overnight` Python 3.10 runtime remains separate because RoboTwin's native stack uses NumPy 1.26.4. Policy and simulator will communicate through the established split-process interface after Track A.

Training-setting qualification: the `robotwin_unified` dataset has no dataset card. Its 27,500-episode total and description of varied layouts/lighting/backgrounds are consistent with the separately documented 50-clean + 500-randomized episodes per task corpus, but the checkpoint repository does not provide a direct per-episode clean/random manifest. Track C will therefore conservatively describe Hard as a harder/more variable evaluation condition and will never call it OOD generalization.

Sources checked 2026-09-02:

- Checkpoint and model card: <https://huggingface.co/SidneyXie/pi05_robotwin>
- Dataset repository: <https://huggingface.co/datasets/lerobot/robotwin_unified>
- Official RoboTwin benchmark setting: <https://robotwin-platform.github.io/leaderboard>
- Official RoboTwin OpenPI evaluation documentation: <https://robotwin-platform.github.io/doc/usage/Pi0.html>

## Alternatives audited

| Candidate | Provenance/coverage | Training data | Compatibility | Decision |
|---|---|---|---|---|
| `motus-robotics/pi0.5_robotwin2@effc6e1a…` | Public 50-task π0.5, 20.95 GB repository | Explicit 50 clean + 500 randomized episodes/task | Bespoke modified pipeline, delta-joint representation, no license metadata in card | Reference only; less auditable local execution route |
| `Avakn/robotwin2-checkpoints@338ca908…` | π0.5 only for `place_phone_stand`; training stopped at step 9,000 | 50 clean demos for that task | H200 provenance and single-task coverage | Too narrow for >=4-task gate |
| `SeonghoonYu/RACE_Robotwin@5c672c4…` | Six single-task π0.5 families | 50 clean demos/task | Very large repository; “best” checkpoint selected over a dense outcome sweep | Not used for a prospective baseline gate |
| `JackieMM/RoboTwin-pi05-30000-checkpoints` | 92 checkpoints, 813 GiB | Heterogeneous local training runs | Excessive bounded-audit/download burden | Not pursued |

No repeated checkpoint hunting beyond this bounded audit is authorized.

## C-1 boundary

The 4–8 feasibility tasks will be frozen only after every Track-A GPU worker exits, using public task-level baseline evidence and task semantics, not any result from this project. C-1 begins with a policy-load/inference canary and a deterministic simulator canary. Only a strong-policy baseline is permitted, Easy and Hard, under the overall 800-rollout cap. A pass records usable dynamic range; it does not launch RoboTwin method development.

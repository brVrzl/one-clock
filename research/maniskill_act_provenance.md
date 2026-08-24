# ManiSkill ACT provenance

The policy experiments use the official ManiSkill ACT state trainer and DETR-
VAE/ACT architecture from the ManiSkill repository:

- source README: https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/act/README.md
- source trainer: https://raw.githubusercontent.com/mani-skill/ManiSkill/main/examples/baselines/act/train.py
- source evaluator: https://raw.githubusercontent.com/mani-skill/ManiSkill/main/examples/baselines/act/act/evaluate.py

Local source snapshot: `/tmp/maniskill_official_act`. Two compatibility fixes
are applied for ManiSkill 3.0.1 on this host: normalization statistics are
computed over the loaded demonstrations rather than an undefined dataset
index, and the evaluator accepts the current direct `episode` info layout.
The temporal-aggregation buffer is also given one extra boundary slot to avoid
the current wrapper's post-truncation index. The model, ACT loss, action chunk
length (30), optimizer, and evaluation semantics are otherwise unchanged.

Environment: ManiSkill 3.0.1, SAPIEN 3.0.3, `physx_cpu`, state observations,
Panda `pd_ee_pose`, 10 original demonstrations per task, 24 added successful
branch trajectories for each augmentation method, batch size 32, matched
training budget 7,000 iterations for the augmentation screen, one evaluation
environment, five evaluation episodes, and fixed seed 1.

The failed custom MLP smoke run remains provenance-only and is not used in the
ACT comparison.

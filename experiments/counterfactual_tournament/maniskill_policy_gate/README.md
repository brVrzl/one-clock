# Matched policy smoke gate

This is the first post-Gate-0 policy smoke, not a final track result. It uses
the same 20 validated expert trajectories and the same 500 optimizer steps for
UniformBC, CriticalBC, and ContrastBC, with five held-out seeds per task.

All six method/task evaluations scored 0/5. The checkpoint and episode-level
JSONL logs are preserved here. This result kills further tuning of this custom
state-vector MLP as the policy backbone; the next policy gate should use the
existing ManiSkill ACT/Diffusion Policy training/evaluation pipeline.

# ManiSkill Gate 0 report

## Decision

**COUNTERFACTUAL SIGNAL SURVIVES** for the two validated causal-screen tasks.
This is a mechanism-screen result, not yet a three-task benchmark result:
PegInsertionSide-v1 initialized successfully, but its bundled fallback expert
did not reliably grasp the peg, so it is excluded from the causal summary.

The final causal run used 10 seeds per validated task, `physx_cpu`, the
official ManiSkill task/state APIs, and a fixed six-branch perturbation set:
four 3-mm object x/y pose offsets and two 0.02-rad Panda joint-state offsets
(a local EEF perturbation proxy). The zero branch was used only for validation
and was excluded from `c_t`.

## Counts and runtime

| task | expert trajectories | nominal successes | control actions | sampled states | perturbed branches | invalid branches | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| PickCube-v1 | 10 | 10 | 339 | 72 | 432 | 0 | 41.12 s |
| StackCube-v1 | 10 | 10 | 560 | 120 | 720 | 0 | 90.36 s |
| Total | 20 | 20 | 899 | 192 | 1,152 | 0 | 131.48 s |

There were 1,344 total branch records including 192 zero-perturbation
controls. Every sampled state passed exact restore/suffix replay and repeated
zero-branch isolation checks. The branch records contain a SHA-256 state ID,
perturbation metadata, branch validity, branch success, and repeated `c_t`.
Raw trajectory/state files and CSV branch logs are under
`experiments/counterfactual_tournament/maniskill_gate0_final/`.

## Criticality distributions

| task | mean | sample SD | min | max | range |
|---|---:|---:|---:|---:|---:|
| PickCube-v1 | 0.146 | 0.156 | 0.000 | 0.333 | 0.333 |
| StackCube-v1 | 0.185 | 0.156 | 0.000 | 0.333 | 0.333 |

The signal is repeatably non-uniform across both tasks. The 3-mm object-only
pilot was flat on StackCube, which is why the final fixed screen includes the
allowed robot-state/EEF perturbations. This perturbation-family change was
made before the 10-seed final screen and is recorded in the protocol.

## Heuristic comparisons

Pearson / Spearman correlation with state criticality:

| heuristic | PickCube | StackCube |
|---|---:|---:|
| action magnitude | 0.224 / 0.122 | -0.154 / -0.164 |
| action velocity | -0.076 / -0.078 | -0.074 / -0.025 |
| action acceleration | -0.074 / -0.036 | -0.065 / 0.135 |
| gripper transition | -0.076 / -0.077 | -0.068 / -0.067 |
| EEF-object distance | -0.101 / -0.039 | -0.053 / 0.023 |
| object-goal distance | 0.342 / 0.430 | 0.585 / 0.595 |

Object-goal distance is the nearest cheap heuristic, particularly for
StackCube, but it is not an equivalent explanation: it does not approach
perfect ranking agreement, while the velocity, acceleration, and gripper
event heuristics are near zero. The result therefore clears the internal
nontriviality gate, with the explicit caveat that only two tasks are screened.

## Qualitative examples

The following repeatable high-criticality states have `c_t = 0.333`, while
action velocity and gripper transition are both zero:

| task | episode | timestep | phase | EEF-object distance | object-goal distance |
|---|---:|---:|---:|---:|---:|
| PickCube-v1 | 0 | 5 | 0.139 | 0.0720 m | 0.2760 m |
| PickCube-v1 | 0 | 10 | 0.278 | 0.0314 m | 0.2760 m |
| PickCube-v1 | 0 | 15 | 0.417 | 0.0016 m | 0.2760 m |
| StackCube-v1 | 0 | 5 | 0.091 | 0.0688 m | 0.2094 m |
| StackCube-v1 | 0 | 10 | 0.182 | 0.0308 m | 0.2094 m |
| StackCube-v1 | 0 | 15 | 0.273 | 0.0014 m | 0.2094 m |

These are not claims that the causal score is independent of geometry. They
are examples where event and action-change heuristics do not flag the state.

## Infrastructure and limitations

- ManiSkill 3.0.1 was installed in `/home/wjq/workspace/venvs/maniskill`.
- The official `get_state_dict` / `set_state_dict` path was used for every
  branch, with exact named actor/articulation state hashes.
- The official Panda `mplib.Planner` constructor segfaults on this host, so
  the expert source is labeled honestly as official ManiSkill task geometry
  plus its bundled CPU IK controller and deterministic waypoints. No planner
  success is claimed.
- PegInsertionSide-v1 construction passed, but one diagnostic expert episode
  was unsuccessful; no Peg criticality values enter the decision.
- The RoboTwin target benchmark remains separate and is not replaced by this
  sandbox result.

The implementation was based on the local tournament scaffold at parent SHA
`638ee1a`; the final local commit SHA is recorded by Git after this report is
added.

## Next experiment

The first matched state-vector MLP smoke was run with 500 optimizer steps and
five held-out seeds per task. UniformBC, CriticalBC, and ContrastBC each scored
0/5 on both PickCube and StackCube. The checkpoints and episode-level logs are
preserved in `experiments/counterfactual_tournament/maniskill_policy_gate/`.
This is a negative backbone/pipeline result, not evidence that the causal
signal has no policy value, and no further tuning of this MLP is warranted.

Next run the same three-way comparison through an existing ManiSkill
ACT/Diffusion Policy state/image pipeline with matched steps and seeds. In
parallel, repair the Peg expert source or obtain recorded demonstrations, then
repeat Gate 0 before adding recovery branch generation.

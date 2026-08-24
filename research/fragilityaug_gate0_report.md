# FragilityAug causal screen

Status: `COUNTERFACTUAL SIGNAL SURVIVES` (mechanism gate only; policy gate pending).

The multi-scale screen reuses the 10 cached successful expert trajectories per
task from ManiSkill 3.0.1. States are sampled every 5 control steps. At each
state, the original action suffix is replayed after six local perturbations at
three scales: 1.5, 3.0, and 6.0 mm object/EEF-proxy displacement. The score is

`FragilityScore = 0.2 p_fail(1.5 mm) + 0.3 p_fail(3 mm) + 0.5 p_fail(6 mm)`.

The 50% robustness margin is the smallest tested scale with at least half of
valid branches failing. Invalid branches are excluded from the denominator.

| task | states | branches | invalid | runtime | mean score | score range | p_fail small / medium / large |
|---|---:|---:|---:|---:|---:|---:|---|
| PickCube-v1 | 72 | 1,296 | 0 | 255.2 s total | 0.248 | 0.000–0.467 | 0.046 / 0.150 / 0.387 |
| StackCube-v1 | 120 | 2,160 | 0 |  | 0.176 | 0.000–0.333 | 0.014 / 0.185 / 0.235 |

Cheap-heuristic Spearman correlations with the scalar score:

| task | phase | object-goal | EEF-object | action magnitude | velocity | acceleration | gripper transition |
|---|---:|---:|---:|---:|---:|---:|---:|
| PickCube-v1 | -0.011 | 0.411 | 0.049 | 0.205 | -0.045 | -0.012 | -0.053 |
| StackCube-v1 | -0.554 | 0.795 | 0.268 | -0.118 | -0.054 | 0.148 | 0.056 |

The held-out-episode ridge diagnostic using all cheap features gives PickCube
R² = -0.538, rank correlation = -0.041, and top-20% overlap = 0.00. For
StackCube, the corresponding values are R² = 0.654, rank correlation = 0.732,
and top-20% overlap = 0.33. Object-goal distance is the strongest single
heuristic on both tasks, with top-20% overlap 0.60 and 0.33 respectively.
This is a genuine novelty risk for StackCube and argues for reporting the task
wise difference rather than claiming universal superiority over geometry.

Examples missed by object-goal ranking include PickCube episode 3 at timesteps
25, 30, 20, and 35 (phases 0.694, 0.833, 0.556, and 0.972; scores 0.417,
0.417, 0.383, and 0.383), and StackCube episode 5 timestep 35 (phase 0.636;
score 0.300). These states are high-fragility states that do not have the
largest object-goal distance.

The policy gate must still establish that this training-free causal score
allocates additional successful branch trajectories more effectively than
random and cheap heuristic selection under the same 24-branch budget.

# FragilityAug policy gate

## Decision

**KEEP FRAGILITY FAMILY / CHANGE MECHANISM**

The causal signal survives, but the first official ACT allocation screen does
not yet establish a closed-loop advantage. All augmentation methods were
evaluated under the same 7,000-iteration training command and compared at the
6,000-iteration checkpoint. Each received 24 additional successful
medium-scale branch trajectories on top of the same 10 original expert
demonstrations. Every method was evaluated on five episodes.

| method | PickCube success_once / end | StackCube success_once / end |
|---|---:|---:|
| UniformACT | 0/5, 0/5 | 0/5, 0/5 |
| RandomAug | 0/5, 0/5 | 0/5, 0/5 |
| LateAug | 0/5, 0/5 | 0/5, 0/5 |
| GoalDistanceAug | 0/5, 0/5 | 0/5, 0/5 |
| MotionAug | 0/5, 0/5 | 0/5, 0/5 |
| FragilityAug | 0/5, 0/5 | 0/5, 0/5 |

The ACT pipeline is not categorically broken: a longer PickCube UniformACT
run reached 1/5 at iteration 8,000, and a StackCube run reached 1/5
`success_once` at iteration 2,000, but these events were not at the matched
augmentation checkpoint and StackCube did not reach `success_at_end`. They are
health evidence, not evidence for a method comparison.

The matched screen therefore has no separability: FragilityAug neither beats
RandomAug nor a heuristic. The correct conclusion is not that the causal
signal is false. It is that this first augmentation construction, action
representation, and ACT budget do not expose the signal in closed loop.

## Closest prior work and defensible distinction

The closest conceptual neighbor is *Perfect Demo Makes Poor Teacher*, which
also argues that data-side coverage of critical/recovery segments matters more
than simply reweighting fluent demonstrations. ISR is a close heuristic
baseline because it resamples using velocity and acceleration information.
Dream2Fix and FLARE are close in using counterfactual/failure or perturbation
data, but focus on generating or deploying recovery behavior. Geometry-aware
Policy Imitation uses trajectory geometry directly for policy construction.

The narrow defensible distinction for FragilityAug is:

> A training-free simulator measurement of downstream suffix failure under
> controlled local perturbations is used to rank *where a fixed number of new
> successful imitation branches should be collected*, with random, motion,
> gripper/event, and geometry selectors matched on branch count and training
> budget.

This is a selection principle, not a new controller, and it should remain
framed as such. The heuristic diagnostic shows a substantial novelty risk on
StackCube: object-goal distance has Spearman ρ = 0.795 with fragility and the
cheap-feature ridge has held-out R² = 0.654. PickCube is more promising
(object-goal ρ = 0.411; cheap-feature ridge R² = -0.538), but the policy gate
has not yet converted that diagnostic separation into success.

## Next experiment

Change the mechanism before porting to RoboTwin: preserve the exact same
fragility score and equal-budget selector, but generate branch trajectories in
the official delta-pose/action representation or use short corrective bridges
that produce longer aligned coverage. First verify that replayed augmented
branches train an ACT policy to a healthy success rate, then rerun only
UniformACT, RandomAug, strongest geometry heuristic, and FragilityAug at the
same budget. Do not run Diffusion Policy, ContrastBC, or a full RoboTwin method
port until that four-way screen separates.

RoboTwin remains the target benchmark. Its lightweight asset/replay recovery
track remains active, but this policy result does not satisfy the port gate.

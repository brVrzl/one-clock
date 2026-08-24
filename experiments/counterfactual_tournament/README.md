# ICRA 2027 counterfactual policy-learning tournament

This directory is a clean research namespace for the new sprint. The prior
execution-time mechanisms in the repository are historical only and are not
inputs to this tournament.

## First environment choice

The first causal gate uses the local RoboTwin checkout and its pinned SAPIEN
runtime. RoboTwin is selected for engineering speed because the available
runtime exposes `scene.get_physx_system().pack()` and `unpack()`, task classes
provide scripted `play_once()` continuations and `check_success()` labels, the
existing data format includes images, and the local environment already has
Torch/SAPIEN/Gymnasium. ManiSkill, MuJoCo, robosuite, and LIBERO are not
available in the active environment at protocol creation time.

The RoboTwin checkout is user-dirty. This sprint does not edit it; every run
records its current Git SHA and the project Git SHA.

## Locked causal question

Under a fixed extra-data budget, does selecting branch starts using downstream
counterfactual success loss improve closed-loop policy success over random,
motion, and gripper/event selection?

The primary method is **BranchBC**. CriticalBC, ContrastBC, and GeoAux are
secondary tracks sharing the same fork dataset where possible. No RL,
test-time filtering, action smoothing, post-policy correction, adaptive
horizons, or execution-time scheduling is part of this sprint.

## Gates

* Gate 0: three tasks, initially 1--2 successful scripted episodes each for
  smoke screening; expand to roughly 10--20 only after the fork engine passes.
  Forks are placed every 5--10 scripted control segments. The first perturbation
  set is limited to small robot-joint, object-position, and action-pose changes.
* Gate 1: UniformBC versus CriticalBC on identical demonstrations and optimizer
  budgets.
* Gate 2: RandomBranch versus CriticalBranch with exactly the same added branch
  trajectories and training budget. This is the primary causal comparison.
* Gate 3: ContrastBC and GeoAux reuse the fork dataset. Privileged state is
  training-only for GeoAux.

The unit of replication is the episode/task seed, not an individual fork. Forks
are nested measurements used to construct selection scores and are never treated
as independent policy-evaluation episodes.

## Artifact contract

Every run writes a manifest containing seed, task, dataset composition, training
steps when applicable, checkpoint path, project/upstream Git SHAs, and runtime
versions. Rollout outcomes are episode-level JSONL. Gate-0 raw fork states are
stored separately from summaries. A result is marked `not_run`, `failed`, or
`complete`; no missing run is silently scored as zero.

The live matrix is `results/tournament_matrix.csv`.

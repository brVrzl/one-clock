# ICRA 2027 counterfactual policy-learning tournament

This directory is a clean research namespace for the new sprint. The prior
execution-time mechanisms in the repository are historical only and are not
inputs to this tournament.

## Backend split

The first causal gate now uses ManiSkill as the fast development sandbox.
RoboTwin remains the eventual scalable benchmark. ManiSkill exposes official
`get_state_dict` / `set_state_dict` restore, recorded-state-compatible task
APIs, vectorization, and image-policy-compatible observations. The final Gate-0
artifacts are under `maniskill_gate0_final/` and the protocol is in
`maniskill_protocol.yaml`.

The local RoboTwin checkout and pinned SAPIEN runtime are preserved as a
separate recovery track. Its runtime exposes
`scene.get_physx_system().pack()` and `unpack()`, task classes provide scripted
`play_once()` continuations and `check_success()` labels, and the existing data
format includes images. Asset and planner recovery must not block the ManiSkill
causal gate.

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

* Gate 0: ManiSkill currently has 10 successful episodes each for PickCube-v1
  and StackCube-v1, sampled every 5 control steps with six small state
  perturbations. PegInsertionSide-v1 initialized but has no reliable fallback
  expert yet. RoboTwin remains queued for replay-based recovery after the
  mechanism survives.
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

# Bounded group-triggered joint re-query status

## Complete

- ACT protocol frozen before adaptive outcomes.
- Repaired fresh-environment pairing used for every method/state condition.
- One-state M0--M3 live prefix smoke passed, including initial observations,
  processed inputs, initial raw chunks, simulator states, and actions until the
  earliest method-specific re-query.
- Adaptive ACT workload complete: 3 methods × 40 episodes = 120 episodes.
- M0 authoritative repaired baseline reused: 32/40.
- M1 arm phase: 30/40.
- M2 gripper event: 35/40.
- M3 combined group event: 31/40.
- No historical action averaging, temporal ensemble, CogACT aggregation,
  independent group action source, predictor, or H_temp-controlled execution.
- SmolVLA was not launched; a minimal M2-only confirmation is prepared in the
  protocol for separate approval.

## Decision

`SINGLE_TRIGGER_BETTER`

M2 is the smallest development winner. It reaches 35/40 versus M0's 32/40,
with 3 candidate-only and 0 reference-only paired outcomes. M1 and M3 are
below M0; M3 is also below M2.

## Current jobs

No bounded-requery or SmolVLA jobs are active. Unrelated pre-existing
untracked experiment directories were preserved.

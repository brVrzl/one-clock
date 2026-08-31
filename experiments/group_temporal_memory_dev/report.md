# Group-conditioned temporal memory development

## Status

Preparation is complete, but the development rollout is gated pending Sol’s repaired h16 trio commit. No simulator rollout, success file, or GPU job has been started from this directory.

Sol’s audit commit is `33463ab4eb0ff1c64f794df7c76330bb7b56c143`. It establishes that the fresh identically seeded environment repair gives strict common-prefix equality. Sol selected the shared baseline `dense_equivalent_te`, with oldest-to-newest weights proportional to `exp(-0.01 * (q - q_oldest))`.

The runners additionally require the forthcoming commit containing the repaired `hard_h16`, `candidate_index_te_h16`, and `dense_equivalent_te_h16` trio. The previous reset-reuse results are excluded.

## Frozen module scope

The prepared ladder fixes h16, the four exposed development tasks, states 10--19, ACT horizon 100, SmolVLA horizon 50, and identical same-target candidate pools. M2 uses whole-action cosine compatibility with frozen alpha 0.3. M3 independently weights the 6-D arm and scalar gripper using the existing CogACT-compatible rules, then applies ordinary weighted action averaging. H_temp is outcome-blind and analysis-only.

M4 is marked `UNAVAILABLE_RELIABILITY_INTERFACE`: the current checkout has no frozen online-compatible group reliability predictor, so no reliability score or fallback was fabricated.

## Semantic validation

The CPU semantic suite passes 8 tests. It covers query scheduling, same-target `q + offset = t` alignment, group slicing, delay masking, one-candidate identity, shared-weight invariance, M3 reduction to M2 under identical compatibility, dense-equivalent prior orientation, missing-reliability refusal, and outcome-blind development-only H_temp freezing. The ACT and SmolVLA CPU semantic smokes also pass.

## Decision

**BLOCKED_BY_PAIRING_AUDIT**

This is a coordination gate, not a scientific null result. After Sol pushes the repaired trio, update `protocol.json`, run strict pairing smokes for each policy, and only then launch the development ladder.


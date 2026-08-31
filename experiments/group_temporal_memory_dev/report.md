# Group-conditioned temporal memory development

## Status

Preparation is complete, and Sol’s repaired ACT h16 trio handoff is recorded. No group-memory simulator rollout, success file, or GPU job has been started from this directory.

Sol’s audit commit is `33463ab4eb0ff1c64f794df7c76330bb7b56c143`, and the repaired trio commit is `b0b2a6d18ccc9da9ded0057d9f512ad8b535dac0`. The repaired protocol requires a fresh environment for each condition/state with exactly identical task/state/environment seed settings; its strict common-prefix equality validator passed. Sol selected the shared baseline `dense_equivalent_te`, with oldest-to-newest weights proportional to `exp(-0.01 * (q - q_oldest))`.

The corrected repaired ACT h16 trio is complete: hard `32/40`, candidate-index TE `24/40`, and dense-equivalent TE `23/40`. The decision is `DENSE_EQ_TE_HARMFUL`; repaired candidate-index TE remains harmful. The previous reset-reuse results are excluded.

## Frozen module scope

The prepared ladder fixes h16, the four exposed development tasks, states 10--19, ACT horizon 100, SmolVLA horizon 50, and identical same-target candidate pools. M2 uses whole-action cosine compatibility with frozen alpha 0.3. M3 independently weights the 6-D arm and scalar gripper using the existing CogACT-compatible rules, then applies ordinary weighted action averaging. H_temp is outcome-blind and analysis-only.

M4 is marked `UNAVAILABLE_RELIABILITY_INTERFACE`: the current checkout has no frozen online-compatible group reliability predictor, so no reliability score or fallback was fabricated.

## Semantic validation

The CPU semantic suite passes 8 tests. It covers query scheduling, same-target `q + offset = t` alignment, group slicing, delay masking, one-candidate identity, shared-weight invariance, M3 reduction to M2 under identical compatibility, dense-equivalent prior orientation, missing-reliability refusal, and outcome-blind development-only H_temp freezing. The ACT and SmolVLA CPU semantic smokes also pass.

## Decision

**SOL_REPAIRED_H16_RECORDED_GROUP_ROLLOUT_NOT_LAUNCHED**

This is a coordination handoff, not a group-memory outcome. If the development ladder resumes, preserve fresh-env pairing and run strict pairing smokes before launch.

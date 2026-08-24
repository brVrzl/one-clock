# Research decision log

## 2026-08-24: platform choice

The machine contains one RoboTwin checkpoint for one task, and its frozen policy is at floor performance. Gate 0 proved the runtime path but this cannot support a multi-task repair claim. Reusable execution infrastructure was retained; multi-task discrimination moved to the existing 10-task LIBERO ACT checkpoint and aligned cache.

## 2026-08-24: reject localized constant repair

Constant bias explains only 0.11% of error, while residuals are strongly phase-correlated and gripper-dominated. Implemented a small affine state/phase calibrator instead of a transformer. It passed offline MSE but remained conditional on Gate 2.

## 2026-08-24: kill full and selective affine repair

```text
PIVOT DECISION
Previous hypothesis: A gated state/phase affine residual can selectively repair a frozen ACT chunk.
Evidence: Offline episode MSE improved 24.0%, but the validation q25 gate activated on 100% of online queries; task 1 success fell from 3/5 to 1/5.
Why insufficient: The gate did not transfer and the correction induced closed-loop covariate shift.
New hypothesis: Restrict correction magnitude or dimensions to preserve nominal control.
Cheapest discriminating experiment: Scale residuals to 0.25 on tasks 6/8 and test gripper-only residuals on all three tasks.
```

## 2026-08-24: kill continuous ChunkFix

```text
PIVOT DECISION
Previous hypothesis: Conservative or gripper-only continuous residuals retain the offline gain without destabilizing arm control.
Evidence: Frozen ACT scored 9/15; EMA 3/15; sequential affine settings 4/15; gripper-only affine 2/15 on paired initial states.
Why insufficient: Every continuous intervention was directionally worse, including arm-preserving gripper repair. Offline action MSE is not a reliable closed-loop selector.
New hypothesis: A tiny discrete vocabulary of gripper advance/delay candidates may capture timing error while keeping the raw chunk as an explicit candidate.
Cheapest discriminating experiment: Measure raw, learned-selector, and oracle MSE for shifts {-8,-4,0,+4,+8} before implementing rollout selection.
```

The discriminating experiment was run. Linear selection reduced held-out frame MSE by 8.1%, and the oracle by 15.7%, but the learned selector chose no-change for only 1.7% of chunks. This supports further study of conservative candidate selection, not immediate deployment.

## Sprint decision

**PIVOT**

The original ChunkFix hypothesis does not survive. The current defensible statement is only a hypothesis:

> Discrete gripper-timing reranking may be a lower-risk correction space for deterministic ACT than continuous residual regression.

No novelty or success claim should be made until a conservative selector improves closed-loop success on a newly frozen evaluation protocol.

## Single next experiment

Train a reject-option selector over raw and ±4/±8-step gripper shifts. Calibrate its confidence and no-change cost on validation only, freeze the threshold, then evaluate offline on a fresh, previously uninspected episode split. Proceed to paired rollouts only if it improves gripper sign accuracy and total MSE while altering at most 25% of chunks. This is higher value than scaling the failed residual model because it directly tests whether selectivity can be made real.

## 2026-08-24: closed-loop EventAlign upper bound

The proposed next step was superseded by a direct causal oracle test, which is more informative than classifier fitting. Across 30 paired LIBERO initial states, frozen ACT succeeded on 15 and the per-state timing oracle on 18. Only 3/15 baseline failures were recoverable by any gripper shift; task 6 had no recoverable failure. All fixed shifts underperformed the baseline in pooled success and frequently broke baseline successes.

```text
PIVOT DECISION
Previous hypothesis: Sparse gripper-event realignment has enough causal success headroom to justify a learned reject-option selector.
Evidence: The closed-loop oracle improves 15/30 to 18/30, recovers 3/15 failures, has zero headroom on task 6, and the best fixed shift scores only 12/30.
Why insufficient: A deployable selector can only underperform this small, task-inconsistent oracle ceiling, while incorrect interventions often destroy successes.
New hypothesis: None selected in this experiment; preserve the evidence and search outside gripper-only timing.
Cheapest discriminating experiment: Completed; no selector training is warranted.
```

Decision: **KILL EVENTALIGN**.

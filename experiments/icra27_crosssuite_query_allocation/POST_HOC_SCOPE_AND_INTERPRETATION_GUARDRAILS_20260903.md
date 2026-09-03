# Post-hoc scope and interpretation guardrails

Recorded `2026-09-03` after Track-A suite effects were known. This document
does not modify a preregistration, scientific condition, or decision label.

## Outcome-motivated scope characterization

Label: `OUTCOME_MOTIVATED_POST_HOC_SCOPE_CHARACTERIZATION`

The following specification is frozen before calculation. Use all 30 Track-A
tasks and every demonstration episode listed by each task-specific ACT
checkpoint's frozen `train_config.json`. Do not reroll out anything.

Freeze exactly two task-level explanatory variables:

1. Demonstration gripper transition density: within-episode changes in
   `sign(action[6])`, with zero a separate state, divided by total eligible
   adjacent-pair duration at the verified physical rate of 20 Hz. Units are
   transitions per physical second.
2. Demonstration trajectory duration: the task-level mean across training
   episodes of `(episode_frame_count - 1) * 0.05` seconds.

Relate each explanatory variable separately to:

- `Delta_G4(task) = success(ARM4_GRIP32) - success(H4)`;
- `Delta_G2(task) = success(ARM2_GRIP16) - success(H2)`.

Report four descriptive Spearman associations across all 30 tasks, using
average ranks for ties and reporting every task value. Do not add alternative
task descriptors after observing the associations, select LIBERO-10 alone, or
convert suite concentration into a confirmatory applicability claim.

## Track-A suite discipline

For ARM4_GRIP32-H4, preserve:

- LIBERO-10: +13.333 percentage points;
- Goal: 0.000 percentage points;
- Spatial: +0.667 percentage points;
- leave-LIBERO-10-out aggregate: +0.333 percentage points.

The preregistered all-LOSO-positive criterion passed, but with minimal
numerical margin outside LIBERO-10.

For ARM2_GRIP16-H2, preserve the separate suite effects:

- LIBERO-10: +14.000 percentage points;
- Goal: +2.000 percentage points;
- Spatial: +1.333 percentage points.

The two operating points must not be described as having equal cross-suite
robustness.

## Measured-path decomposition discipline

It is valid to report the measured path
`H2 -> ARM2_GRIP16 -> H16`, whose conditional edge effects are +5.778 and
+8.000 percentage points. Call this a `measured-path decomposition`.

Do not call the edges unique arm/gripper contributions, assign percentages of
the total effect to components, divide effects by action dimensionality to
claim a per-dimension contribution, or imply additivity without the missing
alternative factorial path.

## B3 configuration check

B3 froze the complete contiguous chunk-index range `0..32`, not a sparse
subset selected to represent nominal physical times. It is a predeclared broad
curve with exact stored-frame targets. The offsets are index-valid; corrected
physical labels are `offset/20` seconds, covering 0..1.60 seconds. The frozen
anchor stride of 10 frames is 0.50 seconds. No new offsets may be added.

## SmolVLA scope language

R2A remains eligible for preregistered execution according to its existing
technical gate. However, the SmolVLA checkpoint model card records its
training dataset as unknown. Unless independent provenance resolves its
training/chunk physical timebase, R2A cannot establish a physically matched
same-target cross-policy replication. It remains a scope experiment on the
frozen discrete executor construction.

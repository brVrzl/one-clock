# Track-A paper-support interpretation hygiene

This is a future-writing support record, not manuscript text. It changes no
rollout, analysis rule, decision label, manuscript, LaTeX source, or
`CLAIMS.md`.

## Canonical per-suite outcomes

| Suite | H16 | H4 | ARM4_GRIP32 | H2 | ARM2_GRIP16 | TE_DENSE |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-10 | 82/150 (54.667%) | 61/150 (40.667%) | 81/150 (54.000%) | 53/150 (35.333%) | 74/150 (49.333%) | 55/150 (36.667%) |
| Goal | 137/150 (91.333%) | 125/150 (83.333%) | 125/150 (83.333%) | 117/150 (78.000%) | 120/150 (80.000%) | 120/150 (80.000%) |
| Spatial | 138/150 (92.000%) | 128/150 (85.333%) | 129/150 (86.000%) | 125/150 (83.333%) | 127/150 (84.667%) | 113/150 (75.333%) |

The tidy machine-readable tables are
`track_a/te_dense_characterization/track_a_per_suite_absolute.csv` and
`track_a/te_dense_characterization/track_a_per_suite_contrasts.csv`.

## Cross-suite robustness disclosure

ARM4_GRIP32-H4 is +13.333 percentage points in LIBERO-10, 0.000 in Goal,
and +0.667 in Spatial. Its preregistered pooled contrast is positive under both
paired and task-cluster inference. The preregistered all-LOSO-positive
subcriterion technically passes, but the leave-LIBERO-10-out point estimate is
only +0.333 percentage points. This minimal margin applies specifically to the
positive non-LIBERO-10 LOSO subcriterion, not to the pooled paired or
task-cluster result.

ARM2_GRIP16-H2 is +14.000 percentage points in LIBERO-10, +2.000 in Goal,
and +1.333 in Spatial. The two operating points must not be described as
having equal cross-suite robustness.

Canonical leave-one-suite-out point estimates exist, but LOSO intervals were
not generated. For ARM4_GRIP32-H4, omitting LIBERO-10, Goal, and Spatial gives
+0.333, +7.000, and +6.667 pp, respectively. For ARM2_GRIP16-H2, the
corresponding estimates are +1.667, +7.667, and +8.000 pp. Do not manufacture
a post-hoc LOSO interval procedure.

## Main Track-A hierarchy

The main component-specific result is that, at two frozen arm replanning
cadences, extending only gripper commitment improves success:

- H4 to ARM4_GRIP32: +4.667 percentage points, with policy-query rates
  0.251156 and 0.251318 per executed environment step;
- H2 to ARM2_GRIP16: +5.778 percentage points, with policy-query rates
  0.500720 and 0.500907 per executed environment step.

Use `policy-query rate`, `replanning cadence`, or `queries per executed
environment step`. Do not say identical total query count, identical compute,
or identical FLOPs. Total queries differ because interventions change episode
lengths.

The coherent `H16 > H4 > H2` sequence is robust supporting context for a
uniform frequent-replanning penalty, but it is not the main novelty claim.
Coherent H16 remains the strongest frozen operating point. The
component-resolved result is therefore an operational consequence under
constrained replanning cadence, not global method dominance.

## Measured-path discipline

`H2 -> ARM2_GRIP16 -> H16` is a valid measured path:

- first edge: arm stays 2 while gripper changes 2 to 16, +5.778 pp;
- second edge: gripper stays 16 while arm changes 2 to 16, an additional
  +8.000 pp.

Call this a `measured-path decomposition`. Do not call the edges unique
arm/gripper contributions, assign contribution percentages, divide by action
dimensionality, or assume path-independent additivity without the missing
alternative factorial path.

Do not construct an analogous decomposition
`H4 -> ARM4_GRIP32 -> H16`. Its second transition changes arm 4 to 16 and
gripper 32 to 16 simultaneously. It is not a single-component edge, and the H2
and H4 paths must not be shown as symmetric factorial decompositions.

## Reviewer-style operational consequence

The operational-consequence objection is substantially addressed because
A4G32 exceeds H4 near qrate 0.25 and A2G16 exceeds H2 near qrate 0.50.
However, both remain below H16. Future writing should state:

> If replanning cadence is freely selectable in this static LIBERO setting,
> coherent H16 remains preferable. The component-resolved advantage is
> relevant to regimes in which a higher replanning cadence is externally
> imposed or otherwise required; such dynamic/reactivity-constrained regimes
> are not evaluated here.

Do not claim that LIBERO establishes value under real external disturbances.

## Moderator provenance

The original preregistered Track-A moderator is the fraction of task-specific
ACT training action steps with closed-gripper intent, `action[6] < 0`: a
gripper occupancy/duty-cycle measure. Its exact canonical result is 30 tasks,
Spearman rho `0.19222126827715721`, two-sided p
`0.30885273529563795`, descriptive/null unsupported.

This is not transition density. It must not be relabeled as a
transition-frequency test or silently replaced. The separately frozen
transition-density/trajectory-duration proposal remains
`OUTCOME_MOTIVATED_POST_HOC_SCOPE_CHARACTERIZATION`, proposed after observing
suite concentration. It is not elevated into the preregistered moderator and
does not authorize alternative descriptor search.

## SmolVLA scope

Preserve R2A if it passes its preregistered technical gate. SmolVLA's
training/chunk physical time remains
`NOT_IDENTIFIABLE_FROM_AVAILABLE_PROVENANCE`. Unless independent provenance
resolves that limitation, it is supplementary scope evidence, not a physically
matched same-target replication, central cross-policy evidence, or support for
a general action-chunked-policy claim. Likely main-paper scope should remain
ACT-focused. Do not finalize the title before R1B is known.

## Temporal and reproducibility inputs

The superseding temporal audit remains authoritative: actual LIBERO evaluation
is 20 Hz; d20 is 1.00 s; H2/H4/H16/H32 are
0.10/0.20/0.80/1.60 s. The pinned LeRobot 0.4.4 path did not propagate sealed
nominal `fps=10` metadata to LIBERO `control_freq`, so execution used LIBERO's
20-Hz default. Future reproduction must configure and verify 20 Hz explicitly.
Do not call this an upstream bug without independent establishment. Preserve
the old audit as superseded history.

B3 retains its predeclared contiguous index range 0..32 with physical label
`k/20` seconds. Do not add offsets or redesign it. A null/non-discriminative B3
result must be retained and mechanism search must then stop.

## Provisional conceptual arc

1. Q1, action-group target-time assignment: frozen 140-block ACT same-target
   factorial.
2. Q2, temporal scale and component identity: R1A sensitivity plus R1B
   translation/rotation.
3. Q3, simple persistence/predictability accounts: B1 localization kill, B2
   insufficient, preregistered occupancy moderator null, B3 pending, and
   command discontinuity non-identifying/post-hoc.
4. Q4, execution consequence: Track-A conditional gripper-commitment gains
   plus R1C query-matched decomposition.

Do not reframe the paper around generic frequent replanning or around coherence
unless future identifiable evidence warrants it.

## Future Track-A figure information hierarchy

No artwork is authorized here. The future main panel should place pooled H16,
H4, ARM4_GRIP32, H2, ARM2_GRIP16, and TE_DENSE against log policy-query rate
and success. Use no regression and no line through all six conditions. Mark
TE_DENSE with distinct aggregation semantics and annotate only the paired
H4-to-ARM4_GRIP32 and H2-to-ARM2_GRIP16 comparisons.

A small inset or second panel should show suite effects for those two contrasts
and visibly retain the LIBERO-10 concentration, Goal zero, and near-zero
Spatial ARM4_GRIP32-H4 effect. Wall-clock results belong in a table.

## Stop rule

After the currently frozen R1A, R1B, R1C, R1D, technically eligible R2A, B3,
and already authorized completed-artifact analyses, launch no rescue executor,
horizon/lag sweep, mechanism family, RoboTwin experiment, consensus method,
RTC reproduction, seed expansion, or adaptive method. The only exception is a
concrete technical-integrity defect that invalidates or materially
reinterprets an existing main-paper claim. Otherwise proceed to final analysis
freeze and paper writing.

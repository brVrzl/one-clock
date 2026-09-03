# ICRA 2027 final scientific claim freeze

Freeze date: 2026-09-03 (Asia/Shanghai)

Status: `FINAL_SCIENTIFIC_CLAIM_FREEZE`

This is a scientific writing-support artifact, not manuscript prose. It governs the next phase: `FINAL CLAIM FREEZE -> MANUSCRIPT REWRITE -> FINAL FIGURES`. Scientific experimentation remains closed.

## A. Confirmatory strongest evidence

The strongest evidence is the original preregistered 140-block ACT same-target confirmation on Goal tasks 4, 6, 7, 8, 9 and LIBERO-10 tasks 0, 2, 4, 6, 7, with 14 task-state blocks per task and ten task-specific ACT checkpoints.

Primary sign convention: `S(A0G20)-S(A20G0)`.

- `A0G20`: fresh arm plus a 20-step-old gripper prediction for the same current physical target time.
- `A20G0`: a 20-step-old arm prediction plus fresh gripper for the same current physical target time.
- Same-target identity: `q+k=t`.
- At 20 Hz, d=20 is 1.00 s.

Canonical primary result: `83/140 (59.29%) - 38/140 (27.14%) = +32.14 pp`; discordance 48:3; exact two-sided McNemar `p=1.96749e-11`; paired 95% CI `[+23.6,+40.7] pp`; task-cluster 95% CI `[+21.4,+44.3] pp`.

d=20 was frozen before R1A existed and was not selected from the later sweep. R1A subsequently preserved `S(A0 G_d)>S(A_d G0)` at every frozen lag from 0.10 to 1.60 s. On the separate 126-block exposed-development R1A cohort, observed separation ranged from `+21.43 to +54.76 pp`; d=20 attained the largest observed separation, +54.76 pp. No preregistered lag-versus-lag inference establishes that d=20 is a statistically distinct peak. Accordingly, +32.14 pp remains the confirmatory cross-suite magnitude, while +54.76 pp is a later cohort's maximum observed grid separation, not a typical across-lag effect.

Same-target probe conditions query the policy once per executed environment step (`policy-query rate=1.0`). They are diagnostic instruments for separating source-observation age while holding physical target time fixed, not deployable operating points and not evidence that a qrate=1 executor is practically preferable.

## B. Characterization evidence

### R1A temporal-scale structure

Governance: frozen reviewer-directed `EXPOSED_DEVELOPMENT_CHARACTERIZATION`, not confirmatory evidence.

- Complete lags: d=2, 4, 8, 12, 16, 20, 32, corresponding to 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 1.60 s.
- Stale-arm/fresh-gripper success: 38.89%, 38.89%, 28.57%, 16.67%, 18.25%, 9.52%, 25.40%.
- Fresh-arm/stale-gripper success: 60.32%, 65.08%, 69.84%, 69.84%, 71.43%, 64.29%, 62.70%.
- Between-branch separation: +21.43, +26.19, +41.27, +53.17, +53.17, +54.76, +37.30 pp.
- Both branches are non-monotone. The stale-arm branch reaches its minimum at d=20 and recovers at d=32. The stale-gripper branch peaks at d=16, then degrades at d=20 and d=32, while remaining above Fresh throughout the frozen grid.
- Exact d=16 -> d=20 -> d=32 branch values are 18.25% -> 9.52% -> 25.40% for stale arm and 71.43% -> 64.29% -> 62.70% for stale gripper. The corresponding separations are +53.17 -> +54.76 -> +37.30 pp.
- The d=20 to d=32 narrowing is descriptively composed of a +15.87 pp stale-arm recovery and a -1.59 pp stale-gripper decline. This is not causal attribution.

All curve-shape descriptions are `POST_HOC_DESCRIPTIVE` within the exposed-development characterization. No failure threshold, tolerance limit, collapse point, or significant change point is defined.

### R1B within-arm identity structure

Governance: frozen reviewer-directed, exposed Object-development characterization; not confirmatory evidence.

Primary sign convention: `S(translation-stale)-S(rotation-stale)`.

At d=20, translation-stale success was 11/126 (8.73%), rotation-stale success was 53/126 (42.06%), and Fresh was 56/126 (44.44%). The primary contrast was `-33.33 pp`; discordance 3:45; exact McNemar `p=1.31259e-10`; paired 95% CI `[-42.06,-24.60] pp`; task-cluster 95% CI `[-44.44,-22.22] pp`. Both canonical intervals exclude zero and lie strictly below zero. All nine frozen leave-one-task-out estimates are negative.

This supports `component-dependent temporal sensitivity within ACT`, with the explicit scope restriction to the evaluated ACT checkpoint family and exposed Object cohort.

B1 ACT same-target dispersion orders rotation 0.1484 > translation 0.1364 > gripper 0.0790. R1B behavioral sensitivity orders translation above rotation. The narrow classification is `CONTRADICTED_AS_AN_ORDERING_PREDICTOR_WITHIN_ARM`: within the arm, same-target dispersion ranks rotation above translation, whereas behavioral temporal sensitivity ranks translation above rotation. The measured dispersion itself remains valid and no causal relationship is tested.

## C. Query-schedule identification

Governance: frozen reviewer-directed `POST_HOC_QUERY_MATCHED_EXTENSION`.

The old query-schedule confound was that fixed-source same-target probes made one whole-policy query every step while sparse coherent H16 queried every 16 steps. Thus query schedule/count differed alongside temporal source assignment.

R1C makes one whole-policy query per executed step in all four cells:

- `C00`, Fresh arm/Fresh gripper: 77/140 (55.00%).
- `C10`, scheduled-H16 arm/Fresh gripper: 76/140 (54.29%).
- `C01`, Fresh arm/scheduled-H16 gripper: 81/140 (57.86%).
- `C11`, scheduled-H16 arm/scheduled-H16 gripper: 93/140 (66.43%).

The strongest conditional result is `C11-C10=+12.14 pp`, discordance 20:3, exact `p=0.000488281`, paired CI `[+5.71,+18.57]`, task-cluster CI `[+3.57,+21.43]`. The other scheduled-component conditional effect is `C11-C01=+8.57 pp`, discordance 26:14, exact `p=0.0806905`, paired CI `[+0.00,+17.14]`, task-cluster CI `[-2.14,+21.43]`. The frozen risk-difference interaction is +9.29 pp; no new uncertainty is attached.

The frozen C11 canary required unused dense forward passes to leave executed actions, simulator trajectory, terminal result, and length identical to sparse HARD-H16. Persisted checks passed for exact executed actions, episode length, terminal success, completion step, and `trajectory_identity_from_same_initial_state_and_exact_actions`. The trajectory definition relies on the deterministic evaluator, same initial state, and exact actions; it is not an independently persisted elementwise environment-state array comparison.

Narrow conclusion: additional discarded policy queries did not alter the executed frozen C11 behavior under the tested deterministic ACT evaluator. The matched-query factorial therefore removes the old policy-query schedule/count confound for this tested ACT component comparison. This conclusion does not extend to arbitrary stochastic policies, inference stacks, or randomized environments.

## D. Execution consequence

Governance: frozen preregistered Track-A execution evidence.

The matched quantity is `policy queries per executed environment step`, not total query count and not identical compute.

- `H4 -> ARM4_GRIP32`: `+4.667 pp` (314/450 to 335/450), paired CI `[+2.00,+7.56]`, task-cluster CI `[+0.67,+9.11]`, policy-query rate approximately 0.251 in both conditions.
- `H2 -> ARM2_GRIP16`: `+5.778 pp` (295/450 to 321/450), paired CI `[+3.56,+8.22]`, task-cluster CI `[+2.67,+9.56]`, policy-query rate approximately 0.501 in both conditions.

Component-resolved temporal allocation therefore has an operational consequence under nearly matched replanning cadence. Coherent H16 remains the strongest overall frozen operating point at 357/450 (79.33%), exceeding ARM4_GRIP32 by 4.889 pp and ARM2_GRIP16 by 8.000 pp. The component-resolved methods are not globally superior executors. If replanning cadence is freely selectable in the present static LIBERO benchmark, H16 remains preferable.

## E. Mechanism boundary

| Diagnostic or interpretation | Classification |
|---|---|
| B1 same-target dispersion values | `SUPPORTED` as descriptive measurement |
| Original ACT-specific gripper-localization story | `CONTRADICTED` |
| ACT-versus-SmolVLA localization difference, whose interval crosses zero | `UNSUPPORTED` |
| B1 within-arm dispersion ordering as a behavioral ordering predictor | `CONTRADICTED_AS_AN_ORDERING_PREDICTOR_WITHIN_ARM` |
| B2 “gripper stays unchanged” as a complete explanation; `P(no transition at 1.00 s)=0.675018` | `INSUFFICIENT_AS_A_COMPLETE_EXPLANATION` |
| Preregistered occupancy moderator, n=30, rho=0.1922, p=0.3089 | `NULL / UNSUPPORTED` |
| Command discontinuity/coherence account | `NON_IDENTIFYING_POST_HOC_CHARACTERIZATION` |
| B3 forecastability | `NO_FROZEN_CRITERION`; descriptively non-corresponding/mixed |
| Positive causal mechanism | `UNRESOLVED` |

No tested diagnostic warrants a positive explanatory or causal mechanism claim. The observed temporal sensitivity remains mechanistically unresolved by the diagnostics tested here.

TE_DENSE remains a scoped negative result: under the canonical frozen LeRobot v0.4.4 coefficient 0.01 and chunk length 100 in this ACT/LIBERO evaluation, dense temporal aggregation places substantial weight on old predictions and produces substantially more near-boundary gripper commands. It is not an implementation bug, not chatter, and does not show that near-boundary commands cause the full success loss.

## F. Policy scope

Central claims are ACT-focused. SmolVLA does not support physically matched same-target behavioral replication because its training/chunk physical timebase is not identifiable from available provenance. R2A was not run because its frozen runtime eligibility window expired: `R2A_NOT_RUN_FROZEN_GATE_INELIGIBLE`. SmolVLA must not be used to justify policy-general claims.

## G. Suite, benchmark, and Spatial scope

Track-A component-resolved gains are concentrated in LIBERO-10:

| Contrast | LIBERO-10 | Goal | Spatial |
|---|---:|---:|---:|
| `ARM4_GRIP32-H4` | +13.333 pp | +0.000 pp | +0.667 pp |
| `ARM2_GRIP16-H2` | +14.000 pp | +2.000 pp | +1.333 pp |

H16 baseline success is 54.7% on LIBERO-10, 91.3% on Goal, and 92.0% on Spatial. Suite identity, baseline difficulty/ceiling, task semantics, and component-resolved gain covary, so the source of this heterogeneity is not identifiable.

R1D separately completes the previously missing Spatial Reverse20/`A20G0` factorial cell. On Spatial tasks 0-9 and the frozen ten states per task, `A0G20-A20G0=+28.00 pp` (40/100 versus 12/100), discordance 33:5, exact `p=4.25596e-06`, paired CI `[+17.00,+39.00]`, task-cluster CI `[+13.00,+45.00]`. R1D preserves the sign of the original +32.14 pp asymmetry and has a 4.14 pp smaller, broadly similar descriptive point estimate. This `POST_HOC_SPATIAL_FACTORIAL_COMPLETION` is not pooled into the original confirmatory 140 blocks.

## H. Governance status ledger

| Major result | Provenance/status | Permitted role |
|---|---|---|
| Original 140-block ACT `A0G20-A20G0` | `PREREGISTERED_CONFIRMATORY` | Strongest central evidence |
| Track-A matched-cadence contrasts | preregistered frozen execution evidence | Operational consequence |
| R1A | frozen reviewer-directed; `EXPOSED_DEVELOPMENT_CHARACTERIZATION` | Temporal-scale characterization only |
| R1A curve-shape statements | `POST_HOC_DESCRIPTIVE` | Describe observed curves; no change-point or threshold claim |
| R1B | frozen reviewer-directed; exposed Object-development characterization | Within-arm heterogeneity; not confirmatory |
| R1C | frozen reviewer-directed; `POST_HOC_QUERY_MATCHED_EXTENSION` | Query-schedule identification under deterministic ACT canary contract |
| R1D | frozen reviewer-directed; `POST_HOC_SPATIAL_FACTORIAL_COMPLETION` | Separate Spatial scope check; never pooled with 140 blocks |
| B1/B2/B3 mechanism diagnostics | supplementary-only and provenance-limited; B3 `NO_FROZEN_CRITERION` | Mechanism boundary, not causal support |
| Command discontinuity | `NON_IDENTIFYING_POST_HOC_CHARACTERIZATION` | No coherence support or falsification |
| R2A | `FROZEN_GATE_INELIGIBLE` | Not run; no SmolVLA behavioral replication claim |
| SmolVLA same-target behavioral scope | supplementary-only provenance-limited evidence | No policy-general claim |
| TE_DENSE characterization | frozen condition plus post-outcome scoped characterization with resolved runtime provenance | Narrow implementation-specific negative result |

## Final recommended claim scope

For the evaluated ACT policies, temporal source and execution choices have component-dependent behavioral effects. Preregistered same-target evidence establishes a robust arm-gripper asymmetry; frozen exposed-development characterization shows that the ordering persists across 0.10-1.60 s and that translation and rotation staleness differ within the arm. At deployable periodic schedules, component-resolved gripper commitment improves success relative to nearly query-matched global short horizons, while coherent H16 remains the strongest overall operating point. The causal mechanism, cross-policy generality, and source of benchmark heterogeneity remain unresolved.

## Title-scope decision

Both canonical R1B uncertainty intervals exclude zero in the same negative direction under `S(translation-stale)-S(rotation-stale)`, and every frozen LOTO estimate is negative. Therefore `component-dependent` is defensible, provided the title remains explicitly ACT-focused.

Recommended titles:

1. **Same-Target Probes Reveal Component-Dependent Temporal Sensitivity in ACT Action-Chunk Execution**
2. **Component-Dependent Temporal Sensitivity and Query Allocation in ACT Action-Chunk Execution**
3. **One Clock Does Not Fit All: Component-Dependent Temporal Execution in ACT**

Candidate evaluation: `Same-Target Probes Reveal Component-Dependent Temporal Sensitivity in ACT Action-Chunk Execution` is supported and is the preferred option. It names the diagnostic, the within-arm-plus-arm-gripper scope, and the ACT policy boundary without implying cross-policy generality or a globally superior executor.

## Abstract claim skeleton

1. Temporal execution in action-chunking policies usually applies one setting across heterogeneous action components.
2. Same-target probes isolate source-observation age while holding current physical target time fixed through `q+k=t`.
3. These probes query the policy once per executed environment step (`policy-query rate=1.0`); they are diagnostic instruments that separate temporal source-age effects from deployment scheduling, not deployable operating points, and the +32.1 pp asymmetry does not show that a qrate=1 executor is practically preferable.
4. In the preregistered 140-block ACT confirmation, fresh-arm/stale-gripper minus stale-arm/fresh-gripper success at d=20 was +32.14 pp, while later exposed-development R1A preserved the ordering over 0.10-1.60 s with observed separations from +21.43 to +54.76 pp; d=20 was frozen before that sweep and no lag-versus-lag test establishes a distinct peak.
5. R1A resolves non-monotone branch-specific scale structure, and R1B finds translation staleness more damaging than rotation staleness by 33.33 pp under the opposite success sign, with both canonical intervals strictly below zero.
6. At two fixed arm cadences, component-resolved gripper commitment improves success by +4.667 and +5.778 pp over nearly query-matched global short-horizon execution, while coherent H16 remains strongest overall.
7. Simple dispersion, persistence, occupancy, forecastability, and command-discontinuity diagnostics do not provide a complete mechanism; claims remain ACT-specific and explicitly acknowledge suite concentration and benchmark heterogeneity.

## FUTURE WORK / POST-ICRA

- Causal identification of the mechanism producing component-dependent temporal sensitivity.
- Generalization beyond ACT once physical timebase provenance permits matched interpretation.
- Temporal execution under dynamically constrained or reactive settings.
- Scope across broader benchmarks without conflating suite identity, task semantics, and baseline ceiling.

These are post-ICRA research topics only. They do not reopen the closed ICRA 2027 experiment program or authorize additional execution.

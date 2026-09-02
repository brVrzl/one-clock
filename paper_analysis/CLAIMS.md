## 1. Proposed central claim

Controlled same-target perturbations reveal a large component-dependent temporal asymmetry in ACT: holding physical target time fixed while assigning different source ages to action components produces sharply different success rates, with FO20 outperforming Reverse20 by 32.1 percentage points on the frozen 140-block cohort under query schedules matched up to episode length. This observation did not translate into reliable gains under the tested fixed-clock and event-triggered executor interventions, and the strongest query-budget-matched ACT fixed-clock effect did not reproduce in SmolVLA. Component-dependent temporal sensitivity is therefore measurable but is not a generic consequence of emitting action chunks, and its practical value depends on policy and evaluation properties beyond chunking alone.

## 2. Three contributions

1. A same-target controlled measurement framework that compares action components at a common physical target time. Its identity is `q + k = t`, with source age `d = t - q`, so natural same-target execution has `k = d`. Source age and chunk offset are coupled, not orthogonal; the framework assigns different source ages to different components while holding `t` fixed.
2. A quantitative characterization of component-temporal sensitivity in ACT. The large FO20-versus-Reverse20 asymmetry is preregistered frozen evidence; the supporting factorial decomposition is explicitly post-hoc analysis of that frozen cohort; and the broader gripper-horizon landscape is exposed-development characterization.
3. A controlled falsification and evaluation methodology using a matched-query fixed-horizon control and a horizon-distribution-matched shuffled-trigger control. The CARE/M2 construction is a tested executor hypothesis with final status `METHOD_NULL`, not a method contribution.

## 3. Allowed claims

- “Holding physical target time fixed while assigning different source ages to ACT action components revealed a large component-dependent temporal asymmetry.”
- “FO20 achieved 83/140 successes and Reverse20 achieved 38/140 on the frozen confirmation cohort, a paired difference of +32.1 percentage points (exact two-sided McNemar `p = 1.97e-11`; paired bootstrap 95% CI `[23.6, 40.7]` percentage points; task-cluster bootstrap 95% CI `[21.4, 44.3]`).”
- “FO20 and Reverse20 are query-schedule matched up to episode length: each makes exactly one query per executed environment step in every block, but success-dependent termination produces different realized episode lengths and therefore different total query counts.”
- “In the ACT execution landscapes we measured, temporal sensitivity is strongly concentrated in the gripper channel, while arm-horizon changes are comparatively weak over the tested range.” This statement must immediately note that the positive landscape evidence is exposed development, that the strongest fixed-clock ACT effect did not reproduce in SmolVLA, that gripper age is not established as a universal sufficient statistic, and that ARM4_GRIP32 and ARM16_GRIP32 come from different cohorts and do not define a clean same-cohort interaction.
- “The observed ACT asymmetry did not translate into reliable gains under the tested fixed-clock and event-triggered executor interventions.”
- “Across the evaluated ACT cohorts, H8, H13, and H16 form a competitive moderate-horizon band; point-estimate ordering changes by cohort, and no unique optimum reproduces.”
- “Component-dependent temporal sensitivity is not a generic consequence of action chunking. The strongest ACT fixed-clock effect did not transfer under a query-budget-matched SmolVLA test, indicating dependence on policy properties beyond the fact that both emit action chunks.” This does not establish zero component asymmetry for SmolVLA everywhere.
- “The RoboTwin pilot was a negative feasibility and statistical-dynamic-range finding under a severe floor effect.” It is not evidence that temporal structure is absent in RoboTwin.

### Fresh-gripper collapse across constructions

- Frozen evidence: Reverse20, which uses old-arm/fresh-gripper same-target execution, achieved 38/140 on frozen confirmation. Its contrast with FO20 is large and stable.
- Development characterization: C2 with an h16 committed arm and fresh gripper achieved 42/126, versus 56/126 for Fresh, on the exposed Object cohort. The paired development penalty was -11.1 percentage points.
- Failed replication: on frozen confirmation, C2 achieved 76/140 and Fresh achieved 77/140, a -0.7 percentage-point null contrast. C2 therefore has different evidential roles across cohorts, and the development C2 penalty must not be presented as frozen or replicated.
- Exposed grid characterization: on the task-0 5-by-5 grid, the gripper-horizon-1 success rates across arm horizons 1, 2, 4, 8, and 16 were 58%, 52%, 56%, 52%, and 58%; the corresponding gripper-horizon-2 rates were 60%, 62%, 62%, 66%, and 60%, while gripper-horizon-16 rates were 80%, 88%, 94%, 90%, and 84%. Across tasks 1–9, ARM2_GRIP16 versus ARM2_GRIP2 was 122/180 versus 96/180, ARM4_GRIP16 versus ARM4_GRIP4 was 128/180 versus 112/180, and ARM8_GRIP16 versus ARM8_GRIP8 was 123/180 versus 114/180. These are exposed-development patterns, not confirmation.

Together, these constructions answer the narrow objection that Reverse20 is the only condition showing fresh-gripper sensitivity. They do not establish that every fresh-gripper executor collapses, as shown directly by the frozen C2 null.

### Sparse coherent-horizon evidence

Each evaluated cell reports successes/N, success rate, query rate, and, when available, the paired delta relative to H16 with its paired-bootstrap 95% CI. A dash means not evaluated; the rows are distinct cohorts and must not be rendered as a rectangular cross-cohort experiment.

| Actual cohort | H8 | H13 | H16 | H32 |
|---|---|---|---|---|
| ACT Object development, 126, `EXPOSED_DEVELOPMENT` | 82/126, 65.1%, q=0.1263; -4.8 pp `[-12.7, 3.2]` | — (not evaluated) | 88/126, 69.8%, q=0.0652; reference | 76/126, 60.3%, q=0.0330; -9.5 pp `[-19.0, 0.0]` |
| Former confirmation cohort, 140, `POST_HOC_ON_EXPOSED_COHORT` | 100/140, 71.4%, q=0.1264; +5.0 pp `[-1.4, 12.1]` | — (not evaluated) | 93/140, 66.4%, q=0.0643; reference | — (not evaluated) |
| Gate M, 130, `HELD_OUT_PREREGISTERED` | — (not evaluated) | 95/130, 73.1%, q=0.0797; +2.3 pp `[-3.1, 7.7]` | 92/130, 70.8%, q=0.0651; reference | — (not evaluated) |

H13 versus H16 has nine positive LOTO deltas but both bootstrap intervals include zero, so it does not satisfy the strict stable-positive rule. The supported wording is a competitive moderate coherent band, not an H13, H8, or H16 optimum.

### SmolVLA factual boundary and absolute-performance context

ACT and SmolVLA may be described as two action-chunk-generating policies. ACT may be described as an ACT-style chunk-regression or latent-variable imitation architecture. SmolVLA may be described as using a VLM backbone with a flow-matching action expert to generate continuous future action chunks. Architecture, training objective, pretraining and data, action representation and normalization, chunk length, checkpoint, and absolute performance all change simultaneously, so no causal attribution to flow matching is supported.

| Suite | Historical standard baseline | Coherent-H16 paired subset | Raw direction |
|---|---:|---:|---|
| Spatial | 85/100 (85.0%) | 27/40 (67.5%) | paired subset lower |
| Object | 93/100 (93.0%) | 38/40 (95.0%) | paired subset higher |
| Goal | 78/100 (78.0%) | 27/40 (67.5%) | paired subset lower |
| Long / LIBERO-10 | 42/100 (42.0%) | 20/40 (50.0%) | paired subset higher |

“Our SmolVLA absolute rates differ from the larger standard evaluation in both directions across suites; our componentwise conclusions use only within-protocol paired comparisons and never subtract against those external aggregate rates.” The cause of these absolute-rate differences is not established.

## 4. Forbidden claims

- “Source age and chunk offset are independently varied.”
- “Same-target makes source age and chunk offset orthogonal.”
- “H16 is globally optimal.”
- “H8 is globally optimal.”
- “H13 is globally optimal.”
- “The gripper carries all temporal benefit universally.”
- “Flow matching causes the SmolVLA null.”
- “SmolVLA has no temporal asymmetry.”
- “Component-specific timing is universally non-exploitable,” or “the effect is not exploitable.”
- “CARE works.”
- “The fixed two-clock executor works generally.”
- “RoboTwin proves there is no temporal effect.”
- Any coherent-horizon label that converts changing, statistically uncertain point-estimate orderings into a declared optimum.
- Any same-cohort interaction inferred by directly comparing ARM4_GRIP32 with ARM16_GRIP32, because those conditions were evaluated on different cohorts.
- Any subtraction of historical SmolVLA standard-baseline rates from the paired-protocol rates as if they shared a cohort.

## 5. Evidence map

- Central frozen ACT same-target asymmetry: `act_same_target_fo20_vs_reverse20_confirm140`, supported by `act_same_target_fo20_vs_fresh_confirm140` and `act_same_target_fullold20_vs_fo20_confirm140`; the additional `act_same_target_fullold20_vs_reverse20_confirm140` and `act_same_target_factorial_interaction_confirm140` entries are explicitly post-hoc analyses of the frozen outcomes.
- Query-schedule qualification for the central contrast: `act_same_target_fo20_vs_reverse20_confirm140` plus `fo20_reverse20_query_budget_audit` at the top level of `numbers.json`.
- Fresh-gripper construction packet: `act_same_target_fo20_vs_reverse20_confirm140`, `act_c2_vs_fresh_dev126`, `act_c2_vs_fresh_confirm140`, `act_h16_vs_c2_confirm140`, `act_static_grid_task0_context`, `act_landscape_arm2_grip16_vs_arm2_grip2`, `act_landscape_arm4_grip16_vs_arm4_grip4`, and `act_landscape_arm8_grip16_vs_arm8_grip8`.
- ACT exposed landscape concentration and plateau: `act_landscape_arm2_grip16_vs_arm2_grip2`, `act_landscape_arm4_grip16_vs_arm4_grip4`, `act_landscape_arm4_grip32_vs_arm4_grip4`, `act_landscape_arm4_grip32_vs_arm4_grip16`, and `act_landscape_arm8_grip16_vs_arm8_grip8`.
- Fixed-clock executor falsification: `act_fixedclock_arm16_grip32_vs_h16_dev126`, `act_fixedclock_h32_vs_h16_dev126`, and `act_fixedclock_arm16_grip32_vs_h32_dev126`.
- Event-triggered executor falsification and controls: `act_gate_m2_vs_h16_heldout130`, `act_gate_m2_vs_h13_heldout130`, `act_gate_m2_vs_shuffled_heldout130`, `act_coherent_h13_vs_h16_heldout130`, `act_gate_shuffled_vs_h16_heldout130`, and `act_gate_h13_vs_shuffled_heldout130`.
- Moderate coherent-horizon band: `act_coherent_h8_vs_h16_dev126`, `act_coherent_h8_vs_h16_posthoc140`, `act_coherent_h13_vs_h16_heldout130`, and `act_fixedclock_h32_vs_h16_dev126`.
- Strongest ACT effect does not reproduce in SmolVLA: `smolvla_arm8_grip16_vs_h8_pooled` and its four per-suite entries, reinforced by `smolvla_h16_vs_h8_pooled`, `smolvla_arm8_grip16_vs_h16_pooled`, and their per-suite entries.
- Second SmolVLA non-replication: `smolvla_arm4_grip32_vs_arm4_grip4_pooled` and its four per-suite entries.
- SmolVLA absolute-performance boundary: the four `smolvla_standard_baseline_*` context entries together with the H16 counts encoded in `smolvla_absolute_performance_context`.
- RoboTwin feasibility only: `robotwin_feasibility_600_context` and `robotwin_fo1s_vs_newest_100`.

## 6. Main-paper candidate results

`MUST_MAIN`

- Frozen FO20 versus Reverse20 same-target result, including the `MATCHED_UP_TO_EPISODE_LENGTH` query qualification.
- Tested-executor falsification: fixed-clock ARM16_GRIP32 versus H16 and held-out Gate M M2 versus its controls.
- SmolVLA pooled non-replication of ARM8_GRIP16 versus H8, with query matching and a concise per-suite direction summary.

`SHOULD_MAIN`

- Post-hoc factorial support on frozen outcomes and the C2 development-to-confirmation non-replication.
- Selected exposed ACT landscape contrasts, visibly labeled development characterization.
- Sparse coherent-horizon evidence establishing a competitive moderate band without a unique optimum.
- Pooled SmolVLA ARM4_GRIP32 versus ARM4_GRIP4 robustness result.

`SUPPLEMENT_ONLY`

- Full per-task and LOTO results, all bootstrap details, the complete query-count-difference distribution, and full grids.
- SmolVLA per-suite contrast panels and the separate historical absolute-performance context.
- RoboTwin negative feasibility and floor-effect packet.

## 7. Supplementary plan

Tier A, claim-critical evidence:

- Complete frozen same-target contrasts, the explicitly post-hoc factorial decomposition and interaction, discordances, paired and task-cluster intervals, LOTO values where available, and the FO20/Reverse20 query-budget audit.
- C2 development characterization, frozen non-replication, and H16-versus-C2 decomposition.
- Fixed-clock falsification, all six Gate M contrasts, the corrected task-cluster bootstrap seed provenance, and the sparse coherent-horizon table.
- SmolVLA pooled and per-suite paired results, plus the historical standard-baseline context shown only as a separate protocol table.

Tier B, reproducibility and full grids:

- Full exposed ACT landscape, task-0 static grid, task/state manifests, seed rules, checkpoints, episode limits, query rates, and exposure annotations.
- Full cohort/exposure ledger, exact raw-artifact provenance for internal assembly, and anonymous-release transformations.

Tier C, RoboTwin feasibility:

- The 600-cell, five-task, six-method, 20-seed feasibility study; zero technical reruns; pooled NATIVE_ACT 19/100; severe floor effect; and FO_1S-versus-NEWEST 96/100 ties. Label the result `NO_SIGNAL` and interpret it only as insufficient statistical dynamic range.

Omit entirely:

- Abandoned adaptive-method variants, exploratory sweeps that neither define a final contrast nor establish cohort exposure, implementation-debug rollouts, and historical candidate rankings superseded by the fixed-clock discriminator and final Gate M.
- Any method-success narrative for CARE/M2 or any cross-cohort pseudo-interaction between ARM4_GRIP32 and ARM16_GRIP32.

## 8. Proposed three titles

1. `Same-Target Probes Reveal Component-Dependent Temporal Sensitivity in Action-Chunk Execution` — preferred. It states the measurement contribution without implying a universal method or impossibility result.
2. `Measurable but Hard to Exploit: Component-Dependent Temporal Sensitivity in Action-Chunk Execution` — acceptable if the abstract immediately limits “hard to exploit” to the tested fixed-clock and event-triggered interventions.
3. `Measurable but Not Exploitable: Component-Dependent Temporal Sensitivity in Action-Chunk Execution` — not recommended in its current form. “Not Exploitable” is too absolute because the evidence falsifies specific executor hypotheses, not every possible intervention.

## 9. Reviewer attack map

- No method or performance gain: **answered as positioning, genuine limitation for a methods claim.** The paper is a measurement and diagnostic study. Controlled fixed-clock and shuffled-trigger comparisons make the negative method result informative, but no performance contribution is claimed.
- Positive landscape evidence is exposed development: **partially answered, genuine limitation.** Every landscape entry is labeled `EXPOSED_DEVELOPMENT`; the central same-target asymmetry is separately supported by frozen confirmation. The broad landscape cannot be relabeled confirmation.
- Only one strong policy/suite signal: **partially answered, genuine limitation.** The strongest frozen positive signal is ACT on the selected Goal plus LIBERO-10 cohort, while SmolVLA provides a controlled non-replication rather than a second positive signal. The conclusion is explicitly policy-dependent and non-universal.
- Why H16: **answered for control selection, partially answered as a scientific optimum.** H16 is a coherent reference inherited from the completed program, not a globally optimal horizon. H8 and H13 are competitive in their evaluated cohorts; H32 is weaker on the 126-block cohort; the evidence remains sparse and does not identify a unique optimum.
- Reverse20 is a degenerate condition: **partially answered, genuine limitation.** Similar fresh-gripper sensitivity appears in exposed C2 and grid constructions, so the pattern is not unique to Reverse20. However, frozen C2 versus Fresh is null, proving that the broader construction-dependent claim must remain qualified.

Additional unresolved limitations are the episode-length-driven difference in realized FO20/Reverse20 query totals; the absence of a rectangular H8/H13/H16/H32 evaluation on one untouched cohort; simultaneous architecture, objective, data, representation, chunk-length, checkpoint, and performance changes in the ACT-to-SmolVLA comparison; unknown causes of the SmolVLA absolute-rate differences; and the RoboTwin floor effect. No claim-freeze wording resolves these limitations, and none is converted into a new experiment proposal.

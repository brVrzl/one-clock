# ICRA 2027 direction reset: constrained divergence (Brainstorm pass 2)

## Workflow record

- **Skill / pass:** `scientific-brainstorming`, pass 2, reopened after the first literature audit.
- **Prompt:** Generate formulations that survive the located literature; repair wording-only failures, combine complementary gaps, and abandon occupied ideas. This is an independent post-check ideation round, not a vote or a selection.
- **Located constraints:** Global adaptive horizons (AAC/AutoHorizon/PACE/HiPolicy), whole-chunk async execution (RTC/REMAC/FutureRTC/DEFLECT), and slow semantic policy plus fast tactile residual (PhaForce/RETAF/TouchWorld/T-Rex) are occupied.
- **Criteria declared before any later selection:** measurable phenomenon; distinct causal unit from a shared chunk/horizon; equal-compute comparison; a strong null/negative; first discriminative test within 72 hours; works despite RoboTwin baseline uncertainty. No scorecard or winner is implied here.

## Refined register

| ID | Idea | Assumption | Falsifiable prediction | Key rival / rejection condition | 72-hour discriminative test |
|---|---|---|---|---|---|
| R01 | **Validity-field execution:** predict a validity map over future time × action subspace; retain valid cells and replace invalid ones. | A component's action validity can be estimated separately from policy confidence. | Map-calibrated replacement gain exceeds global entropy/horizon at matched policy queries. | Map collapses to a single shared horizon or fails to predict intervention benefit. | Build counterfactual labels from already buffered queries; test calibration and selective replacement offline/LIBERO. |
| R02 | **Recoverability-budgeted partial replanning:** refresh a subspace only if its remaining action suffix is unlikely to recover after an observed deviation. | Recoverability, not age alone, controls the value of a refresh. | A recoverability trigger has higher success/query than age, entropy, and periodic/global replan. | Recoverability cannot predict a different decision from age/uncertainty. | Controlled deviations at several chunk positions; estimate suffix recovery under hold/replan interventions. |
| R03 | **Subspace-staleness law:** establish that action subspaces have phase-dependent validity horizons, then use the minimum intervention needed. | Observed LIBERO asymmetry is part of a broader causal law, not a quirk. | Arm/gripper/hand age-response curves cross by phase/embodiment; fixed FO can therefore succeed in one setting and fail in another. | One global source-age curve explains controlled interventions; RoboTwin null persists in stable tasks. | Explicit time × group × phase factorial replay with native aggregation retained. |
| R04 | **Compute-aware action refresh allocation:** given a delayed VLA, decide whether a new expensive inference should update arm, end-effector, or contact residual. | Existing outputs/buffers permit partial action reuse without retraining the entire VLA. | Selective allocation dominates full re-query and global continuation at fixed latency/call budget. | Cross-component coherence loss cancels compute advantage. | Simulate measured pi0.5 inference delays in an existing executor and compare fixed equal-call schedules. |
| R05 | **Coherence-constrained partial refresh:** identify when a refreshed local action must synchronize with retained components, using geometric/contact constraints. | Partial refresh is valuable only if coupling is explicitly measured. | Coherence constraint removes the failure mode of naive partial replacement and shows where independence is invalid. | Constraints require nearly full recomputation, eliminating partial-refresh benefit. | Arm-hold/gripper-refresh and converse conditions near grasp/insert; quantify discontinuity/contact errors. |
| R06 | **Cross-modal freshness routing:** track age/novelty of wrist/workspace/contact streams and route each to the action subspace it can correct. | Sensor freshness, not only action age, creates component-specific invalidation. | Delay/mask effects form a sparse sensor-to-action causal matrix. | Uniform fusion/requery matches it; collection synchronization dominates results. | Independent artificial delay/mask of camera and contact signals in replay; measure action-group replacement gain. |
| R07 | **Failure-response taxonomy:** classify chunk failures by remedy value (continue, local repair, partial refresh, global replan), not visual symptom. | Failure modes have distinct counterfactual optimal response. | A small diagnostic matrix predicts response choice across tasks, including a null class where no refresh helps. | The same remedy wins universally / labels are not reproducible. | Perturbation suite with direct intervention outcomes, before learning any classifier. |
| R08 | **Role-conditioned bimanual temporal contract:** refresh each arm/finger group by current support/manipulate/contact role and synchronize only at coupling events. | Bimanual coupling is intermittent and role-driven. | Role-dependent timing helps under unilateral delay while preserving contact geometry at handoff/insertion. | Global timing is as good or coupling events are too pervasive. | RoboTwin bimanual delay intervention on a verified-fidelity task. |
| R09 | **Hold/refresh action interface:** action chunks emit commitment, local target, expiry, and refresh permission, making partial replanning an explicit representation. | Dense action vectors hide the execution decision. | Interface enables controllable partial update where post-hoc masks cannot. | It is reducible to a gate with no generality/result. | Small ACT head/interface prototype only after R03/R04 diagnostic supports the premise. |
| R10 | **Source-age calibration benchmark:** a diagnostic benchmark/protocol that exposes action-subspace temporal asymmetry across simulators and embodiments. | Current benchmarks conceal timing with aggregate success. | The protocol discriminates global horizon, native aggregation and subspace interventions reliably. | Effects are simulator-/task-specific and not reproducible. | Package the existing preregistered LIBERO test plus one stable RoboTwin task (after baseline reconciliation). |

## Abandoned, not repaired

- **DCTA alone:** literature and RoboTwin null leave it as a weak gating/ablation story.
- **Global adaptive horizon/multi-rate action policy:** occupied by AAC, AutoHorizon, PACE, HiPolicy and related work.
- **Slow arm + fast tactile correction:** occupied by RETAF, PhaForce, TouchWorld, T-Rex, FAWAM and M2-ResiPolicy.
- **Generic event-triggered VLA:** event timing alone is not distinct from existing event/memory/phase-aware systems.

## Deliberate alternative (minority view)

**Idea:** The real paper should be a tactile dexterity/force-control paper unrelated to action chunking. **Why retain as dissent:** the available RH56 and force signals make it visually strong. **Why not advance now:** the 2026 tactile literature makes a generic architecture claim crowded; without a novel sensing capability or benchmark, its gap appears weaker than the diagnostic evidence already in hand.

## Status

All R01–R10 are **ideas** with `mixed` / `search-incomplete` evidence status. The final Nature-style stress test must test their precise claims and nearest papers before any reviewer handoff.

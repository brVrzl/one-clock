# Narrative spine for approval

Date: 2026-09-03 (Asia/Shanghai)

This is a pre-rewrite narrative plan, not manuscript prose.

## Title evaluation

### Candidate A

`Which Action Components Can Go Stale? Temporal Sensitivity in ACT Action-Chunk Execution`

Clear before the paper's terminology and framed around the scientific question. The ACT qualifier prevents a cross-policy claim, and the title does not imply a superior executor. The question form is slightly broader than the primary confirmatory arm-versus-gripper result, but the later Object characterization supports discussing translation and rotation if their development-cohort status remains clear in the paper.

### Candidate B

`Stale Translation, Not Stale Rotation: Component-Dependent Temporal Sensitivity in ACT`

Concrete and memorable, but too strong for the evidence hierarchy. The translation-versus-rotation contrast is statistically strong yet comes from reviewer-directed development-cohort characterization rather than the primary confirmatory cohort. “Not stale rotation” also risks turning a small, uncertain Fresh-relative cost into a categorical absence of cost.

### Candidate C

`Same-Target Probes: Component-Resolved Temporal Sensitivity in ACT Action-Chunk Execution`

Accurate in scope, but specialized terminology appears before the reader understands the intervention. Leading with the probe name makes the paper sound more like a method paper even though its main contribution is measurement and characterization.

### Candidate D

`Component-Dependent Sensitivity to Stale Predictions in ACT Action-Chunk Execution`

This is the recommended title. It uses ordinary language, states the measured result, keeps ACT scope explicit, and does not claim a globally superior executor or cross-policy generality. It also rests on the primary arm-versus-gripper confirmation rather than requiring the development-only translation-versus-rotation result to carry the title.

## Figure-1 staircase decision

The five Object-126 points are exactly comparable, but the proposed staircase belongs in **Fig. 2**, not Fig. 1. Fig. 1 should remain centered on the same-target concept and the primary 140-block confirmatory result. In Fig. 2, the ordered Object-cohort reference can sit with R1A/R1B characterization, provided the graphic distinguishes audited historical anchors from new reviewer-directed component probes and does not present the five points as levels of one preregistered factorial. It should state that translation alone nearly reproduces full-arm degradation, that rotation has little detectable cost relative to Fresh under the existing paired comparison, and that the gripper-above-Fresh ordering on Object does not establish a generically beneficial effect.

## Interaction placement

**`INTERACTION_SUPPLEMENT_ONLY`**

The R1C risk-difference interaction formula was frozen before its outcomes, but only the +9.29 pp point estimate is canonical; there is no interaction-specific p-value or interval. The original 140-block interaction is a post-hoc supporting result whose small-cluster sensitivity crosses zero, and R1D has no named canonical interaction estimate. Positive arithmetic signs in two reported cohorts are insufficient for a replication claim, and interaction is not required for R1C's central identification result that the H16 advantage survives query-schedule matching.

## 15-sentence narrative spine

1. An action-chunk policy predicts commands for future control steps, but its executor must decide how old a prediction can become before it is refreshed.
2. Existing executors generally apply one temporal setting to translation, rotation, and gripper commands even though these components may tolerate stale predictions differently.
3. At control step `t`, our diagnostic can execute one component from the prediction queried at `t` with offset 0 and another from the chunk queried 20 steps earlier with offset 20, so both commands target the same physical step.
4. In the preregistered 140-block ACT evaluation, fresh-arm/stale-gripper execution achieved 83/140 successes versus 38/140 for stale-arm/fresh-gripper execution, a +32.14 percentage-point difference at 1.00 s.
5. On a separate Object development cohort, the fresh-arm branch remained above the stale-arm branch at every tested age from 0.10 to 1.60 s, with the largest observed separations from 0.60 to 1.00 s but no evidence that 1.00 s is an optimum.
6. Within the arm on that Object cohort, translation-stale success was 11/126 and rotation-stale success was 53/126, a -33.33 percentage-point paired contrast under the translation-minus-rotation convention.
7. Same-target dispersion ranked rotation above translation, whereas behavioral sensitivity ranked translation above rotation, so this simple disagreement metric predicts the within-arm ordering incorrectly.
8. Because holding the physical target fixed couples older observations to longer-lookahead predictions, the probes do not separate those factors, and the tested diagnostics do not identify a positive causal mechanism.
9. A dense-query factorial showed that coherent H16's advantage persists when every condition queries the policy at every step and unused predictions are discarded, removing query schedule as the explanation under the deterministic ACT evaluator.
10. At fixed arm cadences, extending gripper commitment improved success by +4.667 percentage points over H4 and by +5.778 percentage points over H2 at nearly matched policy-query rates.
11. Coherent H16 nevertheless remained the strongest overall operating point at 357/450 successes, so the component-resolved schedules are not globally superior executors.
12. The Track-A gains were concentrated in LIBERO-10, while suite identity, baseline ceiling, and task semantics covaried too strongly to identify the cause of that heterogeneity.
13. The behavioral claims apply to the evaluated ACT checkpoints; SmolVLA cannot provide a physically matched behavioral replication because its training and chunk timebase is not identifiable from the available provenance.
14. Same-target probes query the policy once per executed environment step and are diagnostic interventions, not deployment executors or evidence that dense querying is computationally preferable.
15. The narrow takeaway is that temporal source and execution choices have component-dependent behavioral effects in the evaluated ACT policies, while their mechanism and generality remain unresolved.

## Three alternative three-sentence Abstract openings

### Opening 1: executor decision

How old can a predicted action be before it should be refreshed? An ACT policy predicts a chunk of future commands, and we compare commands aimed at the same physical control step while changing which prediction source supplies the arm or gripper. In a preregistered 140-block evaluation, fresh-arm/stale-gripper execution exceeded stale-arm/fresh-gripper execution by 32.14 percentage points at a source age of 1.00 s.

### Opening 2: heterogeneous components

Translation, rotation, and gripper commands need not tolerate stale predictions equally. We test this by holding the executed physical target step fixed while supplying different action components from newer or older ACT predictions. The primary evaluation found a 32.14 percentage-point arm-gripper asymmetry, and a separate Object development cohort found translation staleness substantially more damaging than rotation staleness.

### Opening 3: prediction reuse

Action-chunk policies forecast many future robot commands from each observation, but execution rules usually reuse those predictions uniformly across action components. We align newer and older predictions to the same physical target step to measure how source age changes the success of arm and gripper commands. The resulting ACT evaluations show robust component dependence, while query-matched controls and deployment-oriented schedules separate this diagnostic effect from the practical choice of how often to replan.

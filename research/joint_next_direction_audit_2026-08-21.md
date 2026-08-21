# Joint Next-Direction Audit — 2026-08-21

**PRACTICAL PRIMARY DIRECTION:** PROCEED WITH SIMPLIFICATION — test a
control-semantic **scalar** temporal ensemble with one shared source-weight
vector; do not make independent group routing the minimum method.

**FALLBACK:** consistency-constrained translation/rotation/gripper residual,
only if dense Gate-3A1 proves group oracle headroom under predeclared
constraints; otherwise stop the group branch.

**NOVELTY LEVEL:** MARGINAL BUT DEFENSIBLE, conditional on isolated and
repeatable closed-loop gains over exact ACT and CogACT baselines.

**EXPECTED IMPLEMENTATION RISK:** LOW for the scalar method; MEDIUM for the
conditional group residual.

**EXPECTED RESULT PROBABILITY:** MEDIUM, qualitative and not calibrated; sparse
offline evidence is positive against cosine but tied with age-exponential
weighting.

**BIGGEST SCIENTIFIC RISK:** teacher-forced semantic error does not predict
closed-loop success.

**BIGGEST NOVELTY RISK:** reviewers reasonably describe the method as “CogACT
with a robotics-aware distance.”

**CHEAPEST DISCRIMINATING TEST:** dense held-out ACT cache on the existing 41
validation and 41 test episodes, with exact scalar baselines and frozen
validation selection.

**NEXT EXPERIMENT IF APPROVED:** Gate-3A1 dense temporal evidence funnel; 12,294
planned ACT calls, one per eligible 10 Hz demonstration step, before any policy
wrapper or rollout.

**KILL CONDITION:** stop the temporal-selection direction if dense temporal
aggregation has no robust held-out advantage over newest-only; stop the proposed
semantic method if it does not beat the strongest query-matched scalar baseline;
stop the group contribution if its advantage disappears under control-aligned
metrics or modest consistency constraints.

## Decision

**PROCEED WITH SIMPLIFICATION.** The provisional Group-Aware Temporal Action
Ensemble (“GATE”) is rejected as the immediate method. Its conceptual delta is
distinguishable but currently unsupported: validation-selected deployable group
weighting does not beat the scalar ensemble, and the consistency gate yields no
held-out benefit. The fastest credible path is to determine whether the surviving
scalar control-semantic kernel produces dense and then closed-loop improvement.

This is a gate decision, not authorization to implement the policy method. No
dense cache, policy modification, or rollout was run in this phase.

## 1. Verified evidence relevant to the direction

The source of truth remains [the verified fact sheet](verified_fact_sheet.md),
supported by the [zero-trust report](icra2027_zero_trust_reaudit_2026-08-21.md).
The observations below are separated from interpretations.

### Facts carried forward from the zero-trust audit

1. Execution horizon affects success for the audited frozen ACT checkpoint on
   LIBERO task 0. This does not establish a universal horizon mechanism.
2. A post-hoc `(arm=4, gripper=16)` static pair is positive on task 0 but does
   not establish ten-task generalization and is not uniformly query matched.
3. The direct selective retain/refresh experiment is negative relative to global
   replacement: -0.26, -0.20, and -0.29 absolute success at queries 4, 8, and 16,
   respectively. This falsifies that implementation, not every group-aware
   method.
4. Gate-2B uses fixed fractions of the 280-step environment limit, not semantic
   phases; its optima are often broad, and the earlier late group point was
   misreported.
5. `Y_refresh` labels are learnable but are not a useful execution-horizon target
   in the tested pipeline.
6. LIBERO actions are translation `[0:3]`, axis-angle rotation `[3:6]`, and
   gripper `[6]`; the environment consumes gripper sign. Continuous gripper
   magnitude MSE is therefore not the primary control metric.
7. Saved sparse overlapping predictions show temporal diversity and offline
   ensemble benefit, but they are teacher-forced and are not dense ACT temporal
   ensembling.

### New Gate-3A0 observations

The new read-only analysis is reproducible from
[`gate3a0_sparse_group_consistency.py`](audit_tools/gate3a0_sparse_group_consistency.py)
and its compact
[`JSON output`](audit_outputs/gate3a0_sparse_group_consistency.json). It uses
55,634 sparse overlap targets overall, selects parameters on 5,126 validation
targets, and evaluates 5,118 targets from 41 held-out episodes covering all ten
tasks. Source spacing is dominated by 25 dataset steps but contains additional
irregularly placed sources. The unit of inference is the episode; all intervals
below are paired episode bootstraps.

Primary offline metric:

\[
L_{sem}=\frac{3L_{trans}+3L_{rot}+L_{grip}}{7},
\]

where `L_trans` is component-normalized translation MSE, `L_rot` is squared
SO(3) geodesic angle normalized by the audited rotation variance scale, and
`L_grip` is sign error. Equal-group and arm/gripper-balanced summaries remain
sensitivity metrics, not privileged objectives.

| Scalar method | Held-out `L_sem` | Gripper sign error | Observation |
|---|---:|---:|---|
| Newest only | 0.65754 | 0.27941 | Reference |
| Uniform raw ensemble | 0.65818 | 0.27237 | No primary-metric gain over newest |
| Validation-selected age-exponential, raw aggregation | 0.62702 | 0.25967 | Strong scalar baseline |
| Released CogACT cosine rule, `alpha=0.1` | 0.65313 | 0.26221 | Exact released rule, not retuned for this checkpoint |
| Validation-selected CogACT cosine, raw aggregation | 0.63401 | 0.27823 | Fair tuned cosine baseline |
| Validation-selected control-semantic similarity, raw aggregation | **0.62471** | **0.25401** | Best tested deployable scalar method |
| Same semantic weights with SO(3)+sign aggregation | 0.62524 | 0.25772 | Aggregation change does not help |

Source: `test_methods` and `validation_selection` in the Gate-3A0 JSON.

Paired comparisons:

- Semantic similarity versus age-exponential raw aggregation is -0.00231 in
  `L_sem`, CI [-0.00715, 0.00211]: **statistically unresolved**.
- With the aggregation operator held semantic for both methods, semantic
  distance beats validation-tuned CogACT cosine by -0.00894, CI
  [-0.01613, -0.00203]. This isolates a sparse offline lead for the distance
  kernel, not a control benefit.
- Changing the semantic-similarity output from raw interpolation to SO(3)+sign
  aggregation changes `L_sem` by +0.00053, CI [-0.00070, 0.00174]. The more
  elaborate aggregation operator is not justified as a core contribution.

Deployable group variants are negative or null:

| Method minus scalar semantic similarity | Mean `L_sem` difference | 95% episode-bootstrap CI | Audit result |
|---|---:|---:|---|
| Arm/gripper group similarity | +0.00460 | [0.00267, 0.00724] | Worse |
| Translation/rotation/gripper similarity | +0.00771 | [0.00019, 0.01302] | Worse |
| Validation-selected consistency-gated semantic-three residual | +0.00058 | [-0.00060, 0.00207] | Tied; no benefit |

The validation-selected consistency gates use a high group temperature (10.0)
and tight scale (0.01), behavior close to the scalar fallback. This is empirical
evidence against making the group residual the minimum method.

Teacher-forced hard oracles retain control-semantic headroom:

| Oracle | Held-out `L_sem` | Improvement from scalar oracle | 95% CI for difference |
|---|---:|---:|---|
| Complete-action scalar source | 0.48137 | — | — |
| Arm/gripper independent source | 0.46698 | 0.01439 | [-0.01691, -0.01070] as group minus scalar |
| Translation/rotation/gripper independent source | 0.41665 | 0.06472 | [-0.07095, -0.05758] as group minus scalar |

The semantic-three oracle's `L_sem` is 0.48137 with zero source-age disparity,
0.46807 with at most 16 dataset steps (1.6 s), 0.43548 with at most 25 steps
(2.5 s), and 0.41665 unconstrained. Its selected mixed-source fraction grows
from 0 at zero disparity to 0.341, 0.828, and 0.916 for those settings. This is
headroom, not a method: it sees the demonstrated target, uses sparse irregular
sources, and may select against demonstration noise.

### Interpretation

- Temporal diversity survives the control-aligned re-analysis.
- The new deployable evidence favors **shared** source weights.
- Control-semantic similarity is a lead because it beats tuned cosine offline;
  it is not yet better than age-exponential with statistical confidence.
- SO(3)+sign aggregation is not supported as an improvement and should remain an
  ablation.
- Semantic-three oracle headroom is not eliminated by replacing continuous
  gripper MSE, but no deployable group rule captures it.

### Falsifiable hypothesis and rivals

**Observation:** a validation-selected control-semantic similarity kernel has a
sparse held-out offline advantage over validation-selected full-vector cosine
when the aggregation operator is held fixed.

**Hypothesis H1:** raw full-vector cosine miscalibrates temporal source trust in
heterogeneous robot actions because translation scale, rotation geometry, and
binary gripper mode do not share one Euclidean direction. A shared
control-semantic kernel will therefore select/weight better temporal candidates
and improve closed-loop task success without independent group recomposition.

Falsifiable predictions:

1. The kernel beats tuned cosine on dense held-out actions, especially on rows
   where candidates disagree in gripper sign or rotation but not translation.
2. Its advantage remains when both methods receive identical candidate windows,
   query counts, aggregation, and validation-tuning budget.
3. In matched rollouts, the kernel reduces control-relevant gripper/rotation
   failures and increases success; offline gain without success gain falsifies
   the policy-performance part of H1.

Incompatible observations:

- dense candidates eliminate the advantage;
- age-exponential or retuned cosine matches the kernel on all affected strata;
- the effect is confined to demonstration noise or one task;
- closed-loop success is unchanged or worse.

Rival explanations include extra hyperparameter flexibility, correlation with
source age, local action smoothness, task/episode imbalance, near-zero vector
instability in cosine, and a target-specific benefit from demonstration noise.
The dense factorial comparison is the smallest experiment that separates the
kernel from the first five rivals; only a later matched rollout addresses the
last one and control relevance.

## 2. Adversarial audit of ChatGPT's GATE proposal

| Attack question | Evidence-bounded answer |
|---|---|
| Why should group adaptation improve success? | No direct evidence. A control-semantic oracle has offline headroom, but deployable group similarities are worse and no closed-loop benefit exists. |
| Why did naive group retention lose 20–29 points? | It mixed action components from predictions generated under different observations and altered refresh semantics. The experiment falsifies that rule. It does not isolate whether the cause is mode inconsistency, staleness, action discontinuity, or another implementation interaction. |
| Is the negative result evidence against the premise? | Yes, against unconstrained mixed-generation execution; no, not a proof that every bounded group residual fails. It raises the burden of proof substantially. |
| Could scalar similarity be enough? | It is currently more promising: scalar semantic similarity is the best tested deployable sparse method. |
| Could exact ACT temporal ensemble capture most benefit? | Yes. Age-exponential and semantic similarity are statistically tied in sparse Gate-3A0. Dense exact ACT comparison is mandatory. |
| Is group headroom a weighting artifact? | Not entirely: semantic-three oracle headroom survives dimension-weighted control metrics. The arm/gripper advantage becomes much smaller, and all oracle results remain teacher-forced. |
| Does gripper sign semantics remove the lead? | It removes the justification for magnitude MSE and reduces the arm/gripper story, but a semantic-three oracle lead remains. Deployable group weighting worsens gripper sign error. |
| Is translation/rotation/gripper preferable? | Yes for metrics and diagnostic grouping because it follows the controller contract. It is not yet preferable for independent source selection. |
| Can group mixing leave the policy manifold? | Plausibly, but unproved. Nearest-full-source distance is only a diagnostic and has not predicted rollout failure. |
| Can a consistency penalty detect harm? | UNKNOWN. The tested nearest-source gate does not improve held-out offline error. No score has been calibrated against direct compositional rollout failures. |
| Is boundary jerk causal? | UNKNOWN. It can be cause, symptom, or harmless correlate. Success and safety must be primary; jerk is secondary. |
| Is a training-free gate sufficiently novel? | Only `MARGINAL BUT DEFENSIBLE` if the constraint is necessary and improves closed-loop results. Current evidence does not meet that bar. |
| Would a neural router reproduce TAS? | Largely yes at the full-action level; adding group outputs alone is an incremental extension with high baseline burden. Do not start there. |
| Would a reviewer call this CogACT plus actuator groups? | Yes, accurately, unless a direct failure mechanism and necessary consistency constraint are demonstrated. |
| Strongest single experiment against that criticism | Query-matched closed-loop factorial comparison of exact CogACT cosine, semantic scalar weights, independent semantic group weights, and constrained group residual across tasks. It is authorized only after the dense oracle funnel passes. |

**Verdict:** reject GATE as the primary method now. Retain the group-consistency
idea as a conditional branch, not as a protected paper story.

## 3. Competing low-risk directions

Ratings are qualitative decisions, not calibrated probabilities.

| Direction | Scientific distinction | Difficulty | Expected upside | Novelty | Baseline burden | Closed-loop risk | Time to decision | Decision |
|---|---|---|---|---|---|---|---|---|
| Control-semantic scalar temporal similarity | Shared weights like CogACT, but translation scale, SO(3), and gripper sign define source similarity | Low | Medium | `MARGINAL BUT DEFENSIBLE` | Moderate | Low–medium | Short | **Primary** |
| Exact/retuned CogACT scalar AAE | Released full-vector cosine weighting | Low | Medium | `TOO CLOSE TO PRIOR WORK` | Low | Low | Short | Baseline only |
| ACT age-exponential ensemble | Exact temporal ensemble | Low | Medium | `NOT NOVEL` | Low | Low | Short | Baseline only |
| Gripper-event safeguard with shared full-action weights | Preserve one joint source distribution; add transition hysteresis/consensus | Low | Medium if failures concentrate at events | `MARGINAL BUT DEFENSIBLE` | Moderate | Medium | Short | Secondary ablation; Gate-3A0 sign vote was not positive |
| Consistency-gated semantic-group residual | Limited group deviation around scalar weights | Medium | Medium–high oracle upside | `MARGINAL BUT DEFENSIBLE` | High | High | Medium | Conditional fallback only |
| Hard full-action semantic selection | Choose one complete cached candidate | Low | Medium | `TOO CLOSE TO PRIOR WORK` (TAS without learning) | High | Medium | Short | Heuristic baseline |
| Boundary/disagreement-aware scalar selector | Penalize discontinuity or disagreement | Low | Medium | `TOO CLOSE TO PRIOR WORK` or crowded by PACE/SGAC/SEAM | High | Medium | Short | Baseline, not headline |
| Source-age trust-region group selector | Permit group freedom only within age disparity | Medium | Medium | `MARGINAL BUT DEFENSIBLE` | High | Medium–high | Medium | Oracle diagnostic before method |
| Learned selector/value-of-freshness head | Predict full-action source or replanning value | High | Potentially high | `TOO CLOSE TO PRIOR WORK` (TAS/DEHP/BCP) | Very high | High | Long | Stop for now |
| Explicit mixture of prediction horizons | Train horizon branches | High | Potentially high | `TOO CLOSE TO PRIOR WORK` (MoH) | Very high | High | Long | Stop for this deadline |

The brainstorming and hypothesis-generation skills were used to keep rival
explanations explicit. The leading observation is compatible with at least:
scale calibration, gripper-mode separation, rotation geometry, a more favorable
temperature grid, age/smoothness correlation, task imbalance, or demonstration
noise. The dense factorial test is designed to discriminate among them.

## 4. Literature overlap

The full comparison and primary links are in
[the joint literature audit](joint_direction_literature_audit_2026-08-21.md).
The hard conclusions are:

- dynamic execution horizon alone is `NOT NOVEL`;
- scalar similarity-weighted overlap is `TOO CLOSE TO PRIOR WORK` because of
  CogACT;
- learned full-action cached selection is `TOO CLOSE TO PRIOR WORK` because of
  TAS;
- explicit prediction-horizon experts are `TOO CLOSE TO PRIOR WORK` because of
  MoH;
- AAC already uses translation/rotation/gripper component uncertainties but
  reduces them to one scalar prefix;
- the bounded search did not locate the exact combination of independent
  semantic-group source weights and an explicit cross-source recomposition
  constraint. This is not a firstness guarantee.

The practical novelty opportunity is not “temporal experts.” It is the
interaction between established temporal aggregation and heterogeneous robot
action semantics, provided a controlled experiment demonstrates necessity.

## 5. What can honestly be claimed as novelty

Provisional, evidence-conditional contribution language:

> Building on ACT temporal ensembling and CogACT adaptive action aggregation, we
> study temporal aggregation under heterogeneous robot action semantics. We use
> a shared temporal source distribution whose similarity kernel respects
> normalized translation, SO(3) rotation, and discrete gripper events, avoiding
> unconstrained cross-source group recomposition.

Only if dense and closed-loop evidence later supports a group residual:

> We show that independent component mixing offers oracle headroom but can
> produce inconsistent joint actions, and introduce a trust-region residual that
> recovers useful group adaptation while remaining near a shared temporal mode.

Claims currently forbidden:

- “one clock does not fit all” as a general policy result;
- “GATE improves robot control”;
- “group-specific timescales cause the benefit”;
- “cross-source inconsistency caused the negative rollout”;
- “the first adaptive ensemble/router/horizon method”;
- any policy gain inferred from the offline Gate-3A0 error.

## 6. Likely reviewer objections

The complete three-reviewer simulation and process limitations are in
[the internal review](joint_direction_internal_review_2026-08-21.md); its
deterministic structural lint is
[here](audit_outputs/joint_direction_peer_review_lint.json).

The convergent objections are:

1. no dense or closed-loop method evidence exists;
2. the scalar delta is very close to CogACT and must be isolated as a distance
   change, not marketed as a new ensemble principle;
3. exact released and validation-tuned CogACT are both required;
4. group routing cannot be the headline while deployable group baselines are
   negative;
5. single-task or single-checkpoint improvement is insufficient;
6. no fixed “5% is enough” rule is defensible before paired rollout variance is
   known.

All three reviewers rate current submission readiness 1/5. They agree that a
clear incremental method can be publishable if the delta is necessary,
query-matched, repeatable, and changes closed-loop task success.

## 7. Proper action grouping

### A. Full action scalar

Use one temporal-source distribution over the complete seven-dimensional action.
This is the primary deployable method because it preserves source coupling and
currently performs best among tested non-oracle variants.

### B. Arm `[0:6]` plus gripper sign `[6]`

Retain for comparison with historical analyses and for a two-group oracle. Do
not treat the gripper as a continuous regression target. This grouping hides the
geometric difference between translation and rotation and is not the preferred
scientific decomposition.

### C. Translation `[0:3]`, rotation `[3:6]`, gripper sign `[6]`

Use as the primary **metric and diagnostic** decomposition. It matches the
audited LIBERO controller semantics. Independent temporal weights are not
authorized unless Gate-3A1 Level 3 and Level 4 pass.

### D. Controller-justified alternatives

For another benchmark or robot, derive groups from its actual action contract
(joint velocity, Cartesian delta, gripper mode, base, torso, etc.). Do not carry
LIBERO groups by analogy. LIBERO is single-arm; the older left/right bimanual
contract does not apply.

## 8. Control-aligned metrics

Primary offline outcomes:

- translation L2 in action units and component-normalized translation MSE;
- SO(3) geodesic rotation error in radians and variance-normalized squared error;
- gripper sign error;
- false gripper transition and missed gripper transition, using the prior
  demonstrated sign only for teacher-forced offline evaluation;
- dimension-weighted semantic error as defined above.

Sensitivity outcomes:

- equal translation/rotation/gripper weighting;
- arm/gripper-balanced weighting, explicitly labeled as giving one gripper
  dimension 50% weight;
- raw normalized-dimension MSE for continuity with earlier reports;
- per-task, temporal-age, and normalized-time decompositions.

Closed-loop outcomes, when later authorized:

- task success is primary;
- episode length, policy queries, wall time/latency, and failure stage;
- gripper false/missed transitions from executed commands;
- boundary velocity/acceleration/jerk as secondary diagnostics, not surrogate
  success;
- safety/collision events where the environment exposes them.

## 9. Cheap sparse analyses completed

Gate-3A0 performed only read-only computations on already saved predictions:

- exact released CogACT cosine and validation-tuned cosine;
- newest, uniform, and validation-selected age-exponential baselines;
- control-semantic scalar similarity;
- raw versus SO(3)+gripper-sign aggregation;
- arm/gripper and translation/rotation/gripper similarity weights;
- jointly validation-selected consistency gates;
- scalar and group hard oracles under control-aligned metrics;
- source-age-disparity oracle frontiers;
- nearest-full-source and teacher-forced boundary diagnostics;
- paired episode-bootstrap comparisons and per-task summaries.

The accompanying
[Gate-3A1 inventory script](audit_tools/gate3a1_inventory.py) and
[JSON](audit_outputs/gate3a1_inventory.json) show that the existing episode-level
split contains 454 episodes total. The validation and test splits each contain
41 episodes (four per task for tasks 0–8 and five for task 9), totaling 6,151 and
6,143 demonstration steps. Dense inference on both therefore requires 12,294
planned ACT calls before retries or resume verification.

## 10. Proposed dense Gate-3A1 — design only

### Objective

Determine whether source-time diversity and the control-semantic scalar kernel
survive exact dense sampling before implementing any execution method.

### Cohort and split

- Use the existing episode-level validation/test split: 41 validation and 41
  held-out test episodes, balanced as recorded in the inventory JSON.
- Query the frozen audited ACT checkpoint once at every demonstration step.
- Use validation episodes for all temperatures, age coefficients, window sizes,
  and optional thresholds. Freeze them before opening test summaries.
- Do not use demonstrated future actions to compute deployable weights.

### Cache contract

Store per episode, resume-safely and without overwriting historical artifacts:

- git SHA, checkpoint/config SHA256, dataset root/digest reference;
- episode/task/split identifiers and observation-frame index;
- policy query index, full predicted chunk, chunk length, action dimension;
- source dataset step, target dataset step, temporal age in steps and seconds;
- preprocessing and temporal-aggregation settings;
- completion marker and artifact SHA256.

The demonstration dataset is 10 Hz: one dataset step is 0.1 s. Later audited
rollouts are 20 Hz: one controller tick is 0.05 s. Never compare an integer age
without also reporting seconds and the originating clock.

### Level 1 — dense temporal diversity and scalar baselines

Compare, with one candidate cache and equal query budget:

1. newest only;
2. oldest valid;
3. uniform raw ensemble;
4. exact ACT temporal ensemble with the audited/released coefficient;
5. validation-selected age-exponential ensemble;
6. released CogACT cosine with `alpha=0.1`;
7. validation-selected CogACT cosine with the same tuning budget as the new
   kernel;
8. control-semantic scalar similarity with raw aggregation;
9. semantic aggregation as a factorial ablation, not an assumed improvement;
10. best validation-selected fixed age.

Report primary and sensitivity metrics, per task, per episode, age/window
sensitivity, and paired episode-bootstrap intervals.

**Level-1 stop:** if no temporal method robustly improves held-out outcomes over
newest-only, stop temporal selection. If temporal aggregation helps but the
semantic kernel does not beat the strongest scalar comparator or isolate the
hypothesized gripper/rotation failure, stop the proposed method and retain it as
a negative baseline.

### Level 2 — contextual scalar headroom

Compare strongest non-oracle scalar method with:

- scalar hard oracle source;
- scalar convex-mixture oracle;
- per-task fixed age and per-sample scalar oracle.

Report absolute and relative error reductions, episode-bootstrap intervals, and
task distributions. If contextual scalar headroom is small, unstable, or
restricted to a few episodes, do not train a selector.

### Level 3 — group freedom

Compare scalar oracle, arm/gripper oracle, and
translation/rotation/gripper oracle under all predeclared metrics. The
dimension-weighted semantic metric is primary. If the advantage exists only
under equal-group or arm/gripper-balanced weighting, stop the group contribution.

### Level 4 — consistency-constrained frontier

Constrain group choices by predeclared source-age disparity and
nearest-full-source semantic distance. Report error, mixed-source fraction,
boundary diagnostics, and constraint occupancy. Use validation quantiles for
any distance threshold; never select a test threshold.

If modest constraints remove the group advantage, stop the group-consistency
paper direction. If robust headroom remains, design one simple residual around
the shared scalar weights before considering a network.

## 11. Oracle funnel and decision statistics

The funnel is sequential to prevent a large router from rescuing a nonexistent
signal:

```text
dense temporal gain?
  no  -> stop temporal selection
  yes -> scalar contextual oracle headroom?
           no  -> keep best fixed/scalar ensemble; no router
           yes -> control-aligned group advantage?
                    no  -> scalar method only
                    yes -> survives consistency constraints?
                             no  -> stop group method
                             yes -> test smallest group residual
```

No arbitrary success threshold is set before variance. Each gate uses effect
size, uncertainty, per-task direction, and practical magnitude. A confidence
interval excluding zero is necessary for the primary offline comparison but not
sufficient for a policy claim.

## 12. Minimum publishable method

Working description: **control-semantic scalar temporal ensemble**. Do not use
the GATE acronym.

For candidate `E_{t,k}` and newest candidate `E_{t,0}`, define

\[
D_{sem}(E_{t,k},E_{t,0}) =
\frac{3D_{trans}+3D_{rot}+D_{grip}}{7},
\]

where translation uses audited scale normalization, rotation uses SO(3)
geodesic distance, and gripper uses sign disagreement. Then

\[
w_{t,k}=\operatorname{softmax}_k(-D_{sem}/T), \qquad
\tilde a_t=\sum_k w_{t,k}E_{t,k}.
\]

One shared `w` is used for the complete action. The minimum method keeps raw ACT
action interpolation because Gate-3A0 did not support the more complex SO(3)+sign
aggregation. An age prior, hard source selection, hysteresis, group residual, or
learned network is an ablation/extension only after independent evidence.

Required baselines:

- newest, oldest, uniform;
- exact ACT temporal ensemble;
- released and validation-tuned CogACT cosine;
- best fixed age and age-exponential;
- PACE-like kinematic boundary heuristic if execution horizons are later varied;
- TAS as literature context and, if feasible, an implementation comparator for
  any learned selector claim;
- unconstrained and consistency-constrained group variants only if group freedom
  passes the dense oracle.

Required ablations:

- raw cosine versus normalized Euclidean versus control-semantic distance;
- translation normalization, SO(3) term, and gripper sign term removed one at a
  time;
- raw interpolation versus semantic aggregation;
- window length/age prior;
- official versus validation-tuned baseline hyperparameters;
- identical policy queries and runtime accounting.

## 13. Fallback direction

The conditional fallback is a **small group residual around the shared scalar
weights**, not independent group routing:

\[
w^g=(1-\alpha_g)w+\alpha_g\bar w^g,
\]

with `alpha_g` bounded by a validation-calibrated source-consistency trust
region. Translation/rotation/gripper is the only currently justified grouping.
Start training-free. A learned gate is allowed only if the dense constrained
oracle leaves clear headroom and every simple rule fails.

If dense Gate-3A1 does not justify this fallback, do not replace it with a TAS-
like neural router. Pivot away from the group paper. A narrower empirical study
of when control-semantic temporal kernels help may remain possible only with
multi-policy and closed-loop evidence.

## 14. Exact next implementation plan — after joint approval only

1. Add a standalone, read-only dense-cache generator under
   `research/audit_tools/`; do not change policy execution code.
2. Pin the audited checkpoint, config, preprocessing, inference mode, chunk
   length, and disabled/enabled temporal aggregation state in a manifest.
3. Write one artifact per episode plus a completion index; use atomic writes,
   resume checks, and SHA256.
4. Generate validation episodes first. Verify counts, shapes, finite values,
   duplicated queries, and a hand-traced action-time/source-time example.
5. Freeze hyperparameter grids and the primary metric in a decision manifest.
6. Generate held-out test episodes without revisiting validation choices.
7. Run Level 1. Stop or continue according to the gate above.
8. Only if Level 1 passes, compute oracle Levels 2–4 from the same cache.
9. Commit only scripts, manifests, and compact summaries. Keep dense predictions
   local with path, size, SHA256, and claim dependency.
10. Request independent review before any policy wrapper or rollout experiment.

Experimental-design controls for a later rollout gate:

- block by task and initial-condition seed;
- pair methods on identical seeds;
- randomize method execution order within blocks where feasible;
- match policy-query opportunity and report actual calls;
- predeclare success, exclusions, resets, truncation, and missing-run handling;
- retain all negative tasks and failures;
- use task-level and seed-level intervals, not episode pooling alone.

## 15. Stop-doing list

- Do not rescue `Y_refresh` binary reliability without a new target-validity
  argument.
- Do not implement independent arm/gripper retain/refresh again.
- Do not call fixed environment fractions semantic phases.
- Do not optimize or report argmax horizons without full curves and uncertainty.
- Do not treat continuous gripper magnitude MSE as control quality.
- Do not infer rollout gain from teacher-forced error.
- Do not call semantic similarity a new temporal ensemble paradigm.
- Do not start a neural router, MoH policy modification, dense rollout sweep, or
  real-robot experiment before the dense evidence gate and independent review.
- Do not tune baseline and proposed-method hyperparameters with unequal budgets.
- Do not hide the negative group results if the scalar method succeeds.

## Provisional figure and claim architecture

The figure-planning and manuscript-optimizer skills were used only for claim
logic, not to draft a paper. A future Figure 1 should make one claim:

> Full-vector temporal similarity ignores heterogeneous control semantics; a
> shared control-semantic source distribution changes source trust without
> independently recomposing action groups.

Suggested panels: (a) overlapping ACT candidates for one physical time; (b)
raw cosine versus translation/SO(3)/gripper-sign distance; (c) the negative
unconstrained group-mixing counterexample; (d) query-matched closed-loop outcome.
Panel (d) does not exist and must not be mocked as data.

## Decision log

| Decision | Reason |
|---|---|
| Reject GATE as immediate method | Deployable group weighting is negative/null; only oracle headroom is positive. |
| Drop GATE acronym | The minimum method is scalar, and no exhaustive acronym collision search was performed. |
| Prefer shared source weights | Best sparse deployable result and lower mixed-mode risk. |
| Keep control-semantic distance | Held-out sparse lead over validation-tuned CogACT cosine with matched aggregation. |
| Do not promote semantic aggregation | SO(3)+sign aggregation did not improve held-out primary error. |
| Run dense Gate-3A1 next if approved | It is the cheapest decisive test and needs only 12,294 planned calls on existing balanced validation/test episodes. |
| Defer all policy work | Required by the collaboration protocol and unsupported by current evidence. |

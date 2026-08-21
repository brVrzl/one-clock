# Comments to authors

## Review status and evidence-bounded summary

This is an author-requested internal adversarial simulation, not an assigned
ICRA review and not an editorial decision. The assessors are project
collaborators. They reviewed the committed zero-trust fact sheet, the new
[Gate-3A0 output](audit_outputs/gate3a0_sparse_group_consistency.json), its
[read-only script](audit_tools/gate3a0_sparse_group_consistency.py), and the
[direction literature audit](joint_direction_literature_audit_2026-08-21.md).
No dense cache, new rollout, second-policy result, or real-robot result exists.

The original proposal allowed action groups to use different source weights and
interpolated back toward shared weights when a consistency score was high.
Gate-3A0 instead favors a simpler candidate: one shared full-action source
distribution computed with a control-semantic similarity kernel. Its current
evidence is sparse, teacher-forced, and offline.

## Strengths shared by the three reviewers

- The project preserved the strongest negative closed-loop result: the tested
  selective retain/refresh rule loses 0.20–0.29 absolute success relative to
  global replacement in its evaluated conditions.
- Gate-3A0 uses an episode-held-out split, paired episode bootstraps, SO(3)
  rotation error, and gripper sign/event metrics rather than relying on a
  generic group-balanced MSE.
- The proposed minimum method is compatible with a frozen ACT policy and has a
  low implementation burden.
- The related-work position is honest: ACT, CogACT, TAS, MoH, AAC, and recent
  adaptive-horizon work are treated as prior art rather than renamed.

## Reviewer A — general ICRA manipulation

### Major comment M1

- Location: Proposed practical direction and planned Gate-3A1.
- Observation: There is no new closed-loop evidence for the proposed scalar
  semantic ensemble, and the only direct closed-loop group-aware mechanism is
  strongly negative.
- Evidence or criterion: The verified fact sheet records the negative selective
  retain/refresh result. Gate-3A0 evaluates demonstrated actions at
  teacher-forced states and explicitly disclaims inference to rollout success.
- Why it matters: ICRA manipulation readers care about task completion and
  robustness. An offline action-error improvement can reflect demonstration
  noise or alternative valid actions and cannot carry the paper's practical
  claim.
- Requested action: First pass the dense held-out evidence funnel; then run a
  query-matched, paired-seed closed-loop comparison against newest, exact ACT
  temporal ensemble, released CogACT AAE, validation-tuned CogACT, and the best
  simple fixed/exponential baseline. Report every task and seed, policy queries,
  failures, and uncertainty.

What would make the incremental contribution publishable?

- A clearly isolated semantic-kernel delta that improves task success across
  several tasks, with matched policy calls and no selective task reporting.
- A second policy, benchmark, or real-robot transfer showing that the rule is
  not a quirk of one frozen LIBERO checkpoint.
- Failure-mode evidence connecting the gain to discrete gripper events or
  rotation/action-mode ambiguity rather than merely reporting a tuned average.

What would make it too trivial?

- Only offline MSE gains; only task 0; only comparison to newest; or a result
  obtainable by retuning ACT's exponential coefficient.
- Calling the method group-aware when the deployed method uses one scalar weight
  vector and no group-specific decision.

Baseline that isolates the contribution: keep the same candidate cache, source
window, query schedule, and interpolation; change only CogACT cosine versus the
control-semantic kernel. Add a second ablation that changes only the aggregation
operator. Gate-3A0 already suggests the operator ablation is unnecessary or
negative.

Minimum credible gain: no universal percentage is justified before measuring
paired rollout variance. The effect must have a confidence interval and
task/seed pattern incompatible with tuning noise, and its magnitude must matter
relative to the checkpoint's baseline failure rate and experimental cost.

Current submission readiness: **1/5**. The question and audit are credible; the
method-level closed-loop evidence is absent.

## Reviewer B — ACT/CogACT/TAS/action-chunking expert

### Major comment M2

- Location: Novelty and method definition.
- Observation: The scalar core is extremely close to CogACT AAE; TAS already
  learns full-action cached-candidate selection, and ACT already supplies the
  overlapping temporal predictions.
- Evidence or criterion: CogACT's released code uses one full-vector cosine
  similarity weight per cached prediction. TAS constructs the same source-time
  candidate set and learns a full-action selector with a coherence reward.
- Why it matters: A different distance function is an incremental contribution,
  not a new routing paradigm. The paper must show why raw cosine is wrong for
  robot control and why the proposed semantic distance fixes a measurable
  failure.
- Requested action: Use the exact released CogACT formula at `alpha=0.1`, a
  validation-tuned CogACT alpha, exact ACT temporal ensembling, uniform, and
  newest-only. Do not label normalized-MSE-to-newest weighting “CogACT-style.”
  Predeclare the control-semantic distance and keep its parameters fixed for the
  test rollouts.

What would make the incremental contribution publishable?

- A controlled demonstration that full-vector cosine confuses scale, rotation
  geometry, or gripper modes, followed by a shared-weight semantic kernel that
  corrects those cases and improves closed-loop success.
- Evidence that the gain survives exact dense ACT temporal candidates, not only
  irregular sparse sources, and that it is robust across source-window sizes.
- A compact property statement: invariance to gripper magnitude under
  sign-preserving control and valid SO(3) comparison, without claiming a new
  ensemble principle.

What would make it too trivial?

- A temperature sweep that wins only because CogACT's published alpha is used
  without its own validation-tuned counterpart.
- A method that is simply `exp(-MSE/T)` on a seven-dimensional vector with new
  terminology.
- Adding a learned router after the heuristic fails; that would move directly
  into TAS territory and increase baseline burden.

Baseline that isolates the contribution: validation-tuned CogACT cosine and the
semantic kernel must share the same candidate set, aggregation rule, window,
and tuning budget. Compare raw linear aggregation and semantic aggregation as a
factorial ablation. Gate-3A0 currently shows the semantic aggregation operator
does not improve the primary metric.

Minimum credible gain: the paired confidence interval should exclude no effect
for the primary predeclared metric, but that condition alone is insufficient.
The same direction should appear in closed-loop success on a meaningful share
of tasks without regressions concentrated at grasp/release events.

Current submission readiness: **1/5**. The new kernel has a sparse offline lead
over cosine, but the core paper claim is not established.

## Reviewer C — skeptical incremental-novelty reviewer

### Major comment M3

- Location: Provisional GATE proposal and contribution language.
- Observation: The proposed group method can currently be summarized as
  “CogACT plus actuator groups plus a heuristic fallback,” while the deployable
  group variants are worse than the scalar method and the consistency-gated
  variant is statistically tied with it.
- Evidence or criterion: In Gate-3A0, arm/gripper similarity is worse than scalar
  semantic similarity by 0.00460 in dimension-weighted semantic error with a
  paired episode-bootstrap interval of [0.00267, 0.00724]. The semantic-three
  group similarity is worse by 0.00771 with interval [0.00019, 0.01302]. The
  validation-selected consistency-gated semantic-three residual differs from
  scalar by +0.00058 with interval [-0.00060, 0.00207].
- Why it matters: Oracle freedom is not a deployable mechanism. Making group
  routing the headline despite these results would look post-hoc and would
  repeat the project's earlier claim drift.
- Requested action: Remove independent group weighting from the minimum method.
  Treat the dense group oracle as a conditional diagnostic. Reintroduce a group
  residual only if its advantage survives control-aligned metrics and a
  predeclared consistency constraint and a simple deployable rule beats the
  shared-weight baseline.

What would make the incremental contribution publishable?

- For the simplified scalar paper: strong, repeatable success gains and an
  ablation showing the control-semantic kernel is the necessary delta from
  CogACT.
- For a later group paper: a direct, query-matched composition experiment that
  shows unconstrained group mixing fails, a constraint predicts that failure,
  and the constrained method recovers a nontrivial portion of verified dense
  oracle headroom.

What would make it too trivial?

- A single-checkpoint success bump with several tuned distances and no held-out
  selection protocol.
- Rebranding action-vector normalization as heterogeneous temporal reasoning.
- Using the name GATE without a collision search or when no group-adaptive gate
  remains in the deployed method.

Baseline that isolates the contribution: a factorial table with source weights
(`ACT age`, `CogACT cosine`, `semantic distance`) crossed with aggregation
(`raw`, `SO(3)+sign`) under identical queries. If group freedom returns, add
`unconstrained`, `age-trust-region`, and `nearest-full-source trust-region`
rows.

Minimum credible gain: do not set a fixed 5% or 10% threshold without variance.
Credibility requires a predeclared primary comparison, paired confidence
intervals, task-level consistency, and enough absolute successes to change the
practical conclusion rather than a rounded average.

Current submission readiness: **1/5**. The simplified direction is an honest
candidate; the GATE paper is not supported.

## Minor comments

### Minor comment m1

- Location: Working acronym and terminology.
- Observation: “GATE” is not yet justified and may collide with unrelated
  robotics/ML methods; the simplified method is not group-adaptive.
- Evidence or criterion: The literature search was method-focused, not a global
  acronym/trademark search, and Gate-3A0 favors one shared source distribution.
- Why it matters: A misleading acronym obscures the actual incremental delta.
- Requested action: Drop GATE for now. Use a descriptive working name such as
  “control-semantic scalar temporal ensemble” until a pre-submission collision
  search and a stable method exist.

## Methods, statistics, and reproducibility

The dense gate must preserve episode-level train/validation/test separation,
task balance, exact checkpoint/config/code hashes, full chunks, source and
target timestamps, and resume-safe per-episode artifacts. Temporal age must be
reported in 10 Hz dataset steps and seconds; later 20 Hz rollouts must use their
own physical timing. Paired episode or paired-seed intervals and per-task
results are required. Hyperparameters are selected on validation episodes only.
Offline errors, gripper events, boundary discontinuity, policy calls, wall time,
and rollout success must remain separate outcomes.

## Ethics, transparency, figures, tables, and citations

No misconduct concern is raised. Negative results must remain visible. Figure 1
should make one claim: standard full-vector similarity ignores heterogeneous
control semantics, while the proposed shared-weight kernel changes source trust
without recomposing groups. It must label all unexecuted/dense-oracle panels as
offline. ACT, CogACT, TAS, MoH, AAC, and the 2026 adaptive-horizon methods must be
cited directly. ICRA 2027's current author guidance requires disclosure of
AI-generated paper content; the final authors remain accountable for every
claim and citation.

## Limitations of this review

The review is internal and conflicted by collaboration. It assesses a direction
specification, not a finished manuscript. It does not verify future dense data,
rollouts, a second policy, hardware transfer, or causal mechanism. The 2026
literature search is current only through 2026-08-21.

# Confidential comments to editor

## Reviewer disclosures

- Conflicts and editor clearance: the assessors are author-side collaborators;
  this document must not be represented as an independent external review.
- Competence limits or specialist review needed: future real-robot control and
  VLA-generalization claims need specialist human review.
- Assistance or tools used and required disclosure: local analysis tools and an
  AI research collaborator were used under author instruction; any generated
  manuscript content must follow the current ICRA/IEEE-RAS disclosure policy.
- Confidentiality or retention issue: unpublished project evidence remained
  local. Web searches used generic literature queries and public citations, not
  manuscript text or private raw artifacts.

## Editorial-process or integrity concerns

There is no substantiated integrity allegation. The material is not
submission-ready because the principal method lacks dense and closed-loop
evidence. The appropriate author-side action is to preserve the negative group
results and enforce the stated kill conditions before drafting performance
claims.

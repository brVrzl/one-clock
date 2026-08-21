# Peer-review Working Draft: Three ICRA 2027 Red-Team Reviews

This is an author-requested pre-submission simulation, not an official ICRA
review or editorial decision. The project collaborator producing it is not an
independent reviewer. The human research team must verify all claims and use its
own judgment. [ICRA 2027 uses double-anonymous review and its official reviewer
policy prohibits reviewers from processing manuscripts through AI](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/information-for-icra-reviewers/);
therefore this artifact must not be used as AI assistance on an assigned
manuscript after submission.

The scores use the official ICRA 2027 scale: A/5.0, B+/4.5, B/4.0, B-/3.5,
C/3.0, C-/2.5, D/2.0, or U/1.0.

# Comments to authors

## Reviewer A — General ICRA manipulation reviewer

### Evidence-bounded summary

The proposed paper would study whether a frozen ACT policy can improve control
by selecting among temporally overlapping action predictions, with optional
arm/gripper-specific selection constrained to avoid inconsistent joint actions.
The current evidence is a thorough retrospective audit, one strongly negative
matched-query group-retention experiment, and a positive sparse offline
temporal-ensemble/oracle analysis. The proposed method and decisive closed-loop
evaluation have not been run.

### What is actually new?

Potentially new: explicitly allowing actuator groups to deviate from a common
temporal source only when a joint-consistency criterion permits it, coupled to a
control-aligned arm/gripper loss. Dynamic horizon, temporal ensembling, temporal
selection, and a generic learned router are not new. The contribution remains a
proposal until the consistency mechanism is formalized and evaluated.

### Is the empirical evidence credible?

The audit itself is unusually credible: it distinguishes raw traces from later
summaries, recomputes accounting and statistics, preserves negative results, and
identifies missing provenance. The evidence for the candidate method is not yet
credible enough for a paper. The positive result is teacher-forced, sparse in
source time, from one checkpoint and one benchmark suite. The only direct
matched-query group-selection rollout result is strongly negative.

### Could the improvement be a metric artifact?

Yes. The arm/gripper oracle advantage changes from .0658 under group-balanced
loss to .0301 under normalized dimension weighting. Continuous gripper
magnitude is behaviorally irrelevant in this LIBERO controller, demonstration
actions may be multimodal, and the 10 Hz demonstration / 20 Hz rollout mismatch
changes temporal meaning. These are central validity issues, not cosmetic
ablations.

### Could a simpler baseline explain it?

Yes: ACT exponential ensembling, uniform averaging, CogACT-style similarity
weighting, a scalar learned selector, or a smoothness/disagreement heuristic.
The current best non-oracle is already a simple similarity ensemble. The paper
must show value beyond it before adding group machinery.

### Does offline action improvement matter?

Unknown. No result links offline temporal-selection gain to rollout success.
Given action multimodality and closed-loop distribution shift, MSE improvement
could be irrelevant or harmful.

### What baseline is missing?

The most important missing baselines are dense ACT temporal ensemble,
CogACT-style adaptive action ensemble, TAS-style full-action selection,
scalar-only selection with identical capacity, and a control-aligned
smoothness-only baseline. If a re-query claim is added, AutoHorizon, AAC, A3,
PACE, DVAC, DEHP, and BCP become relevant.

### What experiment would change the score?

A preregistered paired, query-matched ten-task rollout on untouched states in
which the constrained method beats the strongest scalar/similarity baseline
with a task-clustered interval excluding zero, while the unconstrained group
variant is worse and no extra policy queries are used. Before that, the dense
offline gate must reproduce robust group advantage under control-aligned loss.

### Major comment A1

- Location: Candidate specification, verified observation and method sections.
- Observation: The manuscript-shaped claim begins at group consistency, but the
  current positive evidence establishes only sparse offline temporal diversity.
- Evidence or criterion: The matched-query group-retention experiment loses
  .20–.29 absolute success, while no CCTS rollouts exist.
- Why it matters: The central method premise and practical effect are untested.
- Requested action: Complete Gate-3A and the bounded paired rollout, or narrow
  the work to an empirical analysis without a policy-improvement claim.

### Current score if submitted today

**C- / 2.5 (Reject), confidence 4/5.** The audit is strong, but an ICRA method
paper cannot rest on a proposed mechanism and offline oracle headroom.

## Reviewer B — ACT/action-chunking/temporal-ensemble expert

### Evidence-bounded summary

This work reframes ACT's overlapping chunk outputs as temporal experts and
proposes a control-aligned selector with constrained group deviations. It audits
fixed-prefix ACT execution on LIBERO and reports sparse teacher-forced gains from
similarity weighting and temporal oracles.

### What is actually new?

The temporal-expert interpretation is not new: ACT temporal ensembling already
combines predictions of the same action from several source observations;
CogACT adapts those weights by similarity; TAS explicitly caches and selects
temporal actions; recent mechanistic work describes action chunking as implicit
ensembling across delayed temporal relationships. Only the group-specific
consistency constraint appears potentially distinct.

### Is the empirical evidence credible?

The system-contract and artifact audit is credible. However, the sparse cache
does not reproduce standard dense ACT temporal ensembling: candidates come from
roughly 25-step-spaced source queries and only 2–6 experts are available. Calling
this evidence for dense temporal routing would overstate the experiment. The
checkpoint is deterministic and temporal aggregation was disabled in the
rollouts, so comparisons to original ACT must be reconstructed explicitly.

### Could the improvement be a metric artifact?

Yes. Source age and chunk output index are confounded; the demonstrated target
is only one possible action; gripper sign semantics are mismatched by magnitude
loss; and the sparse candidate cadence changes candidate diversity. Oracle
mixtures can directly fit demonstration noise.

### Could a simpler baseline explain it?

Very likely. Similarity weighting already gives the best non-oracle error (.7084)
and differs only slightly from exponential weighting (.7112). A properly tuned
ACT exponential ensemble, including its exact age convention and physical-time
rate, may be sufficient. Hard selection may also merely avoid cross-mode
averaging, which CogACT was designed to address.

### Does offline action improvement matter?

Not by itself. ACT was motivated partly by temporal smoothness and multimodal
demonstrations, so lower pointwise error can select an undesirable average or a
noncausal teacher-forced expert. The paper needs paired control results and
boundary smoothness, not just action MSE.

### What baseline is missing?

Exact ACT temporal ensemble with coefficient sensitivity; CogACT AAE; uniform
ensemble; newest-only; best validation-selected fixed age; TAS; source-age-only
and chunk-index-only selectors; and an ensemble that is forced to use a single
joint source. Report age curves and full distributions, not only oracle minima.

### What experiment would change the score?

Dense per-step ACT predictions on held-out episodes, with physical-time-aligned
age curves and exact ACT/CogACT/TAS comparisons, followed by a paired rollout
showing that constrained group routing improves success rather than only MSE.
An ablation must demonstrate that the consistency constraint—not additional
capacity or smoothing—causes the gain.

### Major comment B1

- Location: Candidate specification, novelty and required baselines sections.
- Observation: The distinction from TAS and CogACT currently depends entirely
  on a group-consistency mechanism for which no oracle-constrained result exists.
- Evidence or criterion: The sparse independent group oracle is not a realizable
  policy and its normalized-dimension advantage is .0301.
- Why it matters: Without evidence that constrained group freedom retains this
  benefit, the method reduces to known temporal selection.
- Requested action: Compute scalar, independent-group, and
  consistency-constrained group oracle frontiers on a dense test cache before
  defining or training the router.

### Current score if submitted today

**D / 2.0 (Definitely Reject), confidence 5/5.** The closest known methods occupy
the core idea, and the proposed distinguishing mechanism has no result yet.

## Reviewer C — Skeptical 2025–2026 adaptive-horizon/MoH expert

### Evidence-bounded summary

The work proposes a frozen-policy temporal selector with optional group-specific
weights, motivated by audited ACT/LIBERO data. It seeks a niche between adaptive
execution-horizon methods and explicit mixtures of prediction horizons.

### What is actually new?

Dynamic execution is not new: SGAC, AutoHorizon, AAC, A3, PACE, DVAC, DEHP, BCP,
and MoH cover training-free heuristics, internal uncertainty, verification,
learned heads, online RL, and explicit horizon experts. Temporal source routing
is also directly covered by TAS. The only plausible novelty is physically
consistent group-specific source selection. That is a narrow difference, not
yet a demonstrated advantage.

### Is the empirical evidence credible?

The zero-trust audit is credible as a negative result and artifact correction.
The paper direction is not. Gate-2B maps were selected and evaluated on the same
230 conditions, phases were fixed time-limit thirds, most candidates were tied,
and the combined maps did not robustly beat static controls. Those experiments
cannot motivate a learned phase/group method. The sparse offline oracle is a new
lead, not validation.

### Could the improvement be a metric artifact?

Yes, and the audit itself shows several mechanisms: threshold sensitivity,
cumulative-prefix information loss, task reweighting, dimension weighting,
gripper sign semantics, phase segmentation, in-sample map selection, and
teacher-forced oracle selection. The current evidence does not reject these
simple explanations.

### Could a simpler baseline explain it?

Yes. PACE/smoothness, AAC entropy, DVAC variance, A3 consensus, CogACT similarity,
or a scalar continuation head can each explain why one should avoid an old or
inconsistent action. A consistency penalty plus scalar selection may match all
benefit without any independent group clock.

### Does offline action improvement matter?

No policy claim follows until it predicts closed-loop success. The strongest
historical warning is that the tested matched-query group rule substantially
reduced success even though the project was motivated by offline support.

### What baseline is missing?

In addition to ACT/CogACT/TAS, the missing novelty baselines are MoH for explicit
horizon mixtures and at least one strong adaptive-prefix method appropriate to
the chosen backbone. If the method claims value of re-querying, BCP or DEHP is
mandatory. If it claims training-free uncertainty, AAC/A3/DVAC is mandatory.

### What experiment would change the score?

First, show a dense, task-held-out, control-aligned, consistency-constrained
oracle advantage over scalar routing. Second, show a paired closed-loop gain
over CogACT/TAS and a strong adaptive-prefix baseline at equal queries and
latency. Third, reproduce on a modern VLA or remove the VLA/general claim.

### Major comment C1

- Location: Candidate specification, novelty and generalization sections.
- Observation: The proposed title-level idea can be read as another adaptive
  horizon/router paper in a saturated literature.
- Evidence or criterion: Multiple peer-reviewed 2025–2026 methods already adapt
  execution; TAS selects cached temporal candidates; MoH learns horizon experts.
- Why it matters: A reviewer cannot infer novelty from different terminology.
- Requested action: Center the paper on a formal and experimentally isolated
  group-consistency problem, or abandon the method claim. Include an explicit
  claim-by-claim comparison with TAS, CogACT, and MoH.

### Current score if submitted today

**D / 2.0 (Definitely Reject), confidence 5/5.** The current submission would be
premature and its central novelty would be vulnerable to direct prior-art
rejection.

## Cross-review consensus

- The audit is a strength; the current method evidence is insufficient.
- Dynamic horizon and generic temporal routing are occupied prior art.
- The only plausible novelty is consistency-constrained group choice.
- Dense oracle analysis is required before training.
- A query-matched closed-loop comparison against ACT, CogACT, and TAS is the
  decisive evidence.
- No reviewer would support submission in the current state.

# Confidential comments to editor

This is not an editor-facing review. No integrity allegation or editorial-process
concern is raised. Process disclosure: the text is an AI-assisted, author-requested
internal simulation produced by a project collaborator; it is not independent,
and specialist human review in real-robot control and current VLA systems is
needed. No unpublished project text was sent to an external service during the
peer-review step. Public literature and public ICRA policy were searched
separately. The project files and this derivative are retained in the
author-controlled workspace at the author's request.


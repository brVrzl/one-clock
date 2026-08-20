# Dynamic group-wise execution-horizon method design memo

**Date:** 2026-08-20
**Status:** literature and method-design study only. No scheduler was implemented, no
experiment or rollout was run, and no paper file was changed.

## Executive summary

The project has established that a frozen action-chunk policy can be deployed
with a **group-specific execution horizon**: a group whose commitment expires
accepts its slice from a fresh full policy chunk, while a non-expired group
continues its previously accepted slice. This is more specific than ordinary
adaptive action chunking, where one scalar prefix length is selected and every
action component is replaced together.

The static results warrant this research question, but not the conclusion that
an online scheduler will help. On LIBERO Object, the universal static pair
`(arm=4, gripper=16)` is strong (macro success 0.734 on the common ten-task
set) and leave-one-task-out selection chose it on every split. The retrospective
per-task static group-wise oracle is 0.779, leaving a 0.045 macro gap. Thus the
next study must first distinguish **task-level selection of a fixed pair** from
**within-episode variation in the safe retention time of each group**. The
current results show assignment sensitivity and task variation; they do not yet
show phase-dependent group horizons.

The recommended research direction is not “PACE independently per group.” It
is a **self-supervised group-wise chunk-persistence estimator**:

1. Keep the chunk policy frozen.
2. On demonstrations, use future observations to measure how much an action
   predicted now for group `g` disagrees with the frozen policy's fresh
   prediction at a future time (and, separately, optionally with the
   demonstrated action).
3. Train a small auxiliary estimator of a *group- and age-conditioned retention
   risk/survival curve*, rather than a categorical horizon head trained on task
   return.
4. At deployment, select each group's commitment from that calibrated curve;
   only the execution acceptance schedule changes.

This has a clearer scientific premise than a hand-tuned kinematic rule: the
quantity being estimated is the future **persistence of a group prediction under
new observations**. It also creates decisive ablations: scalar aggregation of
the same signal, static group-wise horizons, a phase-shuffled schedule with the
same horizon histogram/query budget, and a learned scalar-horizon head.

This is a recommendation, not a claim of novelty or expected performance. The
method has two material risks: the self-supervised disagreement target may not
predict task success, and independently retained components may be
cross-group-incompatible. Both must be tested before interpreting a gain.

## Scope and evidence used

This memo is a targeted systematic review of primary papers and official project
materials most directly relevant to execution-time chunk selection, supplemented
by the repository's audited experiment records. The search included the current
paper's cited adaptive-execution methods and title/identifier searches for
action-chunking, uncertainty, adaptive horizons, and event/self-triggered MPC.
Only primary papers or official author/project material are used for method
descriptions. It is not a bibliometric claim that every related paper has been
found.

Repository evidence consulted:

- [50-state LIBERO task-0 static landscape](libero_static_grid_50.md): task-0
  global best `h=8` is 45/50; `(arm=4, gripper=16)` is 47/50 and has the same
  approximate query rate as global `h=4`.
- [Cross-task static summary](libero_object_cross_task_summary.md) and
  [dynamic-readiness analysis](libero_object_dynamic_readiness.md): task-0 has
  50 states, tasks 1–9 have 20; macro analyses weight tasks equally.
- [Execution audit](libero_execution_audit.md) and the ICRA draft's problem and
  executor descriptions: the verified LIBERO action partition is six relative
  end-effector controls versus one gripper control; policy queries are shared,
  but group buffers and source generations can differ.
- [PACE-to-LIBERO source audit](pace_libero_source_audit.md): a source-faithful
  PACE implementation is currently blocked, rather than merely deferred.

The existing experiment facts should be read narrowly. In particular, the
`(4,16)` result does **not** identify a causal “gripper stability” mechanism and
does **not** establish that `h_arm(t)` and `h_gripper(t)` should change within an
episode.

## Terminology landscape

### Recommended wording

Use **action chunk** and **prediction horizon** for what the policy predicts,
and **execution horizon** for how many predicted actions are followed before a
new observation-conditioned policy query. This is the cleanest vocabulary in
Diffusion Policy and recent adaptive-execution work. For the proposed direction,
use **group-specific adaptive execution horizons** or **group-wise adaptive
execution horizons**, explicitly defining the term at first use.

Use **group refresh** or **group acceptance** for replacing that group's active
chunk slice. This matters: a group-specific horizon is not an independent policy
query interval. One full policy query may be caused by any expiring group; only
the groups that accept it change source chunk.

| Term | Status in the literature | Recommendation here |
|---|---|---|
| Action chunk / action sequence | Established in ACT, Diffusion Policy, and flow-VLA work | Use. |
| Prediction horizon | Established; length predicted by the policy | Use, and keep separate from execution horizon. |
| Action execution horizon / execution horizon | Established in Diffusion Policy and the adaptive-execution papers | Preferred technical term. |
| Receding-horizon execution / replanning | Established control interpretation | Use for a fresh observation and policy query. |
| Open-loop prefix | Established descriptive term | Use for the portion executed without a new observation. |
| Adaptive action chunking / adaptive execution horizon | Established but broad | Use when discussing scalar prior methods; qualify the proposed direction as group-specific. |
| Event-triggered replanning | Established control-language analogy | Use only if the method is actually a trigger rule, rather than a predicted horizon. |
| Temporal abstraction | Established in hierarchical RL/options | Avoid as the primary name: it normally concerns temporally extended skills/options, not retention of components of one low-level chunk. |
| Commitment horizon | Intuitive but not the dominant primary-paper term | Fine as explanatory prose after defining execution horizon; do not make it the method's formal name. |
| Commitment-duration allocation | Not found as a standard term in the reviewed chunk-execution or control sources | Avoid. It suggests resource allocation rather than replanning/retention. |
| Multi-clock scheduling / temporal urgency | Not established in the reviewed sources | Avoid. |

No acronym is recommended. The transparent phrase “group-specific adaptive
execution horizons” is more defensible than a branded name while the mechanism
is still unsettled.

### Control-theory connection and its limit

There is a useful analogy, but not an equivalence.

| Control concept | What it adapts | Connection | Important difference |
|---|---|---|---|
| Receding-horizon/MPC | Repeatedly solves a future-control optimization; often applies a short prefix | Supplies the replan/execute-prefix framing | A learned chunk is not an online optimal control sequence with terminal feasibility guarantees. |
| Adaptive-horizon MPC | Usually changes the prediction/optimization horizon, often for stability/computation | Shows that horizon choice can be state-dependent | It is generally not about which physical coordinates retain different source plans. [Krener](https://arxiv.org/abs/1602.08619) is an example. |
| Event-triggered control | Updates control when a trigger condition is met | Gives the vocabulary “trigger,” “update,” and “inter-event time” | Classical triggers are commonly one controller update; they do not justify mixing action coordinates from distinct learned chunk generations. |
| Self-triggered MPC | Chooses a future update/sampling time from current information | Closest conceptual analogy to predicting a commitment duration | The object is usually a whole control input/plant, with model-based guarantees not available here. |
| Distributed/asynchronous event-triggered control | Different agents/subsystems can update asynchronously | Suggests that asynchronous updates are not conceptually alien | The theory assumes specified subsystem coupling/communication; it cannot be imported as a safety argument for arm/gripper chunks. |

The relevant established control terms are **inter-event time**, **sampling
interval**, **update interval**, and **control/prediction horizon**. “Commitment
duration allocation” is not a standard replacement. The proposed direction
should be framed as an execution-time policy-deployment problem informed by
event-triggered control, not as MPC with inherited guarantees.

## Related work and collision map

### Action chunks and VLA execution

| Work | How chunk/execution length is handled | Decision signal and output | Why it matters here |
|---|---|---|---|
| [ACT](https://arxiv.org/abs/2304.13705) | Predicts action sequences. Its reported temporal-ensemble deployment queries repeatedly and aggregates overlapping full actions. | No group-specific execution decision. | Establishes chunked behavioral cloning, but temporal ensembling must remain disabled in a controlled executor study. |
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) | Distinguishes observation, prediction, and action-execution horizons; the execution horizon is a fixed deployment hyperparameter. | One full-vector prefix `T_a`. | Best conceptual source for the prediction-vs-execution distinction and receding-horizon terminology. |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | The original paper predicts one 7-D action as discretized tokens, not an action chunk. | No execution-horizon rule in the original model. | Important VLA context, but not a direct adaptive-chunking baseline. Do not imply that original OpenVLA already has the same interface. |
| [$\pi_0$](https://arxiv.org/abs/2410.24164) | Predicts `H=50` actions; reported deployment runs inference after 16 actions at 20 Hz or 25 at 50 Hz, without temporal ensembling. | Fixed, embodiment-dependent full-vector prefix. | Demonstrates that training horizon and deployed prefix can differ. Its interface is compatible with execution-only research, but its reported setting is not adaptive. |
| [RTC](https://arxiv.org/abs/2506.07339) | Generates a next flow/diffusion chunk while the current chunk executes; freezes unavoidable actions and inpaints the remainder. | Full-vector timing/inpainting rule driven by inference latency. | A close boundary-consistency/latency paper, but not independent physical-group retention. |
| [HiPolicy](https://arxiv.org/abs/2604.06067) | Learns and fuses multi-frequency action chunks, then selects an execution frequency with entropy. | Whole hierarchical policy/output, not group-wise source-buffer acceptance. | Important collision with “multi-frequency,” but it changes policy training and temporal representation. It is not an execution-only frozen-policy method. |
| [Temporal Action Selection](https://arxiv.org/abs/2511.04421) | Caches full chunks from multiple times and selects actions with a learned selector. | Full-action selection. | Shows that retaining/replacing chunks is already active research; the action-vector granularity still differs. |

**Takeaway.** Chunk length is commonly fixed per policy, embodiment, or task;
recent work increasingly makes it adaptive. The standard final decision is a
single full-vector prefix, even when methods inspect different action components
or action chunks at multiple temporal resolutions.

### Adaptive-horizon methods

| Method | Signal | Where decision lives | Output | Training | Weakness relative to this problem |
|---|---|---|---|---|---|
| [PACE](https://arxiv.org/abs/2606.00537) | Predicted arm kinematic speed profile; low-speed valleys and task-calibrated acceptance | External test-time executor | Scalar `h_i`; accepted candidates from arms are pooled and the earliest is chosen | Training-free | Uses arm-specific signals but synchronizes all action dimensions at one boundary. It describes phase segmentation, not independent retention validity. The current LIBERO relative-OSC representation cannot support a source-faithful implementation without missing PACE details; see the repository audit. |
| [AutoHorizon](https://arxiv.org/abs/2602.21445) | Cross-/self-attention statistics in a flow VLA | Test-time policy-side analysis | Scalar execution horizon | Training-free, but architecture-access dependent | Attention is architecture-specific and its final decision is one prefix for the chunk. Not available from black-box frozen ACT without a new internal audit. |
| [AAC](https://arxiv.org/abs/2604.04161) | Sampled action entropy: translation, rotation, and gripper entropies are estimated separately then summed/averaged across components and time | External inference-time rule | Scalar `h*` | No retraining; needs multiple action samples | The closest uncertainty/gripper collision. Its component signals are aggregated before selecting one synchronized prefix. |
| [DVAC](https://arxiv.org/abs/2606.03847) | Variance of final denoising clean-action estimates | External inference-time rule | Scalar prefix ending before first high-variance index | Training-free for flow policies with denoising trace | Its variance score explicitly sums over all action dimensions. Current ACT does not expose an analogous denoising trajectory. |
| [DEHP](https://arxiv.org/abs/2606.11408) | Current state plus flattened predicted chunk | Lightweight policy head trained with PPO | One categorical scalar horizon | Online RL while base chunk policy is frozen | Closest “frozen policy plus horizon head” collision. Extending it naively to two logits/vectors would invite the “just added a horizon head” critique. |
| [ACH](https://arxiv.org/abs/2605.10044) / [AQC](https://arxiv.org/abs/2605.05544) | Multi-horizon Q/advantage estimates | RL policy/critic | Scalar chunk length | Offline-to-online RL | Value-based state-dependent scalar duration, not a black-box execution-only group refresh rule. |

### Uncertainty and confidence: what does and does not transfer

AAC makes the key point that a manipulation action has heterogeneous
translation, rotation, and gripper uncertainty. Its published selection,
however, turns the three entropies into one average entropy curve and one
`h*`. DVAC similarly produces a per-future-index variance only after summing
variance over all action dimensions. These works support **component-aware
measurement**, not yet component-specific execution schedules.

Group-wise uncertainty can in principle produce `h_arm` and `h_gripper`, but
only after four design choices are justified:

1. **Distribution source.** Multiple generative samples, an ensemble, a
   calibrated latent posterior, or dropout are different uncertainty objects.
   Current frozen ACT inference must be audited before assuming that stochastic
   samples are meaningful or cheap.
2. **Within-group metric.** Translation and rotation have different physical
   units; a six-dimensional relative OSC arm group cannot be assigned a norm by
   convenience. Gripper uncertainty can be discrete/continuous depending on the
   controller representation. Calibration must be per group rather than based
   on incomparable raw magnitudes.
3. **From uncertainty to retention.** Dispersion is not automatically the
   probability that retaining a stale prediction harms execution. It needs a
   target or calibration demonstrating that relation.
4. **Coupling.** Low uncertainty for each marginal group does not imply that a
   mixed-generation joint action is compatible.

Therefore “uncertainty-aware group-wise commitment” is a viable family, but it
is not equivalent to a scientifically specified method merely by computing two
variances.

## Collision analysis and design requirements

The requested collision analysis is included here rather than in a separate
file, because this task's strict deliverable allows only this memo.

| Likely reviewer objection | When it would be fair | Design requirement for a defensible distinction |
|---|---|---|
| “You just split PACE.” | If the method separately finds arm/gripper kinematic valleys and assigns their indices as horizons. | Do not make group-wise kinematic valley splitting the main method. A stronger method should estimate group **prediction persistence/staleness**, use a different target from kinematic phase boundaries, and compare to a scalar aggregation of the same signal. PACE remains a scalar global baseline when it is reproducible faithfully. |
| “You just added a horizon head.” | If an auxiliary network maps observation/chunk to two categorical lengths trained from task reward, especially with a vectorized DEHP-like loss. | Supervise an explicitly defined group-age persistence quantity from demonstrations/frozen-policy predictions, not a reward-selected length. Output a calibrated survival/risk curve, then derive horizons. Include a same-capacity scalar head and a task-conditioned static selector. |
| “This is uncertainty-based replanning.” | If the contribution is only per-group sampled variance followed by thresholds. | Treat uncertainty as one input feature/ablation, not the defining target. The decision must be about whether a *specific group slice from a specific source chunk remains valid* as it ages. Include a scalar uncertainty aggregation baseline and report sampling/inference cost. |
| “Mixed generations are incoherent, so this is unsafe.” | Always a legitimate concern; a joint chunk was predicted under one observation. | Make cross-group compatibility an explicit failure mode and measured quantity. Compare against synchronized replacement, log source ages/generations, and test whether gains survive cases with high inter-group disagreement. Do not claim independent clocks are universally safe. |
| “The method just learns the task identity.” | If one task or task label predicts a fixed pair and phase variation is not shown. | Evaluate against per-task static selection, use within-task time-shuffled controls with the same horizon/query distribution, and report within-episode horizon variation. A dynamic claim requires a benefit beyond a task-conditioned static pair. |
| “It spends more queries.” | If group expiry makes policy calls more frequent than its comparator. | Treat policy-query rate as a primary budget. Match or trace the query budget; compare to global and group-wise static points at the same effective cadence whenever possible. |

The narrow potential contribution is consequently not “adaptive chunking” in
general. It is: **a validity-based, group-specific acceptance schedule for slices
of a frozen joint action chunk, evaluated against scalar versions of the same
information and against static group-wise schedules.** This is only a potential
contribution until the target, calibration, coupling test, and experimental
controls are validated.

## Candidate method families

The candidates below are alternatives, not a proposed combined system. Ratings
are relative to the current frozen-ACT project and are deliberately conservative.

### 1. Group-wise kinematic phase boundaries (PACE-style)

| Aspect | Assessment |
|---|---|
| Core idea | Construct a phase/speed profile per physical group and let each group's first accepted valley determine its next commitment. |
| Signal source | Predicted within-chunk kinematics. |
| Output | `h_t^g`. |
| Training required | None, if the profile and calibration are supplied. |
| Implementation complexity | Medium in joint space; high and scientifically blocked for current relative translation-plus-axis-angle actions. |
| Reviewer perception | Easy to understand, but likely “PACE split.” |
| Advantages | Training-free; direct phase interpretation; potentially portable where joint trajectories are available. |
| Disadvantages / risk | PACE's scalar pooling is deliberately designed to synchronize arms; blindly reversing that design does not prove a group-validity rationale. Relative OSC units make group motion profiles hard to compare. Current PACE source details are insufficient for faithful implementation on this path. |

**Verdict:** useful ablation or diagnostic after a faithful scalar PACE baseline
is available; weak primary ICRA direction.

### 2. Calibrated cross-query prediction consistency

| Aspect | Assessment |
|---|---|
| Core idea | At a policy query caused by any group, compare each retained group's active old action at its current cursor with that group's action from the fresh current-observation chunk. Refresh a non-expired group only when calibrated disagreement says its old slice is stale. |
| Signal source | Free overlap between an old buffer and the already-produced fresh chunk; no extra policy query. |
| Output | Per-group retain/refresh decision at an existing query, rather than a standalone horizon prediction. |
| Training required | No learned model is necessary, but the discrepancy scale/acceptance level must be calibrated from demonstrations or a held-out calibration set. |
| Implementation complexity | Low to medium. |
| Reviewer perception | A principled event-triggered correction if calibrated; a heuristic if the threshold is hand-picked. |
| Advantages | Directly measures the thing stale execution suppresses: disagreement with an up-to-date policy prediction. Fits the existing shared-query semantics. |
| Disadvantages / risk | It cannot decide to query before any group reaches a scheduled cap; therefore it is a correction/acceptance mechanism, not a complete dynamic scheduler by itself. Fresh-policy disagreement need not correlate with task error. |

**Verdict:** strongest simple *training-free diagnostic*, provided its
calibration is specified before evaluation. It should not be positioned as the
main method unless it proves phase-sensitive benefit beyond a static cap.

### 3. Group-wise posterior-dispersion commitment

| Aspect | Assessment |
|---|---|
| Core idea | Draw multiple stochastic chunks at the current observation; estimate a group- and future-index-specific distributional dispersion, then select the longest acceptable commitment for each group. |
| Signal source | Posterior samples, ensemble predictions, or another explicitly verified uncertainty mechanism. |
| Output | `h_t^g` or group-specific survival probabilities. |
| Training required | None if meaningful samples exist; otherwise ensemble/dropout training changes the scope. |
| Implementation complexity | Medium to high because sampling cost and per-group calibration are material. |
| Reviewer perception | Natural connection to AAC, but potentially strong if the contribution is asynchronous group acceptance rather than two entropy thresholds. |
| Advantages | Directly operationalizes “higher predictive certainty permits longer commitment”; naturally generalizes to hands, wrists, and multiple arms. |
| Disadvantages / risk | Multiple samples may be expensive or unavailable for the frozen ACT runtime; posterior spread can represent multimodality rather than error; normalizing heterogeneous groups is nontrivial. |

**Verdict:** a promising second path, especially for diffusion/flow VLAs. It is
not the first recommendation for this ACT study until the policy's sampling
semantics are verified.

### 4. Learned vector-valued horizon predictor

| Aspect | Assessment |
|---|---|
| Core idea | Attach a small predictor to observation and predicted chunk, output one categorical horizon per group, and train with return/RL or labels. |
| Signal source | Policy observation, chunk features, and optionally history. |
| Output | `h_t^g`. |
| Training required | Yes; usually online RL or task-reward supervision. |
| Implementation complexity | Medium in simulation, high for real robots and cross-task generalization. |
| Reviewer perception | Straightforward but directly collides with DEHP; a vector output alone is not a sufficient conceptual advance. |
| Advantages | Can optimize downstream success and model nonlinear group interactions. |
| Disadvantages / risk | Sparse reward/sample efficiency; likely task overfitting; weak frozen-policy story; difficult to disentangle benefit of vector scheduling from ordinary learned horizon selection. |

**Verdict:** reserve for a later stage, and only after a self-supervised target
or dynamic effect has been demonstrated.

### 5. Self-supervised group-wise chunk-persistence estimation

| Aspect | Assessment |
|---|---|
| Core idea | From demonstration trajectories and a frozen chunk policy, estimate for every group and age how well an action predicted at time `t` remains consistent with a fresh policy prediction or demonstration action at `t+k`. Learn a calibrated persistence/survival curve, then derive each group's commitment. |
| Signal source | Current observation/chunk and self-supervised future-observation overlap; optional demonstrated-action deviation as a separate target. |
| Output | A group- and age-conditioned retention risk/survival curve, from which `h_t^g` is selected. |
| Training required | An auxiliary supervised estimator; the action generator remains frozen and no success labels are needed for the main target. |
| Implementation complexity | Medium. The key work is careful target definition/calibration, not a large network. |
| Reviewer perception | Stronger than a horizon head if the target is explicit, pre-registered, and scalar-ablation-controlled. |
| Advantages | Directly asks the execution question: how long is this group slice expected to remain usable after the source observation becomes stale? Uses available demonstrations; transfers to chunked ACT/VLAs without policy-internal attention. |
| Disadvantages / risk | Fresh-policy disagreement may not equal physical error; demonstration distributions may be narrow; the learned estimator can still learn task identity unless within-task controls are strong. |

**Verdict:** recommended primary direction.

### 6. Hybrid persistence estimator with posterior and cross-query evidence

| Aspect | Assessment |
|---|---|
| Core idea | Use self-supervised persistence as the main target, with group-wise sampled dispersion at source time and cross-query disagreement as calibrated features/monitors. |
| Signal source | Demonstration overlap, stochastic chunk spread when available, and free fresh-vs-buffer comparisons at shared queries. |
| Output | `h_t^g` plus a group refresh/retain gate at opportunities created by another group. |
| Training required | Auxiliary training and calibration; no base-policy update. |
| Implementation complexity | High. |
| Reviewer perception | Scientifically rich if ablated carefully; vulnerable to looking like an over-engineered collection of proxies. |
| Advantages | Separates prospective confidence, posterior uncertainty, and realized staleness; can expose why a group changed source. |
| Disadvantages / risk | Too many moving parts for the first dynamic paper; calibration leakage and query-budget confounds become harder to audit. |

**Verdict:** a later extension, not the first implementation.

### 7. Visual/state-residual group triggers

| Aspect | Assessment |
|---|---|
| Core idea | Predict expected visual/proprioceptive evolution under a chunk and trigger a group's refresh when observed state deviates from the forecast. |
| Signal source | World model, visual feature prediction, proprioceptive residual, or contact residual. |
| Output | Per-group event trigger or horizon. |
| Training required | Usually an auxiliary dynamics/world model. |
| Implementation complexity | High. |
| Reviewer perception | Potentially compelling for disturbance recovery, but close to online VLA-correction/event-triggered replanning and hard to attribute to a group. |
| Advantages | Measures actual post-action mismatch rather than only policy self-disagreement. |
| Disadvantages / risk | Needs action-to-observation attribution; risks turning the project into a world-model/corrector paper; one residual normally calls for a global replan. |

**Verdict:** reject as the first method. It is a different project unless a
clean group attribution is demonstrated.

### Ranking for the current project

| Rank | Candidate | Scientific novelty potential | Frozen-ACT fit | Chunked-VLA fit | LIBERO / RoboTwin / real-robot validation | Reviewer acceptance | Main condition |
|---:|---|---|---|---|---|---|---|
| 1 | Self-supervised group-wise chunk-persistence estimation | High | High | High | Medium–high | High if controls are strong | Demonstrate a valid persistence target and within-task adaptation. |
| 2 | Hybrid persistence + posterior/cross-query evidence | High | Medium | High | Medium | Medium | Add only after the single-target estimator is understood. |
| 3 | Group-wise posterior-dispersion commitment | Medium | Unknown for current ACT | High for diffusion/flow VLAs | Medium | Medium | Verify meaningful stochastic samples and calibration. |
| 4 | Calibrated cross-query consistency | Medium | High | High | High | Medium | Present as a minimal event-triggered baseline/diagnostic, not a threshold heuristic. |
| 5 | Learned vector horizon predictor | Medium | Medium | Medium | Low–medium | Low–medium | Must beat the “two DEHP heads” critique with a distinct target/evidence. |
| 6 | PACE-style group-wise phase boundaries | Low–medium | Low on current LIBERO actions | Medium | Medium in joint-space robots | Low | Require a faithful global PACE and a justified group metric first. |
| 7 | Visual/state-residual triggers | Medium | Low | Medium | Low initially | Low–medium | Requires a separate dynamics/attribution contribution. |

## Formalizing group-specific temporal predictability

Let a frozen policy issue a full chunk at source time `s`:

\[
\hat{A}_{s} = [\hat a_{s\mid s}, \ldots, \hat a_{s+L-1\mid s}],
\qquad \hat a^{g}_{s+k\mid s}\ \text{for group }g.
\]

The relevant quantity is not generic action magnitude. It is a group-specific
**retention risk** at future age `k`, conditional on information available when
the chunk was accepted:

\[
R_g(s,k) = \mathbb E[\ell_g(\hat a^{g}_{s+k\mid s}, a^{g,\mathrm{target}}_{s+k})
\mid I_s].
\]

`I_s` can contain the current policy observation, the full predicted chunk,
proprioception/history already available to the base policy, and the group
identity. The target must be specified before implementation. Two nonidentical
candidates are:

\[
\ell^{\mathrm{fresh}}_g =
 d_g(\hat a^g_{s+k\mid s},\hat a^g_{s+k\mid s+k}),
\]

where the second action is a fresh frozen-policy prediction at the future
demonstration observation, and

\[
\ell^{\mathrm{demo}}_g =
 d_g(\hat a^g_{s+k\mid s}, a^{g,\mathrm{demo}}_{s+k}).
\]

The first is a **policy-persistence/overlap** target: it asks whether re-querying
the same frozen policy would change group `g`'s action. The second is a
**demonstration-deviation** target: it asks whether the old action tracks the
recorded expert action. They should not be silently combined, because they test
different hypotheses. The first is most directly aligned with source-chunk
staleness; the second may better reflect control error but inherits demonstration
ambiguity.

An estimator can produce a calibrated conditional distribution or survival
probability instead of an arbitrary raw score:

\[
C_g(s,k) = \Pr\{\ell_g(s,k) \leq \epsilon_g \mid I_s\},
\qquad
h_g(s)=\max\{k\leq H_{\max}:C_g(s,k)\geq 1-\alpha_g\}.
\]

Here `C_g` is best called **retention confidence** only if calibration is
checked on held-out demonstrations; otherwise call it an estimated persistence
score. `\epsilon_g` and `\alpha_g` are group-specific because the action spaces
and errors are not comparable. They must be set using a training/validation
split, never the evaluation success sweep.

This formulation makes the intended causal chain explicit:

\[
\text{source observation/chunk} \rightarrow
\text{estimated group retention risk} \rightarrow
\text{group acceptance duration} \rightarrow
\text{mixed-generation execution trace}.
\]

It does not guarantee that a low marginal risk produces a good joint action. A
future method must therefore condition on enough shared chunk/observation
context to represent dependence and must report cross-group failure cases.

### Measurable signals for `C_g(t)`

| Signal | General-purpose? | What it measures | Caveat |
|---|---|---|---|
| Cross-query overlap disagreement | Yes, at existing query opportunities | Difference between retained old group action and a fresh current-observation group action | Cannot by itself request an earlier query. |
| Future-observation chunk consistency | Yes, offline from demonstrations | How predictions at source time differ from fresh predictions at future observed states | Proxy for staleness, not direct task loss. |
| Demonstration deviation | Yes, offline | Difference from recorded expert action | Demonstrator multimodality and off-distribution recovery limit it. |
| Posterior/sample dispersion | Policy dependent | Predictive uncertainty/multimodality by group and future index | Requires verified sample semantics; scale calibration is essential. |
| Policy internals (attention, logits, latent statistics) | Architecture dependent | Internal predictive-limit proxy | Not black-box portable; does not automatically mean calibrated confidence. |
| Kinematic smoothness/curvature/valleys | General only with semantically defined action geometry | Planned phase structure | Motion magnitude is not validity; relative translation and rotation cannot be mixed casually. |
| Visual/proprioceptive prediction residual | Potentially general, but requires model/attribution | Realized deviation after execution | Usually motivates a global correction, not a group-specific one. |
| Contact state, force/torque, tactile slip | Dexterous/contact-specific | Physical interaction uncertainty and grasp state | Valuable later, but not required for the general-purpose first method and unavailable in the current LIBERO path. |

## What the static results imply—and what they do not

The task-0 result `(arm=4, gripper=16)` compared with global `h=4` is
especially useful because both have approximately the same query rate (about
0.252). The difference cannot be attributed simply to more full policy
evaluations. The complete task-0 grid also shows that swapping a pair of horizon
values changes success while leaving the nominal query schedule unchanged. This
is strong evidence that the physical assignment of retention duration matters
for this policy/task/executor.

It is reasonable to hypothesize that the arm and gripper have different
temporal predictability or different tolerance for source-chunk replacement.
For example, longer gripper retention may preserve a coherent grasp/release
intent while the arm benefits from more frequent spatial correction. But the
same results have other compatible explanations: source-transition continuity,
controller nonlinearities, dataset conventions, or mixed-generation coupling.
The existing trace variation statistics are descriptive and cannot choose among
these mechanisms.

Most importantly, a static `(4,16)` win does not establish `h_arm(t) !=
h_gripper(t)` *within a single episode*. It is consistent with either:

- **A. Task-level static selection:** different object tasks simply prefer
  different fixed pairs; a universal pair already captures most of the gain.
- **B. Within-task phase variation:** the same task has phases in which each
  group's safe retention duration changes, and a dynamic schedule can beat its
  best fixed pair at comparable budget.

### Future experiment needed to separate A from B

Pre-register a phase-variation test before method tuning:

1. Use a fixed checkpoint, fixed task, paired official states, and one
   predeclared group partition.
2. Compare the best deployable universal static pair, per-task static selection
   (clearly marked retrospective if selected from the same data), a scalar
   dynamic version of the proposed signal, and the group-wise dynamic rule.
3. Log each `h_g`, group source age, source generation, and full policy-query
   count. Report success/query Pareto points, not success alone.
4. Construct a **within-task time-shuffled control**: preserve the empirical
   per-group horizon histogram and nominal query budget, but shuffle the
   scheduling decisions within an episode or across matched episodes. If the
   original schedule does not beat this control, the evidence supports a
   distribution of horizons, not phase-aware timing.
5. Where benchmark semantics allow it, use predeclared, observation-derived
   phase labels (approach, grasp/contact, transport, release) only for analysis,
   not as scheduler input. Test whether selected horizons differ across these
   labels and whether the relation is consistent across tasks.
6. Inspect whether group updates increase near the relevant phase *without*
   selectively changing only successful episodes. Report successful and failed
   traces separately.

This test is more informative than first expanding a horizon grid. It also
guards against a learned selector that merely predicts task identity or episode
time.

## Recommended research path

### Primary recommendation

Proceed only after an offline **group-wise chunk-persistence measurement study**
shows both of the following on held-out demonstrations:

1. the proposed persistence target is calibrated or at least predictive of its
   held-out error; and
2. its group- and time-dependence is not reducible to a task-specific constant.

If both hold, the first dynamic method should be the self-supervised
group-wise chunk-persistence estimator (Candidate 5), with the action generator
frozen. This is a clearer ICRA story than a group-wise PACE heuristic:

- **Question:** when does a group component of a frozen joint chunk remain
  prediction-consistent as observations evolve?
- **Signal:** demonstration-derived, group- and future-age-specific prediction
  persistence.
- **Decision:** a vector of group acceptance durations, not a global boundary.
- **Novelty test:** scalar aggregation of exactly the same persistence signal
  cannot explain the group-wise result.

Do not call the approach “confidence-based” unless calibration is actually
demonstrated. Do not call the method dynamic until it has shown nontrivial
within-episode variation and a benefit over the best static group-wise pair at a
reported query budget.

### Staged roadmap

| Stage | Work | Expected value | Main risk | Relative effort |
|---|---|---|---|---|
| 0 — completed | Static global and group-specific landscapes, paired seeds, query accounting, source-age traces | Establishes an execution-only phenomenon and strong static baselines | Limited to one policy/suite; no causal mechanism | Completed |
| 1 — offline falsification | Measure `\ell_g^fresh` and `\ell_g^demo` by group/age on held-out demonstrations; test calibration, task effects, and within-trajectory variation | Determines whether there is an observable group-persistence signal before building a scheduler | Offline disagreement may not predict closed-loop success | Low–medium |
| 2 — minimal dynamic diagnostic | Calibrated cross-query consistency gate at existing shared queries, with explicit static caps and scalar counterpart | Cheap test of group-specific stale-vs-fresh replacement | Cannot initiate all desired early queries; threshold risk | Low–medium |
| 3 — primary method | Self-supervised group-wise chunk-persistence estimator with risk-calibrated horizons | Direct, frozen-policy answer to the scientific question | Learns task identity or fails to transfer from demonstrations | Medium |
| 4 — robustness extension | Add posterior dispersion/cross-query evidence only if Stage 3 shows target limitations; validate on LIBERO, then a semantically verified joint-space RoboTwin path and real robot | Tests whether uncertainty improves a persistence target | Scope/compute/coupling complexity | Medium–high |
| 5 — learned return optimization | Vector horizon policy/critic only if self-supervision leaves a meaningful oracle gap | Can optimize task return and interaction effects | Collides directly with DEHP and requires online data/rewards | High |

### Required comparators for a future primary method

The future evaluation must not compare only against one global static horizon.
At minimum include:

1. best universal global fixed horizon;
2. best universal static group-wise pair, currently `(4,16)` on the common
   LIBERO Object set;
3. per-task best static global and group-wise values explicitly labeled as
   retrospective oracles;
4. a **scalar dynamic** baseline using the same proposed persistence signal;
5. a same-compute or reported-query-rate static comparison;
6. a time-/decision-shuffled group-wise control with the same horizon
   distribution; and
7. the global PACE baseline only after the authors provide enough information to
   make the current relative-OSC adaptation faithful, or after moving to an
   action representation for which PACE is fully specified.

This list prevents an apparent gain from being attributed solely to extra policy
queries, a learned task ID, or any scalar dynamic signal that would work without
independent group retention.

## Rejected directions for the first dynamic study

- **Independent PACE per group.** It is too easy to characterize as a direct
  split of a scalar phase-boundary method. The current relative end-effector
  action representation also lacks a source-defined PACE kinematic profile.
- **Two hand-tuned thresholds on arm/gripper action norms.** This is not
  calibrated confidence, mixes action units, and creates an arbitrary heuristic.
- **Immediate vectorized DEHP.** It would be a horizon-head extension trained on
  return before the project has established that within-task adaptation is
  necessary.
- **Force/tactile-first scheduler.** Contact signals are promising for
  dexterous manipulation but would obscure whether the core execution effect is
  general. They belong in a later modality-specific extension.
- **World-model correction as the core method.** It changes the scientific
  problem toward observation prediction and correction, with unclear group
  attribution.
- **Task-ID/static lookup masquerading as dynamic execution.** Useful as a
  diagnostic baseline, not the contribution.

## Open research questions

1. Does frozen-policy fresh-vs-stale disagreement predict actual closed-loop
   error or merely policy indecision?
2. Which target is more useful: fresh-policy consistency, demonstration
   deviation, or a calibrated combination? This must be decided using held-out
   demonstration criteria before rollout success is examined.
3. Can the arm group's relative translation and rotation be represented in a
   controller-consistent error metric without introducing arbitrary weights?
4. Does a group-specific confidence estimator require group-specific action
   normalization, and can that calibration transfer across embodiments?
5. How often do independently retained components become cross-group
   incompatible, and can that be measured without introducing a separate
   synchronization method?
6. Does the remaining static-oracle gap come from phase variation, task
   variation, finite-sample noise, or the coarse horizon grid?
7. Are group source ages, not just selected `h_g`, the right explanatory
   variables once shared policy-query boundaries are accounted for?
8. Can a method trained only on demonstration observations remain useful during
   recovery/off-demonstration states?

## Sources inspected

Primary/official sources inspected for this memo:

- Zhao et al., [*Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware* (ACT)](https://arxiv.org/abs/2304.13705).
- Chi et al., [*Diffusion Policy: Visuomotor Policy Learning via Action
  Diffusion*](https://arxiv.org/abs/2303.04137).
- Kim et al., [*OpenVLA: An Open-Source Vision-Language-Action
  Model*](https://arxiv.org/abs/2406.09246).
- Black et al., [$\pi_0$: *A Vision-Language-Action Flow Model for General Robot
  Control*](https://arxiv.org/abs/2410.24164).
- Black et al., [*Real-Time Execution of Action Chunking Flow
  Policies*](https://arxiv.org/abs/2506.07339).
- Wang et al., [*VLA Knows Its Limits: Adaptive Execution Horizons for Robot
  Policies*](https://arxiv.org/abs/2602.21445).
- Liang et al., [*Adaptive Action Chunking at Inference-time for
  Vision-Language-Action Models*](https://arxiv.org/abs/2604.04161).
- Zhang et al., [*HiPolicy: Hierarchical Multi-Frequency Action Chunking for
  Policy Learning*](https://arxiv.org/abs/2604.06067).
- Shin et al., [*Adaptive Action Chunking via Multi-Chunk Q Value
  Estimation*](https://arxiv.org/abs/2605.10044), and Gireesh et al.,
  [*Adaptive Q-Chunking for Offline-to-Online Reinforcement
  Learning*](https://arxiv.org/abs/2605.05544).
- Nie et al., [*PACE: Phase-Aware Chunk Execution for Robot Policies with
  Action Chunking*](https://arxiv.org/abs/2606.00537).
- Feng et al., [*Denoising Tells When to Replan: Denoising-Variance Adaptive
  Chunking for Flow-Based Robot Policies*](https://arxiv.org/abs/2606.03847).
- Zhao et al., [*Dynamic Execution Horizon Prediction for Chunk-based Robot
  Policies*](https://arxiv.org/abs/2606.11408).
- Weng et al., [*Temporal Action Selection for Action
  Chunking*](https://arxiv.org/abs/2511.04421).
- Krener, [*Adaptive Horizon Model Predictive
  Control*](https://arxiv.org/abs/1602.08619), plus the event-/self-triggered
  control references identified in the targeted control-theory search.

The project-local PACE audit is a material source for this memo because it
documents a reproducibility boundary specific to the verified LIBERO relative
OSC action contract. No secondary PACE reimplementation was treated as a source
of algorithmic truth.

# ICRA 2027 Candidate Specification

Status: **conditional paper specification; Gate-3 implementation is not yet
recommended**. This document follows the paper-before-experiments discipline but
does not treat the proposed method as established.

## Problem

A chunked robot policy can produce several predictions for action time (t), each
conditioned on an observation from a different source time. Fixed ACT temporal
ensembling applies a context-independent weighting to these predictions. Pure
independent group routing may lower offline component error but can compose an
action vector that the policy never predicted jointly. The problem is to use
temporally overlapping predictions without either discarding contextual
competence differences or violating joint-action consistency.

## Verified observation

On 5,118 held-out teacher-forced LIBERO actions with sparse saved predictions,
a validation-selected similarity ensemble reduces group-balanced normalized
error from .7728 (newest only) to .7084; the episode-bootstrap CI for the
difference is [-.0864,-.0453]. A hard scalar oracle reaches .4788 and an
independent arm/gripper oracle .4130. The group advantage falls from .0658 under
group-balanced weighting to .0301 under normalized dimension weighting. These
are offline sparse-source results and do not establish control improvement.

Separately, a matched-query group-wise accept/retain rule decreases success by
.20 to .29 absolute across three query periods. Thus the evidence supports
temporal candidate diversity and simultaneously warns that unconstrained
cross-generation composition can be harmful.

## Hypothesis

> A selector trained on control-aligned temporal competence can outperform fixed
> full-action temporal ensembles in closed loop, but independent group selection
> helps only when constrained by the consistency of the jointly predicted
> action.

This hypothesis is falsified if dense held-out analysis removes the non-oracle
advantage, if group benefit is not robust to justified control losses, or if
offline gains do not improve paired rollouts.

## Mechanism

For a target time (t), let (E_{t,k}) be the full action predicted from source
observation (o_{t-k}). Different ages encode different temporal relationships:
fresh predictions can react to current state, while older predictions may be
smoother or remain in a previously selected action mode. A full-action selector
chooses or mixes those joint candidates. Group freedom can correct a component
whose competence differs by age, but incurs a distributional cost when arm and
gripper come from incompatible source chunks. The proposed constraint trades
off the predicted component gain against observable cross-source disagreement,
boundary jerk, and joint-source distance.

## Method

Working name: **Consistency-Constrained Temporal Selection (CCTS)**.

Given available temporal candidates, CCTS first produces scalar source weights
(w_k) for the complete action. It may add a group-specific residual
(delta w^g_k) only inside a bounded trust region:

\[
\tilde a_t^g = \sum_k (w_k + \delta w^g_k) E^g_{t,k}.
\]

The core contribution is not merely the router. It is a **joint-action
consistency constraint that grants group-specific temporal freedom only when
the predicted control-aligned gain exceeds the incompatibility cost**. A
minimal objective is

\[
L = L_{control}(\tilde a_t,a_t^*)
  + \lambda_c L_{cross\text{-}source}
  + \lambda_s L_{boundary\ smoothness}.
\]

For LIBERO, (L_control) must separate translation, rotation, and gripper
sign/transition timing. It must not penalize behaviorally irrelevant gripper
magnitude. The base ACT policy stays frozen. A scalar-only model, hard
joint-source selection, and unconstrained independent group routing are required
ablations.

This formulation remains provisional until the dense oracle shows that a
consistency-constrained group oracle retains meaningful advantage.

## Novelty

- ACT uses fixed exponential weights over overlapping predictions.
- CogACT adapts full-action weights using candidate similarity.
- TAS learns to select cached temporal actions with online RL.
- MoH trains explicit prediction-horizon experts and a fusion gate.
- Adaptive-horizon methods choose how long to execute one chunk before re-query.

CCTS would be different only through the conjunction of: (1) an explicit
control-aligned competence target, (2) optional actuator-group temporal choice,
and (3) an explicit joint-consistency constraint governing that choice. Removing
the group-consistency mechanism collapses the proposal into crowded prior art.

## Killer experiment

After the dense offline gate passes, run one preregistered, paired,
policy-query-matched LIBERO experiment on untouched initial states across all ten
tasks. Compare:

1. newest prediction;
2. ACT exponential temporal ensemble;
3. uniform ensemble;
4. CogACT-style similarity ensemble;
5. validation-selected fixed age;
6. learned scalar temporal selector;
7. unconstrained group selector;
8. CCTS;
9. a smoothness-only scheduler/ensemble baseline.

Primary outcome is paired task success. Secondary outcomes are ACT queries,
wall-clock feasibility, boundary jerk, action discontinuity, and predeclared
control-aligned offline error. The convincing result is not an isolated mean:
CCTS must beat the strongest scalar/nonlearned baseline with a task-clustered
interval excluding zero, without increased query count, and its improvement must
persist across tasks rather than come from one task.

## Kill condition

Stop or reframe before model implementation if any of the following occurs:

- In the dense cache, similarity/exponential ensembling no longer beats newest
  prediction on held-out task/episode splits.
- The dense independent-group oracle fails to improve over the scalar oracle
  under both control-aligned and normalized-dimension losses, or the improvement
  is not stable across tasks and bootstrap resamples.
- A consistency-constrained oracle loses essentially all independent-group gain.
- A smoothness-only or similarity heuristic matches the proposed selector.
- The paired closed-loop experiment shows no success improvement over the
  strongest query-matched scalar baseline, regardless of offline MSE gain.

The first two checks require only dense inference and analysis; no policy change
is justified before they pass.

## Required baselines

- newest-only and oldest-valid prediction;
- best validation-selected fixed age;
- fixed execution horizons at matched physical time;
- ACT exponential temporal ensemble;
- uniform temporal ensemble;
- CogACT-style similarity weighting;
- TAS-style full-action selection, reproduced faithfully if code is obtainable;
- scalar learned router with the same capacity/features;
- independent group router without consistency constraint;
- PACE-style kinematic/smoothness baseline where action semantics permit;
- phase/time-only, velocity/curvature, and disagreement-only predictors;
- global replacement and the historical tested selective-retain rule;
- oracle scalar, oracle independent group, and oracle consistency-constrained
  group bounds.

Adaptive prefix methods such as AutoHorizon/AAC/DVAC/A3/BCP are mandatory in a
broader VLA evaluation if CCTS also makes a re-query-efficiency claim; they are
not substitutes for temporal-selection baselines.

## Required ablations

- scalar versus group residual weights;
- hard selection versus convex mixture;
- translation, rotation, and gripper sign/transition loss components;
- normalized-dimension versus group-balanced weighting;
- no consistency term, source-age-disparity term only, prediction-disagreement
  term only, and boundary-smoothness term only;
- with and without temporal age, observation/state, policy latent, action
  similarity, and local smoothness features;
- dense versus sparse candidate cadence;
- 10 Hz demonstration time versus 20 Hz rollout physical-time alignment;
- task-held-out, episode-held-out, and event-held-out generalization;
- contact/event labels versus normalized time;
- deterministic and stochastic chunk policies where available.

## Generalization claim

The minimal defensible claim requires consistent results across the ten LIBERO
Object tasks with an untouched task/state selection protocol and a second policy
family. A VLA claim requires at least one modern VLA/flow policy; ACT-only results
cannot support it. Group-specific claims require an action space with meaningful
group semantics and cannot be generalized from one binary gripper alone.

## Real-robot relevance

The mechanism could transfer to arm/gripper or bimanual systems because temporal
candidate caches already exist in many chunked controllers. Transfer requires:
exact per-group units and semantics; a consistency metric in physical or induced
trajectory space; safety bounds on mixed-generation actions; real-time feature
latency; and paired evaluation under identical query compute. The current
RoboTwin and dexterous-system artifacts provide no empirical evidence of such
transfer.

## Figure 1 concept

A three-panel figure:

1. overlapping chunks predict the same action from observation ages 0, 1, 2,
   ...;
2. scalar ACT/CogACT/TAS choices preserve a joint source, whereas unconstrained
   arm/gripper choices can create an off-manifold composite;
3. CCTS permits a group deviation only inside a consistency region, followed by
   the decisive plot: closed-loop success versus query rate for ACT ensemble,
   similarity, unconstrained group routing, and CCTS.

The figure must visually distinguish prediction horizon, temporal age, query
interval, and execution horizon.

## Next experimental Gate

**Gate-3A: dense temporal-evidence gate (offline, no training).**

- Query the frozen ACT checkpoint at every demonstration step for a stratified,
  task-balanced subset and save full chunks with hashes and exact provenance.
- Evaluate all requested non-oracles and oracles under control-aligned losses,
  task/episode bootstrap, physical-time resampling, smoothness conditioning, and
  consistency constraints.
- Predeclare selection on train/validation episodes and report untouched test
  episodes/tasks.
- If Gate-3A passes, proceed to a small scalar-versus-group selector prototype;
  otherwise stop group routing.

**Current decision: do not implement Gate-3 policy code yet.**


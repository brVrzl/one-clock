# Candidate Research Ranking

Audit date: 2026-08-21. Scores are decision aids, not measurements. Each score is
1 (unfavorable) to 5 (favorable). For `ease`, `cost`, and `speed`, 5 means easier,
cheaper, or faster. For `calibration`, `heuristic`, and `offline metric`, 5 means
less dependence/risk. Evidence scores are constrained by the verified fact sheet;
novelty scores are constrained by the literature re-audit.

## Candidate definitions

- **A — Better reliability estimator:** improve the existing binary/prefix target
  pipeline without changing the basic estimand.
- **B — Dynamic group-wise execution horizon:** choose separate arm/gripper
  commitment lengths; no explicit ensemble.
- **C — Group-wise Temporal Expert Routing:** choose or mix cached source-time
  action predictions independently by group.
- **D — Group-wise Mixture of Horizons:** retrain the action policy with explicit
  horizon experts and group gates.
- **E — Value of Freshness:** predict the marginal value of a new observation and
  policy query rather than a reliability label.
- **F — Temporal routing plus Value of Freshness:** jointly route cached experts
  and decide whether to re-query.
- **G — Consistency-Constrained, Control-Aligned Temporal Selection:** start with
  a joint/scalar temporal selector, permit group deviations only when their
  predicted control-aligned gain exceeds a cross-generation consistency cost.

## Scientific and publication scores

| Family | Empirical support | Evidence robustness | Novelty | Policy upside | Closed-loop mechanism | Scientific depth | ICRA fit | Reviewer defensibility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 1 |
| B | 2 | 1 | 3 | 2 | 3 | 3 | 4 | 2 |
| C | 3 | 2 | 2 | 4 | 3 | 4 | 4 | 2 |
| D | 1 | 1 | 2 | 4 | 4 | 4 | 4 | 2 |
| E | 1 | 1 | 2 | 3 | 5 | 5 | 4 | 2 |
| F | 1 | 1 | 2 | 5 | 4 | 5 | 4 | 1 |
| **G** | **3** | **2** | **3** | **4** | **4** | **5** | **5** | **3** |

## Execution and transfer scores

| Family | Implementation ease | Experiment cost | Speed to decisive evidence | ACT compatibility | VLA extensibility | Real-robot extensibility | Low calibration dependence | Low heuristic risk | Low dependence on questionable offline metric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 4 | 4 | 5 | 5 | 4 | 4 | 1 | 2 | 1 |
| B | 3 | 3 | 3 | 5 | 4 | 3 | 3 | 2 | 2 |
| C | 3 | 3 | 4 | 5 | 4 | 3 | 3 | 3 | 2 |
| D | 1 | 1 | 2 | 2 | 5 | 2 | 4 | 4 | 3 |
| E | 2 | 2 | 3 | 5 | 5 | 4 | 2 | 4 | 4 |
| F | 1 | 1 | 1 | 4 | 4 | 2 | 2 | 4 | 3 |
| **G** | **3** | **3** | **5** | **5** | **4** | **3** | **4** | **4** | **3** |

## Rank and rationale

| Rank | Family | Decision |
|---:|---|---|
| 1 | **G** | Primary **research gate**, not yet an authorized method implementation. It is the only candidate that directly addresses the strongest positive observation (temporal ensemble headroom), the strongest negative observation (harmful cross-generation group composition), and the gripper/loss artifact. Its novelty is still only conditional. |
| 2 | C | Useful ablation and simpler precursor, but a generic learned selector is too close to TAS and a generic similarity mixture too close to CogACT. |
| 3 | E | Scientifically clean estimand and strong mechanism, but the project has no valid value target yet and the operational space is crowded by BCP/DEHP/A2C2/adaptive-horizon work. |
| 4 | B | Easy to explain and potentially distinct at group level, but direct matched-query evidence for the tested rule is strongly negative and marginal Gate-2B optima do not prove group timescales. |
| 5 | D | Potential upside, but MoH occupies the central idea and this would require policy retraining before the key empirical premise is established. |
| 6 | A | Can remain a diagnostic baseline. The binary target is lossy, threshold-sensitive, and disconnected from rollout success. |
| 7 | F | Premature composition of two unverified mechanisms; highest scope and confounding risk. |

No aggregate score is reported because equal-weight averaging would conceal the
fatal novelty and evidence constraints.

## Primary direction

**Investigate consistency-constrained, control-aligned selection among temporal
predictions, with group-specific freedom treated as an ablated extension.**

The precise scientific question is:

> When overlapping action chunks offer different predictions for the same
> physical action, can a selector improve closed-loop control beyond ACT and
> CogACT-style ensembles, and does independently selecting actuator groups add
> benefit after accounting for action semantics and cross-generation
> inconsistency?

This is not yet a recommendation to implement Gate-3. The first gate is a dense
every-step oracle and baseline analysis with no policy changes.

## Fallback direction

**A scalar, mechanism-focused temporal-ensemble study.** If group advantage
fails, drop all “one clock does not fit all” claims. Study when ACT's implicit
temporal experts help, compare fixed/exponential/uniform/similarity/learned
full-action selection, and connect disagreement and boundary consistency to
paired closed-loop outcomes. This may support an analysis-plus-simple-baseline
paper; it is not novel merely as temporal ensembling.

## Stop-doing list

- Do not train another binary prefix-reliability head on the current target.
- Do not implement a generic dynamic-horizon head; the novelty is occupied.
- Do not use Gate-2B's point-max maps as supervision or ground truth.
- Do not call fixed time-limit thirds semantic task phases.
- Do not claim arm/gripper timescales from marginal argmax differences.
- Do not mix independently aged action groups without a direct consistency and
  closed-loop test.
- Do not use continuous gripper magnitude MSE as a LIBERO control objective.
- Do not infer policy improvement from teacher-forced MSE.
- Do not launch broad RoboTwin or real-robot evaluation until an exact policy,
  checkpoint, upstream commit, and action contract are pinned.
- Do not combine freshness and routing until each has independent headroom.


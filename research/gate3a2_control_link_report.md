# Gate-3A2 temporal aggregation control-link report

**GATE DECISION: CONTROL-LINK-POSITIVE**

**TIME CONTRACT: TIME-CONTRACT-MISMATCH (RESOLVED BEFORE ROLLOUT)**

**PRIMARY RESULT:** Gate-3A1's newest-favoring age exponential succeeds on
54/100 frozen task-state blocks versus 41/100 for exact ACT. The paired
difference is +0.13, with 95% paired-state bootstrap CI `[+0.05,+0.22]` and
task-cluster CI `[+0.03,+0.23]`.

**BOUNDARY OF THE RESULT:** The age rule is numerically above newest-only
(54/100 versus 44/100) and tuned CogACT (54/100 versus 48/100), but both paired
intervals include zero. This is not `STRONG-CONTROL-LINK`, does not establish
superiority to all strong baselines, and does not make newest-favoring ACT
weighting novel.

**AUTHORIZED SCIENTIFIC CONCLUSION:** Temporal-source weighting is a real
closed-loop control lever relative to exact ACT on this frozen checkpoint and
cohort. Gate-3A1's offline ranking has deployment relevance for the age-rule
versus exact-ACT contrast. It is not yet a generally validated surrogate for
closed-loop ranking.

## 1. Question and preregistration

Gate-3A2 asked:

> Does dense teacher-forced temporal-aggregation ranking predict
> query-cadence-matched closed-loop LIBERO task success?

The time contract, initial-state cohort, treatment order, methods, primary
outcome, comparisons, bootstrap units, and decision logic were frozen in
[`gate3a2_preregistered_protocol.md`](gate3a2_preregistered_protocol.md) and
pushed as commit `e3ff506caed44fb685a16e4a1158b6c5de6ac2bc` before an official outcome
was generated. No protocol amendment, episode exclusion, failed run, or rerun
occurred.

The complete design contains ten LIBERO Object tasks, ten deterministically
sampled official initial states per task, and four scalar aggregation rules in
randomized within-block order: 100 task-state blocks and 400 episodes. Each
condition queries the same frozen ACT once per surviving 20 Hz controller step
and differs only in scalar weighting of valid predictions for the current
action.

## 2. Corrected time and prior-art contracts

The blocking audit in
[`gate3a2_time_contract_audit.md`](gate3a2_time_contract_audit.md) found that
the local demonstration copy's 10 Hz metadata is wrong as physical provenance.
The saved sequence is an unreduced copy of 20 Hz LIBERO content. One saved
action index and one rollout tick both represent 0.05 s; a 100-action chunk
represents 5 s. Gate-3A1's selected `beta=0.03/index` therefore maps to
`beta=0.03/tick`, or `0.6 s^-1`. No rollout used the proposed but incorrect
`0.015/tick` conversion.

Pinned LeRobot orders ACT contributors oldest to newest and assigns
`exp(-m i)`, with original ACT `m=+0.01` favoring older predictions. It
explicitly interprets negative coefficients as favoring newer predictions.
[LeRobot PR #319](https://github.com/huggingface/lerobot/pull/319) publicly
evaluated negative coefficients before this project. Newest-favoring temporal
weighting is therefore prior art and is not the contribution of Gate-3A2.

## 3. Frozen methods

For physical action time `t`, source `q` contributes when
`q <= t < q+100`. Candidates are ordered oldest to newest and combined by one
shared scalar weight vector over the full seven-dimensional action.

| Label | Rule | Frozen parameter |
|---|---|---:|
| A | Newest prediction only | age 0 |
| B | Exact pinned ACT source-order exponential | `m=+0.01` |
| C | [Released CogACT full-action cosine weighting](https://github.com/microsoft/CogACT/blob/b174a1b86deedfab4d198d935207e7bb0527994e/sim_cogact/adaptive_ensemble.py) | validation-selected `alpha=0.3` |
| D | Newest-favoring age exponential | validation-selected `beta=0.03/tick` |

All methods use the same raw arithmetic action aggregation operator. No method
uses a demonstration target, future observation, extra ACT query, group-wise
weights, learned selector, or policy update.

## 4. Provenance and execution validity

- Model SHA256:
  `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`.
- Config SHA256:
  `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d`.
- Pinned LeRobot commit:
  `f66e5128ecb2456e8c54a63d15404fa59c16aebc`.
- Controller: 20 Hz, relative LIBERO controller, 280-step limit.
- Schedule SHA256:
  `d27563df989d7f497cf7aafb2f6d6a1ecafa7fe797f0555e59acc24b2e389d66`.
- Completed episodes: 400/400.
- Environment steps and ACT queries: 85,942 each.

The committed [rollout manifest](audit_outputs/gate3a2_rollout_manifest.json)
links every cell to an atomic local log and SHA256. The post-result read-only
[validation output](audit_outputs/gate3a2_rollout_validation.json) establishes:

- 400 unique task-state-method cells with no missing cell;
- one ACT query per executed environment step in every episode;
- exact dense candidate counts `min(t+1,100)`;
- finite 7-D executed actions throughout;
- fixed-rule effective ages exactly matching their registered equations;
- identical first actions across all four methods in all 100 paired blocks.

The final point is a useful deterministic-reset check: before trajectories can
diverge, every condition receives the same initial information and executes
the same single available candidate.

## 5. Primary outcome

| Method | Success | Rate | Environment steps | ACT queries | Queries / surviving step |
|---|---:|---:|---:|---:|---:|
| A — newest | 44/100 | .44 | 22,341 | 22,341 | 1.000 |
| B — exact ACT, `m=+.01` | 41/100 | .41 | 22,058 | 22,058 | 1.000 |
| C — tuned CogACT, `alpha=.3` | 48/100 | .48 | 21,499 | 21,499 | 1.000 |
| D — newest-age exponential, `beta=.03/tick` | **54/100** | **.54** | 20,044 | 20,044 | 1.000 |

Total queries differ because successful episodes can terminate early. The
conditions are matched in policy-query cadence per surviving step, not in total
episode compute.

### Frozen paired comparisons

The right method minus left method is positive when the right method succeeds
more often.

| Comparison | Difference | Paired-state 95% CI | Task-cluster 95% CI | Right-only / left-only | Exact McNemar p | Frozen stability |
|---|---:|---:|---:|---:|---:|---|
| D − A, age rule minus newest | +.10 | [-.02,+.21] | [-.08,+.28] | 23 / 13 | .1325 | unresolved |
| D − B, age rule minus exact ACT | **+.13** | **[+.05,+.22]** | **[+.03,+.23]** | **17 / 4** | **.0072** | stable positive |
| D − C, age rule minus tuned CogACT | +.06 | [-.01,+.13] | [.00,+.13] | 10 / 4 | .1796 | unresolved |
| C − B, tuned CogACT minus exact ACT | +.07 | [+.02,+.13] | [.00,+.16] | 8 / 1 | .0391 | unresolved by frozen task-stability rule |

The McNemar values are diagnostics, not a replacement for the preregistered
episode and task-cluster bootstraps. Complete machine-readable values are in
[`gate3a2_pairwise_comparisons.csv`](audit_outputs/gate3a2_pairwise_comparisons.csv).

### Task decomposition

| Task | Newest | Exact ACT | CogACT | Age exponential | D−B | D−A | D−C |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5/10 | 3/10 | 7/10 | 7/10 | +.4 | +.2 | .0 |
| 1 | 1/10 | 4/10 | 4/10 | 4/10 | .0 | +.3 | .0 |
| 2 | 5/10 | 5/10 | 5/10 | 7/10 | +.2 | +.2 | +.2 |
| 3 | 3/10 | 7/10 | 9/10 | 9/10 | +.2 | +.6 | .0 |
| 4 | 7/10 | 9/10 | 8/10 | 9/10 | .0 | +.2 | +.1 |
| 5 | 3/10 | 3/10 | 4/10 | 7/10 | +.4 | +.4 | +.3 |
| 6 | 3/10 | 2/10 | 2/10 | 1/10 | −.1 | −.2 | −.1 |
| 7 | 4/10 | 1/10 | 1/10 | 1/10 | .0 | −.3 | .0 |
| 8 | 3/10 | 1/10 | 1/10 | 2/10 | +.1 | −.1 | +.1 |
| 9 | 10/10 | 6/10 | 7/10 | 7/10 | +.1 | −.3 | .0 |

The D−B advantage is positive in six tasks, zero in three, and negative in one;
every leave-one-task-out mean remains positive. D−A is positive on tasks 0–5
but negative on tasks 6–9, explaining its wide task-cluster interval. The
experiment therefore does not justify a universal claim that temporal
aggregation beats newest-only.

## 6. Secondary control diagnostics

| Method | Mean effective age | Translation action delta | Rotation action delta | Raw action jerk | Gripper transitions |
|---|---:|---:|---:|---:|---:|
| Newest | 0.00 ticks / 0.00 s | .04795 | .00528 rad | .14863 | 3.51 |
| Exact ACT | 42.04 ticks / 2.10 s | .04187 | .00394 rad | .01739 | 2.79 |
| CogACT | 35.54 ticks / 1.78 s | .04316 | .00398 rad | .03279 | 2.50 |
| Age exponential | 21.45 ticks / 1.07 s | .04510 | .00405 rad | .02045 | 2.55 |

These are episode-weighted descriptive summaries along each method's different
closed-loop trajectories, not paired causal mediators. Exact ACT is the
smoothest by the raw jerk diagnostic yet has the lowest success rate, so
smoothness alone cannot explain the observed success ordering. That observation
does not identify the true mechanism.

## 7. Scientific-critical interpretation

### What survived

Gate-3A1 ranked D above B under held-out teacher-forced `L_sem`; Gate-3A2 ranks
D above B in closed-loop success with stable paired and task-cluster evidence.
The only manipulated execution variable is temporal-source weighting. For this
specific contrast, offline temporal aggregation contained deployment-relevant
information.

The numerical four-method success order is D, C, A, B. Gate-3A1's offline order
was D, C, B, A. The top two agree, but exact ACT and newest swap. With only four
methods and unresolved D−A/D−C contrasts, this is not evidence that `L_sem`
generally predicts policy ranking.

### What did not survive or remains unresolved

- D does not clearly outperform tuned CogACT.
- D does not clearly outperform newest-only across tasks.
- The result does not validate Gate-3A1's large target-informed scalar oracle
  headroom or authorize a learned selector.
- One checkpoint, one benchmark suite, one deterministic trial per official
  state, and ten states per task do not establish transfer or small effects.
- Bootstraps describe variation over the selected state/task structure; they do
  not estimate stochastic simulator-seed sensitivity.
- Earlier termination lowers D's total query count. This is an outcome, not a
  compute advantage at a fixed trajectory length.

### Causal claim boundary

Because policy, per-step query cadence, candidate-validity rule, raw
aggregation, environment, initial state, and seed are controlled, the D−B
result supports a causal effect of assigning the scalar source-weight rule on
this cohort. Candidate values and observations appropriately diverge after the
first treatment-dependent action; they are downstream parts of the closed-loop
effect, not matched inputs. The result does not establish why the effect occurs,
nor that freshness bias is the uniquely responsible feature among differences
in the two weighting profiles.

## 8. Internal peer-review assessment

**General ICRA manipulation reviewer.** The complete blocked design, frozen
success outcome, exact provenance, and task-cluster analysis make the control
result credible. It is not a paper contribution by itself because the winning
rule is prior art and gains are unresolved against tuned CogACT and newest.

**ACT/CogACT reviewer.** The exact-ACT baseline is correctly ordered and the
CogACT comparison uses its released full-action cosine rule with a
validation-selected alpha. The next contribution must isolate a new adaptive
delta over D and C; calling D a new method would be rejectable.

**Skeptical reviewer.** Ten states per task are enough to detect the large D−B
contrast but leave D−A and D−C uncertain. A publishable next result needs
replicated gains over both the strong age prior and tuned CogACT, not only over
original ACT, plus a second policy/benchmark or substantially stronger
mechanistic evidence.

## 9. Decision and next authorization boundary

The preregistered outcome is **CONTROL-LINK-POSITIVE**, not strong. The dense
offline result has enough control relevance to justify a later minimal adaptive
method study built on the strong age prior, provided its baselines include D,
C, B, and newest and its first rollout is equally query-cadence-matched.

This report does not choose or implement that method. It does not revive the
semantic kernel, GATE, CCTS, group routing, or a learned selector. Work stops at
Gate-3A2 pending the requested joint review.

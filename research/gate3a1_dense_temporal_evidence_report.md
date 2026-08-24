# Gate-3A1 Dense Temporal Evidence Report

> **2026-08-24 time-contract correction:** The later primary-evidence audit in
> [`gate3a2_time_contract_audit.md`](gate3a2_time_contract_audit.md) shows that
> the local copy's 10 Hz metadata relabels an unreduced 20 Hz LIBERO sequence.
> Gate-3A1's index-domain numerical results are unchanged, but each source-age
> index is physically 0.05 s, `beta=0.03/index` is `0.6 s^-1`, and the original
> `0.015` per 20 Hz tick conversion in Section 12 was wrong. The frozen
> preregistration preserves the assumption made before this evidence was known.

**GATE DECISION:** `FAIL-SEMANTIC`

**DENSE TEMPORAL AGGREGATION:** passes against newest-only under the frozen
offline metric.

**CONTROL-SEMANTIC HYPOTHESIS H1:** rejected for dense teacher-forced offline
prediction error. Control-semantic weighting does not beat fairly tuned CogACT
cosine and is worse than the tuned age-exponential baseline.

**SCALAR CONTEXTUAL ORACLE HEADROOM:** large but target-informed and not yet
evidence that a selector is learnable or useful in closed loop.

**CLOSED-LOOP CLAIM:** none. No rollout or policy implementation was performed.

## 1. Confirmatory answer

The sparse control-semantic advantage did not replicate under dense sampling.
On 6,143 held-out actions from 41 episodes, scalar control-semantic similarity
has episode-weighted `L_sem = 0.62707`. Validation-tuned CogACT cosine has
`0.62581`. Their paired difference, semantic minus CogACT, is `+0.00126` with
95% episode-bootstrap CI `[-0.00132, 0.00376]`. Only five of ten task means favor
the semantic kernel. This fails the preregistered consistency rule.

The strongest validation-selected method is instead a simple newest-favoring
age exponential with `beta = 0.03` per stored dataset index. Its held-out
`L_sem = 0.60242`. Semantic weighting is worse by `+0.02465`, CI
`[0.01920, 0.03035]`, and loses on all ten task means. Therefore, the primary
result is not “semantic similarity approximately wins.” It is
`FAIL-SEMANTIC`.

Dense temporal aggregation itself survives. The tuned age exponential improves
over newest-only by `-0.13729`, CI `[-0.16042, -0.11574]`, with all ten task
means favoring aggregation. This is a verified offline phenomenon for one
frozen ACT checkpoint. It is not evidence of higher rollout success.

## 2. Frozen audit trail

The analysis was preregistered before any dense metric in
[`gate3a1_preregistered_protocol.md`](gate3a1_preregistered_protocol.md), commit
`d163f5a76a46c9368adbb8c2f56f09e248b3a81c`. The validation cache contained 41
episodes and 6,151 source queries. Validation-only selection produced an
immutable [selection lock](audit_outputs/gate3a1_validation_lock.json), which
was committed with the analysis code as
`07bfc40670f5f8ee692d210ea7dd7bff686ee4e6` before the test cache was generated.

The held-out analysis command verifies that the lock is tracked in the current
Git history. It also checks that the protocol and analysis-script SHA256 values
still match the lock. A repeated test evaluation produced byte-identical JSON
and CSV files. The test result was not used to alter a grid, distance, metric,
window, aggregation operator, or decision rule.

## 3. Cache completeness and provenance

The dense cache contains exactly 82 episode artifacts and 12,294 successful
source queries. Validation contributes 6,151 queries and test contributes
6,143. Each query stores the full `(100,7)` postprocessed ACT chunk. The cache
therefore contains 8,605,800 predicted scalars. Every expected source frame
appears once. No duplicate or missing source frame, non-finite output, shape
error, task mismatch, episode mismatch, or provenance mismatch was found.

The prediction files occupy 32,905,812 bytes. Their content-tree SHA256 is
`87e97a5711a7b51ea53da908774040d2b23ca57e9c29699bfd25f28ebe31908c`.
The cache is local-only at
`/home/thor/projects/one-clock/experiments/gate3a1_dense_temporal_cache` and is
not committed. The compact [cache manifest](audit_outputs/gate3a1_dense_cache_manifest.json)
contains every per-episode path, size, and SHA256.

The frozen checkpoint is
`/home/thor/projects/checkpoints/zeromidnight_act_libero_object/model.safetensors`,
SHA256 `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`.
The config SHA256 is
`a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d`.
The pinned LeRobot commit is
`f66e5128ecb2456e8c54a63d15404fa59c16aebc`. Inference used an NVIDIA Thor,
PyTorch 2.11.0+cu130, LeRobot 0.6.2, `policy.eval()`, inference mode, no AMP,
and deterministic algorithms.

The dataset metadata says 10 Hz, but the later time-contract audit establishes
that one stored source-age index is one 0.05 s LIBERO controller tick. All
Gate-3A1 methods used the same index axis, so this correction changes physical
labels rather than index-domain results.

## 4. Exact temporal contracts

For target time `t`, the analysis uses every source `q` satisfying
`q <= t < q + 100`. Candidates are ordered oldest to newest. The newest source
has temporal age zero.

The exact ACT baseline follows both the pinned LeRobot implementation and the
[original ACT code](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py).
For oldest-to-newest source index `i`, it assigns weight proportional to
`exp(-0.01 i)`. Positive `0.01` therefore favors older contributors. This is
not the same method as the separately tuned newest-favoring rule
`exp(-beta * temporal_age)`.

The CogACT baseline follows the
[released adaptive ensemble](https://github.com/microsoft/CogACT/blob/main/sim_cogact/adaptive_ensemble.py):
weights are proportional to `exp(alpha * cosine(candidate,newest))`. The
official released setting uses `alpha = 0.1`; Gate-3A1 also tunes `alpha` on
validation with the same nine-choice budget used for the semantic temperature.

## 5. Metric and selection

The primary target-level loss is

\[
L_{sem}=\frac{3L_{trans}+3L_{rot}+L_{grip}}{7}.
\]

Translation is training-scale-normalized Cartesian MSE. Rotation is squared
SO(3) geodesic distance normalized by the sum of the three audited rotation
variances. Gripper loss is sign error. The primary point estimate averages
within episode and then averages the 41 episode means.

Validation selected the following frozen parameters:

| Method | Frozen parameter | Validation `L_sem` |
|---|---:|---:|
| Newest-only | none | 0.82479 |
| Uniform | none | 0.70776 |
| Exact ACT | oldest-favoring `m=0.01` | 0.72073 |
| Tuned ACT-direction exponential | oldest-favoring `m=0.001` | 0.70910 |
| Tuned age exponential | newest-favoring `beta=0.03` | **0.68281** |
| Official CogACT | `alpha=0.1` | 0.70495 |
| Tuned CogACT | `alpha=0.3` | 0.70075 |
| Control-semantic similarity | `T=1.0` | 0.70067 |

The nearly identical validation scores for tuned CogACT and semantic weighting
did not justify treating semantic weighting as the expected winner. The
preregistered tie rule selected the age exponential as the strongest
non-oracle comparator.

## 6. Held-out non-oracle results

| Method | `L_sem` | Translation normalized MSE | Rotation normalized squared error | Gripper sign error | Mean source age, steps |
|---|---:|---:|---:|---:|---:|
| Newest-only | 0.73971 | 0.65856 | 0.96716 | 0.30083 | 0.00 |
| Uniform | 0.63381 | 0.53597 | 0.86686 | 0.22822 | 32.98 |
| Exact ACT, `m=0.01` | 0.64822 | 0.55869 | 0.87367 | 0.24044 | 37.57 |
| Tuned ACT-direction, `m=0.001` | 0.63521 | 0.53821 | 0.86748 | 0.22939 | 33.45 |
| Tuned newest-age exponential, `beta=0.03` | **0.60242** | **0.48556** | **0.85689** | **0.18960** | 20.50 |
| Official CogACT, `alpha=0.1` | 0.63086 | 0.53226 | 0.86631 | 0.22030 | 32.77 |
| Tuned CogACT, `alpha=0.3` | 0.62581 | 0.52604 | 0.86545 | 0.20619 | 32.34 |
| Control-semantic, `T=1.0` | 0.62707 | 0.52789 | 0.86285 | 0.21728 | 31.87 |

The complete component metrics and teacher-forced transition denominators are
in [`gate3a1_dense_metrics.json`](audit_outputs/gate3a1_dense_metrics.json).
Task-level values are in
[`gate3a1_dense_per_task.csv`](audit_outputs/gate3a1_dense_per_task.csv).

### Preregistered paired comparisons

Negative difference favors semantic weighting.

| Semantic minus comparator | Mean `L_sem` difference | 95% episode CI | 95% task-cluster CI | Tasks favoring semantic | Result |
|---|---:|---:|---:|---:|---|
| Tuned CogACT | +0.00126 | [-0.00132, +0.00376] | [-0.00167, +0.00386] | 5/10 | Tied; primary failure |
| Tuned newest-age exponential | +0.02465 | [+0.01920, +0.03035] | [+0.01608, +0.03313] | 0/10 | Semantic is worse |
| Exact ACT, `m=0.01` | -0.02115 | [-0.02692, -0.01543] | [-0.02836, -0.01337] | 9/10 | Semantic is better |
| Tuned ACT-direction, `m=0.001` | -0.00814 | [-0.01183, -0.00446] | [-0.01272, -0.00301] | 8/10 | Semantic is better |
| Official CogACT, `alpha=0.1` | -0.00379 | [-0.00675, -0.00093] | [-0.00722, -0.00041] | 8/10 | Semantic is better |
| Newest-only | -0.11264 | [-0.13844, -0.08863] | [-0.14957, -0.08469] | 10/10 | Temporal aggregation helps offline |

The relevant fair comparison is tuned CogACT, not only its official parameter.
Beating the untuned release setting does not rescue H1.

## 7. Metric tradeoffs and frozen diagnostics

Against tuned CogACT, semantic weighting reduces mean SO(3) geodesic error by
`0.000132` radians, CI `[-0.000238, -0.000034]`. It increases gripper sign error
by `0.01109`, CI `[0.00248, 0.02024]`, and increases raw 7-D MSE by `0.00149`,
CI `[0.00031, 0.00264]`. Translation error is unresolved. Thus the proposed
semantic kernel does not produce a uniform control-semantic improvement. Its
small rotation gain is offset by worse discrete gripper decisions.

When both kernels use the same SO(3)-projection and gripper-sign aggregation,
their primary difference remains unresolved: `+0.00048`, CI
`[-0.00512, 0.00577]`. The output operator therefore does not reveal a hidden
primary advantage.

The preregistered validation-quartile diagnostics do not supply a clean rescue.
Semantic weighting is descriptively better than tuned CogACT in the highest
rotation-disagreement quartile by `0.00689`, but worse in the two lowest
quartiles. It is worse when any candidate disagrees in gripper sign and only
slightly better when no sign disagreement exists. These are descriptive
interactions without clustered subgroup intervals. They cannot reverse the
primary decision or support a post-hoc rollout claim.

## 8. Scalar oracle headroom

The hard scalar source oracle reaches `L_sem = 0.33846`. The strongest
deployable baseline minus this oracle is `+0.26396`, CI
`[0.24165, 0.28714]`. Headroom is positive in all ten task means.

The conservative scalar convex-mixture oracle reaches `0.32018`. Its headroom
is `+0.28224`, CI `[0.25937, 0.30605]`. The algorithm starts from the hard
oracle and performs preregistered greedy convex-improvement rounds. It improved
4,786 of 6,143 targets. Some targets used all 32 rounds, so this is not a
certified global convex optimum.

The oracle result establishes target-informed offline diversity. It does not
show that the best candidate is identifiable from deployment-available inputs,
that demonstration action is the unique valid target, or that reducing this
loss increases success. It is sufficient to keep contextual selection as a
long-term scientific possibility. It is not sufficient to start a learned
selector now, especially given the overlap with Temporal Action Selection.

## 9. Scientific-critical assessment

The strongest design features are the episode-safe validation/test split, a
frozen primary outcome, matched candidate/query sets, equal CogACT/semantic
tuning budgets, exact implementation baselines, episode-level uncertainty,
task-cluster sensitivity, complete cache accounting, and a committed selection
lock before test exposure. The result is unlikely to be a frame-level
pseudoreplication artifact because the gate uses episode means and reports task
concentration.

The main validity limits remain substantial. Every state is teacher-forced.
Demonstration action can be one of several valid actions. The primary metric is
a constructed surrogate rather than success. Only one checkpoint, policy
family, benchmark suite, and successful-demonstration dataset are evaluated.
The dense cache comes from demonstrations whose metadata misstates their 20 Hz
physical cadence. Finally, oracle selection sees the target and can exploit
demonstration noise.

These limits bound both positive and negative claims. Gate-3A1 supports “the
semantic kernel does not improve the frozen dense offline comparison.” It does
not support “semantic geometry can never affect closed-loop behavior.” However,
the burden of proof now lies with an independent control mechanism, not a
semantic rollout undertaken to rescue a failed offline primary result.

## 10. Internal peer-review assessment

| Reviewer perspective | Evidence-bounded assessment | Required action |
|---|---|---|
| General ICRA manipulation | Dense temporal aggregation is credible offline, but there is no control result and no proposed method survives the strongest simple baseline. | Do not present Gate-3A1 as policy improvement. Establish a task-success link before method work. |
| ACT/CogACT expert | Exact ACT, released CogACT, tuned CogACT, and a tuned age prior are correctly separated. The age prior explains more than the semantic kernel. | Treat control-semantic similarity as a negative baseline. Preserve exact source-order and physical-time distinctions in rollout. |
| Skeptical novelty reviewer | A metric change that ties tuned CogACT and loses to exponential age weighting is not a contribution. Oracle headroom alone is already close to selector prior art. | Stop the semantic-paper direction unless a new independent mechanism and closed-loop result emerge. |

The peer-review workflow was used as an internal, author-authorized assessment.
It is not an editorial decision or a submitted review.

## 11. Claim architecture after Gate-3A1

The verified observation is that dense overlapping ACT predictions contain
offline temporal-aggregation benefit across this ten-task held-out cohort. The
failed hypothesis is that heterogeneous control semantics make a shared
semantic distance superior to tuned full-vector cosine. The surviving rival is
that temporal age itself supplies the dominant useful prior under this
teacher-forced distribution. The largest unknown is whether any of these
offline differences change closed-loop task success.

No paper claim is currently stable. A truthful future claim must connect a
deployment-available temporal rule to query-matched rollout success. Dense
offline error alone cannot support the final link.

## 12. Cheapest defensible next control gate, if separately approved

Do not implement or run this gate as part of Gate-3A1. The smallest useful
closed-loop test of the surviving observation is a three-condition,
query-matched experiment:

1. newest-only;
2. exact ACT temporal ensemble with its official source-order `m=0.01`;
3. validation-selected newest-favoring age exponential.

All methods should query at every 20 Hz environment tick and use the same
complete candidates. The later time-contract audit corrects the tuned decay to
`0.6 s^-1`: `beta=0.03` applies to each stored 0.05 s action index and therefore
remains `0.03` per controller tick. The exact ACT baseline retains its official
per-source-order coefficient and must be labeled as such.

Block on all ten tasks and identical official initial states. A practical first
gate is ten paired states per task and three methods, totaling 300 episodes.
Randomize method order within task-state blocks with a recorded seed. Primary
outcome is task success. Report actual queries, episode length, failures, and
paired episode and task-cluster intervals. Do not select tasks using Gate-3A1
test effects.

This experiment could establish the first direct offline-to-control link. It
would not by itself make tuned exponential weighting novel enough for ICRA.
If the age prior does not improve success over newest and exact ACT with a
task-consistent direction, stop temporal-selection work. If it does, the next
research question should concern a deployment-available mechanism that
outperforms this age prior, not revival of the failed semantic kernel.

## 13. Final decision

- `FAIL-SEMANTIC`: do not implement or rollout the control-semantic method now.
- Preserve tuned CogACT and tuned age exponential as mandatory baselines.
- Do not implement independent group routing, GATE, CCTS, or a learned selector.
- Retain scalar oracle headroom as a bounded unknown, not a method result.
- Request independent review before any closed-loop Gate-3A2.

The exact compact evidence is in
[`gate3a1_dense_metrics.json`](audit_outputs/gate3a1_dense_metrics.json),
[`gate3a1_dense_pairwise_comparisons.csv`](audit_outputs/gate3a1_dense_pairwise_comparisons.csv),
[`gate3a1_dense_per_task.csv`](audit_outputs/gate3a1_dense_per_task.csv), and
[`gate3a1_dense_oracle_headroom.json`](audit_outputs/gate3a1_dense_oracle_headroom.json).

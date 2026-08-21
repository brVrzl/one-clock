# Gate-3A1 Dense Temporal Evidence — Preregistered Protocol

Registration date: 2026-08-21

Registration commit parent: `ab95224cf6b049a6785a46a0d50dc9a40f598fc9`

Status: **FROZEN BEFORE DENSE VALIDATION OR TEST METRICS**

This protocol tests one question: does a scalar control-semantic similarity
kernel improve dense teacher-forced temporal aggregation over exact ACT
temporal ensembling and a fairly tuned CogACT cosine kernel? It does not test
independent group routing, GATE, CCTS, a learned selector, a horizon head,
policy training, or closed-loop control.

No dense validation or test metric was read before this document was frozen.
The test cache may be generated after registration, but test targets and test
method summaries remain inaccessible to model selection. The validation-selected
parameters will be written to a compact lock file and committed before the test
analysis command is run.

## 1. Question, hypotheses, and inference unit

The confirmatory hypothesis is:

> With identical dense temporal candidates, raw action aggregation, query
> count, and validation-selection budget, one shared control-semantic source
> distribution has lower held-out dimension-weighted control-semantic error
> than validation-tuned full-action CogACT cosine weighting.

The main rivals are exact ACT's fixed older-source-favoring exponential rule
and a validation-tuned newest-source-favoring age exponential. The secondary
question is whether a target-informed scalar oracle retains enough contextual
headroom to justify a later selector study.

The target action is the unit of measurement. The episode is the primary unit
of inference and weighting: each method's primary point estimate is the mean of
41 episode-level mean losses. Frame-weighted means are descriptive secondary
summaries. Task means and a task-cluster bootstrap assess task concentration.

## 2. Frozen cohort and split

The cohort is copied without alteration from
[`gate3a1_inventory.json`](audit_outputs/gate3a1_inventory.json), which derives
episode-safe membership from the portable Y-refresh split bundle.

| Split | Episodes | Dataset steps / ACT queries | Use |
|---|---:|---:|---|
| Validation | 41 | 6,151 | Select every non-fixed hyperparameter and validation-derived diagnostic cutpoint |
| Test | 41 | 6,143 | One held-out evaluation after selection is locked |
| Total | 82 | 12,294 | Dense cache scope |

Tasks 0–8 contribute four episodes to each split; task 9 contributes five.
The exact episode IDs are those already recorded in the inventory JSON. No
episode, frame, or task may be added or removed because of its result. A cache
failure is repaired and documented; it is not an exclusion rule.

The dataset clock is 10 Hz. One dataset step is 0.1 s. All ages will be stored
and reported in both dataset steps and seconds. These ages are not 20 Hz
deployment ticks.

## 3. Frozen policy, data, and preprocessing provenance

### Policy

- Checkpoint directory:
  `/home/thor/projects/checkpoints/zeromidnight_act_libero_object`
- Model SHA256:
  `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`
- Config SHA256:
  `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d`
- Policy-preprocessor JSON SHA256:
  `e7e3815a9e23eabe88e3dc5697cbccf8c59e61b59cf916d947dd673123426450`
- Policy-postprocessor JSON SHA256:
  `c27cf6f42b42352f9b8f9c40da155fd4459e0ee9b85b9f23072941eb52b3ffb5`
- Normalizer/unnormalizer tensor SHA256:
  `3cb90679b116d22c960772f75e567c32b51778df2ca065cc4784bd6cd593e941`
- ACT chunk length: 100; action dimension: 7; observation steps: 1.
- Saved `temporal_ensemble_coeff`: null. Dense tooling calls
  `predict_action_chunk`, never `select_action`, and performs every ensemble
  offline from the saved full chunks.

### LeRobot and public rule references

- Pinned local LeRobot checkout:
  `/home/thor/projects/embodied_lab/third_party/lerobot`, commit
  `f66e5128ecb2456e8c54a63d15404fa59c16aebc`.
- The pinned `ACTTemporalEnsembler` orders contributors oldest to newest and
  assigns `exp(-m*i)` to source-order index `i`, so positive `m` favors older
  sources. Its documented original coefficient is `m=0.01`.
- The independently inspected original ACT repository was commit
  `742c753c0d4a5d87076c8f69e5628c79a8cc5488`; its evaluation loop implements
  the same oldest-to-newest `exp(-0.01*i)` rule
  ([official source](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)).
- The independently inspected CogACT repository was commit
  `b174a1b86deedfab4d198d935207e7bb0527994e`; its released rule is
  `exp(alpha*cosine(candidate,newest))`, and its deployment path uses
  `alpha=0.1`
  ([official source](https://github.com/microsoft/CogACT/blob/main/sim_cogact/adaptive_ensemble.py)).

### Dataset

- Root: `/home/thor/datasets/libero_object_25_08_23_lerobotv2.1`
- Recorded repository/revision:
  `DorayakiLin/libero_object_25_08_23_lerobotv2.1@cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a`
- Local content-tree SHA256:
  `2c7b87d23936dcd9d511c77234907f99e2da8ac4d23b68bb7b23af9b71297608`
- Dataset metadata: v2.1, 454 episodes, 66,984 frames, 10 tasks, 10 Hz.

### Observation and action contracts

For each stored demonstration frame, the audit reconstructs the exact runner
observation from two RGB images and the 8-D state. It applies
`preprocess_observation`, the official LIBERO environment processor, and the
saved ACT policy preprocessor. A round-trip check requires recovered state to
match the stored state within `2e-4` absolute error. The saved policy
postprocessor and environment postprocessor convert each full chunk back to
the seven-dimensional environment action.

The action is translation `[0:3]`, rotation axis-angle `[3:6]`, and gripper
command `[6]`. The LIBERO controller consumes gripper sign. Zero is assigned
positive sign consistently for targets, candidates, and predictions.

### Determinism and environment

The generator uses seed `20260821`, `policy.eval()`, `torch.inference_mode()`,
no AMP, deterministic algorithms, deterministic cuDNN, and disabled cuDNN
benchmarking. ACT inference uses the zero latent because no training action is
passed. The registered runtime is Python 3.12.3, LeRobot 0.6.2, PyTorch
2.11.0+cu130, CUDA 13.0, NumPy 2.2.6, pandas 2.3.3, SciPy 1.18.0, PyArrow
25.0.1, PyAV 15.1.0, and an NVIDIA Thor device. The final manifest records
the observed versions again.

## 4. Dense cache contract

The local-only cache root is
`experiments/gate3a1_dense_temporal_cache/`. One compressed artifact is written
per episode with an atomic same-filesystem rename. A completed artifact is
verified and skipped on resume. A partial or corrupt artifact is quarantined
with an explicit diagnostic rather than appended silently.

Every eligible frame is queried exactly once during a successful generation
pass. Batching policy inputs is allowed, but it does not duplicate or omit
source observations. Each episode artifact stores:

- episode ID, task ID, split, dataset frame index;
- source time in dataset steps and seconds;
- full postprocessed predicted chunk `(frames,100,7)`;
- chunk length, action dimension, dataset frequency;
- checkpoint/config/processor hashes and software provenance.

The resumable working manifest and the compact committed manifest record
expected frames, completed frames, cache filename, byte size, SHA256, and
status for every episode. Completion requires all 12,294 source frames exactly
once, no duplicate source frame, no missing source frame, exact chunk shape,
finite values, correct episode/task mapping, and matching provenance hashes.
The prediction arrays remain outside Git.

## 5. Temporal candidate construction

For target dataset action time `t`, a source query `q` contributes exactly when

\[
q \le t < q+100.
\]

Its temporal age is `k=t-q` dataset steps, or `0.1k` seconds. Candidates are
stored and processed in ascending source time, equivalently oldest to newest
for a fixed target. The newest candidate has age zero. The primary candidate
window contains every valid source, at most 100 candidates. Window length is
not tuned.

Fixed descriptive window sensitivities use maximum candidate counts
`8,16,32,64,100`, equivalently maximum ages `7,15,31,63,99` dataset steps.
They do not alter the primary decision or any selected hyperparameter.

## 6. Frozen metrics

The audited training-action standard deviations are

```text
[0.2681190073, 0.4384443760, 0.4475117326,
 0.0244482197, 0.0493620895, 0.0421034954,
 0.9974462986]
```

For prediction `a` and demonstration target `a*`:

\[
L_{trans}=\frac{1}{3}\sum_{j=0}^{2}
\left(\frac{a_j-a_j^*}{\sigma_j}\right)^2.
\]

Let `R(a)` convert the axis-angle increment to SO(3), and let
`theta=acos(clip((tr(R(a)^T R(a*))-1)/2,-1,1))`. Then

\[
L_{rot}=\frac{\theta^2}{\sum_{j=3}^{5}\sigma_j^2},\qquad
L_{grip}=1[sign(a_6)\ne sign(a_6^*)].
\]

The primary target-level loss is

\[
L_{sem}=\frac{3L_{trans}+3L_{rot}+L_{grip}}{7}.
\]

The primary reported method estimate averages target losses within episode and
then averages the 41 episode means.

Secondary metrics are frame-weighted `L_sem`; normalized translation error;
raw translation L2 in action units; rotation geodesic radians; normalized
squared rotation error; gripper sign error; raw environment-unit 7-D MSE;
`(L_trans+L_rot+L_grip)/3`; and
`0.5*(0.5*(L_trans+L_rot)+L_grip)`.

Offline transition diagnostics exclude the first episode frame. They use the
previous demonstrated gripper sign as the assumed previous command. False
transition rate is false predicted transitions divided by target
non-transition opportunities. Missed transition rate is missed transitions
divided by target transition opportunities. Counts and denominators accompany
the rates. These are teacher-forced diagnostics, not rollout event rates.

## 7. Frozen non-oracle methods

Every method uses the same full candidate set and exactly the same 12,294 ACT
queries. Unless explicitly marked as a secondary operator ablation, output is
the weighted linear mean of the postprocessed seven-dimensional action
candidates. No method accesses the demonstration target when constructing a
deployable prediction.

### B0 — newest

Use the age-zero candidate.

### B1 — uniform

Assign equal weight to every valid candidate.

### B2 — exact upstream ACT temporal ensemble

Order candidates oldest to newest. For source-order index `i=0,...,n-1`, use

\[
w_i\propto\exp(-0.01i).
\]

This exactly reproduces the pinned LeRobot/original ACT convention, including
its greater weight on older candidates. `B2-tuned` applies the same
oldest-to-newest convention and selects `m` on validation from

```text
[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0].
```

### B3 — newest-favoring age exponential

For temporal age `k`, use `w_k proportional to exp(-beta*k)`. Select `beta` on
validation from the same seven-value grid used by B2-tuned. B2 and B3 are
separate because their weighting directions differ.

### B4 — official released CogACT cosine

Let the age-zero action be `e_0`. For each candidate `e_k`, compute full-vector
cosine with denominator epsilon `1e-7`, then use

\[
w_k\propto\exp(0.1\,cos(e_k,e_0)).
\]

### B5 — validation-tuned CogACT cosine

Select `alpha` on validation from

```text
[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0].
```

The rule, candidates, epsilon, and raw aggregation otherwise equal B4.

### B6 — control-semantic scalar similarity

For candidate `e_k` and newest `e_0`, define `D_sem` by the same translation,
rotation, gripper, and 3:3:1 construction used for `L_sem`. Use one shared
weight vector for the complete action:

\[
w_k\propto\exp(-D_{sem}(e_k,e_0)/T).
\]

Select `T` on validation from

```text
[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0].
```

B5 and B6 therefore receive nine validation candidates each. Selection
minimizes the episode-weighted validation `L_sem`. Exact numeric ties within
`1e-12` choose the first value in the displayed order. Test data never break a
tie or alter a grid.

### Matched semantic-aggregation ablation

As a secondary factorial comparison, B5 and B6 are also evaluated with the
same semantic output operator: Euclidean translation mean, projected chordal
SO(3) mean, and weighted gripper-sign vote. Each method selects its parameter
from the same nine-value validation grid under that operator. This comparison
isolates the distance/kernel effect. It cannot replace the raw-aggregation
primary comparison.

## 8. Frozen diagnostic strata

All strata are descriptive and cannot rescue a failed primary gate.

- Candidate count: `1`, `2–4`, `5–8`, `9–16`, `17–32`, `33–64`, `65–100`.
- Method-weighted mean source age: `0`, `(0,1]`, `(1,3]`, `(3,7]`, `(7,15]`,
  `(15,31]`, `(31,63]`, `(63,99]` dataset steps; seconds are also reported.
- Gripper disagreement: all candidates share the newest sign versus at least
  one disagreement; the disagreeing-candidate fraction is also retained.
- Translation disagreement: maximum normalized translation discrepancy from
  newest.
- Rotation disagreement: maximum SO(3) geodesic angle from newest.
- Overall candidate disagreement: mean `D_sem(candidate,newest)`.

Translation, rotation, and overall disagreement are divided by quartile
cutpoints computed once on validation and then applied unchanged to test.
Normalized episode time deciles may be shown as a diagnostic only; they are
not phases and do not enter the gate decision.

## 9. Frozen oracles and contextual headroom

O1 is the hard scalar source oracle: choose the one complete candidate with
minimum exact target `L_sem`. Ties choose the newest candidate.

O2 is a conservative scalar convex-mixture oracle with one shared weight
vector over the complete action. It starts from O1 and performs 32 deterministic
greedy convex-improvement rounds. Each round searches every candidate and step
size in

```text
[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
```

for the mixture `(1-lambda)*current + lambda*candidate`, choosing the largest
exact `L_sem` reduction. It stops if improvement is below `1e-12`. The result
is a valid shared convex combination and cannot be worse than O1, but it is not
a certified global optimum. This limitation will be stated wherever O2 is
reported.

No group oracle enters the Gate-3A1 decision. An optional
translation/rotation/gripper independent hard oracle may be reported only as
an unattainable, potentially inconsistent diagnostic and must not motivate a
method in this phase.

Contextual headroom is reported as the strongest validation-selected
non-oracle test `L_sem` minus O1 and O2, with paired episode-bootstrap
intervals and per-task effects. A later learned selector is not justified when
this headroom is small, unstable, or concentrated in a few tasks/episodes.

## 10. Frozen statistical analysis

- Bootstrap draws: 10,000.
- Episode-bootstrap seed: `20260821`.
- Task-cluster-bootstrap seed: `20260822`.
- Primary paired difference: mean of the 41 episode-level paired differences.
- Episode 95% CI: percentile interval from resampling 41 episodes with
  replacement.
- Task-cluster 95% CI: percentile interval from resampling the ten task-level
  paired means with replacement.
- Report every task-level paired mean and leave-one-task-out primary mean.
- Negative method-minus-baseline difference favors the named method.

Confirmatory comparisons are B6 minus B2, B6 minus B3, and B6 minus B5. B6
minus B2-tuned, B6 minus B4, B6 minus B1, and the strongest
validation-selected method among B1, B2, B2-tuned, B3, B4, B5, and B6 minus B0
are supporting comparisons. No frame-level CI is interpreted as independent
evidence.

## 11. Frozen gate decision

“Consistent advantage” requires all of the following: the episode-bootstrap
CI upper bound is below zero, at least seven of ten task-level mean differences
are negative, and every leave-one-task-out primary mean remains negative.

- `FAIL-TEMPORAL`: the method selected as best on validation among B1, B2,
  B2-tuned, B3, B4, B5, and B6 does not show a consistent held-out advantage
  over B0 newest.
- `PASS-SEMANTIC`: B6 shows a consistent held-out advantage over B5 tuned
  CogACT under raw aggregation, but the stronger condition below is not met.
- `STRONG-PASS`: B6 meets `PASS-SEMANTIC` and also shows a consistent advantage
  over B2 exact ACT, B2-tuned, and B3 tuned newest-favoring age exponential.
- `PARTIAL`: B6 meets the consistent-advantage rule versus B5, but is unresolved
  against B2 or B3. This label takes precedence over `PASS-SEMANTIC` when the
  ACT/age comparisons are tied.
- `FAIL-SEMANTIC`: temporal aggregation passes, but B6 does not show a
  consistent advantage over B5. Predeclared strata may identify a future
  diagnostic, but they do not change this primary status.

`PASS-SEMANTIC` without `STRONG-PASS` is retained only for completeness; under
the stated hierarchy it will normally be reported as `PARTIAL` when a strong
age baseline remains tied. No fixed percentage-effect threshold is invented.
Absolute effect, relative effect, task consistency, and uncertainty will be
reported so a later rollout can be sized from observed variance.

## 12. Blind analysis sequence and amendments

1. Commit this protocol.
2. Implement deterministic cache and analysis tooling without test summaries.
3. Generate and validate the validation cache.
4. Run validation-only selection; write a lock file containing selected
   parameters, validation-derived quartiles, code SHA256, and cache hashes.
5. Commit the lock file before running the test evaluation command.
6. Generate/validate test cache if not already generated.
7. Run the test analysis exactly once from the committed lock.
8. Write compact outputs, the evidence report, fact-sheet update, and
   local-only cache manifest.

If a software defect makes the registered computation invalid, analysis stops.
A dated amendment will identify the defect, affected rule, evidence that the
change is necessary, and whether any test metric was already exposed. Both
pre-amendment and amended outputs will be preserved. Scientific dislike of an
outcome is not an amendment reason.

## 13. Interpretation boundary

Gate-3A1 is teacher-forced and offline. It can verify dense prediction
diversity, an offline kernel advantage, or oracle headroom for this frozen ACT
checkpoint and LIBERO Object cohort. It cannot establish better closed-loop
control, causality, policy generalization, VLA transfer, real-robot benefit, or
an ICRA-ready contribution. Any later rollout requires a separate, query-matched
experimental design and approval.

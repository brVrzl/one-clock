# Gate-3A2 temporal aggregation control-link test — preregistered protocol

Registration date: 2026-08-24

Registration parent: `1ce9bf0eb1443abb7452086ac85a7c4ed0ea5752`

Status: **FROZEN BEFORE ANY OFFICIAL GATE-3A2 OUTCOME IS GENERATED OR READ**

This gate asks one question: does the dense teacher-forced temporal-aggregation
ranking from Gate-3A1 predict query-cadence-matched closed-loop success on the
same frozen ACT/LIBERO system? It does not evaluate a novel method. It does not
test GATE, CCTS, independent group routing, a selector, a horizon head, or
policy training.

An implementation or environment bug may be corrected only through a dated
amendment that identifies affected episodes. Outcomes from invalid episodes
remain quarantined and are never silently replaced. The rules below are not
changed in response to success results.

## 1. Gate audit and hypothesis

Scientific-critical-thinking and experimental-design review found no fatal
comparison confound after resolving the time-contract mismatch in
[`gate3a2_time_contract_audit.md`](gate3a2_time_contract_audit.md):

- Every condition receives the same information type: one current observation
  and one newly inferred full ACT chunk per surviving 20 Hz controller step.
- Every condition retains all still-valid overlapping predictions for the
  current action. Only scalar source weighting differs.
- Conditions share policy, checkpoint, preprocessing, environment,
  controller, initial state, seed, maximum steps, and query cadence.
- Closed-loop trajectories and therefore total steps/queries may differ. The
  precise fairness claim is **same policy-query cadence per surviving step**,
  not equal total compute.
- Newest-favoring exponential weighting is prior art and is evaluated only as
  a scientific control link.
- Ten states per task has limited resolution for small effects, but the paired
  complete-block design can detect a large, task-stable first-gate effect. The
  prescribed 400 episodes are the smallest complete design retaining all ten
  tasks and all four methods.

Confirmatory hypothesis H1 is:

> Gate-3A1's validation-selected newest-favoring age exponential improves
> paired closed-loop binary success over exact ACT temporal ensembling, is
> directionally favorable to newest-only, and does so without additional
> policy queries per surviving control step.

H1 is about ranking-to-control relevance, not novelty or mechanism.

## 2. Frozen system and provenance

- Branch parent: `1ce9bf0eb1443abb7452086ac85a7c4ed0ea5752`.
- Checkpoint:
  `/home/thor/projects/checkpoints/zeromidnight_act_libero_object/model.safetensors`.
- Required model SHA256:
  `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`.
- Required config SHA256:
  `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d`.
- Required pinned LeRobot commit:
  `f66e5128ecb2456e8c54a63d15404fa59c16aebc`.
- ACT chunk length: 100; action dimension: 7; deterministic inference; no
  policy training; saved policy temporal ensembling disabled.
- LIBERO Object tasks 0–9; relative controller; two 256×256 RGB observations;
  `pixels_agent_pos`; hard reset; official initial states; 280-action maximum.
- Controller frequency: 20 Hz. One action/chunk index is 0.05 s. A 100-action
  chunk spans 5.0 s.

The runner aborts before an episode if the checkpoint, config, LeRobot commit,
chunk shape, action dimension, or temporal-ensemble setting differs. It records
runtime device and library versions in the manifest.

## 3. Frozen task-state cohort and randomization

All ten official LIBERO Object tasks have 50 available initial states, IDs
0–49. A NumPy `default_rng` seeded with `20260824` sampled ten IDs without
replacement; IDs were sorted only for legible execution order:

```text
[0, 7, 11, 13, 25, 30, 36, 41, 42, 43]
```

Selection did not inspect any previous state-level outcome. The same state IDs
are used for every task. For block `(task_id,state_id)`, all four methods use
the same episode seed:

```text
310000 + 100 * task_id + state_id
```

Within each of the 100 task-state blocks, method order is independently drawn
from a continuing NumPy `default_rng(20260825)` permutation stream. The exact
400-run order is frozen in
[`gate3a2_run_schedule.json`](audit_outputs/gate3a2_run_schedule.json).
Method labels are not reordered after outcomes. Tasks and states are traversed
in ascending order, while treatment order within block is randomized.

## 4. Frozen conditions

At controller step `t`, the runner first queries ACT once at observation `o_t`
and saves the complete 100×7 chunk. Source `q` is valid for the current action
when `q <= t < q+100`. Candidates are ordered oldest source to newest source.
No target or future observation is available to any rule.

All methods use the raw weighted arithmetic mean over the seven-dimensional
postprocessed LIBERO actions.

### A — `newest`

Execute the age-zero candidate from the current query.

### B — `exact_act_m001`

Reproduce pinned upstream ACT source-order semantics:

\[
w_i \propto \exp(-0.01i),
\]

where `i=0` is the oldest valid source. This favors older contributors.

### C — `cogact_a03`

Use the released full-action CogACT cosine rule with Gate-3A1
validation-selected `alpha=0.3`:

\[
w_q \propto \exp(0.3\cos(E_{t,q}, E_{t,t})).
\]

The newest full action is the reference; the cosine denominator uses the
released `+1e-7` numerical constant. Alpha is not retuned on rollouts.

### D — `newest_age_exp_b003`

Use Gate-3A1's validation-selected decay in the correct physical index domain:

\[
w_q \propto \exp(-0.03(t-q)).
\]

The time audit establishes that one stored source age equals one 20 Hz rollout
tick, so `beta_tick=0.03` and continuous decay is `0.6 s^-1`. The originally
proposed `0.015/tick` conversion is rejected as a factor-of-two mismatch.

## 5. Execution, logging, and resume contract

Every episode begins from its frozen official state and seed, resets the policy
and temporal cache, and runs until success or 280 actions. The runner asserts
one and only one ACT query per executed environment step. It records actual
steps, queries, inference time, episode wall time, candidate counts, effective
source ages in ticks and seconds, executed actions, gripper transitions, and
action-discontinuity diagnostics.

Each completed episode is written as one gzip JSON artifact using a
same-filesystem temporary file and atomic rename. The compact committed
manifest records task, state, method, status, success, steps, queries, local
path, log SHA256, and provenance. Resume validates a complete artifact and
skips it; a malformed or provenance-mismatched artifact aborts rather than
being overwritten. No historical rollout artifact is modified.

## 6. Outcomes and estimands

The primary outcome is binary task success. The primary estimand for method
`X` versus `Y` is the mean within-block difference
`success_X - success_Y` over the 100 frozen task-state blocks. Because every
task has ten blocks, this is also the macro mean of the ten task differences.

The four preregistered comparisons, in order, are:

1. D minus A;
2. D minus B;
3. D minus C;
4. C minus B.

Secondary outcomes are episode length, actual policy queries, wall-clock ACT
inference, mean effective source age, raw translation-action discontinuity,
SO(3) geodesic rotation-action discontinuity, gripper transitions, and
finite-difference action acceleration/jerk diagnostics. They cannot override
the binary-success decision and are not used to invent post-hoc mechanisms.

## 7. Frozen statistics

Each comparison reports method success counts and rates, absolute paired
difference, all ten task differences, and discordant counts. Uncertainty uses:

- 20,000 paired task-state bootstrap draws resampling the 100 blocks;
- 20,000 task-cluster bootstrap draws resampling ten whole tasks, retaining all
  ten states inside each selected task;
- exact two-sided McNemar/binomial diagnostic on discordant blocks;
- ten leave-one-task-out paired differences as a concentration diagnostic.

Bootstrap RNG seeds are `20260826 + comparison_index` for paired-state draws
and `20261826 + comparison_index` for task-cluster draws, with comparison order
as listed above. Percentile 2.5% and 97.5% bounds form the reported 95% CIs.
Episodes and task-state blocks—not target frames or controller steps—are the
inferential units. No multiple-comparison-corrected discovery claim is made;
the four comparisons answer the fixed gate rather than a broad method search.

For decision wording, a comparison is **stable positive** only when both
bootstrap CI lower bounds exceed zero and every leave-one-task-out point
estimate exceeds zero. It is **stable negative** only when both upper bounds
are below zero and every leave-one-task-out point estimate is below zero. This
is a concentration rule, not a minimum percentage-point effect threshold.

## 8. Frozen gate decision

- **STRONG-CONTROL-LINK:** D−B is stable positive, D−A has positive point
  estimate, every method has one query per surviving step, and D−C is stable
  positive.
- **CONTROL-LINK-POSITIVE:** D−B is stable positive, D−A has positive point
  estimate, every method has one query per surviving step, but D−C is not
  stable positive.
- **CONTROL-LINK-NEGATIVE:** D−B or D−A is stable negative.
- **CONTROL-LINK-NULL:** every other complete result, including numerical
  advantages whose interval or task concentration is unresolved.

No arbitrary success-percentage threshold is introduced. Effect magnitude,
discordant counts, and interval width determine whether any later sample-size
extension is justified.

If positive, the only authorized conclusion is that temporal-source weighting
is a real closed-loop control lever on this system and Gate-3A1 ranking carries
deployment-relevant information. Newest-favoring decay is not claimed as
novel. If null or negative, demonstration `L_sem` and scalar-oracle mining stop
as the main method-development signal.


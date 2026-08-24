# RTX 5080 cross-generation offline composition protocol

Freeze date: 2026-08-24 (Asia/Shanghai). This is a post-Gate-3A2 exploratory
mechanism audit frozen before computing the new composition losses. It is not a
historical preregistration. No Thor Gate-3B outcome was inspected in defining
this analysis, and no Gate-3B result will be used to change it.

## Question and scope

The sole question is whether the frozen Gate-3A1 teacher-forced action metric
detects a penalty when the six arm dimensions and gripper sign are taken from
different temporal source generations. This is a matched offline audit, not a
method, selector, temporal-weight comparison, or causal test of closed-loop
coherence.

The analysis uses all 82 episodes in the committed Gate-3A1 validation and test
cohorts (41 each). There is no fitting, tuning, threshold selection, age sweep,
or method selection. Within every episode of length `T`, eligible target indices
are exactly `t = 20, ..., T-1`, giving 10,654 targets in total. The dataset's
nominal 10 Hz field relabels unreduced 20 Hz source content: `d=20` stored action
indices is retained exactly and corresponds to 20 controller ticks, or 1.0 s.

## Fixed 2x2 construction

For each eligible target `t`, the cache stores a chunk queried at source `q`.
Define

```text
F_t = E[t, t]    = predicted_chunks[t, 0]
O_t = E[t, t-20] = predicted_chunks[t-20, 20]

FF = [F_t[0:6], F_t[6]]
OO = [O_t[0:6], O_t[6]]
FO = [F_t[0:6], O_t[6]]
OF = [O_t[0:6], F_t[6]]
```

No source interpolation, target-informed choice, or alternative age is allowed.
Zero gripper command is assigned positive sign, matching Gate-3A1.

## Frozen losses and aggregation

The demonstrated action at `t` is the target. For a prediction `a`, use the
unchanged Gate-3A1 component losses:

- `L_trans`: mean squared Cartesian translation error after division by the
  frozen standard deviations `[0.2681190073490143, 0.4384443759918213,
  0.4475117325782776]`;
- `L_rot`: squared SO(3) geodesic angle divided by the sum of the squared three
  frozen rotation standard deviations `[0.024448219686746597,
  0.04936208948493004, 0.042103495448827744]`;
- `L_grip`: gripper sign mismatch indicator.

The primary loss is

```text
L_sem = (3 L_trans + 3 L_rot + L_grip) / 7.
```

Continuous gripper magnitude MSE is excluded. Raw SO(3) geodesic radians are
reported as an additional component-scale description, but are not substituted
into `L_sem` and do not define another combined metric.

Target losses are averaged within episode first. The primary condition values
are the unweighted means of the 82 episode means. Task summaries first average
episode means within each task. All FF, OO, FO, and OF losses and their
translation, normalized-rotation, raw-geodesic, and gripper components are
reported; no condition may be omitted based on its result.

## Primary contrast and pre-result algebra check

For each episode and for the overall episode-weighted estimand, define

```text
C_offline = 0.5 (L_FO + L_OF) - 0.5 (L_FF + L_OO).
```

Positive means mixed actions have higher teacher-forced error; zero means the
metric detects no penalty; negative means mixed actions look better.

Before reading any composition loss, note an exact property of the frozen
design: `L_sem` is an additive sum of an arm-only term and a gripper-only term.
Therefore the specified symmetric contrast cancels target-by-target:

```text
L_FO + L_OF = L_FF + L_OO.
```

The empirical computation is consequently an implementation/provenance audit
of whether the frozen metric contains any cross-component term. A nonzero
contrast beyond absolute tolerance `1e-12` is a failure to reproduce the stated
construction or metric, not an effect to interpret. This algebraic limitation
does not imply any particular closed-loop Gate-3B outcome.

## Frozen inference

- Inference unit: episode (`n=82`), never target/frame.
- Primary point estimate: mean of the 82 episode contrasts.
- Paired episode bootstrap: 20,000 draws, resample episode contrasts with
  replacement, percentile 95% interval, seed `20260826`.
- Task-cluster bootstrap: compute ten task-level episode means, resample ten
  task labels with replacement for 20,000 draws, percentile 95% interval, seed
  `20260827`. This estimates the macro-task contrast.
- Report all ten task contrasts.
- Leave-one-task-out: for each omitted task, report the unweighted mean of the
  remaining nine task contrasts.
- Sign classification uses the frozen numerical tolerance: positive if
  `C > 1e-12`, negative if `C < -1e-12`, otherwise zero.

No frame-level test, p-value search, outlier removal, multiple alternative
contrasts, or equivalence margin will be added after inspection.

## Frozen descriptive disagreement strata

Fresh-old disagreement is computed without demonstration targets. Translation
uses normalized MSE and rotation uses SO(3) geodesic radians. The pooled 10,654
eligible targets fixed the following quartiles before composition losses were
computed:

| Signal | Q25 | Q50 | Q75 |
|---|---:|---:|---:|
| translation normalized disagreement | 0.09562705994409045 | 0.2215019542367106 | 0.4240999982124295 |
| rotation geodesic disagreement (rad) | 0.01704259893576156 | 0.026375850435693467 | 0.03981205950930166 |

Bins are `x <= Q25`, `Q25 < x <= Q50`, `Q50 < x <= Q75`, and `x > Q75`.
Gripper is stratified only as same sign versus different sign; 3,098 of 10,654
targets have different signs. Stratum counts, four condition losses, and the
fixed contrast are descriptive. No subgroup is selected and no stratum-specific
significance claim will be made.

## Outputs and stopping rule

One deterministic analysis script will write:

- `research/audit_outputs/fast5080_cross_generation_metrics.json`;
- `research/audit_outputs/fast5080_cross_generation_per_task.csv`;
- `research/audit_outputs/fast5080_cross_generation_contrast.csv`;
- `research/fast5080_cross_generation_offline_report.md`.

The contrast CSV contains one row per episode so the bootstrap is auditable;
the per-task CSV and JSON contain figure-ready FF/OO/FO/OF, component, and task
contrast summaries. The report will state the additive-metric limitation and
will not infer closed-loop behavior or inspect Gate-3B results. After these
fixed outputs are generated, validated, committed, and pushed, the RTX track
stops.

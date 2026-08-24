# Gate-3B directional post-hoc analysis

**Status:** completed bounded post-hoc characterization, 2026-08-24.

**Scientific boundary:** Gate-3B did not confirm its preregistered generic
coherence hypothesis. The directional analyses below were specified only after
the complete four-cell outcome was known. They are secondary, post-hoc results
that motivate the untouched-state Gate-3C experiment. They are not substitutes
for Gate-3B's preregistered primary result.

## Source and validation

The analysis reads the final Gate-3B rollout manifest and success summary
directly from Thor commit `2817411a4210b8611dc8dae5d32ec99fc6b94cf3`.
The manifest contains 400 unique completed episodes: four conditions in each of
100 paired task-state blocks spanning ten tasks and ten states per task. All
four outcomes were present in every block. The condition-level success totals
reproduced the final report exactly:

| Condition | Arm source | Gripper source | Successes | Rate |
|---|---|---|---:|---:|
| `FF` | fresh | fresh | 44/100 | .44 |
| `OO` | old20 | old20 | 40/100 | .40 |
| `FO` | fresh | old20 | 62/100 | .62 |
| `OF` | old20 | fresh | 17/100 | .17 |

The preregistered coherence contrast was only `+.025`, with paired-state 95%
CI `[-.030,+.085]` and task-cluster CI `[-.005,+.055]`. Therefore, Gate-3B
does not establish that cross-generation composition is generally harmful.

## Post-hoc estimands and inference

For paired block (i), let (S_{i,c}\in\{0,1\}) be success under condition
(c\). The arm-freshness main effect is

\[
M_{\mathrm{arm}}=\frac{1}{N}\sum_i
\left[\frac{S_{i,FF}+S_{i,FO}}{2}
-\frac{S_{i,OO}+S_{i,OF}}{2}\right].
\]

The gripper-old-source main effect is

\[
M_{\mathrm{grip-old}}=\frac{1}{N}\sum_i
\left[\frac{S_{i,OO}+S_{i,FO}}{2}
-\frac{S_{i,FF}+S_{i,OF}}{2}\right].
\]

The three requested cell contrasts are (FO-FF), (FO-OO), and (FO-OF).
All five effects were computed within paired task-state blocks. Uncertainty was
characterized with 20,000 paired-block bootstrap draws and 20,000 task-cluster
bootstrap draws. The fixed seeds were `20260830` and `20261830`, respectively.
Intervals are unadjusted exploratory percentile intervals. Leave-one-task-out
estimates retain the paired-block construction.

Exact block diagnostics were also computed. Pairwise cell comparisons use the
two-sided exact McNemar/binomial diagnostic on discordant paired blocks. The two
factorial effects use an exact two-sided sign-flip test conditional on each
block's absolute contrast. The raw diagnostic probabilities and Holm-adjusted
values across the five reported contrasts are supplied for transparency. They
do not convert the post-hoc analyses into confirmatory tests.

## Directional results

| Post-hoc contrast | Estimate | Paired-block 95% CI | Task-cluster 95% CI | Task signs +/0/- | LOTO range | Exact diagnostic, raw / Holm |
|---|---:|---:|---:|---:|---:|---:|
| Fresh-arm main effect | +.245 | [.155, .330] | [.170, .330] | 10/0/0 | [.211, .267] | 5.11e-7 / 1.53e-6 |
| Old-gripper main effect | +.205 | [.140, .275] | [.105, .305] | 8/1/1 | [.172, .233] | 2.91e-8 / 1.16e-7 |
| `FO-FF` | +.180 | [.100, .270] | [.090, .280] | 8/2/0 | [.144, .200] | 1.21e-4 / 2.26e-4 |
| `FO-OO` | +.220 | [.120, .320] | [.150, .300] | 10/0/0 | [.189, .233] | 1.13e-4 / 2.26e-4 |
| `FO-OF` | +.450 | [.340, .560] | [.300, .590] | 9/1/0 | [.411, .500] | 1.97e-11 / 9.84e-11 |

The marginal averages reproduce the requested decomposition. Fresh-arm cells
average `.530`, compared with `.285` for old-arm cells. Old-gripper cells
average `.510`, compared with `.305` for fresh-gripper cells. Thus the observed
main effects are `+.245` for a fresh arm source and `+.205` for an old gripper
source. These estimates describe the completed Gate-3B sample. Their strong
directionality does not repair the unresolved preregistered coherence result.

At the paired-block level, the arm contrast was positive, negative, and zero in
44, 8, and 48 blocks. The gripper contrast was positive, negative, and zero in
38, 4, and 58 blocks. For `FO-FF`, the discordant counts were 20 versus 2. For
`FO-OO`, they were 27 versus 5. For `FO-OF`, they were 48 versus 3.

## Task consistency

The exact task-level rates verify the pattern highlighted in the Gate-3B
report.

| Task | FF | OO | FO | OF | Fresh-arm effect | Old-gripper effect | FO-FF | FO-OO |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .30 | .30 | .60 | .10 | +.25 | +.25 | +.30 | +.30 |
| 1 | .30 | .40 | .60 | .00 | +.25 | +.35 | +.30 | +.20 |
| 2 | .90 | .90 | 1.00 | .60 | +.20 | +.20 | +.10 | +.10 |
| 3 | .50 | .20 | .50 | .20 | +.30 | .00 | .00 | +.30 |
| 4 | .30 | .50 | .80 | .00 | +.30 | +.50 | +.50 | +.30 |
| 5 | .40 | .60 | .70 | .10 | +.20 | +.40 | +.30 | +.10 |
| 6 | .40 | .30 | .40 | .40 | +.05 | -.05 | .00 | +.10 |
| 7 | .30 | .20 | .40 | .00 | +.25 | +.15 | +.10 | +.20 |
| 8 | .10 | .10 | .20 | .00 | +.10 | +.10 | +.10 | +.10 |
| 9 | .90 | .50 | 1.00 | .30 | +.55 | +.15 | +.10 | +.50 |

`FO` was at least as successful as `FF` on all ten tasks. It was higher on
eight tasks and tied on two. `FO` was higher than `OO` on all ten tasks. The
fresh-arm main effect was positive on all ten tasks. The old-gripper main
effect was positive on eight, zero on one, and negative on one. These task-wise
signs are more consistent than the preregistered coherence contrast, which
averaged `FF` with `OO` and `FO` with `OF`.

## Offline versus closed-loop source preference

The frozen RTX 5080 audit at the same `d=20` source age provides a direct
descriptive comparison. Lower teacher-forced loss is better.

| Component | Fresh-source offline loss | Old-source offline loss | Offline preference | Gate-3B marginal success preference |
|---|---:|---:|---|---|
| Arm translation | .5957820292 | .5066667976 | old | fresh arm (`.530` vs `.285`) |
| Arm rotation, normalized | 1.1296224520 | 1.0987722486 | old | fresh arm (`.530` vs `.285`) |
| Gripper sign error | .3076001092 | .2740024419 | old | old gripper (`.510` vs `.305`) |

The additive offline target descriptively favors the old source for both arm
terms and for the gripper. Gate-3B instead descriptively favors a fresh arm and
an old gripper. The arm preference therefore reverses between teacher-forced
component error and closed-loop marginal success. This comparison does not show
causation, and it does not imply that the demonstration action is erroneous. It
shows that teacher-forced component accuracy and closed-loop temporal-source
utility are not equivalent objectives in these two frozen evaluations.

Delayed-prediction analyses of action chunking provide a plausible reason that
an older observation-conditioned prediction can better match demonstrated
behavior. The present post-hoc pattern asks a narrower question: that advantage
may differ across heterogeneous action components, and delayed teacher-forced
quality may not identify the source age that maximizes closed-loop success.
This remains a candidate interpretation until Gate-3C tests the directional
pattern on its untouched states.

## Reproducibility artifacts

The deterministic analysis is implemented in
`research/audit_tools/gate3b_directional_posthoc.py`. It writes:

- `research/audit_outputs/gate3b_directional_summary.json`;
- `research/audit_outputs/gate3b_directional_contrasts.csv`;
- `research/audit_outputs/gate3b_directional_per_task.csv`;
- `research/audit_outputs/gate3b_directional_leave_one_task_out.csv`;
- `research/audit_outputs/gate3b_directional_block_contrasts.csv`.

No robot rollout, source-age sweep, policy fitting, threshold selection, or
method search was performed. The next scientific evidence is Gate-3C; this
post-hoc analysis stops here.

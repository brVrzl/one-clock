# Post-hoc command-discontinuity characterization

Label: `POST_HOC_COMMAND_DISCONTINUITY_CHARACTERIZATION`

Status at freeze: **SPECIFICATION FROZEN BEFORE COMPUTATION**

Specification and applicability amendment frozen at
`2026-09-03T11:37:06+08:00`. A repository/path search immediately before this
freeze found no command-discontinuity computation, output, marker, or prior
implementation. This is post-hoc mechanism characterization, not a
preregistered mechanism gate. An inconsistent result must be retained.

## Differential-mechanism hypothesis

- Arm temporal benefits may be associated with reduced cross-chunk command
  discontinuity or coherence loss.
- Gripper temporal benefits may instead depend more strongly on forecasting
  the timing of sparse state transitions.

These statements were recorded after the main experiments and before this
characterization was computed. They do not change any decision label.

## Inputs and fixed comparisons

Use existing completed trajectories only. Do not reroll out any condition.

1. Frozen 140-block primary confirmation cohort, excluding its Object bridge:
   Fresh, Reverse20 (`A20G0`), and `HARD_H16` (coherent H16).
2. All 450 Track-A paired blocks for H4 versus ARM4_GRIP32.
3. All 450 Track-A paired blocks for H2 versus ARM2_GRIP16.

No success outcome is required for this characterization. Do not select
episodes, tasks, conditions, or time windows by success.

## Frozen quantities

Let `u_t` be the executed seven-dimensional controller-native command.
Exclude `t=0` from first differences and `t<2` from second differences.

- Translation first difference:
  `D1_translation(t) = sqrt(mean_j=0..2((u_tj-u_(t-1)j)^2))`.
- Rotation first difference:
  `D1_rotation(t) = sqrt(mean_j=3..5((u_tj-u_(t-1)j)^2))`.
- Gripper first difference:
  `D1_gripper(t) = abs(u_t6-u_(t-1)6)`.
- Gripper state switch: `sign(u_t6) != sign(u_(t-1)6)`, retaining zero as a
  separate state.
- Secondary second difference for each group:
  apply the same group RMS/absolute definitions to
  `u_t - 2*u_(t-1) + u_(t-2)`.

Call these quantities `command discontinuity` or `action variation`, never
physical jerk. Translation, rotation, and gripper remain separate because
their controller-native scales and meanings differ.

## Source-transition applicability amendment

For each group, reconstruct source query exactly where possible as
`source_q(t)=t-source_age(t)`, or use the persisted source-query field. A
transition is a `source_chunk_switch` when that group's source query differs
between `t-1` and `t`; otherwise it is `same_source_chunk`.

Within-condition switch-versus-same-source statistics are defined only for a
condition/group in which both transition types actually occur. In sliding
source conditions such as Fresh and A20G0, every eligible step changes source
query. Their same-source statistic is therefore
`STRUCTURALLY_UNAVAILABLE`, not zero and not evidence encoded as NaN. Missing
same-source cells must never be interpreted as scientific outcomes.

Between-condition command-discontinuity comparisons remain defined for all
three fixed comparisons above, independently of whether a condition has both
source-transition types.

Temporal-ensemble actions without a unique contributing source ID would not be
assigned a fabricated source identity. None is needed for the fixed requested
comparisons.

## Summaries and uncertainty

For each trajectory and group, report mean and median D1, mean and median D2,
and gripper state-switch probability. Where both source-transition types are
present, additionally report the two class means and
`mean(D1 | switch) - mean(D1 | same source)`.

For each fixed between-condition comparison, pair by the exact task-state
block and report the mean paired difference, median paired difference, every
task's mean paired difference, and a task-cluster percentile 95% interval.
Use 20,000 bootstrap draws with seed `20260903`. The unit passed to the
bootstrap is task; sampled tasks carry all their paired state blocks.

Because intervention trajectories may visit different states, between-method
differences characterize rollout-level command variation. They are not
state-matched causal action differences.

Outputs are canonical CSV/JSON/Markdown only. No paper-facing artwork is
authorized.

# Final analysis-only mechanism amendment

Status: **FROZEN BEFORE TRACK-A CANONICAL ANALYSIS AND BEFORE REVIEWER-SUPPLEMENT OUTCOMES**

Frozen at `2026-09-03T09:54:08+08:00`, after the Track-B analyses already
reported in `track_b/analysis.json` and the B1/B2 addendum outputs, but before
opening any Track-A scientific outcome or producing any reviewer-supplement
outcome. This amendment changes no rollout, cohort, condition, checkpoint,
state, seed, statistic, decision label, or launch order. The machine-readable
authority is `final_analysis_only_amendment.json`.

## M1. Conditional gripper-persistence prediction

The analysis population is exactly the ten primary tasks in the frozen
140-block cross-suite confirmation: Goal tasks 4, 6, 7, 8, and 9 and LIBERO-10
tasks 0, 2, 4, 6, and 7. For each task, the demonstration population is every
training episode explicitly listed in that task's frozen ACT checkpoint
`train_config.json`. If those lists cannot be mapped exactly to the dataset,
the analysis is unavailable; no language-based or nearest-task substitution is
allowed.

Use the established B2 gripper state `sign(action[6])`, with zero retained as
its own state. For each task, pool only adjacent pairs within demonstrations and
define transition density as

`number of adjacent sign changes / total adjacent-pair duration in seconds`.

The duration uses the audited native dataset rate only. Define
`Delta_G(task) = success(A0G20) - success(A0G0)` from the task's 14 paired
confirmation blocks. The primary association is Spearman rho across all ten
tasks, using average ranks for ties. The prospective direction is `rho < 0`:
higher transition density predicts a smaller or more-negative `Delta_G`.
Report all ten task values. Do not substitute `A0G20-A20G0`, select tasks, or
try alternative gripper metrics after observing the association.

## M2. Symmetric arm-side prediction

For the same task demonstrations, use each task checkpoint's frozen ACT action
standard deviations. For every within-episode adjacent pair, compute the six
arm-component change in checkpoint-normalized action space and define

`V_A(task) = sqrt(mean_{pairs,j=0..5}(((a[t+1,j]-a[t,j])/sigma_task[j])^2))`.

This is reported as normalized controller-action change per native dataset
step, with that step also reported in seconds after the timebase audit. It is
not converted to millimetres or degrees. Define
`Delta_A(task) = success(A20G0) - success(A0G0)` from the same 14 paired blocks.
The primary association is Spearman rho across all ten tasks, average ranks for
ties, with prospective direction `rho < 0`. Report all ten values. If frozen
normalization or exact episode membership cannot be recovered, mark this
analysis unavailable. Do not search alternative arm metrics.

## M3. B1-derived prospective R1B prediction

Label: `B1_DERIVED_PROSPECTIVE_PREDICTION`.

B1 found the same-target normalized dispersion ordering
`rotation > translation > gripper` for both ACT and SmolVLA. If same-target
source disagreement predicts behavioral temporal sensitivity, then at d=20
stale rotation should be at least as damaging as stale translation:

`success(T0_R20_G0) <= success(T20_R0_G0)`.

Equivalently, the frozen direct contrast
`T20_R0_G0 - T0_R20_G0` is predicted to be non-negative. Report the observed
contrast and uncertainty regardless of direction. A negative contrast is
reported as mechanism dissociation, not used to revise B1 or select a new
metric.

## M4. B3-to-R1A physical-time characterization

If and only if the temporal-contract audit establishes an unambiguous mapping,
compare the complete ACT B3 future-action forecast curves with the complete
R1A behavioral sensitivity curves on a seconds-first axis. Use every frozen
R1A age `d={2,4,8,12,16,20,32}` and the corresponding B3 offsets, with no lag
selection. The table must include B3 arm, translation, rotation, gripper-value,
and gripper-sign forecast quantities and R1A `Fresh-A_d_G0` and
`Fresh-A0_G_d`. This is cross-cohort descriptive characterization, not a
significance-gated hypothesis. If time alignment is ambiguous, report the
comparison as not identifiable and keep native step axes separate.

## M5. Censoring-aware B2 persistence

Replace any interpretation of the uncensored mean distance-to-next-transition
as a population mean. Use a lightweight Kaplan-Meier product-limit estimate on
the existing per-action-step time-to-next-gripper-transition observations,
with episode-end right censoring and no cross-episode interval. At the audited
10 Hz dataset rate, report `P(no transition within 0.5 s)`,
`P(no transition within 1.0 s)`, and `P(no transition within 2.0 s)` as
`S(5)`, `S(10)`, and `S(20)`, where `S(h)=P(T>h)` after events at step `h`.
Use 20,000 episode-cluster bootstrap draws, seed 27302, and percentile 95%
intervals. Retain the censoring fraction. The old 30.77-step uncensored mean
may be listed only as an explicitly biased complete-case descriptive value.

## M6. Outcome-aligned Track-B diagnostics

Without rerollout, invert each checkpoint's frozen MEAN_STD action transform
only if code and processor provenance establish the exact mapping. In the
16-source primary window, report per policy:

- gripper same-target unordered-pair sign/state disagreement probability;
- gripper native controller-action RMS source dispersion;
- low-margin minus high-margin sign disagreement under the already-frozen
  target terciles;
- translation and rotation native controller-action RMS source dispersion;
- age-resolved fresh-referenced native RMS differences for translation,
  rotation, and gripper over every age 0..15.

Native grouped dispersion is
`mean_target sqrt(mean_{source,dimensions}((A_source-mean_source A)^2))`;
the gripper formula has one dimension. Fresh-referenced native difference is
`RMS_{episode,target,dimensions}(A_{t-a}[a]-A_t[0])`. Use only
controller-native units unless a source-level physical-unit mapping is proven.
Normalized dispersion remains descriptive context and is not privileged as a
causal mechanism variable. Existing Track-B decision labels remain unchanged:
`ACT_LOCALIZATION_KILL` and no cross-policy mechanism support.

## Interpretation guardrail

Demonstration persistence, policy forecastability, and same-target cross-source
disagreement are related evidence levels, not independent replications and not
interchangeable metrics. SmolVLA is a cross-policy dissociation. The killed
ACT-specific gripper-instability/chatter hypothesis remains killed. Negative
M1--M4 findings are retained.

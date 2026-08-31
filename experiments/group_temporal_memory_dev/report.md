# Group-conditioned temporal memory development

Sol audit commit: `33463ab4eb0ff1c64f794df7c76330bb7b56c143`; repaired baseline commit: `b0b2a6d18ccc9da9ded0057d9f512ad8b535dac0`. Shared baseline: `dense_equivalent_te`. The panel contains only the four frozen development tasks and 40 paired episodes per method.

## ACT

The repaired protocol used a fresh identically seeded LIBERO environment for
each method/state condition. The four development tasks were object3,
spatial0, goal2, and libero_10 task3, with states 10--19 and seeds 2000--2009.
M0 and M1 are the authoritative repaired Sol baselines, reused rather than
rerun; M2 and M3 are the 80 new episodes. All methods queried at exactly
`q=0,16,32,...`, with ACT prediction horizon 100. The strict prefix validator
passed for M0--M3, including raw observations, processed inputs, initial
chunks, actions, simulator states, and post-action observations.

M1 uses Sol's dense-equivalent prior, with candidates ordered oldest to
newest: `b_q ∝ exp(-0.01 * (q-q_oldest))`. M2 multiplies that prior by one
whole-action cosine-compatibility factor with frozen `alpha=0.3` and applies
one normalized weight vector to all seven dimensions. M3 uses the same prior
and alpha, but normalizes cosine compatibility over the six arm dimensions and
sign compatibility over the scalar gripper independently. It still uses an
ordinary weighted average, not sign voting.

| method | success /40 | object3 | spatial0 | goal2 | L10-3 | queries | query rate | mean candidates | arm age | gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0_h16 | 32/40 | 8/10 | 7/10 | 10/10 | 7/10 | 458 | 0.06506 | 4.16 | 7.32 | 7.32 |
| M1_shared_te_h16 | 23/40 | 4/10 | 5/10 | 9/10 | 5/10 | 567 | 0.06468 | 4.48 | 39.99 | 39.99 |
| M2_shared_cogact_h16 | 23/40 | 4/10 | 5/10 | 9/10 | 5/10 | 566 | 0.06460 | 4.48 | 39.39 | 39.39 |
| M3_group_cogact_h16 | 23/40 | 4/10 | 5/10 | 9/10 | 5/10 | 568 | 0.06458 | 4.49 | 39.54 | 39.50 |

| contrast | candidate-only | reference-only | paired net | exact McNemar p |
|---|---:|---:|---:|---:|
| M1_shared_te_h16 vs M0_h16 | 2 | 11 | -9 | 0.0224609375 |
| M2_shared_cogact_h16 vs M1_shared_te_h16 | 0 | 0 | +0 | 1.0 |
| M3_group_cogact_h16 vs M2_shared_cogact_h16 | 0 | 0 | +0 | 1.0 |
| M3_group_cogact_h16 vs M0_h16 | 2 | 11 | -9 | 0.0224609375 |

## Causal sequence

1. M1 versus M0: shared dense-equivalent temporal averaging is harmful (23/40 versus 32/40; repaired Sol baseline).
2. M2 versus M1: whole-action compatibility filtering does not recover any paired episode outcomes (0 candidate-only, 0 reference-only).
3. M3 versus M2: group-conditioned compatibility adds no paired episode outcomes (0 candidate-only, 0 reference-only).
4. M3 versus M0: group-conditioned fusion remains below newest-chunk execution (23/40 versus 32/40; net -9).

## H_temp post-hoc association

H_temp was frozen before outcome files were loaded and was not available to the executor.

- `M1_shared_te_h16_over_M0_h16`: Spearman(H_temp, success-rate gain) = `0.9486832980505139`; counterexamples are listed in `analysis.json`.
- `M2_shared_cogact_h16_over_M1_shared_te_h16`: Spearman(H_temp, success-rate gain) = `None`; counterexamples are listed in `analysis.json`.
- `M3_group_cogact_h16_over_M2_shared_cogact_h16`: Spearman(H_temp, success-rate gain) = `None`; counterexamples are listed in `analysis.json`.
- `M3_group_cogact_h16_over_M0_h16`: Spearman(H_temp, success-rate gain) = `0.9486832980505139`; counterexamples are listed in `analysis.json`.

These correlations are descriptive only. The M2-minus-M1 and M3-minus-M2
gains are zero for every development task, so H_temp provides no evidence that
the group mechanism selectively helps. H_temp was not used by execution or
method selection.

## Reliability interface

M4 was not run. The checkout contains prior outcome-blind reliability work,
but no frozen group-specific reliability value with a validated online runtime
interface for this ladder. No reliability predictor was retrained or
invented. Thus this null result tests M2/M3 compatibility filtering, not the
soft-reliability M4 hypothesis. The earlier failure of a hard predicted
execution horizon does not by itself invalidate continuous soft reliability,
but there is no result here that supports deploying it.

## SmolVLA gate

SmolVLA was not launched. The pre-specified gate required ACT to be
`GROUP_COGACT_STRONG` or `GROUP_COGACT_RECOVERS_HISTORY`; ACT instead returned
`GROUP_COGACT_NULL`. No blind task was touched.

## Decision

**GROUP_COGACT_NULL**

M1 is harmful relative to hard execution; M2 and M3 produce identical episode outcomes to M1 on all 40 paired episodes, so neither recovers useful history and group conditioning adds no outcome benefit.

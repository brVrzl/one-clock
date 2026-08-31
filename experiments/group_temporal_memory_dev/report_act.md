# Group-conditioned temporal memory development

Sol audit commit used: `33463ab4eb0ff1c64f794df7c76330bb7b56c143`. Shared baseline: `dense_equivalent_te`. The panel contains only the four frozen development tasks and 40 paired episodes per method.

## ACT

| method | success /40 | object3 | spatial0 | goal2 | L10-3 | query rate | mean candidates | arm age | gripper age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0_h16 | 32/40 | 8/10 | 7/10 | 10/10 | 7/10 | 0.06506 | 4.16 | 7.32 | 7.32 |
| M1_shared_te_h16 | 23/40 | 4/10 | 5/10 | 9/10 | 5/10 | 0.06468 | 4.48 | 39.99 | 39.99 |
| M2_shared_cogact_h16 | 23/40 | 4/10 | 5/10 | 9/10 | 5/10 | 0.06460 | 4.48 | 39.39 | 39.39 |
| M3_group_cogact_h16 | 23/40 | 4/10 | 5/10 | 9/10 | 5/10 | 0.06458 | 4.49 | 39.54 | 39.50 |

| contrast | candidate-only | reference-only | paired net | exact McNemar p |
|---|---:|---:|---:|---:|
| M2_shared_cogact_h16 vs M1_shared_te_h16 | 0 | 0 | +0 | 1.0 |
| M3_group_cogact_h16 vs M2_shared_cogact_h16 | 0 | 0 | +0 | 1.0 |
| M3_group_cogact_h16 vs M0_h16 | 2 | 11 | -9 | 0.0224609375 |

## H_temp post-hoc association

H_temp was frozen before outcome files were loaded and was not available to the executor.

- `M2_shared_cogact_h16_over_M1_shared_te_h16`: Spearman(H_temp, success-rate gain) = `None`; counterexamples are listed in `analysis.json`.
- `M3_group_cogact_h16_over_M2_shared_cogact_h16`: Spearman(H_temp, success-rate gain) = `None`; counterexamples are listed in `analysis.json`.
- `M3_group_cogact_h16_over_M0_h16`: Spearman(H_temp, success-rate gain) = `0.9486832980505139`; counterexamples are listed in `analysis.json`.

## Decision

**GROUP_COGACT_WHOLE_ONLY**

M2 and M3 changed actions but produced no paired episode-outcome gains over the harmful shared reference; M3 did not add outcome benefit beyond whole-action compatibility.

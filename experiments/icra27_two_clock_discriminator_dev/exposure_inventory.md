# Historical LIBERO exposure inventory

This is a human-readable audit of repository protocols, results, analyses, reports, and relevant reachable Git history. `OUTCOME_EXPOSED` means a rollout outcome is recorded. `PROTOCOL_ONLY` means the cell was named in a plan without a corresponding rollout in that experiment. `NO_EVIDENCE_OF_EXPOSURE` means this audit found neither a completed nor planned artifact. This inventory does not select a later confirmation cohort.

## Recorded outcome exposure

| Suite | Task ID(s) | State IDs/range | Experiment path or history artifact | Exposure class |
|---|---:|---|---|---|
| `libero_object` | 0 | 0–49 | history commits `20d14e4` and `4d20b6c`: `experiments/libero_static_grid_20.json`, `experiments/libero_static_grid_50.json` | `OUTCOME_EXPOSED` |
| `libero_object` | 1–9 | 0–19 | history commit `2a1f1fa`: `experiments/libero_object_cross_task/task_*/result.json` | `OUTCOME_EXPOSED` |
| `libero_object` | 1–9 | 20,21,22,23,27,31,34,35,38,39,44,45,47,48 | `experiments/group_delay_factorial_act20/results/`, `experiments/asymmetric_chunk_reuse_dev/results/` | `OUTCOME_EXPOSED` |
| `libero_object` | 1,4 | 20–29 | `experiments/component_temporal_reuse/fixed_horizon_blind/results/`, `act_temporal_ensemble_blind*` | `OUTCOME_EXPOSED` |
| `libero_object` | 3 | 0–9 | `experiments/component_temporal_reuse/` dense-query SmolVLA cache/results | `OUTCOME_EXPOSED` |
| `libero_object` | 3 | 10–19 | `experiments/sparse_temporal_ensemble_dev/act/results/`, `sparse_temporal_ensemble_age_audit/act_h16/results/`, `group_temporal_memory_dev/act/results/`, `bounded_group_requery_dev/act/results/` | `OUTCOME_EXPOSED` |
| `libero_object` | 6 | 10–19 | `experiments/component_temporal_reuse/{cdta_dev,dynamic_horizon_dev,dynamic_horizon_h16_dev,two_clock_dev}/` and ACT confirmation | `OUTCOME_EXPOSED` |
| `libero_object` | 6 | 25–29 | `experiments/component_temporal_reuse/rapid_component_smoke/` | `OUTCOME_EXPOSED` |
| `libero_object` | 1,5,9 | 0–13 | `experiments/cross_suite_confirmation/results/` bridge panel | `OUTCOME_EXPOSED` |
| `libero_spatial` | 0–9 | 1,13,15,19,21,24,31,37,40,47 | `experiments/gate4a2_spatial_analysis/analysis.json` and its completed preregistered source logs | `OUTCOME_EXPOSED` |
| `libero_spatial` | 0 | 0–9 | `experiments/component_temporal_reuse/` SmolVLA results | `OUTCOME_EXPOSED` |
| `libero_spatial` | 0 | 10–19 | sparse temporal ensemble, group-memory, and bounded-requery result directories | `OUTCOME_EXPOSED` |
| `libero_spatial` | 2 | 10–19 | `experiments/component_temporal_reuse/{cdta_dev,dynamic_horizon_dev,dynamic_horizon_h16_dev,two_clock_dev}/` and ACT confirmation | `OUTCOME_EXPOSED` |
| `libero_spatial` | 2 | 25–29 | `experiments/component_temporal_reuse/rapid_component_smoke/` | `OUTCOME_EXPOSED` |
| `libero_spatial` | 3,7 | 20–29 | `experiments/component_temporal_reuse/fixed_horizon_blind/results/`, `act_temporal_ensemble_blind*` | `OUTCOME_EXPOSED` |
| `libero_spatial` | 4 | 0–9 | `experiments/component_temporal_reuse/` SmolVLA results | `OUTCOME_EXPOSED` |
| `libero_goal` | 1 | 10–19 | ACT temporal-source, aggregation, dynamic-horizon, historical two-clock, and related four-task panels | `OUTCOME_EXPOSED` |
| `libero_goal` | 1 | 25–29 | `experiments/component_temporal_reuse/rapid_component_smoke/` | `OUTCOME_EXPOSED` |
| `libero_goal` | 2 | 0–9 | `experiments/component_temporal_reuse/` SmolVLA results | `OUTCOME_EXPOSED` |
| `libero_goal` | 2 | 10–19 | sparse temporal ensemble, group-memory, and bounded-requery result directories | `OUTCOME_EXPOSED` |
| `libero_goal` | 0,3 | 20–29 | `experiments/component_temporal_reuse/fixed_horizon_blind/results/`, `act_temporal_ensemble_blind*` | `OUTCOME_EXPOSED` |
| `libero_goal` | 4,6,7,8,9 | 0–13 | `experiments/cross_suite_confirmation/results/`, `experiments/candidate1_c2_cross_suite/results/` | `OUTCOME_EXPOSED` |
| `libero_goal` | 5 | 0–9 | `experiments/component_temporal_reuse/` SmolVLA results | `OUTCOME_EXPOSED` |
| `libero_10` | 3 | 0–9 | `experiments/component_temporal_reuse/` SmolVLA results | `OUTCOME_EXPOSED` |
| `libero_10` | 3 | 10–19 | ACT confirmation, sparse temporal ensemble, group-memory, bounded-requery, historical two-clock, and dynamic-horizon result directories | `OUTCOME_EXPOSED` |
| `libero_10` | 3 | 25–29 | `experiments/component_temporal_reuse/rapid_component_smoke/` | `OUTCOME_EXPOSED` |
| `libero_10` | 1,9 | 20–29 | `experiments/component_temporal_reuse/fixed_horizon_blind/results/`, `act_temporal_ensemble_blind*` | `OUTCOME_EXPOSED` |
| `libero_10` | 0,2,4,6,7 | 0–13 | `experiments/cross_suite_confirmation/results/`, `experiments/candidate1_c2_cross_suite/results/` | `OUTCOME_EXPOSED` |
| `libero_10` | 5 | 0–9 | `experiments/component_temporal_reuse/` SmolVLA results | `OUTCOME_EXPOSED` |
| all four standard suites | 0–9 | 10 evaluation episodes per task; exact init-state IDs are not preserved in the aggregate result | `experiments/standard_libero_baselines/results.json` and `report.md` | `OUTCOME_EXPOSED` |

The standard-baseline row is conservatively treated as outcome exposure at task level. Its aggregate files record per-episode outcomes and seed 1000 but not an auditable explicit init-state list, so they should not be used to certify a supposedly untouched state without checking the underlying evaluator artifacts.

## Protocol-only exposure

| Suite | Task ID(s) | State IDs/range | Experiment path | Exposure class |
|---|---:|---|---|---|
| `libero_object` | 0 | 20,21,22,23,27,31,34,35,38,39,44,45,47,48 | `experiments/group_delay_factorial_act20/protocol.json` secondary task; no task-0 result shard exists in that experiment | `PROTOCOL_ONLY` |
| `libero_object` | 1–9 | 20,21,22,23,27,31,34,35,38,39,44,45,47,48 | `experiments/icra27_two_clock_discriminator_dev/protocol.json`; two new result conditions pending at protocol-freeze time | `PROTOCOL_ONLY` |

The first protocol-only row is not scientifically untouched: Object task 0 states 0–49 already have static-grid outcomes in history. Protocol-only class describes that particular experiment artifact, not the union of all exposure evidence.

## No evidence found in this audit

| Suite | Task ID(s) | State IDs/range | Experiment path | Exposure class |
|---|---:|---|---|---|
| `libero_goal` | 4,6,7,8,9 | 14–49 | repository audit; no matching completed or planned artifact found | `NO_EVIDENCE_OF_EXPOSURE` |
| `libero_10` | 0,2,4,6,7 | 14–49 | repository audit; no matching completed or planned artifact found | `NO_EVIDENCE_OF_EXPOSURE` |
| `libero_object` | 1,4 | 30,32,33,36,37,40,41,42,43,46,49 | repository audit after combining cross-task, fixed-blind, factorial, asymmetric, and confirmation artifacts | `NO_EVIDENCE_OF_EXPOSURE` |
| `libero_object` | 2,3,5,6,7,8,9 | 24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49 | repository audit after combining cross-task, factorial, asymmetric, and confirmation artifacts | `NO_EVIDENCE_OF_EXPOSURE` |

These `NO_EVIDENCE_OF_EXPOSURE` rows are audit findings, not a recommendation or a selected confirmation cohort.

## Important contamination findings

- Every LIBERO Object task has prior recorded outcomes; Object task 0 is outcome-exposed across all official states 0–49.
- Object tasks 1–9 are all outcome-exposed on states 0–19 and on the full 14-state development cohort used in this experiment. A later confirmation cannot be called untouched merely by choosing a different Object task from 1–9.
- The former cross-suite confirmation tasks in Goal and LIBERO-10, states 0–13, are now outcome-exposed, including the later Candidate-1/C2 round.
- All ten Spatial tasks have outcome exposure on the preregistered Gate-4A2 state set, in addition to standard-baseline and task-specific development exposure.
- Some old result artifacts survive only in reachable Git history rather than the fallback working tree. They still count as scientific exposure.

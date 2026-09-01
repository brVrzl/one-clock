# Repository-wide exposure inventory

`OUTCOME_EXPOSED` means a rollout outcome exists on at least one reachable remote ref. `PROTOCOL_ONLY` means a planned cell exists but no matching outcome was found. `NO_EVIDENCE_OF_EXPOSURE` is an audit finding, not a recommendation. All `origin/*` refs were inspected.

| Suite | Task(s) | State(s) | Experiment / branch or path | Classification |
|---|---:|---|---|---|
| Object | 0 | 0–49 | `libero_static_grid_50`, reachable from `origin/main` and older experimental refs | `OUTCOME_EXPOSED` |
| Object | 1–9 | 0–19 | `libero_object_cross_task`, reachable from `origin/main` and older experimental refs | `OUTCOME_EXPOSED` |
| Object | 1–9 | 20,21,22,23,27,31,34,35,38,39,44,45,47,48 | `group_delay_factorial_act20`, `asymmetric_chunk_reuse_dev`, completed discriminator | `OUTCOME_EXPOSED` |
| Object | 6 | 10–19 | `dynamic_horizon_dev`, `dynamic_horizon_h16_dev`, historical two-clock and ACT confirmation | `OUTCOME_EXPOSED` |
| Object | 3 | 10–19 | sparse temporal ensemble, `group_temporal_memory_dev`, `bounded_group_requery_dev` | `OUTCOME_EXPOSED` |
| Spatial | 0 | 10–19 | sparse temporal ensemble, `group_temporal_memory_dev`, `bounded_group_requery_dev` | `OUTCOME_EXPOSED` |
| Goal | 2 | 10–19 | sparse temporal ensemble, `group_temporal_memory_dev`, `bounded_group_requery_dev` | `OUTCOME_EXPOSED` |
| Long (`libero_10`) | 3 | 10–19 | sparse temporal ensemble, `group_temporal_memory_dev`, `bounded_group_requery_dev`, ACT confirmation | `OUTCOME_EXPOSED` |
| Goal | 4,6,7,8,9 | 0–13 | `cross_suite_confirmation` and `candidate1_c2_cross_suite` | `OUTCOME_EXPOSED` |
| Long | 0,2,4,6,7 | 0–13 | `cross_suite_confirmation` and `candidate1_c2_cross_suite` | `OUTCOME_EXPOSED` |
| Object | 1,5,9 | 0–13 | `cross_suite_confirmation` bridge | `OUTCOME_EXPOSED` |
| Spatial | 0–9 | recorded task-level standard baseline; Gate-4A2 state set 1,13,15,19,21,24,31,37,40,47 | standard baseline and Gate-4A2 | `OUTCOME_EXPOSED` |
| All four suites | 0–9 | standard evaluation: 10 episodes/task, aggregate artifact lacks explicit state IDs | `standard_libero_baselines` | `OUTCOME_EXPOSED` at task level |
| All four suites | 0–9 | 0–3 | tonight's frozen SmolVLA primary/capacity manifest (outcomes pending at inventory creation) | `PROTOCOL_ONLY` |
| Goal | 4,6,7,8,9 | 14–49 | remote-ref audit | `NO_EVIDENCE_OF_EXPOSURE` |
| Long | 0,2,4,6,7 | 14–49 | remote-ref audit | `NO_EVIDENCE_OF_EXPOSURE` |
| Object | 1,4 | 30,32,33,36,37,40,41,42,43,46,49 | combined remote-ref audit | `NO_EVIDENCE_OF_EXPOSURE` |
| Object | 2,3,5,6,7,8,9 | 24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49 | combined remote-ref audit | `NO_EVIDENCE_OF_EXPOSURE` |

Tonight does not select a future confirmation cohort.


# Gate-3C historical LIBERO state-usage audit

Audit date: 2026-08-24

Scientific parent: `2817411a4210b8611dc8dae5d32ec99fc6b94cf3`

Purpose: freeze an outcome-blind official-state cohort for Gate-3C. This audit
read only rollout identity fields (`task_id`, `state_id`, `init_state_id`, or
manifest-level state lists). It did not read, print, rank, or condition on any
historical success field or per-state outcome.

## Audited closed-loop sources

| Scientifically used rollout source | Identity source | Task 0 official IDs | Tasks 1–9 official IDs |
|---|---|---:|---:|
| Initial pilots and sanity runs | `experiments/runs/**/metadata.json` | subsets of 0–19 | subsets of 0–19 |
| Task-0 static grids | `experiments/runs/libero_static_grid_20/**/metadata.json`; `experiments/runs/libero_static_grid_50_extension/**/metadata.json` | 0–49 | not run |
| Cross-task static grid | `experiments/runs/libero_object_cross_task/task_{1..9}/**/metadata.json` | reused task-0 grid | 0–19 |
| Gate-2B phase-conditioned grid | `experiments/phase_conditioned_oracle/config_results.json` identity fields | 0–49 | 0–19 |
| Matched-query selective retention | `experiments/groupwise_selective_commitment/paired_seed_manifest.json` | 0–19 | 0–19 |
| Gate-3A2 | `research/audit_outputs/gate3a2_rollout_manifest.json` | `[0,7,11,13,25,30,36,41,42,43]` | same |
| Gate-3B | `research/audit_outputs/gate3b_rollout_manifest.json` | `[24,26,28,29,32,33,37,40,46,49]` | same |

The repository source-of-truth map identifies these as the scientifically used
closed-loop LIBERO rollout families. Offline dense caches, teacher-forced
analyses, and non-LIBERO experiments are not closed-loop official-state uses.
Duplicate executions of an already used state do not change the set audit.

## Per-task union

Task 0 has used every official state ID `0–49`. There is no untouched-state
task-0 cohort. Gate-3C therefore includes task 0 only as a preregistered
secondary generalization/sensitivity task.

For every task 1–9, the union is identical:

```text
0–19,
24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49
```

The common genuinely unused set for tasks 1–9 is consequently:

```text
[20, 21, 22, 23, 27, 31, 34, 35, 38, 39, 44, 45, 47, 48]
```

Its count is 14, within the preregistered 10–15 range. All 14 IDs are selected;
the reserved `numpy.default_rng(20260830)` sampling rule is not invoked. The
same ordered state list is frozen for all ten tasks. Historical outcomes were
not inspected while deriving or freezing this cohort.

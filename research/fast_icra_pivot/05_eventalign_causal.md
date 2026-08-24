# EventAlign causal timing sweep

Date: 2026-08-24. Status: exploratory paired closed-loop upper-bound diagnostic.

## Question

Can changing only gripper open/close event timing rescue failures of a frozen ACT policy while leaving the continuous arm trajectory unchanged?

## Protocol

- Tasks: LIBERO-Object task 1 (cream cheese), task 6 (butter), and task 8 (chocolate pudding).
- Frozen checkpoint, observations, global execution horizon 8, and arm actions held fixed.
- Gripper sequence shifts: `{-8, -4, 0, +4, +8}` control steps. Negative advances and positive delays the nominal sequence; edge values are replicated.
- Ten official initial states per task, IDs 0–9, with paired seeds 1000–1009 for every candidate.
- Shift 0 was rerun inside this sweep and is the causal baseline.
- The implementation copies dimensions 0–5 exactly, changes only dimension 6, and asserts exact arm equality before returning each chunk.
- All 150 episodes completed and are retained. Success is the primary endpoint; no action-MSE selection was performed.

## Aggregate results

| Task | -8 | -4 | 0 baseline | +4 | +8 | Timing oracle | Recoverable baseline failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 cream cheese | 4/10 | 7/10 | 6/10 | 4/10 | 1/10 | 8/10 | 2/4 (50%) |
| 6 butter | 1/10 | 4/10 | 7/10 | 4/10 | 5/10 | 7/10 | 0/3 (0%) |
| 8 chocolate pudding | 1/10 | 1/10 | 2/10 | 2/10 | 1/10 | 3/10 | 1/8 (12.5%) |
| **Pooled** | 6/30 | 12/30 | **15/30** | 10/30 | 7/30 | **18/30** | **3/15 (20%)** |

The per-state oracle adds 3 successes over 30 paired states, an absolute upper-bound headroom of 10 percentage points. This upper bound assumes knowledge of which candidate will succeed after execution and therefore cannot be achieved by a deployable selector.

## Baseline-success breakage

| Task | -8 | -4 | +4 | +8 |
|---|---:|---:|---:|---:|
| 1 cream cheese | 3/6 (50%) | 1/6 (16.7%) | 2/6 (33.3%) | 5/6 (83.3%) |
| 6 butter | 6/7 (85.7%) | 3/7 (42.9%) | 3/7 (42.9%) | 2/7 (28.6%) |
| 8 chocolate pudding | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) | 2/2 (100%) |

No fixed shift improves pooled success over baseline. The best fixed candidate, −4, scores 12/30 versus baseline 15/30. Its apparent task-1 gain trades two rescued failures against one broken baseline success and does not transfer to tasks 6 or 8.

## Per-initial-state outcomes

The complete paired table is in `artifacts/eventalign_analysis/report.md`; machine-readable outcomes are in `per_state.csv` and `summary.json`. A `1` denotes task success.

| Task | State | -8 | -4 | 0 | +4 | +8 | Failure rescued by any shift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| 1 | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| 1 | 2 | 1 | 1 | 1 | 1 | 1 | 0 |
| 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 4 | 1 | 1 | 0 | 0 | 0 | 1 |
| 1 | 5 | 0 | 0 | 1 | 0 | 0 | 0 |
| 1 | 6 | 0 | 1 | 0 | 0 | 0 | 1 |
| 1 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 8 | 1 | 1 | 1 | 1 | 0 | 0 |
| 1 | 9 | 1 | 1 | 1 | 1 | 0 | 0 |
| 6 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 6 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| 6 | 2 | 0 | 0 | 1 | 0 | 0 | 0 |
| 6 | 3 | 1 | 1 | 1 | 1 | 1 | 0 |
| 6 | 4 | 0 | 1 | 1 | 1 | 1 | 0 |
| 6 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 7 | 0 | 0 | 1 | 0 | 1 | 0 |
| 6 | 8 | 0 | 1 | 1 | 1 | 1 | 0 |
| 6 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 4 | 0 | 0 | 1 | 1 | 0 | 0 |
| 8 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 7 | 1 | 1 | 1 | 0 | 0 | 0 |
| 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 9 | 0 | 0 | 0 | 1 | 1 | 1 |

## Decision

**KILL EVENTALIGN.**

The oracle headroom is too small and inconsistent to support a three-task mechanism: only 20% of baseline failures are timing-recoverable, no failure is recoverable on task 6, and every nonzero shift frequently breaks correct executions. Training a selector would attempt to approximate an oracle whose ceiling is already only 60% pooled success. The user-specified kill rule therefore triggers, and no discrete selector is trained.

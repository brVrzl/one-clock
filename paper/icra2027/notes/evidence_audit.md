# Evidence audit

Audit date: 2026-08-20. Repository: `brVrzl/one-clock`. Audited starting HEAD:
`466ce0a5942bc7660b1ddff4a54f8758e7854727`. The prompt's expected HEAD
`2a1f1fab34c086a60bf92e23b4dde1b5a0bb59d8` is the parent of the audited
HEAD; the intervening commit adds the committed dynamic-readiness analysis.

## Implementation contract

- `src/one_clock/executor.py` defines `FixedChunkExecutor`. In group-wise mode,
  a policy query is issued whenever at least one group expires. The new full
  chunk is installed only for expired groups; non-expired groups retain their
  earlier chunk and advance their own cursor. The executed action can therefore
  combine components from different policy-query generations. This is both the
  mechanism under study and a possible source of cross-group inconsistency.
- `scripts/run_libero_gate0.py` verifies a chunked policy output of shape
  `(100, 7)`, requires temporal ensembling to be disabled, and constructs two
  groups: arm/end-effector indices `0:6` and gripper index `6`. The runner uses
  relative control, official initial states, and a 20 Hz control rate.
- The frozen checkpoint recorded by the artifacts is
  `zeromidnight/act_libero_object` (local cache path omitted from the paper).
  Runtime records identify LeRobot 0.6.2 and `hf-libero` 0.1.4. No experiment
  in this audit retrains or fine-tunes the policy.

## Experiment-level evidence

| Experiment | Task(s) | Checkpoint | States / seeds | Configurations | Metric | Exact artifacts | Status | Claims supported |
|---|---|---|---|---|---|---|---|---|
| Task-0 full static landscape | LIBERO Object 0, `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | Frozen `zeromidnight/act_libero_object` | Official states 0--49; seeds 1000--1049 | Global `h in {1,2,4,8,16}`; complete arm/gripper 5x5 grid over the same set | Episode success, success count/rate, policy-query rate, paired exact McNemar diagnostic | `experiments/libero_static_grid_50.json`; `experiments/libero_static_grid_50.md` | Committed | Horizon sensitivity; 5x5 landscape; diagonal controls; best evaluated `(4,16)=47/50`; budget-matched comparison; assignment directionality |
| Task-0 paired execution audit | Same task as above | Same | Same paired state IDs | Global and group-specific fixed horizons | Episode-level equality and discordant counts; executor semantics | `experiments/libero_execution_audit.md`; `src/one_clock/executor.py`; `scripts/run_libero_gate0.py` | Committed | Pairing validity, action partition, fresh-query/retained-buffer behavior, absence of temporal ensembling |
| Cross-task static diagnostic | LIBERO Object tasks 1--9 | Same frozen checkpoint | Official states 0--19 per task; seeds 1000--1019 | Common coarse set: diagonal/global controls at 2, 4, 8, 16 and evaluated off-diagonal pairs among 2, 4, 8, 16; not a full 5x5 grid | Success, query rate, paired counts, per-task retrospective best configuration | `experiments/libero_object_cross_task_summary.json`; `experiments/libero_object_cross_task_summary.md`; `experiments/libero_object_cross_task/task_{1..9}/result.json`; corresponding `result.md` | Committed | Macro static oracle-style diagnostic (.700 best global vs .761 best off-diagonal); configuration classes 6/3/0; directionality 24/6/6; six tasks with a no-higher-query-rate off-diagonal improvement |
| Checkpoint/task coverage | All ten LIBERO Object tasks | Same | Benchmark task/state inventory; no rollout metric | N/A | Dataset/runtime coverage audit | `experiments/libero_object_cross_task/task_coverage_audit.md` | Committed | The checkpoint/dataset and runtime expose all ten object tasks and official state sets; this is not performance evidence |
| Static universal-pair readiness analysis | LIBERO Object tasks 0--9 | Same; no new policy rollout | Uses the already completed paired rollouts above | Common set across all tasks: global 2/4/8/16 and 12 group pairs | Macro success/query rate, bootstrap task resampling, leave-one-task-out static selection, retrospective oracle gap | `experiments/libero_object_dynamic_readiness.json`; `experiments/libero_object_dynamic_readiness.md` | Committed | A single common static `(4,16)` pair is a strong baseline (.734 macro); the common-set per-task group oracle is .779; the .045 oracle gap motivates but does not prove a need for dynamic scheduling |
| Gate-1 group prediction-persistence audit | All ten LIBERO Object demonstration tasks | Frozen ACT checkpoint; no rollout | 454 demonstration episodes; 1,098 full-chunk observation points; 100-step predictions | Teacher-forced future-action comparison; arm translation/rotation and gripper metrics | Group-wise prediction-error curves, AUC/slope summaries, bootstrap intervals | `experiments/group_prediction_persistence_audit.md`; `experiments/group_prediction_persistence/summary.json` and plots | Committed offline audit | Group-dependent temporal error structure; no monotone persistence horizon; no dynamic-execution result |
| Gate-2A phase-conditioned persistence audit | All ten LIBERO Object demonstration tasks | Frozen ACT checkpoint; no rollout | 454 episodes; 3,740 points; early/middle/late normalized phases; common `k=0..37` window | Episode-balanced phase-conditioned prediction-error AUCs and paired bootstrap effects | Phase-conditioned arm translation/rotation and gripper error profiles | `experiments/phase_conditioned_persistence_audit.md`; `experiments/phase_conditioned_persistence/summary.json` and plots | Committed offline audit | Phase dependence for arm translation and gripper error; mixed arm-rotation direction; no dynamic-execution result |
| Exploratory execution-trace analysis | Task 0 | Same | 50 states aggregated; step traces for evaluated runs | Global 4/8 and selected group pairs including `(4,16)` and `(16,4)` | Gripper total variation/sign changes; arm action variation | `trace_aggregates`, `trace_comparisons`, and `trace_episode_metrics` in `experiments/libero_static_grid_50.json`; narrative in `.md` | Committed aggregate; underlying step logs local-only | Descriptive differences in executed trajectories. Supports no causal mechanism and is omitted from primary claims |
| Selected replay cases | Task 0 states 20 and 41 | Same | Individual paired states | State 41: global 8 failure vs `(4,16)` success; state 20: `(4,16)` success vs `(16,4)` failure | Qualitative replay outcome | `experiments/libero_video_cases_50.md`; `experiments/runs/libero_static_grid_50/videos/*.mp4` | Index committed; MP4 and replay logs local-only/ignored | Illustrative cases only; not independent quantitative evidence |

## Exact quantitative checks used by the manuscript

### Task 0 global sweep

| Horizon | Success | Query rate |
|---:|---:|---:|
| 1 | 29/50 (0.58) | 1.000 |
| 2 | 31/50 (0.62) | 0.501 |
| 4 | 42/50 (0.84) | 0.252 |
| 8 | 45/50 (0.90) | 0.128 |
| 16 | 42/50 (0.84) | 0.065 |

The best evaluated group-specific configuration is `(h_arm,h_gripper)=(4,16)`:
47/50 (0.94), query rate 0.252. Against global `h=4`, paired counts are
42 both successful, 0 global-only, 5 group-specific-only, and 3 both failed;
the two-sided exact McNemar diagnostic is `p=0.0625`. This is described as a
consistent paired improvement, not as statistically significant.

All ten task-0 symmetric comparisons favor the assignment with the longer
gripper horizon. The success rates, in `(short,long)` versus `(long,short)`
order, are: 0.60/0.52, 0.68/0.56, 0.80/0.52, 0.80/0.58, 0.84/0.62,
0.84/0.66, 0.88/0.60, 0.90/0.82, 0.94/0.74, and 0.90/0.82. These data
support assignment sensitivity, not a universal rule for gripper timing.

### Cross-task diagnostic

- Macro retrospective per-task best global success: 0.700.
- Macro retrospective per-task best evaluated off-diagonal success: 0.761.
- Difference: +0.061. This is an oracle-style diagnostic, not one deployable
  method.
- Best-configuration classes: off-diagonal on 6 tasks; diagonal/off-diagonal
  tie on 3; diagonal-only on 0.
- Symmetric assignment outcomes: longer-gripper 24, reversed 6, ties 6.
- On 6/9 tasks, at least one off-diagonal point strictly improves a global
  point at no higher measured query rate.
- Task 1 budget control: diagonal/global-equivalent 2 gives 6/20 (0.30) at
  query rate 0.5004; `(2,16)` gives 14/20 (0.70) at 0.5005. Discordant counts
  are 0 versus 8; exact `p=0.0078125`.
- Task 4 budget control: diagonal/global-equivalent 2 gives 10/20 (0.50) at
  query rate 0.5006; `(2,16)` gives 17/20 (0.85) at 0.5018. Discordant counts
  are 0 versus 7; exact `p=0.015625`.

These task-level p-values are descriptive diagnostics from small paired samples
and are not corrected for multiple comparisons.

## Local-only raw evidence inspected

The following ignored artifacts were read to verify aggregation, metadata, and
mixed-generation behavior. They must not be added to Git.

- `experiments/runs/libero_static_grid_20/group_arm4_grip16/{metadata,summary}.json`
  and `steps.jsonl`: task-0 states 0--19; 19/20 success; query rate 0.2523;
  temporal ensembling recorded as null.
- `experiments/runs/libero_static_grid_50_extension/group_arm4_grip16/` and
  `group_arm4_grip16_states21_49/`: task-0 extension through state 49.
- The `(4,16)` step trace shows both groups using generation 0 at step 0;
  at steps 4, 8, and 12 the arm uses fresh generations 1, 2, and 3 while the
  gripper continues generation 0 at cursor positions 4, 8, and 12; both accept
  generation 4 at step 16.
- `experiments/runs/libero_object_cross_task/task_1/group_arm2_grip2/` and
  `group_arm2_grip16/`: raw 6/20 versus 14/20 outcomes and nearly identical
  query rates.
- `experiments/runs/libero_object_cross_task/task_4/group_arm2_grip2/` and
  `group_arm2_grip16/`: raw 10/20 versus 17/20 outcomes and nearly identical
  query rates.
- `experiments/runs/libero_static_grid_50/videos/` and associated replay logs:
  task-0 state 20 and 41 examples indexed by the committed video-cases note.
- `experiments/runs/libero_object_cross_task/videos/` and `video_replays/` were
  inventoried but are not used for numerical manuscript claims.

The committed aggregates remain the manuscript's primary numerical source.

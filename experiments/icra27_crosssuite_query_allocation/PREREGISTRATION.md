# Track-A cross-suite query-allocation preregistration

Status: **FROZEN BEFORE TRACK-A OUTCOMES**

This preregistration governs a prospective evaluation of query-allocation conditions frozen from Object development on non-Object task-state cells selected without reference to their query-allocation outcomes. It does not describe the suites or policies as globally unseen or globally executor-unexposed.

## Scientific question and prior-art boundary

Frequent replanning degradation and the action-chunk consistency/reactivity trade-off are prior art (ACT, BID, RTC). The new question is whether the penalty is non-uniform across action components and whether preserving gripper commitment while refreshing the arm mitigates it under the same periodic policy-query schedule. We report policy-query rate/budget and replanning frequency; wall-clock is separate. Policy queries are not FLOPs or compute scaling.

## Frozen cohort

- Valid task-specific non-Object ACT policies: 30.
- Frozen states per task: 15.
- Paired task-state blocks: 450.
- Conditions per block: 6.
- Planned scientific episodes: 2700.
- Object checkpoints/results are development and hypothesis-generation evidence only.
- The deterministic selection rule in `confirmation_cohort.json` conservatively excluded every state with any prior executor-variant outcome. No success rate, difficulty, or previous effect entered selection.
- Historical non-Object task-specific ACT outcomes under any exact H4/H2/ARM4_GRIP32/ARM2_GRIP16 condition before freeze: 0 cells.

Exact ordered states:

- `libero_10:task0`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_10:task1`: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
- `libero_10:task2`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_10:task3`: [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
- `libero_10:task4`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_10:task5`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_10:task6`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_10:task7`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_10:task8`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_10:task9`: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
- `libero_goal:task0`: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
- `libero_goal:task1`: [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
- `libero_goal:task2`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_goal:task3`: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
- `libero_goal:task4`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_goal:task5`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_goal:task6`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_goal:task7`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_goal:task8`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_goal:task9`: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
- `libero_spatial:task0`: [20, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38]
- `libero_spatial:task1`: [20, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38]
- `libero_spatial:task2`: [20, 22, 23, 30, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44]
- `libero_spatial:task3`: [30, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 48]
- `libero_spatial:task4`: [20, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38]
- `libero_spatial:task5`: [20, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38]
- `libero_spatial:task6`: [20, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38]
- `libero_spatial:task7`: [30, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 48]
- `libero_spatial:task8`: [20, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38]
- `libero_spatial:task9`: [20, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38]

## Six frozen conditions

1. `H16`: coherent arm16/gripper16.
2. `H4`: coherent arm4/gripper4.
3. `ARM4_GRIP32`: arm refresh 4, gripper commitment 32; exact historical 4x Object contrast.
4. `H2`: coherent arm2/gripper2.
5. `ARM2_GRIP16`: arm refresh 2, gripper commitment 16; exact historical 8x Object contrast.
6. `TE_DENSE`: canonical dense upstream ACT temporal ensembling, query every step, coefficient 0.01, oldest-to-newest exponential weights, all seven normalized action dimensions aggregated before checkpoint denormalization; no tuning and no sparse approximation.

H4 must equal group arm4/grip4 and H2 must equal group arm2/grip2 step-by-step on exposed technical canaries. The frozen workers are task-major: load one task-specific checkpoint once, run all ordered states and six methods, release policy/environment, then load the next task. Static sorted-task modulo-three sharding is fixed. There is no result-dependent scheduling. Scientific failures are outcomes; only technical failures may be retried, at most twice after the initial attempt.

## Questions, contrasts, and inference

The task policy is the primary generality unit and the task-state cell is the paired block. Every contrast reports success counts/rates, paired discordances, percentage-point delta, exact two-sided McNemar, 20,000-draw paired percentile CI, 20,000-draw task-cluster percentile CI, all per-task deltas, leave-one-task-out, per-suite descriptive deltas, leave-one-suite-out, policy queries/rate, environment steps, and wall-clock.

- Q-A1 primary: `H16-H4`; secondary dose response: `H4-H2`.
- Q-A2 primary matched schedule: `ARM4_GRIP32-H4`; secondary: `ARM2_GRIP16-H2`.
- Q-A3 primary: `ARM4_GRIP32-H16`; secondary: `ARM2_GRIP16-H16`.
- Q-A4 standard-practice reference: `TE_DENSE-H16` and `TE_DENSE-ARM4_GRIP32`; TE is not query-budget matched.

Decision-label definitions are frozen verbatim in `track_a_manifest.json`. Strict H4>H2 monotonicity is not required to interpret the 4x mechanism. The 3 pp TE practical-equivalence reference is descriptive only: at this cohort size it is at most about 14 net successes, predeclared as a small materiality tolerance against an approximately fourfold query-rate difference, not a universal equivalence or Pareto theorem.

## Throughput decision

The conservative task-major prediction is 6.31 h for the slowest worker against the predeclared 18 h Track-A window. Therefore 15 states/task are frozen uniformly for all six conditions. TE_DENSE is retained.

## Frozen exploratory moderator

`gripper_activity_moderator.json` fixes, from training-demonstration metadata only, each task's fraction of action steps with closed gripper intent. After outcome freeze, all valid tasks enter one exploratory Spearman correlation with task-level `ARM4_GRIP32-H4`; there are no post-hoc categories or selected tasks.

## Track-B diagnostic

Track B has 80 mechanism-only episodes on already outcome-exposed development cells; outcomes are not used for method selection. ACT and SmolVLA execute exact H16 trajectories while extra per-step queries are logging-only. The primary window is target t>=15 and source ages 0..15. Dispersion, sign metrics, bootstrap, and ACT/cross-policy labels are fixed in `track_b_manifest.json`. Dense logging must pass exact action/state/terminal/length canaries. Passing Track B does not authorize debounce, consensus, or any new ICRA method development.

## Prior evidence and interpretation constraints

`evidence_tension_and_factorial_notes.md` is incorporated by reference. In particular, the +32.14 pp FO20-Reverse20 diagonal is not assigned uniquely to either component, the weak FO20-Fresh evidence is kept in tension with Object ARM4_GRIP32-H4 development, differing semantics are hypotheses only, and no reconciliation experiment is authorized. The 126-block Object and 140-block cross-suite factorial interactions retain distinct provenance and opposite sign conventions.

## Frozen exclusions

No fixed-clock search, CARE/M2 variant, grip64, horizon sweep, adaptive threshold, outcome-selected task search, consensus/debounce, RTC pivot, PACE reproduction, or new horizon cell may be launched. The manuscript and `CLAIMS.md` remain untouched.

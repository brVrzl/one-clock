# Track-A exposure audit

The audit used the frozen 28-ref snapshot and parsed 1963 unique outcome JSON/JSONL artifacts plus path-identified compressed episode artifacts. No success magnitude entered state selection.

Exposure is recorded per task-specific ACT task/state cell with four distinct fields: `BASELINE_EXPOSED`, `OTHER_EXECUTOR_EXPOSED`, `QUERY_ALLOCATION_CONDITION_EXPOSED`, and `TRACK_A_CELL_PROSPECTIVE`. Standard baseline exposure is recorded at task-policy granularity because every policy has an existing eval10; it does not exclude a state.

Across the 1,500 candidate non-Object ACT cells, 355 have another executor outcome and 0 have an exact H4/H2/ARM4_GRIP32/ARM2_GRIP16-family outcome. Cross-policy SmolVLA outcomes are not treated as exposure of the task-specific ACT policy cell.

The deterministic rule selected **450 blocks across 30 task policies**. Every selected cell is `TRACK_A_CELL_PROSPECTIVE=true` and conservatively free of any prior ACT executor-variant outcome.

The preregistered scientific wording is: **query-allocation conditions frozen from Object development were prospectively evaluated on non-Object task-state cells selected without reference to their query-allocation outcomes.** The policies and suites are not described as unseen or globally executor-unexposed.

## Deterministic selected states

- `libero_spatial:task0`: 20,22,23,25,26,27,28,29,30,32,33,34,35,36,38
- `libero_spatial:task1`: 20,22,23,25,26,27,28,29,30,32,33,34,35,36,38
- `libero_spatial:task2`: 20,22,23,30,32,33,34,35,36,38,39,41,42,43,44
- `libero_spatial:task3`: 30,32,33,34,35,36,38,39,41,42,43,44,45,46,48
- `libero_spatial:task4`: 20,22,23,25,26,27,28,29,30,32,33,34,35,36,38
- `libero_spatial:task5`: 20,22,23,25,26,27,28,29,30,32,33,34,35,36,38
- `libero_spatial:task6`: 20,22,23,25,26,27,28,29,30,32,33,34,35,36,38
- `libero_spatial:task7`: 30,32,33,34,35,36,38,39,41,42,43,44,45,46,48
- `libero_spatial:task8`: 20,22,23,25,26,27,28,29,30,32,33,34,35,36,38
- `libero_spatial:task9`: 20,22,23,25,26,27,28,29,30,32,33,34,35,36,38
- `libero_goal:task0`: 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
- `libero_goal:task1`: 20,21,22,23,24,30,31,32,33,34,35,36,37,38,39
- `libero_goal:task2`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_goal:task3`: 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
- `libero_goal:task4`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_goal:task5`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_goal:task6`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_goal:task7`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_goal:task8`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_goal:task9`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_10:task0`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_10:task1`: 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
- `libero_10:task2`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_10:task3`: 20,21,22,23,24,30,31,32,33,34,35,36,37,38,39
- `libero_10:task4`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_10:task5`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_10:task6`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_10:task7`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_10:task8`: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
- `libero_10:task9`: 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44

## Exclusion provenance

Every conservatively excluded state with an outcome has experiment, remote ref(s), introducing commit, artifact path, and blob recorded in `exposure_audit.json`. States not chosen merely because they followed the first 15 eligible IDs are not exposure exclusions.

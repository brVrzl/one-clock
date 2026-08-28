# Complete frozen-pilot analysis

The analysis guard accepted all 80 planned task-condition blocks. Results below are paired by initial state; no partial shard is treated as complete.

## Per-task success and execution metrics

| suite | task | fresh | fo4 | full_old4 | reverse4 | fo8 | full_old8 | reverse8 | fo16 | full_old16 | reverse16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| libero_object | 3 | 9/10 (90.0%) | 10/10 (100.0%) | 9/10 (90.0%) | 9/10 (90.0%) | 7/10 (70.0%) | 9/10 (90.0%) | 9/10 (90.0%) | 9/10 (90.0%) | 9/10 (90.0%) | 5/10 (50.0%) |
| libero_object | 5 | 8/10 (80.0%) | 8/10 (80.0%) | 9/10 (90.0%) | 7/10 (70.0%) | 8/10 (80.0%) | 9/10 (90.0%) | 7/10 (70.0%) | 8/10 (80.0%) | 4/10 (40.0%) | 1/10 (10.0%) |
| libero_spatial | 0 | 6/10 (60.0%) | 7/10 (70.0%) | 7/10 (70.0%) | 7/10 (70.0%) | 7/10 (70.0%) | 7/10 (70.0%) | 7/10 (70.0%) | 8/10 (80.0%) | 8/10 (80.0%) | 5/10 (50.0%) |
| libero_spatial | 4 | 8/10 (80.0%) | 9/10 (90.0%) | 8/10 (80.0%) | 8/10 (80.0%) | 9/10 (90.0%) | 8/10 (80.0%) | 8/10 (80.0%) | 8/10 (80.0%) | 6/10 (60.0%) | 9/10 (90.0%) |
| libero_goal | 2 | 9/10 (90.0%) | 7/10 (70.0%) | 10/10 (100.0%) | 8/10 (80.0%) | 9/10 (90.0%) | 10/10 (100.0%) | 9/10 (90.0%) | 9/10 (90.0%) | 9/10 (90.0%) | 10/10 (100.0%) |
| libero_goal | 5 | 9/10 (90.0%) | 9/10 (90.0%) | 9/10 (90.0%) | 9/10 (90.0%) | 10/10 (100.0%) | 10/10 (100.0%) | 9/10 (90.0%) | 7/10 (70.0%) | 9/10 (90.0%) | 10/10 (100.0%) |
| libero_10 | 3 | 9/10 (90.0%) | 10/10 (100.0%) | 6/10 (60.0%) | 7/10 (70.0%) | 5/10 (50.0%) | 5/10 (50.0%) | 5/10 (50.0%) | 6/10 (60.0%) | 3/10 (30.0%) | 2/10 (20.0%) |
| libero_10 | 5 | 8/10 (80.0%) | 8/10 (80.0%) | 10/10 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) | 6/10 (60.0%) | 10/10 (100.0%) | 8/10 (80.0%) |

## Per-task execution provenance

| task | condition | queries/step | mean arm age | mean gripper age | success completion steps |
|---|---|---|---|---|---|
| libero_object:task3 | fresh | 1.000 | 0.00 | 0.00 | 135.7 |
| libero_object:task3 | fo4 | 1.000 | 0.00 | 3.88 | 131.2 |
| libero_object:task3 | full_old4 | 1.000 | 3.88 | 3.88 | 124.9 |
| libero_object:task3 | reverse4 | 1.000 | 3.89 | 0.00 | 137.6 |
| libero_object:task3 | fo8 | 1.000 | 0.00 | 7.58 | 127.7 |
| libero_object:task3 | full_old8 | 1.000 | 7.52 | 7.52 | 125.3 |
| libero_object:task3 | reverse8 | 1.000 | 7.57 | 0.00 | 153.8 |
| libero_object:task3 | fo16 | 1.000 | 0.00 | 14.17 | 133.6 |
| libero_object:task3 | full_old16 | 1.000 | 14.11 | 14.11 | 128.2 |
| libero_object:task3 | reverse16 | 1.000 | 14.50 | 0.00 | 123.4 |
| libero_object:task5 | fresh | 1.000 | 0.00 | 0.00 | 165.2 |
| libero_object:task5 | fo4 | 1.000 | 0.00 | 3.90 | 152.0 |
| libero_object:task5 | full_old4 | 1.000 | 3.90 | 3.90 | 149.9 |
| libero_object:task5 | reverse4 | 1.000 | 3.91 | 0.00 | 165.9 |
| libero_object:task5 | fo8 | 1.000 | 0.00 | 7.59 | 140.6 |
| libero_object:task5 | full_old8 | 1.000 | 7.62 | 7.62 | 169.6 |
| libero_object:task5 | reverse8 | 1.000 | 7.68 | 0.00 | 194.6 |
| libero_object:task5 | fo16 | 1.000 | 0.00 | 14.28 | 134.6 |
| libero_object:task5 | full_old16 | 1.000 | 14.62 | 14.62 | 124.2 |
| libero_object:task5 | reverse16 | 1.000 | 14.96 | 0.00 | 117.0 |
| libero_spatial:task0 | fresh | 1.000 | 0.00 | 0.00 | 74.7 |
| libero_spatial:task0 | fo4 | 1.000 | 0.00 | 3.84 | 79.6 |
| libero_spatial:task0 | full_old4 | 1.000 | 3.84 | 3.84 | 76.7 |
| libero_spatial:task0 | reverse4 | 1.000 | 3.84 | 0.00 | 90.4 |
| libero_spatial:task0 | fo8 | 1.000 | 0.00 | 7.32 | 73.6 |
| libero_spatial:task0 | full_old8 | 1.000 | 7.38 | 7.38 | 86.1 |
| libero_spatial:task0 | reverse8 | 1.000 | 7.34 | 0.00 | 75.7 |
| libero_spatial:task0 | fo16 | 1.000 | 0.00 | 13.26 | 93.1 |
| libero_spatial:task0 | full_old16 | 1.000 | 13.06 | 13.06 | 74.4 |
| libero_spatial:task0 | reverse16 | 1.000 | 13.84 | 0.00 | 75.0 |
| libero_spatial:task4 | fresh | 1.000 | 0.00 | 0.00 | 133.1 |
| libero_spatial:task4 | fo4 | 1.000 | 0.00 | 3.89 | 132.8 |
| libero_spatial:task4 | full_old4 | 1.000 | 3.89 | 3.89 | 129.5 |
| libero_spatial:task4 | reverse4 | 1.000 | 3.89 | 0.00 | 131.1 |
| libero_spatial:task4 | fo8 | 1.000 | 0.00 | 7.54 | 131.2 |
| libero_spatial:task4 | full_old8 | 1.000 | 7.57 | 7.57 | 131.8 |
| libero_spatial:task4 | reverse8 | 1.000 | 7.56 | 0.00 | 131.0 |
| libero_spatial:task4 | fo16 | 1.000 | 0.00 | 14.26 | 131.5 |
| libero_spatial:task4 | full_old16 | 1.000 | 14.45 | 14.45 | 130.3 |
| libero_spatial:task4 | reverse16 | 1.000 | 14.15 | 0.00 | 131.2 |
| libero_goal:task2 | fresh | 1.000 | 0.00 | 0.00 | 101.3 |
| libero_goal:task2 | fo4 | 1.000 | 0.00 | 3.87 | 95.7 |
| libero_goal:task2 | full_old4 | 1.000 | 3.83 | 3.83 | 96.2 |
| libero_goal:task2 | reverse4 | 1.000 | 3.85 | 0.00 | 90.9 |
| libero_goal:task2 | fo8 | 1.000 | 0.00 | 7.40 | 100.9 |
| libero_goal:task2 | full_old8 | 1.000 | 7.29 | 7.29 | 90.7 |
| libero_goal:task2 | reverse8 | 1.000 | 7.33 | 0.00 | 89.0 |
| libero_goal:task2 | fo16 | 1.000 | 0.00 | 13.56 | 98.1 |
| libero_goal:task2 | full_old16 | 1.000 | 13.41 | 13.41 | 92.1 |
| libero_goal:task2 | reverse16 | 1.000 | 13.18 | 0.00 | 91.0 |
| libero_goal:task5 | fresh | 1.000 | 0.00 | 0.00 | 143.6 |
| libero_goal:task5 | fo4 | 1.000 | 0.00 | 3.89 | 134.1 |
| libero_goal:task5 | full_old4 | 1.000 | 3.88 | 3.88 | 125.6 |
| libero_goal:task5 | reverse4 | 1.000 | 3.88 | 0.00 | 124.9 |
| libero_goal:task5 | fo8 | 1.000 | 0.00 | 7.56 | 151.1 |
| libero_goal:task5 | full_old8 | 1.000 | 7.50 | 7.50 | 128.6 |
| libero_goal:task5 | reverse8 | 1.000 | 7.55 | 0.00 | 144.3 |
| libero_goal:task5 | fo16 | 1.000 | 0.00 | 14.55 | 157.6 |
| libero_goal:task5 | full_old16 | 1.000 | 14.13 | 14.13 | 129.8 |
| libero_goal:task5 | reverse16 | 1.000 | 14.08 | 0.00 | 135.2 |
| libero_10:task3 | fresh | 1.000 | 0.00 | 0.00 | 259.1 |
| libero_10:task3 | fo4 | 1.000 | 0.00 | 3.93 | 230.3 |
| libero_10:task3 | full_old4 | 1.000 | 3.94 | 3.94 | 223.5 |
| libero_10:task3 | reverse4 | 1.000 | 3.94 | 0.00 | 243.9 |
| libero_10:task3 | fo8 | 1.000 | 0.00 | 7.80 | 223.6 |
| libero_10:task3 | full_old8 | 1.000 | 7.79 | 7.79 | 218.4 |
| libero_10:task3 | reverse8 | 1.000 | 7.84 | 0.00 | 354.0 |
| libero_10:task3 | fo16 | 1.000 | 0.00 | 15.11 | 222.3 |
| libero_10:task3 | full_old16 | 1.000 | 15.30 | 15.30 | 218.0 |
| libero_10:task3 | reverse16 | 1.000 | 15.38 | 0.00 | 222.0 |
| libero_10:task5 | fresh | 1.000 | 0.00 | 0.00 | 289.2 |
| libero_10:task5 | fo4 | 1.000 | 0.00 | 3.93 | 215.6 |
| libero_10:task5 | full_old4 | 1.000 | 3.92 | 3.92 | 203.7 |
| libero_10:task5 | reverse4 | 1.000 | 3.93 | 0.00 | 226.6 |
| libero_10:task5 | fo8 | 1.000 | 0.00 | 7.74 | 273.8 |
| libero_10:task5 | full_old8 | 1.000 | 7.67 | 7.67 | 198.7 |
| libero_10:task5 | reverse8 | 1.000 | 7.68 | 0.00 | 206.0 |
| libero_10:task5 | fo16 | 1.000 | 0.00 | 15.11 | 254.2 |
| libero_10:task5 | full_old16 | 1.000 | 14.62 | 14.62 | 188.2 |
| libero_10:task5 | reverse16 | 1.000 | 14.78 | 0.00 | 187.8 |

## Per-task core paired contrasts

Candidate minus reference; contingency is candidate-only/reference-only success. Exact McNemar values are Holm-adjusted across the 15 core comparisons within each task.

| task | age | contrast | success | C-only/R-only | delta | paired bootstrap CI | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|---|
| libero_object:task3 | 4 | FO - fresh | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_object:task3 | 4 | FO - full-old | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_object:task3 | 4 | FO - reverse | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_object:task3 | 4 | reverse - fresh | 9/10 vs 9/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_object:task3 | 4 | full-old - fresh | 9/10 vs 9/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_object:task3 | 8 | FO - fresh | 7/10 vs 9/10 | 1/3 | -0.200 | [-0.600, +0.200] | 0.625 | 1 |
| libero_object:task3 | 8 | FO - full-old | 7/10 vs 9/10 | 1/3 | -0.200 | [-0.600, +0.200] | 0.625 | 1 |
| libero_object:task3 | 8 | FO - reverse | 7/10 vs 9/10 | 1/3 | -0.200 | [-0.600, +0.200] | 0.625 | 1 |
| libero_object:task3 | 8 | reverse - fresh | 9/10 vs 9/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_object:task3 | 8 | full-old - fresh | 9/10 vs 9/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_object:task3 | 16 | FO - fresh | 9/10 vs 9/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_object:task3 | 16 | FO - full-old | 9/10 vs 9/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_object:task3 | 16 | FO - reverse | 9/10 vs 5/10 | 5/1 | +0.400 | [+0.000, +0.800] | 0.2188 | 1 |
| libero_object:task3 | 16 | reverse - fresh | 5/10 vs 9/10 | 1/5 | -0.400 | [-0.800, +0.000] | 0.2188 | 1 |
| libero_object:task3 | 16 | full-old - fresh | 9/10 vs 9/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_object:task5 | 4 | FO - fresh | 8/10 vs 8/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_object:task5 | 4 | FO - full-old | 8/10 vs 9/10 | 0/1 | -0.100 | [-0.300, +0.000] | 1 | 1 |
| libero_object:task5 | 4 | FO - reverse | 8/10 vs 7/10 | 3/2 | +0.100 | [-0.300, +0.500] | 1 | 1 |
| libero_object:task5 | 4 | reverse - fresh | 7/10 vs 8/10 | 2/3 | -0.100 | [-0.500, +0.300] | 1 | 1 |
| libero_object:task5 | 4 | full-old - fresh | 9/10 vs 8/10 | 2/1 | +0.100 | [-0.200, +0.400] | 1 | 1 |
| libero_object:task5 | 8 | FO - fresh | 8/10 vs 8/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_object:task5 | 8 | FO - full-old | 8/10 vs 9/10 | 0/1 | -0.100 | [-0.300, +0.000] | 1 | 1 |
| libero_object:task5 | 8 | FO - reverse | 8/10 vs 7/10 | 2/1 | +0.100 | [-0.200, +0.400] | 1 | 1 |
| libero_object:task5 | 8 | reverse - fresh | 7/10 vs 8/10 | 1/2 | -0.100 | [-0.400, +0.200] | 1 | 1 |
| libero_object:task5 | 8 | full-old - fresh | 9/10 vs 8/10 | 2/1 | +0.100 | [-0.200, +0.400] | 1 | 1 |
| libero_object:task5 | 16 | FO - fresh | 8/10 vs 8/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_object:task5 | 16 | FO - full-old | 8/10 vs 4/10 | 5/1 | +0.400 | [+0.000, +0.800] | 0.2188 | 1 |
| libero_object:task5 | 16 | FO - reverse | 8/10 vs 1/10 | 8/1 | +0.700 | [+0.300, +1.000] | 0.03906 | 0.5859 |
| libero_object:task5 | 16 | reverse - fresh | 1/10 vs 8/10 | 1/8 | -0.700 | [-1.000, -0.300] | 0.03906 | 0.5859 |
| libero_object:task5 | 16 | full-old - fresh | 4/10 vs 8/10 | 1/5 | -0.400 | [-0.800, +0.000] | 0.2188 | 1 |
| libero_spatial:task0 | 4 | FO - fresh | 7/10 vs 6/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_spatial:task0 | 4 | FO - full-old | 7/10 vs 7/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_spatial:task0 | 4 | FO - reverse | 7/10 vs 7/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_spatial:task0 | 4 | reverse - fresh | 7/10 vs 6/10 | 2/1 | +0.100 | [-0.200, +0.400] | 1 | 1 |
| libero_spatial:task0 | 4 | full-old - fresh | 7/10 vs 6/10 | 2/1 | +0.100 | [-0.200, +0.400] | 1 | 1 |
| libero_spatial:task0 | 8 | FO - fresh | 7/10 vs 6/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_spatial:task0 | 8 | FO - full-old | 7/10 vs 7/10 | 2/2 | +0.000 | [-0.400, +0.400] | 1 | 1 |
| libero_spatial:task0 | 8 | FO - reverse | 7/10 vs 7/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_spatial:task0 | 8 | reverse - fresh | 7/10 vs 6/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_spatial:task0 | 8 | full-old - fresh | 7/10 vs 6/10 | 3/2 | +0.100 | [-0.300, +0.500] | 1 | 1 |
| libero_spatial:task0 | 16 | FO - fresh | 8/10 vs 6/10 | 4/2 | +0.200 | [-0.300, +0.600] | 0.6875 | 1 |
| libero_spatial:task0 | 16 | FO - full-old | 8/10 vs 8/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_spatial:task0 | 16 | FO - reverse | 8/10 vs 5/10 | 3/0 | +0.300 | [+0.000, +0.600] | 0.25 | 1 |
| libero_spatial:task0 | 16 | reverse - fresh | 5/10 vs 6/10 | 2/3 | -0.100 | [-0.500, +0.300] | 1 | 1 |
| libero_spatial:task0 | 16 | full-old - fresh | 8/10 vs 6/10 | 3/1 | +0.200 | [-0.200, +0.600] | 0.625 | 1 |
| libero_spatial:task4 | 4 | FO - fresh | 9/10 vs 8/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_spatial:task4 | 4 | FO - full-old | 9/10 vs 8/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_spatial:task4 | 4 | FO - reverse | 9/10 vs 8/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_spatial:task4 | 4 | reverse - fresh | 8/10 vs 8/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_spatial:task4 | 4 | full-old - fresh | 8/10 vs 8/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_spatial:task4 | 8 | FO - fresh | 9/10 vs 8/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_spatial:task4 | 8 | FO - full-old | 9/10 vs 8/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_spatial:task4 | 8 | FO - reverse | 9/10 vs 8/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_spatial:task4 | 8 | reverse - fresh | 8/10 vs 8/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_spatial:task4 | 8 | full-old - fresh | 8/10 vs 8/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_spatial:task4 | 16 | FO - fresh | 8/10 vs 8/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_spatial:task4 | 16 | FO - full-old | 8/10 vs 6/10 | 3/1 | +0.200 | [-0.200, +0.600] | 0.625 | 1 |
| libero_spatial:task4 | 16 | FO - reverse | 8/10 vs 9/10 | 1/2 | -0.100 | [-0.400, +0.200] | 1 | 1 |
| libero_spatial:task4 | 16 | reverse - fresh | 9/10 vs 8/10 | 2/1 | +0.100 | [-0.200, +0.400] | 1 | 1 |
| libero_spatial:task4 | 16 | full-old - fresh | 6/10 vs 8/10 | 0/2 | -0.200 | [-0.500, +0.000] | 0.5 | 1 |
| libero_goal:task2 | 4 | FO - fresh | 7/10 vs 9/10 | 1/3 | -0.200 | [-0.600, +0.200] | 0.625 | 1 |
| libero_goal:task2 | 4 | FO - full-old | 7/10 vs 10/10 | 0/3 | -0.300 | [-0.600, +0.000] | 0.25 | 1 |
| libero_goal:task2 | 4 | FO - reverse | 7/10 vs 8/10 | 2/3 | -0.100 | [-0.500, +0.300] | 1 | 1 |
| libero_goal:task2 | 4 | reverse - fresh | 8/10 vs 9/10 | 1/2 | -0.100 | [-0.400, +0.200] | 1 | 1 |
| libero_goal:task2 | 4 | full-old - fresh | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_goal:task2 | 8 | FO - fresh | 9/10 vs 9/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_goal:task2 | 8 | FO - full-old | 9/10 vs 10/10 | 0/1 | -0.100 | [-0.300, +0.000] | 1 | 1 |
| libero_goal:task2 | 8 | FO - reverse | 9/10 vs 9/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_goal:task2 | 8 | reverse - fresh | 9/10 vs 9/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_goal:task2 | 8 | full-old - fresh | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_goal:task2 | 16 | FO - fresh | 9/10 vs 9/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_goal:task2 | 16 | FO - full-old | 9/10 vs 9/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_goal:task2 | 16 | FO - reverse | 9/10 vs 10/10 | 0/1 | -0.100 | [-0.300, +0.000] | 1 | 1 |
| libero_goal:task2 | 16 | reverse - fresh | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_goal:task2 | 16 | full-old - fresh | 9/10 vs 9/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_goal:task5 | 4 | FO - fresh | 9/10 vs 9/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_goal:task5 | 4 | FO - full-old | 9/10 vs 9/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_goal:task5 | 4 | FO - reverse | 9/10 vs 9/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_goal:task5 | 4 | reverse - fresh | 9/10 vs 9/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_goal:task5 | 4 | full-old - fresh | 9/10 vs 9/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_goal:task5 | 8 | FO - fresh | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_goal:task5 | 8 | FO - full-old | 10/10 vs 10/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_goal:task5 | 8 | FO - reverse | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_goal:task5 | 8 | reverse - fresh | 9/10 vs 9/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_goal:task5 | 8 | full-old - fresh | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_goal:task5 | 16 | FO - fresh | 7/10 vs 9/10 | 1/3 | -0.200 | [-0.600, +0.200] | 0.625 | 1 |
| libero_goal:task5 | 16 | FO - full-old | 7/10 vs 9/10 | 1/3 | -0.200 | [-0.600, +0.200] | 0.625 | 1 |
| libero_goal:task5 | 16 | FO - reverse | 7/10 vs 10/10 | 0/3 | -0.300 | [-0.600, +0.000] | 0.25 | 1 |
| libero_goal:task5 | 16 | reverse - fresh | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_goal:task5 | 16 | full-old - fresh | 9/10 vs 9/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_10:task3 | 4 | FO - fresh | 10/10 vs 9/10 | 1/0 | +0.100 | [+0.000, +0.300] | 1 | 1 |
| libero_10:task3 | 4 | FO - full-old | 10/10 vs 6/10 | 4/0 | +0.400 | [+0.100, +0.700] | 0.125 | 1 |
| libero_10:task3 | 4 | FO - reverse | 10/10 vs 7/10 | 3/0 | +0.300 | [+0.000, +0.600] | 0.25 | 1 |
| libero_10:task3 | 4 | reverse - fresh | 7/10 vs 9/10 | 1/3 | -0.200 | [-0.600, +0.200] | 0.625 | 1 |
| libero_10:task3 | 4 | full-old - fresh | 6/10 vs 9/10 | 1/4 | -0.300 | [-0.700, +0.100] | 0.375 | 1 |
| libero_10:task3 | 8 | FO - fresh | 5/10 vs 9/10 | 0/4 | -0.400 | [-0.700, -0.100] | 0.125 | 1 |
| libero_10:task3 | 8 | FO - full-old | 5/10 vs 5/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_10:task3 | 8 | FO - reverse | 5/10 vs 5/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_10:task3 | 8 | reverse - fresh | 5/10 vs 9/10 | 0/4 | -0.400 | [-0.700, -0.100] | 0.125 | 1 |
| libero_10:task3 | 8 | full-old - fresh | 5/10 vs 9/10 | 0/4 | -0.400 | [-0.700, -0.100] | 0.125 | 1 |
| libero_10:task3 | 16 | FO - fresh | 6/10 vs 9/10 | 1/4 | -0.300 | [-0.700, +0.100] | 0.375 | 1 |
| libero_10:task3 | 16 | FO - full-old | 6/10 vs 3/10 | 5/2 | +0.300 | [-0.200, +0.800] | 0.4531 | 1 |
| libero_10:task3 | 16 | FO - reverse | 6/10 vs 2/10 | 5/1 | +0.400 | [+0.000, +0.800] | 0.2188 | 1 |
| libero_10:task3 | 16 | reverse - fresh | 2/10 vs 9/10 | 0/7 | -0.700 | [-1.000, -0.400] | 0.01562 | 0.2344 |
| libero_10:task3 | 16 | full-old - fresh | 3/10 vs 9/10 | 0/6 | -0.600 | [-0.900, -0.300] | 0.03125 | 0.4375 |
| libero_10:task5 | 4 | FO - fresh | 8/10 vs 8/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_10:task5 | 4 | FO - full-old | 8/10 vs 10/10 | 0/2 | -0.200 | [-0.500, +0.000] | 0.5 | 1 |
| libero_10:task5 | 4 | FO - reverse | 8/10 vs 10/10 | 0/2 | -0.200 | [-0.500, +0.000] | 0.5 | 1 |
| libero_10:task5 | 4 | reverse - fresh | 10/10 vs 8/10 | 2/0 | +0.200 | [+0.000, +0.500] | 0.5 | 1 |
| libero_10:task5 | 4 | full-old - fresh | 10/10 vs 8/10 | 2/0 | +0.200 | [+0.000, +0.500] | 0.5 | 1 |
| libero_10:task5 | 8 | FO - fresh | 10/10 vs 8/10 | 2/0 | +0.200 | [+0.000, +0.500] | 0.5 | 1 |
| libero_10:task5 | 8 | FO - full-old | 10/10 vs 10/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_10:task5 | 8 | FO - reverse | 10/10 vs 10/10 | 0/0 | +0.000 | [+0.000, +0.000] | n/a | n/a |
| libero_10:task5 | 8 | reverse - fresh | 10/10 vs 8/10 | 2/0 | +0.200 | [+0.000, +0.500] | 0.5 | 1 |
| libero_10:task5 | 8 | full-old - fresh | 10/10 vs 8/10 | 2/0 | +0.200 | [+0.000, +0.500] | 0.5 | 1 |
| libero_10:task5 | 16 | FO - fresh | 6/10 vs 8/10 | 1/3 | -0.200 | [-0.600, +0.200] | 0.625 | 1 |
| libero_10:task5 | 16 | FO - full-old | 6/10 vs 10/10 | 0/4 | -0.400 | [-0.700, -0.100] | 0.125 | 1 |
| libero_10:task5 | 16 | FO - reverse | 6/10 vs 8/10 | 1/3 | -0.200 | [-0.600, +0.200] | 0.625 | 1 |
| libero_10:task5 | 16 | reverse - fresh | 8/10 vs 8/10 | 1/1 | +0.000 | [-0.300, +0.300] | 1 | 1 |
| libero_10:task5 | 16 | full-old - fresh | 10/10 vs 8/10 | 2/0 | +0.200 | [+0.000, +0.500] | 0.5 | 1 |

## libero_object: task-macro and pooled descriptive summaries

| condition | task-macro success | pooled success | queries/step | arm age | gripper age | success completion steps |
|---|---|---|---|---|---|---|
| fresh | 0.850 | 17/20 (85.0%) | 1.000 | 0.00 | 0.00 | 149.6 |
| fo4 | 0.900 | 18/20 (90.0%) | 1.000 | 0.00 | 3.89 | 140.4 |
| full_old4 | 0.900 | 18/20 (90.0%) | 1.000 | 3.89 | 3.89 | 137.4 |
| reverse4 | 0.800 | 16/20 (80.0%) | 1.000 | 3.90 | 0.00 | 149.9 |
| fo8 | 0.750 | 15/20 (75.0%) | 1.000 | 0.00 | 7.58 | 134.6 |
| full_old8 | 0.900 | 18/20 (90.0%) | 1.000 | 7.57 | 7.57 | 147.4 |
| reverse8 | 0.800 | 16/20 (80.0%) | 1.000 | 7.62 | 0.00 | 171.6 |
| fo16 | 0.850 | 17/20 (85.0%) | 1.000 | 0.00 | 14.23 | 134.1 |
| full_old16 | 0.650 | 13/20 (65.0%) | 1.000 | 14.37 | 14.37 | 127.0 |
| reverse16 | 0.300 | 6/20 (30.0%) | 1.000 | 14.73 | 0.00 | 122.3 |

Task macro is the primary presentation for heterogeneous tasks. Pooled numbers are descriptive only.

| age | contrast | task-macro delta | task bootstrap CI | pooled delta | C-only/R-only | pooled McNemar p | pooled Holm p |
|---|---|---|---|---|---|---|---|
| 4 | FO - fresh | +0.050 | [+0.000, +0.100] | +0.050 | 2/1 | 1 | 1 |
| 4 | FO - full-old | +0.000 | [-0.100, +0.100] | +0.000 | 1/1 | 1 | 1 |
| 4 | FO - reverse | +0.100 | [+0.100, +0.100] | +0.100 | 4/2 | 0.6875 | 1 |
| 4 | reverse - fresh | -0.050 | [-0.100, +0.000] | -0.050 | 2/3 | 1 | 1 |
| 4 | full-old - fresh | +0.050 | [+0.000, +0.100] | +0.050 | 3/2 | 1 | 1 |
| 8 | FO - fresh | -0.100 | [-0.200, +0.000] | -0.100 | 2/4 | 0.6875 | 1 |
| 8 | FO - full-old | -0.150 | [-0.200, -0.100] | -0.150 | 1/4 | 0.375 | 1 |
| 8 | FO - reverse | -0.050 | [-0.200, +0.100] | -0.050 | 3/4 | 1 | 1 |
| 8 | reverse - fresh | -0.050 | [-0.100, +0.000] | -0.050 | 1/2 | 1 | 1 |
| 8 | full-old - fresh | +0.050 | [+0.000, +0.100] | +0.050 | 2/1 | 1 | 1 |
| 16 | FO - fresh | +0.000 | [+0.000, +0.000] | +0.000 | 1/1 | 1 | 1 |
| 16 | FO - full-old | +0.200 | [+0.000, +0.400] | +0.200 | 5/1 | 0.2188 | 1 |
| 16 | FO - reverse | +0.550 | [+0.400, +0.700] | +0.550 | 13/2 | 0.007385 | 0.1108 |
| 16 | reverse - fresh | -0.550 | [-0.700, -0.400] | -0.550 | 2/13 | 0.007385 | 0.1108 |
| 16 | full-old - fresh | -0.200 | [-0.400, +0.000] | -0.200 | 1/5 | 0.2188 | 1 |

## libero_spatial: task-macro and pooled descriptive summaries

| condition | task-macro success | pooled success | queries/step | arm age | gripper age | success completion steps |
|---|---|---|---|---|---|---|
| fresh | 0.700 | 14/20 (70.0%) | 1.000 | 0.00 | 0.00 | 108.1 |
| fo4 | 0.800 | 16/20 (80.0%) | 1.000 | 0.00 | 3.86 | 109.5 |
| full_old4 | 0.750 | 15/20 (75.0%) | 1.000 | 3.86 | 3.86 | 104.9 |
| reverse4 | 0.750 | 15/20 (75.0%) | 1.000 | 3.87 | 0.00 | 112.1 |
| fo8 | 0.800 | 16/20 (80.0%) | 1.000 | 0.00 | 7.43 | 106.0 |
| full_old8 | 0.750 | 15/20 (75.0%) | 1.000 | 7.47 | 7.47 | 110.5 |
| reverse8 | 0.750 | 15/20 (75.0%) | 1.000 | 7.45 | 0.00 | 105.2 |
| fo16 | 0.800 | 16/20 (80.0%) | 1.000 | 0.00 | 13.76 | 112.3 |
| full_old16 | 0.700 | 14/20 (70.0%) | 1.000 | 13.76 | 13.76 | 98.4 |
| reverse16 | 0.700 | 14/20 (70.0%) | 1.000 | 13.99 | 0.00 | 111.1 |

Task macro is the primary presentation for heterogeneous tasks. Pooled numbers are descriptive only.

| age | contrast | task-macro delta | task bootstrap CI | pooled delta | C-only/R-only | pooled McNemar p | pooled Holm p |
|---|---|---|---|---|---|---|---|
| 4 | FO - fresh | +0.100 | [+0.100, +0.100] | +0.100 | 2/0 | 0.5 | 1 |
| 4 | FO - full-old | +0.050 | [+0.000, +0.100] | +0.050 | 2/1 | 1 | 1 |
| 4 | FO - reverse | +0.050 | [+0.000, +0.100] | +0.050 | 2/1 | 1 | 1 |
| 4 | reverse - fresh | +0.050 | [+0.000, +0.100] | +0.050 | 2/1 | 1 | 1 |
| 4 | full-old - fresh | +0.050 | [+0.000, +0.100] | +0.050 | 3/2 | 1 | 1 |
| 8 | FO - fresh | +0.100 | [+0.100, +0.100] | +0.100 | 2/0 | 0.5 | 1 |
| 8 | FO - full-old | +0.050 | [+0.000, +0.100] | +0.050 | 3/2 | 1 | 1 |
| 8 | FO - reverse | +0.050 | [+0.000, +0.100] | +0.050 | 1/0 | 1 | 1 |
| 8 | reverse - fresh | +0.050 | [+0.000, +0.100] | +0.050 | 2/1 | 1 | 1 |
| 8 | full-old - fresh | +0.050 | [+0.000, +0.100] | +0.050 | 4/3 | 1 | 1 |
| 16 | FO - fresh | +0.100 | [+0.000, +0.200] | +0.100 | 5/3 | 0.7266 | 1 |
| 16 | FO - full-old | +0.100 | [+0.000, +0.200] | +0.100 | 4/2 | 0.6875 | 1 |
| 16 | FO - reverse | +0.100 | [-0.100, +0.300] | +0.100 | 4/2 | 0.6875 | 1 |
| 16 | reverse - fresh | +0.000 | [-0.100, +0.100] | +0.000 | 4/4 | 1 | 1 |
| 16 | full-old - fresh | +0.000 | [-0.200, +0.200] | +0.000 | 3/3 | 1 | 1 |

## libero_goal: task-macro and pooled descriptive summaries

| condition | task-macro success | pooled success | queries/step | arm age | gripper age | success completion steps |
|---|---|---|---|---|---|---|
| fresh | 0.900 | 18/20 (90.0%) | 1.000 | 0.00 | 0.00 | 122.4 |
| fo4 | 0.800 | 16/20 (80.0%) | 1.000 | 0.00 | 3.88 | 117.3 |
| full_old4 | 0.950 | 19/20 (95.0%) | 1.000 | 3.86 | 3.86 | 110.1 |
| reverse4 | 0.850 | 17/20 (85.0%) | 1.000 | 3.86 | 0.00 | 108.9 |
| fo8 | 0.950 | 19/20 (95.0%) | 1.000 | 0.00 | 7.48 | 127.3 |
| full_old8 | 1.000 | 20/20 (100.0%) | 1.000 | 7.40 | 7.40 | 109.7 |
| reverse8 | 0.900 | 18/20 (90.0%) | 1.000 | 7.44 | 0.00 | 116.7 |
| fo16 | 0.800 | 16/20 (80.0%) | 1.000 | 0.00 | 14.05 | 124.1 |
| full_old16 | 0.900 | 18/20 (90.0%) | 1.000 | 13.77 | 13.77 | 110.9 |
| reverse16 | 1.000 | 20/20 (100.0%) | 1.000 | 13.63 | 0.00 | 113.1 |

Task macro is the primary presentation for heterogeneous tasks. Pooled numbers are descriptive only.

| age | contrast | task-macro delta | task bootstrap CI | pooled delta | C-only/R-only | pooled McNemar p | pooled Holm p |
|---|---|---|---|---|---|---|---|
| 4 | FO - fresh | -0.100 | [-0.200, +0.000] | -0.100 | 1/3 | 0.625 | 1 |
| 4 | FO - full-old | -0.150 | [-0.300, +0.000] | -0.150 | 1/4 | 0.375 | 1 |
| 4 | FO - reverse | -0.050 | [-0.100, +0.000] | -0.050 | 3/4 | 1 | 1 |
| 4 | reverse - fresh | -0.050 | [-0.100, +0.000] | -0.050 | 2/3 | 1 | 1 |
| 4 | full-old - fresh | +0.050 | [+0.000, +0.100] | +0.050 | 2/1 | 1 | 1 |
| 8 | FO - fresh | +0.050 | [+0.000, +0.100] | +0.050 | 1/0 | 1 | 1 |
| 8 | FO - full-old | -0.050 | [-0.100, +0.000] | -0.050 | 0/1 | 1 | 1 |
| 8 | FO - reverse | +0.050 | [+0.000, +0.100] | +0.050 | 2/1 | 1 | 1 |
| 8 | reverse - fresh | +0.000 | [+0.000, +0.000] | +0.000 | 2/2 | 1 | 1 |
| 8 | full-old - fresh | +0.100 | [+0.100, +0.100] | +0.100 | 2/0 | 0.5 | 1 |
| 16 | FO - fresh | -0.100 | [-0.200, +0.000] | -0.100 | 2/4 | 0.6875 | 1 |
| 16 | FO - full-old | -0.100 | [-0.200, +0.000] | -0.100 | 1/3 | 0.625 | 1 |
| 16 | FO - reverse | -0.200 | [-0.300, -0.100] | -0.200 | 0/4 | 0.125 | 1 |
| 16 | reverse - fresh | +0.100 | [+0.100, +0.100] | +0.100 | 2/0 | 0.5 | 1 |
| 16 | full-old - fresh | +0.000 | [+0.000, +0.000] | +0.000 | 1/1 | 1 | 1 |

## libero_10: task-macro and pooled descriptive summaries

| condition | task-macro success | pooled success | queries/step | arm age | gripper age | success completion steps |
|---|---|---|---|---|---|---|
| fresh | 0.850 | 17/20 (85.0%) | 1.000 | 0.00 | 0.00 | 273.3 |
| fo4 | 0.900 | 18/20 (90.0%) | 1.000 | 0.00 | 3.93 | 223.8 |
| full_old4 | 0.800 | 16/20 (80.0%) | 1.000 | 3.93 | 3.93 | 211.1 |
| reverse4 | 0.850 | 17/20 (85.0%) | 1.000 | 3.94 | 0.00 | 233.7 |
| fo8 | 0.750 | 15/20 (75.0%) | 1.000 | 0.00 | 7.77 | 257.1 |
| full_old8 | 0.750 | 15/20 (75.0%) | 1.000 | 7.73 | 7.73 | 205.3 |
| reverse8 | 0.750 | 15/20 (75.0%) | 1.000 | 7.76 | 0.00 | 255.3 |
| fo16 | 0.600 | 12/20 (60.0%) | 1.000 | 0.00 | 15.11 | 238.2 |
| full_old16 | 0.650 | 13/20 (65.0%) | 1.000 | 14.96 | 14.96 | 195.1 |
| reverse16 | 0.500 | 10/20 (50.0%) | 1.000 | 15.08 | 0.00 | 194.6 |

Task macro is the primary presentation for heterogeneous tasks. Pooled numbers are descriptive only.

| age | contrast | task-macro delta | task bootstrap CI | pooled delta | C-only/R-only | pooled McNemar p | pooled Holm p |
|---|---|---|---|---|---|---|---|
| 4 | FO - fresh | +0.050 | [+0.000, +0.100] | +0.050 | 2/1 | 1 | 1 |
| 4 | FO - full-old | +0.100 | [-0.200, +0.400] | +0.100 | 4/2 | 0.6875 | 1 |
| 4 | FO - reverse | +0.050 | [-0.200, +0.300] | +0.050 | 3/2 | 1 | 1 |
| 4 | reverse - fresh | +0.000 | [-0.200, +0.200] | +0.000 | 3/3 | 1 | 1 |
| 4 | full-old - fresh | -0.050 | [-0.300, +0.200] | -0.050 | 3/4 | 1 | 1 |
| 8 | FO - fresh | -0.100 | [-0.400, +0.200] | -0.100 | 2/4 | 0.6875 | 1 |
| 8 | FO - full-old | +0.000 | [+0.000, +0.000] | +0.000 | 1/1 | 1 | 1 |
| 8 | FO - reverse | +0.000 | [+0.000, +0.000] | +0.000 | 1/1 | 1 | 1 |
| 8 | reverse - fresh | -0.100 | [-0.400, +0.200] | -0.100 | 2/4 | 0.6875 | 1 |
| 8 | full-old - fresh | -0.100 | [-0.400, +0.200] | -0.100 | 2/4 | 0.6875 | 1 |
| 16 | FO - fresh | -0.250 | [-0.300, -0.200] | -0.250 | 2/7 | 0.1797 | 1 |
| 16 | FO - full-old | -0.050 | [-0.400, +0.300] | -0.050 | 5/6 | 1 | 1 |
| 16 | FO - reverse | +0.100 | [-0.200, +0.400] | +0.100 | 6/4 | 0.7539 | 1 |
| 16 | reverse - fresh | -0.350 | [-0.700, +0.000] | -0.350 | 1/8 | 0.03906 | 0.5859 |
| 16 | full-old - fresh | -0.200 | [-0.600, +0.200] | -0.200 | 2/6 | 0.2891 | 1 |

## all_tasks: task-macro and pooled descriptive summaries

| condition | task-macro success | pooled success | queries/step | arm age | gripper age | success completion steps |
|---|---|---|---|---|---|---|
| fresh | 0.825 | 66/80 (82.5%) | 1.000 | 0.00 | 0.00 | 165.2 |
| fo4 | 0.850 | 68/80 (85.0%) | 1.000 | 0.00 | 3.89 | 149.8 |
| full_old4 | 0.850 | 68/80 (85.0%) | 1.000 | 3.88 | 3.88 | 139.9 |
| reverse4 | 0.812 | 65/80 (81.2%) | 1.000 | 3.89 | 0.00 | 152.4 |
| fo8 | 0.812 | 65/80 (81.2%) | 1.000 | 0.00 | 7.57 | 153.7 |
| full_old8 | 0.850 | 68/80 (85.0%) | 1.000 | 7.54 | 7.54 | 140.9 |
| reverse8 | 0.800 | 64/80 (80.0%) | 1.000 | 7.57 | 0.00 | 160.2 |
| fo16 | 0.762 | 61/80 (76.2%) | 1.000 | 0.00 | 14.29 | 146.2 |
| full_old16 | 0.725 | 58/80 (72.5%) | 1.000 | 14.21 | 14.21 | 130.4 |
| reverse16 | 0.625 | 50/80 (62.5%) | 1.000 | 14.36 | 0.00 | 130.0 |

Task macro is the primary presentation for heterogeneous tasks. Pooled numbers are descriptive only.

| age | contrast | task-macro delta | task bootstrap CI | pooled delta | C-only/R-only | pooled McNemar p | pooled Holm p |
|---|---|---|---|---|---|---|---|
| 4 | FO - fresh | +0.025 | [-0.050, +0.088] | +0.025 | 7/5 | 0.7744 | 1 |
| 4 | FO - full-old | +0.000 | [-0.137, +0.138] | +0.000 | 8/8 | 1 | 1 |
| 4 | FO - reverse | +0.037 | [-0.062, +0.138] | +0.037 | 12/9 | 0.6636 | 1 |
| 4 | reverse - fresh | -0.013 | [-0.088, +0.075] | -0.013 | 9/10 | 1 | 1 |
| 4 | full-old - fresh | +0.025 | [-0.087, +0.113] | +0.025 | 11/9 | 0.8238 | 1 |
| 8 | FO - fresh | -0.013 | [-0.150, +0.100] | -0.013 | 7/8 | 1 | 1 |
| 8 | FO - full-old | -0.038 | [-0.100, +0.025] | -0.037 | 5/8 | 0.5811 | 1 |
| 8 | FO - reverse | +0.013 | [-0.050, +0.062] | +0.013 | 7/6 | 1 | 1 |
| 8 | reverse - fresh | -0.025 | [-0.150, +0.075] | -0.025 | 7/9 | 0.8036 | 1 |
| 8 | full-old - fresh | +0.025 | [-0.113, +0.113] | +0.025 | 10/8 | 0.8145 | 1 |
| 16 | FO - fresh | -0.062 | [-0.163, +0.038] | -0.062 | 10/15 | 0.4244 | 1 |
| 16 | FO - full-old | +0.037 | [-0.138, +0.200] | +0.037 | 15/12 | 0.7011 | 1 |
| 16 | FO - reverse | +0.138 | [-0.088, +0.375] | +0.138 | 23/12 | 0.08953 | 1 |
| 16 | reverse - fresh | -0.200 | [-0.438, +0.013] | -0.200 | 9/25 | 0.009041 | 0.1356 |
| 16 | full-old - fresh | -0.100 | [-0.300, +0.075] | -0.100 | 7/15 | 0.1338 | 1 |

## Prespecified sensitivity interpretation

For each age, compare FO minus fresh (fresh arm, stale gripper) with reverse minus fresh (stale arm, fresh gripper). The direction and uncertainty of those two task-macro contrasts answer the temporal-sensitivity question; no monotonicity is assumed.

Figures are saved under `figures/`: `arm_vs_gripper_age_sensitivity.png`, `per_task_source_age_deltas.png`, and `success_vs_source_age.png`. If a disagreement JSON is supplied, a fourth outcome-association plot is added.

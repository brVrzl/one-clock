# Temporal aggregation follow-up

Fixed coefficients, frozen eight-task cohort, paired seeds 1000–1009. Task-macro values are primary; pooled values are descriptive.

## Per-task success

| task | fresh | official_act_m001 | physical_exp_beta003 | cogact_alpha03 | component_arm_fresh_gripper_act | component vs ACT |
|---|---|---|---|---|---|---|
| libero_10:task3 | 8/10 | 1/10 | 4/10 | 5/10 | 5/10 | 5-1 (+0.40) |
| libero_10:task5 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10-10 (+0.00) |
| libero_goal:task2 | 10/10 | 7/10 | 8/10 | 9/10 | 9/10 | 9-7 (+0.20) |
| libero_goal:task5 | 9/10 | 10/10 | 10/10 | 9/10 | 10/10 | 10-10 (+0.00) |
| libero_object:task3 | 9/10 | 10/10 | 10/10 | 10/10 | 9/10 | 9-10 (-0.10) |
| libero_object:task5 | 9/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10-10 (+0.00) |
| libero_spatial:task0 | 9/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10-10 (+0.00) |
| libero_spatial:task4 | 8/10 | 8/10 | 9/10 | 10/10 | 10/10 | 10-8 (+0.20) |

## libero_object

| method | task macro | pooled | queries/step | arm age | gripper age |
|---|---:|---:|---:|---:|---:|
| fresh | 0.900 | 18/20 (90.0%) | 1.000 | 0.00 | 0.00 |
| official_act_m001 | 1.000 | 20/20 (100.0%) | 1.000 | 21.16 | 21.16 |
| physical_exp_beta003 | 1.000 | 20/20 (100.0%) | 1.000 | 15.18 | 15.18 |
| cogact_alpha03 | 1.000 | 20/20 (100.0%) | 1.000 | 19.45 | 19.45 |
| component_arm_fresh_gripper_act | 0.950 | 19/20 (95.0%) | 1.000 | 0.00 | 21.50 |

### Paired comparisons vs fresh

| method | task-macro delta | pooled delta | candidate-only/reference-only | p |
|---|---:|---:|---:|---:|
| official_act_m001 | +0.100 | +0.100 | 2/0 | 0.5 |
| physical_exp_beta003 | +0.100 | +0.100 | 2/0 | 0.5 |
| cogact_alpha03 | +0.100 | +0.100 | 2/0 | 0.5 |
| component_arm_fresh_gripper_act | +0.050 | +0.050 | 2/1 | 1.0 |

Component-aware minus official full-action ACT: task-macro -0.050; pooled -0.050; candidate-only/reference-only 0/1; exact McNemar p=1.0.

## libero_spatial

| method | task macro | pooled | queries/step | arm age | gripper age |
|---|---:|---:|---:|---:|---:|
| fresh | 0.850 | 17/20 (85.0%) | 1.000 | 0.00 | 0.00 |
| official_act_m001 | 0.900 | 18/20 (90.0%) | 1.000 | 19.69 | 19.69 |
| physical_exp_beta003 | 0.950 | 19/20 (95.0%) | 1.000 | 14.21 | 14.21 |
| cogact_alpha03 | 1.000 | 20/20 (100.0%) | 1.000 | 18.01 | 18.01 |
| component_arm_fresh_gripper_act | 1.000 | 20/20 (100.0%) | 1.000 | 0.00 | 19.38 |

### Paired comparisons vs fresh

| method | task-macro delta | pooled delta | candidate-only/reference-only | p |
|---|---:|---:|---:|---:|
| official_act_m001 | +0.050 | +0.050 | 2/1 | 1.0 |
| physical_exp_beta003 | +0.100 | +0.100 | 2/0 | 0.5 |
| cogact_alpha03 | +0.150 | +0.150 | 3/0 | 0.25 |
| component_arm_fresh_gripper_act | +0.150 | +0.150 | 3/0 | 0.25 |

Component-aware minus official full-action ACT: task-macro +0.100; pooled +0.100; candidate-only/reference-only 2/0; exact McNemar p=0.5.

## libero_goal

| method | task macro | pooled | queries/step | arm age | gripper age |
|---|---:|---:|---:|---:|---:|
| fresh | 0.950 | 19/20 (95.0%) | 1.000 | 0.00 | 0.00 |
| official_act_m001 | 0.850 | 17/20 (85.0%) | 1.000 | 21.03 | 21.03 |
| physical_exp_beta003 | 0.900 | 18/20 (90.0%) | 1.000 | 14.76 | 14.76 |
| cogact_alpha03 | 0.900 | 18/20 (90.0%) | 1.000 | 19.02 | 19.02 |
| component_arm_fresh_gripper_act | 0.950 | 19/20 (95.0%) | 1.000 | 0.00 | 20.87 |

### Paired comparisons vs fresh

| method | task-macro delta | pooled delta | candidate-only/reference-only | p |
|---|---:|---:|---:|---:|
| official_act_m001 | -0.100 | -0.100 | 1/3 | 0.625 |
| physical_exp_beta003 | -0.050 | -0.050 | 1/2 | 1.0 |
| cogact_alpha03 | -0.050 | -0.050 | 1/2 | 1.0 |
| component_arm_fresh_gripper_act | +0.000 | +0.000 | 1/1 | 1.0 |

Component-aware minus official full-action ACT: task-macro +0.100; pooled +0.100; candidate-only/reference-only 3/1; exact McNemar p=0.625.

## libero_10

| method | task macro | pooled | queries/step | arm age | gripper age |
|---|---:|---:|---:|---:|---:|
| fresh | 0.900 | 18/20 (90.0%) | 1.000 | 0.00 | 0.00 |
| official_act_m001 | 0.550 | 11/20 (55.0%) | 1.000 | 23.97 | 23.97 |
| physical_exp_beta003 | 0.700 | 14/20 (70.0%) | 1.000 | 16.73 | 16.73 |
| cogact_alpha03 | 0.750 | 15/20 (75.0%) | 1.000 | 21.86 | 21.86 |
| component_arm_fresh_gripper_act | 0.750 | 15/20 (75.0%) | 1.000 | 0.00 | 24.18 |

### Paired comparisons vs fresh

| method | task-macro delta | pooled delta | candidate-only/reference-only | p |
|---|---:|---:|---:|---:|
| official_act_m001 | -0.350 | -0.350 | 0/7 | 0.015625 |
| physical_exp_beta003 | -0.200 | -0.200 | 0/4 | 0.125 |
| cogact_alpha03 | -0.150 | -0.150 | 1/4 | 0.375 |
| component_arm_fresh_gripper_act | -0.150 | -0.150 | 0/3 | 0.25 |

Component-aware minus official full-action ACT: task-macro +0.200; pooled +0.200; candidate-only/reference-only 4/0; exact McNemar p=0.125.

## all_tasks

| method | task macro | pooled | queries/step | arm age | gripper age |
|---|---:|---:|---:|---:|---:|
| fresh | 0.900 | 72/80 (90.0%) | 1.000 | 0.00 | 0.00 |
| official_act_m001 | 0.825 | 66/80 (82.5%) | 1.000 | 21.46 | 21.46 |
| physical_exp_beta003 | 0.887 | 71/80 (88.8%) | 1.000 | 15.22 | 15.22 |
| cogact_alpha03 | 0.912 | 73/80 (91.2%) | 1.000 | 19.58 | 19.58 |
| component_arm_fresh_gripper_act | 0.912 | 73/80 (91.2%) | 1.000 | 0.00 | 21.48 |

### Paired comparisons vs fresh

| method | task-macro delta | pooled delta | candidate-only/reference-only | p |
|---|---:|---:|---:|---:|
| official_act_m001 | -0.075 | -0.075 | 5/11 | 0.210113525390625 |
| physical_exp_beta003 | -0.013 | -0.013 | 5/6 | 1.0 |
| cogact_alpha03 | +0.013 | +0.013 | 7/6 | 1.0 |
| component_arm_fresh_gripper_act | +0.013 | +0.013 | 6/5 | 1.0 |

Component-aware minus official full-action ACT: task-macro +0.088; pooled +0.087; candidate-only/reference-only 9/2; exact McNemar p=0.0654296875.

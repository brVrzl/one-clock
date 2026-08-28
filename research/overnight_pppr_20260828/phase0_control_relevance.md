# Phase-0 control relevance

This report uses only the frozen `phase0_features.npz` table and paired intervention outcomes/source-event logs. For each active source event at physical step `u`, the score is read from the Fresh row `(task, episode, old_query_t=u-d, age_steps=d)`; intervention-trajectory predictions are never used.

- Exact command: `/home/wjq/workspace/venvs/libero_act/bin/python research/overnight_pppr_20260828/analyze_control_relevance.py --force`
- Started (UTC): `2026-08-28T10:58:33.910303+00:00`
- Finished (UTC): `2026-08-28T10:59:49.128543+00:00`
- Runtime seconds: `75.218`
- Pair rows: `720`; decisive rows: `181`
- Source features: `/home/wjq/workspace/one-clock/research/overnight_pppr_20260828/phase0_features.npz`
- Source outcomes/logs: `/home/wjq/workspace/one-clock/experiments/component_temporal_reuse/pilot_results.json`

## Pairing and alignment

`DeltaY = Y_fresh - Y_intervention`; decisive rows have `DeltaY != 0`, and `Z=1` means harmful old source (`DeltaY=+1`). FullOld uses joint scores, Reverse uses arm scores, and FO uses gripper scores at ages 4, 8, and 16. Episode-condition scores are arithmetic means over valid active logged steps. Warm-up age-0 steps are excluded. Rows with no valid Fresh feature are not filled.

- Active logged steps: `130633`; valid aligned steps: `106644`; missing/invalid Fresh rows: `23989`.

## Primary metrics on decisive pairs

### development (primary)

| signal | n | harmful | prevalence | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| age | 98 | 63 | 0.643 | 0.564 | 0.660 |
| event | 98 | 63 | 0.643 | 0.112 | 0.457 |
| raw_ppr | 98 | 63 | 0.643 | 0.439 | 0.600 |
| pppr | 98 | 63 | 0.643 | 0.249 | 0.534 |

Episode-cluster bootstrap (10,000 paired draws; percentile 95% CI): age AUROC [0.447, 0.655], AUPRC [0.476, 0.901]; event AUROC [0.000, 0.293], AUPRC [0.276, 0.742]; raw_ppr AUROC [0.333, 0.556], AUPRC [0.388, 0.848]; pppr AUROC [0.106, 0.410], AUPRC [0.321, 0.796].
Valid AUROC draws: {'age': 9997, 'event': 9997, 'raw_ppr': 9997, 'pppr': 9997}; class-degenerate draws: {'age': 3, 'event': 3, 'raw_ppr': 3, 'pppr': 3}; PPPR-minus-Raw AUROC CI: [-0.290, -0.103].

Task-cluster bootstrap (10,000 paired draws; percentile 95% CI): age AUROC [0.463, 0.713], AUPRC [0.401, 0.933]; event AUROC [0.000, 0.164], AUPRC [0.240, 0.764]; raw_ppr AUROC [0.389, 0.600], AUPRC [0.342, 0.904]; pppr AUROC [0.167, 0.354], AUPRC [0.254, 0.808].
Valid AUROC draws: {'age': 10000, 'event': 10000, 'raw_ppr': 10000, 'pppr': 10000}; class-degenerate draws: {'age': 0, 'event': 0, 'raw_ppr': 0, 'pppr': 0}; PPPR-minus-Raw AUROC CI: [-0.278, -0.103].

### held-out (primary)

| signal | n | harmful | prevalence | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| age | 83 | 41 | 0.494 | 0.649 | 0.743 |
| event | 83 | 41 | 0.494 | 0.494 | 0.465 |
| raw_ppr | 83 | 41 | 0.494 | 0.506 | 0.479 |
| pppr | 83 | 41 | 0.494 | 0.491 | 0.462 |

Episode-cluster bootstrap (10,000 paired draws; percentile 95% CI): age AUROC [0.571, 0.743], AUPRC [0.425, 0.875]; event AUROC [0.149, 0.796], AUPRC [0.248, 0.896]; raw_ppr AUROC [0.307, 0.662], AUPRC [0.282, 0.762]; pppr AUROC [0.207, 0.716], AUPRC [0.252, 0.808].
Valid AUROC draws: {'age': 9997, 'event': 9997, 'raw_ppr': 9997, 'pppr': 9997}; class-degenerate draws: {'age': 3, 'event': 3, 'raw_ppr': 3, 'pppr': 3}; PPPR-minus-Raw AUROC CI: [-0.159, 0.097].

Task-cluster bootstrap (10,000 paired draws; percentile 95% CI): age AUROC [0.587, 0.688], AUPRC [0.433, 0.791]; event AUROC [0.154, 0.545], AUPRC [0.241, 0.482]; raw_ppr AUROC [0.381, 0.544], AUPRC [0.342, 0.557]; pppr AUROC [0.266, 0.558], AUPRC [0.299, 0.491].
Valid AUROC draws: {'age': 10000, 'event': 10000, 'raw_ppr': 10000, 'pppr': 10000}; class-degenerate draws: {'age': 0, 'event': 0, 'raw_ppr': 0, 'pppr': 0}; PPPR-minus-Raw AUROC CI: [-0.121, 0.014].

### all_data (descriptive only)

| signal | n | harmful | prevalence | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| age | 181 | 104 | 0.575 | 0.602 | 0.661 |
| event | 181 | 104 | 0.575 | 0.352 | 0.461 |
| raw_ppr | 181 | 104 | 0.575 | 0.488 | 0.552 |
| pppr | 181 | 104 | 0.575 | 0.396 | 0.507 |

## Held-out component-matched metrics

| component-matched population | n | harmful | PPPR AUROC | RawPPR AUROC | PPPR−Raw |
|---|---:|---:|---:|---:|---:|
| full_old_joint | 27 | 12 | 0.594 | 0.639 | -0.044 |
| reverse_arm | 33 | 18 | 0.389 | 0.389 | 0.000 |
| fo_grip | 23 | 11 | 0.519 | 0.496 | 0.023 |

## Task-wise direction

For each task/condition, the table reports harmful-old-source and beneficial/other score means/medians, plus AUROC only when both classes are present.

| split | task | condition | n | harmful | PPPR harmful median | PPPR beneficial median | PPPR AUROC |
|---|---|---|---:|---:|---:|---:|---:|
| development | libero_object:task3 | full_old4 | 2 | 1 | 0.057 | 0.079 | 0.000 |
| development | libero_object:task3 | full_old8 | 0 | 0 | NA | NA | NA |
| development | libero_object:task3 | full_old16 | 0 | 0 | NA | NA | NA |
| development | libero_object:task3 | reverse4 | 0 | 0 | NA | NA | NA |
| development | libero_object:task3 | reverse8 | 0 | 0 | NA | NA | NA |
| development | libero_object:task3 | reverse16 | 6 | 5 | 0.077 | 0.100 | 0.200 |
| development | libero_object:task3 | fo4 | 1 | 0 | NA | 0.103 | NA |
| development | libero_object:task3 | fo8 | 4 | 3 | 0.045 | 0.068 | 0.000 |
| development | libero_object:task3 | fo16 | 0 | 0 | NA | NA | NA |
| development | libero_spatial:task0 | full_old4 | 3 | 1 | 0.067 | 0.117 | 0.000 |
| development | libero_spatial:task0 | full_old8 | 5 | 2 | 0.073 | 0.126 | 0.000 |
| development | libero_spatial:task0 | full_old16 | 4 | 1 | 0.111 | 0.157 | 0.000 |
| development | libero_spatial:task0 | reverse4 | 3 | 1 | 0.102 | 0.100 | 0.500 |
| development | libero_spatial:task0 | reverse8 | 1 | 0 | NA | 0.097 | NA |
| development | libero_spatial:task0 | reverse16 | 5 | 3 | 0.129 | 0.143 | 0.333 |
| development | libero_spatial:task0 | fo4 | 1 | 0 | NA | 0.175 | NA |
| development | libero_spatial:task0 | fo8 | 1 | 0 | NA | 0.171 | NA |
| development | libero_spatial:task0 | fo16 | 6 | 2 | 0.063 | 0.151 | 0.000 |
| development | libero_goal:task2 | full_old4 | 1 | 0 | NA | 0.070 | NA |
| development | libero_goal:task2 | full_old8 | 1 | 0 | NA | 0.085 | NA |
| development | libero_goal:task2 | full_old16 | 2 | 1 | 0.078 | 0.126 | 0.000 |
| development | libero_goal:task2 | reverse4 | 3 | 2 | 0.114 | 0.105 | 1.000 |
| development | libero_goal:task2 | reverse8 | 2 | 1 | 0.116 | 0.126 | 0.000 |
| development | libero_goal:task2 | reverse16 | 1 | 0 | NA | 0.160 | NA |
| development | libero_goal:task2 | fo4 | 4 | 3 | 0.054 | 0.022 | 1.000 |
| development | libero_goal:task2 | fo8 | 0 | 0 | NA | NA | NA |
| development | libero_goal:task2 | fo16 | 2 | 1 | 0.044 | 0.089 | 0.000 |
| development | libero_10:task3 | full_old4 | 5 | 4 | 0.063 | 0.109 | 0.000 |
| development | libero_10:task3 | full_old8 | 4 | 4 | 0.071 | NA | NA |
| development | libero_10:task3 | full_old16 | 6 | 6 | 0.085 | NA | NA |
| development | libero_10:task3 | reverse4 | 4 | 3 | 0.087 | 0.099 | 0.000 |
| development | libero_10:task3 | reverse8 | 4 | 4 | 0.092 | NA | NA |
| development | libero_10:task3 | reverse16 | 7 | 7 | 0.114 | NA | NA |
| development | libero_10:task3 | fo4 | 1 | 0 | NA | 0.132 | NA |
| development | libero_10:task3 | fo8 | 4 | 4 | 0.091 | NA | NA |
| development | libero_10:task3 | fo16 | 5 | 4 | 0.042 | 0.160 | 0.250 |
| held_out | libero_object:task5 | full_old4 | 3 | 1 | 0.105 | 0.145 | 0.000 |
| held_out | libero_object:task5 | full_old8 | 3 | 1 | 0.126 | 0.190 | 0.000 |
| held_out | libero_object:task5 | full_old16 | 6 | 5 | 0.076 | 0.221 | 0.000 |
| held_out | libero_object:task5 | reverse4 | 5 | 3 | 0.077 | 0.095 | 0.167 |
| held_out | libero_object:task5 | reverse8 | 3 | 2 | 0.092 | 0.135 | 0.000 |
| held_out | libero_object:task5 | reverse16 | 9 | 8 | 0.102 | 0.122 | 0.250 |
| held_out | libero_object:task5 | fo4 | 2 | 1 | 0.139 | 0.168 | 0.000 |
| held_out | libero_object:task5 | fo8 | 2 | 1 | 0.172 | 0.237 | 0.000 |
| held_out | libero_object:task5 | fo16 | 2 | 1 | 0.219 | 0.317 | 0.000 |
| held_out | libero_spatial:task4 | full_old4 | 2 | 1 | 0.050 | 0.053 | 0.000 |
| held_out | libero_spatial:task4 | full_old8 | 2 | 1 | 0.047 | 0.048 | 0.000 |
| held_out | libero_spatial:task4 | full_old16 | 2 | 2 | 0.060 | NA | NA |
| held_out | libero_spatial:task4 | reverse4 | 0 | 0 | NA | NA | NA |
| held_out | libero_spatial:task4 | reverse8 | 2 | 1 | 0.083 | 0.085 | 0.000 |
| held_out | libero_spatial:task4 | reverse16 | 3 | 1 | 0.098 | 0.128 | 0.000 |
| held_out | libero_spatial:task4 | fo4 | 1 | 0 | NA | 0.012 | NA |
| held_out | libero_spatial:task4 | fo8 | 1 | 0 | NA | 0.013 | NA |
| held_out | libero_spatial:task4 | fo16 | 2 | 1 | 0.026 | 0.051 | 0.000 |
| held_out | libero_goal:task5 | full_old4 | 2 | 1 | 0.052 | 0.053 | 0.000 |
| held_out | libero_goal:task5 | full_old8 | 1 | 0 | NA | 0.066 | NA |
| held_out | libero_goal:task5 | full_old16 | 0 | 0 | NA | NA | NA |
| held_out | libero_goal:task5 | reverse4 | 2 | 1 | 0.094 | 0.106 | 0.000 |
| held_out | libero_goal:task5 | reverse8 | 2 | 1 | 0.103 | 0.136 | 0.000 |
| held_out | libero_goal:task5 | reverse16 | 1 | 0 | NA | 0.166 | NA |
| held_out | libero_goal:task5 | fo4 | 0 | 0 | NA | NA | NA |
| held_out | libero_goal:task5 | fo8 | 1 | 0 | NA | 0.000 | NA |
| held_out | libero_goal:task5 | fo16 | 4 | 3 | 0.000 | 0.000 | 0.500 |
| held_out | libero_10:task5 | full_old4 | 2 | 0 | NA | 0.042 | NA |
| held_out | libero_10:task5 | full_old8 | 2 | 0 | NA | 0.040 | NA |
| held_out | libero_10:task5 | full_old16 | 2 | 0 | NA | 0.046 | NA |
| held_out | libero_10:task5 | reverse4 | 2 | 0 | NA | 0.082 | NA |
| held_out | libero_10:task5 | reverse8 | 2 | 0 | NA | 0.079 | NA |
| held_out | libero_10:task5 | reverse16 | 2 | 1 | 0.101 | 0.099 | 1.000 |
| held_out | libero_10:task5 | fo4 | 2 | 1 | 0.007 | 0.002 | 1.000 |
| held_out | libero_10:task5 | fo8 | 2 | 0 | NA | 0.001 | NA |
| held_out | libero_10:task5 | fo16 | 4 | 3 | 0.013 | 0.007 | 0.667 |

### Held-out task-pooled PPPR direction

Each row pools the nine component-matched conditions (three ages for FullOld, Reverse, and FO) within one held-out task.

| task | n | harmful | PPPR harmful mean/median | PPPR beneficial mean/median | PPPR AUROC |
|---|---:|---:|---:|---:|---:|
| libero_object:task5 | 35 | 23 | 0.106/0.095 | 0.172/0.158 | 0.178 |
| libero_spatial:task4 | 15 | 7 | 0.060/0.057 | 0.065/0.052 | 0.500 |
| libero_goal:task5 | 13 | 6 | 0.041/0.026 | 0.075/0.066 | 0.310 |
| libero_10:task5 | 20 | 5 | 0.029/0.013 | 0.046/0.042 | 0.400 |

## RawPPR versus PPPR distributions

These summaries are descriptive. Candidate feature rows and episode-condition rows are not treated as independent inferential samples.

| split | population | n | RawPPR median [q25,q75] | PPPR median [q25,q75] |
|---|---|---:|---:|---:|
| development | valid feature rows: full_old_joint | 20039 | 0.179 [0.136, 0.246] | 0.050 [0.019, 0.100] |
| development | valid feature rows: reverse_arm | 20039 | 0.344 [0.265, 0.440] | 0.088 [0.035, 0.156] |
| development | valid feature rows: fo_grip | 20039 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| development | episode-condition scores: full_old_joint | 120 | 0.197 [0.177, 0.228] | 0.071 [0.061, 0.097] |
| development | episode-condition scores: reverse_arm | 120 | 0.335 [0.306, 0.371] | 0.100 [0.087, 0.116] |
| development | episode-condition scores: fo_grip | 120 | 0.051 [0.035, 0.089] | 0.049 [0.030, 0.088] |
| held_out | valid feature rows: full_old_joint | 23999 | 0.162 [0.126, 0.209] | 0.043 [0.016, 0.079] |
| held_out | valid feature rows: reverse_arm | 23999 | 0.316 [0.248, 0.395] | 0.081 [0.030, 0.143] |
| held_out | valid feature rows: fo_grip | 23999 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| held_out | episode-condition scores: full_old_joint | 120 | 0.167 [0.148, 0.187] | 0.053 [0.046, 0.067] |
| held_out | episode-condition scores: reverse_arm | 120 | 0.307 [0.281, 0.331] | 0.088 [0.082, 0.100] |
| held_out | episode-condition scores: fo_grip | 120 | 0.017 [0.000, 0.060] | 0.013 [0.000, 0.054] |
| all_data | valid feature rows: full_old_joint | 44038 | 0.170 [0.130, 0.223] | 0.046 [0.017, 0.087] |
| all_data | valid feature rows: reverse_arm | 44038 | 0.328 [0.254, 0.414] | 0.084 [0.032, 0.149] |
| all_data | valid feature rows: fo_grip | 44038 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| all_data | episode-condition scores: full_old_joint | 240 | 0.182 [0.156, 0.213] | 0.063 [0.051, 0.085] |
| all_data | episode-condition scores: reverse_arm | 240 | 0.323 [0.286, 0.346] | 0.093 [0.083, 0.107] |
| all_data | episode-condition scores: fo_grip | 240 | 0.043 [0.017, 0.075] | 0.037 [0.012, 0.071] |

## Gate recommendation

**FAIL** for the held-out combined/component-matched gate.

Held-out PPPR AUROC=0.491 and RawPPR AUROC=0.506; PPPR-minus-Raw AUROC=-0.015. The frozen guide is not met without forcing a positive conclusion.

The gate guide is held-out PPPR AUROC roughly ≥0.65, improvement over RawPPR roughly ≥0.05 or similarly strong consistent rank separation, and no catastrophic suite reversal. The operational checks and all component values are recorded in the JSON output.

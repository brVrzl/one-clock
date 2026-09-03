# Post-hoc command-discontinuity characterization

Status: **COMPLETE**

Computed only from existing completed trajectories. No reviewer-supplement artifact or success outcome was used.

Quantities are controller-native command differences, not physical jerk. Translation, rotation, and gripper are reported separately.

Fresh and A20G0 have no same-source transitions by construction; those comparisons are recorded as `STRUCTURALLY_UNAVAILABLE`, not as outcomes.

## Mean first-difference contrasts

Positive values mean greater command variation in the first named condition. Intervals are task-cluster percentile 95% intervals.

| Contrast | Group | Mean paired difference | 95% interval |
|---|---|---:|---:|
| Fresh-A20G0 | translation | 0.005102 | [0.000535, 0.010218] |
| Fresh-A20G0 | rotation | 0.000561 | [0.000166, 0.000986] |
| Fresh-A20G0 | gripper | 0.008659 | [0.002458, 0.014951] |
| Fresh-coherent_H16 | translation | -0.009169 | [-0.013858, -0.004888] |
| Fresh-coherent_H16 | rotation | -0.001164 | [-0.001625, -0.000717] |
| Fresh-coherent_H16 | gripper | -0.031176 | [-0.053365, -0.012447] |
| A20G0-coherent_H16 | translation | -0.014271 | [-0.018869, -0.009921] |
| A20G0-coherent_H16 | rotation | -0.001725 | [-0.002165, -0.001285] |
| A20G0-coherent_H16 | gripper | -0.039835 | [-0.063960, -0.018249] |
| ARM4_GRIP32-H4 | translation | 0.001231 | [0.000389, 0.002279] |
| ARM4_GRIP32-H4 | rotation | 0.000097 | [-0.000009, 0.000214] |
| ARM4_GRIP32-H4 | gripper | -0.013040 | [-0.018683, -0.007894] |
| ARM2_GRIP16-H2 | translation | 0.001170 | [0.000209, 0.002313] |
| ARM2_GRIP16-H2 | rotation | 0.000124 | [0.000032, 0.000224] |
| ARM2_GRIP16-H2 | gripper | -0.000462 | [-0.007289, 0.006981] |

## Gripper state-switch contrasts

| Contrast | Mean paired probability difference | 95% interval |
|---|---:|---:|
| Fresh-A20G0 | 0.003522 | [0.000277, 0.006345] |
| Fresh-coherent_H16 | -0.015890 | [-0.027037, -0.006131] |
| A20G0-coherent_H16 | -0.019412 | [-0.031601, -0.008406] |
| ARM4_GRIP32-H4 | -0.005382 | [-0.008237, -0.002872] |
| ARM2_GRIP16-H2 | -0.000829 | [-0.004933, 0.003431] |

## Post-hoc interpretation

The requested D1 comparisons do not support a simple account in which arm temporal benefits arise from reduced executed arm-command discontinuity. Coherent H16 had greater translation and rotation D1 than both Fresh and A20G0, and both split Track-A methods had slightly greater arm-D1 point estimates than their matched global-horizon comparators.

ARM4_GRIP32 reduced gripper D1 and executed gripper state-switch probability relative to H4. The corresponding ARM2_GRIP16-H2 differences were small with task-cluster intervals spanning zero. This is compatible with sparse-transition timing mattering at the first operating point, but is not uniform evidence and does not establish a causal or forecasting mechanism.

Canonical numerical outputs are `condition_summaries.csv`, `source_transition_summaries.csv`, `contrasts.csv`, `task_contrasts.csv`, `trajectory_summaries.csv`, and `analysis.json`.

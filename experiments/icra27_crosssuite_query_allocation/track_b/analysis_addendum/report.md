# Track-B B1 per-dimension same-target analysis

All values use each checkpoint's frozen normalized action space and the frozen 16-source primary window. These explanatory analyses do not alter the original Track-B labels.

| Policy | Translation | Rotation | Gripper | Original arm |
|---|---:|---:|---:|---:|
| ACT | 0.136408 | 0.148419 | 0.079009 | 0.146425 |
| SmolVLA | 0.315715 | 0.464434 | 0.173287 | 0.401964 |

## Controller-native diagnostics

Values below invert the exact checkpoint MEAN_STD transform and remain in controller-native action units.

| Policy | Translation source dispersion | Rotation source dispersion | Gripper source dispersion | Gripper sign disagreement | Low-minus-high margin disagreement |
|---|---:|---:|---:|---:|---:|
| ACT | 0.053776 | 0.008785 | 0.078912 | 0.059515 | 0.178281 |
| SmolVLA | 0.121933 | 0.026908 | 0.173073 | 0.072452 | 0.216946 |

Age-resolved normalized and controller-native differences are in the accompanying tidy CSV/JSON files. Normalized dispersion remains descriptive context; no success outcome was loaded.

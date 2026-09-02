# Track-B same-target instability diagnostic

ACT localization: **KILL**. Cross-policy mechanism support: **NO**.

These are mechanism-only diagnostics on already outcome-exposed development cells. Success outcomes were not loaded or used for method selection.

| Policy | Arm dispersion | Gripper dispersion | R | Episode-cluster 95% CI | Low-minus-high margin disagreement | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| ACT | 0.146425 | 0.079009 | 0.540 | [0.397, 0.703] | 0.1792 | [0.1209, 0.2353] |
| SmolVLA | 0.401964 | 0.173287 | 0.431 | [0.351, 0.518] | 0.2169 | [0.1582, 0.2786] |

`R_ACT - R_SMOLVLA = 0.108`, paired episode-cluster 95% CI `[-0.042, 0.282]`.

# B2 training-demonstration temporal persistence

These 173 episodes are training demonstrations, not a held-out split. Dataset actions are at 10 Hz. No rollout success outcome was loaded.

Adjacent gripper sign-transition frequency: `0.015517` (episode-cluster 95% CI `[0.014463, 0.016662]`).
Right-censored action-step fraction: `19.172%`. The `30.775`-step mean among observed transitions is a biased complete-case description, not a population mean.

| Window | P(no gripper transition within window) | Episode-cluster 95% CI |
|---:|---:|---:|
| 0.5 s | 0.921765 | [0.916275, 0.926915] |
| 1.0 s | 0.840878 | [0.829623, 0.851380] |
| 2.0 s | 0.675018 | [0.652380, 0.696128] |

All per-dimension autocorrelation, normalized-difference, group, and lag results are stored in `lag_metrics.csv` and `summary.json`.

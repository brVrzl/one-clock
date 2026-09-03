# B3 open-loop future-action predictability

This is a training-demonstration reference analysis, not held-out evaluation. ACT and SmolVLA are reported in their own frozen normalized spaces; no rollout success outcome was loaded.

Chunk offset k is an exact stored-row target at k/20 physical seconds. The dataset-declared 10 Hz timestamps relabel the retained 20 Hz sequence; no interpolation, resampling, or repetition is used.

All per-offset translation, rotation, gripper, per-dimension, sign-disagreement, and episode-cluster interval results are in `forecast_metrics.csv` and `summary.json`.

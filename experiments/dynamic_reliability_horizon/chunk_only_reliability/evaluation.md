# Chunk-only `Y_refresh` reliability pilot

## Status: blocked before scientific training

The real pilot was not run. The required Thor portable handoff is not present
in this checkout or on the searched shared filesystem/remotes:

```text
experiments/dynamic_reliability_horizon/artifact_handoff/
  minimal_y_refresh_training_bundle.npz
```

Expected bundle SHA256:

```text
45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4
```

No independent reconstruction of `Y_refresh` was attempted. Therefore there
are no learned-model metrics, confidence intervals, calibration figures,
offset CSVs, or horizon-recovery results in this directory. `metrics.json`
records the blocker instead of a partial headline result.

## Intended pilot

The input is the deliberate chunk-only ablation:

```text
R_g(k | A_t, g)
```

The four comparisons are a constant prior, a train-fitted empirical
`P(Y_refresh=1 | g,k)` prior, an independent vector shared MLP, and a
monotone conditional-survival shared MLP. The learned models use three fixed
seeds from `config.json`; the test split is not used for tuning.

The expected Thor bundle contract is 3,740 source windows with source chunks
of shape `(3740, 100, 7)`, offsets `k=1..99`, group-wise labels/censor masks,
episode IDs, and an episode-level split. Offset zero is excluded from loss and
all headline metrics. It is not evidence of a learned horizon.

When the exact artifact is mounted, the report must include pooled, group-wise,
fixed-offset and macro-valid-offset AUROC/AUPRC/Brier/ECE, Brier Skill Score
against the empirical group-offset prior, and arm/gripper
refresh-oracle-horizon recovery. The final verdict must be GO/PARTIAL/NO-GO
according to the pre-registered standard in the task request.

## Code state

The verified estimator/evaluation implementation is committed in
`9855f52` (`Add monotone reliability evaluation heads`). Focused tests pass:

```text
22 passed in 2.25s
```

The blocked report and config are intentionally separate from the earlier
uncommitted result edits in `experiments/dynamic_reliability_horizon/results/`.

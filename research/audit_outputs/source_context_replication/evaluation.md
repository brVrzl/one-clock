# Exact causal source-context ablation

Status: completed offline; final scientific verdict: **PARTIAL**.

The exact portable Y_refresh bundle remains locked at SHA256 `45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4`. The feature artifact is `bf1075076631305fa9a6992baab7b888761d92d7e5531a947e72416740c61dbc`. No Y_refresh regeneration, rollout, executor change, or paper-claim change occurred.

## Frozen cohort and features

The 3,740 source windows are joined by unique `(episode_id, source_step) = (episode_index, frame_index)` keys. State is the causal 8-vector `[EEF position (3), EEF axis-angle orientation (3), gripper qpos (2)]` at source time. The primary frozen-ACT representation is the 512-D first token of the final fused `policy.model.encoder` output, captured before ACT decoding; it includes only the current state and current images and excludes the training-time VAE action-conditioned latent.

Latent extraction invariance: max postprocessed action delta `0.0`, allclose at `1e-6`: `True`. The locked chunk remains canonical; same-path replay drift is recorded separately with max absolute delta `0.005125686526298523`.

## Primary results

All conditions use one shared monotone conditional-survival MLP with the same `(64, 32)` hidden dimensions, seeds, optimizer, episode split, censor mask, and validation-only tau selection. Condition A reproduces the existing same-formulation chunk-only pilot within `1e-12` on the parity fields recorded in `metrics.json`.

| Condition | pooled AUROC | pooled Brier | pooled Brier Skill | arm fixed-offset AUROC | gripper fixed-offset AUROC | arm MAE | gripper MAE | overall MAE | overall within ±2 | Spearman | tau |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_chunk_only | 0.9197 | 0.1159 | 0.2577 | 0.7407 | 0.9404 | 24.06 | 7.51 | 15.78 | 0.217 | 0.627 | 0.70 |
| B_chunk_plus_state | 0.9191 | 0.1153 | 0.2596 | 0.7377 | 0.9425 | 23.65 | 7.76 | 15.71 | 0.191 | 0.633 | 0.65 |
| C_chunk_plus_frozen_ACT_latent | 0.9157 | 0.1180 | 0.2412 | 0.7484 | 0.9298 | 25.79 | 9.04 | 17.41 | 0.191 | 0.598 | 0.60 |
| D_chunk_plus_state_plus_frozen_ACT_latent | 0.9139 | 0.1201 | 0.2318 | 0.7285 | 0.9234 | 24.38 | 8.28 | 16.33 | 0.208 | 0.617 | 0.65 |

Empirical group/offset baseline horizon: overall MAE `21.01`, within ±2 `0.078`. The best test-MAE condition is `B_chunk_plus_state`; this selection is descriptive after the fixed protocol, while each condition's tau was selected on validation only.

## Scientific questions

1. **State beyond chunk-only:** see B versus A in `comparison_table.csv`; fixed-offset and horizon changes are reported without retuning the criterion.
2. **Frozen ACT latent beyond chunk-only:** see C versus A.
3. **Arm versus gripper:** compare the fixed-offset macro columns and group horizon rows; this directly tests whether latent value is concentrated in arm reliability.
4. **Horizon recovery:** the predeclared usefulness criterion is unchanged. High AUROC alone does not overturn PARTIAL; the arm and overall horizon metrics are decisive.
5. **Smallest useful set:** compare A-D in `comparison_table.csv`; if no augmented condition meets the usefulness rule, the simplest useful estimator remains chunk-only and the result is a causal augmentation failure, not a failure of the project-wide chunk signal.

## Artifacts

- `feature_bundle.npz`, `feature_manifest.json`, `feature_provenance.md` — immutable source features and provenance.
- `metrics.json`, `comparison_table.csv`, `per_group_metrics.csv`, `per_offset_metrics.csv`, `per_group_offset_metrics.csv`, `horizon_metrics.csv` — held-out metrics and exact protocol.
- `config.json`, `seeds.json`, `plots/` — reproducibility configuration and figures.

No executor semantics, Y_refresh labels, rollout code, or paper claims were changed.

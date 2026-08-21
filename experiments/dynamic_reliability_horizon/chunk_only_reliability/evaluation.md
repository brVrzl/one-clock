# Chunk-only reliability pilot

Status: completed on CPU with the exact portable bundle (SHA256 `45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4`).

This is the causal chunk-only ablation: one shared predictor receives only the source predicted action chunk and a one-hot arm/gripper identity. No future observations, future actions, phase, progress, episode length, terminal metadata, rollout, or executor call is used.

## Protocol

The bundle contains k=1..99. k=0 is excluded from training and all headline metrics; R(0)=1 is prepended only while decoding horizons. Priors, feature normalization, model selection, and tau selection use train/validation episodes only.

Seeds: 20260820, 20260821, 20260822. The two learned models are an independent-vector shared MLP and a monotone conditional-survival shared MLP.

## Required questions

1. Source-window-conditioned information beyond P(Y=1|g,k): the best learned aggregate is `independent_vector_shared_mlp|aggregate_mean_prediction`. Fixed group/offset AUROC/AUPRC are in `per_group_offset_metrics.csv`; pooled fixed-offset metrics are in `per_offset_metrics.csv`. The pre-registered fixed-offset AUROC threshold is 0.55.
2. Arm signal: {'mean_valid_group_offset_auroc': 0.7389732316358752, 'valid_group_offset_slices': 97, 'slices_at_or_above_threshold': 97, 'signal_fraction_of_valid_slices': 1.0, 'meaningful_signal': True}. Gripper signal: {'mean_valid_group_offset_auroc': 0.9295771986461262, 'valid_group_offset_slices': 91, 'slices_at_or_above_threshold': 91, 'signal_fraction_of_valid_slices': 1.0, 'meaningful_signal': True}.
3. Best learned test probability metrics: pooled AUROC=0.9223891353335086, AUPRC=0.9109292118487539, Brier=0.11361307338352078, ECE=0.03137494587073155, Brier Skill versus empirical=0.30430535098132616. The full pooled, group, offset, and macro reports are in `metrics.json` and the CSV files.
4. Best learned refresh-oracle horizon recovery: overall MAE=16.711143695014663, median absolute error=9.0, exact match=0.04252199413489736, within +/-2=0.20087976539589442, within +/-5=0.3680351906158358, Spearman=0.608840304832739, over-commit=0.4413489736070381, under-commit=0.5161290322580645, mean learned horizon=32.93255131964809, mean oracle horizon=33.21847507331378.
5. Empirical baseline overall test horizon MAE=19.96041055718475, within +/-2=0.11436950146627566; tau was selected on validation only. Selected tau for the best learned aggregate: 0.65.
6. Verdict: **PARTIAL**. This is a scientific reliability diagnostic, not a task-optimal horizon claim.
7. Next scientific experiment: Materialize exact causal source-time observation/state or frozen-policy latent features and rerun the same pre-registered estimator evaluation; keep future observations/actions, phase, progress, terminal metadata, and rollout semantics excluded.

## Undefined slices

Any group/offset or pooled offset containing only one observed label class has AUROC/AUPRC recorded as undefined with positive/negative/sample counts; no discrimination value is fabricated. Brier and ECE remain reported because they are probability metrics.

## Artifacts

- `metrics.json` — complete nested results and fixed decision rule
- `per_group_metrics.csv` — pooled metrics by arm/gripper
- `per_offset_metrics.csv` — pooled fixed-offset metrics
- `per_group_offset_metrics.csv` — critical group/fixed-offset discrimination slices
- `horizon_metrics.csv` — per-group refresh-oracle recovery
- `validation_tau_sweep.csv` — validation-only threshold sweep
- `plots/` — fixed-offset, calibration, and horizon plots

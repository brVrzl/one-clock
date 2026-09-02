# Frozen Track-B analysis-only addendum

Frozen after all 80 diagnostic rollouts completed but before any prediction array or Track-B scientific result was inspected. This addendum adds explanatory analyses only. It does not change the Track-B panel, execution, primary window, normalization, bootstrap unit, original `ACT_LOCALIZATION_PASS` / `ACT_LOCALIZATION_KILL` criteria, or `CROSS_POLICY_MECHANISM_SUPPORT` criterion in `track_b_manifest.json`.

Track B is described as **mechanism-only logging on already outcome-exposed development cells; success outcomes are not used for method selection**. It is not described as outcome-free.

## B1. Per-dimension same-target instability

The frozen primary window remains physical target `t >= 15` and source ages `a=0..15`, with exactly 16 valid predictions of the same physical target. All calculations use each checkpoint's frozen normalized action space.

For every target and dimension `j`, retain the original 16-source population RMS dispersion

`D_j(t) = sqrt(mean_a((A_{t-a}[a,j] - mean_b A_{t-b}[b,j])^2))`.

Report episode means separately for dimensions 0, 1, 2, 3, 4, 5, and 6. Report translation as `sqrt(mean_j(D_j(t)^2))` for `j=0..2`, rotation analogously for `j=3..5`, and gripper as `D_6(t)`. Retain the original arm metric over dimensions 0..5 unchanged.

For an age-resolved explanatory curve, report fresh-referenced same-target disagreement

`E_j(a) = RMS_{episode,target}(A_{t-a}[a,j] - A_t[0,j])`

for every integer `a=0..15`, separately per dimension and for translation, rotation, and gripper groupings. This curve is not substituted for the original decision statistic. Episode-cluster percentile bootstrap uses 20,000 draws, 95% intervals, and seed 27211 for 16-source dimension summaries and 27212 for age curves. These analyses are descriptive and receive no new pass threshold.

The original gripper sign-disagreement, sign-entropy, absolute-margin, margin-tercile, and ratio analyses remain exactly as frozen. No success field enters B1.

## B2. Demonstration temporal persistence

Dataset audit establishes no defensibly unused named validation/test split. The local `HuggingFaceVLA/libero` dataset metadata at revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e` declares only `train: 0:1693`. Each of the four task-specific ACT checkpoints was trained on every dataset episode matching its task language: Object task3 46 episodes, Spatial task0 45, Goal task2 47, and LIBERO-10 task3 35. The SmolVLA checkpoint is local revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`; its model card says `datasets: unknown`, so no unused split can be established for it either.

Consequently, these data are called **training-demonstration temporal characterization**, never held-out. B2 uses all 173 demonstration episodes in the four-task panel. Dataset actions are sampled at the dataset-declared 10 Hz; lags are reported in dataset control steps and seconds and are not conflated with the 20 Hz rollout controller.

For each integer lag `l=0..32`, report:

- per-dimension Pearson autocorrelation across all valid within-episode `(t,t+l)` pairs, with centering and scale computed from that lag's pooled pairs;
- per-dimension RMS `a(t+l)-a(t)` after applying the relevant frozen checkpoint action normalization;
- translation and rotation RMS normalized differences using dimensions 0..2 and 3..5;
- gripper normalized difference, sign agreement, and sign disagreement.

Gripper state is `sign(action[6])`, with zero retained as its own state. Adjacent-step transition frequency is the number of within-episode sign changes divided by the number of adjacent pairs. For every action step, distance to the next sign transition is reported in steps; observations with no later transition are right-censored and their censoring fraction is reported rather than assigned an invented distance. Episode-cluster percentile bootstrap uses 20,000 draws, 95% intervals, and seed 27301. Degenerate per-dimension correlations are reported undefined, without regularization.

ACT-normalized summaries use each task checkpoint's frozen mean/std. SmolVLA-normalized summaries, if shown, use its frozen global mean/std and are labeled as a separate within-policy scale. Raw action autocorrelation and gripper sign quantities do not depend on normalization.

## B3. Open-loop future-action predictability

This is a training-demonstration reference analysis, not held-out evaluation. It uses the same four tasks. To bound GPU time without outcome selection, freeze the numerically lowest ten training episode IDs per task:

- Object task3: 811, 812, 824, 843, 846, 849, 853, 858, 867, 871.
- Spatial task0: 1272, 1273, 1275, 1282, 1300, 1327, 1330, 1344, 1347, 1352.
- Goal task2: 385, 389, 396, 397, 404, 417, 419, 423, 427, 430.
- LIBERO-10 task3: 14, 15, 16, 31, 32, 36, 75, 89, 90, 97.

Within each episode, query anchors are frame indices `t=0,10,20,...` for which `t+32` exists. One frozen-policy chunk prediction is made from the recorded observation at each anchor; no environment is initialized and no action is executed. For every integer chunk offset `k=0..32`, compare the normalized predicted action for physical demonstration target `t+k` with the recorded demonstrated action at `t+k`, normalized using that checkpoint's already-frozen statistics.

Report per-dimension RMSE versus `k`, translation RMS error (dims 0..2), rotation RMS error (dims 3..5), gripper absolute normalized error, and gripper sign-disagreement probability. ACT and SmolVLA are reported separately in their own normalized spaces. The SmolVLA training-data relationship remains unknown from its model card and is stated as such; this uncertainty does not convert the cohort into held-out data.

Episode-cluster percentile bootstrap uses 20,000 draws, 95% intervals, seed 27401 for ACT and 27402 for SmolVLA. No performance outcome, task subset, lag, or offset is selected after results. These associations do not establish that persistence or forecastability causes executor sensitivity.

## B4. Failure-mode feasibility

The frozen Track-B artifacts contain executed actions, predictions, an initial simulator snapshot, terminal metadata, and episode length, but not a per-step object pose, end-effector pose, contact stream, task phase, or independently validated failure label. The existing Track-A schema likewise does not add those streams. Therefore:

`FAILURE_MODE_CLASSIFICATION_NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS`

No standalone rollout is authorized to reconstruct failure types. Any future reviewer-supplement passive logger requires a trajectory-identity canary and must never affect actions.

## Separation of mechanism quantities

The following remain distinct throughout tables, tidy data, prose inputs, and figure specifications:

1. demonstration action temporal persistence (B2);
2. frozen-policy future-action forecast error against a demonstration reference (B3);
3. same-target cross-source prediction disagreement on executed diagnostic trajectories (B1/original Track B).

They must not be collapsed into a generic “prediction error” quantity.


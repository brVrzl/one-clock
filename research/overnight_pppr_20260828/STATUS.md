# PPPR Overnight Status — 2026-08-28

## Repository

- Branch: `exp/libero-component-temporal-reuse`
- Starting and pre-milestone HEAD: `38046a961cd796b30b554c9de407d64aa82518cf`
- Starting tracked status: clean.
- Preserved local/untracked experiment directories: `act_temporal_ensemble_blind/`, `act_temporal_ensemble_blind_corrected_0404/`, and `two_clock_dev/`.
- Coherent PPPR milestone prepared for commit/push after this report. The resulting hash is recorded in the final operator handoff and branch log rather than self-referenced here.

## External ACT ensemble audit

- The prior LeRobot 0.6.2 result `3/80` is invalid and excluded.
- The independent corrected run is complete: `48/80`, with 18,886 policy queries over 18,886 environment steps (query rate `1.000000`).
- Verified runtime: `/home/wjq/workspace/venvs/libero_act/bin/python`, LeRobot 0.4.4, PyTorch 2.7.1+cu128, MuJoCo 3.3.1.
- Numerical parity: official versus repository same-target implementation maximum absolute error `3.761950195801944e-08` over 20 deterministic steps.
- Frozen cohort verified for every episode: requested and actual states 20–29, seeds 4000–4009.
- Per-task successes: object1 `3/10`, object4 `10/10`, spatial3 `6/10`, spatial7 `7/10`, goal0 `6/10`, goal3 `5/10`, libero_10 task1 `3/10`, libero_10 task9 `8/10`.

## PPPR Phase 0

### Label and split

- Development: object3, spatial0, goal2, libero_10 task3.
- Held-out offline: object5, spatial4, goal5, libero_10 task5.
- The split was frozen before PPPR score inspection and was not changed.
- Metric: development-Fresh per-dimension IQR scaling; separate translation/rotation normalized arm distance; bounded arm distance; sign-only gripper intent; fixed joint weight `0.5 arm + 0.5 grip`.
- Geometry: `u=t+k`, `r=2`, `M=4`, targets `{u+2,u+3,u+4,u+5}`, future sources exactly `{u,u+1,u+2}`.
- PPPR is the target median of `max(old-to-future-consensus distance - future-family dispersion, 0)`.
- Development-only IQR scales: `[0.413560, 0.265790, 0.607151, 0.024758, 0.071454, 0.052166]`; no zero-scale guard was used.
- Candidate rows: 249,376 total, 235,936 valid, 13,440 masked; 15,586 Fresh current chunks from 80 episodes.
- All 8 label semantic tests and 3 control-alignment/bootstrap tests pass.

### RawPPR versus PPPR distributions

Descriptive medians over valid feature rows at intervention ages 4/8/16:

| split | component mapping | RawPPR | PPPR |
|---|---|---:|---:|
| development | FullOld / joint | 0.179 | 0.050 |
| development | Reverse / arm | 0.344 | 0.088 |
| development | FO / gripper | 0.000 | 0.000 |
| held-out | FullOld / joint | 0.162 | 0.043 |
| held-out | Reverse / arm | 0.316 | 0.081 |
| held-out | FO / gripper | 0.000 | 0.000 |

The candidate rows are correlated and are not treated as independent inferential samples. Episode-condition gripper medians were nonzero: development RawPPR/PPPR `0.051/0.049`, held-out `0.017/0.013`.

### Control relevance

- Pair rows: 720; all decisive pairs: 181 (`104` harmful, `77` beneficial).
- Development decisive pairs: 98 (`63` harmful, `35` beneficial).
- Held-out decisive pairs: 83 (`41` harmful, `42` beneficial).
- Scores use only the matched Fresh cache at `(task, episode, t=u-d, k=d)` and only logged steps where the intervention actually uses the historical component source.
- Active logged steps: 130,633; valid Fresh-aligned steps: 106,644; 23,989 treatment-schedule steps lacked a valid Fresh-reference label and were reported, not filled.

| split | signal | AUROC | AUPRC |
|---|---|---:|---:|
| development | Age | 0.564 | 0.660 |
| development | Event | 0.112 | 0.457 |
| development | RawPPR | 0.439 | 0.600 |
| development | PPPR | 0.249 | 0.534 |
| held-out | Age | 0.649 | 0.743 |
| held-out | Event | 0.494 | 0.465 |
| held-out | RawPPR | 0.506 | 0.479 |
| held-out | PPPR | 0.491 | 0.462 |

Held-out episode-cluster 95% CIs for AUROC/AUPRC: Age `[0.571,0.743]/[0.425,0.875]`; Event `[0.149,0.796]/[0.248,0.896]`; RawPPR `[0.307,0.662]/[0.282,0.762]`; PPPR `[0.207,0.716]/[0.252,0.808]`. PPPR-minus-Raw AUROC CI: `[-0.159,0.097]`.

Held-out task-cluster 95% CIs for AUROC/AUPRC: Age `[0.587,0.688]/[0.433,0.791]`; Event `[0.154,0.545]/[0.241,0.482]`; RawPPR `[0.381,0.544]/[0.342,0.557]`; PPPR `[0.266,0.559]/[0.299,0.491]`. PPPR-minus-Raw AUROC CI: `[-0.121,0.014]`.

Held-out component-matched PPPR versus RawPPR AUROC: FullOld/joint `0.594/0.639`; Reverse/arm `0.389/0.389`; FO/gripper `0.519/0.496`.

Held-out task-pooled PPPR versus RawPPR AUROC: object5 `0.178/0.348`; spatial4 `0.500/0.554`; goal5 `0.310/0.381`; libero_10 task5 `0.400/0.373`. PPPR direction is not consistently useful and is strongly reversed on object5.

### Phase-0 decision

**FAIL.** Held-out PPPR AUROC is `0.491`, RawPPR is `0.506`, and the difference is `-0.015`. PPPR does not reach the roughly `0.65` guide, does not improve on RawPPR by roughly `0.05`, and does not show consistently strong rank separation.

## PPPR Phase 1

- Not run because Phase 0 failed. No predictor was trained.
- The existing Fresh action cache also has no proprioceptive state, but this limitation was not used to decide the Phase-0 failure.

## ACT confirmation

- Not run because the mandatory Smol Phase-0 gate failed.

## Closed-loop development

- Not run because the mandatory preceding gates failed.

## Active detached jobs

- None. GPUs were left idle by PPPR work.

## Scientific state

`STOP_PPPR`

## Next action

Review the negative Phase-0 report and keep PPPR stopped; do not train a PPPR predictor or launch ACT PPPR data generation.

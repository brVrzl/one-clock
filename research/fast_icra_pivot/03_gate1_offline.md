# Gate 1: offline repairability

Status: exploratory; the test split was inspected during method selection.

## Protocol

- Frozen ACT predictions and expert targets: 41 validation / 41 test episodes.
- Selection/fitting uses validation; reported numbers below use test.
- Ridge value 10, bootstrap seed 20260824, 10,000 paired episode bootstrap replicates.
- All methods use the same aligned targets. No unfavorable episode was removed.

## Results

| Method | Episode MSE | Relative change | Gripper sign accuracy | Paired MSE-difference 95% CI |
|---|---:|---:|---:|---:|
| Frozen ACT | 0.17647 | 0.0% | 70.36% | — |
| Clip `[-1,1]` | 0.17572 | -0.43% | 70.36% | [-0.00097, -0.00056] |
| Previous blend, α=0.25 | 0.17505 | -0.80% | 70.47% | [-0.00195, -0.00091] |
| EMA, α=0.25 | 0.16733 | -5.18% | 70.89% | [-0.01104, -0.00729] |
| Mean residual | 0.17635 | -0.07% | 70.34% | [-0.00046, 0.00022] |
| Task/position residual | 0.16697 | -5.38% | 71.82% | [-0.01298, -0.00598] |
| Affine residual, scale 0.25 | 0.15790 | -10.52% | 71.40% | [-0.02009, -0.01711] |
| Affine residual, scale 1.0 | 0.13431 | -23.89% | 76.47% | [-0.04714, -0.03727] |
| Affine + q25 gate | 0.13405 | -24.04% | 76.77% | [-0.04733, -0.03746] |

The q25 gate threshold was fixed from validation at residual chunk norm 3.1273. It was marginally better than always-repair offline. During online rollout it activated on 100% of queries, exposing distribution shift and eliminating the intended selectivity.

## Gate decision

Gate 1 passes narrowly: cheap state/phase calibration can recover expert-like chunks offline far better than smoothing. It does not establish closed-loop utility. The key risk is intervention-induced covariate shift, so no larger repair network was built.

## Adjacent candidate diagnostic

After continuous repair failed Gate 2, a structured bank shifted only the gripper sequence by `{-8,-4,0,+4,+8}` control steps. A ridge linear selector trained on validation oracle labels achieved on test:

| Selector | Frame MSE | Gripper MSE | Gripper sign accuracy |
|---|---:|---:|---:|
| Frozen ACT | 0.18045 | 0.93779 | 70.36% |
| Linear shift selector | 0.16592 | 0.83602 | 73.03% |
| Oracle shift selector | 0.15221 | 0.74007 | 76.24% |

The learned selector reduces overall MSE 8.1%, but chooses “no shift” on only 1.7% of chunks. It therefore has a plausible oracle gap but a serious closed-loop conservatism problem. Artifact: `artifacts/gripper_shift_reranking_20260824.json`.

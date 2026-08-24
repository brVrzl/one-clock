# ACT action-chunk error audit

## Data contract

The audit aligns frozen ACT chunks from the existing dense cache with expert actions from the LIBERO-Object LeRobot dataset. For a query at dataset frame `t`, chunk position `k` is compared with expert action `t+k`, truncated at episode end. The model was selected/fitted on 41 validation episodes (412,150 aligned targets) and evaluated on 41 test episodes (411,350 targets).

The test split was inspected during method selection, so all results are explicitly exploratory and cannot later be called confirmatory. Raw per-episode and per-dimension tables, plots, and JSON are under `artifacts/gate1_v2_20260824/`.

## Error structure

- Frozen ACT frame-weighted MSE: 0.18045; episode-weighted MSE: 0.17647.
- Constant per-dimension bias explains only 0.11% of MSE. A constant-offset ChunkFix formulation is unsupported.
- Lag-1 residual correlation is 0.95–0.99 in every action dimension. Errors are temporally structured rather than isolated spikes.
- The top 10% highest-error targets contain 36.4% of total squared error: localized failures exist, but errors are not sparse enough to justify spike removal alone.
- Gripper error accounts for 74.2% of total squared error. Frozen gripper sign accuracy is 70.36%.
- Residual standard deviations by dimension are `[0.185, 0.322, 0.426, 0.0216, 0.0518, 0.0357, 0.968]`.
- Error varies across chunk phase: position-bin MSE ranges from 0.142 to 0.242.
- Predicted actions have lower mean magnitude than expert actions (0.920 versus 1.211).

## Interpretation

The data reject “mostly correct chunks with a few isolated corruptions” as the main description. The useful structure is low-frequency/phase-correlated, dominated by gripper sign timing, with secondary state-dependent arm errors. This motivated an affine state/phase calibrator and later the discrete gripper-shift pivot. The plots in `error_structure.png` and `baseline_comparison.png` were generated with the project figure-style helper and visually checked after rendering.

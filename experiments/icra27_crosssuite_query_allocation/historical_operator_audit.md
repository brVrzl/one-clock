# Historical operator and implementation audit

Status: **PASS**. This audit used source code and synthetic predictions only; it loaded no Track-A outcomes.

## Canonical dense ACT temporal ensembling

- Implementation: `lerobot.policies.act.modeling_act.ACTTemporalEnsembler`.
- Query every environment step (`query_frequency=1`).
- For a physical target, candidates are ordered oldest source to newest source and weighted `exp(-0.01*i)`, with the oldest at `i=0`.
- The online LeRobot implementation matched an explicit offline same-target weighted sum in normalized action space across all seven dimensions.
- Synthetic maximum absolute discrepancy: `1.19e-07`.
- The checkpoint postprocessor/unnormalizer and environment postprocessor are applied only after temporal aggregation.

This is canonical dense ACT TE. It is not the historical sparse approximation in `temporal_operators.py`.

## Historical operators

| Operator | Nominal query rate | Candidate ages | Aggregation | Dimensions | Sign-vote output? |
|---|---:|---|---|---|---|
| M1_shared_te_h16 | 0.0625 | native same-target sparse sources, up to ACT chunk validity | continuous weighted average | 0..6 with one shared weight vector | False |
| M2_shared_cogact | 0.0625 | same pool as M1 | continuous weighted average | 0..6 with one compatibility-weighted vector | False |
| M3_group_cogact | 0.0625 | same pool as M1 | continuous weighted average with separately normalized arm/gripper weights | arm 0..5 and gripper 6 | False |
| canonical_dense_ACT_TE | 1 | dense sources ages 0..min(t,99) | continuous exponential weighted average | 0..6 with one shared temporal weight vector | False |

M3's `np.sign` forms compatibility weights only; its executed gripper scalar is still a continuous weighted average. CDTA likewise uses signs for weights, dynamic-horizon code uses them for triggers, and offline code uses them for metrics.

A distinct historical discrete output already exists: `rapid_component_smoke`'s `SIGN_VOTE` calls `weighted_gripper_vote`, chooses weighted open/close support, and executes an original candidate scalar from the winning sign. It must not be reinvented in this session.

## ACT and SmolVLA normalization buffers

- ACT: 40/40 audited checkpoint exports have matching preprocessor and postprocessor action mean/std buffers.
- SmolVLA: its frozen MEAN_STD action buffers match between preprocessing and postprocessing.
- Track B logs native model outputs before the checkpoint postprocessor, so dispersion is measured in each checkpoint's own frozen normalized space. No scale is refit on the diagnostic panel.

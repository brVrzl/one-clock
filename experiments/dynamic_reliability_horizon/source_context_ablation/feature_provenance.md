# Source-context feature provenance

Status: materialized and verified from the exact 3,740-row portable cohort.

## Cohort lock

- Portable bundle SHA256: `45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4`.
- Source key: `(episode_id, source_step) = (episode_index, frame_index)`.
- Duplicate source keys: `0`.
- Ambiguous state matches: `0`.
- Missing state matches: `0`.
- Train/validation/test episode leakage: `0`.
- `Y_refresh` was read from the existing bundle and was not regenerated. The historical target-side cache is absent in this checkout and that absence is recorded explicitly in `feature_manifest.json`.

## Exact causal state

`source_state` is the original LIBERO `observation.state` row at source time
`t`, stored as float32 shape `(8,)`: EEF Cartesian position (indices `0:3`),
EEF axis-angle orientation (indices `3:6`), and the two-finger gripper qpos
(indices `6:8`). No future row, episode-length value, normalized progress,
phase, success, or action is included in the feature vector.

## ACT representation candidates considered before fitting

1. Per-camera ResNet-18 `layer4` feature map: causal and reusable, but camera-local and not the fused policy context.
2. Final fused ACT encoder output: causal, shared with action prediction, and available before decoding. **Selected.**
3. Final ACT decoder token sequence: causal and action-proximal, but decoder-output features are more tightly coupled to the action head and carry a larger sequence.

The primary representation is therefore:

`z_t = policy.model.encoder(batch)[0, :, :]`, where the first token is the
source-conditioned ACT latent/context token after the final encoder layer.
For each source it is shape `(512,)`, float32. The encoder input contains the
current normalized `observation.state`, current agent-view and wrist images,
and ACT's all-zero inference latent token. It is extracted by a forward hook
on `policy.model.encoder` immediately before `policy.model.decoder`; the hook
does not alter the returned action chunk. The training-time VAE encoder was
not selected because its input includes demonstration actions.

## Verification

- Frozen ACT action chunks recomputed from the exact source state/images match
  the bundle within the recorded tolerance in `feature_manifest.json`.
- On a deterministic subset, enabling the latent hook changes the postprocessed
  frozen ACT chunk by the recorded maximum absolute delta and passes the
  `1e-6` allclose check.
- Feature array SHA256, shapes, dtypes, checkpoint checksums, and dataset
  provenance are recorded in `feature_manifest.json`.

The feature artifact is immutable; the materializer refuses to overwrite it.

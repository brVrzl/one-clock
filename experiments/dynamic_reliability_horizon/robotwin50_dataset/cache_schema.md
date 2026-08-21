# RoboTwin 50 policy-response cache schema

The cache is policy-specific and lives outside git at
`/home/wjq/robotwin_reliability_cache/<policy_revision>/`.

## Policy-output episode shard

`policy_outputs/<canonical_task_or_task_index>/episode_<episode_index>.npz`
is written through a temporary file and `os.replace` only after validation.

`manifests/cache_manifest.json` records the exact dataset/checkpoint revisions,
normalization mapping, group-schema path, and the rule that future outputs are
label-side only. `checkpoint_contract.json` records the locally audited model
and camera contract.

Before inference, the checkpoint must also provide `action_schema.json` with an
`action_names` (or `ordering`) list equal to the dataset's verified motor
ordering. A matching width alone is not sufficient to define the four clocks.

- `episode_index`: `int64 [N]` repeated episode identifier.
- `frame_index`: `int64 [N]`, dataset frame indices in episode order.
- `task_index`: `int64 [N]`.
- `task`: `str [N]`, exact dataset prompt.
- `action_chunks`: `float32 [N, 50, 14]`, complete de-normalized frozen-policy output.
- `z_t`: `float32 [N, D]` only when a non-invasive prefix representation is exposed;
  an empty second dimension means unavailable.
- `noise_seed`: `int64 [N]`, deterministic inference seed used for the frozen call.

There is exactly one policy call per unique `frame_index`. The cache never stores
future observations or future demonstrated actions as estimator features.

## Offline label episode shard

`labels/<task_key>/episode_<episode_index>.npz` contains arrays with shape
`[N, 4, 50]`:

- `raw_distances`: `float32`, NaN only where the future frame is censored.
- `validity`: `uint8`, thresholded instantaneous validity.
- `y_refresh`: `uint8`, prefix product of observed validity from offset zero.
- `censor_mask`: `uint8`, true only where `t+k` exists in the same episode.
- `offset_k`: `int64 [50]`, `0..49`.
- `group_names`: `str [4]`, verified action groups.

`A_{t+k}` appears only through this label-side construction. The target builder
retains raw distances and does not expose future values to estimator tensors.

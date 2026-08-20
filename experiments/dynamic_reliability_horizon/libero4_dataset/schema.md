# LIBERO-4 reliability-data schema

The canonical source is the pinned LeRobot dataset `lerobot/libero` at
revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. The source lives outside
Git because its videos are large. `dataset_manifest.json` records its absolute
path, source metadata checksums, and the observed scale.

## Policy-independent corpus

`frame_index.parquet` has one row per source frame, in dataset `index` order:

| field | meaning |
|---|---|
| `frame_id` / `dataset_index` | contiguous canonical source-frame ID |
| `episode_id`, `frame_index` | episode and frame-local coordinates |
| `suite`, `task_id`, `task_name` | verified LIBERO-4 task metadata |
| `timestamp` | source tabular timestamp in seconds |
| `data_path`, `data_row_index` | reference to pinned LeRobot parquet row |
| `state_ref` | reference to `observation.state` in the pinned parquet row |
| `observation_image_ref`, `observation_image2_ref` | video path and timestamp references; image bytes are not copied |
| `demonstrated_action` | raw source 7-D action, float32-compatible |
| `episode_start_frame_id`, `episode_end_frame_id` | episode boundary lookup |
| `split` | episode-level train/validation/test membership |

`episode_split.json` is the only split authority. The split unit is an episode,
and the same deterministic SHA-256 task-stratified rule is applied to every
task. No frame is independently assigned.

## Future lookup

`source_window_index.npz` contains:

- `future_frame_ids`: int32 `[273465, 100]`, offsets `1..100`, `-1` when right-censored;
- `observed`: bool `[273465, 100]`, false after an episode boundary;
- `offsets`: int16 `[100]`.

No lookup entry crosses an episode boundary. `K=20`, `K=50`, and `K=100`
policies use the same canonical lookup; a cache chunk shorter than 100 simply
marks unavailable old-chunk offsets as unobserved during target construction.

## Policy-response cache

Each atomic `policy_cache/shard-*.npz` contains one response per unique current
frame: `frame_id`, episode/task metadata, `source_chunks` `[N, chunk_length, 7]`,
and optional `source_latents` `[N, latent_dim]`. The adapter is called once per
frame. The cache manifest records normalization and checkpoint metadata. The
episode/task/frame/split columns are grouping and audit fields only, not
estimator features.

## Reliability labels

Each `reliability_labels/shard-*.npz` keeps label-side values separate from
estimator-visible source cache values. `raw_group_distances` is `[N,100,2]`
float32 and is stored before thresholding. `Y_refresh` is `[N,100,2]` prefix
survival; `label_observed` is the right-censoring mask. Future observations and
future actions are never source features.

The fixed compatible convention is:

- arm indices `[0:6]`: translation and rotation normalized RMS; raw arm distance
  is their maximum and validity requires it to be at most `1.0`;
- gripper index `[6]`: normalized absolute error, with threshold `1.0` and
  command-sign agreement.

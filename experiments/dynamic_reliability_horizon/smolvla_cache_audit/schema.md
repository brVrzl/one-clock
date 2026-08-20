# SmolVLA cache schema

This audit keeps causal model-side inputs separate from offline label-side
supervision.

## Model side: `model_side_cache.npz`

- `source_frame_id`: `int64 [N]`, the unique dataset frame identifier.
- `predicted_action_chunk`: `float32 [N, 50, 7]`, the full de-normalized
  SmolVLA chunk before execution truncation.
- `z_t`: `float32 [N, D]` when the non-invasive prefix hook succeeds. The
  current candidate is the mean over the sequence of the final hidden states
  returned by the normal source-prefix VLM call. It is source-time only and is
  shared by all action offsets from that frame.
- `group_names`: `str [2]`, `['arm', 'gripper']`.

The action groups are `arm = [0,1,2,3,4,5]` and `gripper = [6]`.

## Label side: `label_side_cache.npz`

- `source_frame_row`: `int64 [W]`, row indices into the model-side cache.
- `future_frame_row`: `int64 [W, 2, K]`, lookup rows for source `t+k`, or `-1`
  when censored.
- `y_refresh`: `uint8 [W, 2, K]`, group-wise prefix-survival labels for
  offsets `k = 1..K`.
- `censor_mask`: `uint8 [W, 2, K]`, true only when `t+k` exists in the same
  episode.
- `offset_k`: `int64 [K]`, explicitly `[1, ..., K]`; there is no k=0 label.
- `group_names`: `str [2]`.

The smoke harness uses an explicit action-space L-infinity tolerance for its
correctness check. The production target must use the already audited Thor
target rule when the portable handoff is available.

## Complexity contract

The cache path performs one frozen-policy forward per unique frame. A naive
direct path performs one source query plus one future query per `(source,
offset)` pair. The harness reports both forward counts and their ratio.

No future observation or future policy response is written into
`model_side_cache.npz`.

# LIBERO-4 data audit

Pinned source: `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.

## Coverage

- all four suites represented: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-Long;
- all 40 task indices and all 1,693 episode indices represented;
- all 273,465 frame IDs are contiguous and appear once;
- every action is finite and 7-D; every state is finite and 8-D;
- image data remains referenced through the pinned video files; no image bytes are copied into the corpus.
- The pinned source download listing reports no failed files.

## Split and leakage checks

- `episode_split.json` applies one deterministic task-stratified SHA-256 rule at episode level;
- each episode has one split and every frame in that episode has the same split;
- `source_window_index.npz` uses `-1`/`observed=false` for right-censored offsets and is constructed inside episode ranges only;
- estimator-visible source records contain current-frame references and demonstrated action only; future lookup is a separate label-side artifact;
- no episode length, progress, phase, terminal flag, or future observation/action is stored in source features.

## Verified action contract evidence

The pinned data metadata verifies action shape `[7]`. The installed LIBERO/robosuite source verifies OSC_POSE ordering as three end-effector position deltas, three axis-angle rotation deltas, then one gripper command. Group definitions are therefore arm `[0:6]` and gripper `[6]`; they were not inferred from cached values.

- `/home/thor/projects/upstreams/lerobot-env/lib/python3.12/site-packages/libero/libero/benchmark/libero_suite_task_map.py` SHA256 `0c950df0a785aa55de968bb38ccd865d2017f71ddbe6f48cfd05ac0742b6d62d`
- `/home/thor/projects/upstreams/lerobot-env/lib/python3.12/site-packages/libero/libero/envs/env_wrapper.py` SHA256 `a782fb76c9792268d28979474fe72849e1e98ada49c8e997a65359a8d6b6acd0`
- `/home/thor/projects/upstreams/lerobot-env/lib/python3.12/site-packages/robosuite/utils/input_utils.py` SHA256 `f792c59fdc2860e29bf3fb5619796b94b2aea1b041822be1a3598fe12febe8ac`

The per-task episode/frame min/median/max statistics are in `episode_split.json` and are copied into `dataset_manifest.json` by reference through the split artifact.

## Optional SmolVLA compatibility

- checkpoint: `HuggingFaceVLA/smolvla_libero@6721902bc4d61e50a3bfdb11dfb4cb626f05d102`;
- current-frame observation decode/preprocessing reached two 256x256 RGB images and an 8-D state;
- policy construction was blocked on Thor because the installed environment lacks `num2words`, required by the SmolVLM processor;
- no SmolVLA policy-response cache or reliability-label shards were started;
- the complete traceback and the pinned runtime details are in `smolvla_compatibility.json`.

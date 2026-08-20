# Handoff to 5080

The primary deliverable is policy-independent and can be transferred without
the downloaded dataset videos:

- `dataset_manifest.json`
- `episode_split.json`
- `frame_index.parquet`
- `source_window_index.npz`
- `schema.md`
- `data_audit.md`

The source revision is fixed at
`lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. Do not combine these
files with another revision. Large-file absolute paths, byte sizes, and SHA256
values are in the manifests.

On Thor, the bounded SmolVLA check reached observation decoding but policy
construction was blocked because the active environment lacks `num2words`, a
SmolVLM processor dependency. The exact traceback is in
`smolvla_compatibility.json`; no SmolVLA cache was started. The pinned
checkpoint is
`HuggingFaceVLA/smolvla_libero@6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.

If the SmolVLA compatibility result is feasible, transfer the separately
recorded policy-cache and reliability-label directories listed in
`progress.json`, together with their manifests. Otherwise, on the 5080 use the
same canonical source revision and run:

```bash
cd experiments/dynamic_reliability_horizon/libero4_dataset
python3 build_policy_cache.py \
  --dataset-root /path/to/pinned/lerobot_libero \
  --corpus-dir "$PWD" \
  --output-dir /path/to/policy_cache/<policy-id> \
  --adapter <module>:<factory> \
  --episodes-per-shard 20
```

The adapter must receive only one current observation and return the complete
frozen-policy action chunk (and optionally one current representation). Never
feed future observations, future actions, episode length, progress, phase, or
terminal metadata to the estimator. Then build raw distances and labels with
`build_reliability_targets.py`; raw distances must be retained for threshold
sensitivity audits.

`episode_id`, `task_id`, `frame_index`, `frame_id`, and split membership remain
grouping/split fields. Do not pass them as estimator features.

No new phase-oracle search, rollout, estimator training, executor change, or
paper change is part of this handoff.

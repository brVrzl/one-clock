# Portable Y_refresh handoff

Generated from repository commit `bd9245e152e549f4252579ee2f9f04a256c8dd6b`. This is
an offline data handoff only. No reliability network was trained, no rollout
was run, and executor semantics and paper claims were not changed.

## What to transfer

Transfer these two files together:

* `minimal_y_refresh_training_bundle.npz` — SHA256 `45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4` (9955742 bytes)
* `episode_split_manifest.json` — SHA256 `1b57c9d12e17292e66276002966e0aea44ff1c7e54f35d335cc5670ce9de439a` (5587 bytes)

The bundle contains only the known source-time predicted ACT chunk, group IDs,
candidate offsets `k=1..99`, `Y_refresh` prefix labels, label-observation masks,
episode IDs, and episode-level split membership. It contains no future
observations, future demonstration actions, episode length, progress, phase,
or terminal metadata. Do not pass `episode_index` or `split_membership` as model
features; they are grouping/split fields.

Array schema and all provenance checksums are in `handoff_manifest.json`.

## Contract limitation

The requested estimator contract commit `928ffba` is not present in this
clone, and no materialized source-time image/state/ACT-latent tensor exists in
the cached re-query artifact. Therefore this is the minimal known causal
source-chunk handoff, not a certification that the unavailable contract is
fully satisfied. Training can proceed from this bundle only if the 5080
implementation consumes the source chunk (plus group/offset context) as its
causal feature input. If it requires current observation images/state or a
frozen-ACT latent, transfer those exact causal inputs separately and record
their checksums; do not use future data or silently redesign the estimator.

## External provenance

The original cache was built from the local dataset
`/home/thor/datasets/libero_object_25_08_23_lerobotv2.1` (Hugging Face
`DorayakiLin/libero_object_25_08_23_lerobotv2.1`, local revision
`cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a`) and frozen ACT checkpoint
`/home/thor/projects/checkpoints/zeromidnight_act_libero_object`. These large
external files are not committed. They are required to reproduce the cached
Y_refresh labels, but not for bundle-only training when the contract condition
above holds. The manifest records checkpoint config/model checksums without
copying credentials.

## Verification on 5080

After transferring the directory, replace `/path/to` in the following exact
command with its destination root:

```bash
HANDOFF_DIR=/path/to/experiments/dynamic_reliability_horizon/artifact_handoff; sha256sum "$HANDOFF_DIR/minimal_y_refresh_training_bundle.npz" "$HANDOFF_DIR/episode_split_manifest.json"; python3 -c 'import hashlib, pathlib, sys; expected={"minimal_y_refresh_training_bundle.npz":"45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4", "episode_split_manifest.json":"1b57c9d12e17292e66276002966e0aea44ff1c7e54f35d335cc5670ce9de439a"}; [(_ for _ in ()).throw(SystemExit("checksum mismatch:"+p)) if hashlib.sha256((pathlib.Path(sys.argv[1])/p).read_bytes()).hexdigest()!=v else None for p,v in expected.items()]' "$HANDOFF_DIR"
```

The expected result is two matching SHA256 lines and no `checksum mismatch`
exception. Then load the NPZ with `allow_pickle=False` and assert every shape
and dtype in the manifest before any training job is started.

## Label semantics

`Y_refresh` is teacher-forced frozen-policy replanning consistency: a source
chunk is compared with the same policy queried at a demonstrated future
observation. It is not rollout success supervision. `k=0` is omitted from the
portable labels as a trivial identity check. Censored offsets are represented
by `label_observed=False` and must not be treated as negative labels.

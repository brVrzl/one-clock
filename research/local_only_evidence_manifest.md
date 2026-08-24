# Local-only evidence manifest

Audit date: 2026-08-21; updated 2026-08-24 through Gate-3A2. This manifest records primary artifacts intentionally
excluded from the audit commit because they are raw rollouts, prediction/data
arrays, checkpoints, or large datasets. They remain available at the paths
below in the current workspace. No claim below should be treated as
reproducible from the Git commit alone unless the corresponding compact audit
output is also committed.

For directories, `content-tree SHA256` is the SHA-256 of the concatenation of
sorted per-file `sha256sum` lines generated from relative paths inside that
directory. It is an integrity digest for the directory contents, not a Git
object ID. Directory sizes are apparent byte sizes from `du -sb`; file counts
are included for scope.

## Frozen policy and dataset

| Local-only artifact | Size | SHA256 / content-tree digest | Audit claims that depend on it |
|---|---:|---|---|
| `/home/thor/projects/checkpoints/zeromidnight_act_libero_object/model.safetensors` | 206,712,028 bytes | `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410` | ACT checkpoint identity, action semantics, every offline/rollout recomputation |
| `/home/thor/projects/checkpoints/zeromidnight_act_libero_object/config.json` | 1,761 bytes | `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d` | ACT chunk length, feature/action dimensions, preprocessing contract |
| `/home/thor/datasets/libero_object_25_08_23_lerobotv2.1` (4,104 files) | 542,093,962 bytes | content-tree `2c7b87d23936dcd9d511c77234907f99e2da8ac4d23b68bb7b23af9b71297608` | 454 demonstrations, 66,984 frames, task identities, demonstration-consistency and temporal-expert audits, Gate-3A0 control-semantic targets, Gate-3A1 inventory |
| `/home/thor/datasets/libero_object_25_08_23_lerobotv2.1/meta/episodes.jsonl` | 46,501 bytes | `63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea` | Episode/task metadata and dataset accounting |
| `/home/thor/datasets/libero_object_25_08_23_lerobotv2.1/meta/episodes_stats.jsonl` | 1,119,439 bytes | `5bf31fb80b359c9fd1d56a0eaa27f8e7c76a7e39678487fdf76986af8fe88dca` | Frame/episode statistics and source-window construction |

## Raw rollout traces

| Local-only artifact | Size | SHA256 / content-tree digest | Audit claims that depend on it |
|---|---:|---|---|
| `/home/thor/projects/one-clock/experiments/runs/libero_static_grid_20` (140 files) | 65,729,472 bytes | content-tree `696f3e5a70ab328815d3dede9457b396c5310d7d311dc506ef0f6c4aa5f16cd8` | Task-0 20-state Gate-0 success/query curves |
| `/home/thor/projects/one-clock/experiments/runs/libero_static_grid_50_extension` (128 files) | 96,539,228 bytes | content-tree `01e7b951be9b6dc0afd6ecdc227027e8de098d6c95084eb0672f86e09e34eed5` | Task-0 50-state horizon and `(4,16)` comparison |
| `/home/thor/projects/one-clock/experiments/runs/libero_object_cross_task` (512 files) | 245,537,731 bytes | content-tree `1684ccf77d26784d1818824de705ef67c5584793a33cb13d6dc9b7995f6798dc` | Ten-task static configuration analysis |
| `/home/thor/projects/one-clock/experiments/groupwise_selective_commitment` (140 files) | 23,591,126 bytes | content-tree `d295c50aaf35a3d0209fb9d4074b1c37b4aa6611c2a45da9e907d104baff836b` | 1,200-episode matched-query selective-commitment negative result |
| `/home/thor/projects/one-clock/experiments/gate3a2_temporal_aggregation` (400 compressed episode files) | 10,191,635 bytes | content-tree `d61f9850d0ee283dc823a8e7a208c02bac7b33bee196c27bcb1d6dd565131a4c` | Gate-3A2's 400/400 binary outcomes, 85,942 executed steps/ACT queries, per-step candidate/action integrity, success differences, secondary action diagnostics, and `CONTROL-LINK-POSITIVE` decision |

The committed `research/audit_outputs/rollout_artifact_inventory.csv` retains
per-run metadata, trace hashes, and validation issues for the historical
rollout roots. Gate-3A2 uses its separate committed
`research/audit_outputs/gate3a2_rollout_manifest.json`, which records the
absolute path, byte size, SHA256, outcome, step/query count, and provenance for
each of the 400 local-only logs.

## Raw prediction and aggregate inputs

| Local-only artifact | Size | SHA256 | Audit claims that depend on it |
|---|---:|---|---|
| `/home/thor/projects/one-clock/experiments/group_prediction_persistence/predictions.npz` | 3,617,038 bytes | `ef089cdac3fe46a4c40e7f1453ef1e1fd0a2fff189fb4e8a42dc00b06d9da535` | Gate-1 sparse temporal error profiles |
| `/home/thor/projects/one-clock/experiments/temporal_reliability/reliability_dataset.npz` | 12,857,759 bytes | `6e72b16a1acc90b6187f4c007848345e73090db3e362ba39f0e7f3095b4c0c60` | Demonstration-support target, threshold sensitivity, smoothness analysis, sparse temporal candidates for Gate-3A0, Gate-3A1 episode/task inventory |
| `/home/thor/projects/one-clock/experiments/dynamic_reliability_horizon/artifact_handoff/minimal_y_refresh_training_bundle.npz` | 9,955,742 bytes | `45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4` | Y_refresh estimator replication, source-context ablation, episode-level validation/test membership for Gate-3A0 and Gate-3A1 |
| `/home/thor/projects/one-clock/experiments/phase_conditioned_oracle/config_results.json` | 1,838,818 bytes | `33f014aa81b640bc05b1fc3aaec85e516009d50eb6cd624d3e1753c499d3434b` | Gate-2B candidate-map accounting, curves, and selected maps |
| `/home/thor/projects/one-clock/experiments/gate3a1_dense_temporal_cache` (82 prediction NPZ files; 83 files including the local manifest) | 32,949,112 bytes | content-tree `87e97a5711a7b51ea53da908774040d2b23ca57e9c29699bfd25f28ebe31908c` over the 82 sorted prediction-file hashes | Gate-3A1 dense validation/test analysis: 12,294 full ACT chunks with logical shape `(12,294, 100, 7)`, non-oracle comparisons, stratified diagnostics, and scalar oracle headroom |

The original fresh-action arrays and construction script for the Y_refresh
target (`target_comparison.npz`, `refresh_first_actions.npz`, and
`compare_targets.py`) are absent from the workspace. Their exact size and
SHA256 therefore cannot be reported; this absence is itself recorded in the
source-of-truth map and claims ledger.

## Deliberately excluded generated files

The audit commits exclude generated PNG plots, Python `__pycache__` bytecode,
and raw prediction/rollout caches. Non-plot replication tables and metrics
needed for the numerical claims are committed under
`research/audit_outputs/chunk_only_replication/` and
`research/audit_outputs/source_context_replication/`, with Gate-3A1 compact
outputs directly under `research/audit_outputs/`; the omitted plots and dense
prediction arrays are not needed to inspect the reported aggregate claims.

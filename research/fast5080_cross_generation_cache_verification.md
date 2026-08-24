# RTX 5080 cross-generation cache verification

Verification date: 2026-08-24 (Asia/Shanghai). This note records the reusable
assets completed before the cross-generation offline protocol was evaluated.
No Thor Gate-3B result was read.

## Frozen assets

| Asset | Local path | Verification |
|---|---|---|
| ACT weights | `/home/wjq/checkpoints/zeromidnight_act_libero_object/model.safetensors` | 206,712,028 bytes; SHA256 `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410` |
| ACT config | `/home/wjq/checkpoints/zeromidnight_act_libero_object/config.json` | 1,761 bytes; SHA256 `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d` |
| Dataset | `/home/wjq/datasets/libero_object_25_08_23_lerobotv2.1` | `DorayakiLin/libero_object_25_08_23_lerobotv2.1@cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a`; 454 episodes, 66,984 frames, 10 tasks; immutable payload-tree SHA256 `7c5cb7e88722e0aead2fe0853bdf54e076afe77364a3204ecf46f1e5e7a05b7b` excluding downloader cache records |
| LeRobot | `/home/wjq/workspace/upstreams/lerobot` | Git HEAD `f66e5128ecb2456e8c54a63d15404fa59c16aebc`; package `0.6.2` |

The dataset's audited `episodes.jsonl` and `episodes_stats.jsonl` hashes are
respectively `63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea`
and `5bf31fb80b359c9fd1d56a0eaa27f8e7c76a7e39678487fdf76986af8fe88dca`.
Its nominal 10 Hz metadata is retained as file provenance but is not used as
physical cadence; the committed time-contract audit establishes one stored
index as one 20 Hz controller tick.

## Independent dense cache

The RTX cache is local-only at
`/home/wjq/workspace/one-clock/experiments/gate3a1_dense_temporal_cache`.
Gate-3A1's validator recomputed every NPZ hash and confirmed 41 validation plus
41 test episodes, 6,151 plus 6,143 queries, and logical prediction shape
`(12,294,100,7)`. The 82 NPZ files occupy 32,935,012 bytes and have content-tree
SHA256 `7e14e1f341bc2425cb3304cc3f35b0075184b0b1f33225e2dcf05cfe67e50f65`.
The working manifest SHA256 after validation is
`dac1ea14913234dff226896687183a155375cda4dc3adf45590a98c6fdddfa7e`.

Generation used Python 3.12.3, PyTorch `2.11.0+cu130`, CUDA 13.0, NumPy 2.2.6,
pandas 2.3.3, SciPy 1.18.0, PyArrow 25.0.1, PyAV 15.1.0, deterministic
algorithms, no AMP, and an NVIDIA GeForce RTX 5080. RTX arrays are an
independent replication and are not expected to be byte-identical to Thor.

## Gate-3A1 reproducibility gate

The committed analysis self-test passed, both cache splits validated, and the
held-out 41-episode ordering reproduced under `L_sem`:

| Method | RTX episode-weighted `L_sem` |
|---|---:|
| newest-age exponential | 0.6020883782 |
| tuned CogACT | 0.6255272419 |
| control-semantic | 0.6269329789 |
| newest-only | 0.7395639097 |

Thus newest-age is best, tuned CogACT and semantic are close, and newest-only is
worst. The small numerical offsets from Thor do not change the scientific
ordering, so the cache is accepted for the fixed composition audit.

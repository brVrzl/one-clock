# RTX 5080 Gate-3A1 replication setup

Setup date: 2026-08-24 (Asia/Shanghai). Host role: independent RTX 5080
replication; no Thor cache or rollout artifact was transferred.

## Frozen public assets

| Asset | Public identity | Local path | Verification |
|---|---|---|---|
| ACT checkpoint | `zeromidnight/act_libero_object@9cb23a1fda9e0cf319af4a3b4aefddb1ee02910f` | `/home/wjq/checkpoints/zeromidnight_act_libero_object` | `model.safetensors` SHA256 `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`; `config.json` SHA256 `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d` |
| LIBERO Object dataset | `DorayakiLin/libero_object_25_08_23_lerobotv2.1@cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a` | `/home/wjq/datasets/libero_object_25_08_23_lerobotv2.1` | 454 episodes, 66,984 frames, 10 tasks, 10 Hz; `episodes.jsonl` SHA256 `63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea`; `episodes_stats.jsonl` SHA256 `5bf31fb80b359c9fd1d56a0eaa27f8e7c76a7e39678487fdf76986af8fe88dca` |
| LeRobot source | `huggingface/lerobot@f66e5128ecb2456e8c54a63d15404fa59c16aebc` | `/home/wjq/workspace/upstreams/lerobot` | exact Git HEAD; package version `0.6.2` |

The 39-character dataset revision written in the fast-track request
(`cbf7122d530...`) is not a public commit. The 40-character revision above is
the repository-audited pin already recorded by Gate-3A1 and is still public.
No alternate dataset was used.

The downloaded scientific payload contains 1,367 files (454 parquet files,
908 videos, four metadata files, and `.gitattributes`) with 541,468,353 regular
file bytes. Its stable relative-path payload-tree SHA256 is
`7c5cb7e88722e0aead2fe0853bdf54e076afe77364a3204ecf46f1e5e7a05b7b`.
The historical Thor digest
`2c7b87d23936dcd9d511c77234907f99e2da8ac4d23b68bb7b23af9b71297608`
covered a 4,104-file local directory containing Hub downloader metadata and
lock files. The current Hub client/mirror produced 4,102 files and embeds
download timestamps/endpoints in those cache records, so that whole-local-dir
digest is not portable. The immutable revision, complete payload accounting,
and the two preregistered metadata hashes reproduce exactly; this is a layout
change in disposable download metadata, not a scientific-content substitution.

## Local runtime and cache contract

The project runtime is a local Python 3.12 virtual environment with PyTorch
`2.11.0+cu130`, CUDA runtime `13.0`, LeRobot `0.6.2`, NumPy `2.2.6`, pandas
`2.3.3`, SciPy `1.18.0`, PyArrow `25.0.1`, and PyAV `15.1.0`. A CUDA smoke
test identified `NVIDIA GeForce RTX 5080`, and the frozen checkpoint loaded as
an `ACTPolicy` with chunk size 100, policy-side temporal ensembling disabled,
and parameters on `cuda:0`. The pinned LeRobot checkout is supplied explicitly
through the new `--lerobot-root`
argument; the cache builder still rejects any HEAD other than the audited
commit. Checkpoint hashes, dataset metadata, episode inventory, observation
round trips, full `(100,7)` outputs, finite values, and source-frame coverage
remain fail-closed checks.

The local cache target is
`/home/wjq/workspace/one-clock/experiments/gate3a1_dense_temporal_cache` and is
Git-ignored. The expected independent-replication scope is 41 validation plus
41 test episodes, 6,151 plus 6,143 source queries, and logical shape
`(12,294,100,7)`. RTX numerical arrays are not required to be byte-identical
to Thor; the preregistered scientific ordering and pairwise conclusions are.

# RTX 5080 branch synchronization audit

- Timestamp: `2026-08-24T10:08:38+08:00`
- Host: `xdl-MS73-HB1-000`
- Repository: `brVrzl/one-clock`
- Original local branch: `integrate/5080-into-main`
- Original local HEAD: `50df73931843eb564a669c74321e8c1cf88f38fd`
- Original worktree: clean (`git status --short` produced no entries)
- Audit branch: `exp/fast5080-adaptive-recency`
- Audit branch creation point: `1ce9bf0eb1443abb7452086ac85a7c4ed0ea5752`

## Remote synchronization

`origin` is `git@github.com:brVrzl/one-clock.git`. A broad
`git fetch --all --prune --tags` was attempted, but the local Git process was
killed by signal 9. SSH authentication and repository access were then verified
independently. Remote heads were enumerated by bounded Git protocol-v2 prefix
queries, and every discovered branch was fetched explicitly. The remote has no
tags. The discovered remote heads are:

| Remote branch | HEAD SHA |
|---|---|
| `origin/main` | `6ed5d06516aaddb382095e3343430c7e31cd22d7` |
| `origin/exp/groupwise-selective-commit-act` | `e3ff506caed44fb685a16e4a1158b6c5de6ac2bc` |
| `origin/exp/robotwin-static-validation` | `d04cdf5fa11049a26d18f38138c4f912a4534c0f` |

Neither `origin/exp/gate3a2-control-link` nor
`origin/exp/fast5080-adaptive-recency` existed at audit time. No unexplained
remote 5080 history therefore required preservation or recovery.

## Branch relationship table

Counts are relative to scientific base
`1ce9bf0eb1443abb7452086ac85a7c4ed0ea5752`; "ahead" means branch-only
commits and "behind" means base-only commits.

| Remote branch | Merge-base with scientific base | Ahead | Behind |
|---|---|---:|---:|
| `origin/main` | `6ed5d06516aaddb382095e3343430c7e31cd22d7` | 0 | 7 |
| `origin/exp/groupwise-selective-commit-act` | `1ce9bf0eb1443abb7452086ac85a7c4ed0ea5752` | 1 | 0 |
| `origin/exp/robotwin-static-validation` | `4df1a12e0d2db4364ec7b47fa993f00b6efd18b9` | 5 | 25 |

The required base commit exists locally and is an ancestor of
`origin/exp/groupwise-selective-commit-act` (`git merge-base --is-ancestor`
returned success).

## Cross-branch content audit

- `origin/main` contains no commits absent from the scientific base.
- The sole Gate-3A2 branch-only commit,
  `e3ff506caed44fb685a16e4a1158b6c5de6ac2bc`, adds Gate-3A2-specific
  schedules, rollout/analyzer code, manifests, protocol, and tests. It is
  in-progress scientific-track code and was not imported.
- The five RoboTwin-only commits (`a059123`, `62b1cbc`, `022fa84`, `7ededfb`,
  `d04cdf5`) add a RoboTwin/SmolVLA-specific dataset/cache pipeline, manifests,
  contract audits, and tests. They do not supply missing LIBERO/ACT fast-track
  infrastructure and were not imported.
- The selected base already contains the verified LIBERO environment/action
  contract, ACT preprocessing and postprocessing path, dense Gate-3A1 analysis,
  bootstrap/statistical audit utilities, manifests, and tests needed for this
  track.

No commits were cherry-picked or merged. No Thor results were inspected, and
no scientific conclusion crossed branches.

## Local-only asset audit

| Asset | Local path/status | Size | Verified identity |
|---|---|---:|---|
| Gate-3A1 dense prediction cache | absent; expected transfer source `/home/thor/projects/one-clock/experiments/gate3a1_dense_temporal_cache` | — | expected content-tree SHA256 `87e97a5711a7b51ea53da908774040d2b23ca57e9c29699bfd25f28ebe31908c` |
| Frozen ACT checkpoint | absent; expected source `/home/thor/projects/checkpoints/zeromidnight_act_libero_object` | — | expected `model.safetensors` SHA256 `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`; expected `config.json` SHA256 `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d` |
| LIBERO Object dataset | absent; expected source `/home/thor/datasets/libero_object_25_08_23_lerobotv2.1` | — | expected HF identity `DorayakiLin/libero_object_25_08_23_lerobotv2.1`, local revision `cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a`, 454 episodes / 66,984 frames / 10 Hz |
| Portable Y-refresh training bundle | `experiments/dynamic_reliability_horizon/artifact_handoff/minimal_y_refresh_training_bundle.npz` | 9.5 MiB | SHA256 `45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4` |
| Episode split manifest | `experiments/dynamic_reliability_horizon/artifact_handoff/episode_split_manifest.json` | 5.5 KiB | SHA256 `1b57c9d12e17292e66276002966e0aea44ff1c7e54f35d335cc5670ce9de439a` |

The two present metadata/bundle artifacts match their committed handoff
manifest. No external asset was copied because this host has no configured Thor
hostname, IP address, SSH alias, or mounted Thor storage. Any later transfer
must be byte/content-tree verified before analysis or rollout. Large artifacts
remain outside Git.

## Decision

**SCIENTIFIC BASE:**
`1ce9bf0eb1443abb7452086ac85a7c4ed0ea5752`

**NO HISTORICAL BRANCH MERGE REQUIRED**

Repository provenance is clean and the fast-track branch is correctly
isolated. Scientific execution is not yet authorized by the asset gate: the
dense cache, frozen ACT checkpoint, and LIBERO dataset must be transferred and
verified first.

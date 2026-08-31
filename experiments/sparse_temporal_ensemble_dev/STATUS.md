# Sparse temporal ensemble development status

- Branch: `exp/libero-component-temporal-reuse`
- Starting commit: `7385d64be7f24c3406ab812ffac88a3f20d9dcd5`
- Protocol: frozen before outcomes in `protocol.json`
- CPU semantic tests: passed (7/7 shared tests; both runner smokes passed)
- ACT one-state smoke: passed and validated
- SmolVLA one-state smoke: passed and validated
- SmolVLA real paired-flow check: passed, `(50, 7)` raw chunks exactly equal (`max_abs_error=0.0`)
- ACT full panel: complete and validated (160 episodes)
- SmolVLA full panel: complete and validated (160 episodes); GPU 0 executed the final Smol shard after ACT
- Analysis and report: complete
- Active jobs: none
- Decision: `SPARSE_TE_HARMFUL`

## Result snapshot

| policy | hard h8 | h8+TE | hard h16 | h16+TE |
|---|---:|---:|---:|---:|
| ACT | 34/40 | 23/40 | 33/40 | 20/40 |
| SmolVLA | 30/40 | 28/40 | 30/40 | 29/40 |

ACT paired net wins were -11 at h8 (`p=0.007385`) and -13 at h16 (`p=0.002350`). SmolVLA paired net wins were -2 at h8 (`p=0.687500`) and -1 at h16 (`p=1.000000`). See `report.md` for per-task outcomes and query-budget metrics.

Pre-existing untracked experiment directories under `experiments/component_temporal_reuse/` are preserved and are not part of this experiment.

The stopped provisional fixed-GPU Smol launch produced only partial logs/progress under `smolvla/invalidated/provisional_fixed_gpu_launch/`; it produced no result files or completion markers and is excluded from analysis.

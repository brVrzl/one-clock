# Thor → RTX 5080 scientific handoff

This bundle freezes Thor science at `3b9f1209df7266160c47453e8ee66a142ea8688c` and transfers only local-only evidence needed to reproduce or audit the final paper story. No rollout was rerun and no new scientific outcome was generated.

## Staging and archive

- Staging root: `/home/thor/projects/one-clock_handoff_20260824/thor_to_5080_20260824/`
- Archive: `/home/thor/projects/thor_to_5080_20260824.tar.zst`
- Archive sidecar: `/home/thor/projects/thor_to_5080_20260824.tar.zst.sha256`
- The binary archive is intentionally outside Git. The committed copy of this manifest is the operational index.

## Included evidence

| Logical asset | Original Thor path | Transferred files / bytes | Destination | Status |
|---|---|---:|---|---|
| Gate-3A1 Thor reference cache | `/home/thor/projects/one-clock/experiments/gate3a1_dense_temporal_cache` | 83 / 32,949,112 | `evidence/gate3a1_thor_reference_cache/` | PASS; 82-NPZ tree `87e97a…31908c`; label `THOR-REFERENCE-CACHE` |
| Gate-3A2 closed-loop temporal aggregation | `/home/thor/projects/one-clock/experiments/gate3a2_temporal_aggregation` | 400 / 10,191,635 | `evidence/gate3a2_rollouts/` | PASS; tree `d61f9850…131a4c` |
| Gate-3B cross-generation composition | `/home/thor/projects/one-clock/experiments/gate3b_cross_generation_composition` | 400 / 20,618,916 | `evidence/gate3b_rollouts/` | PASS; tree `046eedc6…0eac83d` |
| Gate-3C asymmetric temporal reuse | `/home/thor/projects/one-clock/experiments/gate3c_asymmetric_temporal_reuse` | 700 / 60,101,395 | `evidence/gate3c_rollouts/` | PASS; tree `0df106c2…506403` |
| Historical selective-retention raw/provenance | `/home/thor/projects/one-clock/experiments/groupwise_selective_commitment` | 130 / 23,337,424 | `evidence/selective_retention/groupwise_selective_commitment/` | 120 raw files all pass per-file manifest hashes; full-root expected tree mismatch recorded |
| Task-0 static 20-state grid | `/home/thor/projects/one-clock/experiments/runs/libero_static_grid_20` | 140 / 65,729,472 | `evidence/static_horizon/task0_grid_20/` | Transferred; historical ledger tree mismatch recorded |
| Task-0 static 50-state extension | `/home/thor/projects/one-clock/experiments/runs/libero_static_grid_50_extension` | 128 / 96,539,228 | `evidence/static_horizon/task0_grid_50_extension/` | Transferred; historical ledger tree mismatch recorded |
| Cross-task static evidence | `/home/thor/projects/one-clock/experiments/runs/libero_object_cross_task` | 512 / 245,537,731 | `evidence/static_horizon/cross_task/` | Transferred; historical ledger tree mismatch recorded |

The full per-file SHA256 list is `provenance/SHA256SUMS`; scoped and full observed trees, expected ledger values, and validation notes are in `provenance/TREE_HASHES.json`. The JSON manifest contains the complete absolute-path and provenance record.

## Discrepancy handling

Gate-3A2, Gate-3B, and Gate-3C are hard verification passes against their committed episode-tree definitions. Gate-3A1 passes against the committed 82-entry cache manifest. The old selective-retention and static ledgers contain expected tree values that do not reproduce from the present committed directories under the documented generic definition. Their current bytes and observed hashes are preserved; no artifact was regenerated, replaced, or silently marked as matching. Selective-retention raw logs independently pass all 120 per-file hashes in `rollout_log_manifest.json`.

## Deliberate exclusions

The ACT checkpoint/config and LIBERO dataset are not duplicated because RTX 5080 independently verified exact copies. Abandoned RoboTwin, PACE, incomplete SmolVLA, learned-reliability, and non-final Gate-2B bulk are omitted. Selective-retention generated scripts, figures, and `__pycache__` are omitted; the raw logs and provenance required for the historical claim are included.

## Rehydration

Copy the archive contents into a work area, verify the archive sidecar, then verify `provenance/SHA256SUMS` and `provenance/TREE_HASHES.json`. Retrieve all committed protocols, schedules, reports, statistics, and paper sources from Git at the source commit/branch; do not overwrite the independently regenerated RTX Gate-3A1 cache with `THOR-REFERENCE-CACHE`.

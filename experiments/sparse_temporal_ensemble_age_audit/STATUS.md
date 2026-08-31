# Sparse temporal-ensemble age audit status

- Part A read-only audit: complete.
- Preliminary implementation smoke: `libero_object:task3`, states 10--12, seeds 2000--2002, h16 passed exactly; retained under `pairing_audit/`.
- Formal Part B pairing cohort: `libero_10:task3`, states 10, 11, and 16 with seeds 2000, 2001, and 2006, h16 passed exactly after repair.
- Root cause: repeated resets of one task-10 environment randomize static fixture model positions outside the saved MuJoCo dynamic state. The repaired audit constructs a fresh, identically seeded environment per condition/state.
- No new outcome rollout has started.
- No audit or rollout jobs remain active.
- Historical results under `../sparse_temporal_ensemble_dev/` remain unchanged.
- Physical-age operator implementation and all outcome rollouts are paused until the requested h1-equivalence orientation is reconciled with validated LeRobot ACT semantics.

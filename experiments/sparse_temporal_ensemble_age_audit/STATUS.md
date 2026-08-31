# Sparse temporal-ensemble age audit status

- Part A read-only audit: complete.
- Preliminary implementation smoke: `libero_object:task3`, states 10--12, seeds 2000--2002, h16 passed exactly; retained under `pairing_audit/`.
- Formal Part B pairing cohort: `libero_10:task3`, states 10, 11, and 16 with seeds 2000, 2001, and 2006, h16 passed exactly after repair.
- Root cause: repeated resets of one task-10 environment randomize static fixture model positions outside the saved MuJoCo dynamic state. The repaired audit constructs a fresh, identically seeded environment per condition/state.
- Dense-equivalent CPU semantics: 15 tests passed.
- Repaired fresh-environment task-10 trio pairing: states 10, 11, and 16 passed with exact equality.
- One-state complete three-method live smoke: passed validation.
- Full repaired ACT h16 trio: complete, 120/120 episodes; all four task validators passed.
- Results: hard 32/40, candidate-index TE 24/40, dense-equivalent TE 23/40.
- Decision: `DENSE_EQ_TE_HARMFUL`.
- Repaired candidate-index TE remains harmful.
- No jobs remain active.
- Historical results under `../sparse_temporal_ensemble_dev/` remain unchanged.
- Operator resolution: `dense_equivalent_te`, oldest-to-newest weights `exp(-0.01*(q_j-q_0))`; no newest-relative decay.

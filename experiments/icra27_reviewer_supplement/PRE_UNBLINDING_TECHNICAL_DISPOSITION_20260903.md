# Pre-unblinding technical disposition

Timestamp: 2026-09-03T14:56:08+08:00

Status: `FROZEN_BEFORE_REVIEWER_SUPPLEMENT_OUTCOME_INSPECTION`

Governing reviewer-supplement preregistration:
`f44a7605246d4c9ea82f4d19ad61833e8fb13eb8`.

The treatment of R1D and B3 is determined solely by technical and integrity
state, never by the scientific direction of R1A, R1B, or R1C.

- If a phase is technically resumable, it will be resumed regardless of the
  R1A/R1B/R1C scientific direction.
- If a phase is technically irrecoverable, it will be reported as technically
  incomplete regardless of the R1A/R1B/R1C scientific direction.
- A resume may execute only frozen cells lacking a valid completion marker.
- A completed cell must never be rerun, overwritten, or recomputed for
  consistency.
- Original conditions, states, checkpoints, seeds, and manifest identities
  must remain unchanged.
- The scientific retry count for already completed cells must remain zero.

At freeze time, R1D had no result, attempt, failure, or completion artifact for
any of its 100 frozen cells. Its three workers failed during runtime import
before cell execution. B3 had all eight frozen shard completion markers and its
canonical-analysis completion marker. No scientific payload was opened to make
this disposition.

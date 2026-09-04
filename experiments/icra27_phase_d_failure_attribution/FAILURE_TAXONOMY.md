# Frozen Phase-D failure taxonomy

Frozen at `2026-09-04T16:31:26+08:00` before Phase-1 outcome unblinding.

The task map defines an ordered list of stages. At episode end, find the first
stage that was not credited as complete. `focal stage` below means stage 1;
`later stage` means any stage with index greater than 1.

| Top-level category | Deterministic rule |
|---|---|
| `SUCCESS` | The environment reports task success. |
| `PRE_OPPORTUNITY_FAILURE` | The focal stage is incomplete and its opportunity was never true while it was active. |
| `INTERACTION_EXECUTION_FAILURE` | The focal stage reached opportunity but its frozen task-appropriate attempt event never occurred. |
| `ACQUISITION_OR_ENGAGEMENT_FAILURE` | The focal stage had an attempt, but its acquisition/engagement event never occurred, or acquisition/engagement persisted without completing the stage. |
| `POST_ACQUISITION_LOSS` | The focal stage acquired/engaged, later lost that state, and never completed. For acquisition, loss is bilateral grasp becoming false before placement. For a fixture, loss is affordance contact ending after joint progress but before the exact fixture predicate completes. |
| `LATER_STAGE_FAILURE` | The focal stage completed, but a later stage did not. The first incomplete later stage receives a `later_stage_detail` from the four physical rules above. |

For a placement stage, the detail rules are: no target opportunity is
`PRE_OPPORTUNITY_FAILURE`; opportunity but no release attempt is
`INTERACTION_EXECUTION_FAILURE`; release attempt without the exact `In`/`On`
predicate is `ACQUISITION_OR_ENGAGEMENT_FAILURE`; loss of bilateral grasp away
from target opportunity before placement is `POST_ACQUISITION_LOSS`.

`TIMEOUT` is metadata with values `COMMAND_TRACE_EXHAUSTED`,
`EPISODE_CAP_REACHED`, or null. It never replaces the physical attribution.

`BLIND_MANUAL_REVIEW` is a predefined fallback status, not a physical category.
It is used only when a required named body/site/joint/geom is missing or the
automatic stage record is internally inconsistent. The annotator receives the
synchronized video with condition and paired-outcome labels hidden and must
choose among the frozen physical categories above or `UNRESOLVED`. No category
may be added after seeing an episode.


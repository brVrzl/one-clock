# Matched-query group-wise selective commitment

Status at branch creation: predeclared mechanism gate. The canonical parent
was `6ed5d06516aaddb382095e3343430c7e31cd22d7`.

## Scientific question

With the same frozen ACT policy and the same full-policy query times, does
independent group-wise acceptance improve closed-loop LIBERO-Object execution
relative to replacing the entire joint action generation?

This experiment makes no compute-saving claim: both methods run the same full
ACT query at every scheduled cadence.

## Frozen runtime

- checkpoint: `/home/thor/projects/checkpoints/zeromidnight_act_libero_object`
- benchmark: LIBERO-Object, task IDs 0--9
- initial states: IDs 0--19, seed `1000 + init_state_id`
- action: `action[0:6]` arm and `action[6]` gripper
- control mode, preprocessing, postprocessing, and environment semantics are
  inherited from the verified `scripts/run_libero_gate0.py` path
- ACT is not retrained and temporal ensembling remains disabled

## Matched methods

At environment step `t`, a full policy query occurs exactly when
`t % q == 0`, for `q` in `{4, 8, 16}`.

Global Replace queries the full chunk and commits both groups to its new
generation.

Selective Commit queries the same full chunk and independently compares each
group's old current action with the fresh chunk's row-0 action. A missing old
generation or an exhausted old source chunk accepts the fresh group. Otherwise
the group accepts only when `d_g > epsilon_g`; it retains the old generation
when `d_g <= epsilon_g`.

The distance is the already audited action-space normalization:

- arm: `max(translation_normalized_RMS, rotation_normalized_RMS)` using the
  checkpoint action standard deviations
- gripper: normalized absolute difference using `action_std[6]`; a gripper
  sign mismatch is promoted above epsilon, preserving the existing
  normalized-error-plus-sign validity rule
- `epsilon_arm = epsilon_gripper = 1.0`

Distances use only the current old and fresh actions at the current query.
No future observation, `Y_refresh`, phase, progress, or learned estimator is
read online.

## Generation and exhaustion semantics

Each full query receives one fresh source generation ID. Accepted groups reset
their local source position to zero and their local age to zero. Retained
groups keep their generation and continue advancing local age and source
position independently. The emitted action is composed from the two local
group generations.

Because the fixed query cadence can fall between the end of a retained 100-step
source chunk and the next scheduled query, an exhausted group explicitly holds
its final source action until that next scheduled full query; at that query the
fresh generation is force-accepted for that group. This preserves exact query
cadence and makes the edge case visible in logs via
`source_chunk_exhausted`. No extra policy query is inserted.

## Predeclared interpretation

- **GO:** selective commitment has a meaningful matched-query closed-loop
  advantage over Global Replace across a substantial part of the cadence range,
  either by higher success or comparable success with clearly improved
  continuity/fewer unnecessary replacements, and is not driven by one task.
- **PARTIAL:** the effect is task/cadence dependent, small, or mainly an
  execution-level effect rather than a success-level effect.
- **NO-GO:** Global Replace performs as well or better consistently and
  selective commitment gives no meaningful closed-loop advantage.

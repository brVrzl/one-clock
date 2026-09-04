# Phase-D preregistration: failure-stage attribution and command-trace sufficiency

Frozen at `2026-09-04T16:31:26+08:00`, before this session opened or parsed any
Phase-1 success outcome.

## Question and scope

Phase-D asks whether changing only the temporal execution of the gripper command
repairs a failure after the manipulator has reached a task-specific manipulation
opportunity. It is not an executor search. The only suite is LIBERO-10, and all
ten tasks are retained.

The paired unit is one task/initial-state block. The development contrasts are
`H4` versus `ARM4_GRIP32` and `H2` versus `ARM2_GRIP16`. The Phase-1 confirmation
contrast is `H4` versus `ARM4_GRIP32` over every available task/state block with
official state ID 15 through 49. Component-swap replays are deterministic
interventions on recorded commands, not interventions on the online ACT policy.

## Prospective boundary and prior exposure

The historical Track-A outcomes and pairings are already exposed. The supplied
full-cohort discordances are 32 rescue, 11 harm, 104 both fail, and 303 both
succeed for `H4 -> ARM4_GRIP32`, and 28 rescue, 2 harm, 127 both fail, and 293
both succeed for `H2 -> ARM2_GRIP16`. They play no role in the definitions in
this freeze.

No Phase-1 success field, result summary, analysis output, or outcome-bearing
result JSON was opened or parsed before this freeze. Phase-1 source code,
protocol metadata, filenames, and completion-marker filenames were inspected;
these do not encode success outcomes.

Repository inspection found a provenance conflict with the requested cohort
labels. Stored Track-A LIBERO-10 artifacts use 15 task-specific official state
IDs, mostly 20 through 44, selected under the earlier exposure protocol; they
are not state IDs 0 through 14. Phase-1 artifacts cover state IDs 15 through 49.
Consequently, the two artifact sets overlap in official state IDs. Phase-D will
analyze the stored Track-A set as development and the complete Phase-1 execution
round as confirmation, but will not describe the overlap as state-held-out.

## Frozen opportunity construction

The authoritative task map is `TASK_MANIPULATION_OPPORTUNITIES.json`. All
opportunities are position-only and use the MuJoCo state after executing a
recorded command.

For an object or an explicit fixture affordance, let `G` be its frozen set of
collision geoms. For geom `g`, MuJoCo provides world center `geom_xpos[g]` and
compiled bounding radius `geom_rbound[g]`. With EEF grip-site position `p`,

`opportunity = min_g (||p - geom_xpos[g]||_2 - geom_rbound[g]) <= 0.040 m`.

For a box target site, distance is the exact Euclidean point-to-oriented-box
distance computed from `site_xpos`, `site_xmat`, and `site_size`, and the same
0.040 m margin is used. For a non-box target site, its compiled site bounding
radius is used analogously. The 0.040 m margin is the maximum absolute travel of
one Panda finger joint (`finger_joint1` range 0.00--0.04 m and
`finger_joint2` range -0.04--0.00 m). This is a geometry-derived reach margin,
not an outcome-fitted parameter.

Object interaction geoms are the object's runtime `contact_geoms`. The stove
knob, bottom-drawer handle, and microwave handle use the explicit geom lists in
the task map. No demonstration-derived quantity is used.

## Frozen physical events and ordered-stage rules

- `attempted`, acquisition stage: while opportunity is true, command dimension
  6 is positive (the verified robosuite close direction), or either Panda finger
  contacts a manipulated-object contact geom.
- `acquired`: robosuite's bilateral grasp predicate is true, meaning at least
  one left-fingerpad geom and one right-fingerpad geom simultaneously contact a
  manipulated-object contact geom. No duration threshold is added.
- `attempted`, placement stage: after acquisition, opportunity at the target is
  true and command dimension 6 is negative, or the grasp changes from true to
  false at that opportunity.
- `placed`: the exact task predicate listed in the map is true (`In` or `On`),
  using LIBERO's own predicate implementation.
- `attempted`, fixture stage: opportunity is true and a Panda finger geom
  contacts one of the frozen affordance geoms.
- `engaged`, fixture stage: after a fixture attempt, the named fixture joint has
  moved strictly in the required direction from its value when the stage became
  active.
- `completed`, fixture stage: LIBERO's exact `Turnon` or `Close` predicate is
  true.

Stages are evaluated in the frozen task-map order. A completion predicate may
be observed before a stage becomes active, but it is credited only after every
prerequisite stage has completed. For a language conjunction with no explicit
ordering word, the noun order in the official instruction is the frozen order.
This applies equally to successes and failures.

## Frozen failure attribution

The deterministic top-level rules are in `FAILURE_TAXONOMY.md`. `TIMEOUT` is
termination metadata. The fallback `BLIND_MANUAL_REVIEW` is permitted only for
a missing required simulator variable or an internally inconsistent automatic
record; it is not a new physical category and cannot change the frozen category
set.

## Original-trace replay and canary gate

Replay restores the exact saved `initial_sim_state` after a normal reset with
the source task, state, environment seed, control frequency, cameras, relative
control mode, and episode cap. Each saved 7-D `executed_actions[t]` is cast back
to float32 and passed directly to the same LIBERO environment. ACT is never
loaded or queried.

The predefined canary for each task is the lowest official state ID present in
its stored Track-A LIBERO-10 cohort. A canary passes only if:

1. the restored flattened simulator state is elementwise identical to the saved
   state;
2. every replayed float32 command is elementwise identical to the saved command
   after the same float32 cast;
3. terminal success equals the source result; and
4. replay episode length equals the source result.

All ten canaries must pass before scientific replay. A material failure stops
Phase-D for diagnosis.

## Frozen replay logging schema

One gzip-compressed JSONL file is written per replay. Its metadata row records
source path and immutable task/state/seed/runtime fields. Each step row contains:

- zero-based step index and exact 7-D replay command;
- EEF position and quaternion;
- physical gripper qpos and qvel;
- pose of every map-listed manipulated object and target body/site;
- named fixture joint positions;
- relevant contact geom-name pairs only;
- every required stage predicate and the active stage label;
- reward, success, terminated, truncated, and termination reason;
- task-specific manipulation-opportunity state.

Arrays unrelated to those fields are not logged. For discordant source episodes
and command swaps, the agent and wrist RGB streams are stored as synchronized
10-fps MP4 files; the JSONL step index is the frame index.

## Frozen replay selection

For each development contrast, replay every LIBERO-10 baseline failure, both
members of every rescue block, and both members of every harm block. Equivalent
episode requests are deduplicated by source result path. Goal and Spatial are
excluded.

After unblinding, apply the same rule to all Phase-1 LIBERO-10 blocks for
`H4 -> ARM4_GRIP32`. The preregistered overall paired Phase-1 contrast is reported
before stage attribution.

## Frozen component-swap protocol

For every rescue and harm block in the specified cohort, execute both hybrids:

- baseline dimensions 0:6 plus treatment dimension 6;
- treatment dimensions 0:6 plus baseline dimension 6.

The restored initial state is the common paired initial state. Let `L` be the
shorter donor trace length. Exactly commands `0, ..., L-1` are supported. If
success occurs after any of these commands, including command `L-1`, the result
is `SUCCESS`. Otherwise the result is `CENSORED` at common-support exhaustion.
There is no hybrid `FAILURE` label. Commands are never held, repeated, inferred,
or extended.

The primary swap quantity is the number of development rescue blocks in which
baseline arm plus treatment gripper is `SUCCESS`, with the full rescue count as
denominator. The reverse hybrid and both hybrids for every harm block are
reported symmetrically. The same frozen analysis is applied to Phase-1 rescue
and harm blocks after unblinding.

## Frozen reporting

For each contrast report the complete paired table, total blocks, all baseline
failures, opportunity-reaching and non-reaching baseline failures, rescue and
still-fail counts among opportunity-reaching failures, full-cohort rescue and
harm, net rescue minus harm, and stage distributions for both rescues and harms.
No selected-subset success rate is reported.

The final label is selected mechanically:

- `A` only if a strict majority of rescues are post-opportunity, baseline-arm +
  treatment-gripper succeeds in a strict majority of rescue swaps, and rescue
  count exceeds harm count.
- `D` if A would otherwise hold but harms are at least half as numerous as
  rescues or a strict majority of harms share the modal rescue stage.
- `B` if a strict majority of rescues are post-opportunity but the rescue-swap
  majority condition for A is not met.
- `C` otherwise.

`CENSORED` swaps count as not establishing sufficiency; they are never called
failures. The report will not make an online-policy component-causality claim.


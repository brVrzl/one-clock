# RoboTwin ACT cross-benchmark exploratory pilot preregistration

Status: **frozen before the first scientific rollout**

This is an exploratory, outcome-sealed test of component-wise temporal source
selection in RoboTwin. No RoboTwin temporal-method task-success outcome was
inspected when the task set, methods, seeds, or analysis were chosen. This pilot
does not modify the manuscript and does not constitute confirmatory evidence.

## Scientific question and experimental unit

The question is whether heterogeneous action components prefer different source
observations for the same current evaluator decision target in bimanual 14-D ACT.
The paired experimental block is one frozen task and one expert-eligible RoboTwin
seed. Each of the six methods is applied once to every block. There are 5 tasks ×
20 blocks/task × 6 methods = 600 preregistered rollout cells.

Source observation age and chunk alignment are distinct:

- Physical source age is `T_t - T_q`, measured in simulator seconds.
- Current-decision-target alignment is `k = t - q`, measured in evaluator
  decisions. The old candidate is `chunk_from_query_q[k]`.

RoboTwin alignment is described only in current-decision-target terms because TOPP
makes evaluator decisions variable in physical duration.

## Frozen tasks and checkpoints

The task set was selected before temporal-method outcomes:

| Task | External ACT Easy reference | Checkpoint SHA256 | Config SHA256 |
|---|---:|---|---|
| `beat_block_hammer` | 56% | `7f3a058419b82464aeeb48d414a8b948eba55220ff5b4b82f16385a0383862fd` | `9d38e4f1696926fc87facdb3d42bd1ac5e97b8b9339a23446b9ff40833668857` |
| `click_alarmclock` | 32% | `73a475b8a2f97d3998ce1e90d26439cd8a701feb9a2ab1659466bad0da869c1c` | `9d38e4f1696926fc87facdb3d42bd1ac5e97b8b9339a23446b9ff40833668857` |
| `dump_bin_bigbin` | 68% | `5d26180719fded89edf5587674281fa4c3d90470ced743ab504bb40074370781` | `9d38e4f1696926fc87facdb3d42bd1ac5e97b8b9339a23446b9ff40833668857` |
| `handover_block` | 42% | `dfb2801ab20b820a844cbdb896989ff443abb5499d3c10f9749bfc84619dc78c` | `9d38e4f1696926fc87facdb3d42bd1ac5e97b8b9339a23446b9ff40833668857` |
| `open_laptop` | 56% | `PENDING_ARTIFACT_COMPLETION` | `9d38e4f1696926fc87facdb3d42bd1ac5e97b8b9339a23446b9ff40833668857` |

External percentages are fidelity context, not local outcomes and not analysis
targets. No task may be added, dropped, or replaced based on pilot results.

The `open_laptop` checkpoint is prospectively and immutably defined as the final
`policy_last.ckpt` produced by the already-running official seed-0, 6000-epoch
ACT training job using 50 official `demo_clean` trajectories and chunk size 50.
Its SHA256 is pending only because that artifact has not yet been written. The
final SHA256 will be appended as provenance; no alternative run, checkpoint, or
epoch may be substituted. Irrecoverable failure of this fixed job is a technical
event, not an opportunity to select a different checkpoint based on outcomes.

## Frozen methods

Exactly six methods are evaluated.

### `NATIVE_ACT`

Unchanged official RoboTwin ACT: temporal aggregation enabled, query cadence one,
official action postprocessing, and official TOPP execution.

### `NEWEST`

All 14 channels execute `chunk_t[0]`. Executed source age is zero. Temporal
aggregation is disabled for this experimental baseline.

### `FULL_OLD_1S`

At query timestamp `T_t`, select

```text
q*(t) = argmin over q<t, 0<t-q<50 |(T_t-T_q)-1.0 s|.
```

Exact ties choose the more recent query. Set `k=t-q*` and execute all channels
from `chunk_q*[k]`. When no past query exists at decision 0, execute NEWEST. No
age-error tolerance, age-error fallback, or success-dependent rule is allowed.

### `FO_1S`

Use the identical q* and k rule. Execute indices 0–5 and 7–12 from `chunk_t[0]`;
execute grippers 6 and 13 from `chunk_q*[k]`. This is not a hold and never uses
`chunk_q[0]` when k is nonzero.

### `GRIPPER_HOLD`

Arms execute fresh `chunk_t[0]`. Grippers execute the previously executed gripper
command. At decision 0, before a previous command exists, both grippers execute
their fresh command. This initialization is frozen.

### `GRIPPER_EMA_1S`

Arms execute fresh `chunk_t[0]`. In postprocessed command space, separately for
grippers 6 and 13:

```text
tau = 1.0 s
dt = T_t - T_(t-1)
alpha_t = exp(-dt/tau)
g_ema[t] = alpha_t*g_ema[t-1] + (1-alpha_t)*g_fresh[t]
g_ema[0] = g_fresh[0]
```

Tau is not tuned.

## Eligibility and paired schedule

For every task, candidate seeds begin at 100000, matching official evaluator seed
argument 0, and increase by one. The unchanged official expert screen is run
without any study policy. The first 20 eligible seeds in ascending order are
frozen independently per task. Rejected seeds and the reason (`UNSTABLE`,
`EXPERT_FAILED`, or expert exception class) are retained in
`research/audit_outputs/robotwin_exploratory_eligible_seeds.json`.

Every method uses the exact same 20 seeds within a task. There is no method-specific
replacement and no method-specific denominator. Method execution order is frozen
within each task/seed block using randomization seed 20270825: a random method
permutation and randomized cyclic rotations balance execution position in complete
six-block groups.

The machine-readable schedule contains 600 unique cell identifiers. Each identifier
contains task, eligible-seed index, actual seed, method, checkpoint identity, and
configuration SHA256. The prospective `open_laptop` cells use
`PENDING_ARTIFACT_COMPLETION` as their frozen checkpoint identity until the final
artifact is available.

Schedule cells SHA256: `467e11065033b12c1cf865ede301ed368e040014cde8cecf23903702d2ae705a`

## Technical failure and retry policy

- A completed policy rollout is retained regardless of policy success or failure.
- A recognized infrastructure failure may rerun only the identical
  task/method/seed/checkpoint/config cell, up to two retries after the initial
  attempt (three attempts total).
- Seeds are never replaced.
- A persistent infrastructure failure after three attempts yields
  `TECHNICAL_INVALIDATION`.
- A source-index, chunk-offset, action-composition, nonfinite-action, or other
  provenance assertion failure halts the pilot immediately and yields
  `TECHNICAL_INVALIDATION`; it is not automatically retried.
- A technical fix that changes scientific semantics invalidates this
  preregistration and requires a new preregistration before further outcomes.

## Outcome sealing

Per-cell success is written automatically to a separate permission-restricted
outcome file addressed by an opaque cell key. Routine technical status contains
cell completion, attempt count, decision count, crashes, and provenance status but
not success. Per-decision provenance is compressed separately. No success file,
success count, task rate, pooled rate, or between-method comparison is inspected
until all 600 cells are technically complete.

Allowed interim inspection is limited to process health, episode completion,
infrastructure exceptions, finite 14-D actions, checkpoint identity, source/query
indices, chunk offsets, simulator timestamps, and provenance assertions.

## Preregistered analysis

The binary unit is the paired task/seed block. The primary contrast is `FO_1S -
NEWEST`. Secondary contrasts are `FO_1S` minus `NATIVE_ACT`, `FULL_OLD_1S`,
`GRIPPER_HOLD`, and `GRIPPER_EMA_1S`.

After a single unsealing event, report:

1. Success count and rate for every method/task and pooled method.
2. Pooled paired success difference for every preregistered contrast across 100
   task/seed blocks.
3. Paired win/loss/tie counts.
4. The five task-specific paired differences.
5. A 95% percentile task-cluster bootstrap interval using 10,000 resamples of the
   five tasks with replacement and analysis seed 20270826; all 20 paired seeds of a
   sampled task remain together.
6. Five leave-one-task-out pooled paired differences.

No confirmatory p-value is claimed.

## Interpretive gate

- `STRONG_SIGNAL`: FO has a positive pooled paired difference over NEWEST, positive
  task-cluster interval lower bound, positive deltas on at least three tasks, point
  estimates above FULL_OLD/HOLD/EMA, and is within 5 percentage points of or better
  than NATIVE_ACT.
- `MECHANISM_SIGNAL_ONLY`: the same positive FO-versus-NEWEST and simple-control
  criteria hold, but FO is more than 5 percentage points below NATIVE_ACT.
- `NO_SIGNAL`: the criteria above are not met or the apparent FO gain is explained
  by FULL_OLD, HOLD, or EMA.
- `TECHNICAL_INVALIDATION`: the frozen failure policy invalidates the pilot.

This pilot only decides whether a later 100-episode/task confirmatory experiment is
scientifically justified. It cannot itself establish confirmation.

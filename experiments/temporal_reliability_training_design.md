# Temporal reliability estimator: preparation design

**Status:** preparation only. No estimator has been trained, no ACT inference
has been run by this package, and no rollout or benchmark result is produced.

This design follows the frozen-policy persistence audits and the method-design
memo. The next experiment estimates whether a group slice of an ACT action
chunk remains valid as its source observation becomes older. The estimator is
an auxiliary model; ACT, the executor, rollout behavior, and the paper remain
unchanged.

## Inputs and scope

The intended external inputs are:

| Input | Path | Use |
|---|---|---|
| Frozen ACT checkpoint | `/home/thor/projects/checkpoints/zeromidnight_act_libero_object` | Produce frozen action chunks and, later, source-time observation embeddings. |
| LeRobot v2.1 LIBERO Object dataset | `/home/thor/datasets/libero_object_25_08_23_lerobotv2.1` | Episode boundaries, demonstrations, observations, and task metadata. |

The preparation package indexes LeRobot metadata with
`build_lerobot_manifest`, but does not load videos, instantiate ACT, or write to
either external path. A future frozen-inference adapter must materialize the
following `FrozenTrajectory` contract before example construction:

- `episode_id` and optional `task_id`;
- demonstrated actions `[episode_step, action_dim]`;
- source-time observation embeddings, or no embedding while the configured
  zero-filled placeholder is used;
- `policy_chunks[step] = [chunk_step, action_dim]` for each ACT query position.

The action partition for this checkpoint is predeclared as `arm = 0:6` and
`gripper = 6`. The `GroupSpec` interface also supports another disjoint
partition without changing the dataset or target code.

## Data flow

```text
LeRobot metadata + frozen ACT prediction artifact
                    |
                    v
          episode-level split (once)
                    |
                    v
   FrozenTrajectory records, one record per episode
                    |
                    v
 source step s x offset k x action group g examples
                    |
          +---------+----------+
          |                    |
          v                    v
 source-only features       target loss / Y_g(k)
          |                    |
          +---------+----------+
                    v
       untrained estimator interface -> held-out metrics
```

The split happens before windows/examples are expanded. Thus overlapping
source windows from one episode cannot land in different splits. The target
may read a future demonstrated action or a future frozen-policy prediction,
but the feature encoder reads only the source observation embedding and source
action chunk.

## Split strategy

The default is a deterministic episode-level split:

- train/validation/test fractions: `70% / 15% / 15%`;
- seed: `20260820`;
- task-stratified allocation when task IDs are available, so every split can
  represent the ten LIBERO Object tasks when episode counts permit;
- no episode appears in more than one split.

Thresholds, feature normalization statistics, and any model-selection choice
must be fitted on train and validation data only. The test split is touched
once for final offline reporting. The `stratify_by_task=False` option is
reserved for a separate task-held-out generalization study; it must not be
silently mixed with the default within-task estimate.

The split utility returns episode IDs, while the dataset builder returns
examples. It does not create horizon labels and it does not use rollout success
or any benchmark return.

## Example and target definition

For a frozen chunk queried at source time `s`, let `k` be an offset within the
predicted chunk. For group `g`, the source action is

```text
a_old_g(s, k) = policy_chunks[s][k, group_indices[g]]
```

The primary target interface is the fresh-policy consistency target. When a
fresh frozen-policy chunk exists at the future demonstration observation
`s + k`, its first group action is the reference:

```text
loss_fresh_g(s, k) = d_g(
    a_old_g(s, k),
    policy_chunks[s + k][0, group_indices[g]])
```

This measures persistence/staleness of the old policy prediction under a new
observation. It does not claim to be physical task success. The package also
supports the separate demonstration-deviation diagnostic:

```text
loss_demo_g(s, k) = d_g(
    a_old_g(s, k),
    demonstrated_actions[s + k, group_indices[g]])
```

These targets are not silently combined. `TemporalValidityTarget` retains the
continuous loss and can generate

```text
Y_g(k) = 1[loss_g(s, k) <= epsilon_g]
```

only when a caller supplies `epsilon_g` per group or an explicit validity
function. There is no hard-coded arm threshold, gripper threshold, metric
weight, or final horizon. Inclusive versus strict comparison is configurable.
Thresholds are a train/validation calibration choice and must not be selected
from test rollout success.

The default example metric is a raw groupwise RMS only as a transparent
placeholder. A real study should predeclare group-specific normalization. In
particular, arm translation and rotation should not be combined with an
arbitrary physical weighting; the metric can be replaced by a callable that
keeps those components separate or uses checkpoint statistics. Gripper
validity must likewise be calibrated independently.

If `policy_chunks[s + k]` is missing for the fresh-policy target, that example
is skipped and counted. It is never filled from a neighboring future query.
This makes the sampling/cost tradeoff visible and avoids target leakage through
implicit interpolation.

## Feature definition

`FeatureEncoder` emits one row per `(episode, source step, group, offset)`.
The initial schema contains:

1. a source observation embedding slot; the default 16-dimensional slot is
   zero-filled until a frozen ACT-compatible embedding extractor is supplied;
2. channelwise statistics of the **source** group slice over the full action
   chunk: mean, standard deviation, minimum, maximum, first action, last
   action, and last-minus-first;
3. a one-hot group ID for `arm` or `gripper`;
4. `k` and `k / max_offset`.

Statistics are padded to the largest configured group width so the feature
dimension is fixed. They are descriptive features, not a validity metric.
No future observation, future embedding, future action, future policy chunk,
episode outcome, rollout success, or target loss is read by the feature
encoder. The source chunk's own row `k` is allowed because it was available at
source time when ACT produced the chunk.

The estimator input dimension is available as `FeatureEncoder.input_dim`, so
the placeholder embedding can be replaced without hand-editing an MLP.

## Initial model interface

`MLPBaseline(input_dim, hidden_dims=(128, 64))` is a lightweight optional-Torch
module. It accepts a `[batch, feature]` matrix and returns one sigmoid reliability
score per row. Its weights are only initialized; this package provides no
optimizer, loss loop, checkpoint writer, or training command. The class is
intentionally separate from the frozen ACT model.

The first training plan, when authorized, is:

1. materialize frozen ACT chunks and source embeddings once, recording dataset
   and checkpoint provenance;
2. build train/validation/test examples using the predeclared split;
3. choose the error metric and `epsilon_g` on train/validation only;
4. fit the MLP to `Y_g(k)` with binary cross-entropy, with class weighting or
   sampling specified from train counts only;
5. calibrate scores on validation data (for example, temperature scaling or a
   monotonic calibration fit) without changing ACT;
6. freeze the estimator and report test discrimination/calibration by group and
   offset.

This is a future plan, not an action performed by the current change.

## Evaluation utilities

The package provides NumPy-only utilities for:

- AUROC, returning `NaN` when a slice has only one class;
- Brier score for probabilistic validity scores;
- equal-width reliability curves with bin counts, mean score, and observed
  validity;
- expected calibration error (ECE), count-weighted over occupied bins;
- aggregate, per-group, and per-offset evaluation slices.

Scores passed to these utilities must be probabilities in `[0, 1]`, and labels
must be binary. Reporting should include sample counts and the fraction of
undefined AUROC slices rather than hiding sparse offsets.

## Scientific guardrails and future ablations

The preparation pipeline does not implement a scheduler. In a later execution
study, a predicted score must not be presented as a horizon until calibration,
within-episode variation, and query-budget accounting are established. At
minimum, compare against global fixed, static group-wise, a scalar estimator
using the same signal, and a time/decision-shuffled group-wise control with the
same horizon distribution. Track group source ages, source chunk generations,
and policy-query counts. Mixed-generation cross-group incompatibility remains
an explicit failure mode.

The primary self-supervised target is policy persistence, not rollout success.
Demonstration deviation is a separate diagnostic. No future observation is
allowed into features, and no benchmark success label is used to construct
`Y_g(k)`.

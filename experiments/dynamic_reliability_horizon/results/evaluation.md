# Dynamic reliability horizon: first scientific evaluation

## Status: blocked before training

This evaluation was not run because the required external inputs are absent
from the current environment:

- dataset: `/home/thor/datasets/libero_object_25_08_23_lerobotv2.1`
- frozen ACT checkpoint: `/home/thor/projects/checkpoints/zeromidnight_act_libero_object`

There is also no prepared target artifact in the repository or workspace. The
training CLI consumes precomputed `Y_g(k)` examples and intentionally does not
regenerate targets. Therefore no train/validation/test artifact could be
loaded, and no estimator was trained.

## Outputs

`metrics.json` records `status: not_run`. No checkpoint, metric value, or plot
has been fabricated. In particular, AUROC, Brier score, ECE, reliability
diagrams, task-wise results, offset-wise results, and group calibration remain
unavailable.

No rollout or benchmark execution was attempted.

## Reproduction when inputs are mounted

First materialize or provide the existing episode-split prepared target
artifact, then run from the repository root:

```sh
PYTHONPATH=src:. python -m experiments.dynamic_reliability_horizon.train \
  --dataset /path/to/prepared_reliability.npz \
  --output-dir experiments/dynamic_reliability_horizon/results/checkpoints
```

Run the combined, arm-only, and gripper-only checkpoints through the evaluation
CLI. The evaluation will report the learned MLP, constant prior, and empirical
`P(Y=1 | group, offset)` baseline on held-out test episodes and write the
reliability/calibration plots.

This stage can only validate reliability prediction. It cannot establish
dynamic-horizon or robot-success improvement.

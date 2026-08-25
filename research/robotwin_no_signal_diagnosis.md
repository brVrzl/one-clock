# RoboTwin exploratory NO_SIGNAL diagnosis

The preregistered classification is **NO_SIGNAL**. FO minus NEWEST was +0.000, with task-cluster 95% interval [-0.030, +0.030].

## Control diagnosis

- FO minus FULL_OLD_1S: +0.020.
- FO minus GRIPPER_HOLD: +0.060.
- FO minus GRIPPER_EMA_1S: +0.010.
- FO minus NATIVE_ACT: -0.080.

The control closest to FO indicates whether globally old predictions, command retention, simple smoothing, or official shared-clock temporal aggregation accounts for the observed pattern. No source age, task, seed, or smoothing parameter was retuned.

## LIBERO versus RoboTwin

LIBERO Object Gate-3C remains frozen: FO20 63.5%, NEWEST 42.1%, FULL_OLD20 43.7%, Age-exp 49.2%, and CogACT 46.8%. The RoboTwin table and heatmap report the independently preregistered bimanual result without altering those figures.

## Closest-method interpretation

- NATIVE_ACT is the official global temporal-aggregation comparator.
- FULL_OLD_1S tests a shared old temporal source.
- GRIPPER_HOLD and GRIPPER_EMA_1S test retention and smoothing explanations.
- FO_1S is supported only if its paired advantage survives those controls; the frozen gate did not.

## Reviewer risk

The cross-benchmark intervention preserves physical source age but aligns chunks by decision target under variable-duration TOPP. A null or control-explained result therefore bounds generality; it must not be reframed through post-hoc age tuning or task selection.

# One Clock Does Not Fit All

This repository contains the first execution-only Gate-0 experiment for
group-wise fixed commitment horizons on RoboTwin 2.0. The base policy is the
official XPolicyLab ACT implementation; this repository does not train or
modify ACT.

The verified initial embodiment is RoboTwin's clean `aloha-agilex` bimanual
joint-action setting. Its 14-D action is packed as six left-arm joint targets,
one left gripper target, six right-arm joint targets, and one right gripper
target. The experiment groups those verified dimensions as
`left_arm`, `left_gripper`, `right_arm`, and `right_gripper`.

The executor has only two strategies:

- `global_fixed`: query a full chunk and execute its first `h` rows;
- `groupwise_fixed`: query a full chunk when one or more groups expire, keep
  unexpired groups on their current chunk, and compose the environment action.

Run the deterministic executor tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The real runner reuses RoboTwin task setup, observation conversion, and the
XPolicyLab ACT model. It disables the upstream ACT temporal aggregation so the
comparison is fixed execution only. Supply a real ACT checkpoint directory:

```bash
PYTHONPATH=src python scripts/run_gate0.py \
  --robotwin-root /home/thor/projects/upstreams/RoboTwin \
  --checkpoint /path/to/ACT/checkpoints/<run> \
  --strategy global_fixed --horizon 8 \
  --output-dir experiments/runs/global_h8
```

The runner writes `metadata.json`, per-control-step `steps.jsonl`, and
`summary.json`. The initial diagnostic group-wise setting is in
`configs/gate0_place_can_basket.yaml` and can be overridden with, for example,
`--group-horizons left_arm=8,left_gripper=2,right_arm=8,right_gripper=2`.

The upstream RoboTwin and XPolicyLab checkouts are external dependencies. Their
commits used for the audit are recorded in
`experiments/roboTwin_act_execution_audit.md` and should be passed through the
runner metadata for each rollout.

## LIBERO Gate-0 on Jetson Thor

The second Gate-0 path uses the current LeRobot LIBERO integration and the
verified 7-D LIBERO action (`arm` indices 0--5, `gripper` index 6). Use the
native environment with `MUJOCO_GL=egl` and supply a legitimate LeRobot ACT
checkpoint:

```bash
MUJOCO_GL=egl PYTHONPATH=src \
  /home/thor/projects/upstreams/lerobot-env/bin/python scripts/run_libero_gate0.py \
  --checkpoint /absolute/path/to/act_libero_object \
  --strategy global_fixed --horizon 8 \
  --output-dir experiments/runs/libero_object_global_h8
```

The LIBERO source/runtime audit is in
`experiments/libero_execution_audit.md`. The runner calls ACT's full-chunk
prediction method directly; LeRobot's queued action selection and temporal
aggregation remain disabled.

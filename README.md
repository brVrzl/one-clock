# ICRA27 baseline foundation

This branch is the hypothesis-neutral infrastructure for standard LIBERO
policy evaluation. It does not contain research results, temporal
interventions, RoboTwin integrations, or overnight orchestration.

Run the lightweight executor tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The LIBERO runner uses the 7-D action contract (`arm` indices 0--5 and
`gripper` index 6), queries complete ACT chunks, and executes the selected
fixed commitment through the generic executor. The standard foundation path is
`global_fixed`; no temporal intervention or learned horizon selection is
enabled.

Supply a legitimate LeRobot ACT checkpoint in the known LIBERO environment:

```bash
MUJOCO_GL=egl PYTHONPATH=src \
  /home/thor/projects/upstreams/lerobot-env/bin/python scripts/run_libero_gate0.py \
  --checkpoint /absolute/path/to/act_libero_object \
  --strategy global_fixed --horizon 8 \
  --output-dir experiments/runs/libero_object_global_h8
```

The runner records metadata, per-step actions, episode summaries, and query
budgets. Rollouts are intentionally outside this consolidation task.

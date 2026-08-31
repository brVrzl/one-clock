# Runbook

Run from the repository root with the pinned environment:

```bash
/home/wjq/workspace/venvs/libero_act/bin/python -m pytest experiments/group_delay_factorial_act20/tests
/home/wjq/workspace/venvs/libero_act/bin/python experiments/group_delay_factorial_act20/run_pairing_smoke.py --gpu 0
```

The smoke must pass before any outcome rollout. It uses task 1, states 20, 21,
and 22 and creates a fresh environment for every condition/state.

Launch the primary task shards only after the smoke passes:

```bash
experiments/group_delay_factorial_act20/resume.sh
```

The simple default allocation is tasks 1–3 on GPU 0, tasks 4–6 on GPU 1, and
tasks 7–9 on GPU 2. Each task shard evaluates all five methods for every
state, writes one resumable task result, and validates the completed task
before its marker is written. The output is under `results/`, `progress/`, and
`markers/`.

After all 630 episodes have validated:

```bash
/home/wjq/workspace/venvs/libero_act/bin/python experiments/group_delay_factorial_act20/analyze.py
```

`analysis.json`, `per_task.csv`, and `report.md` are generated from the new
outcomes only. Historical Gate-3C results are never merged into the new table.

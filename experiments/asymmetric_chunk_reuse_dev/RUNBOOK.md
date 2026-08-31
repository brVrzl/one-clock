# Asymmetric Temporal Reuse development gate

This directory contains the final executor-development experiment. The only
new conditions are C1 PreviousChunkGrip and C2 H16Arm+FreshGrip. The hard-h16
outcomes are reused from the exact compatible repaired factorial commit.

Before rollout:

```bash
/home/wjq/workspace/venvs/libero_act/bin/python -m pytest experiments/asymmetric_chunk_reuse_dev/tests
/home/wjq/workspace/venvs/libero_act/bin/python experiments/asymmetric_chunk_reuse_dev/run_pairing_smoke.py --gpu 0
```

The smoke must pass before outcome rollout. It uses task 1, states 20, 21, and
22, fresh environments per method/state, exact C1/hard equality through t=15,
state equality through t=16, and C2's t=1 causal boundary.

The 252 new episodes are run as three simple shards:

```bash
experiments/asymmetric_chunk_reuse_dev/resume.sh
```

GPU 0 runs tasks 1–3, GPU 1 runs tasks 4–6, and GPU 2 runs tasks 7–9. Each
task writes a resumable result and validates it before its completion marker.

After all 252 new episodes validate:

```bash
/home/wjq/workspace/venvs/libero_act/bin/python experiments/asymmetric_chunk_reuse_dev/analyze.py
```

Analysis uses new C1/C2 step logs plus the exact reused hard-h16 primary
outcomes. It reports paired contrasts, outcome-stratified descriptive logs,
and the frozen development decision. No additional executor condition is
permitted after this gate.

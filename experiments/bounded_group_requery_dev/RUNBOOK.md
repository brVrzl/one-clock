# Bounded group-triggered joint re-query runbook

The protocol is frozen in `protocol.json`. The executor makes one new ACT
query at each dynamic boundary and executes only that query's chunk. It does
not retain historical chunks for action fusion.

## CPU checks

```bash
/home/wjq/workspace/venvs/libero_act/bin/python -m pytest -q test_bounded_group_requery.py
/home/wjq/workspace/venvs/libero_act/bin/python run_bounded_group_requery.py --semantic-smoke
```

## Pairing gate

```bash
/home/wjq/workspace/venvs/libero_act/bin/python run_bounded_group_requery.py \
  --task libero_object:task3 \
  --pairing-smoke --gpu 2 \
  --output act/pairing_live_smoke_object3.json
```

The smoke compares initial raw and processed observations, initial raw A0,
and common-prefix actions, simulator states, and post-action observations only
until the earliest method-specific re-query.

## ACT shards

The completed panel used one method supervisor per GPU:

```bash
./launch_act_method.sh M1_arm_phase 0
./launch_act_method.sh M2_gripper_event 1
./launch_act_method.sh M3_group_event_joint 2
```

Each supervisor runs the four frozen task shards serially. Results are isolated
under `act/results/<method>/`; progress and logs are method-specific. M0 is
reused from Sol's repaired baseline and is not part of the new workload.

## Analysis

```bash
/home/wjq/workspace/venvs/libero_act/bin/python analyze_bounded_group_requery.py \
  --include-h-temp \
  --decision SINGLE_TRIGGER_BETTER \
  --interpretation 'Frozen development decision recorded in report.md.'
```

H_temp is loaded only after outcomes are frozen and is descriptive. No SmolVLA
or blind evaluation is part of this experiment.

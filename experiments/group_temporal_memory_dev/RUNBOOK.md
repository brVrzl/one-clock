# Group-conditioned temporal memory development runbook

This directory is disjoint from Sol’s audit directory. The commands below are intentionally not executed while Sol’s repaired h16 trio commit is absent.

## Gate and protocol update

First inspect running jobs without stopping them:

```bash
git status --short --branch
git pull --ff-only
pgrep -af 'run_act_group_memory|run_smolvla_group_memory' || true
```

After Sol’s trio commit is visible, update only this directory’s `protocol.json` with the exact SHA:

```json
"coordination": {
  "sol_repaired_rollout_commit": "<exact Sol trio commit SHA>"
}
```

The shared kernel must remain:

```json
"shared_kernel": {
  "status": "selected_by_sol",
  "selected_name": "dense_equivalent_te",
  "coefficient": 0.01
}
```

The dense-equivalent prior is applied to candidates ordered oldest to newest:
`b_q ∝ exp(-0.01 * (q - q_oldest))`. At h16 this is `[1, exp(-0.16), exp(-0.32), ...]`. Do not substitute physical-age or candidate-index weighting.

## CPU semantic tests

Run these before any environment command:

```bash
python3 -m py_compile group_memory_operators.py group_memory_common.py \
  run_act_group_memory.py run_smolvla_group_memory.py \
  analyze_group_memory.py freeze_h_temp.py test_group_memory_semantics.py
python3 -m pytest -q test_group_memory_semantics.py
python3 freeze_h_temp.py
```

The semantic smoke commands in the policy runners are also CPU-only, but they do not clear the Sol rollout gate:

```bash
python3 run_act_group_memory.py --semantic-smoke
python3 run_smolvla_group_memory.py --semantic-smoke
```

## Strict pairing smoke

Use the corrected fresh-environment construction in the runner. At least one state/seed pair must pass for every policy before the panel is launched:

```bash
ACT_PY=/home/wjq/workspace/venvs/libero_act/bin/python
$ACT_PY run_act_group_memory.py \
  --task libero_object:task3 --methods M0_h16,M1_shared_te_h16,M2_shared_cogact_h16,M3_group_cogact_h16 \
  --pairing-smoke --gpu 0 --output act/pairing_smoke_object3.json

# Only after ACT selects/retains a meaningful group mechanism:
python3 run_smolvla_group_memory.py \
  --task libero_object:task3 --methods M2_shared_cogact_h16,M3_group_cogact_h16 \
  --pairing-smoke --gpu 1 --output smolvla/pairing_smoke_object3.json
```

The smoke compares initial observations, initial chunks, actions through t=15, simulator states, and post-action observations. Any mismatch stops the panel.

## Resumable task shards

The launch wrappers accept one frozen task and one GPU, write a per-task result, and skip no episode internally. Run them in separate detached supervisors only after the gate and pairing smoke pass:

```bash
./launch_act_shard.sh libero_object:task3 0
./launch_act_shard.sh libero_spatial:task0 0
./launch_act_shard.sh libero_goal:task2 0
./launch_act_shard.sh libero_10:task3 0
```

The ACT stage is the gate for the conditional SmolVLA stage. M4 is not launchable under this protocol. No final blind task is in the task map.

## Analysis

After all selected method results are complete, run paired analysis. Only then, and only for descriptive association, load the already frozen H_temp artifact:

```bash
python3 analyze_group_memory.py --policy ACT \
  --include-h-temp --decision GROUP_MEMORY_POLICY_DEPENDENT \
  --interpretation 'Fill from the frozen ACT development gate.'
```

Use separate output paths for a later SmolVLA analysis. Do not use H_temp to choose a method, weight, alpha, horizon, or reliability value.


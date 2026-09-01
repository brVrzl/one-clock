# ICRA 2027 two-clock discriminator handoff

## Repository state

- Branch: `exp/icra27-two-clock-discriminator`
- Frozen code/protocol commit: `5cb89a6a2c984b0e34ff05da15c0c22a7172f486`
- Worktree: `/home/wjq/workspace/one-clock-icra27-two-clock`
- Protocol: `experiments/icra27_two_clock_discriminator_dev/protocol.json`
- Exact cohort: LIBERO Object tasks 1--9, state IDs
  `[20,21,22,23,27,31,34,35,38,39,44,45,47,48]`, with environment
  seed `330000 + 100 * task_id + state_id`.
- New conditions only: `H32_COHERENT` and
  `TWO_CLOCK_ARM16_GRIP32`; 126 episodes each.

Run `git rev-parse HEAD` for the operational handoff commit containing this
file and the completed raw rollout snapshot.

## Completion state

- Completed: 252 episodes (126 H32 coherent + 126 two-clock).
- Currently running: 0 episodes and 0 rollout workers.
- Pending: 0 episodes.
- All nine task result files have `finished: true`.
- All nine task validators completed and wrote valid completion markers.
- No validator analysis or outcome interpretation has been run after rollout;
  that work is deliberately left to the next Codex session.

The workers were launched by `resume.sh`, which uses `setsid nohup` for three
independent shards. They therefore would have survived a Codex disconnect.
They finished before this handoff. There are no live worker PIDs or session IDs
to preserve. The obsolete interactive monitor PID/session `2536846` was stopped
after completion; it was not a rollout worker.

## Durable artifacts

- Logs: `experiments/icra27_two_clock_discriminator_dev/logs/gpu{0,1,2}_tasks{123,456,789}.log`
- Results: `experiments/icra27_two_clock_discriminator_dev/results/task_01.json`
  through `task_09.json`
- Progress: `experiments/icra27_two_clock_discriminator_dev/progress/gpu{0,1,2}_tasks{123,456,789}.json`
- Markers: `experiments/icra27_two_clock_discriminator_dev/markers/*.complete`
- Semantic smoke: `experiments/icra27_two_clock_discriminator_dev/semantic_smoke.json`

## Inspection and safe resume

Inspect completion without starting anything:

```bash
cd /home/wjq/workspace/one-clock-icra27-two-clock
for f in experiments/icra27_two_clock_discriminator_dev/progress/*.json; do
  jq -c '{task_id,gpu,completed_episodes,finished}' "$f"
done
find experiments/icra27_two_clock_discriminator_dev/markers -name '*.complete' -type f | sort
pgrep -af 'icra27_two_clock_discriminator_dev/(run_shard.sh|run_fixed_clocks.py)' || true
```

If a later filesystem interruption ever makes resumption necessary, use:

```bash
cd /home/wjq/workspace/one-clock-icra27-two-clock
bash experiments/icra27_two_clock_discriminator_dev/resume.sh
```

`resume.sh` skips a fully marked shard and refuses to duplicate a matching live
worker. Within an unmarked shard, `run_fixed_clocks.py` reads the durable result
file and skips completed method/task/state cells, retaining the same protocol,
method, task, state, seed, checkpoint, and runtime configuration. With the
current nine markers, the command is a no-op.

## Next-session validation and analysis

Run the existing validators, then the frozen analysis:

```bash
cd /home/wjq/workspace/one-clock-icra27-two-clock
for task_id in $(seq 1 9); do
  case "$task_id" in
    1|2|3) slug=gpu0_tasks123 ;;
    4|5|6) slug=gpu1_tasks456 ;;
    7|8|9) slug=gpu2_tasks789 ;;
  esac
  /home/wjq/workspace/venvs/libero_act/bin/python \
    experiments/icra27_two_clock_discriminator_dev/validate_shard.py \
    --result "experiments/icra27_two_clock_discriminator_dev/results/task_$(printf '%02d' "$task_id").json" \
    --protocol experiments/icra27_two_clock_discriminator_dev/protocol.json \
    --marker "experiments/icra27_two_clock_discriminator_dev/markers/${slug}_task${task_id}.complete"
done
PYTHONPATH=src /home/wjq/workspace/venvs/libero_act/bin/python \
  -m pytest experiments/icra27_two_clock_discriminator_dev/tests -q
/home/wjq/workspace/venvs/libero_act/bin/python \
  experiments/icra27_two_clock_discriminator_dev/analyze.py
```

After analysis, update the current experiment row in `exposure_inventory.md`
from `PROTOCOL_ONLY` to `OUTCOME_EXPOSED`, inspect the generated report, commit,
and push. Do not change the frozen manuscript.

No additional scientific experiment was launched: no confirmation, adaptive,
RoboTwin, pi0/pi0.5, SmolVLA, or real-robot run was started.

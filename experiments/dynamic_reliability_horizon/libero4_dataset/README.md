# LIBERO-4 reliability-data foundation

This directory contains the reproducible code, manifests, reports, and tests
for the policy-independent LIBERO-4 corpus. The pinned source is
`lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`; downloaded source
files and generated policy caches live outside Git.

## Detached overnight worker

- tmux session: `oneclock_libero4`
- launch command:

  ```bash
  tmux new-session -d -s oneclock_libero4 'cd /home/thor/projects/one-clock && exec experiments/dynamic_reliability_horizon/libero4_dataset/run_overnight.sh'
  ```

- safe resume command (starts only when the session is absent):

  ```bash
  if ! tmux has-session -t oneclock_libero4 2>/dev/null; then tmux new-session -d -s oneclock_libero4 'cd /home/thor/projects/one-clock && exec experiments/dynamic_reliability_horizon/libero4_dataset/run_overnight.sh'; fi
  ```

  The launcher uses `overnight.lock`, skips verified completed shards, and
  resumes partial work;
- durable stdout/stderr log:
  `/home/thor/projects/one-clock/experiments/dynamic_reliability_horizon/libero4_dataset/overnight.log`;
- durable progress:
  `/home/thor/projects/one-clock/experiments/dynamic_reliability_horizon/libero4_dataset/progress.json`.

Inspect a running or completed job with:

```bash
tmux has-session -t oneclock_libero4
tmux attach -t oneclock_libero4
tail -f /home/thor/projects/one-clock/experiments/dynamic_reliability_horizon/libero4_dataset/overnight.log
cat /home/thor/projects/one-clock/experiments/dynamic_reliability_horizon/libero4_dataset/progress.json
```

The current bounded worker completed the corpus and handoff. SmolVLA cache
generation was not started because the Thor compatibility report records the
missing `num2words` SmolVLM dependency.

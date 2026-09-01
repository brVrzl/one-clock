# Two-clock discriminator runbook

The protocol contains exactly two new conditions: coherent H32 and true independent arm16/grip32. H16 and C1 are historical reuses.

Before outcome rollout:

```bash
PYTHONPATH=src /home/wjq/workspace/venvs/libero_act/bin/python -m pytest experiments/icra27_two_clock_discriminator_dev/tests -q
/home/wjq/workspace/venvs/libero_act/bin/python experiments/icra27_two_clock_discriminator_dev/run_semantic_smoke.py --gpu 0
```

Only after both pass, launch the three fixed task shards:

```bash
bash experiments/icra27_two_clock_discriminator_dev/resume.sh
```

After all nine completion markers exist:

```bash
/home/wjq/workspace/venvs/libero_act/bin/python experiments/icra27_two_clock_discriminator_dev/analyze.py
```

Stop after the development report. No additional horizon, adaptive method, or confirmation run is authorized.

# PPPR Overnight Runbook

All commands run from `/home/wjq/workspace/one-clock` with `/home/wjq/workspace/venvs/libero_act/bin/python`. Phase 0 is CPU-only; no simulator or GPU is used.

## Reproduce Phase-0 labels

```bash
/home/wjq/workspace/venvs/libero_act/bin/python -m unittest -v research/overnight_pppr_20260828/test_pppr_phase0.py
/home/wjq/workspace/venvs/libero_act/bin/python research/overnight_pppr_20260828/build_phase0.py
```

The build is idempotent after `phase0_features.complete`. To deliberately replace a complete table after a scientifically required code correction, pass `--force`; do not combine old and new outputs.

## Reproduce control relevance

```bash
/home/wjq/workspace/venvs/libero_act/bin/python -m unittest -v research/overnight_pppr_20260828/test_control_relevance.py
/home/wjq/workspace/venvs/libero_act/bin/python research/overnight_pppr_20260828/analyze_control_relevance.py
```

Expected decision outputs:

- `phase0_control_relevance.json`
- `phase0_control_relevance.md`
- `phase0_pairs.csv`
- `phase0_analysis.complete`

The analysis is idempotent after the completion marker. Its logged full regeneration command was:

```bash
/home/wjq/workspace/venvs/libero_act/bin/python research/overnight_pppr_20260828/analyze_control_relevance.py --force
```

## Resume

```bash
research/overnight_pppr_20260828/resume.sh
```

`resume.sh` skips valid completed artifacts and reconstructs only a missing phase prerequisite. It launches no simulator and no later phase.

## Stop condition

Phase 0 failed (`STOP_PPPR`). Do not add predictor training, ACT PPPR confirmation, adaptive rollout, or any replacement PPPR variant to this runbook.

## Detached jobs

None.

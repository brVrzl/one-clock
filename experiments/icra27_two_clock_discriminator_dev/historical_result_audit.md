# Exact historical-result audit

Audit scope: the fallback tree at `7ea83e1c0bea4367cc722a3d7b72ac0ca827e009` and every commit reachable from local and remote refs before this experiment was created.

## Findings

| Candidate historical artifact | Finding | Exact reusable result? |
|---|---|:---:|
| `experiments/component_temporal_reuse/two_clock_dev/` | Completed arm8/grip16 and arm16/grip8 at 30 Hz on four tasks, states 10–19, seeds 2000–2009, and task-specific ACT checkpoints. It is not arm16/grip32 and does not use this cohort/runtime. | No |
| History commits `20d14e4`, `4d20b6c`, and `2a1f1fa` | Completed static/global and groupwise grids use horizons no larger than 16. Their cohorts are Object task 0 states 0–49 or Object tasks 1–9 states 0–19. | No H32 |
| Temporal-ensemble, dynamic-horizon, group-memory, requery, and delay artifacts | These use averaging, adaptive triggers, dense queries, lagged sources, different horizons, or different cohorts. | No |

Full-history searches for `HARD_H32`, `H32_COHERENT`, `ARM16_GRIP32`, `grip32`, `global_h32`, `fixed_h32`, and equivalent arm/gripper spellings returned no artifact. Numeric horizon searches found no completed fixed 32-step condition; occurrences near 32 in offline reliability analyses are learned/diagnostic quantities, not exact closed-loop conditions.

Conclusion: no provenance-valid exact H32 or true arm16/grip32 result exists. Both authorized conditions require new 126-episode rollouts. Historical HARD_H16 and C1 remain exact reuses and will not be rerun.

## Audit commands

```text
rg -n -i --glob '!paper/**' 'HARD_H32|H32_COHERENT|ARM16_GRIP32|GRIP32|horizon.?32|hard.?h32|two.?clock' .
git log --all --oneline -- experiments
git grep -I -n -E 'HARD_H32|H32_COHERENT|ARM16_GRIP32|arm[_ -]?16[_ -]?grip(per)?[_ -]?32|grip(per)?[_ -]?32|global[_ -]?h32|fixed[_ -]?h32|hard[_ -]?h32' $(git rev-list --all)
```

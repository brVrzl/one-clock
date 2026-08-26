# Repository consolidation, 2026-08-26

## Decision

`integration/icra27-baseline-foundation` starts at `origin/main` and keeps
only the neutral LIBERO execution foundation: the fixed-chunk executor, the
LIBERO runner/configuration, lightweight executor tests, and repository
metadata. Historical experiment and analysis files are removed from this
snapshot only; their commits and branches are preserved.

The standard path is `global_fixed` in `scripts/run_libero_gate0.py`. No
Gate-3C FO behavior, source-age manipulation, DCTA, component-specific
temporal aggregation, selective commitment, adaptive horizon, partial
replanning, Gate-4A2 logic, RoboTwin task experiment, or overnight supervisor
is included or required by this path.

## Inventory snapshot

After `git fetch --all --prune`, the relevant local state was:

- Current pre-consolidation branch: `research/libero-baseline-handoff-20260826`,
  synchronized with its upstream.
- No tracked modified, staged, or deleted files.
- 501 untracked Gate-4A2 artifacts: 500 compressed rollout logs under
  `experiments/gate4a2_spatial_act_generalization/` and
  `research/audit_outputs/gate4a2_spatial_rollout_manifest.json`.
- The untracked Gate-4A2 artifacts were classified as LIBERO
  experiment-specific and preserved on
  `exp/gate4a2-spatial-act-generalization` in commit `e246d3d` (pushed).
- Ignored virtual-environment/cache/generated files were left untouched and
  were not candidates for the foundation.

Upstream status before preservation:

- Synchronized: `exp/fast5080-adaptive-recency`,
  `exp/fast5080-cross-generation-offline`, `exp/gate4a-spatial-generalization`,
  `exp/robotwin-static-validation`, `ops/overnight-5080-prep-20260824`,
  `research/icra27-direction-reset`, `research/libero-baseline-handoff-20260826`,
  and `research/robotwin-paused-20260826`.
- Ahead by five: `exp/icra27-chunkfix-fast`.
- Behind by 31: local `main`.
- Diverged: `integrate/5080-into-main` (six local-only commits ahead and
  three behind its configured `origin/main` upstream).
- No upstream but already reachable from pushed refs: the local backup and
  Gate-4A2/post-Gate-3C pointers.

## Local-only work preserved

| branch | commit SHA | short purpose | pushed | classification |
|---|---|---|---|---|
| `exp/icra27-chunkfix-fast` | `638ee1a` | start ManiSkill counterfactual policy tournament | yes, same branch | experiment-specific |
| `exp/icra27-chunkfix-fast` | `00403fe` | add ManiSkill counterfactual causal gate | yes, same branch | experiment-specific |
| `exp/icra27-chunkfix-fast` | `730ad42` | run matched ManiSkill policy smoke gate | yes, same branch | experiment-specific |
| `exp/icra27-chunkfix-fast` | `cab7971` | add ManiSkill fragility sweep and ACT allocation gate | yes, same branch | experiment-specific |
| `exp/icra27-chunkfix-fast` | `c87fee1` | add corrective fragility recovery gate | yes, same branch | experiment-specific |
| `exp/robotwin-dcta-development` | `d776d33` | freeze RoboTwin DCTA development run | yes, same branch | RoboTwin-specific |
| `exp/robotwin-dcta-development` | `91fb3a2` | resolve frozen DCTA gate paths | yes, same branch | RoboTwin-specific |
| `exp/robotwin-dcta-development` | `d672da7` | include shared dynamic baseline contrast | yes, same branch | RoboTwin-specific |
| `exp/robotwin-exploratory-sealed` | `dde7099` | preregister sealed RoboTwin exploratory pilot | yes, same branch | RoboTwin-specific |
| `exp/robotwin-exploratory-sealed` | `054b79f` | freeze RoboTwin exploratory evaluation | yes, same branch | RoboTwin-specific |
| `exp/robotwin-exploratory-sealed` | `42ce3b5` | add deterministic RoboTwin follow-up branches | yes, same branch | RoboTwin-specific |
| `exp/robotwin-exploratory-sealed` | `c32defc` | freeze RoboTwin exploratory no-signal result | yes, same branch | RoboTwin-specific |
| `exp/robotwin-exploratory-sealed` | `f36be2d` | add RoboTwin no-signal diagnostics | yes, same branch | RoboTwin-specific |
| `integrate/5080-into-main` | `a059123` | add resumable RoboTwin reliability cache pipeline | yes, preservation branch | RoboTwin-specific |
| `integrate/5080-into-main` | `62b1cbc` | record RoboTwin provenance and preflight | yes, preservation branch | RoboTwin-specific |
| `integrate/5080-into-main` | `022fa84` | clarify RoboTwin resume command | yes, preservation branch | RoboTwin-specific |
| `integrate/5080-into-main` | `7ededfb` | gate cache generation on verified action ordering | yes, preservation branch | OPS/machine-specific |
| `integrate/5080-into-main` | `d04cdf5` | document unresolved policy action semantics | yes, preservation branch and historical branch | OPS/machine-specific |
| `integrate/5080-into-main` | `50df739` | merge 5080 integration history | yes, preservation branch | mixed RoboTwin/OPS |

## Branch-content classification

- Reusable foundation: the neutral LIBERO fixed-chunk executor/action
  contract, LIBERO runner/configuration, and lightweight executor tests kept
  in this branch.
- LIBERO experiment-specific: Gate-0 horizon sweeps, Gate-3A/3B/3C/4A2
  interventions, source-age work, DCTA, selective commitment, adaptive or
  partial replanning, and related result/audit artifacts. Historical branches
  remain available.
- Invalid or historical-only: superseded Gate-4A2 logic and old overnight
  orchestration.
- RoboTwin-specific: all RoboTwin runners, temporal methods, results, and
  task-specific configs.
- Research/analysis: paper sources, preregistrations, reports, and audit
  outputs, including `research/icra27-direction-reset`; not merged.
- OPS/machine-specific: overnight preparation, Thor/5080 handoffs, and the
  diverged integration history; preserved separately.

## Included foundation paths

- `src/one_clock/executor.py` and `src/one_clock/__init__.py`
- `scripts/run_libero_gate0.py`
- `configs/gate0_libero_object.yaml`
- `tests/test_executor.py`
- `README.md`, `.gitignore`, and this inventory

All other tracked files from `origin/main` were excluded from this clean
foundation snapshot because they are experiment, research, analysis, paper,
RoboTwin, or machine-specific material.

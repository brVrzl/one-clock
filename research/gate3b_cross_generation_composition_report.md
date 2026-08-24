# Gate-3B cross-generation composition report

**GATE DECISION: COMPOSITION-HARM-SUGGESTIVE**

**PRIMARY RESULT:** The preregistered coherence contrast is `+0.025`: the two
source-coherent cells average 42.0% success and the two cross-generation cells
average 39.5%. The equivalent standard 2×2 interaction is `+0.050`.

**UNCERTAINTY:** The 20,000-draw paired task-state bootstrap 95% interval is
`[-0.030,+0.085]`; the 20,000-draw task-cluster interval is
`[-0.005,+0.055]`. Both include zero. Every leave-one-task-out contrast remains
positive, ranging from `+0.0167` to `+0.0333`.

**BOUNDARY OF THE RESULT:** The point estimate and concentration diagnostic are
directionally consistent with average composition harm, but neither bootstrap
criterion confirms it. The four cells are also strongly asymmetric: `FO`
succeeds 62/100 while `OF` succeeds 17/100. This experiment therefore does not
establish that mixed-source actions are generally harmful, does not revive a
group-wise method, and does not authorize mechanism design.

## 1. Question and registration

Gate-3B asked:

> Does independently refreshing components of a jointly predicted action chunk
> harm closed-loop control because it recomposes action groups across policy
> source generations?

The exact design, source age, cohort, episode seeds, 400-run order, estimand,
bootstrap seeds, decision rule, and stop conditions were frozen and pushed in
[`gate3b_cross_generation_preregistered_protocol.md`](gate3b_cross_generation_preregistered_protocol.md)
at commit `510908ed438cba72a30a5a679b91b339f4297c65`, before any official
success outcome was generated. The exact schedule SHA256 is
`2cae2712ff00a1bc5bf8c3eb808b69aeaa8208acb62d2f8591e476e7b18ce4ff`.
No official episode was retried, excluded, quarantined, or replaced, and no
protocol amendment occurred.

This is a composition experiment, not temporal ensembling or execution-horizon
selection. At every surviving 20 Hz controller step, every condition queries
the same frozen ACT once, receives the current observation, and stores the full
fresh 100×7 chunk. Only the executed source assignment differs.

## 2. Frozen factorial intervention

At physical time `t`, `F_t=E_{t,t}` is the current query's first action. For
`t>=20`, `O_t=E_{t,t-20}` is chunk offset 20 from source query `q=t-20`. The
fixed 20-tick age is exactly 1.0 second. For `t<20`, all four conditions execute
`F_t` in full.

For `t>=20`:

| Cell | Executed arm | Executed gripper | Source relation |
|---|---|---|---|
| `FF` | `F_t[0:6]` | `F_t[6]` | joint fresh |
| `OO` | `O_t[0:6]` | `O_t[6]` | joint old20 |
| `FO` | `F_t[0:6]` | `O_t[6]` | cross-generation |
| `OF` | `O_t[0:6]` | `F_t[6]` | cross-generation |

No candidate is averaged, smoothed, thresholded, semantically weighted, or
temporally ensembled. The policy's saved temporal-ensemble coefficient is
disabled.

## 3. Cohort and execution validity

The frozen states are `[24,26,28,29,32,33,37,40,46,49]`, sampled without
replacement from the registered higher-index pool using seed `20260827`. They
exclude every Gate-3A2 state and are shared across all ten tasks. Task 0 had
historically used all 50 states, so its Gate-3B states are not described as
previously untouched. Method order is independently randomized within each of
the 100 task-state blocks using seed `20260828`; every block's four methods
share episode seed `320000 + 100*task_id + state_id`.

Post-result validation established:

- 400 unique scheduled task-state-method cells and 400 complete local logs;
- 88,171 environment steps and exactly 88,171 policy queries;
- one query per surviving step in every episode;
- exact `q=t-20` old-source identity and chunk offset 20 at every intervention
  step;
- exact registered action formula at every step, with maximum error 0;
- identical first 20 actions across all four methods in all 100 blocks, with
  maximum difference 0;
- finite seven-dimensional executed actions throughout;
- no policy temporal ensemble, composition ensemble, or action smoothing.

The 400 local gzip logs occupy 20,618,916 bytes at
`/home/thor/projects/one-clock/experiments/gate3b_cross_generation_composition`.
Their content-tree SHA256 is
`046eedc6921205b28eacc6d24f7ce6bbc2d250b305c98383055935c100eac83d`.
The committed [rollout manifest](audit_outputs/gate3b_rollout_manifest.json)
records every absolute path, byte size, file SHA256, outcome, step/query count,
and provenance. The compact integrity result is
[`gate3b_rollout_validation.json`](audit_outputs/gate3b_rollout_validation.json).

## 4. Primary coherence contrast

For every task-state block, the frozen primary estimand is

\[
C_{coherence}=\tfrac12(success_{FF}+success_{OO})
-\tfrac12(success_{FO}+success_{OF}).
\]

The estimate is `+0.025`, and the standard interaction is exactly twice that,
`+0.050`.

| Quantity | Result |
|---|---:|
| Same-source average success, `(FF+OO)/2` | .420 |
| Cross-source average success, `(FO+OF)/2` | .395 |
| `C_coherence` | +.025 |
| Standard 2×2 interaction | +.050 |
| Paired-state bootstrap 95% CI, 20,000 draws | [-.030,+.085] |
| Task-cluster bootstrap 95% CI, 20,000 draws | [-.005,+.055] |
| Leave-one-task-out range | [+.0167,+.0333] |

Task-level coherence contrasts are:

| Task | FF | OO | FO | OF | `C_coherence` |
|---:|---:|---:|---:|---:|---:|
| 0 | .30 | .30 | .60 | .10 | -.05 |
| 1 | .30 | .40 | .60 | .00 | +.05 |
| 2 | .90 | .90 | 1.00 | .60 | +.10 |
| 3 | .50 | .20 | .50 | .20 | .00 |
| 4 | .30 | .50 | .80 | .00 | .00 |
| 5 | .40 | .60 | .70 | .10 | +.10 |
| 6 | .40 | .30 | .40 | .40 | -.05 |
| 7 | .30 | .20 | .40 | .00 | +.05 |
| 8 | .10 | .10 | .20 | .00 | .00 |
| 9 | .90 | .50 | 1.00 | .30 | +.05 |

The task contrasts are positive for five tasks, zero for three, and negative
for two. Removing any one task leaves a positive mean, so the small positive
estimate is not created by one favorable task. The bootstrap intervals still
include zero; this is why the frozen gate is suggestive rather than confirmed.

## 5. Four cells and descriptive pairwise comparisons

| Cell | Successes | Rate | Environment steps | Policy queries |
|---|---:|---:|---:|---:|
| `FF` | 44/100 | .44 | 21,524 | 21,524 |
| `OO` | 40/100 | .40 | 22,064 | 22,064 |
| `FO` | 62/100 | .62 | 19,085 | 19,085 |
| `OF` | 17/100 | .17 | 25,498 | 25,498 |

All six pairwise differences are descriptive and secondary. The first cell
minus the second cell is reported below.

| Comparison | Success difference | First-only / second-only |
|---|---:|---:|
| `FF−OO` | +.04 | 18 / 14 |
| `FF−FO` | -.18 | 2 / 20 |
| `FF−OF` | +.27 | 32 / 5 |
| `OO−FO` | -.22 | 5 / 27 |
| `OO−OF` | +.23 | 25 / 2 |
| `FO−OF` | +.45 | 48 / 3 |

The four direct decompositions do not tell one uniform story. With a fresh arm,
an old gripper (`FO`) is descriptively better than a fresh gripper (`FF`) by
18 points. With an old arm, an old gripper (`OO`) is descriptively better than
a fresh gripper (`OF`) by 23 points. With an old gripper, a fresh arm (`FO`) is
descriptively better than an old arm (`OO`) by 22 points; with a fresh gripper,
a fresh arm (`FF`) is better than an old arm (`OF`) by 27 points.

Marginally, fresh-arm cells average 53.0% versus 28.5% for old-arm cells, while
fresh-gripper cells average 30.5% versus 51.0% for old-gripper cells. These are
secondary source-age main effects, not the preregistered coherence mechanism.
The balanced interaction remains the only primary test and is much smaller
than the cell asymmetry.

## 6. Secondary action diagnostics

| Cell | Mean steps | Mean arm age | Mean grip age | Translation delta | SO(3) delta, rad | Raw jerk | Grip transitions | Distance to nearest joint source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `FF` | 215.24 | 0.00 | 0.00 | .04698 | .00494 | .13709 | 3.13 | .00000 |
| `OO` | 220.64 | 17.90 | 17.90 | .07537 | .00679 | .23824 | 8.80 | .00000 |
| `FO` | 190.85 | 0.00 | 17.58 | .06407 | .00665 | .16813 | 4.41 | .01691 |
| `OF` | 254.98 | 18.29 | 0.00 | .06150 | .00563 | .11243 | 1.03 | .05925 |

Mean source ages are below 20 because every episode begins with 20 common fresh
actions and successful episodes can terminate early. The diagnostics are
episode-weighted summaries along different treatment-dependent trajectories.
They are not paired causal mediators. In particular, `OF` has the lowest raw
jerk among the four cells but also the lowest success, so the data do not
support a claim that jerk explains the success ordering. Distance from the two
jointly predicted source actions is a control-semantic action distance, not an
action-manifold measurement.

## 7. Scientific interpretation

The experiment cleanly removes several confounds from the earlier selective
retention failure. Every method queries on every surviving step; there is no
threshold rule, source exhaustion, age search, scalar aggregation, or changed
execution horizon. Marginal fresh/old source assignments are balanced in the
coherence contrast.

The resulting primary point estimate is positive and survives every
leave-one-task-out omission, but both registered bootstrap intervals include
zero. The correct conclusion is therefore limited:

> Gate-3B provides suggestive, not confirmatory, evidence that source-coherent
> arm/gripper execution has higher average success than cross-generation
> recomposition in this frozen ACT/LIBERO system at `d=20`.

The large `FO`/`OF` asymmetry is a real observed cell pattern but is not the
primary result. It shows why reporting only a favorable pairwise comparison
would be misleading. This result does not establish a policy-manifold
violation, universal joint-action coherence law, general harm from component
adaptation, or causation through discontinuity/jerk. It does not generalize to
bimanual, dexterous, VLA, other policy, or real-robot settings.

Because the gate is not confirmed, the stronger preregistered positive wording
is not used. The old group-wise method is not revived, and secondary
diagnostics cannot rescue the unresolved primary success result.

## 8. Tests and exact invocations

The targeted implementation tests passed in the rollout environment:

```text
/home/thor/projects/upstreams/lerobot-env/bin/python -m pytest -q \
  tests/test_gate3b_composition.py tests/test_gate3a2_temporal_aggregation.py
21 passed in 6.26s
```

The initial broad repository invocation failed during collection because
`src/` was absent from `PYTHONPATH`; it did not exercise or invalidate a
Gate-3B test. The exact corrected invocation was:

```text
PYTHONPATH=src:. /home/thor/projects/upstreams/lerobot-env/bin/python \
  -m pytest -q tests
58 passed in 5.95s
```

The non-outcome runtime check loaded the pinned task-0 policy/environment and
validated the contract without resetting or executing an episode. After
rollout, `gate3b_rollout.py --verify-only`, the frozen analyzer, and the
post-result validator all completed successfully.

## 9. Decision and stop

The final registered label is **COMPOSITION-HARM-SUGGESTIVE**. The experiment
ends at the completed 400-episode result. No consistency-constrained method,
adaptive group horizon, learned router, age sweep, PACE experiment, or new
benchmark is started.

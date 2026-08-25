# Gate-4A2 Spatial ACT generalization report

Status: **INVALIDATED BEFORE OUTCOME ANALYSIS**

Gate-4A2 completed all 500 scheduled environment episodes, but the cohort
failed a preregistered technical integrity requirement before success outcomes
were opened. The experiment therefore has no valid success-rate comparison and
no frozen Spatial-generalization decision. No episode was excluded or rerun,
and the manuscript remains unchanged.

## Asset and preregistration identity

The originally proposed `zeromidnight/act_libero_spatial` candidate remains
recorded as an asset-gate failure before any outcome. The single replacement
candidate passed the non-outcome asset and execution-contract audit:

- repository: `ishandotsh/act_libero_spatial_test`
- immutable revision:
  `8f04de1472975d62db214238b2fc07e78bde2474`
- model SHA256:
  `912f41808962d80ca9084435aa01eccccdd97b7eae3a841c9f4ac71caaf9f8b0`
- config SHA256:
  `0e783369890d33a714cef603185c10dff4215328a9862b181eb7f511f3f1a93c`
- preprocessor SHA256:
  `8a5df04ea1f67ab515898ba211bc64b6c38020e259bc0bd520ddd7b38a660128`
- postprocessor SHA256:
  `c27cf6f42b42352f9b8f9c40da155fd4459e0ee9b85b9f23072941eb52b3ffb5`
- normalizer and unnormalizer SHA256:
  `a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9`
- training-config SHA256:
  `551dd7bdb8b4ffb109f3ebc40a26856b72953188a74b4a02d597ba2989528b5f`
- training provenance: **MULTI-SUITE**, spanning Spatial, Object, Goal, and
  LIBERO-10, not Spatial-only

The evaluation dataset was
`zeromidnight/libero_spatial_lerobot_v3.0` revision
`38927e939de5d2bfd40effcf27d16710aea6f864`. The runtime used vanilla LIBERO
Spatial tasks 0--9, official states, relative 7-D control, hard reset, a
280-step horizon, and a directly audited 20 Hz controller. Thus the frozen
`d=20` intervention was 20 controller ticks, or 1.0 s.

The preregistration was committed and pushed at
`fbcd910d4f2aa6fad7a6228708249bd4dc1bf04e` before the first official episode.
The ordered schedule SHA256 was
`00ba884d488891569008ab44214e924143f1cb7992fd57383ab057f5b6c16833`.
The same selected state IDs were used for every task:

```text
[1, 13, 15, 19, 21, 24, 31, 37, 40, 47]
```

## Execution completeness

The runner produced 500 unique scheduled task-state-method cells and 500 local
compressed episode traces. All 110,624 environment steps had exactly one policy
query. Schedule identities, episode seeds, official initial-state IDs and
vectors, checkpoint/config/processor hashes, temporal source identities, and
trace hashes matched the registration. No extra, missing, excluded, or retried
episode file was found.

The local trace content-tree SHA256 is
`e11cacfc5dd8f15dcd8374961229df3fb33641a1ba4669032e67c89f71ec8e7c`.
The 500 compressed traces occupy 51,276,946 bytes and remain local. The rollout
manifest SHA256 is
`6037134863168645605aa03cc99db881bc79defd8983694c55075bb0d9290470`.

## Preregistered validation failure

The fixed-source A/B/C formulas were exact at every recorded step (maximum
formula error 0), the age-exponential weights exactly matched beta 0.03
(maximum error 0), CogACT retained the frozen Gate-3C source identity, all
actions were finite 7-D vectors, `q=t-20` and chunk offset 20 were exact, and
policy temporal ensembling and action smoothing were both disabled.

The preregistered validator nevertheless failed the requirement that the first
20 executed actions be identical for A, B, and C in every paired task-state
block. Only 89 of 100 blocks passed. Eleven failed:

- Spatial task 3, state 1, first mismatch at controller step 8;
- Spatial task 4, every selected state, first mismatch at controller step 0.

Task 3 is `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`.
Task 4 is
`pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`.
The maximum absolute A/B/C first-prefix action difference was
`0.19852697849273682`, at task 4, state 37, step 7.

This is not a formula mismatch: all three methods execute their own fresh action
for `t<20`. It shows that the supposedly paired executions did not produce the
same fresh-action prefix in those blocks despite shared state IDs and episode
seeds. The registered fairness criterion is therefore false for the completed
cohort. The validation record does not assign a post-outcome causal explanation
for the divergence.

## Statistical and scientific disposition

The primary success fields were not analyzed after the validation failure.
Consequently the five success rates, four contrasts, paired and task-cluster
bootstrap intervals, exact McNemar/binomial diagnostics, per-task differences,
and leave-one-task-out estimates were deliberately not generated. None of the
registered labels (`STRONG`, `DIRECTIONAL`, `SUGGESTIVE`, `NULL`, or
`CONTRADICTED`) is assigned to an integrity-invalid cohort.

The failure is technical rather than outcome-triggered. Under the frozen rules,
rerunning, excluding, or replacing these official episodes would violate the
registered no-rerun and no-exclusion contract. Gate-4A2 therefore stops here.
It does not provide evidence for or against external generalization of the
Gate-3C Object result.

## Software and tests

The frozen stack was LeRobot 0.6.2 at clean commit
`f66e5128ecb2456e8c54a63d15404fa59c16aebc`, `hf_libero` 0.1.4,
robosuite 1.4.0, MuJoCo 3.8.1, Python 3.12.3, PyTorch 2.11.0+cu130,
CUDA 13.0, NumPy 2.2.6, SciPy 1.18.0, Gymnasium 1.3.0, EGL rendering,
NVIDIA driver 595.84, and an NVIDIA GeForce RTX 5080 on Linux
6.8.0-136-generic.

Before outcome generation, five focused synthetic tests passed for direct
Gate-3C executor reuse, source indexing, fresh-prefix formulas, shared scalar
baseline weights, state/schedule randomization, resume order, and one-query-per-
step bookkeeping. Model/environment construction also passed without executing
an outcome. After 500 episodes, the preregistered validator correctly stopped
on the A/B/C prefix-integrity failure. The machine-readable record is
[`audit_outputs/gate4a2_spatial_rollout_validation.json`](audit_outputs/gate4a2_spatial_rollout_validation.json).

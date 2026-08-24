# Gate-3A2 LIBERO time-contract audit

Audit date: 2026-08-24

Verdict: **TIME-CONTRACT-MISMATCH (RESOLVED BEFORE ROLLOUT)**

The frozen demonstration copy says 10 Hz, but its stored frames and actions are
an unreduced copy of a 20 Hz LIBERO sequence. The historical project rollout
path advances LIBERO once per 20 Hz controller tick. Consequently, one stored
dataset index corresponds to one 20 Hz environment action (0.05 s), a
100-action ACT chunk spans 5 s, and Gate-3A1's selected `beta=0.03` per stored
index must be used as `beta_tick=0.03` online. The proposed conversion to
`0.015` per rollout tick would halve the decay and is rejected.

This mismatch invalidates prior physical-time labels derived only from the
frozen dataset's `fps=10` metadata. It does **not** invalidate index-domain
Gate-3A1 method rankings or the historical 20 Hz rollout path.

## 1. Evidence hierarchy and question-by-question resolution

### Actual environment/control frequency

The active rollout configuration sets `control_freq: 20` in
[`configs/gate0_libero_object.yaml`](../configs/gate0_libero_object.yaml).
[`scripts/run_libero_gate0.py`](../scripts/run_libero_gate0.py) passes that
value to the pinned LeRobot `LiberoEnv`, queries one ACT chunk inside each
runner iteration, and calls `env.step` once for the selected action. The pinned
LeRobot environment:

- defaults LIBERO to 20 Hz and says it must match robosuite's default;
- passes `control_freq` to LIBERO's `OffScreenRenderEnv`;
- advances one underlying environment step per wrapper `step` call.

Installed LIBERO also defaults `control_freq=20`. Installed robosuite defines
`control_timestep = 1/control_freq`, advances simulation by that amount in each
environment step, and increments its clock by that timestep. The actual
project rollout control frequency is therefore **20 Hz (0.05 s/action)**.

Primary local source paths and inspected definitions:

- `/home/thor/projects/embodied_lab/third_party/lerobot/src/lerobot/envs/configs.py`
- `/home/thor/projects/embodied_lab/third_party/lerobot/src/lerobot/envs/libero.py`
- `/home/thor/projects/upstreams/lerobot-env/lib/python3.12/site-packages/libero/libero/envs/env_wrapper.py`
- `/home/thor/projects/upstreams/lerobot-env/lib/python3.12/site-packages/robosuite/environments/base.py`

### How the frozen demonstration frames map to physical control

The frozen local copy
`DorayakiLin/libero_object_25_08_23_lerobotv2.1@cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a`
reports 10 Hz, 454 episodes, and 66,984 frames. That metadata alone would imply
0.1 s between stored actions.

An independent public LeRobot conversion of the same LIBERO no-noops data,
`IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot@15657dac2ad1c01b4e94bf54ab0493b46a8d63f9`,
reports 20 Hz, 454 episodes, and 66,984 frames. The following primary-content
checks establish that the local 10 Hz label did not arise from keeping every
second source action:

1. The two `episodes.jsonl` files are byte-identical: 46,501 bytes, SHA256
   `63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea`.
2. Episodes 0, 212, and 417 have identical lengths. Their 8-D state arrays and
   action dimensions 0–5 are exactly equal elementwise.
3. Their gripper values differ only by a deterministic convention:
   `local_gripper = 1 - 2 * IPEC_gripper` (local sign command versus IPEC
   0/1 command). There is no temporal subsampling.
4. Local timestamps advance by 0.1 s; IPEC timestamps advance by 0.05 s.
   Because the state/action sequences are otherwise identical, this is a
   timestamp relabeling, not a different physical cadence.
5. Episode 49's agent-view videos both contain 153 frames. IPEC encodes them at
   20 fps and the local copy at 10 fps; decoded frames have mean absolute pixel
   difference 1.44 from codec re-encoding, not a factor-of-two frame drop.

The exact script that produced the DorayakiLin copy is absent, so the cause of
the incorrect 10 Hz metadata is **UNKNOWN**. A public conversion script from
the same conversion lineage iterates every RLDS step while setting `fps=10`,
which is compatible with the observed relabeling but is not proof of this
copy's creator path.

### Learned chunk axis and historical execution

The checkpoint config records `chunk_size=100` and `n_action_steps=100`. ACT
predicts ordered action indices; it contains no physical-time resampler. The
training copy retained every source action index, and the runner consumes one
chunk index per 20 Hz LIBERO step. Therefore the learned and executed index
axes agree:

| Quantity | Verified interpretation |
|---|---|
| One stored dataset step | One 20 Hz LIBERO action, 0.05 s |
| One historical rollout tick | One 20 Hz LIBERO action, 0.05 s |
| One ACT chunk index | One rollout tick |
| 100-action ACT chunk | 5.0 s |

The checkpoint's saved training config also contains an `env.fps=30` field.
No environment was used inside ACT's index-based supervised loss, and the
frozen runtime uses the explicit 20 Hz project config. This stale field does
not define either training sample spacing or rollout timing.

## 2. Consequence for Gate-3A1 and Gate-3A2

Gate-3A1 selected `beta=0.03` using candidate age measured in stored action
indices. The weights were:

\[
w(k) \propto \exp(-0.03 k).
\]

Since one index is 0.05 s, the continuous decay is `0.03/0.05 = 0.6 s^-1`.
At 20 Hz the physically identical online rule is therefore:

\[
w(k_{tick}) \propto \exp(-0.03 k_{tick}),
\]

not `exp(-0.015 k_tick)`. The latter corresponds to `0.3 s^-1` and was never
selected by Gate-3A1.

The Gate-3A1 numerical comparisons remain valid in their stored-index domain,
because every method used the same candidate indices. Its statements that the
dataset was physically 10 Hz, that one age step was 0.1 s, and that `beta=0.03`
was `0.3 s^-1` are contradicted by this audit and must not be reused.

## 3. Prior-art coefficient audit

The pinned LeRobot checkout is commit
`f66e5128ecb2456e8c54a63d15404fa59c16aebc`. Its
`ACTTemporalEnsembler` orders contributing sources oldest to newest and uses
`w_i=exp(-m i)`, with source-order index zero the oldest. It documents original
ACT's default `m=+0.01`: positive values favor older sources and negative
values favor newer sources. The
[official original ACT evaluation source](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)
uses the same oldest-to-newest `exp(-0.01 i)` convention.

[LeRobot PR #319, “Fix ACT temporal ensembling”](https://github.com/huggingface/lerobot/pull/319)
was merged on 2024-07-16. Its ALOHA transfer-cube experiments report 500-episode
success rates of 87.6% without temporal ensembling, 73.8% for `m=+0.01`, 76.8%
for `m=0`, and 79.0% for `m=-0.01` (plus other coefficients in the PR). These
are prior-work results, not one-clock results. They make “favor newer chunks”
ineligible as a novelty claim.

## 4. Scientific stopping decision

The mismatch is resolved from primary sequence content and active execution
code. It changes only the physical-time annotation and the coefficient
conversion. Gate-3A2 can proceed safely with a 20 Hz controller and
`beta_tick=0.03`, provided all methods query once per surviving controller
step. Had only the 10 Hz metadata been available, the mapping would have
remained unresolved and rollout execution would have stopped here.

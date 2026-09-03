# Superseding temporal-contract audit

Status: **CURRENT; SUPERSEDES THE 2026-09-03 NOMINAL-FPS AUDIT**

Audit completed at `2026-09-03T11:05:14+08:00`. This document was written
without opening any reviewer-supplement result or aggregating any partial
reviewer-supplement outcome.

## Superseded record

This audit supersedes, but does not modify or erase:

- `TEMPORAL_CONTRACT_AUDIT.md`, introduced by Git commit
  `43e0b50ac79cfb5dbbce5358967f53b3d651d954`, file SHA-256
  `d234909adb5016a0df181cff54544fe40c182da55f4423ff6e279f2da4f2aabe`;
- `temporal_contract_audit.json`, introduced in the same commit, file SHA-256
  `1281315c4f485a943c9a831510369de9f43e90e5248e3446c8f3a7beb77f1eb8`.

The prior audit inferred physical sampling and control time primarily from
nominal dataset and evaluator `fps` fields. That inference was incorrect. The
dataset timestamps say 10 Hz, but the stored sequence retains every frame and
action from a 20 Hz LIBERO sequence. In the pinned LeRobot 0.4.4 evaluator,
the configured LIBERO `fps` field is not forwarded to `control_freq`, so the
environment executes at LIBERO's default 20 Hz.

## Demonstration and ACT chunk indexing

The following evidence establishes that one ACT training action index is one
physical 20 Hz control step, or 0.05 seconds:

1. ACT defines `action_delta_indices` as the complete integer range
   `0..chunk_size-1`.
2. The LeRobot dataset factory converts these integers to nominal timestamps,
   then `LeRobotDataset` maps them back to the same integer offsets and reads
   exact rows `abs_idx + delta`. It does not interpolate, repeat, or drop
   action rows; only episode-end padding clamps out-of-range indices.
3. Content comparisons between the nominal-10-Hz
   `/home/wjq/research-assets/datasets/HuggingFaceVLA_libero` copy and the
   independent 20-Hz
   `/home/wjq/research-assets/libero_dcta/datasets/nvidia_libero_v3` conversion
   found identical row counts and ordered state/arm-action arrays for Goal,
   Object, and Spatial. Gripper actions matched under the documented
   deterministic sign/state convention. The timestamp increments alone were
   0.1 versus 0.05 seconds.
4. The merged dataset counts are LIBERO-10 101,469, Goal 52,042, Object
   66,984, and Spatial 52,970 frames, exactly matching the independent 20-Hz
   conversions. The independent Object audit also found byte-identical episode
   records, equal episode lengths, and equal video frame counts.
5. The standard task-specific ACT checkpoints name this merged dataset. The
   historical Object checkpoint uses the nominal-10-Hz Object copy covered by
   the independent content audit. The Spatial checkpoint uses the same
   standard LIBERO content lineage.

Therefore ACT chunk element `k` is supervised against stored frame `t+k`, and
its physical target is `k * 0.05` seconds after the observation. The nominal
10-Hz timestamps relabel the sequence but do not change which action row is
the target.

## Evaluator indexing

In installed LeRobot 0.4.4, `LiberoEnv.gym_kwargs` includes observation and
rendering fields but not `fps`. The environment factory creates
`OffScreenRenderEnv` without `control_freq`; LIBERO 0.1.1 and robosuite 1.4.0
therefore use their default `control_freq=20`. Each `env.step` advances 0.05
seconds. Robosuite performs 25 internal 0.002-second simulator/controller
substeps per environment step, holding one high-level goal over that interval.
There is no additional executor action repeat, frame skip, or interpolation.

R1D uses the separately pinned LeRobot source commit
`f66e5128ecb2456e8c54a63d15404fa59c16aebc`, which explicitly forwards
`control_freq=self.fps`; its sealed value is 20 Hz. It therefore has the same
0.05-second executor step.

For the relevant ACT rollouts, `q`, `t`, source age `d`, and chunk offset `k`
are expressed in the same 20-Hz step clock. The executable same-target check
`q+k=t` consequently also satisfies
`k * dt_policy = d * dt_executor = 0.05*k = 0.05*d`.

## Family classifications

| Family | Classification | Reason |
|---|---|---|
| Frozen 140-block ACT confirmation | `PHYSICAL_SAME_TARGET_VALID` | ACT index and executor step are both 0.05 s |
| Historical Object 126-block cohort | `PHYSICAL_SAME_TARGET_VALID` | Object ACT index and executor step are both 0.05 s |
| Track A | `PHYSICAL_SAME_TARGET_VALID` | Sealed nominal 10-Hz field was not applied; actual evaluator and ACT index are 20 Hz |
| Track B ACT | `PHYSICAL_SAME_TARGET_VALID` | Same standard ACT and evaluator paths |
| R1A/R1B | `PHYSICAL_SAME_TARGET_VALID` | Object ACT and executor clocks both resolve to 20 Hz |
| R1C | `PHYSICAL_SAME_TARGET_VALID` | Standard ACT and executor clocks both resolve to 20 Hz |
| R1D | `PHYSICAL_SAME_TARGET_VALID` | Standard Spatial ACT and explicitly configured 20-Hz evaluator |
| Track B / R2A SmolVLA | `NOT_IDENTIFIABLE_FROM_AVAILABLE_PROVENANCE` | Evaluator is actually 20 Hz, but the checkpoint model card records its training dataset as unknown |

No ACT family requires a `PHYSICAL_SAME_TARGET_VALID_VIA_EXPLICIT_RESAMPLING`
qualification: the validity comes from retained one-to-one 20-Hz content, not
from an explicit resampling operation.

## Corrected physical-time mappings

- Track A: H2 = 0.10 s, H4 = 0.20 s, H16 = 0.80 s, H32 = 1.60 s.
- Track B B1 source ages 0..15 = 0..0.75 s.
- B2/B3 lags or chunk offsets 0..32 = 0..1.60 s.
- B3 anchor stride 10 indices = 0.50 s.
- R1A `d={2,4,8,12,16,20,32}` =
  `{0.10,0.20,0.40,0.60,0.80,1.00,1.60}` s.
- The B2 Kaplan-Meier evaluations already stored as `S(5)`, `S(10)`, and
  `S(20)` correspond to 0.25, 0.50, and 1.00 s, not 0.5, 1.0, and 2.0 s.
- The conditional-mechanism gripper transition-density values calculated with
  nominal `fps=10` must be multiplied by two for physical transitions/second.
  This common positive rescaling does not change task ranks or Spearman rho.

B3 froze the full contiguous offset curve 0..32 before prediction
interpretation. Its indices remain valid and no offsets should be added. Its
seconds labels must use `offset/20`. For ACT, R1A ages match B3 at the same
integer offsets `k=d`; the prior factor-two mapping is superseded.

## Scientific-integrity conclusion

The nominal-FPS error changes physical-time labels and some cross-artifact
time mappings, but it does not change the executed ACT actions, cohorts,
conditions, seeds, source identities, or `q+k=t` alignment. Existing ACT
same-target scientific results therefore remain valid. Future reports must use
the corrected seconds axes. SmolVLA evidence must not be described as a
physically time-matched cross-policy replication unless independent training
timebase provenance becomes available.

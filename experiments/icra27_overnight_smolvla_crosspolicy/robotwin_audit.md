# RoboTwin audit (no rollout)

The sealed exploratory artifacts on `origin/exp/robotwin-exploratory-sealed` and `origin/research/icra27-direction-reset` were audited without launching RoboTwin.

- Design: randomized complete blocks, five tasks × six methods × 20 expert-eligible seeds = 600 completed cells; `technical_reruns=0`.
- Embodiment/setting: RoboTwin 2.0 / XPolicyLab ACT, `aloha_agilex`, bimanual 14-D absolute joint targets; left arm 0–5, left gripper 6, right arm 7–12, right gripper 13. The task context is the official **Easy** reference setting with clean demonstrations.
- Tasks: `beat_block_hammer`, `click_alarmclock`, `dump_bin_bigbin`, `handover_block`, `open_laptop`.
- Checkpoints: final seed-0, 6000-epoch `policy_last.ckpt` per task under `demo_clean-<task>-aloha_agilex-joint-0/`. Verified SHA256 values are respectively `7f3a058419b82464aeeb48d414a8b948eba55220ff5b4b82f16385a0383862fd`, `73a475b8a2f97d3998ce1e90d26439cd8a701feb9a2ab1659466bad0da869c1c`, `5d26180719fded89edf5587674281fa4c3d90470ced743ab504bb40074370781`, `dfb2801ab20b820a844cbdb896989ff443abb5499d3c10f9749bfc84619dc78c`, and (locally verified for the preregistered pending artifact) `9e8366772f163c78e18f91cbb1685423c34fca859bad2e03308ee60700444b4e` for `open_laptop`. Common config SHA256 is `9d38e4f1696926fc87facdb3d42bd1ac5e97b8b9339a23446b9ff40833668857`.
- Seeds: the first 20 official-expert-eligible seeds in ascending order from 100000, selected independently per task and frozen in `robotwin_exploratory_eligible_seeds.json`; no method-specific replacement.
- Methods: official temporally aggregated `NATIVE_ACT`; nonaggregated `NEWEST`; same-target `FULL_OLD_1S`; fresh arms plus same-target old grippers `FO_1S`; `GRIPPER_HOLD`; and postprocessed-command `GRIPPER_EMA_1S` with tau=1 s.
- Seal: task/method/seed schedule hash `467e11065033b12c1cf865ede301ed368e040014cde8cecf23903702d2ae705a`; per-cell outcomes were sealed until all 600 cells completed; valid policy failures were retained; only identical-cell technical retry was allowed (maximum three attempts).
- Preregistered interpretation: exploratory only, no confirmatory p-value; the pilot could only justify a later preregistered confirmation.

Verified pooled outcomes: NATIVE_ACT 19/100, NEWEST 11/100, FO_1S 11/100, FULL_OLD_1S 9/100, GRIPPER_EMA_1S 10/100, GRIPPER_HOLD 5/100. The preregistered classification is **NO_SIGNAL**. `FO_1S - NEWEST` is exactly 0 pp (2 wins, 2 losses, 96 ties; task-cluster interval −3 to +3 pp). Follow-up status says component-wise temporal aggregation is not justified. Reinterpretation is constrained by the frozen gate and exploratory scope.

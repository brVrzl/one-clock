# Close literature neighbors

Search date: 2026-08-24. This is a compact novelty screen, not a systematic review.

## StateTrack / progress-tracking screen

| Method | Timing / state mechanism | Distinction from StateTrack |
|---|---|---|
| [FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking](https://arxiv.org/abs/2607.24008) (2026) | Predicts execution-time visual and proprioceptive context so a frozen VLA can generate an asynchronous chunk aligned to its future state | Changes the policy-conditioning context and predicts a new chunk; StateTrack would only index rows inside an already predicted chunk and keeps the query schedule fixed. |
| [LeRobot RTC documentation](https://github.com/huggingface/lerobot/blob/main/docs/source/rtc.mdx) | Asynchronous chunk production with prefix guidance / blending at chunk boundaries | Acts on inter-chunk continuity and is designed for high-latency generative policies; it is not measured-state progress indexing. |
| [LeRobot action representations](https://github.com/huggingface/lerobot/blob/main/docs/source/action_representations.mdx) | Defines absolute, relative, and delta actions; relative actions share the query-time reference state, and RTC re-anchors leftovers for supported policies | This audit verified LIBERO ACT is instead sent through the environment's relative OSC controller; StateTrack reconstructs nominal EEF targets from that controller, not by re-anchoring the policy output. |
| [TempoWAM / Rethink Before You Execute](https://arxiv.org/abs/2608.09492) (2026) | Monitors task progress and decides whether a world-action-model chunk should continue or replan | Closest conceptual progress monitor found, but it changes the replan decision and uses a trained recurrent monitor; StateTrack tests row indexing without changing policy queries. |

The search found no prior method whose defining operation is a training-free,
monotonic nearest-state index *inside* a deterministic frozen ACT chunk while
leaving both numerical actions and the policy-query schedule untouched. This is
only a novelty-risk observation, not a novelty claim. The negative LIBERO gate
means this distinction is not currently sufficient for a paper.

## Requested execution and correction taxonomy

| Method | What is changed | Mechanism / supervision | Relation to sparse gripper-event realignment |
|---|---|---|---|
| [PACE: Phase-Aware Chunk Execution](https://arxiv.org/abs/2606.00537) (2026) | Whole-chunk execution horizon / replanning boundary | Training-free speed-profile transitions in the predicted chunk | Keeps action values intact but changes how much of the entire chunk is executed. EventAlign instead holds arm execution fixed and shifts only the gripper sequence. |
| [RACE: Time Optimal Execution of Action Chunk Policies Beyond Demonstration Speed](https://openreview.net/forum?id=INsLvSCJ4z) (ICLR 2026) | Timing of the whole desired-state trajectory | Desired-state imitation, time-optimal path parameterization, and test-time aligned-chunk search | Re-times all robot motion for speed and reachability and changes the policy target. EventAlign is a fixed-policy, single-channel discrete timing intervention. |
| [REMAC: Real-Time Robot Execution with Masked Action Chunking](https://openreview.net/forum?id=r0RGJ1j9on) (ICLR 2026) | Corrective adjustments across action chunks | Masked policy adaptation plus prefix-preserved sampling for asynchronous mismatch | Learns whole-chunk corrections and targets inference-latency inconsistency. EventAlign applies no continuous arm correction and needs no generative resampling. |
| [BID: Bidirectional Decoding](https://openreview.net/forum?id=qZmn2hkuzw) (ICLR 2025) | Selects among complete sampled chunks | Multiple generative samples ranked by backward coherence and forward contrast | Closest candidate-selection precedent, but its candidates vary the complete trajectory. EventAlign's five candidates share exactly the same arm trajectory and differ only in gripper-event timing. |
| [ORPA: Online Residual Policy Adaptation](https://arxiv.org/abs/2608.17323) (2026) | Continuous joint-space residual over frozen ACT | Human-feedback-conditioned residual module | The closest frozen-ACT correction neighbor, but it uses feedback and continuous residuals. EventAlign uses a sparse discrete timing vocabulary with no arm residual. |
| [RETAF: Reactive Tactile Adaptation of Force](https://arxiv.org/abs/2602.10013) (2026) | High-frequency gripper force, decoupled from arm pose and base open/close | Wrist vision, tactile feedback, force demonstrations, and a force-controlled gripper | Shares the principle of gripper-specific adaptation while preserving the base arm policy. It regulates continuous contact force with new sensing/hardware, whereas EventAlign only shifts open/close events and uses the existing deterministic ACT interface. |

The categorical distinction is therefore:

- PACE and RACE alter whole-trajectory execution timing.
- REMAC and ORPA learn continuous corrections.
- BID resamples and reranks whole action chunks.
- RETAF is gripper-specific, but adds tactile continuous-force control.
- EventAlign tested sparse discrete gripper-event shifts while preserving all arm values from a frozen deterministic ACT policy.

This distinction would be methodologically real, but the present causal sweep kills EventAlign empirically: its per-state oracle improves only 15/30 to 18/30 successes across the three tasks. It must not be presented as a contribution on the strength of taxonomy alone.

- [ORPA: Online Residual Policy Adaptation for Robot Manipulation Control with Human Feedback](https://arxiv.org/abs/2608.17323) (2026): the closest direct neighbor. It adds feedback-conditioned joint-space residuals to a frozen ACT policy and trains from human correction signals. A no-feedback offline residual would need a clear empirical advantage or different supervision to be defensible.
- [VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon](https://arxiv.org/abs/2607.01804) (2026): detects visual-dynamics deviation and responds by truncating a stale chunk and gradient-guided replanning. It repairs execution through visual monitoring and re-querying, not a single deterministic post-policy correction.
- [Real-Time Robot Execution with Masked Action Chunking (REMAC)](https://arxiv.org/abs/2601.20130) (ICLR 2026): learns masked corrective adjustments for asynchronous inference and targets intra-chunk mismatch plus inter-chunk continuity. It is training-time policy adaptation for latency mismatch, rather than a small external module trained on aligned ACT errors.
- [Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling](https://openreview.net/forum?id=qZmn2hkuzw) (ICLR 2025): samples multiple chunks and selects using backward coherence and forward contrast. It requires a generative policy and multiple samples; a deterministic ACT timing-vocabulary selector would differ in candidate construction and cost.
- [Test-Time Scaling for World Action Models via Zero-Shot Geometric Evaluation](https://arxiv.org/abs/2607.17454) (2026): ranks sampled action/future rollouts with geometric consistency and selectively invokes extra sampling. It is a high-compute world-model selector, whereas the current pivot considers five deterministic gripper-timing variants.
- [Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling](https://computationalrobotics.seas.harvard.edu/GPC/) (2025): uses an action-conditioned world model to rank and refine diffusion-policy candidates. This is the closest conceptual baseline for candidate scoring, but it needs a predictive world model and random-exploration data.
- [From Imitation to Refinement: Residual RL for Precise Assembly](https://arxiv.org/abs/2407.16677) (2024/2025 versions): learns residual actions over a pretrained chunked policy with RL and privileged/current state. It establishes that action-space residual adaptation is not novel by itself.
- [SEAM: Smooth Execution of Action-Chunked Motion for Vision-Language-Action Policies](https://arxiv.org/abs/2607.04609) (2026): training-free steering of flow-matching VLAs using the previous unexecuted tail. It targets chunk-boundary smoothness; our EMA baseline shows that smoothness alone can reduce success for deterministic ACT.

## Novelty implication before the causal sweep

The original statement “lightweight residual repair for frozen ACT” is too close to ORPA and residual-policy literature, and this sprint found no closed-loop gain. The only surviving nearby hypothesis is narrower:

> For deterministic action-chunk policies, predict a discrete gripper-timing correction from the current state and nominal chunk, then rerank a tiny structured candidate vocabulary without policy resampling or a world model.

That statement was tested directly by the paired closed-loop timing oracle. The small and inconsistent oracle gap is insufficient, so no selector or novelty claim is pursued.

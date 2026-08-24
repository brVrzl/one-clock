# Close literature neighbors

Search date: 2026-08-24. This is a compact novelty screen, not a systematic review.

- [ORPA: Online Residual Policy Adaptation for Robot Manipulation Control with Human Feedback](https://arxiv.org/abs/2608.17323) (2026): the closest direct neighbor. It adds feedback-conditioned joint-space residuals to a frozen ACT policy and trains from human correction signals. A no-feedback offline residual would need a clear empirical advantage or different supervision to be defensible.
- [VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon](https://arxiv.org/abs/2607.01804) (2026): detects visual-dynamics deviation and responds by truncating a stale chunk and gradient-guided replanning. It repairs execution through visual monitoring and re-querying, not a single deterministic post-policy correction.
- [Real-Time Robot Execution with Masked Action Chunking (REMAC)](https://arxiv.org/abs/2601.20130) (ICLR 2026): learns masked corrective adjustments for asynchronous inference and targets intra-chunk mismatch plus inter-chunk continuity. It is training-time policy adaptation for latency mismatch, rather than a small external module trained on aligned ACT errors.
- [Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling](https://openreview.net/forum?id=qZmn2hkuzw) (ICLR 2025): samples multiple chunks and selects using backward coherence and forward contrast. It requires a generative policy and multiple samples; a deterministic ACT timing-vocabulary selector would differ in candidate construction and cost.
- [Test-Time Scaling for World Action Models via Zero-Shot Geometric Evaluation](https://arxiv.org/abs/2607.17454) (2026): ranks sampled action/future rollouts with geometric consistency and selectively invokes extra sampling. It is a high-compute world-model selector, whereas the current pivot considers five deterministic gripper-timing variants.
- [Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling](https://computationalrobotics.seas.harvard.edu/GPC/) (2025): uses an action-conditioned world model to rank and refine diffusion-policy candidates. This is the closest conceptual baseline for candidate scoring, but it needs a predictive world model and random-exploration data.
- [From Imitation to Refinement: Residual RL for Precise Assembly](https://arxiv.org/abs/2407.16677) (2024/2025 versions): learns residual actions over a pretrained chunked policy with RL and privileged/current state. It establishes that action-space residual adaptation is not novel by itself.
- [SEAM: Smooth Execution of Action-Chunked Motion for Vision-Language-Action Policies](https://arxiv.org/abs/2607.04609) (2026): training-free steering of flow-matching VLAs using the previous unexecuted tail. It targets chunk-boundary smoothness; our EMA baseline shows that smoothness alone can reduce success for deterministic ACT.

## Novelty implication

The original statement “lightweight residual repair for frozen ACT” is too close to ORPA and residual-policy literature, and this sprint found no closed-loop gain. The only surviving nearby hypothesis is narrower:

> For deterministic action-chunk policies, predict a discrete gripper-timing correction from the current state and nominal chunk, then rerank a tiny structured candidate vocabulary without policy resampling or a world model.

That statement is not yet a contribution. It requires closed-loop improvement and a broader search specifically for discrete gripper-timing correction before any novelty claim.

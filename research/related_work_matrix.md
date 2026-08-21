# Related Work Matrix

Audit cutoff: 2026-08-21. `Reported rollout gain` means a gain claimed by the
paper, not independently replicated here. `Frozen` refers to the base policy.

| Paper | Date / status | Backbone | Chunking / prediction horizon | Execution horizon and adaptivity | Learned scheduler / uncertainty | Temporal ensemble or multiple experts | Group-specific execution | Base frozen? | Direct action-error objective? | Reported rollout gain? | Closest overlap with one-clock | Remaining gap relevant here |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [ACT](https://roboticsproceedings.org/rss19/p016.html) | 2023; RSS | CVAE Transformer | Trains fixed-length action chunks | Naive fixed query/execution or query every step for TE | No scheduler; fixed exponential decay | Yes: overlapping source-time predictions are averaged | No | N/A | L1 imitation loss for chunk | Yes | Exact source of the project's ACT semantics and temporal experts | No contextual selection; no group routing |
| [CogACT](https://arxiv.org/abs/2411.19650) | 2024 preprint | VLM + diffusion action transformer | Fixed predicted chunk | Ensemble at action time rather than a prefix scheduler | Similarity heuristic, not learned scheduler | Yes: cosine-weighted prior/current predictions | No; one weight per full action | N/A | Diffusion training, not router action MSE | Yes | Very close non-neural temporal ensemble baseline | No independent group choice or causal freshness estimate |
| [RTC](https://proceedings.neurips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html) | 2025; NeurIPS | Diffusion/flow VLA | Fixed native chunk | Asynchronous; commitment governed by latency/system pipeline | No learned scheduler; inpainting constraints | Reuses committed prefix/tail, not expert routing | No | Yes | No | Yes | Mandatory real-time/stale-chunk baseline | Does not decide group-specific source age |
| [SGAC](https://proceedings.neurips.cc/paper_files/paper/2025/hash/79ce24f9e8d3c4ff5919240eac78a782-Abstract-Conference.html) | 2025; NeurIPS | Generative BC / diffusion | Fixed prediction chunk | Adaptive scalar update/commitment | Action-similarity criterion | Reconciles old and new chunks | No | Method augments inference/training | No | Yes | Early peer-reviewed adaptive chunking collision | No group-specific routing |
| [A2C2](https://arxiv.org/abs/2509.23224) | 2025 preprint | Off-the-shelf VLA + residual head | Base chunk unchanged | Corrects every executed action using latest observation | Learned residual, not horizon scheduler | One stale base action plus fresh correction | No | Yes | Supervised residual action target | Yes | Strong alternative to re-querying/routing: correct stale action | Adds a model and needs per-step observation features |
| [TAS](https://arxiv.org/abs/2511.04421) | 2025 preprint, v2 2026 | Diverse chunk policies + selector | Base chunk unchanged | Selects one action from cached source times | Learned PPO selector | Explicit cached temporal candidates | No; full-action source selected | Yes | No; online reward | Yes | **Closest collision to Temporal Expert Routing** | No group-specific choice; its benefit must be reproduced against ACT/CogACT |
| [MoH](https://arxiv.org/abs/2511.19433) | 2025; ICML 2026 | pi0/pi0.5/regression action transformer | Explicit parallel short/long horizon segments | Cross-horizon consensus supports adaptive scalar inference | Learned linear gate | Explicit horizon experts | No independent actuator groups found | No; trains action module | Yes, policy training objective | Yes | Closest collision to explicit Mixture of Horizons family | No cached source-age experts or group consistency analysis |
| [REMAC](https://arxiv.org/abs/2601.20130) | 2026; ICLR | Flow/diffusion VLA | Masked chunk training | Asynchronous continuation | Learned masked-conditioning policy | Old prefix conditions new suffix | No | No; retraining required | Generative policy objective | Yes | Strong stale-chunk correction/continuation baseline | Different training regime; no group-age choice |
| [AutoHorizon](https://arxiv.org/abs/2602.21445) | 2026; ECCV | Flow VLA | Native fixed prediction horizon | Per-query scalar adaptive execution prefix | Self-attention proxy; training-free | No | No | Yes | No | Yes | Direct dynamic execution horizon collision | Proxy is architecture-specific; no group routing |
| [AAC](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.html) | 2026; CVPR | VLA; multiple sampled chunks | Native fixed chunk | Entropy selects one scalar prefix | Sample entropy; training-free | Multiple stochastic chunks, aggregated | Measures components but executes one scalar horizon | Yes | No | Yes | Direct training-free adaptive-horizon baseline | Component uncertainty is not independently executed |
| [A3](https://arxiv.org/abs/2605.11567) | 2026 preprint | pi0/pi0.5/GR00T | Native fixed chunk | Longest verified scalar prefix | Sampling + conditional-invariance verification | Multiple stochastic chunks | No | Yes | No | Yes | Strong consensus/verification baseline | Extra sampling/re-decoding; no group-age routing |
| [PACE](https://arxiv.org/abs/2606.00537) | 2026 preprint | Chunked robot policies | Native fixed chunk | Low-speed boundary chooses scalar prefix | Kinematic heuristic; training-free | No | Arm profile may pool arms; one global prefix | Yes | No | Yes | Direct phase/smoothness baseline | Action-space and calibration dependence; no group execution |
| [DVAC](https://arxiv.org/abs/2606.03847) | 2026 preprint | Flow policies | Native fixed chunk | Low-variance prefix, scalar | Denoising variance + rolling calibration | Multiple denoising states, not source ages | No | Yes | No | Yes | Strong uncertainty-driven adaptive-prefix baseline | Flow-only internal signal; no group scheduling |
| [DEHP](https://arxiv.org/abs/2606.11408) | 2026 preprint | Frozen chunk policy + head | Native fixed chunk | Categorical scalar horizon | Online-RL learned head | No | No | Yes | No; trajectory reward | Yes | Direct learned dynamic-horizon baseline | No group-specific action source; online training cost |
| [SEAM](https://arxiv.org/abs/2607.04609) | 2026 preprint | Flow VLA | Native fixed chunk | Fixed execution; improves boundary consistency | Analytic tail-guided correction | Previous tail is a reference | No | Yes | No | Preserves success, improves jerk | Mandatory smoothness/consistency baseline if mixing chunks | Does not decide when to re-query |
| [BCP](https://arxiv.org/abs/2608.03483) | 2026 preprint | LingBot-VLA/pi0.5 + 16.4M head | Native fixed chunk | Sequential continue/replan decisions | Online-RL ordinal continuation head | No | No | Yes | No; trajectory success/efficiency | Yes | Strongest learned value-of-requerying collision | No group-specific source selection; substantial online RL |
| [Why Action Chunking Works](https://arxiv.org/abs/2608.02547) | 2026 preprint | BC policies | Experimental manipulation of chunk/delay | Analysis, not scheduler | No | Identifies implicit temporal ensemble | No | N/A | BC losses | Yes, mechanistic comparisons | Directly supports temporal-expert interpretation but threatens novelty | Does not solve contextual or group-consistent routing |

## Definitions enforced in this matrix

- **Prediction horizon:** number/range of future actions produced or modeled by a
  policy query.
- **Execution horizon:** number of those actions committed before a new decision.
- **Query interval:** physical control steps between policy forward passes.
- **Temporal age:** current action time minus the observation/source time that
  generated a candidate prediction.

These quantities coincide only in special execution schemes. The project must
report all four separately.

## Answer to the novelty question

**No. “Dynamic execution horizon” alone is not a defensible ICRA 2027 novelty.**
It is directly occupied by peer-reviewed SGAC, AAC, and MoH and by AutoHorizon,
A3, PACE, DVAC, DEHP, and BCP. Rebranding horizon selection as reliability,
phase awareness, confidence, freshness, or continuation does not create a new
technical contribution.

A potentially distinguishable question remains: whether independently choosing
the source age of actuator groups yields control benefit beyond full-action ACT,
CogACT, and TAS, while enforcing physical consistency. That question is not yet
supported by the project's closed-loop evidence and must pass the dense oracle
and direct compositional gates before implementation.


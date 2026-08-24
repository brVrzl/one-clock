# Fast RTX 5080 execution-signal literature update

Search date: 2026-08-24 (Asia/Shanghai). Scope: a bounded update of the named
original papers, official proceedings pages, project pages, and released code.
This is not a systematic review and does not establish universal novelty.

## Practical source map

| Work | Original source inspected | Execution signal or mechanism | Consequence for the fast track |
|---|---|---|---|
| ACT | [RSS 2023 paper](https://roboticsproceedings.org/rss19/p016.html), [official code](https://github.com/tonyzhaozh/act) | Predicts action chunks; its released temporal ensemble combines overlapping source-time predictions with a fixed exponential rule. | Action chunking, temporal ensembling, and age weighting are prior art and baselines, not contributions. |
| CogACT | [paper](https://arxiv.org/abs/2411.19650), [official code](https://github.com/microsoft/CogACT) | Adaptive Action Ensemble weights overlapping complete actions by cosine similarity to the newest prediction; released deployment uses `alpha=0.1`. | Similarity-weighted temporal ensembling and caching overlapping predictions are occupied. |
| AutoHorizon | [paper](https://arxiv.org/abs/2602.21445), [project/code](https://hatchetproject.github.io/autohorizon/) | Uses flow-VLA action-token self-attention as a training-free proxy for one scalar execution horizon. | Generic model-internal horizon prediction and non-monotonic horizon curves are occupied. |
| AAC | [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.html), [project/code](https://lance-lot.github.io/adaptive-chunking.github.io/) | Samples multiple chunks, computes translation, rotation, and gripper uncertainty, then aggregates them into one synchronized prefix length. | Component-aware uncertainty is prior art; the current design preserves one shared 7-D source-weight vector and does not claim component uncertainty as novel. |
| PACE | [paper](https://arxiv.org/abs/2606.00537) | Finds low-speed valleys in predicted arm-motion profiles and uses accepted transitions as training-free replanning boundaries. | Kinematic transitions and phase-dependent horizons are a mandatory simple baseline/fallback, not the headline idea. |
| DEHP | [paper](https://arxiv.org/abs/2606.11408), [project](https://dehp-chunking.github.io/) | Adds an online-RL-trained categorical horizon branch while freezing the base chunk policy. | A learned horizon head is occupied and outside the fast method budget. |
| TAS | [paper](https://arxiv.org/abs/2511.04421) | Caches candidates predicted at different timesteps and learns a PPO selector for one complete current action, with a coherence objective. | Generic learned cached-source selection is occupied; no learned selector will be pursued. |
| When to Trust Imagination | [paper](https://arxiv.org/abs/2605.06222) | Verifies a world-action model's predicted visual future against real observations using a learned causal-attention verifier; adaptive chunk length follows from future/reality consistency. | Generic temporal trust and learned future-consistency replanning are occupied; this is normal prior work, not concurrent work. |
| TempoWAM / Rethink Before You Execute | [paper](https://arxiv.org/abs/2608.09492) | A recurrent progress monitor and online calibration decide whether a world-action-model chunk is still advancing the task. | Progress-based keep/replan decisions further crowd generic adaptive execution; they do not answer the frozen-ACT offline-to-control transfer question. |
| Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control? | [paper](https://arxiv.org/abs/2608.02547) | Controlled experiments attribute benefits to non-Markovian expressivity, reduced compounding error, and implicit ensembling over delayed temporal relationships, rather than the tested temporal-consistency explanation alone. | “Temporal experts” or implicit temporal ensembling is not a novelty claim. The result strengthens the need to separate teacher-forced action metrics from closed-loop utility. |
| Revisiting Open-Loop Execution in Robotics | [paper](https://arxiv.org/abs/2608.15938) | Studies why open-loop execution helps short-context policies, emphasizing expert non-Markovianity; reports that sufficiently long context can favor maximally reactive execution. | Treat as very recent concurrent work (first posted 2026-08-16). Acknowledge briefly; do not redesign experiments or make it a central baseline. |

## Locked positioning and test

The occupied space includes horizon-dependent success, non-monotonic and
phase-dependent horizons, generic temporal trust, temporal ensembling,
similarity- or recency-weighted aggregation, cached temporal candidates, and
generic uncertainty-based replanning. Kinematics-adaptive recency is therefore
an intentionally small empirical probe, not a broad novelty claim.

The candidate question is narrower:

> Under one frozen ACT policy and an episode-safe split, do deployment-available
> offline or model-internal execution signals select temporal aggregation rules
> that improve held-out teacher-forced action error, and does that ranking
> transfer to closed-loop task success?

The adaptive-recency test keeps one scalar source-weight vector for the full
7-D action. Validation selects among only fixed recency, PACE-transition,
disagreement, and their fused two-level rule. The test split is opened once
after freezing. If no adaptive rule clearly and stably beats fixed
`beta=0.03`, the result is `KARE-OFFLINE-NULL` and the project immediately
moves to a faithful PACE-style baseline plus one predefined gripper-event
boundary extension.

## Claim boundary

Teacher-forced error, smoothness, agreement, attention, entropy, kinematics,
or future/reality consistency may be useful signals, but none alone establishes
closed-loop execution utility. Until matched rollouts are available, the only
supported statement is an offline ranking under the frozen checkpoint and
dataset. The very recent Zeng et al. preprint is concurrent mechanistic work;
its exact final claims must be rechecked before final Related Work wording.

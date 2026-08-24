# Final novelty audit — 2026-08-24

## Scope and conclusion

This was a bounded verification of public primary records available through
2026-08-24. It re-checked the execution-time action-chunking papers named in
the manuscript plan and searched directly for component-specific reuse of
observation-time generations. The search did not establish priority and is not
a basis for a "first" claim.

Bounded manuscript wording supported by this audit:

> We are not aware of prior controlled evaluations that independently assign
> temporal source generations to heterogeneous components of a jointly
> predicted action chunk.

This statement is narrower than claiming that no prior work specializes action
components, reuses old chunks, routes temporal information, or makes an
adaptive execution decision.

## Search log

Search date: 2026-08-24 (Asia/Shanghai). Sources were checked on official
publisher pages, project pages, or arXiv records and full text.

Queries:

- `ACT temporal ensemble action chunk RSS 2023`
- `CogACT adaptive action ensemble action chunk`
- `Adaptive Action Chunking inference time translation rotation gripper entropy`
- `Temporal Action Selection action chunk cached candidate actions`
- `Real-Time Execution Action Chunking Flow Policies RTC`
- `REMAC masked action chunking`
- `A2C2 real-time correction VLA action chunks`
- `SEAM action dimensions overlap guidance`
- `AutoHorizon adaptive execution horizon robot`
- `PACE phase-aware chunk execution arm speed profile`
- `Why Does Action Chunking Improve Behavioral Cloning delayed policies`
- `TempoWAM adaptive execution world action model`
- `When to Trust Imagination adaptive action execution`
- `HiPolicy multi-frequency action chunking`
- `TRACT chronological phase authority action chunks`
- `DAM-VLA arm gripper specialized action models`
- `robot action chunk "component-specific temporal source"`
- `robot action chunk "arm gripper temporal source"`
- `robot "asynchronous component action chunk"`
- `robot "component-wise temporal ensemble"`
- `robot action chunk "different source observation" action dimensions`
- `robot action chunk "mixed-generation action components"`
- `robot action chunk component temporal generations source observation arm gripper`
- `robot temporal ensemble per-dimension action source age`

The six quoted component-specific formulations and their two broader variants
returned no exact controlled counterpart. One additional adjacent paper was
found: ARP mixes action types with different chunk sizes in a learned
autoregressive representation, but does not mix observation-time source
generations at execution.

## Paper-by-paper distinctions

| Work | What the primary record establishes | Exact distinction from this paper |
|---|---|---|
| ACT (Zhao et al., RSS 2023) | Overlapping predictions for the same physical time can be exponentially averaged. | The temporal weights apply to the complete action vector; ACT does not independently assign source age to arm and gripper. |
| CogACT (Li et al., arXiv:2411.19650) | Adaptive Action Ensemble changes shared weights over historical complete-action predictions using action similarity. | A single weight vector is used for the full action; our intervention assigns different source generations to components under a matched query cadence. |
| AAC (Liang et al., CVPR 2026) | Translation, rotation, and gripper uncertainty signals are computed separately and combined to select an execution prefix. | Component-aware evidence is aggregated into one chunk-level prefix rather than used to select different observation-time generations for components. |
| TAS (Weng et al., arXiv:2511.04421) | A learned selector chooses among cached candidate actions originating from different temporal contexts. | The selected candidate is a complete action. TAS does not perform a controlled arm/gripper source-age factorial. |
| RTC (Black et al., NeurIPS 2025) | During asynchronous inference, actions guaranteed to execute are frozen and the remainder of a flow-policy chunk is inpainted. | RTC handles latency and chunk-prefix commitment; it does not independently select source generations by action component. |
| REMAC (Wang et al., arXiv:2601.20130) | A learned masked-action correction improves asynchronous execution and prefix-preserved sampling. | REMAC changes the policy/correction process and preserves a shared prefix; it does not compose one frozen action from component-specific observation times. |
| A2C2 (Sendai et al., arXiv:2509.23224) | A lightweight head uses the newest observation to add a per-step correction to a base VLA action. | It learns a residual correction rather than selecting arm and gripper from distinct cached generations. |
| SEAM (Zhan et al., arXiv:2607.04609) | The previous chunk tail guides the overlap of a newly generated flow-policy chunk; guidance may be restricted to a subset of dimensions. | Dimension-selective correction is the closest technical case, but the method steers a new chunk toward a tail. It does not execute heterogeneous components directly from different observation-time source generations or evaluate the corresponding 2x2 factorial. |
| AutoHorizon (Wang et al., arXiv:2602.21445) | Attention structure is used to estimate one execution horizon for each predicted chunk. | It changes when the full action chunk is refreshed; our query frequency is fixed and only component source assignment changes. |
| PACE (Nie et al., arXiv:2606.00537) | Arm-specific kinematic profiles propose phase boundaries, which are pooled to choose one execution horizon. | Component-specific signals still produce a shared re-query/prefix decision. PACE changes observation frequency, while our design queries every controller step. |
| Lazzati et al. (arXiv:2608.02547) | Delayed policies and implicit ensembles explain benefits of older observation-conditioned predictions in audited settings. | The analysis motivates utility of older sources, but does not test whether different components of the same action prefer different delays in closed loop. |
| TempoWAM (Ye et al., arXiv:2608.09492) | A progress monitor decides whether a world-action-model chunk should continue or be replanned. | The decision applies to the complete remaining action chunk, not separately to arm and gripper sources. |
| When to Trust Imagination (Wang et al., arXiv:2605.06222) | A future-reality verifier adapts action chunk size from prediction-observation agreement. | It returns a shared execution length and changes replanning timing. |
| HiPolicy (Zhang et al., arXiv:2604.06067) | A learned policy predicts action sequences at multiple frequencies and selects frequency using uncertainty. | This is frequency adaptation inside a learned architecture, not component-specific reuse of observation-time generations at fixed 20 Hz. |
| TRACT (Liu et al., arXiv:2607.29285) | Current/next procedural-phase query paths route future positions within a newly predicted chunk, with an arm-specific response-deficit correction. | TRACT routes future chunk semantics and corrects arm execution. It does not assign arm and gripper of one executed action to different past observation-time generations from a frozen jointly predicted chunk. |
| DAM-VLA (Peng et al., arXiv:2603.00926; ICRA 2026) | Specialized diffusion action models, routing, and dual-scale weighting coordinate arm movement and gripper manipulation. | DAM-VLA already establishes architectural arm/gripper specialization, so component heterogeneity is not our novelty. Our result concerns temporal-source assignment for components of one frozen jointly predicted action. |
| ARP (Zhang et al., RA-L 2025; arXiv:2410.03132) | A learned autoregressive policy mixes action types and uses different chunk sizes for those types. | It changes action representation and generation; it does not independently reuse observation-time generations across components at execution. |

## Novelty boundary carried into the manuscript

The manuscript may claim a controlled intervention and a confirmed result in
the audited ACT/LIBERO Object system. It must not claim novelty for action
chunking, temporal ensembling, adaptive execution, arm/gripper specialization,
dimension-selective correction, or the general usefulness of older
predictions. It must also avoid a universal arm-fresh/gripper-old rule.

Closest collision set for reviewer comparison:

1. **TAS:** closest direct selection among temporally distinct cached actions,
   but it selects a complete action.
2. **TRACT:** closest combination of temporal routing and arm-specific
   execution treatment, but its temporal variable is future procedural phase,
   not observation-time source generation.
3. **DAM-VLA:** closest explicit arm/gripper specialization, but the
   specialization is architectural rather than a frozen executor's source-age
   assignment.

## Primary records inspected

- ACT: <https://roboticsproceedings.org/rss19/p016.html>
- CogACT: <https://arxiv.org/abs/2411.19650>
- AAC: <https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.html>
- TAS: <https://arxiv.org/abs/2511.04421>
- RTC: <https://arxiv.org/abs/2506.07339>
- REMAC: <https://arxiv.org/abs/2601.20130>
- A2C2: <https://arxiv.org/abs/2509.23224>
- SEAM: <https://arxiv.org/abs/2607.04609>
- AutoHorizon: <https://arxiv.org/abs/2602.21445>
- PACE: <https://arxiv.org/abs/2606.00537>
- Lazzati et al.: <https://arxiv.org/abs/2608.02547>
- TempoWAM: <https://arxiv.org/abs/2608.09492>
- When to Trust Imagination: <https://arxiv.org/abs/2605.06222>
- HiPolicy: <https://arxiv.org/abs/2604.06067>
- TRACT: <https://arxiv.org/abs/2607.29285>
- DAM-VLA: <https://arxiv.org/abs/2603.00926>
- ARP: <https://arxiv.org/abs/2410.03132>

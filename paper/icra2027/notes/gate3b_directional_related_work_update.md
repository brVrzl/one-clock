# Bounded related-work update for the Gate-3B directional story

Status: primary-paper recheck completed 2026-08-24. This note updates only the
closest method boundary needed for the post-Gate-3B manuscript. It is not an
open-ended or systematic review.

## Question and search boundary

The review asked one operational question:

> Does a closest published method explicitly execute heterogeneous action
> components, such as arm and gripper, from different source-observation
> generations?

The bounded set was ACT, CogACT, AAC, TAS, RTC, REMAC, A2C2, SEAM, Lazzati et
al., AutoHorizon, and PACE. The papers and available official code/project
sources were inspected for their executed temporal decision, not merely for
component-aware features or losses.

## Closest methods

| Work | Temporal object used at execution | Component treatment | Independent component-source assignment located? |
|---|---|---|---|
| [ACT](https://arxiv.org/abs/2304.13705) | Exponential ensemble over overlapping predictions for one physical action | One age weight applies to the complete joint action | No |
| [CogACT](https://arxiv.org/abs/2411.19650) | Similarity-weighted present and historical predictions | One scalar similarity weight per complete action | No |
| [AAC](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.html) | One selected chunk prefix | Translation, rotation, and gripper uncertainties are computed separately, then aggregated | No |
| [TAS](https://arxiv.org/abs/2511.04421) | One complete action selected from cached source-time candidates | Learned selection remains at full-action level | No |
| [RTC](https://proceedings.neurips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html) | Inpainted full chunk around actions committed during inference latency | Full-action asynchronous correction | No |
| [REMAC](https://arxiv.org/abs/2601.20130) | Masked or prefix-conditioned regenerated chunk | Full-action chunk correction | No |
| [A2C2](https://arxiv.org/abs/2509.23224) | Per-step residual correction using the current observation | Corrects the base action rather than assigning group-specific source generations | No |
| [SEAM](https://arxiv.org/abs/2607.04609) | Flow correction aligned to the previous unexecuted overlap | Guidance can be restricted to a dimension subset, but the method does not form a controlled arm/gripper source-generation factorial | No exact match |
| [Lazzati et al.](https://arxiv.org/abs/2608.02547) | Delayed single-action predictions and explicit or implicit temporal ensembles | Studies non-Markovian demonstrations and temporal prediction reuse | No |
| [AutoHorizon](https://arxiv.org/abs/2602.21445) | One attention-derived execution horizon | Global chunk prefix | No |
| [PACE](https://arxiv.org/abs/2606.00537) | One kinematic transition boundary and execution prefix | Arm trajectory cues determine one shared execution boundary | No |

SEAM is the closest nuance because its correction can use only a subset of
dimensions. That operation should be acknowledged if dimensional guidance is
discussed. It is not the same as deterministically assigning arm and gripper
from two different source observations while balancing the four source
margins.

## Lazzati boundary

Lazzati et al. support the proposition that older observation-conditioned
predictions can better match expert behavior. Their experiments attribute part
of action chunking's benefit to non-Markovian demonstration structure, reduced
compounding error, and implicit temporal ensembling in the tested settings.
The present candidate claim is not that older predictions can help.

The relevant distinction is narrower. Older predictions may have different
utility across heterogeneous action components, and teacher-forced delayed
prediction quality may not identify the temporal source that maximizes
closed-loop control. Gate-3B offers post-hoc evidence for that interpretation;
Gate-3C is the untouched-state test. This wording must remain conditional until
Gate-3C is complete.

## Permitted manuscript wording

The bounded review did not locate an exact controlled counterpart. The
manuscript may therefore use:

> We are not aware of prior controlled evaluations that independently assign
> temporal source generations to heterogeneous action components.

This sentence is deliberately scoped to controlled evaluations and independent
source-generation assignment. It must be rechecked before submission. Do not
replace it with “we are the first,” “prior work assumes one clock,” or a claim
that no component-aware execution method exists.

## Consequences for positioning

ACT and CogACT are full-action temporal-aggregation controls. AAC is the
mandatory example of a method that measures component uncertainty but returns
one shared temporal execution decision. TAS establishes that learned selection
among cached full-action generations is occupied. RTC, REMAC, A2C2, and SEAM
cover stale-chunk correction and consistency. AutoHorizon and PACE are relevant
only to distinguish global execution-boundary selection from component-source
assignment.

The paper should not claim novelty for older temporal predictions, temporal
ensembling, adaptive execution, stale-action correction, or component-aware
uncertainty. Its candidate empirical contribution is the controlled
component-source intervention and, only if Gate-3C confirms it, the asymmetric
fresh-arm/old-gripper result.

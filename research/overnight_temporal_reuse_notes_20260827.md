# Overnight temporal-reuse research notes

Date: 2026-08-27, Beijing time.  Branch: `exp/libero-component-temporal-reuse`.

## Frozen pilot status

The frozen 8-task × 10-condition × 10-initial-state cohort is complete and was
merged into `experiments/component_temporal_reuse/pilot_results.json`.  The
complete-pilot guard reports 80 unique task-condition blocks.  The CPU
analysis is under `experiments/component_temporal_reuse/final_analysis/`.

Initial read: full-old is 68/80 at ages 4 and 8, versus fresh 66/80, then falls
to 58/80 at age 16.  FO is 68/80, 65/80, 61/80 at ages 4/8/16.  Reverse is
65/80, 64/80, 50/80.  The task-macro FO-minus-reverse contrast is +0.138 at age
16, but the direction is not uniform: the two Goal tasks favor reverse at that
age.  This supports age-dependent utility and a possible component contrast,
but also shows strong task heterogeneity.  It does not justify a universal
component-specific claim or coefficient tuning on this cohort.

The minimum component-aware follow-up is justified as a fixed diagnostic,
because the age-16 FO/reverse difference is positive on four of eight tasks and
the pooled paired contrast is directionally large, while it remains exploratory
and is not a confirmatory method-selection result.

## Fixed follow-up protocol

After semantic smoke validation, the aggregation follow-up uses the same eight
tasks, seeds 1000–1009, native SmolVLA checkpoint, LIBERO runtime, one query per
environment step, and same-target candidate alignment.  Fixed methods are:

* fresh;
* installed LeRobot ACT temporal weights, `m=0.01`;
* physical source-age exponential weights, `beta=0.03`;
* CogACT cosine weights, `alpha=0.3`;
* component diagnostic: fresh arm plus ACT-aggregated gripper.

No coefficient sweep, task-specific tuning, learned selector, horizon change,
or frozen-cohort rerun is authorized.

## Literature and novelty audit

* [ACT, Zhao et al. (2023)](https://arxiv.org/abs/2304.13705) establishes
  action-sequence prediction and the action-chunking setup; temporal ensembling
  is part of the deployment lineage.
* [Lazzati et al. (2026)](https://arxiv.org/abs/2608.02547) is the closest
  conceptual collision.  It argues that delayed observation-conditioned
  predictions and implicit ensembling explain much of action chunking's gain,
  and explicitly studies same-target predictions from different temporal
  relationships.  A paper claim that “older predictions can help” therefore
  needs to be framed as a component-resolved, task-conditional measurement, not
  as a first observation of historical prediction utility.
* [CogACT](https://arxiv.org/abs/2411.19650) uses a full-action adaptive
  similarity-weighted ensemble to avoid averaging incompatible action modes.
  Our fixed cosine control is an implementation-level comparator, not a new
  full-action aggregation principle.
* [TAS](https://arxiv.org/abs/2511.04421) caches chunks and learns a selector;
  it is closer to a learned action-selection method than the present fixed
  operator comparison and is not an overnight training target.
* [AutoHorizon](https://arxiv.org/abs/2602.21445), [AAC](https://arxiv.org/abs/2604.04161),
  and [PACE](https://arxiv.org/abs/2606.00537) address adaptive execution
  horizon or chunk length.  They collide with horizon-adaptation claims but do
  not, by themselves, establish component-wise source assignment at a common
  physical target time.
* [REMAC](https://arxiv.org/abs/2601.20130) and related asynchronous-inference
  work concern inference delay, intra-chunk mismatch, or continuity.  They are
  relevant to deployment semantics but are distinct from this synchronous
  same-target aggregation comparison.

## RoboTwin gate

The official/current path is [RoboTwin-Platform/RoboTwin](https://github.com/robotwin-Platform/robotwin)
and its [official documentation](https://robotwin-platform.github.io/doc/),
with [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) as the benchmark paper.
The public protocol is a 50-task dual-arm benchmark using SAPIEN, with official
policy support including ACT.  The local worktree contains historical
RoboTwin-related bytecode and branch-consolidation references, but no clean
current standard checkout was identified in the active research tree.  No
RoboTwin job is started while the justified LIBERO aggregation follow-up is
using the research GPUs.

## Decision rule

Use task-level paired outcomes first.  A stable component result requires
directional support across multiple tasks/suites and an advantage over shared
full-action aggregation, not merely a pooled percentage.  If shared
aggregation explains the gain, stop component-specific development.  If only
fixed historical selection helps, perform CPU/offline source-utility analysis
before considering any selector.  If the pattern remains heterogeneous, keep
the paper framing centered on conditional temporal-source utility and report
the negative cases.

## Independent ACT confirmation

The predeclared matched-query minimum panel was launched on completed native
100k task-specific ACT checkpoints for Object task 6, Spatial task 2, and Goal
task 1. It uses fresh, FO16, full-old16, and reverse16, with one policy query
per environment step, explicit independent initial-state IDs 10--19, and
frozen checkpoint weights. The intervention is separate from native ACT
deployment, whose installed `n_action_steps=100` remains unchanged. The
runner's source-assignment semantic smoke test passed before rollout.

The confirmation completed with 40 paired episodes per condition after the
LIBERO-10 task-3 checkpoint became available:

| task | fresh | FO16 | full-old16 | reverse16 | FO16 minus reverse16 |
|---|---:|---:|---:|---:|---:|
| Object task 6 | 8/10 | 7/10 | 2/10 | 1/10 | +0.60 |
| Spatial task 2 | 10/10 | 10/10 | 10/10 | 4/10 | +0.60 |
| Goal task 1 | 10/10 | 9/10 | 10/10 | 9/10 | +0.00 |
| LIBERO-10 task 3 | 9/10 | 7/10 | 6/10 | 3/10 | +0.40 |
| all completed tasks | 37/40 | 33/40 | 28/40 | 17/40 | +0.40 |

The pooled paired FO16 versus reverse16 contrast is 14 candidate-only versus
2 reference-only successes (exact McNemar p=0.00040). This is cross-policy
directional evidence for component identity at age 16, but it is not evidence
that FO16 improves over fresh: FO16 is 33/40 versus fresh 37/40, and full-old
is 28/40. The result therefore supports a conditional component asymmetry
interpretation, while preserving the frozen pilot's non-monotonic and
task-dependent conclusions. No ACT intervention outcomes were used to add or
remove the task.

Artifacts: `experiments/component_temporal_reuse/act_confirmation/`.

## Final aggregation readout

The frozen eight-task aggregation follow-up produced Fresh 72/80, official ACT
temporal ensemble 66/80, physical age decay (`beta=0.03`) 71/80, CogACT-style
shared aggregation (`alpha=0.3`) 73/80, and component-aware aggregation 73/80.
All used one policy query per environment step. CogACT captures most of the
descriptive gain, while component-aware aggregation has no clear advantage over
CogACT on this cohort. The result does not support a universal component rule.

The completed standard native ACT baseline is Spatial 64/100, Object 48/100,
Goal 70/100, LIBERO-10 41/100, overall 223/400 (55.8%). Standard SmolVLA is
85%, 93%, 78%, and 42%, respectively (74.5% average). These native baselines
remain separate from the intervention studies.

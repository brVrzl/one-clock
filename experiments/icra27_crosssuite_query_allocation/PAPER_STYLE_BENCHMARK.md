# Paper style benchmark for `one-clock`

Date checked: 2026-09-02. This memo extracts communication patterns only. It does not authorize copying wording, artwork, color palettes, icons, or panel layouts.

## Reference set and status

| Reference | Status used here | Primary source |
|---|---|---|
| ACT, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* | Robotics: Science and Systems 2023 | [RSS proceedings](https://roboticsproceedings.org/rss19/p016.html) |
| *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* | Robotics: Science and Systems 2023 | [RSS proceedings](https://roboticsproceedings.org/rss19/p026.html) |
| BID, *Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling* | ICLR 2025 conference paper | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d78dd998f7b9ac79604d47a2d79bb0d-Abstract-Conference.html) |
| PACE, *Phase-Aware Chunk Execution for Robot Policies with Action Chunking* | arXiv preprint, v2 dated 2026-07-29; no archival venue claimed in the paper record | [arXiv](https://arxiv.org/abs/2606.00537) |
| *VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies* (AutoHorizon) | arXiv preprint, v2 dated 2026-06-20. The project page labels it ECCV 2026, but this memo does not treat that as an independently verified proceedings record. | [arXiv](https://arxiv.org/abs/2602.21445) |

## ACT

### 1. Abstract structure

- Opens with a concrete capability barrier: fine manipulation requires precision, contact coordination, and feedback.
- Introduces a cost/access constraint and turns it into a direct question.
- Presents the hardware/data system, then states the learning-specific failure mode.
- Introduces ACT as the response and ends with memorable absolute evidence: six real tasks, 80–90% success, and roughly ten minutes of demonstrations.
- The abstract carries about four claim types: problem importance, system accessibility, algorithmic contribution, and demonstrated capability. It uses one compact quantitative result rather than an audit trail.

### 2. Introduction structure

1. Grounds fine manipulation in a detailed task sequence and explains why millimeter-scale errors matter.
2. Contrasts expensive precision systems with an accessible learned alternative.
3. Separates the system into two needs: dexterous demonstration collection and an imitation learner robust to compounding error and non-stationarity.
4. Gives one paragraph to the teleoperation system.
5. Gives one paragraph to action chunking and temporal ensembling, including the mechanism each is intended to address.
6. Closes with a short synthesis of the hardware and learning contributions.

The central question appears immediately. Prior work is contrasted before the method detail. The main empirical promise appears in the abstract and first-page figure; the introduction focuses more on why the components are needed.

### 3. Results narrative

- Results progress from hardware/teleoperation validation to imitation-learning performance, then to ablations.
- The action-chunking ablation varies chunk size and makes a specific design choice interpretable rather than presenting only a winning number.
- Mechanism is modest: temporal aggregation is motivated through prediction smoothness and evaluated through ablations, not elevated into a causal theorem.
- Weak baselines remain in the complete comparison instead of being removed from the story.

### 4. Figure rhetoric

- Figure 1 is a miniature system story: hardware, operator interaction, and example learned behaviors. It answers “what capability does the complete system enable?” before technical notation appears.
- The method figures separate architecture from execution semantics. The action-chunking/temporal-ensemble figure uses a small number of visual objects to explain overlapping predictions.
- The chunk-size ablation uses a simple response curve whose x-axis is one true intervention family. It does not connect semantically different conditions as if they were a dose curve.
- Robot photographs are used when embodiment matters; plots are used for comparative evidence. Decorative visual material is limited.

### 5. Caption style

- Captions define what each side or panel contains, then state why the panel is relevant.
- Method captions explain the transformation represented by arrows or overlapping chunks.
- Result captions include task/metric context but leave dense numerical detail to the surrounding text or table.

### 6. Table style

- Tables carry absolute success by task and method, which is more auditable than a plot of only aggregate gains.
- Precision is coarse and appropriate to rollout counts.
- Best results receive restrained typographic emphasis; the table does not use decorative color.
- Ablations and headline comparisons are separated so the main table remains legible.

## Diffusion Policy

### 1. Abstract structure

- Introduces a new policy representation immediately.
- Places a broad quantitative benchmark result near the front: 12 tasks across four benchmarks and a reported average improvement.
- Explains three properties of the representation: multimodality, high-dimensional action generation, and stable training.
- Ends by naming the implementation ingredients needed for robot control: receding-horizon execution, visual conditioning, and a time-series diffusion architecture.
- It makes roughly five claims, but groups them under one representation thesis rather than listing experiments chronologically.

### 2. Introduction structure

1. Defines why robot action prediction differs from ordinary regression.
2. Organizes prior attempts by action representation.
3. Introduces diffusion as a representation-level answer.
4. Enumerates the properties that matter for robot policies.
5. Converts those properties into three robot-specific technical contributions.
6. States the breadth of the evaluation and the aggregate empirical result.

The main empirical evidence is introduced early, while experiment mechanics are deferred. Prior work is contrasted through a representation taxonomy rather than a long citation catalog.

### 3. Results narrative

- Evaluation breadth is established first across simulation and real-robot tasks.
- Comparative results are followed by analyses of design choices, robustness, control representation, and latency.
- Results are grouped by scientific property—multimodality, stability, precision, robustness—rather than by repository experiment names.
- Failure cases and behavior visualizations are used to clarify scope, not hidden in prose.

### 4. Figure rhetoric

- Figure 1 is a compact representation taxonomy leading from explicit and implicit policies to diffusion policies. It answers one conceptual question: what is different about the output representation?
- The overview figure separates observations, the denoising process, and the executed action horizon.
- Real-world sequence figures use a few key frames plus an aggregate image where motion structure matters.
- Consistent encoding is maintained across conceptual and empirical panels; each panel omits implementation details not needed for its question.

### 5. Caption style

- Conceptual captions define each representation and state its intended advantage.
- Rollout captions explain what the selected frames or averaged images reveal.
- Captions are self-contained enough to orient a reader, but equations and full experimental contracts stay in the text.

### 6. Table style

- Tables are used for cross-task benchmark breadth and architecture/control ablations.
- Means are reported at a precision commensurate with the evaluation, with consistent units.
- Highlighting is sparse and comparison-specific.
- The paper does not try to encode every benchmark and ablation in one plot.

## BID

### 1. Abstract structure

- Opens with the established action-chunking practice and conflicting empirical reports.
- States the consistency/reactivity trade-off as the analytical result.
- Introduces BID as a test-time response with two named criteria.
- Ends with breadth rather than a single cherry-picked task: two policy families, seven simulation benchmarks, and two real-world tasks.
- The abstract makes four linked claims: diagnosis, trade-off, method, and general empirical improvement.

### 2. Introduction structure

1. Defines action chunking and why it can both help and hurt.
2. Uses a first-page comparison figure to distinguish open-loop chunking, receding-horizon execution, and BID.
3. Poses the technical question of how chunking changes learner–expert divergence.
4. Gives the analytical consistency/reactivity interpretation.
5. Derives backward coherence and forward contrast as responses to two sides of the trade-off.
6. Summarizes empirical scope and contributions.

The paper’s research question precedes the algorithm. The method is positioned as a response to the analyzed trade-off, not merely as another decoder.

### 3. Results narrative

- Theory and controlled one-dimensional analysis establish the interpretive vocabulary before full robot benchmarks.
- Main comparisons evaluate multiple inference strategies; sampling-size and compatibility analyses follow.
- Real-robot results arrive after simulation evidence and are supported by stage-based rollout examples.
- Negative or non-monotonic horizon effects are part of the motivating evidence, not treated as inconvenient ablations.

### 4. Figure rhetoric

- Figure 1 compares inference strategies with a shared visual grammar and answers “where does BID sit between coherence and reactivity?”
- The expert/learner diagrams isolate the formal source of divergence before presenting the full method.
- The one-dimensional horizon plot provides an interpretable controlled case before high-dimensional tasks.
- Later figures each have a narrow role: main comparative performance, scalability/compatibility, or real-robot stages.

### 5. Caption style

- Captions define the compared inference contracts, not just their names.
- Controlled-analysis captions specify what is held constant.
- Result captions state aggregation and evaluation scope while leaving proofs and long parameter lists outside the artwork.

### 6. Table style

- Tables carry success and inference time together when both are central to the practical claim.
- Comparisons are grouped by policy/inference family.
- Precision and emphasis are restrained; tables expose baselines and method variants rather than only gains.
- Uncertainty is not the primary table language in every result, a pattern `one-clock` should improve on with paired and task-cluster intervals.

## PACE

### 1. Abstract structure

- Starts from the deployment interface: a policy predicts a chunk and an executor chooses how much to execute.
- Gives the empirical observation that fixed-horizon performance is task-dependent and non-monotonic.
- Introduces a phase-aware hypothesis and a training-free rule based on predicted speed valleys.
- Uses simulation and real-robot absolute before/after results as evidence.
- Ends with behavior-level analysis of when horizons shorten or lengthen.
- Its five principal claims are observation, mechanism hypothesis, method, cross-domain performance, and rollout-level consistency with the hypothesis.

### 2. Introduction structure

1. Establishes action chunking and the unresolved execution-horizon choice.
2. Shows fixed-horizon sensitivity and argues a single horizon is unreliable.
3. Frames manipulation as locally coherent phases separated by transitions.
4. Introduces speed valleys as observable replanning candidates.
5. States the method and why it is training-free/policy-agnostic.
6. Summarizes large simulation and real-robot evidence plus analysis.

The central question appears on the first page. The headline empirical observation precedes the proposed method. Related work is contrasted by the level at which it acts: training/representation versus test-time execution.

### 3. Results narrative

- Main results are organized around evaluation questions: aggregate performance, task breadth, sensitivity to policy/horizon, and real-world behavior.
- Diagnostic fixed-horizon sweeps are explicitly distinguished from the adaptive method evaluation.
- Rollout-level plots follow aggregate tables and test whether the selected horizons align with hypothesized manipulation phases.
- A visible failure case is included, making the boundary condition part of the narrative.

### 4. Figure rhetoric

- Figure 1 works as a miniature paper: fixed-horizon sensitivity, a phase-aware execution concept, and a headline comparison.
- Figure 2 narrows to the executor pipeline and omits benchmark clutter.
- Figure 3 aligns camera frames, kinematics, and chosen horizons over time, letting a reader assess the proposed mechanism directly.
- Later rollout figures mark policy-query times consistently and use selected frames only where they explain phase changes.

The transferable principle is the hierarchy—from problem to mechanism to consequence—not the specific panel arrangement or appearance.

### 5. Caption style

- Captions define markers and signals, state the evaluated task, and give a one-sentence behavioral interpretation.
- The framework caption explains what is computed and what is executed.
- Failure-case captions state why the method failed rather than merely labeling the rollout unsuccessful.

### 6. Table style

- Headline tables report absolute baseline and method performance plus a simple delta.
- Task-level detail is moved to later tables, preserving main-text hierarchy.
- Fixed-horizon diagnostic sweeps are tabulated separately from adaptive-method results.
- Precision is generally one decimal point for percentages; bolding identifies a comparison winner without implying statistical certainty.

## AutoHorizon / VLA Knows Its Limits

### 1. Abstract structure

- Establishes action chunking in flow-based VLAs and identifies execution horizon as underexplored.
- States a peaked empirical horizon-response pattern.
- Introduces two attention observations as the proposed explanation.
- Converts those observations into an attention-guided dynamic horizon method.
- Ends with scope, overhead, and generalization claims.
- The abstract contains about five claim types: sensitivity, two mechanistic observations, method, and empirical breadth.

### 2. Introduction structure

1. Gives broad VLA/action-chunking context.
2. Presents the fixed-horizon sensitivity observation in Figure 1.
3. Contrasts human/exhaustive horizon selection with the desired dynamic choice.
4. States the consistency/reactivity trade-off as established background.
5. Analyzes visual-language and action self-attention to propose a predictive-limit interpretation.
6. Introduces the bidirectional soft-pointer mechanism.
7. Ends with three explicit contributions: analysis, method, and empirical validation.

The central research question appears immediately after the motivating empirical curve. Mechanism evidence is presented before the method is fully specified.

### 3. Results narrative

- The narrative progresses from fixed-horizon sensitivity to attention observations, then to the dynamic method and benchmark evaluation.
- Simulation and real-world results are separated.
- Attention visualizations and method ablations are used to test internal logic after the headline comparison.
- Static-oracle comparisons and fixed-horizon tables make clear that a dynamic method is being compared with strong fixed choices.

### 4. Figure rhetoric

- Figure 1 asks one question and answers it directly with success versus execution horizon.
- Figure 2 compares conventional execution and the adaptive mechanism alongside a rollout phase sequence.
- Attention figures separate cross-attention and self-attention phenomena rather than collapsing them into one generic heat map.
- Method and evidence use consistent terminology for prediction horizon, execution horizon, and inferred boundary.

### 5. Caption style

- Captions state the empirical pattern or attention phenomenon that the visual supports.
- Overview captions define left/right components and execution semantics.
- They are informative but do not substitute attention structure for causal proof; `one-clock` should preserve that restraint explicitly.

### 6. Table style

- Main tables compare success across suites and models; secondary tables document chosen static horizons and oracle sensitivity.
- Tables are preferred when many suite/model combinations would make a plot unreadable.
- Values use consistent percentage precision and limited bolding.
- Uncertainty is not always prominent; `one-clock` should retain paired and task-cluster uncertainty even when adopting this hierarchy.

## Transferable synthesis

Across the five papers, the strongest recurring pattern is a short causal-looking communication chain without overloading any one display: define the execution problem, show one interpretable empirical observation, state a bounded mechanism hypothesis, and then test a practical consequence. First figures often act as miniature papers. Main tables preserve absolute performance and task breadth. Mechanism figures align a predicted signal with rollout behavior, but the papers vary in how strongly uncertainty and alternative explanations are handled.

For `one-clock`, the useful adaptation is therefore:

1. Make same-target component assignment visually and mathematically unambiguous.
2. Lead with the robust diagonal assignment asymmetry, while showing that simple effects are conditional.
3. Place the three mechanism quantities side by side but keep them named separately: demonstration persistence, future-action forecast error, and cross-source prediction disagreement.
4. Treat Track A as the practical consequence at matched policy-query schedule, not as proof that generic frequent replanning can hurt.
5. Put full paired discordances, cluster uncertainty, exposure lineage, and task-level results in tables/supplement rather than crowding the conceptual figures.

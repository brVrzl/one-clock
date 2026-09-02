# `one-clock` paper and figure-specification style guide

This guide is an input to later human/co-PI paper-rewrite and figure-design sessions. It does not authorize manuscript editing or final figure artwork in the present execution session.

## A. Writing principles

- Organize the paper around scientific questions, not experiment chronology or internal experiment names.
- Begin every Results subsection with its question or hypothesis. State the qualitative answer before dense statistics.
- Keep only the minimum statistics needed to support the sentence in prose. Move complete success counts, discordances, intervals, per-task effects, exposure classifications, and audit detail to tables or supplement.
- Distinguish three evidence roles with stable labels: **preregistered confirmation**, **development characterization**, and **post-hoc supporting analysis**. Establish a section’s role once, then write fluidly within it instead of interrupting every sentence with provenance reminders.
- Never inflate the robust diagonal contrast `A0G20 - A20G0` into an additive attribution to arm freshness or gripper commitment. Simple effects depend on the other component’s temporal state.
- Treat negative findings as boundary conditions. Report them in the same question–answer structure as positive findings, without apology or rescue-method speculation.
- Do not claim that the paper discovers generic degradation from frequent replanning. ACT, BID, RTC, and related work establish the consistency/reactivity trade-off.
- Use **policy-query rate**, **policy-query budget**, and **replanning frequency**. Report wall-clock separately. Do not call policy queries FLOPs or “4× compute.”
- Describe the 140-block confirmation checkpoint structure exactly. Do not call it a single-checkpoint experiment. Unless a provenance audit shows otherwise, the limitation is one trained checkpoint/training seed per task.
- Keep the three mechanism quantities distinct in wording and notation:
  1. demonstration action temporal persistence;
  2. frozen-policy future-action forecast error;
  3. same-target cross-source prediction disagreement.
  They answer different questions and must not be collapsed into “prediction error.”
- For Track B use: “mechanism-only logging on already outcome-exposed development cells; success outcomes are not used for method selection.” Do not call the interaction outcome-free.

## B. Abstract target structure

Target approximately seven linked moves:

1. Established context: action chunks provide temporal coherence but deployment requires a replanning schedule.
2. Overlooked problem: common executors implicitly treat all action components as temporally homogeneous.
3. Same-target formulation: compare predictions for the same physical target while assigning source age separately to arm and gripper components.
4. Headline observation: report the robust component-assignment asymmetry without converting it into additive arm/gripper percentages.
5. Mechanism/scale result: summarize temporal-sensitivity curves and the separate persistence, forecast, and cross-source-disagreement findings, including a failed localization criterion if that is the outcome.
6. Practical or scope consequence: summarize Track A’s matched-query component-resolved executor result, or state its negative boundary clearly.
7. Precise final claim: bound the conclusion to the evaluated policy families, tasks, and executor semantics.

Do not turn the abstract into a list of cohorts or experiments. Use at most a small number of memorable quantitative anchors.

## C. Introduction target structure

- **P1:** Why action-chunk execution matters: coherence, reactivity, and the deployment role of execution horizon.
- **P2:** Missing assumption: standard executors largely apply one temporal rule across heterogeneous action components.
- **P3:** Same-target formulation and central question. Define `q + k = t` and component-resolved source assignment in plain language.
- **P4:** Headline empirical observation and why the diagonal contrast does not imply unique additive attribution.
- **P5:** Mechanism and temporal-scale question, explicitly separating demonstration persistence, forecast error, and cross-source disagreement.
- **P6:** Practical consequence and scope: component-resolved periodic execution at matched policy-query schedule, plus cross-policy boundary evidence.
- **P7:** Concise contributions, each corresponding to a scientific question and a specific evidence role.

Contrast prior work when the missing assumption becomes clear, not in a detached catalog. The first mention of the central observation should precede a detailed protocol inventory.

## D. Figure-design principles

Final artwork will be produced elsewhere. Every specification should enforce:

- white or light background;
- a restrained semantic palette whose arm, gripper, and coherent-executor meanings remain fixed across the paper;
- one primary scientific question per figure;
- strong information hierarchy and minimal in-plot prose;
- vector-friendly construction, with no decorative gradients or unnecessary 3D effects;
- legibility at final two-column print size;
- redundant encodings such as color plus line style, marker, or direct label;
- uncertainty visually secondary to point estimates but never omitted where it is inferentially required;
- full p-values, discordances, audit metadata, and exhaustive task results in tables or captions, not crowded into plots.

Physical sizing must come from the repository’s actual ICRA template. The current `paper/icra2027/main.tex` loads `ieeeconf.cls` as `letterpaper, 10 pt, conference`; that class sets `\textwidth=7.0in` and `\columnsep=0.2in`, implying a current single-column width of 3.4in and double-column width of 7.0in. A later figure-design session must recheck these values against the final template before export and match the manuscript typography. Publication plots should be exported as vector graphics then, not in this session.

## E. Figure-1 principle

The first major figure should function as a miniature version of the paper:

`problem / same-target concept -> component assignment -> headline observation`.

Use the information-hierarchy principle visible in strong robot-learning first figures, including PACE, without copying their arrangement, visual appearance, icons, or colors. Make it immediately clear that `A0G20` and `A20G0` are diagnostic probes at policy-query rate 1, not deployment recommendations.

## F. Mechanism-figure principle

The key analysis figure should align behavioral temporal-sensitivity curves with the three distinct mechanism measurements:

- fixed-source behavioral sensitivity versus the exact preregistered age grid;
- demonstration action persistence versus lag;
- frozen-policy future-action forecast error versus chunk offset;
- same-target cross-source prediction disagreement versus source age.

This alignment lets readers assess whether the proposed mechanism tracks the behavioral phenomenon. Parallel axes or panel positions must not imply that the quantities are mathematically interchangeable or causally identified.

For fixed-source sensitivity, use exactly `d in {2,4,8,12,16,20,32}` and represent `d=0` only as the common Fresh anchor. Do not import x-values from reference papers or visual examples.

## G. Result-figure principle

If Track A is scientifically meaningful, use a clean success-versus-policy-query-rate representation and/or explicit matched-condition comparisons for:

`H16`, `H4`, `ARM4_GRIP32`, `H2`, `ARM2_GRIP16`, and `TE_DENSE`.

The x-axis is **policy-query rate**, never compute. Show wall-clock separately. Make the matched comparisons `ARM4_GRIP32 - H4` and `ARM2_GRIP16 - H2` visually unambiguous.

Do not connect `A0G20`/`A20G0` same-target diagnostic probes into a coherent-horizon dose-response line. Their fixed-source, dense-query intervention semantics differ from periodic executor schedules.

If Track A is not main-paper-worthy but the frozen SmolVLA scope experiment is informative, Figure 4 may instead compare ACT and SmolVLA component-assignment effects. Do not force a negative result into a planned figure number.

## H. Caption principle

Every caption should answer:

1. What is shown?
2. How do the conditions differ?
3. What does the uncertainty represent and what is the inference unit?
4. What one takeaway does the display support?
5. What important limitation or evidence-role qualifier is needed?

Do not merely repeat axis labels. Define unfamiliar notation once and identify reused data versus newly run conditions.

## I. Terminology

- Prefer descriptive notation `A0G0`, `A0G20`, `A20G0`, and `A20G20`.
- Retain `FO20` and `Reverse20` only where historical provenance or artifact identity requires the aliases.
- Use **same-target source age**, **chunk offset**, **execution horizon**, and **policy-query rate** precisely.
- Say **continuous gripper policy output** unless the actual execution path applies a discrete threshold or sign operation. Distinguish the emitted value from how the environment interprets it.
- Use **interaction magnitude** when discussing an unsigned value. For signed values, state the canonical difference-in-differences formula.
- Label the 140-block factorial interaction `POST_HOC_SUPPORTING_INTERACTION`; the earlier 126-block Object interaction was predeclared descriptive and its status does not transfer.

## J. No style copying

Do not copy exact sentences, layouts, icons, palettes, schematics, or artwork from reference papers. Do not reuse a panel arrangement when it does not match the `one-clock` evidence chain. Extract only general communication principles such as question-first organization, consistent encoding, and separation of concepts from audit detail.

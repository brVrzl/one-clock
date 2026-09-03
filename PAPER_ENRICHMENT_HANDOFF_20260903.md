# ICRA 2027 targeted manuscript enrichment handoff

Date: 2026-09-03 (Asia/Shanghai)

## Scope and title

This pass enriches the scientifically frozen manuscript without adding an
experiment, statistical analysis, comparison, interval, moderator, mechanism,
or scientific claim. Canonical writing artifacts remain unchanged.

Working title: **Component-Dependent Sensitivity to Stale Predictions in ACT
Action-Chunk Execution**.

## 1. Build and page count

- The manuscript compiles successfully with Tectonic 0.17.0, including BibTeX
  and cross-reference reruns.
- Current output: `paper/icra2027/main.pdf`.
- Current page count: **7 pages**, including references and two short supporting
  appendices, with specification-box figure placeholders.
- Paper size: US Letter.
- This is not a target PaperCept/pdflatex pagination result. Final page count
  must be remeasured with the target toolchain and final artwork.
- The log contains no overfull boxes and no undefined citations or references.
  It reports 21 underfull horizontal boxes and 2 underfull vertical boxes,
  concentrated in narrow table cells, number-dense result lines, float
  transitions, and a long bibliography URL. No text crosses a margin in the
  rendered draft.
- Tectonic substitutes Latin Modern for unavailable `TU/ptm` Times shapes. This
  local font substitution is another reason not to infer final pagination.

## 2. References added and why

Two references were added after checking their bibliographic identity and
contribution against primary proceedings records:

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action
  Diffusion* (RSS 2023), to position receding-horizon action-sequence execution.
- Liu et al., *Bidirectional Decoding: Improving Action Chunking via Guided
  Test-Time Sampling* (ICLR 2025), to position guided test-time selection among
  action chunks.

No entry was added from a method name alone. Suggested neighbors without a
verified record in the available project material were omitted rather than
guessed. There are no citation TODOs.

## 3. Final cited-reference count

The manuscript cites **16 unique references**. All 16 citation keys resolve to
the 16 entries emitted by BibTeX; no bibliography entry is unused. The local
citation scan found no placeholders and no duplicate BibTeX keys.

## 4. Tables

The compiled manuscript now contains **5 tables**:

1. the retained reader-facing evidence/provenance table;
2. a main-text mechanism-diagnostic accounting table;
3. the main-text R1C dense-query factorial contrast table;
4. the main-text Track-A pooled and per-suite execution table; and
5. the complete seven-lag R1A table in Appendix II.

The Track-A table reports pooled success, measured policy queries per executed
environment step, and all three suite results for H16, H4, ARM4_GRIP32, H2,
ARM2_GRIP16, and TE_DENSE.

## 5. TE_DENSE scope

TE_DENSE appears in Q4 after the fixed-cadence Track-A results and suite
heterogeneity paragraph. It is explicitly introduced as a secondary frozen
condition, not a headline method comparison. The text reports the canonical
288/450 versus 357/450 comparison, discordance, exact McNemar result, and both
intervals. It then characterizes the frozen LeRobot v0.4.4 temporal-ensembling
configuration, realized prediction ages, near-boundary gripper-command rate,
and low switch rate. The paragraph closes by stating that these measurements do
not establish that age weighting or intermediate gripper commands caused the
performance loss. It does not use bug, harmful, or chatter language.

## 6. R1C table and conditional-effect wording

Table III includes all six frozen first-condition-minus-second-condition R1C
contrasts, discordance, exact McNemar values, paired intervals, and
task-cluster intervals. The main text emphasizes only the conditional gripper
comparisons:

> Under identical dense querying, extending gripper commitment produced a clear
> gain when the arm was also committed.

C11-C10 is reported as +12.14 points with both canonical intervals above zero;
C01-C00 is +2.86 points with both intervals spanning zero. The text concludes
that query schedule alone cannot explain this conditional effect in the tested
deterministic evaluator. It does not call the +9.29-point interaction
statistically established.

## 7. Related Work positioning

Related Work now separates four scientific neighborhoods: action chunking and
temporal ensembling, whole-policy execution adaptation, stale/asynchronous
execution, and heterogeneous action generation. Verified adaptive methods each
receive a concise statement of the temporal decision, signal, and scope. The
section ends with the explicit boundary that this paper does not propose an
alternative to those methods. It measures whether a chunk-level temporal
decision has the same behavioral consequence for every action component, while
Track A is only a bounded operational test at fixed arm cadences.

The Introduction retains three contributions while making the within-arm
heterogeneity result and the reversed diagnostic ordering separately visible.

## 8. Figure layout

The source contains **4 figure placeholders** and no final paper-facing
artwork:

- Fig. 1 remains a full-width `figure*` for the same-target construction and
  preregistered primary factorial.
- Fig. 2 remains a full-width `figure*` for the R1A curves and Object-126
  component characterization.
- Fig. 3 remains a single-column mechanism-diagnostic figure centered on the
  dispersion/behavior ordering disagreement.
- Fig. 4 is now a single-column figure for Track-A success versus log query
  rate, the two fixed-cadence pair annotations, a distinct TE_DENSE marker, and
  a suite inset only if it remains legible.

No regression or line joining all Track-A conditions is specified.

## 9. Placeholder safety

`main.tex` defines a small draft/submission conditional. The current draft uses
`\submissionbuildfalse`. Setting `\submissionbuildtrue` for the submission
build makes compilation fail if any `\draftfigureplaceholder` remains. The
final submission checklist must include both switching to submission mode and
confirming that all four placeholder calls have been replaced by final artwork.

## 10. Evidence intentionally kept supplementary

The complete R1A numerical grid remains in Appendix II rather than the main
text because final Fig. 2 is intended to carry the two absolute curves. The
interaction accounting remains in Appendix I. Its cautious conclusion is
unchanged, and the added sentence only records that the dense-query estimate
used matched policy-query schedules. No interaction estimates are pooled and
no new interaction uncertainty is reported.

## 11. Prose audit

The final source audit found:

- no leaked internal governance labels;
- no unsupported positive causal mechanism;
- no claim that rotation staleness is free or unaffected;
- no generalized stale-gripper benefit;
- no claim that `d=20` is an optimum or statistically distinct peak;
- no equation of policy-query rate with compute, latency, or total query count;
- no claim of global component-schedule or adaptive-method superiority;
- no cross-policy behavioral generalization;
- no description of TE_DENSE as a bug, intrinsically harmful, or chattering;
- no claim of a statistically established R1C interaction; and
- no em dash in manuscript prose.

The negative phrases that remain, such as “not globally superior,” are the
required scientific boundaries rather than superiority claims.

## 12. Scientific-freeze confirmation

No experiment was run, no scientific analysis was rerun, and no new statistic
was calculated. Every numerical result inserted into the manuscript was checked
against the frozen canonical artifacts. Only manuscript-writing artifacts were
modified for this pass.

# Final Pre-Figure Manuscript Handoff

Date: 2026-09-03

## Frozen reader structure

The anonymous manuscript now follows this structure:

1. Introduction
2. Related Work
3. Same-Target Temporal Probe
4. Experimental Setup
5. Results
   - Q1: Same-Target Arm--Gripper Asymmetry
   - Q2: Temporal Scale and Component Identity
   - Q3: Do Simple Diagnostics Explain the Ordering?
   - Q4: Executable Consequences and Query-Matched Controls
6. Discussion and Limitations
7. Conclusion
8. References

There is no standalone appendix or supplementary document in the submitted PDF.

## Appendix migration

- The complete canonical seven-lag R1A table is now in Results Q2.
- The original-cohort and R1C interaction accounting is compressed into a supporting paragraph immediately after the R1C results in Q4. It retains the original intervals and task sign-flip value, states that R1C has no canonical interaction-specific uncertainty, and concludes that the point estimates do not establish replicated interaction.
- The outcome-blind selection and preregistration credibility statement is now in Experimental Setup. Development and reviewer-directed evidence remains explicitly separated from the primary cohort.
- Per-task tables, complete B3 curves, source-age histograms, feasibility inventories, and internal governance records were intentionally not migrated.
- `sections/supplementary.tex` and the former combined setup/method source are removed from the active paper; their retained content is in the two new reader-facing sections.

## Literature and citation audit

Two direct neighbors were added after checking their original/official records:

- Yuanchang Liang et al., *Adaptive Action Chunking at Inference-time for Vision-Language-Action Models*, CVPR 2026. The paper is cited for inference-time chunk sizing from action entropy in the current prediction.
- Dong Jing et al., *Mixture of Horizons in Action Chunking*, ICML 2026. The official ICML program confirms the title, authors, year, and venue. No unverified PMLR volume or page range was invented. The paper is cited for multiple prediction horizons, learned gating, and cross-horizon consensus.

Archival metadata changes:

- RTC was upgraded from its arXiv entry to the NeurIPS 2025 Main Conference proceedings version, volume 38, pages 33383--33407, DOI `10.52202/085713-1122`.
- LeRobot remains cited as an ICLR 2026 paper; its author order was corrected against the official record and the citation now points to that record.
- ACT and Diffusion Policy remain their RSS 2023 versions; LIBERO remains the NeurIPS 2023 Datasets and Benchmarks version; BID remains the ICLR 2025 version; and ARP remains the IEEE Robotics and Automation Letters version.

All 18 cited entries were checked against an original paper or official proceedings record. The remaining arXiv entries are retained only where no archival version was verified: SmolVLA, the two action-chunk/open-loop analyses, AutoHorizon, PACE, DEHP, BCP, VLA-Corrector, and A3. Their arXiv identifiers and the manuscript clauses they support were checked against the original abstracts. The offline bibliography audit reports 18 cited entries, zero unused entries, zero duplicate keys, zero missing required fields, and zero undefined citation keys. No proposed citation pairing was rejected after verification; two metadata defects in the existing bibliography were corrected as described above.

Related Work now distinguishes the temporal decision and signal used by each directly neighboring method without presenting a citation catalog. It explicitly states that this paper does not propose a competing adaptive-horizon executor; it measures whether a chunk-level temporal decision is behaviorally uniform across translation, rotation, and gripper components.

## Evidence and display inventory

The current draft contains five tables:

1. reader-facing evidence provenance;
2. complete Object-126 seven-lag R1A sweep;
3. mechanism-diagnostic accounting;
4. dense-query R1C contrasts;
5. Track-A pooled and per-suite execution results.

It contains four draft figure specifications, all protected by the hard-failure safeguard:

- Fig. 1: two-column same-target construction and primary preregistered result;
- Fig. 2: two-column R1A curves and Object-126 component characterization;
- Fig. 3: single-column mechanism diagnostic;
- Fig. 4: single-column Track-A query-rate/success result with compact suite information if legible.

Fig. 4 is not a `figure*`. No final figure artwork was created in this pass.

## Build, page budget, and warnings

The best available local ICRA-compatible toolchain was Tectonic; standalone `pdflatex` and `bibtex` executables are not installed on this machine. The repository draft and a clean unpacked export both compile successfully from source, including a fresh BibTeX pass.

- Total pages, including references: **7**
- Cited references: **18**
- Tables: **5**
- Figure placeholders: **4**
- Undefined citations/references: **0**
- Overfull hboxes/vboxes: **0 / 0**
- Underfull hboxes/vboxes: **21 / 1**

The underfull warnings arise mainly from compact table cells, long method names, and bibliography URLs. Visual inspection found no defect worth aggressive spacing changes before final artwork. Final pagination must be remeasured with the target PaperCept/pdflatex build after the four figures are replaced.

## Clean export tests

The ZIP was unpacked into a new temporary directory outside the repository and compiled without cached `.aux`, `.bbl`, or repository-relative files. Normal draft mode succeeded at seven pages with all citations and references resolved.

For the required negative test, the unpacked copy was changed to `\submissionbuildtrue` while the placeholders remained. Compilation exited with status 1 at Fig. 1 with the expected `Draft figure placeholder remains` package error. The safeguard remains a hard failure, not a warning.

The clean package contains exactly 17 files: `main.tex`, `ieeeconf.cls`, `IEEEtran.bst`, `references.bib`, the eight active section files, the four figure specification files, and the anonymous compile note. It contains no build products, experiment outputs, canonical artifacts, governance files, repository metadata, or files outside the compile dependency set.

## Anonymity and prose compliance

The active LaTeX sources, exported package, rendered PDF text, and PDF metadata were scanned. The author declaration is empty; the PDF has no Author, Subject, or Keywords metadata. No project/repository name, GitHub URL, branch name, manuscript commit, preregistration SHA, local path, username, machine name, affiliation, or institution-identifying comment was found. Bibliography author names are, of course, retained as scholarly citations.

The final prose audit found no leaked internal governance labels; unsupported causal claim; rotation described as free, unaffected, or costless; generalized stale-gripper benefit; positive optimum, peak, plateau, or tolerance-threshold claim at `d=20`; query rate equated with compute; global method-superiority claim; cross-policy behavioral generalization; TE_DENSE bug, intrinsic-harm, or chatter language; or claim of a statistically established/replicated R1C interaction. Matches for words such as “cause,” “peak,” “compute,” and “replicated” occur only in explicit limitations or negations.

## Export paths

- Anonymous LaTeX package: `/home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation/paper/icra2027/export/icra2027_anonymous_latex.zip`
- Current anonymous draft PDF: `/home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation/paper/icra2027/export/icra2027_current_draft.pdf`

## Final submission checklist

- Replace all four placeholders with final local vector artwork, then build with `\submissionbuildtrue`.
- Recheck the full PaperCept/pdflatex PDF against the eight-page total limit after artwork is inserted.
- Repeat citation, reference, anonymity, PDF-metadata, and layout checks on that final build.
- **AI-use disclosure must be resolved by the authors against the final ICRA 2027 policy and the final human-edited manuscript before submission.** If article content beyond editing or grammar enhancement remains AI-generated, follow the official ICRA/RAS disclosure requirement in force at submission.

No experiment, rollout, statistical calculation, confidence-interval calculation, moderator analysis, mechanism analysis, or other new scientific analysis was performed. Only manuscript-writing, bibliographic verification, compliance auditing, compilation, and export packaging were carried out.

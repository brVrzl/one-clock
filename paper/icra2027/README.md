# ICRA 2027 paper workspace

> **Status note (2026-08-24):** Gate-3C is complete and is the authoritative
> scientific backbone. The RTX 5080 is now the primary simulation, analysis,
> and manuscript machine; Thor is retained for historical evidence and
> lightweight or hardware-specific verification.
> `main.tex` and the existing section files remain historical prose from the
> discontinued group-specific-horizon direction. They are not the scientific
> source of truth. The active post-Gate-3C architecture lives in
> [`notes/post_gate3b_paper_plan.md`](notes/post_gate3b_paper_plan.md), and the
> fill-ready prose, methods, results, and tables live in
> [`notes/pre_gate3c_manuscript_skeleton.tex`](notes/pre_gate3c_manuscript_skeleton.tex).
> Gate-3B's directional result remains developmental evidence. Gate-3C's
> validated primary outputs are now integrated in the active skeleton and
> Figure 4 interface.

Historical provisional title: **One Clock Does Not Fit All: Group-Specific Execution
Horizons for Action-Chunking Robot Policies**.

Recommended current title: **Fresh Feedback, Retained Intent: Asymmetric Temporal
Reuse for Action-Chunked Robot Policies**. The historical title above is not a
candidate for the active paper.

The bounded closest-work recheck is recorded in
[`notes/gate3b_directional_related_work_update.md`](notes/gate3b_directional_related_work_update.md).
Figure 2 has a validated Gate-3B data interface, Figure 3 has a frozen
offline-versus-closed-loop interface, and Figure 4 contains only validated
Gate-3C primary-cohort data under [`figures/`](figures/).

## Submission format audit (2026-08-20)

The current official [ICRA 2027 call for papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
specifies:

- an eight-page limit for the complete initial paper, including text, figures,
  tables, acknowledgments, bibliography, and references;
- ICRA/IEEE double-column format;
- double-anonymous review, with author names and affiliations excluded; and
- disclosure in the acknowledgments of AI-generated article content, naming the
  system and the affected content.

The official [IEEE RAS double-anonymous rules](https://www.ieee-ras.org/publications/rules-for-the-double-anonymous-review-process/)
were also checked. The draft contains no author, affiliation, lab, funding,
private repository, or local-path identifiers. Its AI-use acknowledgment is
non-identifying and included to satisfy the conference-specific disclosure rule.

The unchanged template files were downloaded from the official
[PaperCept manuscript-support page](https://ras.papercept.net/conferences/support/tex.php):

- `ieeeconf.cls` from `ieeeconf.zip`, SHA-256
  `4befef671c2a996889d325f5170d3387bf42aac9a37dcaa93724ad49816e4ec2`;
- `IEEEtran.bst` from `IEEEtranBST.zip`, SHA-256
  `b11af8e5096681f1eccdce6c72c047dc056ddeef52dff340213104019bcf3409`.

`main.tex` uses the official US-letter conference declaration:

```tex
\documentclass[letterpaper,10pt,conference]{ieeeconf}
```

## Regenerating figures

From the repository root:

```sh
python paper/icra2027/generate_figures.py
```

The script reads the committed experiment JSON and writes vector PDFs under
`paper/icra2027/figures/`. It does not read or modify raw runs.

## Building

With a standard LaTeX/BibTeX installation:

```sh
cd paper/icra2027
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The host environment has no `pdflatex`, `bibtex`, or `latexmk`, but an existing
local `embodied-latex:texlive2023` Docker image was used without installing a TeX
distribution. The complete draft compiles to eight US-letter, double-column
pages, including acknowledgments and references. The final build has no
unresolved citations, undefined labels, BibTeX errors, or overfull boxes; it
emits only non-fatal underfull-box warnings.

# Scientific skill installation report

Audit date: 2026-08-21 (Asia/Shanghai)

## Installation mechanism

Installed with Codex's supported `skill-installer` helper using git sparse checkout and immutable commit pins. The upstream `curl | bash` installer was inspected but not executed.

## Scientific Agent Skills

- Source: `K-Dense-AI/scientific-agent-skills`
- Inspected and installed commit: `390f5146bf3c1877cf15636a3dd7b775e4f0f185`
- Installed: `scientific-brainstorming`, `hypothesis-generation`, `scientific-critical-thinking`, `experimental-design`, `literature-review`, `peer-review`, `citation-management`
- Not installed: the K-Dense `scientific-writing` skill, because the requested Nature skill set provides a same-named skill and Codex skill names must not collide.

## Nature Paper Skills

- Source: `Boom5426/Nature-Paper-Skills`
- Inspected and installed commit: `47a06b35f1eed23fb943f65e86777eb37c605090`
- Installed: `paper-workflow`, `paper-bootstrap`, `scientific-writing`, `manuscript-optimizer`, `results-section-revision`, `figure-planner`, `citation-verifier`, `data-availability`, `submission-audit`, `rebuttal-response`, `nature-portfolio-playbook`
- The full skill directories were installed, including local scripts and references.

## Safety review

- No selected installer or skill performs a destructive action at install time.
- The repository installer that advertises `curl | bash` also replaces existing skill directories with `rm -rf`; it was not used.
- The selected local validation scripts either do not delete data or only remove their own temporary files. No selected directory contains symlinks.
- `literature-review` includes optional OpenRouter schematic scripts that can read `OPENROUTER_API_KEY`, including from a local `.env`, and send prompts to OpenRouter. These scripts will not be invoked in this audit.
- `citation-management` can optionally read the explicitly documented `NCBI_API_KEY`, `NCBI_EMAIL`, and `OPENALEX_EMAIL` variables for their named services. No credential is required, and those scripts will not be given project secrets.
- The inspected copies compiled successfully. Installed directories match the inspected, commit-pinned sources (excluding audit-generated `__pycache__` files in the temporary checkout).

## Relevance and venue control

The K-Dense skills are relevant to evidence grading, falsifiable hypotheses, experimental design, literature reconstruction, and adversarial review. The Nature skills are relevant to claim/evidence discipline, figure logic, citation checks, data availability, and submission auditing. ICRA 2027 conventions and robotics evaluation requirements remain authoritative; Nature-specific venue or formatting defaults do not control this project.

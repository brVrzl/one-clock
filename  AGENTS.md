# AGENTS.md

## Project Intent

This is a research codebase for controlled experiments on execution-time
scheduling of action-chunking robot policies.

The primary goal is to test research hypotheses with simple, auditable
experiments. Favor research clarity and iteration speed over production-grade
infrastructure.

## Research Scope / Anti-Overengineering

The current research question and its deciding evidence take priority over
hypothetical future requirements. This supplements the engineering and scope
rules below.

- Prefer the smallest implementation or experiment that distinguishes the
  current hypotheses. Start with focused smoke tests and small discriminative
  experiments; scale only when their result justifies it.
- Before running a test, audit, or benchmark, identify the concrete failure it
  can detect and the decision that would change if it fails. Do not repeat
  audits, reviews, or tests once sufficient evidence exists.
- Do not add speculative infrastructure, repeated scoring/review loops, or
  subagents unless they materially help the current task. Necessary callers,
  fixtures, interfaces, and data-path changes are in scope when the requested
  work genuinely requires them.
- Interpret task intent strictly: review, inspect, analyze, investigate, and
  explain are read-only unless changes are explicitly authorized; implement,
  change, fix, try, and prototype authorize only the requested work and its
  necessary consequences.

## Robot Safety Exception

Anti-overengineering rules never authorize weakening justified physical robot
safety. Preserve existing actuator and joint limits, workspace limits,
action-domain validation, watchdogs, emergency stops, collision/contact and
force/torque checks, command validation, and safety interlocks. Do not invent
additional defensive layers without reachable evidence or an explicit request.

## Completion

When the requested result has enough evidence to answer the current question,
stop and summarize the result and important evidence. Do not begin another
audit, review, or refactor cycle unless requested.

## Architecture

- Keep base policy training/inference separate from execution scheduling.
- Execution strategies must be comparable under the same policy checkpoint,
  observations, task configuration, and evaluation protocol.
- Treat RoboTwin, XPolicyLab, policy repositories, and other research codebases
  as upstream dependencies.
- Modify upstream code only at the narrowest necessary integration point when a
  clean external integration is not practical.
- Never guess action tensor semantics from indices. Establish action semantics
  from source code, configuration, or verified runtime behavior before defining
  action groups.
- Keep experiment configuration explicit and reproducible.
- Prefer repository-native mechanisms over introducing parallel infrastructure.

## Engineering Style

- Implement the smallest direct solution that tests the current hypothesis.
- Do not over-engineer.
- Avoid hypothetical defensive guards and premature abstractions.
- Do not build speculative plugin systems, registries, compatibility layers,
  factories, wrappers, or fallback stacks for possible future requirements.
- Do not perform unrelated refactors, renaming, formatting, cleanup, dependency
  upgrades, or architectural rewrites.
- Do not add helper classes or functions used only once unless they materially
  improve readability or isolate important research semantics.
- Validate real external boundaries and observed failure modes. Do not add
  defensive checks for states that cannot occur on the verified execution path.
- Do not hide unexpected failures behind broad exception handling or silent
  fallbacks. Unexpected failures should fail loudly with useful context.
- Do not create backup copies or version-suffixed source files. Git is the
  version history.
- Do not add SHA, MD5, checksum manifests, or file-integrity machinery for
  ordinary local files, generated outputs, checkpoints, configs, or downloads
  unless an upstream source explicitly requires a checksum or the task
  explicitly asks for one.
- Git commit IDs are appropriate for recording dependency and experiment
  provenance.
- Keep one source of truth for each configuration or semantic definition.
- Comments should explain non-obvious research semantics, not restate obvious
  code.
- Do not optimize performance before a correct measurable baseline exists.

## Scope Discipline

- Stay within the requested task.
- Do not change adjacent systems merely because they could be improved.
- Do not generalize an implementation for hypothetical future use unless the
  current task requires the generalization.
- Preserve unrelated user changes.
- Never rewrite Git history or discard unrelated work.
- When several implementations are possible, prefer the one with the smallest
  diff and fewest new concepts that correctly tests the hypothesis.

## Research Integrity

- Never fabricate experiments, rollout results, metrics, assets, checkpoints,
  commands, or successful tests.
- Distinguish clearly between not run, failed, and passed.
- Code inspection is not evidence that an experiment succeeded.
- Preserve evaluation seeds and experimental settings when comparing execution
  strategies.
- Do not change the base checkpoint, dataset, observation pipeline, task
  definition, or unrelated policy parameters while presenting a comparison as
  execution-only.
- Record enough provenance to reproduce meaningful experiment results without
  adding unnecessary bookkeeping infrastructure.

## Execution

- Inspect the real code path before editing it.
- Trace data and action flow far enough to understand the integration point
  rather than relying on names or assumptions.
- Make one coherent change at a time.
- Validate changes using the narrowest test or smoke run that exercises the real
  behavior being changed.
- Prefer real-path tests over large collections of mocks.
- Keep diffs scoped and reviewable.
- If action semantics or upstream behavior are unclear, establish them from
  source code or runtime evidence before implementation.
- Do not claim completion until the relevant runnable path has actually been
  exercised when the environment permits it.

## Git Workflow

- At the end of a task, if tracked files were meaningfully changed and the
  resulting state is coherent, commit the changes and push them to the current
  remote branch.
- Run the relevant narrow validation before committing.
- Do not create commits for tasks that made no tracked-file changes.
- Do not create many incremental checkpoint commits during one task.
- Do not force-push or rewrite history.
- A task may still be committed when externally blocked if the completed code
  changes are independently useful and their validation status is reported
  accurately.

## Completion Report

At the end of a task, report concisely:

1. What was changed.
2. Which files were changed.
3. Which commands were actually run.
4. What passed, failed, or was not run.
5. Any unresolved blocker or research-relevant finding.

Do not add ceremony beyond what is useful for the research workflow.

# Group-conditioned temporal memory development status

## Current state

Prepared, not launched. Sol's repaired ACT h16 trio is now recorded; no
group-memory simulator rollout or GPU job has been started from this
directory.

- Sol audit commit currently recorded: `33463ab4eb0ff1c64f794df7c76330bb7b56c143`.
- Sol pairing result: valid after the fresh identically seeded environment repair.
- Sol shared baseline decision: `dense_equivalent_te`, canonical oldest-to-newest orientation.
- Sol’s repaired h16 trio commit: `b0b2a6d18ccc9da9ded0057d9f512ad8b535dac0`.
- Repaired pairing is mandatory: fresh environment per condition/state with exactly identical task/state/environment seed settings.
- Strict common-prefix equality validator passed for the repaired trio.
- Corrected repaired ACT h16 results: hard `32/40`, candidate-index TE `24/40`, dense-equivalent TE `23/40`.
- Sol decision: `DENSE_EQ_TE_HARMFUL`; repaired candidate-index TE remains harmful.
- Group-memory rollout remains unlaunched; this coordination update does not authorize a rollout.
- M4 is explicitly unavailable because no frozen online-compatible group reliability interface exists in the current checkout.

## Frozen development scope

The protocol fixes h16, the four exposed development tasks, states 10--19, environment seeds 2000--2009, ACT horizon 100, SmolVLA horizon 50, and the method definitions M0--M3. `H_temp` remains analysis-only and is never read by the executor.

## Required next coordination event

If group-memory development resumes after this handoff, preserve the repaired
Sol pairing and shared-kernel decisions before any rollout:

1. use the recorded Sol trio commit and keep `shared_kernel.selected_name` equal to `dense_equivalent_te`;
2. construct a fresh environment for every group-memory condition/state with identical task/state/environment seed settings;
3. run the strict per-policy common-prefix smoke before any group-memory panel;
4. do not combine prior reset-reuse results with the repaired panel.


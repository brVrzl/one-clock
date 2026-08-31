# Group-conditioned temporal memory development status

## Current state

ACT development complete; SmolVLA not reached. Sol's repaired ACT h16 trio is
recorded, and the new group-memory workload has completed under the repaired
fresh-environment protocol.

- Sol audit commit currently recorded: `33463ab4eb0ff1c64f794df7c76330bb7b56c143`.
- Sol pairing result: valid after the fresh identically seeded environment repair.
- Sol shared baseline decision: `dense_equivalent_te`, canonical oldest-to-newest orientation.
- Sol’s repaired h16 trio commit: `b0b2a6d18ccc9da9ded0057d9f512ad8b535dac0`.
- Repaired pairing is mandatory: fresh environment per condition/state with exactly identical task/state/environment seed settings.
- Strict common-prefix equality validator passed for the repaired trio.
- Corrected repaired ACT h16 results: hard `32/40`, candidate-index TE `24/40`, dense-equivalent TE `23/40`.
- Sol decision: `DENSE_EQ_TE_HARMFUL`; repaired candidate-index TE remains harmful.
- New ACT M2/M3 workload: 80 episodes, complete across the four development tasks.
- ACT result: M0 `32/40`, M1 `23/40`, M2 `23/40`, M3 `23/40`.
- Decision: `GROUP_COGACT_NULL`.
- SmolVLA was not launched because the specified ACT gate requires STRONG or RECOVERS_HISTORY.
- M4 is explicitly unavailable because no frozen online-compatible group reliability interface exists in the current checkout.

## Frozen development scope

The protocol fixes h16, the four exposed development tasks, states 10--19, environment seeds 2000--2009, ACT horizon 100, SmolVLA horizon 50, and the method definitions M0--M3. `H_temp` remains analysis-only and is never read by the executor.

## Completion notes

The completed ACT panel preserved the repaired Sol pairing and shared-kernel decisions:

1. the recorded Sol trio commit is `b0b2a6d18ccc9da9ded0057d9f512ad8b535dac0` and `shared_kernel.selected_name` is `dense_equivalent_te`;
2. a fresh environment was constructed for every group-memory condition/state with identical task/state/environment seed settings;
3. the strict M0--M3 common-prefix smoke passed before the panel;
4. the final focused smoke also compared processed policy inputs exactly, with zero difference for M1, M2, and M3 against M0;
5. prior reset-reuse results were not combined with this panel.

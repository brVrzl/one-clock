# Group-conditioned temporal memory development status

## Current state

Prepared, not launched.

- Sol audit commit currently recorded: `33463ab4eb0ff1c64f794df7c76330bb7b56c143`.
- Sol pairing result: valid after the fresh identically seeded environment repair.
- Sol shared baseline decision: `dense_equivalent_te`, canonical oldest-to-newest orientation.
- Sol’s repaired h16 trio commit is not present in this checkout yet.
- Full rollout is therefore gated. No simulator rollout or GPU job has been started by this directory.
- M4 is explicitly unavailable because no frozen online-compatible group reliability interface exists in the current checkout.

## Frozen development scope

The protocol fixes h16, the four exposed development tasks, states 10--19, environment seeds 2000--2009, ACT horizon 100, SmolVLA horizon 50, and the method definitions M0--M3. `H_temp` remains analysis-only and is never read by the executor.

## Required next coordination event

When Sol pushes the repaired h16 trio (`hard_h16`, `candidate_index_te_h16`, `dense_equivalent_te_h16`):

1. pull that commit fast-forward-only;
2. record its exact SHA in `protocol.json` as `coordination.sol_repaired_rollout_commit`;
3. keep `shared_kernel.selected_name` equal to `dense_equivalent_te`;
4. run the strict per-policy common-prefix smoke;
5. launch ACT M2/M3 (and only protocol-compatible M0/M1 references) before considering SmolVLA.

No prior reset-reuse result may be combined with the repaired panel.


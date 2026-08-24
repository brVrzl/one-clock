# Thor-to-RTX 5080 completion report

Handoff date: 2026-08-24. This report records the final Thor-to-RTX scientific
transfer and the construction of the unified post-Gate-3C branch.

## Archive and archival root

The transferred archive was verified before extraction. Its SHA256 is

```text
51e8444fe612364ef17878208ac6bd17c36d42c252ed3ce507e1b750b25218f8
```

The stable non-Git evidence root is
`/home/wjq/one-clock-evidence/thor-era-20260824/`. The archive's
`provenance/SHA256SUMS` verified 2,495/2,495 files. All required content-tree
digests were independently recomputed and matched:

| Evidence | Transferred content-tree SHA256 | Result |
|---|---|---|
| Gate-3A1 Thor reference cache | `87e97a5711a7b51ea53da908774040d2b23ca57e9c29699bfd25f28ebe31908c` | PASS |
| Gate-3A2 | `d61f9850d0ee283dc823a8e7a208c02bac7b33bee196c27bcb1d6dd565131a4c` | PASS |
| Gate-3B | `046eedc6921205b28eacc6d24f7ce6bbc2d250b305c98383055935c100eac83d` | PASS |
| Gate-3C | `0df106c267c4651ac50182829e85e00a8e2791e68f44982b879facd7df506403` | PASS |
| Selective-retention transferred subset | `531123b0cb45520ba41878e513d2a16cb5906c0d7f8ff6bb27616f24e536b7de` | PASS; source full-root ledger mismatch retained |
| Static horizon task-0 20-state | `c79a6ca90756957c43ac7ef819f03bd4d73350e15571c8a58bec5d24ce76b2b6` | PASS; historical ledger mismatch retained |
| Static horizon task-0 extension | `3f2d1645128496a5c897f15e308244435ada7d9c248a65a4c6e1577e656d3837` | PASS; historical ledger mismatch retained |
| Static horizon cross-task | `99495287b4b882bce12f74c114cc5d644a12139162fb3f312d4585df61fa068c` | PASS; historical ledger mismatch retained |

The selective-retention and static-root discrepancies are not repaired or
relabelled. Per-file hashes and the transferred scopes remain auditable in the
archive provenance files.

## Unified branch and integration

- Gate-3C authoritative source head:
  `3b9f1209df7266160c47453e8ee66a142ea8688c`
- RTX source head:
  `279bcab9415e46f72c8bd2b89c3adbee773c3db6`
- Common ancestor:
  `eb4f6bfeb40a9d1444d3fb1d17c841601ca29a76`
- Unified branch: `exp/icra27-post-gate3c-5080`
- Final unified commit: recorded in the final push report after the metadata
  commit below; the scientific parent and all cherry-pick identities are fixed
  above and in the integration audit.

All six RTX-only commits were selectively retained in chronological order.
The resulting cherry-pick identities and decisions are recorded in
`research/handoffs/post_gate3c_branch_integration_audit.md`. The only conflict
was `.gitignore`; all valid Gate-3A1, Gate-3B, and Gate-3C local-cache
exclusions were preserved. No scientific result or numerical artifact had a
content conflict.

## Frozen assets and independent replication

The RTX-local ACT checkpoint is verified:

- model: `/home/wjq/checkpoints/zeromidnight_act_libero_object/model.safetensors`,
  SHA256 `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`;
- config: `/home/wjq/checkpoints/zeromidnight_act_libero_object/config.json`,
  SHA256 `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d`.

The RTX-local dataset is the verified
`DorayakiLin/libero_object_25_08_23_lerobotv2.1` revision
`cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a`, with 454 episodes, 66,984 frames,
and the audited 20-Hz physical index contract. Its dataset provenance remains
separate from the Thor evidence archive.

The RTX Gate-3A1 cache is independent of the transferred Thor reference cache.
It contains 82 episodes, 6,151 validation queries, 6,143 test queries, and
logical shape `(12,294,100,7)`. Its content-tree SHA256 is
`7e14e1f341bc2425cb3304cc3f35b0075184b0b1f33225e2dcf05cfe67e50f65`. The
important ordering reproduced: newest-age exponential `<` tuned CogACT
`approximately` control-semantic `<` newest-only. It is accepted as an
independent replication, not byte-equivalent to the Thor reference cache.

## Read-only relocated-log validation

The historical validators hard-code Thor absolute rollout roots. They were not
edited and no historical output was overwritten. A relocation-aware read-only
wrapper instead mapped each committed manifest path to the archival root,
rechecked every listed log SHA256, parsed every compressed episode, checked
episode step/query counts, and recomputed the content-tree digest:

| Gate | Episodes | Environment steps | Policy queries | Result |
|---|---:|---:|---:|---|
| Gate-3A2 | 400 | 85,942 | 85,942 | PASS |
| Gate-3B | 400 | 88,171 | 88,171 | PASS |
| Gate-3C | 700 | 145,272 | 145,272 | PASS |

The committed Gate-3C compact validation output also confirms 700 unique cells,
one query per surviving step, exact source age 20, exact fixed-source formulas,
identical first 20 A/B/C actions, finite seven-dimensional actions, and no
temporal ensemble or smoothing.

## Final paper source status

The active manuscript skeleton and plan now contain the validated Gate-3C
primary result. The confirmatory primary cohort is tasks 1--9 with 14 states
per task:

- FO20: `80/126 = 63.5%`;
- newest: `53/126 = 42.1%`;
- full old20: `55/126 = 43.7%`;
- age exponential: `62/126 = 49.2%`;
- CogACT: `59/126 = 46.8%`.

FO20 minus newest, full old20, age exponential, and CogACT are respectively
`+.2143`, `+.1984`, `+.1429`, and `+.1667`; all registered stability criteria
are positive. Gate-3B's FO result remains explicitly exploratory and is not
relabelled as confirmation.

The Gate-3C Figure 4 interface contains the validated five success rates,
paired/task-cluster confidence intervals, and nine primary task rows. No
Gate-3C placeholders remain in the active skeleton or interface. The paper
guardrails exclude universal arm/gripper laws, dynamic horizon claims, d=20
optimality, proven non-Markovian causation, and cross-policy or real-robot
generalization.

## Tests and non-experiment checks

The Gate-3C figure interface passed JSON validation, interface validation, and
PDF rendering from the committed plotting script. The targeted repository
suite passed with the repository audit-tools path enabled:

```text
28 passed in 6.11s
tests/test_gate3c_temporal_reuse.py
tests/test_gate3b_composition.py
tests/test_gate3a2_temporal_aggregation.py
```

No rollout, policy evaluation, or historical episode regeneration was run
during this handoff.

## Explicit machine-role declaration

> RTX 5080 is the primary simulation and manuscript machine after Gate-3C.
> Thor is retained for historical evidence, lightweight validation, and
> hardware-specific work.

No future large simulation experiment is launched by this handoff.

# Gate-4A2 Spatial official-state audit

Audit time: 2026-08-24T21:03:14+08:00

Status: **FROZEN OUTCOME-BLIND BEFORE PREREGISTRATION**

No Spatial task-success outcome was generated or inspected. Vanilla LIBERO
Spatial tasks 0–9 each expose exactly the official state-ID set `0..49`; the
common valid candidate set is therefore:

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
```

Each task's loaded state array has shape `(50,92)` and dtype `float64`.

| Task | Official init-state file SHA256 |
|---:|---|
| 0 | `cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a` |
| 1 | `9b6927b6ef03460ab055141e1a1154ad85e5e290058d8ed3a8a314aff911f864` |
| 2 | `27d5b81d739f969e40c72b6c88699e57a91ea61209946b21c593aae5990689cb` |
| 3 | `0627f5f5ce3ef23be546571012be8ef603d93bcb4032bc80feb34937ba580140` |
| 4 | `deef3df529e0bb57658b973bbb3baf02b02922db065c025a73ebfcd11eb54d46` |
| 5 | `2d2c50b7dcbf861fffa94d576f217ed34153e57487010d7a109a220d4d97189b` |
| 6 | `ab86fb7b5942dc0fec762eae57e72bf636106b1c10ad8ce57227723d8eb9a81b` |
| 7 | `ea9cba2814bc166f3682f3f5a97329467839bcf584069cd419191b6c17adae25` |
| 8 | `f43df2d80042130ac8770590d83f1a1e6fd4df7479aebc392aa2ed6e3bd0b83e` |
| 9 | `08b0ac928dcc524a66d963851a6a4eece9aeb782b38466006e403a70e3287867` |

The frozen selection command is equivalent to:

```python
rng = numpy.random.default_rng(20260825)
rng.choice(common_ids, 10, replace=False)
```

The preserved RNG result is:

```text
[40, 15, 13, 47, 37, 24, 19, 1, 31, 21]
```

The same selected IDs, sorted for reporting and used for every task, are:

```text
[1, 13, 15, 19, 21, 24, 31, 37, 40, 47]
```

The schedule and rollout provenance additionally freeze SHA256 identities for
each of the 100 selected task-state vectors.

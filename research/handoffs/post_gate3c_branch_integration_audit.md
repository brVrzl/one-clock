# Post-Gate-3C branch integration audit

Integration date: 2026-08-24. Target branch: `exp/icra27-post-gate3c-5080`.

## Git ancestry

- Common ancestor: `eb4f6bfeb40a9d1444d3fb1d17c841601ca29a76`
- Gate-3C source head: `3b9f1209df7266160c47453e8ee66a142ea8688c`
- RTX source head: `279bcab9415e46f72c8bd2b89c3adbee773c3db6`
- New branch was created directly from the Gate-3C source head.
- `git cherry` and `git range-diff` showed four Gate-3C-line commits and six
  RTX-only commits after the common ancestor. The RTX commits were not merged
  as a whole branch.

## RTX-only commit decisions

| RTX source commit | Decision | Resulting cherry-pick | Scope |
|---|---|---|---|
| `d9cac3ab69bd1a6d93608ebbd8311134483bc2ac` | KEEP | `98fcc8a` | Frozen offline protocol and cache-directory ignore rule |
| `b6f37e494caa19c772f0da3614474a94a83230f8` | KEEP | `89c6396` | RTX Gate-3A1 cache verification and local evidence entry |
| `194c4fc8230c0ad823601ffc712b900a789d6030` | KEEP | `5d1d2bc` | Frozen RTX cross-generation offline audit and outputs |
| `a8d49834650e17ca9cd6d413a7f64d0c5387fe4c` | KEEP | `ef08fa7` | Deterministic normalized audit CSV outputs |
| `7ef04683bad87f47ac1548bb0af7a67d3fa25e6d` | KEEP | `6f8bdf7` | Conditional paper infrastructure used as the parent of the final paper update |
| `279bcab9415e46f72c8bd2b89c3adbee773c3db6` | KEEP | `925191d` | Gate-3B directional analysis, related work, figures, and manuscript scaffold |

No RTX-only commit was skipped. The last paper commit superseded portions of
the earlier conditional scaffold in-tree, but its original identity and
chronological parent are retained through the two cherry-picks.

## Conflict resolution

The first cherry-pick conflicted only in `.gitignore`. The resolved file keeps
all valid local evidence exclusions from both histories:

```text
experiments/gate3a1_dense_temporal_cache/
experiments/gate3b_cross_generation_composition/
experiments/gate3c_asymmetric_temporal_reuse/
```

No scientific result, manifest, or numerical output had a content conflict.
The local RTX Gate-3A1 cache remains separate from the transferred Thor
reference cache.

## Resulting history

The six RTX commits are represented by the cherry-pick commits listed above.
Gate-3C remains the scientific parent and authoritative result source. The
active branch now contains both the Gate-3C scientific line and the RTX
offline/cache/paper artifacts without rewriting historical provenance.

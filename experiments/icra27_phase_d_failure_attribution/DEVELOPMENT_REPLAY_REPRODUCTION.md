# Phase-D development replay reproduction audit

Audit timestamp: `2026-09-04T17:44:14+08:00`

Status: `PASS`

The first bulk implementation reused one LIBERO environment within each task.
Although restored MuJoCo flat states were bit-identical and commands matched,
9 of 248 terminal outcomes differed from their source JSONs and 4 episode lengths
differed. The original Track-A runner creates a new environment for every cell;
the discrepancy was therefore diagnosed as hidden controller or wrapper state
that is not represented in the flat simulator state.

The invalid reused-environment batch was excluded from analysis and retained
under `diagnostic_archive/reused_env_bulk/`. The replay utility was changed to
construct a fresh environment for every source episode and every hybrid replay.
A targeted known mismatch then reproduced exactly, after which all 248 selected
development source episodes were replayed again.

Corrected full-cohort audit:

- selected source episodes: 248;
- exact restored initial flat states: 248/248;
- terminal success mismatches: 0;
- episode-length mismatches: 0;
- replay-command mismatches: 0.

Component-swap replay is permitted only after this full source-reproduction gate
passes. Phase-1 outcomes remained sealed throughout diagnosis and correction.

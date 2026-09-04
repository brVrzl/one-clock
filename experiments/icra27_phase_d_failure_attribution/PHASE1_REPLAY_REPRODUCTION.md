# Phase-D Phase-1 replay reproduction audit

Status: `PASS`

All 288 protocol-selected Phase-1 source episodes were replayed open-loop in a
fresh LIBERO environment per episode. The selection contains every H4 baseline
failure and both members of every H4 versus ARM4_GRIP32 discordant block, with
deduplication across roles.

- selected source episodes: 288;
- exact restored initial flat states: 288/288;
- terminal success mismatches: 0;
- episode-length mismatches: 0;
- replay-command mismatches: 0.

The Phase-1 component-swap replay gate therefore passed. No ACT inference or
continuation beyond recorded common support was used.

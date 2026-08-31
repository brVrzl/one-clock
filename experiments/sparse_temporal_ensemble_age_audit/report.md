# Sparse temporal-ensemble age audit: interim report

The historical sparse executor has no same-target indexing code bug. Its executed variant is precisely a candidate-index ensemble, which becomes nearly uniform at sparse cadence.

The existing ACT task-10 hard/TE shard is not strictly paired. The cause is repeated hard resets resampling static fixture model positions that are absent from the flattened MuJoCo state. A fresh, identically seeded environment per condition/state repairs initialization: the formal three-state audit produced exact equality for raw state, both cameras, processed inputs, initial chunk, and the complete t=0--15 action/state prefix.

The pooled 38/40 prior fixed-horizon references used three different tasks, so they do not establish drift of the current 34/40 and 33/40 panel. The one shared task does show the reset contamination described above.

No new ACT h16 result is reported because the requested physical-age definition is internally inconsistent with the mandatory h1-equivalence test. Positive-coefficient validated ACT favors the oldest prediction; `exp(-0.01*(q_newest-q))` favors the newest. Running either without resolving that distinction would not settle the stated semantic question.

Status: **blocked before outcome rollout by weight-orientation specification conflict**.

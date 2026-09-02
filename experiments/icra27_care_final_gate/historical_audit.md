# Historical M2 audit

The fallback, overnight, and discriminator tips all resolve `experiments/bounded_group_requery_dev/requery_policy.py` to the same Git blob: `a00528eb41c53c1dcd844f356681196f7bf4066e`.

M2 is exactly `intent(g)=+1` for `g>=0`, otherwise `-1`; it selects the first transition index `k` in `[4,16)`, and uses 16 if none exists. `MIN_HORIZON=4` and `MAX_HORIZON=16`. No magnitude threshold, delta, EMA, hysteresis, learned gate, force signal, or changed bound is present.

The completed development panel is M0 hard16 32/40, M1 arm-phase 30/40, M2 gripper-event 35/40, and M3 combined 31/40. M2 versus M0 discordance is 3:0. M2 used 532 queries over 6844 environment steps (query rate 0.077732320281) with mean execution horizon 13.304511278195.

Frozen M2 horizon histogram: 4:29, 5:25, 6:12, 7:18, 8:22, 9:16, 10:16, 11:10, 12:10, 13:9, 14:10, 15:8, 16:347.

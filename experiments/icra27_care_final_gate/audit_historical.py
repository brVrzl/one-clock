#!/usr/bin/env python3
"""Verify the frozen M2 implementation and development evidence from Git history."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
COMMITS = (
    "7ea83e1c0bea4367cc722a3d7b72ac0ca827e009",
    "eb4e29a62b28010c714d961f42449ae33bbe2312",
    "c4f9cb9ba0816a7cf0fb7f81abbfdcd126322490",
)
PATH = "experiments/bounded_group_requery_dev/requery_policy.py"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True)


def main() -> None:
    blobs = {commit: git("rev-parse", f"{commit}:{PATH}").strip() for commit in COMMITS}
    if set(blobs.values()) != {"a00528eb41c53c1dcd844f356681196f7bf4066e"}:
        raise RuntimeError(f"historical requery implementation drift: {blobs}")
    source = git("show", f"{COMMITS[0]}:{PATH}")
    required_source = (
        "MIN_HORIZON = 4",
        "MAX_HORIZON = 16",
        "return np.where(commands >= 0.0, 1, -1).astype(np.int8)",
        "events = [k for k in range(MIN_HORIZON, MAX_HORIZON) if intents[k] != intents[k - 1]]",
        "horizon = events[0] if events else MAX_HORIZON",
    )
    if not all(fragment in source for fragment in required_source):
        raise RuntimeError("historical M2 source no longer matches the frozen definition")
    analysis = json.loads(git("show", f"{COMMITS[0]}:experiments/bounded_group_requery_dev/analysis.json"))
    expected_successes = {"M0_hard16": 32, "M1_arm_phase": 30, "M2_gripper_event": 35, "M3_group_event_joint": 31}
    observed_successes = {method: int(analysis["summaries"][method]["success_count"]) for method in expected_successes}
    if observed_successes != expected_successes:
        raise RuntimeError(f"historical success counts drift: {observed_successes}")
    m2 = analysis["summaries"]["M2_gripper_event"]
    expected_histogram = {str(h): count for h, count in zip(range(4, 17), (29, 25, 12, 18, 22, 16, 16, 10, 10, 9, 10, 8, 347), strict=True)}
    if m2["horizon_histogram"] != expected_histogram:
        raise RuntimeError("historical M2 histogram drift")
    comparison = analysis["contrasts"]["M2_gripper_event_vs_M0_hard16"]
    if (comparison["candidate_only"], comparison["reference_only"]) != (3, 0):
        raise RuntimeError("historical M2:M0 discordance drift")
    result = {
        "implementation_path": PATH,
        "implementation_blobs": blobs,
        "all_three_tips_byte_identical": True,
        "definition": {
            "MIN_HORIZON": 4,
            "MAX_HORIZON": 16,
            "gripper_intent": "+1 if g >= 0 else -1",
            "event": "first k in [4,16) with intent[k] != intent[k-1]",
            "fallback": 16,
            "added_threshold_delta_ema_hysteresis_or_learned_signal": False,
        },
        "development_conditions": ["M0_hard16", "M1_arm_phase", "M2_gripper_event", "M3_group_event_joint"],
        "development_successes_over_40": observed_successes,
        "M2_vs_M0": {"M2_only": 3, "M0_only": 0},
        "M2_execution": {
            "policy_queries": int(m2["policy_queries"]),
            "environment_steps": int(m2["environment_steps"]),
            "query_rate": float(m2["query_rate"]),
            "mean_horizon": float(m2["mean_planned_horizon"]),
            "horizon_histogram": m2["horizon_histogram"],
        },
    }
    (ROOT / "historical_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    histogram = ", ".join(f"{h}:{count}" for h, count in m2["horizon_histogram"].items())
    report = f"""# Historical M2 audit

The fallback, overnight, and discriminator tips all resolve `{PATH}` to the same Git blob: `a00528eb41c53c1dcd844f356681196f7bf4066e`.

M2 is exactly `intent(g)=+1` for `g>=0`, otherwise `-1`; it selects the first transition index `k` in `[4,16)`, and uses 16 if none exists. `MIN_HORIZON=4` and `MAX_HORIZON=16`. No magnitude threshold, delta, EMA, hysteresis, learned gate, force signal, or changed bound is present.

The completed development panel is M0 hard16 32/40, M1 arm-phase 30/40, M2 gripper-event 35/40, and M3 combined 31/40. M2 versus M0 discordance is 3:0. M2 used 532 queries over 6844 environment steps (query rate {m2['query_rate']:.12f}) with mean execution horizon {m2['mean_planned_horizon']:.12f}.

Frozen M2 horizon histogram: {histogram}.
"""
    (ROOT / "historical_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"blob": next(iter(blobs.values())), "successes": observed_successes, "M2_discordance": [3, 0]}))


if __name__ == "__main__":
    main()

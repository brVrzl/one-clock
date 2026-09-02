#!/usr/bin/env python3
"""Frozen execution-horizon rules for the final CARE method gate."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
BOUNDED_REQUERY_ROOT = REPO_ROOT / "experiments" / "bounded_group_requery_dev"
sys.path.insert(0, str(BOUNDED_REQUERY_ROOT))

# Import the historical implementation rather than reproducing or modifying M2.
from requery_policy import MAX_HORIZON, MIN_HORIZON, choose_horizon  # noqa: E402


M2_DEVELOPMENT_HISTOGRAM = {
    4: 29,
    5: 25,
    6: 12,
    7: 18,
    8: 22,
    9: 16,
    10: 16,
    11: 10,
    12: 10,
    13: 9,
    14: 10,
    15: 8,
    16: 347,
}
M2_DEVELOPMENT_QUERY_COUNT = 532
assert MIN_HORIZON == 4
assert MAX_HORIZON == 16
assert sum(M2_DEVELOPMENT_HISTOGRAM.values()) == M2_DEVELOPMENT_QUERY_COUNT


def shuffled_horizon(cell: dict[str, Any], query_index: int) -> tuple[int, dict[str, Any]]:
    """Draw once from the frozen M2 development marginal with stable identity."""

    key = (
        "icra27-care-final-gate|SHUFFLED_TRIGGER|"
        f"task={int(cell['task_id'])}|state={int(cell['state_id'])}|"
        f"environment_seed={int(cell['environment_seed'])}|query_index={int(query_index)}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big") % M2_DEVELOPMENT_QUERY_COUNT
    cumulative = 0
    for horizon, weight in M2_DEVELOPMENT_HISTOGRAM.items():
        cumulative += weight
        if draw < cumulative:
            return horizon, {
                "trigger_reason": "frozen_shuffled_development_marginal",
                "stable_rng_key": key,
                "sha256_first_8_bytes_hex": digest[:8].hex(),
                "integer_draw_mod_532": draw,
                "h_exec": horizon,
            }
    raise AssertionError("unreachable shuffled-horizon draw")


def gate_horizon(
    method: str,
    chunk: np.ndarray,
    cell: dict[str, Any],
    query_index: int,
) -> tuple[int, dict[str, Any]]:
    """Return the frozen coherent execution horizon for one Gate M query."""

    if method == "M0_HARD16":
        return choose_horizon("M0_hard16", chunk)
    if method == "M2_GRIPPER_EVENT":
        return choose_horizon("M2_gripper_event", chunk)
    if method == "FIXED_H13":
        return 13, {"trigger_reason": "fixed_h13", "h_exec": 13}
    if method == "SHUFFLED_TRIGGER":
        return shuffled_horizon(cell, query_index)
    raise ValueError(f"unknown Gate M method {method!r}")


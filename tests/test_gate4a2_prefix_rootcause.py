from __future__ import annotations

import gzip
import json

import numpy as np

from research.audit_tools.gate4a2_prefix_rootcause import (
    assert_outcome_sealed,
    compare_layered_repeats,
    read_historical_prefix,
)


def test_sealed_historical_reader_never_returns_outcome_keys(tmp_path):
    path = tmp_path / "trace.json.gz"
    payload = {
        "run": {
            "task_id": 4,
            "state_id": 1,
            "method": "A_NEWEST",
            "episode_seed": 340401,
        },
        "summary": {
            "initial_state_vector_sha256": "registered-state",
            "success": True,
            "failure_category": "forbidden",
        },
        "steps": [
            {
                "step": step,
                "fresh_action": [0.0] * 7,
                "action": [0.0] * 7,
                "reward": 1.0,
                "is_success": True,
                "terminated": True,
                "truncated": False,
            }
            for step in range(20)
        ],
        "success": True,
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    result = read_historical_prefix(path)

    assert result["task_id"] == 4
    assert result["initial_state_vector_sha256"] == "registered-state"
    assert len(result["steps"]) == 20
    assert_outcome_sealed(result)
    serialized = json.dumps(
        result, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value
    )
    for forbidden in ("success", "reward", "failure_category", "terminated", "truncated"):
        assert f'"{forbidden}"' not in serialized


def test_layer_order_uses_first_time_then_earliest_layer():
    def repeat(image_delta: float, state_delta: float):
        records = []
        for step in range(20):
            records.append(
                {
                    "L1": {"state": np.asarray([state_delta if step >= 9 else 0.0])},
                    "L2": {"image": np.asarray([image_delta if step >= 8 else 0.0])},
                    "L3": {"input": np.asarray([image_delta if step >= 8 else 0.0])},
                    "L4": {"chunk": np.asarray([image_delta if step >= 8 else 0.0])},
                    "L5": {"chunk": np.asarray([image_delta if step >= 8 else 0.0])},
                    "L6": {"action": np.asarray([image_delta if step >= 8 else 0.0])},
                }
            )
        return records

    comparison = compare_layered_repeats([repeat(0.0, 0.0), repeat(1.0, 1.0)])

    assert comparison["first_divergent_step"] == 8
    assert comparison["earliest_divergent_layer"] == "L2"
    assert comparison["L1"]["first_divergent_step"] == 9

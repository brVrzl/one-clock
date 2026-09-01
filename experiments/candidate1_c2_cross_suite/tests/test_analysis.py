from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from analyze import (  # noqa: E402
    OUTCOME_SENTENCES,
    classify_outcome,
    frozen_regression,
    load_frozen_cross_suite_outcomes,
    load_protocol,
)


PROTOCOL_PATH = EXPERIMENT_ROOT / "protocol.json"


def test_outcome_a_mapping() -> None:
    assert classify_outcome([-0.30, -0.01], [0.01, 0.40]) == "OUTCOME_A"


def test_outcome_b_mapping() -> None:
    assert classify_outcome([-0.10, 0.05], [0.01, 0.40]) == "OUTCOME_B"


def test_outcome_c_mapping() -> None:
    assert classify_outcome([-0.10, 0.05], [-0.01, 0.40]) == "OUTCOME_C"


def test_non_a_non_b_edge_falls_to_c() -> None:
    assert classify_outcome([0.01, 0.20], [0.01, 0.40]) == "OUTCOME_C"


def test_frozen_claim_sentences_are_verbatim() -> None:
    assert OUTCOME_SENTENCES == {
        "OUTCOME_A": "Across the Object development cohort and the frozen cross-suite confirmation cohort, committing only the arm to the fixed h16 schedule while keeping the gripper fully reactive underperforms fully reactive execution, whereas coherent fixed-h16 execution performs substantially better.",
        "OUTCOME_B": "On the frozen cross-suite confirmation cohort, arm commitment alone does not reproduce the benefit of coherent fixed-h16 execution, although it does not reliably underperform fully reactive execution as it did on the Object development cohort.",
        "OUTCOME_C": "The Object executor decomposition does not transfer to the frozen cross-suite confirmation cohort; it remains a bounded development-setting observation, and we do not claim that arm commitment alone fails to explain the fixed-h16 gain in general.",
    }


def test_frozen_fresh_and_hard_reference_join_is_exactly_140_each() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    _, join = load_frozen_cross_suite_outcomes(protocol, ("FRESH", "HARD_H16"))
    assert join == {
        "FRESH": {
            "unique_blocks": 140,
            "expected_blocks": 140,
            "missing_blocks": 0,
            "duplicate_blocks": 0,
            "extra_blocks": 0,
        },
        "HARD_H16": {
            "unique_blocks": 140,
            "expected_blocks": 140,
            "missing_blocks": 0,
            "duplicate_blocks": 0,
            "extra_blocks": 0,
        },
    }


def test_frozen_fo20_vs_reverse20_regression() -> None:
    result = frozen_regression(load_protocol(PROTOCOL_PATH))
    assert result["status"] == "PASS"
    assert np.isclose(result["success_delta_percentage_points"], 32.142857142857146)
    assert result["first_only_wins"] == 48
    assert result["second_only_wins"] == 3
    assert np.isclose(result["exact_two_sided_mcnemar_p"], 1.9674928353197174e-11)
    np.testing.assert_allclose(result["paired_bootstrap_ci"], [0.2357142857, 0.4071428571])
    np.testing.assert_allclose(result["task_cluster_bootstrap_ci"], [0.2142857143, 0.4428571429])

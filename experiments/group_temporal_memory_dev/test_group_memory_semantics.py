#!/usr/bin/env python3
"""CPU-only semantic tests for the group-memory development ladder.

These tests deliberately exercise only the candidate buffer and pure NumPy
operators.  They do not import a policy, simulator, GPU runtime, outcome file,
or intervention result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "sparse_temporal_ensemble_dev"))

from group_memory_common import compose_method  # noqa: E402
from freeze_h_temp import DEV_TASKS, freeze  # noqa: E402
from group_memory_operators import (  # noqa: E402
    ARM_DIM,
    m0_hard,
    m2_shared_cogact,
    m3_group_cogact,
    shared_temporal_prior,
)
from sparse_executor import SparseExecutor  # noqa: E402


def chunk(source: int, horizon: int = 100) -> np.ndarray:
    return np.asarray(
        [[1000.0 * source + 10.0 * offset + dim for dim in range(7)] for offset in range(horizon)],
        dtype=np.float64,
    )


def test_same_target_alignment_and_group_slicing() -> None:
    executor = SparseExecutor(cadence=16, prediction_horizon=100, mode="hard")
    result = None
    for target in range(33):
        result = executor.step(target, lambda target=target: chunk(target))
    assert result is not None
    assert result.candidates.source_query_steps.tolist() == [0, 16, 32]
    assert result.candidates.offsets.tolist() == [32, 16, 0]
    for source, offset, row in zip(
        result.candidates.source_query_steps,
        result.candidates.offsets,
        result.candidates.actions,
        strict=True,
    ):
        assert int(source) + int(offset) == 32
        np.testing.assert_allclose(row, chunk(int(source))[int(offset)])
    assert ARM_DIM == 6
    assert result.candidates.actions[:, :6].shape == (3, 6)
    assert result.candidates.actions[:, 6:].shape == (3, 1)


def test_query_schedule_and_delay_masking() -> None:
    executor = SparseExecutor(cadence=16, prediction_horizon=50, mode="hard")
    calls: list[int] = []
    counts: dict[int, int] = {}
    for target in range(81):
        result = executor.step(target, lambda target=target: calls.append(target) or chunk(target, 50))
        counts[target] = result.candidate_count
        assert all(age == target - source for age, source in zip(result.candidates.ages, result.candidates.source_query_steps, strict=True))
    assert calls == [0, 16, 32, 48, 64, 80]
    assert counts[48] == 4
    # q=0 is valid at t=49 but is excluded at t=50 by the strict H_pred rule.
    assert 0 in executor.same_target_candidates(49).source_query_steps
    assert 0 not in executor.same_target_candidates(50).source_query_steps


def test_fresh_one_candidate_identity_and_group_weight_normalization() -> None:
    candidates = np.asarray([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, -1.0]])
    ages = np.asarray([0.0])
    for method in ("M0_h16", "M1_shared_te_h16", "M2_shared_cogact_h16", "M3_group_cogact_h16"):
        output, diagnostics = compose_method(
            method,
            type("Candidates", (), {"actions": candidates, "ages": ages})(),
            kernel_name="physical_age_te",
        )
        np.testing.assert_array_equal(output, candidates[0])
        np.testing.assert_allclose(diagnostics["arm_weights"], [1.0])
        np.testing.assert_allclose(diagnostics["gripper_weights"], [1.0])


def test_m2_shared_weight_invariant_and_m3_group_split() -> None:
    candidates = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ages = np.asarray([32.0, 16.0, 0.0])
    m2_output, m2_diag = m2_shared_cogact(candidates, ages, kernel_name="physical_age_te")
    assert np.allclose(m2_diag["shared_weights"], m2_diag["arm_weights"])
    assert np.allclose(m2_diag["shared_weights"], m2_diag["gripper_weights"])
    m3_output, m3_diag = m3_group_cogact(candidates, ages, kernel_name="physical_age_te")
    assert np.isclose(m3_diag["arm_weights"].sum(), 1.0)
    assert np.isclose(m3_diag["gripper_weights"].sum(), 1.0)
    assert not np.allclose(m3_diag["arm_weights"], m3_diag["gripper_weights"])
    assert not np.array_equal(m2_output, m3_output)


def test_m3_reduces_to_m2_when_compatibility_is_identical() -> None:
    candidates = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    ages = np.asarray([16.0, 0.0])
    # The arm vectors and scalar gripper intents are intentionally identical
    # in compatibility across candidates, so both independent softmaxes equal
    # the shared whole-action softmax.
    shared, _ = m2_shared_cogact(candidates, ages, kernel_name="physical_age_te")
    grouped, grouped_diag = m3_group_cogact(candidates, ages, kernel_name="physical_age_te")
    np.testing.assert_allclose(grouped, shared, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(grouped_diag["arm_weights"], grouped_diag["gripper_weights"], atol=1e-12)


def test_shared_prior_is_explicit_and_no_h_temp_or_outcome_input() -> None:
    ages = np.asarray([32.0, 16.0, 0.0])
    physical = shared_temporal_prior(ages, kernel_name="physical_age_te")
    dense = shared_temporal_prior(ages, kernel_name="dense_equivalent_te")
    indexed = shared_temporal_prior(ages, kernel_name="candidate_index_te")
    assert np.isclose(physical.sum(), 1.0)
    assert np.isclose(dense.sum(), 1.0)
    assert np.isclose(indexed.sum(), 1.0)
    assert physical[-1] > physical[0]
    np.testing.assert_allclose(dense / dense[0], np.exp(-0.01 * np.asarray([0.0, 16.0, 32.0])))
    assert indexed[0] > indexed[-1]
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert protocol["offline_prior"]["constraint"] == "H_temp is analysis-only and is never used by the executor."
    source = (ROOT / "group_memory_common.py").read_text()
    assert "h_temp" not in source.lower()


def test_m4_refuses_missing_online_reliability() -> None:
    candidates = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    ages = np.asarray([0.0])
    try:
        compose_method(
            "M4_anchored_group_reliability_h16",
            type("Candidates", (), {"actions": candidates, "ages": ages})(),
            kernel_name="physical_age_te",
        )
    except RuntimeError as error:
        assert str(error) == "UNAVAILABLE_RELIABILITY_INTERFACE"
    else:
        raise AssertionError("M4 must not fabricate a reliability interface")


def test_h_temp_freeze_is_outcome_blind_and_development_only(tmp_path: Path) -> None:
    output = tmp_path / "h_temp.json"
    frozen = freeze(ROOT.parent / "group_temporal_memory_offline" / "h_temp_frozen.json", output)
    assert frozen["outcome_blind"] is True
    assert frozen["outcomes_loaded"] is False
    assert [row["task_key"] for row in frozen["task_values"]] == list(DEV_TASKS)
    assert all("success" not in row for row in frozen["task_values"])

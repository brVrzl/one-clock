"""Deterministic same-target tests for the two authorized fixed-clock conditions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
ASYM_ROOT = REPO_ROOT / "experiments" / "asymmetric_chunk_reuse_dev"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ASYM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from asymmetric_executor import C1, make_executor as make_c1_executor  # noqa: E402
from fixed_clock_executor import (  # noqa: E402
    ACTION_DIM,
    CHUNK_LENGTH,
    H16,
    H32,
    H32_COHERENT,
    TWO_CLOCK,
    make_executor,
)
from one_clock import ActionGroup, FixedChunkExecutor  # noqa: E402


def tagged_chunk(query_step: int) -> np.ndarray:
    return np.asarray(
        [
            [10000.0 * query_step + 100.0 * offset + dimension for dimension in range(ACTION_DIM)]
            for offset in range(CHUNK_LENGTH)
        ],
        dtype=np.float64,
    )


def run_new(method: str, steps: int) -> dict:
    executor = make_executor(method)
    query_steps: list[int] = []
    rows = []
    for t in range(steps):
        decision = executor.step(lambda t=t: tagged_chunk(t))
        if decision.policy_query:
            query_steps.append(t)
        source_q = {group: query_steps[chunk_id] for group, chunk_id in decision.source_chunk_ids.items()}
        rows.append({"t": t, "decision": decision, "source_q": source_q})
    return {"query_steps": query_steps, "rows": rows}


def run_h16(steps: int) -> dict:
    executor = FixedChunkExecutor.global_fixed(
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_LENGTH,
        horizon=H16,
        groups=(ActionGroup("arm", tuple(range(6)), H16), ActionGroup("gripper", (6,), H16)),
    )
    query_steps: list[int] = []
    rows = []
    for t in range(steps):
        decision = executor.step(lambda t=t: tagged_chunk(t))
        if decision.policy_query:
            query_steps.append(t)
        source_q = {group: query_steps[chunk_id] for group, chunk_id in decision.source_chunk_ids.items()}
        rows.append({"t": t, "decision": decision, "source_q": source_q})
    return {"query_steps": query_steps, "rows": rows}


def run_c1(steps: int) -> dict:
    executor = make_c1_executor(C1)
    rows = []
    queries = []
    for t in range(steps):
        result = executor.step(t, lambda t=t: tagged_chunk(t))
        if result.queried:
            queries.append(t)
        rows.append(result)
    return {"query_steps": queries, "rows": rows}


def test_chunk_size_exposes_offset_31() -> None:
    assert CHUNK_LENGTH == 100
    assert 31 < CHUNK_LENGTH
    two = run_new(TWO_CLOCK, 32)
    row = two["rows"][31]
    assert row["decision"].source_positions == {"arm": 15, "gripper": 31}


def test_h32_is_coherent_same_target_execution() -> None:
    trace = run_new(H32_COHERENT, 65)
    assert trace["query_steps"] == [0, 32, 64]
    for row in trace["rows"]:
        t = row["t"]
        q = H32 * (t // H32)
        assert row["source_q"] == {"arm": q, "gripper": q}
        assert row["decision"].source_positions == {"arm": t - q, "gripper": t - q}
        np.testing.assert_array_equal(row["decision"].action, tagged_chunk(q)[t - q])


def test_two_clock_has_independent_same_target_sources_and_h16_query_rate() -> None:
    trace = run_new(TWO_CLOCK, 65)
    assert trace["query_steps"] == [0, 16, 32, 48, 64]
    for row in trace["rows"]:
        t = row["t"]
        q_arm = H16 * (t // H16)
        q_grip = H32 * (t // H32)
        assert row["source_q"] == {"arm": q_arm, "gripper": q_grip}
        assert row["decision"].source_positions == {"arm": t - q_arm, "gripper": t - q_grip}
        expected = tagged_chunk(q_arm)[t - q_arm].copy()
        expected[6] = tagged_chunk(q_grip)[t - q_grip, 6]
        np.testing.assert_array_equal(row["decision"].action, expected)
    assert trace["rows"][16]["decision"].refreshed_groups == ("arm",)
    assert trace["rows"][32]["decision"].refreshed_groups == ("arm", "gripper")


def test_required_initial_prefixes_are_identical() -> None:
    h16 = run_h16(16)
    h32 = run_new(H32_COHERENT, 16)
    two = run_new(TWO_CLOCK, 16)
    for t in range(16):
        np.testing.assert_array_equal(h16["rows"][t]["decision"].action, h32["rows"][t]["decision"].action)
        np.testing.assert_array_equal(h16["rows"][t]["decision"].action, two["rows"][t]["decision"].action)


def test_c1_and_true_two_clock_first_differ_in_gripper_source_at_t32() -> None:
    c1 = run_c1(33)
    two = run_new(TWO_CLOCK, 33)
    assert c1["query_steps"] == two["query_steps"] == [0, 16, 32]
    for t in range(32):
        c1_row = c1["rows"][t]
        two_row = two["rows"][t]
        assert (c1_row.arm_source_q, c1_row.grip_source_q) == (
            two_row["source_q"]["arm"],
            two_row["source_q"]["gripper"],
        )
        assert (c1_row.arm_offset, c1_row.grip_offset) == (
            two_row["decision"].source_positions["arm"],
            two_row["decision"].source_positions["gripper"],
        )
        np.testing.assert_array_equal(c1_row.action, two_row["decision"].action)
    assert (c1["rows"][32].grip_source_q, c1["rows"][32].grip_offset) == (16, 16)
    assert (two["rows"][32]["source_q"]["gripper"], two["rows"][32]["decision"].source_positions["gripper"]) == (32, 0)

"""Minimum deterministic semantic tests for the matched-query executors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from one_clock import ActionGroup, FixedChunkExecutor  # noqa: E402


METHODS = ("global_8_8", "arm8_grip16", "arm16_grip8")


def tagged_chunk(query_step: int, chunk_size: int = 32) -> np.ndarray:
    return np.asarray(
        [[10000.0 * query_step + 100.0 * offset + dimension for dimension in range(7)] for offset in range(chunk_size)],
        dtype=np.float64,
    )


def make_executor(method: str) -> FixedChunkExecutor:
    if method == "global_8_8":
        return FixedChunkExecutor.global_fixed(
            action_dim=7,
            chunk_size=32,
            horizon=8,
            groups=(ActionGroup("arm", tuple(range(6)), 8), ActionGroup("gripper", (6,), 8)),
        )
    if method == "arm8_grip16":
        return FixedChunkExecutor.groupwise_fixed(
            action_dim=7,
            chunk_size=32,
            groups=(ActionGroup("arm", tuple(range(6)), 8), ActionGroup("gripper", (6,), 16)),
        )
    if method == "arm16_grip8":
        return FixedChunkExecutor.groupwise_fixed(
            action_dim=7,
            chunk_size=32,
            groups=(ActionGroup("arm", tuple(range(6)), 16), ActionGroup("gripper", (6,), 8)),
        )
    raise ValueError(method)


def run_trace(method: str, steps: int = 33) -> dict:
    executor = make_executor(method)
    queries: list[int] = []
    records = []
    for step in range(steps):
        decision = executor.step(lambda step=step: tagged_chunk(step))
        if decision.policy_query:
            queries.append(step)
        source_queries = {name: queries[chunk_id] for name, chunk_id in decision.source_chunk_ids.items()}
        records.append({"step": step, "decision": decision, "source_queries": source_queries})
    return {"query_steps": queries, "records": records}


def test_all_methods_share_the_fixed_query_schedule() -> None:
    traces = {method: run_trace(method) for method in METHODS}
    assert [trace["query_steps"] for trace in traces.values()] == [[0, 8, 16, 24, 32]] * 3


def test_global_8_8_reproduces_ordinary_fixed_h8() -> None:
    trace = run_trace("global_8_8")
    for row in trace["records"]:
        step = row["step"]
        source = (step // 8) * 8
        expected = tagged_chunk(source)[step - source]
        np.testing.assert_array_equal(row["decision"].action, expected)
        assert row["source_queries"] == {"arm": source, "gripper": source}
        assert row["decision"].source_ages == {"arm": step - source, "gripper": step - source}


def test_arm8_grip16_uses_new_arm_and_old_same_target_gripper() -> None:
    trace = run_trace("arm8_grip16")
    row = trace["records"][8]
    np.testing.assert_array_equal(row["decision"].action[:6], tagged_chunk(8)[0, :6])
    np.testing.assert_array_equal(row["decision"].action[6:], tagged_chunk(0)[8, 6:])
    assert row["source_queries"] == {"arm": 8, "gripper": 0}
    assert row["decision"].source_ages == {"arm": 0, "gripper": 8}
    # The old source predicts the current target; this is not a held scalar.
    assert row["decision"].action[6] != trace["records"][7]["decision"].action[6]
    row = trace["records"][16]
    assert row["source_queries"] == {"arm": 16, "gripper": 16}
    np.testing.assert_array_equal(row["decision"].action, tagged_chunk(16)[0])


def test_arm16_grip8_is_the_exact_reverse() -> None:
    trace = run_trace("arm16_grip8")
    row = trace["records"][8]
    np.testing.assert_array_equal(row["decision"].action[:6], tagged_chunk(0)[8, :6])
    np.testing.assert_array_equal(row["decision"].action[6:], tagged_chunk(8)[0, 6:])
    assert row["source_queries"] == {"arm": 0, "gripper": 8}
    assert row["decision"].source_ages == {"arm": 8, "gripper": 0}
    assert row["decision"].action[:6].tolist() != trace["records"][7]["decision"].action[:6].tolist()
    row = trace["records"][24]
    assert row["source_queries"] == {"arm": 16, "gripper": 24}
    np.testing.assert_array_equal(row["decision"].action[:6], tagged_chunk(16)[8, :6])
    np.testing.assert_array_equal(row["decision"].action[6:], tagged_chunk(24)[0, 6:])


def test_source_ids_ages_and_target_offsets_are_independent() -> None:
    for method in METHODS:
        trace = run_trace(method)
        for row in trace["records"]:
            decision = row["decision"]
            step = row["step"]
            for group in ("arm", "gripper"):
                source = row["source_queries"][group]
                age = decision.source_ages[group]
                position = decision.source_positions[group]
                assert source + age == step
                assert position == age
                assert decision.source_chunk_ids[group] == trace["query_steps"].index(source)
                assert decision.action[list(range(6)) if group == "arm" else [6]].tolist() == tagged_chunk(source)[position, list(range(6)) if group == "arm" else [6]].tolist()

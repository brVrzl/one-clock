from __future__ import annotations

import numpy as np
import pytest

from research.audit_tools.robotwin_temporal_reuse import (
    ACTION_GROUPS,
    CHUNK_LENGTH,
    GRIPPER_CONTROL_METHODS,
    METHODS,
    NOMINAL_SOURCE_AGE_TICKS,
    PHYSICAL_AGE_METHODS,
    PHYSICAL_SOURCE_AGE_SECONDS,
    RoboTwinGripperControlExecutor,
    RoboTwinPhysicalAgeExecutor,
    RoboTwinTemporalExecutor,
    native_act_aggregate,
    postprocess_action,
    select_physical_age_source,
)


def synthetic_chunks(count: int = 18) -> list[np.ndarray]:
    chunks = []
    for source_step in range(count):
        chunk = np.empty((50, 14), dtype=np.float64)
        for offset in range(50):
            chunk[offset] = 10000 * source_step + 100 * offset + np.arange(14)
        chunks.append(chunk)
    return chunks


def test_frozen_robotwin_action_group_contract() -> None:
    assert ACTION_GROUPS == {
        "left_arm": (0, 1, 2, 3, 4, 5),
        "left_gripper": (6,),
        "right_arm": (7, 8, 9, 10, 11, 12),
        "right_gripper": (13,),
    }


@pytest.mark.parametrize("method", METHODS)
def test_pre_offset_fallback_is_full_fresh(method: str) -> None:
    executor = RoboTwinTemporalExecutor(method)
    for step, chunk in enumerate(synthetic_chunks(NOMINAL_SOURCE_AGE_TICKS)):
        result = executor.update(step, chunk)
        np.testing.assert_array_equal(result.action, chunk[0].astype(np.float32))
        assert result.old_action is None
        assert set(result.group_source_ages.values()) == {0}


def test_same_current_target_indexing_and_composition() -> None:
    chunks = synthetic_chunks()
    results = {}
    for method in METHODS:
        executor = RoboTwinTemporalExecutor(method)
        for step, chunk in enumerate(chunks):
            result = executor.update(step, chunk)
        results[method] = result

    fresh = chunks[17][0]
    old_same_target = chunks[0][17]
    wrong_old_first_action = chunks[0][0]

    for result in results.values():
        assert result.target_step == 17
        assert result.fresh_source_step == 17
        assert result.fresh_chunk_offset == 0
        assert result.old_source_step == 0
        assert result.old_chunk_offset == 17
        np.testing.assert_array_equal(result.fresh_action, fresh)
        np.testing.assert_array_equal(result.old_action, old_same_target)
        assert not np.array_equal(result.old_action, wrong_old_first_action)

    np.testing.assert_array_equal(results["NEWEST"].action, fresh.astype(np.float32))
    np.testing.assert_array_equal(
        results["FULL_OLD_17"].action, old_same_target.astype(np.float32)
    )

    asymmetric = results["FO_17"]
    np.testing.assert_array_equal(asymmetric.action[:6], fresh[:6])
    assert asymmetric.action[6] == old_same_target[6]
    np.testing.assert_array_equal(asymmetric.action[7:13], fresh[7:13])
    assert asymmetric.action[13] == old_same_target[13]
    assert asymmetric.group_source_ages == {
        "left_arm": 0,
        "left_gripper": 17,
        "right_arm": 0,
        "right_gripper": 17,
    }
    assert asymmetric.group_chunk_offsets == asymmetric.group_source_ages


def test_invalid_query_stream_or_chunk_is_rejected() -> None:
    executor = RoboTwinTemporalExecutor("NEWEST")
    executor.update(0, synthetic_chunks(1)[0])
    with pytest.raises(ValueError, match="ordered query"):
        executor.update(2, synthetic_chunks(1)[0])
    with pytest.raises(ValueError, match="expected chunk shape"):
        RoboTwinTemporalExecutor("NEWEST").update(0, np.zeros((49, 14)))


def test_native_act_aggregation_matches_official_row_order() -> None:
    chunks = synthetic_chunks(18)
    target = 17
    expected_candidates = np.stack([chunks[source][target - source] for source in range(18)])
    weights = np.exp(-0.01 * np.arange(18))
    weights /= weights.sum()
    expected = np.sum(expected_candidates * weights[:, None], axis=0)
    np.testing.assert_array_equal(
        native_act_aggregate(dict(enumerate(chunks)), target), expected
    )


def test_postprocessing_is_deterministic_and_affine() -> None:
    normalized = np.arange(14, dtype=np.float64)
    mean = np.linspace(-1, 1, 14)
    std = np.linspace(0.1, 1.4, 14)
    first = postprocess_action(normalized, mean, std)
    second = postprocess_action(normalized, mean, std)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, normalized * std + mean)


def physical_results(
    timestamps: list[float],
) -> tuple[dict[str, object], list[np.ndarray]]:
    chunks = synthetic_chunks(len(timestamps))
    results = {}
    for method in PHYSICAL_AGE_METHODS:
        executor = RoboTwinPhysicalAgeExecutor(method)
        for step, (timestamp, chunk) in enumerate(zip(timestamps, chunks)):
            result = executor.update(step, chunk, query_time_seconds=timestamp)
        results[method] = result
    return results, chunks


def test_physical_age_selection_is_past_and_chunk_bounded() -> None:
    timestamps = {step: step * 0.1 for step in range(55)}
    selection = select_physical_age_source(timestamps, 54)
    assert selection is not None
    assert selection.old_source_step < selection.target_step
    assert selection.chunk_offset == selection.target_step - selection.old_source_step
    assert 0 < selection.chunk_offset < CHUNK_LENGTH
    assert selection.old_source_step == 44


def test_physical_age_same_target_indexing_and_composition() -> None:
    timestamps = [0.0, 0.4, 0.9, 1.4, 2.05]
    results, chunks = physical_results(timestamps)
    selection = select_physical_age_source(dict(enumerate(timestamps)), 4)
    assert selection is not None
    assert selection.old_source_step == 2
    assert selection.chunk_offset == 2
    assert selection.realized_source_age_seconds == pytest.approx(1.15)

    fresh = chunks[4][0]
    old_same_target = chunks[2][2]
    wrong_old_first_action = chunks[2][0]
    for result in results.values():
        assert result.old_source_step == selection.old_source_step
        assert result.old_chunk_offset == selection.chunk_offset
        np.testing.assert_array_equal(result.old_action, old_same_target)
        assert not np.array_equal(result.old_action, wrong_old_first_action)

    np.testing.assert_array_equal(results["NEWEST"].action, fresh.astype(np.float32))
    np.testing.assert_array_equal(
        results["FULL_OLD_1S"].action, old_same_target.astype(np.float32)
    )
    full_old = results["FULL_OLD_1S"]
    assert set(full_old.group_source_steps.values()) == {selection.old_source_step}
    assert set(full_old.group_chunk_offsets.values()) == {selection.chunk_offset}

    asymmetric = results["FO_1S"]
    np.testing.assert_array_equal(asymmetric.action[:6], fresh[:6])
    np.testing.assert_array_equal(asymmetric.action[7:13], fresh[7:13])
    assert asymmetric.action[6] == old_same_target[6]
    assert asymmetric.action[13] == old_same_target[13]
    assert asymmetric.group_source_steps == {
        "left_arm": 4,
        "left_gripper": 2,
        "right_arm": 4,
        "right_gripper": 2,
    }


def test_physical_age_warmup_is_newest_and_not_gripper_hold() -> None:
    executor = RoboTwinPhysicalAgeExecutor("FO_1S")
    chunks = synthetic_chunks(3)
    first = executor.update(0, chunks[0], query_time_seconds=0.0)
    second = executor.update(1, chunks[1], query_time_seconds=0.6)
    third = executor.update(2, chunks[2], query_time_seconds=1.2)
    np.testing.assert_array_equal(first.action, chunks[0][0].astype(np.float32))
    assert first.old_action is None
    assert second.action[6] == chunks[0][1, 6]
    assert third.action[6] == chunks[0][2, 6]
    assert third.action[6] != second.action[6]
    assert third.action[13] != second.action[13]


def test_physical_selection_uses_query_time_not_wall_clock_or_success() -> None:
    first_times = {0: 0.0, 1: 0.4, 2: 0.8, 3: 1.2, 4: 1.6}
    second_times = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 1.6}
    first = select_physical_age_source(first_times, 4)
    second = select_physical_age_source(second_times, 4)
    assert first is not None and second is not None
    assert first.old_source_step == 2
    assert second.old_source_step == 3
    with pytest.raises(TypeError):
        select_physical_age_source(  # type: ignore[call-arg]
            first_times, 4, task_success=True
        )


def test_physical_executor_rejects_non_simulator_timestamp_stream() -> None:
    executor = RoboTwinPhysicalAgeExecutor("FO_1S")
    chunks = synthetic_chunks(2)
    executor.update(0, chunks[0], query_time_seconds=1.0)
    with pytest.raises(ValueError, match="strictly increase"):
        executor.update(1, chunks[1], query_time_seconds=1.0)


def test_frozen_physical_target_age() -> None:
    assert PHYSICAL_SOURCE_AGE_SECONDS == 1.0


def test_gripper_hold_initializes_fresh_then_holds_executed_command() -> None:
    executor = RoboTwinGripperControlExecutor("GRIPPER_HOLD")
    first_fresh = np.arange(14, dtype=np.float64)
    second_fresh = first_fresh + 100.0
    first = executor.update(0, first_fresh, query_time_seconds=0.0)
    second = executor.update(1, second_fresh, query_time_seconds=0.4)
    np.testing.assert_array_equal(first.action, first_fresh.astype(np.float32))
    np.testing.assert_array_equal(second.action[:6], second_fresh[:6])
    np.testing.assert_array_equal(second.action[7:13], second_fresh[7:13])
    np.testing.assert_array_equal(second.action[[6, 13]], first.action[[6, 13]])
    np.testing.assert_array_equal(
        second.previous_executed_grippers, first.action[[6, 13]]
    )


def test_gripper_ema_uses_frozen_physical_time_constant() -> None:
    executor = RoboTwinGripperControlExecutor("GRIPPER_EMA_1S")
    first_fresh = np.zeros(14, dtype=np.float64)
    first_fresh[[6, 13]] = [0.2, 0.8]
    second_fresh = np.ones(14, dtype=np.float64)
    second_fresh[[6, 13]] = [0.9, 0.1]
    first = executor.update(0, first_fresh, query_time_seconds=2.0)
    second = executor.update(1, second_fresh, query_time_seconds=2.25)
    expected_alpha = np.exp(-0.25 / 1.0)
    expected_grippers = (
        expected_alpha * first_fresh[[6, 13]]
        + (1.0 - expected_alpha) * second_fresh[[6, 13]]
    )
    assert second.ema_alpha == pytest.approx(expected_alpha)
    np.testing.assert_allclose(second.action[[6, 13]], expected_grippers)
    np.testing.assert_array_equal(second.action[:6], second_fresh[:6])
    np.testing.assert_array_equal(second.action[7:13], second_fresh[7:13])
    np.testing.assert_array_equal(first.action, first_fresh.astype(np.float32))


@pytest.mark.parametrize("method", GRIPPER_CONTROL_METHODS)
def test_gripper_baselines_require_simulator_time_and_order(method: str) -> None:
    executor = RoboTwinGripperControlExecutor(method)
    fresh = np.zeros(14, dtype=np.float64)
    executor.update(0, fresh, query_time_seconds=1.0)
    with pytest.raises(ValueError, match="strictly increase"):
        executor.update(1, fresh, query_time_seconds=1.0)

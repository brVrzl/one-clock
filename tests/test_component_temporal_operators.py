from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "component_temporal_reuse"))

from temporal_operators import (  # noqa: E402
    act_temporal_weights,
    aggregate_components,
    aggregate_full_action,
    component_candidate_features,
    exponential_age_weights,
    one_hot_age,
    same_target_candidates,
    select_component_source,
    selected_component_action,
)


def tagged_chunk(source: int, horizon: int = 4) -> np.ndarray:
    return np.asarray(
        [[100.0 * source + offset + dimension for dimension in range(7)] for offset in range(horizon)]
    )


def test_same_target_semantics_are_source_plus_offset_equals_target() -> None:
    cache = [tagged_chunk(source) for source in range(5)]
    result = same_target_candidates(cache, target_step=4)
    assert result.source_steps.tolist() == [1, 2, 3, 4]
    assert result.ages.tolist() == [3, 2, 1, 0]
    np.testing.assert_array_equal(result.actions[:, 0], [103.0, 202.0, 301.0, 400.0])


def test_identical_group_weights_are_exactly_ordinary_aggregation() -> None:
    candidates = np.arange(28, dtype=np.float64).reshape(4, 7)
    weights = np.asarray([1.0, 2.0, 3.0, 4.0])
    ordinary = aggregate_full_action(candidates, weights)
    grouped = aggregate_components(candidates, {"arm": weights, "gripper": weights})
    np.testing.assert_array_equal(grouped, ordinary)


def test_one_hot_newest_is_exactly_fresh() -> None:
    candidates = np.arange(28, dtype=np.float64).reshape(4, 7)
    ages = np.asarray([3, 2, 1, 0])
    newest = one_hot_age(ages, 0)
    action = aggregate_components(candidates, {"arm": newest, "gripper": newest})
    np.testing.assert_array_equal(action, candidates[-1])


def test_one_hot_historical_reproduces_full_old_and_fo_interventions() -> None:
    candidates = np.arange(28, dtype=np.float64).reshape(4, 7)
    ages = np.asarray([3, 2, 1, 0])
    fresh = one_hot_age(ages, 0)
    old = one_hot_age(ages, 2)

    full_old = aggregate_components(candidates, {"arm": old, "gripper": old})
    np.testing.assert_array_equal(full_old, candidates[1])

    fo = aggregate_components(candidates, {"arm": fresh, "gripper": old})
    np.testing.assert_array_equal(fo[:6], candidates[-1, :6])
    np.testing.assert_array_equal(fo[6:], candidates[1, 6:])

    reverse = aggregate_components(candidates, {"arm": old, "gripper": fresh})
    np.testing.assert_array_equal(reverse[:6], candidates[1, :6])
    np.testing.assert_array_equal(reverse[6:], candidates[-1, 6:])


def test_act_and_physical_age_decay_have_explicit_opposite_positive_sign_semantics() -> None:
    ages = np.asarray([3, 2, 1, 0])
    act = act_temporal_weights(len(ages), coefficient=0.01)
    decay = exponential_age_weights(ages, beta=0.03)
    assert act[0] > act[-1]
    assert decay[0] < decay[-1]


def test_act_weights_match_pinned_lerobot_online_ensembler() -> None:
    import torch
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    generator = np.random.default_rng(20260827)
    chunks = generator.normal(size=(6, 4, 7)).astype(np.float32)
    upstream = ACTTemporalEnsembler(temporal_ensemble_coeff=0.01, chunk_size=4)
    history: list[np.ndarray] = []
    for target_step, chunk in enumerate(chunks):
        history.append(chunk)
        candidates = same_target_candidates(history, target_step).actions
        expected = aggregate_full_action(candidates, act_temporal_weights(len(candidates)))
        actual = upstream.update(torch.from_numpy(chunk[None]))[0].numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_selector_features_and_rules_are_outcome_blind() -> None:
    candidates = np.asarray(
        [
            [0.0] * 7,
            [1.0] * 7,
            [1.1] * 7,
        ]
    )
    ages = np.asarray([2, 1, 0])
    features = component_candidate_features(candidates, ages, slice(0, 6))
    assert set(features) == {
        "source_age",
        "fresh_disagreement_l2",
        "action_magnitude_l2",
        "distance_to_cached_centroid_l2",
        "mean_pairwise_disagreement_l2",
    }
    assert select_component_source(features, rule="newest") == 2
    assert select_component_source(features, rule="consensus_medoid") == 1
    selected = select_component_source(
        features,
        rule="oldest_within_fresh_disagreement",
        disagreement_threshold=0.3,
    )
    assert selected == 1

    action = selected_component_action(candidates, {"arm": selected, "gripper": 2})
    np.testing.assert_array_equal(action[:6], candidates[1, :6])
    np.testing.assert_array_equal(action[6:], candidates[2, 6:])

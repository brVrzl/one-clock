from __future__ import annotations

import unittest

import numpy as np

from experiments.temporal_reliability_training import (
    DEFAULT_LIBERO_GROUPS,
    FeatureConfig,
    FeatureEncoder,
    FrozenTrajectory,
    SplitConfig,
    TargetConfig,
    TemporalReliabilityDatasetBuilder,
    TemporalValidityTarget,
    MLPBaseline,
    evaluate_reliability,
    expected_calibration_error,
    reliability_curve,
    roc_auc,
    split_episode_ids,
)


def make_trajectory(episode_id: int = 0) -> FrozenTrajectory:
    chunks = {
        step: np.asarray(
            [
                [step + row + column / 10 for column in range(7)]
                for row in range(4)
            ],
            dtype=np.float32,
        )
        for step in range(5)
    }
    embeddings = np.asarray(
        [[step, step + 0.5, -step] for step in range(5)], dtype=np.float32
    )
    actions = np.asarray(
        [[step + column / 10 for column in range(7)] for step in range(5)],
        dtype=np.float32,
    )
    return FrozenTrajectory(
        episode_id=episode_id,
        task_id="task",
        policy_chunks=chunks,
        demonstrated_actions=actions,
        observation_embeddings=embeddings,
        source_steps=(0, 1),
    )


class DatasetAndFeatureTest(unittest.TestCase):
    def test_episode_split_is_disjoint_deterministic_and_complete(self) -> None:
        episodes = list(range(20))
        tasks = {episode: episode % 2 for episode in episodes}
        config = SplitConfig(seed=7)
        first = split_episode_ids(episodes, task_by_episode=tasks, config=config)
        second = split_episode_ids(episodes, task_by_episode=tasks, config=config)

        self.assertEqual(first, second)
        parts = [set(first.train), set(first.validation), set(first.test)]
        self.assertTrue(parts[0].isdisjoint(parts[1]))
        self.assertTrue(parts[0].isdisjoint(parts[2]))
        self.assertTrue(parts[1].isdisjoint(parts[2]))
        self.assertEqual(set.union(*parts), set(episodes))

    def test_builder_supports_arm_and_gripper_and_counts_examples(self) -> None:
        builder = TemporalReliabilityDatasetBuilder(
            groups=DEFAULT_LIBERO_GROUPS,
            offsets=(0, 1, 2),
            target_mode="fresh_policy",
        )
        examples = builder.build([make_trajectory()])

        self.assertEqual(len(examples), 2 * 3 * 2)
        self.assertEqual({example.group for example in examples}, {"arm", "gripper"})
        self.assertEqual(builder.last_build_summary["examples"], len(examples))

    def test_features_ignore_future_target_fields(self) -> None:
        builder = TemporalReliabilityDatasetBuilder(
            groups=DEFAULT_LIBERO_GROUPS,
            offsets=(1,),
            target_mode="fresh_policy",
        )
        example = builder.build([make_trajectory()])[0]
        encoder = FeatureEncoder(
            DEFAULT_LIBERO_GROUPS,
            FeatureConfig(observation_embedding_dim=3, max_offset=4),
        )
        before = encoder.encode(example)
        object.__setattr__(
            example,
            "future_policy_chunk",
            np.full_like(example.future_policy_chunk, 999.0),
        )
        object.__setattr__(
            example,
            "future_demonstrated_action",
            np.full_like(example.future_demonstrated_action, -999.0),
        )
        np.testing.assert_array_equal(before, encoder.encode(example))


class TargetTest(unittest.TestCase):
    def test_threshold_is_not_assumed_and_can_be_supplied_per_group(self) -> None:
        builder = TemporalReliabilityDatasetBuilder(
            groups=DEFAULT_LIBERO_GROUPS,
            offsets=(0,),
            target_mode="fresh_policy",
        )
        examples = builder.build([make_trajectory()])
        target = TemporalValidityTarget(
            DEFAULT_LIBERO_GROUPS,
            TargetConfig(mode="fresh_policy"),
        )
        without_threshold = target.generate(examples)
        self.assertIsNone(without_threshold.labels)
        with_threshold = target.generate(
            examples,
            threshold_by_group={"arm": 0.0, "gripper": 0.0},
        )
        self.assertIsNotNone(with_threshold.labels)
        np.testing.assert_array_equal(with_threshold.labels, np.ones(len(examples)))

    def test_demonstration_target_is_separate_from_fresh_policy_target(self) -> None:
        trajectory = make_trajectory()
        builder = TemporalReliabilityDatasetBuilder(
            groups=DEFAULT_LIBERO_GROUPS,
            offsets=(1,),
            target_mode="demonstration",
        )
        examples = builder.build([trajectory])
        target = TemporalValidityTarget(
            DEFAULT_LIBERO_GROUPS,
            TargetConfig(mode="demonstration"),
        )
        result = target.generate(examples, threshold_by_group={"arm": 0.0, "gripper": 0.0})
        self.assertEqual(result.labels.shape, (4,))


class EvaluationTest(unittest.TestCase):
    def test_metrics_and_curve(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)
        self.assertAlmostEqual(expected_calibration_error(labels, scores, n_bins=2), 0.15)
        curve = reliability_curve(labels, scores, n_bins=2)
        np.testing.assert_array_equal(curve.count, [2, 2])
        result = evaluate_reliability(labels, scores, n_bins=2)
        self.assertAlmostEqual(result.brier_score, 0.025)
        self.assertEqual(set(result.as_dict()), {"auroc", "brier_score", "calibration_error", "reliability_curve"})

    def test_untrained_mlp_returns_one_score_per_feature_row(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is an optional model-interface dependency")
        model = MLPBaseline(input_dim=5, hidden_dims=(4,))
        scores = model(torch.zeros((3, 5)))
        self.assertEqual(tuple(scores.shape), (3,))
        self.assertTrue(bool(torch.all((scores >= 0.0) & (scores <= 1.0))))


if __name__ == "__main__":
    unittest.main()

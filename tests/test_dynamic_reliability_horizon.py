from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from one_clock import ActionGroup

from experiments.dynamic_reliability_horizon.adaptive_executor import (
    AdaptiveGroupwiseExecutor,
)
from experiments.dynamic_reliability_horizon.artifacts import PreparedReliabilityDataset
from experiments.dynamic_reliability_horizon.config import TrainingConfig
from experiments.dynamic_reliability_horizon.decoder import (
    GroupHorizonDecoder,
    HorizonDecodeConfig,
)
from experiments.dynamic_reliability_horizon.horizon_analysis import (
    compare_horizon_sources,
    rows_to_curves,
)
from experiments.dynamic_reliability_horizon.scheduler import AdaptiveHorizonScheduler
from experiments.dynamic_reliability_horizon.training import (
    predict_scores,
    train_reliability_model,
)
from experiments.temporal_reliability_training.config import FeatureConfig
from experiments.temporal_reliability_training.features import FeatureEncoder
from experiments.temporal_reliability_training.schema import DEFAULT_LIBERO_GROUPS


def make_dataset() -> PreparedReliabilityDataset:
    rows = 60
    features = np.zeros((rows, 5), dtype=np.float32)
    groups = np.asarray(["arm" if row % 2 == 0 else "gripper" for row in range(rows)])
    offsets = np.asarray([row % 4 for row in range(rows)], dtype=np.int64)
    labels = np.asarray([int((row % 4) < 2) for row in range(rows)], dtype=np.int64)
    features[:, 0] = labels
    features[:, 1] = offsets / 4.0
    split = np.asarray(["train"] * 40 + ["validation"] * 10 + ["test"] * 10)
    return PreparedReliabilityDataset(
        features=features,
        labels=labels,
        groups=groups,
        offsets=offsets,
        episode_ids=np.asarray([f"episode-{row // 2}" for row in range(rows)]),
        task_ids=np.asarray(["task"] * rows),
        feature_names=("label_proxy", "offset", "f2", "f3", "f4"),
        source_steps=np.asarray([row // 2 for row in range(rows)]),
        losses=1.0 - labels,
        split=split,
        metadata={"synthetic": True},
    )


class DecoderTest(unittest.TestCase):
    def test_strict_prefix_decoder_and_fallback(self) -> None:
        decoder = GroupHorizonDecoder(
            HorizonDecodeConfig(threshold_tau=0.5, min_horizon=1, max_horizon=4)
        )
        self.assertEqual(decoder.decode_curve([0.9, 0.8, 0.4, 0.9]), 2)
        self.assertEqual(decoder.decode_curve([0.1, 0.9, 0.9, 0.9]), 1)

    def test_source_rows_decode_fixed_learned_and_oracle_schedules(self) -> None:
        rows = rows_to_curves(
            episode_ids=["e0"] * 8,
            source_steps=[0] * 8,
            groups=["arm"] * 4 + ["gripper"] * 4,
            offsets=[0, 1, 2, 3] * 2,
            scores=[0.9, 0.8, 0.1, 0.9, 0.9, 0.9, 0.2, 0.9],
        )
        oracle = rows_to_curves(
            episode_ids=["e0"] * 8,
            source_steps=[0] * 8,
            groups=["arm"] * 4 + ["gripper"] * 4,
            offsets=[0, 1, 2, 3] * 2,
            scores=[1.0] * 8,
        )
        summaries = compare_horizon_sources(
            rows,
            oracle,
            decoder=GroupHorizonDecoder(
                HorizonDecodeConfig(threshold_tau=0.5, min_horizon=1, max_horizon=4)
            ),
            static_horizons={"arm": 4, "gripper": 16},
        )
        self.assertEqual(summaries["learned_reliability"].by_group["arm"]["mean"], 2.0)
        self.assertEqual(summaries["oracle_reliability"].by_group["gripper"]["mean"], 4.0)


class ArtifactAndTrainingTest(unittest.TestCase):
    def test_artifact_round_trip_and_small_bce_training(self) -> None:
        dataset = make_dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.npz"
            dataset.save(path)
            loaded = PreparedReliabilityDataset.load(path)
            np.testing.assert_array_equal(loaded.features, dataset.features)
            np.testing.assert_array_equal(loaded.split, dataset.split)

            try:
                import torch  # noqa: F401
            except ImportError:
                self.skipTest("torch is an optional training dependency")
            result = train_reliability_model(
                loaded,
                mode="combined",
                config=TrainingConfig(
                    epochs=3,
                    batch_size=16,
                    patience=1,
                    hidden_dims=(8,),
                    seed=4,
                ),
                checkpoint_path=Path(directory) / "model.pt",
            )
            self.assertTrue(result.checkpoint_path.is_file())
            self.assertGreaterEqual(result.best_epoch, 1)
            scores = predict_scores(result.model, loaded.features[:4])
            self.assertEqual(scores.shape, (4,))
            self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))


class SchedulerAndExecutorTest(unittest.TestCase):
    def test_scheduler_produces_group_curves_and_horizons(self) -> None:
        encoder = FeatureEncoder(
            DEFAULT_LIBERO_GROUPS,
            FeatureConfig(observation_embedding_dim=0, max_offset=4),
        )
        scheduler = AdaptiveHorizonScheduler(
            groups=DEFAULT_LIBERO_GROUPS,
            feature_encoder=encoder,
            scorer=lambda features: np.full(features.shape[0], 0.9),
            decoder=GroupHorizonDecoder(
                HorizonDecodeConfig(threshold_tau=0.5, min_horizon=1, max_horizon=4)
            ),
        )
        prediction = scheduler.predict(None, np.zeros((4, 7), dtype=np.float32))
        self.assertEqual(prediction.horizons, {"arm": 4, "gripper": 4})
        self.assertEqual(prediction.reliability_by_group["arm"].shape, (4,))

    def test_adaptive_executor_refreshes_only_expired_groups(self) -> None:
        class FakeScheduler:
            def predict_horizons(self, observation_embedding, chunk):
                del observation_embedding, chunk
                return {"arm": 1, "gripper": 3}

        generations = iter(range(10))

        def query() -> np.ndarray:
            generation = next(generations)
            return np.asarray(
                [[generation * 100 + row * 10 + column for column in range(4)] for row in range(4)],
                dtype=np.float32,
            )

        executor = AdaptiveGroupwiseExecutor(
            action_dim=4,
            chunk_size=4,
            groups=(
                ActionGroup("arm", (0, 1), horizon=1),
                ActionGroup("gripper", (2, 3), horizon=3),
            ),
            scheduler=FakeScheduler(),
        )
        first = executor.step(query)
        second = executor.step(query)
        self.assertEqual(first.refreshed_groups, ("arm", "gripper"))
        self.assertEqual(second.refreshed_groups, ("arm",))
        self.assertEqual(second.source_chunk_ids, {"arm": 1, "gripper": 0})
        np.testing.assert_array_equal(second.action, [100, 101, 12, 13])


if __name__ == "__main__":
    unittest.main()

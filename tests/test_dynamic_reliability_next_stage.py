from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.dynamic_reliability_horizon.causal_features import CausalFeatureContract
from experiments.dynamic_reliability_horizon.config import TrainingConfig
from experiments.dynamic_reliability_horizon.decoder import GroupHorizonDecoder, HorizonDecodeConfig
from experiments.dynamic_reliability_horizon.evaluation import (
    evaluate_vector_horizon_regret,
    evaluate_vector_predictions,
)
from experiments.dynamic_reliability_horizon.horizon_analysis import horizon_regret
from experiments.dynamic_reliability_horizon.model import SharedReliabilityMLP
from experiments.dynamic_reliability_horizon.scheduler import SharedHorizonScheduler
from experiments.dynamic_reliability_horizon.split_manifest import EpisodeSplitManifest
from experiments.dynamic_reliability_horizon.vector_dataset import build_vector_dataset
from experiments.dynamic_reliability_horizon.vector_training import (
    predict_reliability_curves,
    train_shared_reliability_model,
)
from experiments.temporal_reliability_training.config import SplitConfig
from experiments.temporal_reliability_training.schema import (
    DEFAULT_LIBERO_GROUPS,
    TemporalExample,
)
from experiments.temporal_reliability_training.targets import TargetBatch


def make_examples(episode_count: int = 8, horizon: int = 3) -> tuple[TemporalExample, ...]:
    examples: list[TemporalExample] = []
    for episode_index in range(episode_count):
        episode_id = f"episode-{episode_index}"
        chunk = np.arange(4 * 7, dtype=np.float32).reshape(4, 7) + episode_index
        embedding = np.asarray([episode_index, episode_index + 0.5], dtype=np.float32)
        for group in DEFAULT_LIBERO_GROUPS:
            for offset in range(horizon):
                examples.append(
                    TemporalExample(
                        episode_id=episode_id,
                        task_id=f"task-{episode_index % 2}",
                        source_step=0,
                        future_step=offset,
                        offset=offset,
                        group=group.name,
                        source_chunk=chunk,
                        source_observation_embedding=embedding,
                        future_policy_chunk=np.full((1, 7), offset + 10, dtype=np.float32),
                        future_demonstrated_action=np.full(7, offset + 20, dtype=np.float32),
                    )
                )
    return tuple(examples)


def make_vector_dataset():
    examples = make_examples()
    labels = np.asarray(
        [int((offset + (group == "gripper")) % 3 != 2)
         for example in examples
         for offset, group in [(example.offset, example.group)]],
        dtype=np.int64,
    )
    manifest = EpisodeSplitManifest.create(
        [f"episode-{index}" for index in range(8)],
        config=SplitConfig(
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
            seed=17,
            stratify_by_task=False,
        ),
    )
    contract = CausalFeatureContract.for_groups(
        DEFAULT_LIBERO_GROUPS, observation_embedding_dim=2
    )
    dataset = build_vector_dataset(
        examples,
        TargetBatch(losses=1.0 - labels, labels=labels),
        feature_contract=contract,
        horizon_dim=3,
        episode_split=manifest.as_episode_split(),
    )
    return dataset, contract


class EpisodeSplitManifestTest(unittest.TestCase):
    def test_deterministic_disjoint_episode_only_manifest(self) -> None:
        episode_ids = [f"episode-{index}" for index in range(20)]
        config = SplitConfig(seed=123, stratify_by_task=False)
        first = EpisodeSplitManifest.create(episode_ids, config=config)
        second = EpisodeSplitManifest.create(episode_ids, config=config)
        self.assertEqual(first, second)
        self.assertEqual(set(first.episode_ids), set(episode_ids))
        self.assertEqual(len(first.episode_ids), len(set(first.episode_ids)))
        self.assertFalse(set(first.train) & set(first.validation))
        self.assertFalse(set(first.train) & set(first.test))
        self.assertFalse(set(first.validation) & set(first.test))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            first.save(path)
            loaded = EpisodeSplitManifest.load(path)
            self.assertEqual(loaded, first)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("frame_ids", payload)


class CausalContractTest(unittest.TestCase):
    def test_contract_has_only_source_inputs_and_ignores_future_fields(self) -> None:
        contract = CausalFeatureContract.for_groups(DEFAULT_LIBERO_GROUPS, observation_embedding_dim=2)
        parameters = tuple(inspect.signature(contract.encode).parameters)
        self.assertEqual(parameters, ("observation_embedding", "action_chunk", "group"))
        forbidden = ("future", "offset", "phase", "length", "terminal")
        self.assertFalse(any(token in name.lower() for name in contract.feature_names for token in forbidden))

        base = make_examples(episode_count=1, horizon=1)[0]
        changed_future = TemporalExample(
            **{
                **base.__dict__,
                "future_policy_chunk": np.full((1, 7), 999, dtype=np.float32),
                "future_demonstrated_action": np.full(7, -999, dtype=np.float32),
                "future_step": 99,
            }
        )
        np.testing.assert_array_equal(contract.encode_example(base), contract.encode_example(changed_future))

    def test_vector_dataset_is_one_row_per_source_group_and_episode_split_is_preserved(self) -> None:
        dataset, _ = make_vector_dataset()
        self.assertEqual(dataset.labels.shape, (16, 3))
        self.assertTrue(np.all(dataset.label_mask))
        for split_name in ("train", "validation", "test"):
            selected = dataset.split == split_name
            self.assertEqual(len(set(dataset.episode_ids[selected])), int(selected.reshape(-1).sum()) // 2)
        split_sets = [set(dataset.episode_ids[dataset.split == name]) for name in ("train", "validation", "test")]
        self.assertFalse(split_sets[0] & split_sets[1])
        self.assertFalse(split_sets[0] & split_sets[2])
        self.assertFalse(split_sets[1] & split_sets[2])


class SharedEstimatorTest(unittest.TestCase):
    def test_shared_head_returns_a_curve_not_a_horizon(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is an optional training dependency")
        model = SharedReliabilityMLP(5, 4, hidden_dims=(8,))
        output = model(torch.zeros((3, 5), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertTrue(bool(torch.all((output >= 0.0) & (output <= 1.0))))

    def test_shared_training_and_vector_metrics(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is an optional training dependency")
        dataset, _ = make_vector_dataset()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "shared.pt"
            result = train_shared_reliability_model(
                dataset,
                config=TrainingConfig(epochs=3, batch_size=4, patience=1, hidden_dims=(8,), seed=3),
                checkpoint_path=checkpoint,
            )
            self.assertTrue(checkpoint.is_file())
            self.assertGreaterEqual(result.best_epoch, 1)
            scores = predict_reliability_curves(result.model, dataset.features)
            self.assertEqual(scores.shape, dataset.labels.shape)
            report = evaluate_vector_predictions(
                dataset.labels,
                scores,
                dataset.label_mask,
                groups=dataset.groups,
                task_ids=dataset.task_ids,
            )
            self.assertIn("overall", report)
            self.assertIn("task", report)
            self.assertEqual(report["observed_cells"], 48)

    def test_shared_scheduler_decodes_one_curve_per_group(self) -> None:
        contract = CausalFeatureContract.for_groups(DEFAULT_LIBERO_GROUPS, observation_embedding_dim=0)
        scheduler = SharedHorizonScheduler(
            groups=DEFAULT_LIBERO_GROUPS,
            feature_contract=contract,
            scorer=lambda features: np.full((features.shape[0], 3), 0.9, dtype=np.float64),
            decoder=GroupHorizonDecoder(HorizonDecodeConfig(threshold_tau=0.5, max_horizon=3)),
        )
        prediction = scheduler.predict(None, np.zeros((4, 7), dtype=np.float32))
        self.assertEqual(prediction.horizons, {"arm": 3, "gripper": 3})
        self.assertEqual(prediction.reliability_by_group["arm"].shape, (3,))


class DecoderAndRegretTest(unittest.TestCase):
    def test_decoder_monotonic_threshold_and_curve_clipping(self) -> None:
        decoder = GroupHorizonDecoder(HorizonDecodeConfig(threshold_tau=0.5, max_horizon=10))
        self.assertEqual(decoder.decode_curve([0.9, 0.8, 0.7]), 3)
        self.assertEqual(decoder.decode_curve([0.6, 0.5, 0.99]), 1)
        clipped = GroupHorizonDecoder(HorizonDecodeConfig(threshold_tau=0.5, max_horizon=2))
        self.assertEqual(clipped.decode_curve([0.9, 0.9, 0.9, 0.9]), 2)

        non_prefix = GroupHorizonDecoder(
            HorizonDecodeConfig(threshold_tau=0.5, max_horizon=4, require_prefix=False)
        )
        self.assertEqual(non_prefix.decode_curve([0.1, 0.8, 0.2, 0.9]), 4)

    def test_horizon_regret_reports_signed_and_absolute_error(self) -> None:
        regret = horizon_regret(
            [{"arm": 2, "gripper": 4}, {"arm": 1, "gripper": 2}],
            [{"arm": 3, "gripper": 3}, {"arm": 1, "gripper": 4}],
        )
        self.assertEqual(regret.count, 2)
        self.assertAlmostEqual(regret.by_group["arm"]["mean_absolute_regret"], 0.5)
        self.assertAlmostEqual(regret.by_group["arm"]["mean_signed_regret"], -0.5)
        self.assertAlmostEqual(regret.by_group["gripper"]["mean_absolute_regret"], 1.5)
        self.assertAlmostEqual(regret.by_group["gripper"]["overcommit_rate"], 0.5)

    def test_vector_horizon_regret_uses_test_episode_source_rows(self) -> None:
        dataset, _ = make_vector_dataset()
        report = evaluate_vector_horizon_regret(
            dataset,
            dataset.labels.astype(np.float64),
            decoder=GroupHorizonDecoder(HorizonDecodeConfig(threshold_tau=0.5, max_horizon=3)),
        )
        self.assertEqual(report["count"], 2)
        self.assertEqual(set(report["by_group"]), {"arm", "gripper"})
        self.assertEqual(report["by_group"]["arm"]["mean_absolute_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()

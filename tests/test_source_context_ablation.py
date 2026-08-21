from __future__ import annotations

import unittest

import numpy as np

from experiments.dynamic_reliability_horizon.source_context_ablation.run_ablation import (
    CONDITIONS,
    build_pilot_data,
    load_feature_data,
)


class SourceContextAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.arrays, cls.manifest = load_feature_data()

    def test_exact_feature_artifact_schema_and_cohort(self) -> None:
        self.assertEqual(self.arrays["source_chunk_actions"].shape, (3740, 100, 7))
        self.assertEqual(self.arrays["source_state"].shape, (3740, 8))
        self.assertEqual(self.arrays["source_policy_latent"].shape, (3740, 512))
        self.assertEqual(self.arrays["y_refresh"].shape, (3740, 2, 99))
        self.assertEqual(self.arrays["label_observed"].shape, (3740, 2, 99))
        keys = set(zip(self.arrays["episode_id"].tolist(), self.arrays["source_step"].tolist(), strict=True))
        self.assertEqual(len(keys), 3740)
        self.assertEqual(len(np.unique(self.arrays["episode_id"])), 454)

    def test_split_is_episode_pure(self) -> None:
        split = self.arrays["split_membership"]
        episodes = self.arrays["episode_id"]
        sets = [set(episodes[split == code].tolist()) for code in (0, 1, 2)]
        self.assertTrue(all(not (sets[left] & sets[right]) for left, right in ((0, 1), (0, 2), (1, 2))))

    def test_four_conditions_share_labels_and_group_rows(self) -> None:
        pilot_data = {
            name: build_pilot_data(self.arrays, features)
            for name, features in CONDITIONS.items()
        }
        self.assertEqual(pilot_data["A_chunk_only"].features.shape[1], 702)
        self.assertEqual(pilot_data["B_chunk_plus_state"].features.shape[1], 710)
        self.assertEqual(pilot_data["C_chunk_plus_frozen_ACT_latent"].features.shape[1], 1214)
        self.assertEqual(pilot_data["D_chunk_plus_state_plus_frozen_ACT_latent"].features.shape[1], 1222)
        reference = pilot_data["A_chunk_only"]
        for current in pilot_data.values():
            np.testing.assert_array_equal(current.labels, reference.labels)
            np.testing.assert_array_equal(current.label_mask, reference.label_mask)
            np.testing.assert_array_equal(current.split, reference.split)
            np.testing.assert_array_equal(current.group_ids, reference.group_ids)
            self.assertEqual(current.features.shape[0], 7480)

    def test_latent_and_state_are_source_only_finite_features(self) -> None:
        self.assertEqual(self.arrays["source_state"].dtype, np.float32)
        self.assertEqual(self.arrays["source_policy_latent"].dtype, np.float32)
        self.assertTrue(np.isfinite(self.arrays["source_state"]).all())
        self.assertTrue(np.isfinite(self.arrays["source_policy_latent"]).all())
        extraction = self.manifest["extraction"]
        self.assertTrue(extraction["invariance"]["allclose_atol_1e-6_rtol_1e-6"])
        self.assertFalse(extraction["latent"]["demonstration_action_input"])
        self.assertFalse(extraction["latent"]["future_input"])


if __name__ == "__main__":
    unittest.main()

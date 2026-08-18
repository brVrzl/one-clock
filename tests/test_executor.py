from __future__ import annotations

import unittest

import numpy as np

from one_clock import ActionGroup, FixedChunkExecutor
from scripts.run_libero_gate0 import make_episode_record
from scripts.run_gate0 import summarize_run


def tagged_chunk(generation: int, chunk_size: int = 4, action_dim: int = 4) -> np.ndarray:
    return np.asarray(
        [
            [generation * 100 + row * 10 + column for column in range(action_dim)]
            for row in range(chunk_size)
        ],
        dtype=np.float32,
    )


class FixedChunkExecutorTest(unittest.TestCase):
    def test_run_summary_counts_policy_query_budget(self) -> None:
        episode_records = [
            [
                {"policy_query": True, "source_ages": {"arm": 0, "gripper": 0}},
                {"policy_query": False, "source_ages": {"arm": 1, "gripper": 1}},
            ],
            [
                {"policy_query": True, "source_ages": {"arm": 0, "gripper": 0}},
            ],
        ]

        summary = summarize_run(episode_records, successes=1)

        self.assertEqual(summary["environment_steps"], 3)
        self.assertEqual(summary["policy_queries"], 2)
        self.assertAlmostEqual(summary["policy_queries_per_episode"], 1.0)
        self.assertAlmostEqual(summary["policy_query_rate"], 2 / 3)
        self.assertEqual(
            summary["mean_source_age_by_group"],
            {"arm": 1 / 3, "gripper": 1 / 3},
        )

    def test_global_fixed_repeats_the_ordinary_fixed_horizon_sequence(self) -> None:
        generations = iter(range(10))
        executor = FixedChunkExecutor.global_fixed(
            action_dim=4,
            chunk_size=4,
            horizon=2,
        )

        decisions = [
            executor.step(lambda: tagged_chunk(next(generations))) for _ in range(6)
        ]

        np.testing.assert_array_equal(
            np.stack([decision.action for decision in decisions]),
            np.asarray(
                [
                    [0, 1, 2, 3],
                    [10, 11, 12, 13],
                    [100, 101, 102, 103],
                    [110, 111, 112, 113],
                    [200, 201, 202, 203],
                    [210, 211, 212, 213],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(sum(decision.policy_query for decision in decisions), 3)
        self.assertEqual([decision.new_chunk_id for decision in decisions], [0, None, 1, None, 2, None])

    def test_groupwise_composes_actions_from_different_chunk_generations(self) -> None:
        generations = iter(range(10))
        executor = FixedChunkExecutor.groupwise_fixed(
            action_dim=4,
            chunk_size=4,
            groups=(
                ActionGroup("left_arm", (0, 1), horizon=3),
                ActionGroup("right_arm", (2, 3), horizon=1),
            ),
        )

        decisions = [
            executor.step(lambda: tagged_chunk(next(generations))) for _ in range(4)
        ]

        np.testing.assert_array_equal(decisions[1].action, [10, 11, 102, 103])
        self.assertTrue(decisions[1].policy_query)
        self.assertEqual(decisions[1].refreshed_groups, ("right_arm",))
        self.assertEqual(decisions[1].source_chunk_ids, {"left_arm": 0, "right_arm": 1})
        self.assertEqual(decisions[1].source_ages, {"left_arm": 1, "right_arm": 0})
        self.assertEqual(decisions[1].source_positions, {"left_arm": 1, "right_arm": 0})
        self.assertEqual(decisions[1].remaining_commitments, {"left_arm": 2, "right_arm": 1})
        self.assertEqual(decisions[1].configured_horizons, {"left_arm": 3, "right_arm": 1})

    def test_groupwise_does_not_query_when_no_group_expired(self) -> None:
        query_count = 0

        def query() -> np.ndarray:
            nonlocal query_count
            query_count += 1
            return tagged_chunk(query_count - 1)

        executor = FixedChunkExecutor.groupwise_fixed(
            action_dim=4,
            chunk_size=4,
            groups=(
                ActionGroup("a", (0, 1), horizon=3),
                ActionGroup("b", (2, 3), horizon=3),
            ),
        )
        decisions = [executor.step(query) for _ in range(3)]

        self.assertEqual(query_count, 1)
        self.assertEqual([decision.policy_query for decision in decisions], [True, False, False])
        self.assertEqual(decisions[-1].source_chunk_ids, {"a": 0, "b": 0})

    def test_libero_arm_gripper_groups_preserve_mixed_generations(self) -> None:
        generations = iter(range(10))
        executor = FixedChunkExecutor.groupwise_fixed(
            action_dim=7,
            chunk_size=4,
            groups=(
                ActionGroup("arm", tuple(range(6)), horizon=3),
                ActionGroup("gripper", (6,), horizon=1),
            ),
        )

        decisions = [
            executor.step(lambda: tagged_chunk(next(generations), action_dim=7))
            for _ in range(2)
        ]

        np.testing.assert_array_equal(decisions[1].action[:6], [10, 11, 12, 13, 14, 15])
        self.assertEqual(decisions[1].action[6], 106)
        self.assertEqual(decisions[1].source_chunk_ids, {"arm": 0, "gripper": 1})
        self.assertEqual(decisions[1].source_positions, {"arm": 1, "gripper": 0})

    def test_libero_episode_record_logs_pairing_and_query_budget(self) -> None:
        records = [
            {"policy_query": True, "source_ages": {"arm": 0, "gripper": 0}},
            {"policy_query": False, "source_ages": {"arm": 1, "gripper": 1}},
        ]

        result = make_episode_record(
            episode=0,
            init_state_id=7,
            seed=1007,
            strategy="groupwise_fixed",
            configured_horizons={"arm": 8, "gripper": 2},
            success=True,
            records=records,
            task_name="task",
            task_description="description",
            initial_eef_pos=np.asarray([1.0, 2.0, 3.0]),
            initial_image_means={"image": 4.0},
        )

        self.assertEqual(result["init_state_id"], 7)
        self.assertEqual(result["seed"], 1007)
        self.assertEqual(result["environment_steps"], 2)
        self.assertEqual(result["policy_queries"], 1)
        self.assertAlmostEqual(result["policy_query_rate"], 0.5)
        self.assertAlmostEqual(result["mean_source_age_arm"], 0.5)
        self.assertEqual(result["arm_horizon"], 8)
        self.assertEqual(result["gripper_horizon"], 2)


if __name__ == "__main__":
    unittest.main()

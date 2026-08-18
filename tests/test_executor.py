from __future__ import annotations

import unittest

import numpy as np

from one_clock import ActionGroup, FixedChunkExecutor


def tagged_chunk(generation: int, chunk_size: int = 4, action_dim: int = 4) -> np.ndarray:
    return np.asarray(
        [
            [generation * 100 + row * 10 + column for column in range(action_dim)]
            for row in range(chunk_size)
        ],
        dtype=np.float32,
    )


class FixedChunkExecutorTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

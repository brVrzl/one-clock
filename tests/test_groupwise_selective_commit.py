from __future__ import annotations

import unittest

import numpy as np

from experiments.groupwise_selective_commitment.executor import (
    CommitGroup,
    ScheduledCommitExecutor,
    normalized_current_distance,
)


GROUPS = (
    CommitGroup("arm", tuple(range(6))),
    CommitGroup("gripper", (6,)),
)
STD = np.ones(7, dtype=np.float64)


def chunk(arm_value: float, gripper_value: float, *, size: int = 8) -> np.ndarray:
    result = np.zeros((size, 7), dtype=np.float32)
    result[:, :6] = arm_value
    result[:, 6] = gripper_value
    return result


class SelectiveCommitExecutorTest(unittest.TestCase):
    def make_executor(self, method: str, cadence: int = 2) -> ScheduledCommitExecutor:
        return ScheduledCommitExecutor(
            method=method,  # type: ignore[arg-type]
            query_cadence=cadence,
            chunk_size=8,
            action_dim=7,
            groups=GROUPS,
            action_std=STD,
        )

    def test_distance_reuses_audited_normalization_and_sign_rule(self) -> None:
        self.assertEqual(normalized_current_distance(np.zeros(6), np.ones(6), group="arm", action_std=STD), 1.0)
        self.assertEqual(normalized_current_distance(np.zeros(1), np.asarray([0.5]), group="gripper", action_std=STD), 0.5)
        self.assertGreater(normalized_current_distance(np.asarray([-0.1]), np.asarray([0.1]), group="gripper", action_std=STD), 1.0)

    def test_global_replaces_both_groups_at_every_scheduled_query(self) -> None:
        executor = self.make_executor("global_replace", cadence=2)
        generations = iter((chunk(0.0, 0.0), chunk(5.0, 5.0), chunk(9.0, 9.0)))
        decisions = [executor.step(lambda: next(generations)) for _ in range(5)]
        self.assertEqual([decision.policy_query for decision in decisions], [True, False, True, False, True])
        for decision in decisions:
            if decision.policy_query:
                self.assertEqual(decision.acceptance, {"arm": "accept", "gripper": "accept"})
                self.assertEqual(decision.current_source_generation_ids["arm"], decision.current_source_generation_ids["gripper"])

    def test_selective_executor_exercises_all_four_decision_pairs(self) -> None:
        executor = self.make_executor("selective_commit", cadence=1)
        # The first query initializes both groups.  Each later chunk changes
        # only the intended group by more than epsilon.
        chunks = iter(
            (
                chunk(0.0, 0.0),
                chunk(2.0, 0.0),
                chunk(2.0, 2.0),
                chunk(0.0, 2.0),
            )
        )
        decisions = [executor.step(lambda: next(chunks)) for _ in range(4)]
        pairs = [(item.acceptance["arm"], item.acceptance["gripper"]) for item in decisions]
        self.assertEqual(
            pairs,
            [("accept", "accept"), ("accept", "retain"), ("retain", "accept"), ("accept", "retain")],
        )

    def test_query_schedule_is_cadence_exact_and_query_callback_is_full(self) -> None:
        executor = self.make_executor("selective_commit", cadence=4)
        calls: list[int] = []

        def query() -> np.ndarray:
            calls.append(len(calls))
            return chunk(float(len(calls)), float(len(calls)))

        decisions = [executor.step(query) for _ in range(10)]
        self.assertEqual([item.environment_step for item in decisions if item.policy_query], [0, 4, 8])
        self.assertEqual(calls, [0, 1, 2])
        self.assertTrue(all(item.action.shape == (7,) for item in decisions))

    def test_exhaustion_is_explicit_and_next_scheduled_query_accepts(self) -> None:
        executor = ScheduledCommitExecutor(
            method="selective_commit",
            query_cadence=4,
            chunk_size=3,
            action_dim=7,
            groups=GROUPS,
            action_std=STD,
        )
        generations = iter((chunk(0.0, 0.0, size=3), chunk(0.0, 0.0, size=3)))
        decisions = [executor.step(lambda: next(generations)) for _ in range(5)]
        self.assertTrue(decisions[3].source_exhausted["arm"])
        self.assertEqual(decisions[4].acceptance["arm"], "accept")
        self.assertFalse(decisions[4].source_exhausted["arm"])


if __name__ == "__main__":
    unittest.main()

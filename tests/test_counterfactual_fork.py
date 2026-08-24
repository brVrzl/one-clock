import unittest

from experiments.counterfactual_tournament.fork_engine import CounterfactualFork


class CounterfactualForkTest(unittest.TestCase):
    def test_each_branch_starts_from_the_same_snapshot(self):
        state = {"value": 0}
        applied = []

        def snapshot():
            return dict(state)

        def restore(saved):
            state.clear()
            state.update(saved)

        def perturb(name):
            state["value"] += {"small": 1, "large": 3}[name]
            applied.append(state["value"])

        def continue_from(_index):
            state["value"] += 10
            return state["value"] < 12, 1

        fork = CounterfactualFork(
            snapshot=snapshot,
            restore=restore,
            perturb=perturb,
            continue_from=continue_from,
        )
        outcomes = fork.evaluate(4, ["small", "large"])
        self.assertEqual(applied, [1, 3])
        self.assertEqual([row.success for row in outcomes], [True, False])
        self.assertEqual(state, {"value": 0})


if __name__ == "__main__":
    unittest.main()

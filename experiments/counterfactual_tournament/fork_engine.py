"""Small, environment-agnostic counterfactual fork primitive.

The RoboTwin runner supplies the concrete snapshot and continuation callbacks.
Keeping the causal operation here explicit makes it testable without importing
the heavyweight simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ForkOutcome:
    perturbation: str
    success: bool
    continuation_steps: int
    error: str | None = None


class CounterfactualFork:
    """Restore one exact snapshot, perturb it, and score continuation."""

    def __init__(
        self,
        *,
        snapshot: Callable[[], Any],
        restore: Callable[[Any], None],
        perturb: Callable[[str], None],
        continue_from: Callable[[int], tuple[bool, int]],
    ) -> None:
        self._snapshot = snapshot
        self._restore = restore
        self._perturb = perturb
        self._continue_from = continue_from

    def evaluate(self, start_index: int, perturbations: list[str]) -> list[ForkOutcome]:
        state = self._snapshot()
        outcomes: list[ForkOutcome] = []
        for name in perturbations:
            self._restore(state)
            try:
                self._perturb(name)
                success, steps = self._continue_from(start_index)
                outcomes.append(ForkOutcome(name, bool(success), int(steps)))
            except Exception as exc:  # branch failures are valid negative labels
                outcomes.append(ForkOutcome(name, False, 0, f"{type(exc).__name__}: {exc}"))
        self._restore(state)
        return outcomes

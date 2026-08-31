"""ACT-orientation-preserving sparse temporal ensembling.

This module adds the dense-equivalent operator used by the repaired ACT
audit.  It reuses the historical executor's query scheduling, candidate
extraction, and reset behavior, but computes temporal-ensemble weights from
the physical separation between same-target query sources::

    w_j proportional to exp(-0.01 * (q_j - q_0))

Candidates are expected in oldest-query-to-newest-query order.  The fixed
coefficient and orientation are intentional: at cadence one this is exactly
the validated canonical ACT kernel, and at cadence ``h`` the relative weights
are ``[1, exp(-0.01*h), ...]``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Literal

import numpy as np


# Keep the historical executor untouched while making this module directly
# importable from the audit directory (including from the rollout runner).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPARSE_ROOT = _REPO_ROOT / "experiments" / "sparse_temporal_ensemble_dev"
if str(_SPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPARSE_ROOT))

from sparse_executor import SparseExecutor, StepResult  # noqa: E402


DENSE_EQUIVALENT_COEFFICIENT = 0.01
DenseEquivalentMode = Literal["hard", "dense_equivalent_te"]


def dense_equivalent_te_weights(source_query_steps: np.ndarray) -> np.ndarray:
    """Return dense-equivalent ACT weights for oldest-to-newest sources.

    ``source_query_steps`` is the same-target candidate source list, ordered
    oldest to newest.  Only the relative source spacing matters, so the
    result is invariant to adding a constant to every query step.  The
    coefficient is fixed at the validated ACT value ``0.01``.
    """

    raw_sources = np.asarray(source_query_steps)
    if raw_sources.ndim != 1 or raw_sources.size == 0:
        raise ValueError("source_query_steps must be a non-empty vector")
    if not np.issubdtype(raw_sources.dtype, np.integer):
        raise ValueError("source_query_steps must contain integer query steps")
    sources = raw_sources.astype(np.int64, copy=False)
    if np.any(np.diff(sources) < 0):
        raise ValueError("source_query_steps must be ordered oldest to newest")

    relative_separation = sources.astype(np.float64) - float(sources[0])
    logits = -DENSE_EQUIVALENT_COEFFICIENT * relative_separation
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


class DenseEquivalentSparseExecutor(SparseExecutor):
    """Sparse executor with ACT-orientation-preserving dense-equivalent TE.

    The constructor, :meth:`step`, and :meth:`reset` follow the historical
    :class:`~sparse_executor.SparseExecutor` interface.  ``mode`` accepts
    ``"dense_equivalent_te"`` for the new operator and ``"hard"`` for a
    compatibility path that delegates unchanged to the historical executor.
    The temporal coefficient is fixed at ``0.01`` and cannot be tuned.
    """

    def __init__(
        self,
        *,
        cadence: int,
        prediction_horizon: int,
        mode: DenseEquivalentMode,
        coefficient: float = DENSE_EQUIVALENT_COEFFICIENT,
        action_dim: int = 7,
    ) -> None:
        if mode not in ("hard", "dense_equivalent_te"):
            raise ValueError(f"unknown mode: {mode}")
        coefficient = float(coefficient)
        if coefficient != DENSE_EQUIVALENT_COEFFICIENT:
            raise ValueError(
                "dense_equivalent_te uses the fixed ACT coefficient 0.01"
            )

        # The parent owns all scheduling, candidate validity, and reset
        # semantics.  It only needs one of its existing internal mode values
        # during construction; this class restores the explicit public name
        # immediately afterward.
        super().__init__(
            cadence=cadence,
            prediction_horizon=prediction_horizon,
            mode="hard" if mode == "hard" else "sparse_te",
            coefficient=coefficient,
            action_dim=action_dim,
        )
        self.mode = mode

    def step(self, target_step: int, query_fn: Callable[[], np.ndarray]) -> StepResult:
        """Query and execute one step using dense-equivalent source weights."""

        result = super().step(target_step, query_fn)
        if self.mode == "hard":
            return result

        weights = dense_equivalent_te_weights(result.candidates.source_query_steps)
        action = weights @ result.candidates.actions
        return StepResult(
            action=action,
            target_step=result.target_step,
            queried=result.queried,
            latest_query_step=result.latest_query_step,
            candidates=result.candidates,
            weights=weights,
        )


__all__ = [
    "DENSE_EQUIVALENT_COEFFICIENT",
    "DenseEquivalentSparseExecutor",
    "dense_equivalent_te_weights",
]

"""Convert group reliability curves into execution horizons."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from .config import DynamicHorizonConfig


@dataclass(frozen=True)
class HorizonDecodeConfig(DynamicHorizonConfig):
    """Named decoder configuration for the scheduler API."""


class GroupHorizonDecoder:
    """Decode ``R_hat_g(k)`` using a configurable reliability threshold.

    Offset zero corresponds to the first action in a chunk.  The returned
    horizon is a number of actions, so a valid prefix through offset ``k`` has
    horizon ``k + 1``.  Prefix decoding is the safe default: an isolated valid
    score after an invalid offset cannot extend execution past the first failed
    reliability test.
    """

    def __init__(self, config: HorizonDecodeConfig | None = None) -> None:
        self.config = config or HorizonDecodeConfig()

    def decode_curve(self, scores: Sequence[float]) -> int:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("reliability curve must be one-dimensional and non-empty")
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("reliability scores must be finite probabilities in [0, 1]")
        maximum = min(values.size, self.config.max_horizon or values.size)
        self.config.validate(chunk_size=values.size)
        valid = values[:maximum] > self.config.threshold_tau
        if self.config.require_prefix:
            horizon = 0
            for is_valid in valid:
                if not is_valid:
                    break
                horizon += 1
        else:
            valid_indices = np.flatnonzero(valid)
            horizon = int(valid_indices[-1] + 1) if valid_indices.size else 0
        if horizon == 0:
            horizon = self.config.min_horizon
        return int(min(maximum, max(self.config.min_horizon, horizon)))

    def decode_curves(self, curves: Mapping[str, Sequence[float]]) -> dict[str, int]:
        if not curves:
            raise ValueError("at least one group curve is required")
        horizons = {name: self.decode_curve(scores) for name, scores in curves.items()}
        return horizons

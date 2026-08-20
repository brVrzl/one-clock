"""Online-facing source-observation -> reliability -> horizon scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from experiments.temporal_reliability_training.features import FeatureEncoder
from experiments.temporal_reliability_training.schema import GroupSpec, TemporalExample

from .decoder import GroupHorizonDecoder


class ReliabilityScorer(Protocol):
    def __call__(self, features: Any) -> Any:
        """Return one probability per feature row."""


class TorchModelScorer:
    """Adapt a Torch reliability model to the NumPy scheduler interface."""

    def __init__(self, model: object, *, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    def __call__(self, features: np.ndarray) -> np.ndarray:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - host-dependent
            raise ImportError("TorchModelScorer requires torch") from error
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(tensor).detach().cpu().numpy()


@dataclass(frozen=True)
class HorizonPrediction:
    reliability_by_group: dict[str, np.ndarray]
    horizons: dict[str, int]
    feature_matrix: np.ndarray


class AdaptiveHorizonScheduler:
    """Score every group/offset of a fresh chunk and decode group horizons."""

    def __init__(
        self,
        *,
        groups: Sequence[GroupSpec],
        feature_encoder: FeatureEncoder,
        scorer: ReliabilityScorer,
        decoder: GroupHorizonDecoder,
        max_offset: int | None = None,
    ) -> None:
        if not groups:
            raise ValueError("at least one group is required")
        self.groups = tuple(groups)
        self.feature_encoder = feature_encoder
        self.scorer = scorer
        self.decoder = decoder
        self.max_offset = max_offset
        if max_offset is not None and max_offset < 1:
            raise ValueError("max_offset must be positive")

    def _source_examples(
        self,
        observation_embedding: np.ndarray | None,
        action_chunk: np.ndarray,
    ) -> tuple[TemporalExample, ...]:
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] < 1:
            raise ValueError("action_chunk must have shape [chunk_step, action_dim]")
        offsets = range(min(chunk.shape[0], self.max_offset or chunk.shape[0]))
        return tuple(
            TemporalExample(
                episode_id="online",
                task_id=None,
                source_step=0,
                future_step=offset,
                offset=offset,
                group=group.name,
                source_chunk=chunk,
                source_observation_embedding=observation_embedding,
                future_policy_chunk=None,
                future_demonstrated_action=None,
            )
            for group in self.groups
            for offset in offsets
        )

    def predict(
        self,
        observation_embedding: np.ndarray | None,
        action_chunk: np.ndarray,
    ) -> HorizonPrediction:
        examples = self._source_examples(observation_embedding, action_chunk)
        batch = self.feature_encoder.encode_many(examples)
        raw_scores = self.scorer(batch.features)
        if hasattr(raw_scores, "detach"):
            raw_scores = raw_scores.detach().cpu().numpy()
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if scores.shape != (len(examples),):
            raise ValueError("reliability scorer must return one score per feature row")
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("reliability scorer must return probabilities in [0, 1]")
        horizon_length = min(np.asarray(action_chunk).shape[0], self.max_offset or np.asarray(action_chunk).shape[0])
        reliability_by_group: dict[str, np.ndarray] = {}
        cursor = 0
        for group in self.groups:
            reliability_by_group[group.name] = scores[cursor : cursor + horizon_length].copy()
            cursor += horizon_length
        horizons = self.decoder.decode_curves(reliability_by_group)
        return HorizonPrediction(reliability_by_group, horizons, batch.features)

    def predict_horizons(
        self,
        observation_embedding: np.ndarray | None,
        action_chunk: np.ndarray,
    ) -> dict[str, int]:
        return self.predict(observation_embedding, action_chunk).horizons

"""Prepared feature/target artifacts for training without target regeneration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from experiments.temporal_reliability_training.dataset import EpisodeSplit
from experiments.temporal_reliability_training.features import FeatureBatch
from experiments.temporal_reliability_training.schema import TemporalExample
from experiments.temporal_reliability_training.targets import TargetBatch


def _id_array(values: Sequence[object | None]) -> np.ndarray:
    return np.asarray(["" if value is None else str(value) for value in values], dtype=str)


@dataclass(frozen=True)
class PreparedReliabilityDataset:
    """Source-only features plus precomputed survival-validity labels."""

    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    offsets: np.ndarray
    episode_ids: np.ndarray
    task_ids: np.ndarray
    feature_names: tuple[str, ...]
    source_steps: np.ndarray | None = None
    losses: np.ndarray | None = None
    split: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.int64)
        groups = np.asarray(self.groups).astype(str)
        offsets = np.asarray(self.offsets, dtype=np.int64)
        episode_ids = np.asarray(self.episode_ids).astype(str)
        task_ids = np.asarray(self.task_ids).astype(str)
        n = features.shape[0] if features.ndim == 2 else -1
        arrays = (labels, groups, offsets, episode_ids, task_ids)
        if features.ndim != 2 or n < 1:
            raise ValueError("features must have shape [example, feature] and be non-empty")
        if any(array.shape != (n,) for array in arrays):
            raise ValueError("all dataset fields must have one value per feature row")
        if not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("features must be finite and labels must be binary")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "episode_ids", episode_ids)
        object.__setattr__(self, "task_ids", task_ids)
        if len(self.feature_names) != features.shape[1]:
            raise ValueError("feature_names must match feature dimension")
        if self.source_steps is None:
            source_steps = np.arange(n, dtype=np.int64)
        else:
            source_steps = np.asarray(self.source_steps, dtype=np.int64)
            if source_steps.shape != (n,):
                raise ValueError("source_steps must match examples")
        object.__setattr__(self, "source_steps", source_steps)
        if self.losses is not None:
            losses = np.asarray(self.losses, dtype=np.float32)
            if losses.shape != (n,) or not np.isfinite(losses).all():
                raise ValueError("losses must match examples and be finite")
            object.__setattr__(self, "losses", losses)
        if self.split is not None:
            split = np.asarray(self.split).astype(str)
            if split.shape != (n,) or any(value not in {"train", "validation", "test"} for value in split):
                raise ValueError("split must contain train, validation, or test")
            object.__setattr__(self, "split", split)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]

    def mask(self, name: str) -> np.ndarray:
        if self.split is None:
            raise ValueError("dataset has no stored episode split")
        if name not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {name!r}")
        return self.split == name

    def select(self, selected: np.ndarray) -> "PreparedReliabilityDataset":
        selected = np.asarray(selected, dtype=bool)
        if selected.shape != (self.features.shape[0],):
            raise ValueError("selection mask must match dataset rows")
        return PreparedReliabilityDataset(
            features=self.features[selected],
            labels=self.labels[selected],
            groups=self.groups[selected],
            offsets=self.offsets[selected],
            episode_ids=self.episode_ids[selected],
            task_ids=self.task_ids[selected],
            feature_names=self.feature_names,
            source_steps=self.source_steps[selected],
            losses=None if self.losses is None else self.losses[selected],
            split=None if self.split is None else self.split[selected],
            metadata=self.metadata,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            groups=self.groups,
            offsets=self.offsets,
            episode_ids=self.episode_ids,
            task_ids=self.task_ids,
            feature_names=np.asarray(self.feature_names, dtype=str),
            source_steps=self.source_steps,
            losses=np.asarray([] if self.losses is None else self.losses, dtype=np.float32),
            split=np.asarray([] if self.split is None else self.split, dtype=str),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PreparedReliabilityDataset":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"prepared reliability dataset not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            losses = data["losses"]
            split = data["split"]
            metadata = json.loads(str(data["metadata_json"].item()))
            return cls(
                features=data["features"],
                labels=data["labels"],
                groups=data["groups"],
                offsets=data["offsets"],
                episode_ids=data["episode_ids"],
                task_ids=data["task_ids"],
                feature_names=tuple(str(value) for value in data["feature_names"]),
                source_steps=data["source_steps"],
                losses=None if losses.size == 0 else losses,
                split=None if split.size == 0 else split,
                metadata=metadata,
            )


def prepare_dataset(
    examples: Sequence[TemporalExample],
    feature_batch: FeatureBatch,
    target_batch: TargetBatch,
    *,
    episode_split: EpisodeSplit | None = None,
    metadata: Mapping[str, object] | None = None,
) -> PreparedReliabilityDataset:
    """Materialize features and existing targets into one trainable artifact."""

    if target_batch.labels is None:
        raise ValueError("binary labels are required; supply target thresholds first")
    if feature_batch.features.shape[0] != len(examples):
        raise ValueError("feature rows and examples must have the same length")
    split_by_episode: dict[str, str] = {}
    if episode_split is not None:
        for name, ids in episode_split.as_dict().items():
            for episode_id in ids:
                key = str(episode_id)
                if key in split_by_episode:
                    raise ValueError("episode split contains duplicate episode IDs")
                split_by_episode[key] = name
    split = None
    if episode_split is not None:
        split = np.asarray([split_by_episode[str(example.episode_id)] for example in examples], dtype=str)
    return PreparedReliabilityDataset(
        features=feature_batch.features,
        labels=target_batch.labels,
        groups=np.asarray([example.group for example in examples], dtype=str),
        offsets=np.asarray([example.offset for example in examples], dtype=np.int64),
        episode_ids=_id_array([example.episode_id for example in examples]),
        task_ids=_id_array([example.task_id for example in examples]),
        feature_names=feature_batch.feature_names,
        source_steps=np.asarray([example.source_step for example in examples], dtype=np.int64),
        losses=target_batch.losses,
        split=split,
        metadata=metadata,
    )

"""One-row-per-source/group vector target dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from experiments.temporal_reliability_training.dataset import EpisodeSplit
from experiments.temporal_reliability_training.schema import TemporalExample
from experiments.temporal_reliability_training.targets import TargetBatch

from .causal_features import CausalFeatureContract


@dataclass(frozen=True)
class VectorReliabilityDataset:
    """Causal source features and a masked ``Y_g(0...K-1)`` curve."""

    features: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray
    groups: np.ndarray
    episode_ids: np.ndarray
    task_ids: np.ndarray
    source_steps: np.ndarray
    feature_names: tuple[str, ...]
    split: np.ndarray | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.float32)
        label_mask = np.asarray(self.label_mask, dtype=bool)
        groups = np.asarray(self.groups).astype(str)
        episode_ids = np.asarray(self.episode_ids).astype(str)
        task_ids = np.asarray(self.task_ids).astype(str)
        source_steps = np.asarray(self.source_steps, dtype=np.int64)
        if features.ndim != 2 or labels.ndim != 2 or label_mask.shape != labels.shape:
            raise ValueError("features must be 2-D and labels/mask must be 2-D matching arrays")
        n = features.shape[0]
        if labels.shape[0] != n or any(array.shape != (n,) for array in (groups, episode_ids, task_ids, source_steps)):
            raise ValueError("vector dataset metadata must match feature rows")
        if not np.isfinite(features).all() or not np.isin(labels[label_mask], [0.0, 1.0]).all():
            raise ValueError("features must be finite and observed labels must be binary")
        if len(self.feature_names) != features.shape[1]:
            raise ValueError("feature_names must match feature dimension")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_mask", label_mask)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "episode_ids", episode_ids)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "source_steps", source_steps)
        if self.split is not None:
            split = np.asarray(self.split).astype(str)
            if split.shape != (n,) or any(value not in {"train", "validation", "test"} for value in split):
                raise ValueError("split must contain train, validation, or test")
            split_by_episode: dict[str, str] = {}
            for episode_id, split_name in zip(episode_ids, split):
                previous = split_by_episode.setdefault(str(episode_id), str(split_name))
                if previous != str(split_name):
                    raise ValueError("one episode cannot occur in multiple dataset splits")
            object.__setattr__(self, "split", split)

    @property
    def horizon_dim(self) -> int:
        return self.labels.shape[1]

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]

    def mask(self, name: str) -> np.ndarray:
        if self.split is None:
            raise ValueError("dataset has no split assignment")
        return self.split == name

    def select(self, selected: np.ndarray) -> "VectorReliabilityDataset":
        selected = np.asarray(selected, dtype=bool)
        if selected.shape != (self.features.shape[0],):
            raise ValueError("selection mask must match rows")
        return VectorReliabilityDataset(
            features=self.features[selected],
            labels=self.labels[selected],
            label_mask=self.label_mask[selected],
            groups=self.groups[selected],
            episode_ids=self.episode_ids[selected],
            task_ids=self.task_ids[selected],
            source_steps=self.source_steps[selected],
            feature_names=self.feature_names,
            split=None if self.split is None else self.split[selected],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            label_mask=self.label_mask,
            groups=self.groups,
            episode_ids=self.episode_ids,
            task_ids=self.task_ids,
            source_steps=self.source_steps,
            feature_names=np.asarray(self.feature_names, dtype=str),
            split=np.asarray([] if self.split is None else self.split, dtype=str),
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorReliabilityDataset":
        with np.load(Path(path), allow_pickle=False) as data:
            split = data["split"]
            return cls(
                features=data["features"],
                labels=data["labels"],
                label_mask=data["label_mask"],
                groups=data["groups"],
                episode_ids=data["episode_ids"],
                task_ids=data["task_ids"],
                source_steps=data["source_steps"],
                feature_names=tuple(str(value) for value in data["feature_names"]),
                split=None if split.size == 0 else split,
            )


def build_vector_dataset(
    examples: Sequence[TemporalExample],
    target_batch: TargetBatch,
    *,
    feature_contract: CausalFeatureContract,
    horizon_dim: int,
    episode_split: EpisodeSplit | None = None,
    require_complete_curve: bool = True,
) -> VectorReliabilityDataset:
    """Group row-wise existing targets into causal vector-output samples."""

    if target_batch.labels is None:
        raise ValueError("binary target labels are required")
    if len(examples) != target_batch.labels.size:
        raise ValueError("examples and target labels must have matching lengths")
    if horizon_dim < 1:
        raise ValueError("horizon_dim must be positive")
    rows: dict[tuple[str, int, str], dict[str, object]] = {}
    for example, label in zip(examples, target_batch.labels):
        key = (str(example.episode_id), int(example.source_step), example.group)
        row = rows.setdefault(
            key,
            {
                "feature": feature_contract.encode_example(example),
                "labels": np.zeros(horizon_dim, dtype=np.float32),
                "mask": np.zeros(horizon_dim, dtype=bool),
                "task_id": "" if example.task_id is None else str(example.task_id),
            },
        )
        if example.offset >= horizon_dim:
            continue
        mask = row["mask"]
        labels = row["labels"]
        assert isinstance(mask, np.ndarray) and isinstance(labels, np.ndarray)
        if mask[example.offset]:
            raise ValueError("duplicate example for one source/group/offset")
        labels[example.offset] = float(label)
        mask[example.offset] = True

    if not rows:
        raise ValueError("no vector examples were constructed")
    if require_complete_curve and any(not np.asarray(row["mask"]).all() for row in rows.values()):
        raise ValueError("at least one source/group curve is missing an offset")
    split_by_episode: dict[str, str] = {}
    if episode_split is not None:
        for name, ids in episode_split.as_dict().items():
            for episode_id in ids:
                split_by_episode[str(episode_id)] = name

    ordered = sorted(rows.items())
    features = np.stack([row["feature"] for _, row in ordered]).astype(np.float32)
    labels = np.stack([row["labels"] for _, row in ordered]).astype(np.float32)
    masks = np.stack([row["mask"] for _, row in ordered]).astype(bool)
    keys = [key for key, _ in ordered]
    split = None
    if episode_split is not None:
        split = np.asarray([split_by_episode[key[0]] for key in keys], dtype=str)
    return VectorReliabilityDataset(
        features=features,
        labels=labels,
        label_mask=masks,
        groups=np.asarray([key[2] for key in keys], dtype=str),
        episode_ids=np.asarray([key[0] for key in keys], dtype=str),
        task_ids=np.asarray([row["task_id"] for _, row in ordered], dtype=str),
        source_steps=np.asarray([key[1] for key in keys], dtype=np.int64),
        feature_names=feature_contract.feature_names,
        split=split,
    )

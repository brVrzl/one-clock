"""Deterministic episode-only split manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Iterable, Mapping

from experiments.temporal_reliability_training.config import SplitConfig
from experiments.temporal_reliability_training.dataset import EpisodeSplit, split_episode_ids


@dataclass(frozen=True)
class EpisodeSplitManifest:
    """A serializable split containing episode IDs, never frame IDs."""

    seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    stratify_by_task: bool = True
    version: str = "episode_split_v1"

    @classmethod
    def create(
        cls,
        episode_ids: Iterable[str | int],
        *,
        task_by_episode: Mapping[str | int, str | int | None] | None = None,
        config: SplitConfig | None = None,
    ) -> "EpisodeSplitManifest":
        config = config or SplitConfig()
        split = split_episode_ids(
            episode_ids,
            task_by_episode=task_by_episode,
            config=config,
        )
        manifest = cls(
            seed=config.seed,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            train=tuple(str(value) for value in split.train),
            validation=tuple(str(value) for value in split.validation),
            test=tuple(str(value) for value in split.test),
            stratify_by_task=config.stratify_by_task,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        parts = (self.train, self.validation, self.test)
        if any(len(set(part)) != len(part) for part in parts):
            raise ValueError("each split must contain unique episode IDs")
        if set(self.train) & set(self.validation):
            raise ValueError("train and validation episodes overlap")
        if set(self.train) & set(self.test):
            raise ValueError("train and test episodes overlap")
        if set(self.validation) & set(self.test):
            raise ValueError("validation and test episodes overlap")
        if not self.train:
            raise ValueError("train split must not be empty")

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return self.train + self.validation + self.test

    def as_episode_split(self) -> EpisodeSplit:
        self.validate()
        return EpisodeSplit(self.train, self.validation, self.test)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed": self.seed,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "stratify_by_task": self.stratify_by_task,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeSplitManifest":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        manifest = cls(
            seed=int(values["seed"]),
            train_fraction=float(values["train_fraction"]),
            validation_fraction=float(values["validation_fraction"]),
            test_fraction=float(values["test_fraction"]),
            train=tuple(str(value) for value in values["train"]),
            validation=tuple(str(value) for value in values["validation"]),
            test=tuple(str(value) for value in values["test"]),
            stratify_by_task=bool(values.get("stratify_by_task", True)),
            version=str(values.get("version", "episode_split_v1")),
        )
        if manifest.version != "episode_split_v1":
            raise ValueError(f"unsupported split manifest version: {manifest.version!r}")
        manifest.validate()
        return manifest

"""Configuration objects for the preparation pipeline.

The paths identify the intended frozen ACT and LeRobot inputs.  They are not
opened during import, and the preparation pipeline never updates either one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_CHECKPOINT_PATH = Path(
    "/home/thor/projects/checkpoints/zeromidnight_act_libero_object"
)
DEFAULT_DATASET_PATH = Path(
    "/home/thor/datasets/libero_object_25_08_23_lerobotv2.1"
)


@dataclass(frozen=True)
class ExperimentPaths:
    """External inputs for the frozen-policy study."""

    checkpoint: Path = DEFAULT_CHECKPOINT_PATH
    dataset: Path = DEFAULT_DATASET_PATH


@dataclass(frozen=True)
class SplitConfig:
    """Episode-level split policy.

    Splitting is performed before windows/examples are generated.  The
    default preserves task representation in every split while keeping
    episodes disjoint.  ``stratify_by_task=False`` is useful for an explicitly
    task-held-out study.
    """

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 20260820
    stratify_by_task: bool = True

    def validate(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(fraction < 0.0 for fraction in fractions):
            raise ValueError("split fractions must be non-negative")
        if abs(sum(fractions) - 1.0) > 1e-8:
            raise ValueError("split fractions must sum to one")
        if self.train_fraction == 0.0:
            raise ValueError("a training split is required")


@dataclass(frozen=True)
class FeatureConfig:
    """Feature layout for the initial estimator interface.

    The default observation dimension is a zero-filled placeholder.  A future
    frozen-ACT observation encoder can replace it without changing the rest of
    the feature schema by setting ``observation_embedding_dim`` and supplying
    source-time embeddings.
    """

    observation_embedding_dim: int = 16
    max_offset: int = 100
    include_group_one_hot: bool = True
    include_offset: bool = True

    def validate(self) -> None:
        if self.observation_embedding_dim < 0:
            raise ValueError("observation_embedding_dim must be non-negative")
        if self.max_offset < 1:
            raise ValueError("max_offset must be positive")


@dataclass(frozen=True)
class TargetConfig:
    """Target generation settings.

    No threshold is supplied by default.  Thresholds are a validation-set
    decision and must be passed explicitly before binary labels are generated.
    """

    mode: Literal["fresh_policy", "demonstration"] = "fresh_policy"
    inclusive: bool = True
    threshold_by_group: dict[str, float] | None = None

    def validate(self) -> None:
        if self.mode not in {"fresh_policy", "demonstration"}:
            raise ValueError(f"unsupported target mode: {self.mode!r}")
        if self.threshold_by_group is not None:
            if any(not isinstance(name, str) for name in self.threshold_by_group):
                raise ValueError("target threshold group names must be strings")
            if any(
                threshold < 0.0
                for threshold in self.threshold_by_group.values()
            ):
                raise ValueError("target thresholds must be non-negative")

"""Configuration for adaptive reliability experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_PATH = Path(
    "/home/thor/datasets/libero_object_25_08_23_lerobotv2.1"
)
DEFAULT_CHECKPOINT_PATH = Path(
    "/home/thor/projects/checkpoints/zeromidnight_act_libero_object"
)


@dataclass(frozen=True)
class DynamicHorizonConfig:
    """Decode a reliability curve into a valid execution horizon."""

    threshold_tau: float = 0.50
    min_horizon: int = 1
    max_horizon: int | None = None
    require_prefix: bool = True

    def validate(self, *, chunk_size: int | None = None) -> None:
        if not 0.0 <= self.threshold_tau <= 1.0:
            raise ValueError("threshold_tau must be in [0, 1]")
        if self.min_horizon < 1:
            raise ValueError("min_horizon must be positive")
        if self.max_horizon is not None and self.max_horizon < self.min_horizon:
            raise ValueError("max_horizon must be >= min_horizon")
        if chunk_size is not None:
            maximum = self.max_horizon or chunk_size
            if maximum > chunk_size:
                raise ValueError("max_horizon cannot exceed chunk_size")
            if self.min_horizon > chunk_size:
                raise ValueError("min_horizon cannot exceed chunk_size")


@dataclass(frozen=True)
class TrainingConfig:
    """Small BCE training configuration for the auxiliary reliability head."""

    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    seed: int = 20260820
    hidden_dims: tuple[int, ...] = (128, 64)
    device: str = "cpu"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["hidden_dims"] = list(self.hidden_dims)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        values = dict(values)
        if "hidden_dims" in values:
            values["hidden_dims"] = tuple(int(width) for width in values["hidden_dims"])
        config = cls(**values)
        config.validate()
        return config

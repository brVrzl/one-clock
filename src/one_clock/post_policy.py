"""Lightweight post-policy action-chunk correction modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PostPolicyResult:
    """Corrected chunk and diagnostics logged by the rollout runner."""

    action_chunk: np.ndarray
    correction_norm: float
    correction_max_abs: float
    correction_fraction: float
    gate_active: bool

    def as_log_record(self) -> dict[str, float | bool]:
        return {
            "correction_norm": self.correction_norm,
            "correction_max_abs": self.correction_max_abs,
            "correction_fraction": self.correction_fraction,
            "gate_active": self.gate_active,
        }


def _result(base: np.ndarray, corrected: np.ndarray, *, gate_active: bool) -> PostPolicyResult:
    correction = corrected - base
    base_norm = float(np.linalg.norm(base))
    return PostPolicyResult(
        action_chunk=corrected,
        correction_norm=float(np.linalg.norm(correction)),
        correction_max_abs=float(np.max(np.abs(correction))),
        correction_fraction=float(np.linalg.norm(correction) / max(base_norm, 1e-12)),
        gate_active=gate_active,
    )


class IdentityPostPolicy:
    """Frozen-policy baseline."""

    def __call__(
        self,
        *,
        state: np.ndarray,
        action_chunk: np.ndarray,
        task_id: int,
    ) -> PostPolicyResult:
        del state, task_id
        chunk = np.asarray(action_chunk)
        return _result(chunk, chunk.copy(), gate_active=False)


class ExponentialChunkSmoother:
    """Causal exponential smoothing along the predicted chunk."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)

    def __call__(
        self,
        *,
        state: np.ndarray,
        action_chunk: np.ndarray,
        task_id: int,
    ) -> PostPolicyResult:
        del state, task_id
        base = np.asarray(action_chunk)
        corrected = base.copy()
        for position in range(1, len(corrected)):
            corrected[position] = (
                self.alpha * base[position]
                + (1.0 - self.alpha) * corrected[position - 1]
            )
        return _result(base, corrected, gate_active=True)


class GripperTimingShift:
    """Shift only the gripper sequence while preserving every arm value."""

    def __init__(self, shift_steps: int, gripper_dimension: int = 6) -> None:
        self.shift_steps = int(shift_steps)
        self.gripper_dimension = int(gripper_dimension)

    def __call__(
        self,
        *,
        state: np.ndarray,
        action_chunk: np.ndarray,
        task_id: int,
    ) -> PostPolicyResult:
        del state, task_id
        base = np.asarray(action_chunk)
        if not 0 <= self.gripper_dimension < base.shape[1]:
            raise ValueError(
                f"gripper dimension {self.gripper_dimension} outside [0, {base.shape[1]})"
            )
        corrected = base.copy()
        source_indices = np.arange(len(base)) - self.shift_steps
        source_indices = np.clip(source_indices, 0, len(base) - 1)
        corrected[:, self.gripper_dimension] = base[source_indices, self.gripper_dimension]
        arm_dimensions = np.arange(base.shape[1]) != self.gripper_dimension
        if not np.array_equal(corrected[:, arm_dimensions], base[:, arm_dimensions]):
            raise AssertionError("gripper timing shift modified an arm action")
        return _result(base, corrected, gate_active=self.shift_steps != 0)


class AffineResidualCalibrator:
    """Ridge-fit residual correction from state, task, action, and chunk phase."""

    def __init__(
        self,
        *,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        weights: np.ndarray,
        task_count: int,
        correction_scale: float = 1.0,
        gate_threshold: float | None = None,
        correction_dimensions: tuple[int, ...] | None = None,
    ) -> None:
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.task_count = int(task_count)
        self.correction_scale = float(correction_scale)
        self.gate_threshold = gate_threshold
        self.correction_dimensions = correction_dimensions

    @staticmethod
    def features(
        action_chunk: np.ndarray,
        state: np.ndarray,
        task_id: int,
        task_count: int,
    ) -> np.ndarray:
        chunk = np.asarray(action_chunk, dtype=np.float64)
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if chunk.ndim != 2:
            raise ValueError(f"action_chunk must be rank 2, got {chunk.shape}")
        if not 0 <= task_id < task_count:
            raise ValueError(f"task_id {task_id} outside [0, {task_count})")
        positions = np.arange(len(chunk), dtype=np.float64)
        phase = positions / max(len(chunk) - 1, 1)
        task = np.zeros((len(chunk), task_count), dtype=np.float64)
        task[:, task_id] = 1.0
        return np.concatenate(
            (
                chunk,
                np.repeat(state[None], len(chunk), axis=0),
                phase[:, None],
                np.square(phase)[:, None],
                np.sin(np.pi * phase)[:, None],
                task,
            ),
            axis=1,
        )

    @classmethod
    def fit(
        cls,
        *,
        action: np.ndarray,
        state: np.ndarray,
        position: np.ndarray,
        task_id: np.ndarray,
        target: np.ndarray,
        chunk_size: int,
        task_count: int,
        ridge: float = 10.0,
    ) -> "AffineResidualCalibrator":
        action = np.asarray(action, dtype=np.float64)
        state = np.asarray(state, dtype=np.float64)
        position = np.asarray(position, dtype=np.float64)
        task_id = np.asarray(task_id, dtype=np.int64)
        target = np.asarray(target, dtype=np.float64)
        phase = position / max(chunk_size - 1, 1)
        task = np.eye(task_count, dtype=np.float64)[task_id]
        features = np.concatenate(
            (
                action,
                state,
                phase[:, None],
                np.square(phase)[:, None],
                np.sin(np.pi * phase)[:, None],
                task,
            ),
            axis=1,
        )
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-12] = 1.0
        standardized = (features - mean) / scale
        design = np.concatenate((standardized, np.ones((len(features), 1))), axis=1)
        regularizer = ridge * np.eye(design.shape[1])
        weights = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ (target - action),
        )
        return cls(
            feature_mean=mean,
            feature_scale=scale,
            weights=weights,
            task_count=task_count,
        )

    def predict_residual(
        self,
        *,
        state: np.ndarray,
        action_chunk: np.ndarray,
        task_id: int,
    ) -> np.ndarray:
        features = self.features(action_chunk, state, task_id, self.task_count)
        standardized = (features - self.feature_mean) / self.feature_scale
        design = np.concatenate((standardized, np.ones((len(features), 1))), axis=1)
        return design @ self.weights

    def __call__(
        self,
        *,
        state: np.ndarray,
        action_chunk: np.ndarray,
        task_id: int,
    ) -> PostPolicyResult:
        base = np.asarray(action_chunk)
        residual = self.predict_residual(
            state=state,
            action_chunk=base,
            task_id=task_id,
        )
        if self.correction_dimensions is not None:
            selected = np.zeros(base.shape[1], dtype=bool)
            for dimension in self.correction_dimensions:
                if not 0 <= dimension < base.shape[1]:
                    raise ValueError(
                        f"correction dimension {dimension} outside [0, {base.shape[1]})"
                    )
                selected[dimension] = True
            residual = residual * selected[None]
        active = self.gate_threshold is None or float(np.linalg.norm(residual)) > self.gate_threshold
        corrected = base + self.correction_scale * residual if active else base.copy()
        return _result(base, corrected, gate_active=active)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            weights=self.weights,
            task_count=np.asarray(self.task_count),
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        correction_scale: float = 1.0,
        gate_threshold: float | None = None,
        correction_dimensions: tuple[int, ...] | None = None,
    ) -> "AffineResidualCalibrator":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                feature_mean=data["feature_mean"],
                feature_scale=data["feature_scale"],
                weights=data["weights"],
                task_count=int(data["task_count"]),
                correction_scale=correction_scale,
                gate_threshold=gate_threshold,
                correction_dimensions=correction_dimensions,
            )

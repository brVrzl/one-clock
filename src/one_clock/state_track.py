"""Training-free feedback indexing for a frozen LIBERO action chunk.

The tracker does not change any action values.  It only chooses which row of
the already predicted chunk to execute from the measured end-effector state.
LIBERO's current ACT environment uses OSC_POSE delta actions, so the nominal
trajectory is reconstructed by integrating the controller's documented
position / axis-angle scales from the query-time pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ProgressDiagnostics:
    """Per-control-tick evidence emitted by :class:`StateTrackChunk`."""

    progress_index: int
    selected_index: int
    nearest_index: int
    tracking_error: float
    repeated: bool
    skipped: int

    def as_log_record(self) -> dict[str, object]:
        return {
            "progress_index": self.progress_index,
            "selected_index": self.selected_index,
            "nearest_index": self.nearest_index,
            "tracking_error": self.tracking_error,
            "repeated": self.repeated,
            "skipped": self.skipped,
        }


def _current_eef_state(observation: dict[str, object]) -> np.ndarray:
    """Return [xyz, axis-angle] from the raw LeRobot LIBERO observation."""

    eef = observation["robot_state"]["eef"]  # type: ignore[index]
    position = np.asarray(eef["pos"], dtype=np.float64)  # type: ignore[index]
    quaternion_xyzw = np.asarray(eef["quat"], dtype=np.float64)  # type: ignore[index]
    if position.shape != (3,) or quaternion_xyzw.shape != (4,):
        raise ValueError("LIBERO EEF observation must contain pos(3) and quat(4)")
    return np.concatenate((position, Rotation.from_quat(quaternion_xyzw).as_rotvec()))


def nominal_eef_trajectory(
    observation: dict[str, object],
    action_chunk: np.ndarray,
    *,
    active_horizon: int,
) -> np.ndarray:
    """Integrate the first ``active_horizon`` OSC_POSE delta actions.

    The installed robosuite OSC controller maps normalized actions in [-1, 1]
    to translational deltas of +/-0.05 m and rotational deltas of +/-0.5 rad,
    and left-multiplies the rotation increment onto the current orientation.
    """

    chunk = np.asarray(action_chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] < 6:
        raise ValueError(f"expected action chunk [T, >=6], got {chunk.shape}")
    horizon = min(int(active_horizon), chunk.shape[0])
    if horizon < 1:
        raise ValueError("active_horizon must be positive")
    current = _current_eef_state(observation)
    position = current[:3].copy()
    rotation = Rotation.from_rotvec(current[3:])
    position_scale = np.array([0.05, 0.05, 0.05], dtype=np.float64)
    rotation_scale = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    trajectory = np.empty((horizon, 6), dtype=np.float64)
    for index in range(horizon):
        delta = np.clip(chunk[index, :6], -1.0, 1.0)
        position = position + delta[:3] * position_scale
        rotation = Rotation.from_rotvec(delta[3:6] * rotation_scale) * rotation
        trajectory[index, :3] = position
        trajectory[index, 3:] = rotation.as_rotvec()
    return trajectory


class StateTrackChunk:
    """Select action rows by monotonic nearest-state progress.

    ``lookahead`` is intentionally restricted to the two values used by the
    causal gate.  ``max_forward_skip`` bounds progress movement per control
    tick; a lagging robot therefore repeats the current row rather than being
    forced to advance by wall-clock time.
    """

    def __init__(
        self,
        *,
        lookahead: int,
        active_horizon: int,
        max_forward_skip: int = 1,
    ) -> None:
        if lookahead not in (1, 2):
            raise ValueError("StateTrack lookahead must be 1 or 2")
        if active_horizon < 1 or max_forward_skip < 1:
            raise ValueError("active_horizon and max_forward_skip must be positive")
        self.lookahead = int(lookahead)
        self.active_horizon = int(active_horizon)
        self.max_forward_skip = int(max_forward_skip)
        self._chunk: np.ndarray | None = None
        self._nominal: np.ndarray | None = None
        self._progress = -1

    def start_chunk(self, observation: dict[str, object], action_chunk: np.ndarray) -> None:
        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] < 1:
            raise ValueError(f"expected action chunk [T, D], got {chunk.shape}")
        horizon = min(self.active_horizon, chunk.shape[0])
        self._chunk = chunk.copy()
        self._nominal = nominal_eef_trajectory(
            observation, chunk, active_horizon=horizon
        )
        self._progress = -1

    def select(self, observation: dict[str, object]) -> tuple[np.ndarray, ProgressDiagnostics]:
        if self._chunk is None or self._nominal is None:
            raise RuntimeError("start_chunk must be called before select")
        current = _current_eef_state(observation)
        # Normalize position and orientation by one controller action scale so
        # neither component dominates purely due to units.
        scale = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        distances = np.linalg.norm((self._nominal - current) / scale, axis=1)
        nearest = int(np.argmin(distances))
        previous = self._progress
        lower = max(0, previous)
        upper = min(self._nominal.shape[0] - 1, previous + self.max_forward_skip)
        progress = min(max(nearest, lower), upper)
        if previous < 0:
            progress = min(nearest, self.max_forward_skip - 1)
        selected = min(progress + self.lookahead, self._chunk.shape[0] - 1)
        diagnostics = ProgressDiagnostics(
            progress_index=int(progress),
            selected_index=int(selected),
            nearest_index=nearest,
            tracking_error=float(distances[progress]),
            repeated=previous >= 0 and progress == previous,
            skipped=max(0, progress - previous - 1) if previous >= 0 else 0,
        )
        self._progress = progress
        return self._chunk[selected].copy(), diagnostics


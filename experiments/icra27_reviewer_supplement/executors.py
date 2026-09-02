"""Frozen dense-query same-target executors."""

from __future__ import annotations

import re
import numpy as np


def checked(chunk: np.ndarray) -> np.ndarray:
    value = np.asarray(chunk, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 7 or not np.isfinite(value).all():
        raise ValueError(f"invalid policy chunk {value.shape}")
    return value.copy()


class DenseExecutor:
    def __init__(self, method: str):
        self.method = method
        self.chunks: list[np.ndarray] = []

    def step(self, t: int, chunk: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        chunk = checked(chunk)
        self.chunks.append(chunk)
        fresh = chunk[0].copy()
        action = fresh.copy()
        source = {"translation_q": t, "translation_k": 0, "rotation_q": t,
                  "rotation_k": 0, "gripper_q": t, "gripper_k": 0}

        def stale(indices: slice | tuple[int, ...], age: int, label: str) -> None:
            if t < age:
                return
            old = self.chunks[t - age][age]
            action[indices] = old[indices]
            source[f"{label}_q"] = t - age
            source[f"{label}_k"] = age

        if self.method in {"C01", "C11"}:
            q = 16 * (t // 16)
            k = t - q
            scheduled = self.chunks[q][k]
            if self.method == "C11":
                action[:] = scheduled
                for label in ("translation", "rotation", "gripper"):
                    source[f"{label}_q"], source[f"{label}_k"] = q, k
            else:
                action[6] = scheduled[6]
                source["gripper_q"], source["gripper_k"] = q, k
        elif self.method == "T20_R0_G0":
            stale(slice(0, 3), 20, "translation")
        elif self.method == "T0_R20_G0":
            stale(slice(3, 6), 20, "rotation")
        else:
            arm_match = re.fullmatch(r"A(\d+)_G(\d+)", self.method)
            if not arm_match:
                raise ValueError(f"unknown dense condition {self.method}")
            arm, grip = (int(arm_match.group(1)), int(arm_match.group(2)))
            if arm:
                stale(slice(0, 3), arm, "translation")
                stale(slice(3, 6), arm, "rotation")
            if grip:
                stale((6,), grip, "gripper")
        for label in ("translation", "rotation", "gripper"):
            if source[f"{label}_q"] + source[f"{label}_k"] != t:
                raise RuntimeError("same-target q+k=t violation")
        return action, source

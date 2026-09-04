"""Frozen five-condition executor definitions for the Phase-1 discriminator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from one_clock import ActionGroup, ExecutionDecision, FixedChunkExecutor


ARM = tuple(range(6))
GRIPPER = (6,)
ACTION_DIM = 7
CONDITION_ORDER = (
    "H4",
    "ARM4_GRIP32",
    "H8",
    "ARM8_GRIP32",
    "H16",
    "ARM8_GRIP16",
    "ZOH8_GRIP16",
)


@dataclass(frozen=True)
class Condition:
    name: str
    strategy: str
    arm_horizon: int
    gripper_horizon: int


CONDITIONS = {
    "H4": Condition("H4", "global_fixed", 4, 4),
    "ARM4_GRIP32": Condition("ARM4_GRIP32", "groupwise_fixed", 4, 32),
    "H8": Condition("H8", "global_fixed", 8, 8),
    "ARM8_GRIP32": Condition("ARM8_GRIP32", "groupwise_fixed", 8, 32),
    "H16": Condition("H16", "global_fixed", 16, 16),
    "ARM8_GRIP16": Condition("ARM8_GRIP16", "groupwise_fixed", 8, 16),
    "ZOH8_GRIP16": Condition("ZOH8_GRIP16", "zoh_gripper", 8, 16),
}


class ZOH8Grip16Executor:
    """Fresh arm8 chunks with a scalar gripper command held for 16 steps."""

    def __init__(self, chunk_size: int) -> None:
        if chunk_size < 16:
            raise ValueError("ZOH8_GRIP16 requires a chunk length of at least 16")
        self.chunk_size = int(chunk_size)
        self.reset()

    def reset(self) -> None:
        self.environment_step = 0
        self.next_chunk_id = 0
        self.arm_chunk: np.ndarray | None = None
        self.arm_chunk_id: int | None = None
        self.arm_query_step: int | None = None
        self.gripper_value: float | None = None
        self.gripper_chunk_id: int | None = None
        self.gripper_query_step: int | None = None

    def step(self, query_policy: Callable[[], np.ndarray]) -> ExecutionDecision:
        t = self.environment_step
        queried = t % 8 == 0
        if queried:
            chunk = np.asarray(query_policy())
            if chunk.shape != (self.chunk_size, ACTION_DIM) or not np.isfinite(chunk).all():
                raise ValueError("policy chunk has invalid shape or values")
            self.arm_chunk = chunk.copy()
            self.arm_chunk_id = self.next_chunk_id
            self.next_chunk_id += 1
            self.arm_query_step = t
        assert self.arm_chunk is not None and self.arm_chunk_id is not None and self.arm_query_step is not None
        arm_position = t - self.arm_query_step
        if not 0 <= arm_position < 8:
            raise RuntimeError("ZOH arm position left the fresh eight-step prefix")

        gripper_refreshed = t % 16 == 0
        if gripper_refreshed:
            if not queried:
                raise RuntimeError("gripper boundary lacks a fresh policy query")
            self.gripper_value = float(self.arm_chunk[0, 6])
            self.gripper_chunk_id = self.arm_chunk_id
            self.gripper_query_step = t
        assert self.gripper_value is not None and self.gripper_chunk_id is not None and self.gripper_query_step is not None

        action = self.arm_chunk[arm_position].copy()
        action[6] = self.gripper_value
        gripper_age = t - self.gripper_query_step
        decision = ExecutionDecision(
            environment_step=t,
            action=action,
            policy_query=queried,
            new_chunk_id=self.arm_chunk_id if queried else None,
            source_chunk_ids={"arm": self.arm_chunk_id, "gripper": self.gripper_chunk_id},
            source_ages={"arm": arm_position, "gripper": gripper_age},
            source_positions={"arm": arm_position, "gripper": 0},
            remaining_commitments={"arm": 8 - arm_position, "gripper": 16 - gripper_age},
            refreshed_groups=("arm", "gripper") if gripper_refreshed else (("arm",) if queried else ()),
            configured_horizons={"arm": 8, "gripper": 16},
        )
        self.environment_step += 1
        return decision


def make_fixed_executor(name: str, chunk_size: int) -> FixedChunkExecutor | ZOH8Grip16Executor:
    condition = CONDITIONS[name]
    if name == "ZOH8_GRIP16":
        return ZOH8Grip16Executor(chunk_size)
    groups = (
        ActionGroup("arm", ARM, condition.arm_horizon),
        ActionGroup("gripper", GRIPPER, condition.gripper_horizon),
    )
    if condition.strategy == "global_fixed":
        return FixedChunkExecutor.global_fixed(
            action_dim=ACTION_DIM,
            chunk_size=chunk_size,
            horizon=condition.arm_horizon,
            groups=groups,
        )
    return FixedChunkExecutor.groupwise_fixed(
        action_dim=ACTION_DIM,
        chunk_size=chunk_size,
        groups=groups,
    )


__all__ = [
    "ACTION_DIM",
    "ARM",
    "GRIPPER",
    "CONDITION_ORDER",
    "CONDITIONS",
    "ZOH8Grip16Executor",
    "make_fixed_executor",
]

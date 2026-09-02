"""Frozen Track-A execution conditions and their query schedules."""

from __future__ import annotations

from dataclasses import dataclass

from one_clock import ActionGroup, FixedChunkExecutor


ARM = tuple(range(6))
GRIPPER = (6,)
ACTION_DIM = 7
TE_COEFFICIENT = 0.01
CONDITION_ORDER = (
    "H16",
    "H4",
    "ARM4_GRIP32",
    "H2",
    "ARM2_GRIP16",
    "TE_DENSE",
)


@dataclass(frozen=True)
class Condition:
    name: str
    strategy: str
    arm_horizon: int | None
    gripper_horizon: int | None


CONDITIONS = {
    "H16": Condition("H16", "global_fixed", 16, 16),
    "H4": Condition("H4", "global_fixed", 4, 4),
    "ARM4_GRIP32": Condition("ARM4_GRIP32", "groupwise_fixed", 4, 32),
    "H2": Condition("H2", "global_fixed", 2, 2),
    "ARM2_GRIP16": Condition("ARM2_GRIP16", "groupwise_fixed", 2, 16),
    "TE_DENSE": Condition("TE_DENSE", "canonical_act_temporal_ensemble", None, None),
}


def make_fixed_executor(name: str, chunk_size: int) -> FixedChunkExecutor:
    condition = CONDITIONS[name]
    if condition.arm_horizon is None or condition.gripper_horizon is None:
        raise ValueError(f"{name} is not a fixed-chunk condition")
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

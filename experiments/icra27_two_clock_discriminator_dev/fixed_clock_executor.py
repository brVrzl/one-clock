"""The two fixed-clock conditions authorized for this development discriminator."""

from __future__ import annotations

from one_clock import ActionGroup, FixedChunkExecutor


ACTION_DIM = 7
CHUNK_LENGTH = 100
H16 = 16
H32 = 32
ARM = tuple(range(6))
GRIPPER = (6,)
H32_COHERENT = "H32_COHERENT"
TWO_CLOCK = "TWO_CLOCK_ARM16_GRIP32"
METHODS = (H32_COHERENT, TWO_CLOCK)


def make_executor(method: str) -> FixedChunkExecutor:
    groups = (
        ActionGroup("arm", ARM, H32 if method == H32_COHERENT else H16),
        ActionGroup("gripper", GRIPPER, H32),
    )
    if method == H32_COHERENT:
        return FixedChunkExecutor.global_fixed(
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_LENGTH,
            horizon=H32,
            groups=groups,
        )
    if method == TWO_CLOCK:
        return FixedChunkExecutor.groupwise_fixed(
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_LENGTH,
            groups=groups,
        )
    raise ValueError(f"unknown fixed-clock method: {method}")


__all__ = [
    "ACTION_DIM",
    "CHUNK_LENGTH",
    "H16",
    "H32",
    "ARM",
    "GRIPPER",
    "H32_COHERENT",
    "TWO_CLOCK",
    "METHODS",
    "make_executor",
]

"""Policy-cache adapter for the pinned SmolVLA LIBERO checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from smolvla_runtime import CHECKPOINT_REPO_ID, CHECKPOINT_REVISION, infer_full_chunk, load_runtime


class SmolVLAAdapter:
    policy_id = CHECKPOINT_REPO_ID

    def __init__(self) -> None:
        self.dataset_root = Path(os.environ["ONECLOCK_SMOLVLA_DATASET_ROOT"]).resolve()
        self.runtime = load_runtime(self.dataset_root, os.environ.get("ONECLOCK_SMOLVLA_DEVICE"))
        self.metadata = {
            "checkpoint_repo_id": CHECKPOINT_REPO_ID,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "device": self.runtime["device"],
            "chunk_length": int(self.runtime["policy_config"].chunk_size),
            "action_normalization": "pinned policy_preprocessor/postprocessor",
        }

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {"actions": infer_full_chunk(self.runtime, observation)}


def make_adapter() -> SmolVLAAdapter:
    return SmolVLAAdapter()

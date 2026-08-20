"""Shared constants and small I/O helpers for the LIBERO-4 data foundation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


DATASET_REPO_ID = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
K_MAX = 100
FPS = 10.0
SPLIT_RULE_VERSION = "sha256-suite-task-episode-v1"

# This mapping was verified by matching every pinned dataset task description
# against the installed LIBERO benchmark map.  The dataset task order is not
# the benchmark-map order, so this must remain explicit and audited.
TASK_SUITE_BY_INDEX: dict[int, str] = {
    **{index: "LIBERO-Long" for index in range(0, 10)},
    **{index: "LIBERO-Goal" for index in range(10, 20)},
    **{index: "LIBERO-Object" for index in range(20, 30)},
    **{index: "LIBERO-Spatial" for index in range(30, 40)},
}

GROUPS = {
    "arm": {"indices": [0, 1, 2, 3, 4, 5], "distance": "max(translation_normalized_rms, rotation_normalized_rms)"},
    "gripper": {"indices": [6], "distance": "normalized_absolute_error; sign_match is an additional validity criterion"},
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def suite_for_task(task_index: int) -> str:
    try:
        return TASK_SUITE_BY_INDEX[int(task_index)]
    except KeyError as exc:
        raise ValueError(f"Unmapped LIBERO task index: {task_index}") from exc


def split_for_episode(suite: str, task_index: int, episode_index: int) -> str:
    """Deterministic task-stratified 80/10/10 episode split."""

    key = f"{SPLIT_RULE_VERSION}|{suite}|{int(task_index)}|{int(episode_index)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "validation"
    return "test"

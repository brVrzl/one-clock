from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from build_reliability_targets import build_labels  # noqa: E402
from dataset_common import split_for_episode  # noqa: E402
from policy_cache import atomic_save_npz, valid_cache_shard  # noqa: E402


def test_episode_split_is_deterministic_and_partitioned() -> None:
    values = [split_for_episode("LIBERO-Object", 20, episode) for episode in range(100)]
    assert values == [split_for_episode("LIBERO-Object", 20, episode) for episode in range(100)]
    assert set(values) == {"train", "validation", "test"}


def test_future_label_is_censored_and_raw_distance_survives() -> None:
    old = np.zeros((1, 3, 7), dtype=np.float32)
    fresh = np.zeros_like(old)
    fresh[0, 0, :3] = 0.335523718
    fresh[0, 1, 6] = 1.0
    observed = np.asarray([[True, True, False]])
    labels = build_labels(old, fresh, observed, np.asarray([0.335523718, 0.335523718, 0.335523718, 1, 1, 1, 1], dtype=np.float32))
    assert labels["raw_group_distances"].shape == (1, 3, 2)
    assert np.isclose(labels["raw_group_distances"][0, 0, 0], 1.0)
    assert bool(labels["label_observed"][0, 2, 0]) is False
    assert bool(labels["Y_refresh"][0, 2, 0]) is False


def test_cache_shard_is_atomic_and_valid(tmp_path: Path) -> None:
    path = tmp_path / "shard-0000.npz"
    atomic_save_npz(path, {
        "frame_id": np.asarray([0, 1], dtype=np.int64),
        "episode_id": np.asarray([0, 0], dtype=np.int32),
        "frame_index": np.asarray([0, 1], dtype=np.int32),
        "task_id": np.asarray([0, 0], dtype=np.int16),
        "source_chunks": np.zeros((2, 20, 7), dtype=np.float32),
    })
    assert valid_cache_shard(path, expected_frame_count=2)

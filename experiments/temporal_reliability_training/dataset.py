"""Episode-safe splitting, manifest indexing, and example construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from .config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATASET_PATH,
    SplitConfig,
)
from .schema import FrozenTrajectory, GroupSpec, TemporalExample


@dataclass(frozen=True)
class EpisodeManifest:
    """Small, raw-dataset index entry; no trajectory payload is loaded."""

    episode_id: str | int
    length: int | None
    task_id: str | int | None
    data_files: tuple[str, ...] = ()
    video_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetManifest:
    """Metadata needed to reproduce the frozen-policy data source."""

    dataset_path: Path
    checkpoint_path: Path
    info: Mapping[str, object]
    episodes: tuple[EpisodeManifest, ...]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        return (str(value),)
    return tuple(str(item) for item in value)


def build_lerobot_manifest(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> DatasetManifest:
    """Index a LeRobot v2.1 dataset without loading frames or running ACT.

    This is deliberately limited to metadata.  Frozen prediction chunks and
    source-time embeddings are supplied later as ``FrozenTrajectory`` records,
    so this function cannot accidentally train on raw future observations.
    """

    root = Path(dataset_path)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    if not isinstance(info, dict):
        raise ValueError("LeRobot info.json must contain an object")

    episode_rows: list[dict[str, object]] = []
    episodes_path = root / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        episode_rows = _read_jsonl(episodes_path)

    episodes: list[EpisodeManifest] = []
    for row_number, row in enumerate(episode_rows):
        episode_id = row.get("episode_index", row.get("episode_id", row_number))
        raw_length = row.get("length", row.get("dataset_num_frames"))
        length = int(raw_length) if raw_length is not None else None
        raw_task = row.get("task_index", row.get("task_id"))
        if isinstance(raw_task, list):
            task_id = str(raw_task[0]) if raw_task else None
        else:
            task_id = raw_task  # type: ignore[assignment]
        episodes.append(
            EpisodeManifest(
                episode_id=episode_id,  # type: ignore[arg-type]
                length=length,
                task_id=task_id,  # type: ignore[arg-type]
                data_files=_string_tuple(row.get("data_files")),
                video_files=_string_tuple(row.get("video_files")),
            )
        )

    return DatasetManifest(
        dataset_path=root,
        checkpoint_path=Path(checkpoint_path),
        info=info,
        episodes=tuple(episodes),
    )


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[str | int, ...]
    validation: tuple[str | int, ...]
    test: tuple[str | int, ...]

    def as_dict(self) -> dict[str, tuple[str | int, ...]]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def _allocate_counts(total: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.asarray(fractions, dtype=np.float64) * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(int(count) for count in counts)  # type: ignore[return-value]


def split_episode_ids(
    episode_ids: Iterable[str | int],
    *,
    task_by_episode: Mapping[str | int, str | int | None] | None = None,
    config: SplitConfig | None = None,
) -> EpisodeSplit:
    """Make a deterministic, disjoint episode split before window creation."""

    config = config or SplitConfig()
    config.validate()
    raw_ids = list(episode_ids)
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("episode identifiers must be hashable and unique")
    unique_ids = raw_ids
    if not unique_ids:
        raise ValueError("at least one episode is required")

    if config.stratify_by_task and task_by_episode is not None:
        buckets: dict[str, list[str | int]] = {}
        for episode_id in unique_ids:
            task = task_by_episode.get(episode_id)
            key = "<none>" if task is None else str(task)
            buckets.setdefault(key, []).append(episode_id)
        ordered_buckets = [buckets[key] for key in sorted(buckets)]
    else:
        ordered_buckets = [unique_ids]

    rng = np.random.default_rng(config.seed)
    parts: list[list[str | int]] = [[], [], []]
    fractions = (
        config.train_fraction,
        config.validation_fraction,
        config.test_fraction,
    )
    for bucket in ordered_buckets:
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        counts = _allocate_counts(len(shuffled), fractions)
        cursor = 0
        for part_index, count in enumerate(counts):
            parts[part_index].extend(shuffled[cursor : cursor + count])
            cursor += count

    return EpisodeSplit(
        train=tuple(parts[0]),
        validation=tuple(parts[1]),
        test=tuple(parts[2]),
    )


class TemporalReliabilityDatasetBuilder:
    """Turn frozen trajectories into source/group/offset examples.

    A ``fresh_policy`` example is included only when a chunk was materialized
    at the future observation step ``source_step + offset``.  A
    ``demonstration`` example only needs a demonstrated action at that step.
    Missing future records are skipped and counted in ``last_build_summary``;
    they are never forward-filled from a future or neighboring observation.
    """

    def __init__(
        self,
        *,
        groups: Sequence[GroupSpec],
        offsets: Sequence[int] = tuple(range(100)),
        target_mode: str = "fresh_policy",
    ) -> None:
        if not groups:
            raise ValueError("at least one group is required")
        if len({group.name for group in groups}) != len(groups):
            raise ValueError("group names must be unique")
        normalized_offsets = tuple(int(offset) for offset in offsets)
        if any(offset < 0 for offset in normalized_offsets):
            raise ValueError("offsets must be non-negative")
        if len(set(normalized_offsets)) != len(normalized_offsets):
            raise ValueError("offsets must be unique")
        if target_mode not in {"fresh_policy", "demonstration"}:
            raise ValueError(f"unsupported target mode: {target_mode!r}")
        self.groups = tuple(groups)
        self.offsets = normalized_offsets
        self.target_mode = target_mode
        self.last_build_summary: dict[str, int] = {}

    def _validate_action_partition(self, action_dim: int) -> None:
        indices = [index for group in self.groups for index in group.indices]
        if sorted(indices) != list(range(action_dim)):
            raise ValueError(
                "groups must partition every action dimension exactly once"
            )

    def build(self, trajectories: Sequence[FrozenTrajectory]) -> tuple[TemporalExample, ...]:
        if not trajectories:
            raise ValueError("at least one trajectory is required")
        examples: list[TemporalExample] = []
        skipped_no_future = 0
        skipped_short_chunk = 0
        skipped_no_demo = 0
        action_dim: int | None = None

        for trajectory in trajectories:
            if action_dim is None:
                action_dim = trajectory.action_dim
                self._validate_action_partition(action_dim)
            elif trajectory.action_dim != action_dim:
                raise ValueError("all trajectories must share an action dimension")
            for source_step in trajectory.source_steps:
                source_chunk = trajectory.policy_chunks[source_step]
                source_embedding = trajectory.observation_embedding_at(source_step)
                for offset in self.offsets:
                    future_step = source_step + offset
                    if offset >= trajectory.chunk_size:
                        skipped_short_chunk += len(self.groups)
                        continue

                    future_chunk = trajectory.policy_chunks.get(future_step)
                    future_demo = None
                    if trajectory.demonstrated_actions is not None:
                        if future_step < trajectory.demonstrated_actions.shape[0]:
                            future_demo = trajectory.demonstrated_actions[future_step]
                        else:
                            skipped_no_demo += len(self.groups)
                            continue

                    if self.target_mode == "fresh_policy" and future_chunk is None:
                        skipped_no_future += len(self.groups)
                        continue

                    for group in self.groups:
                        examples.append(
                            TemporalExample(
                                episode_id=trajectory.episode_id,
                                task_id=trajectory.task_id,
                                source_step=source_step,
                                future_step=future_step,
                                offset=offset,
                                group=group.name,
                                source_chunk=source_chunk,
                                source_observation_embedding=source_embedding,
                                future_policy_chunk=future_chunk,
                                future_demonstrated_action=future_demo,
                            )
                        )

        self.last_build_summary = {
            "examples": len(examples),
            "skipped_no_future_policy_chunk": skipped_no_future,
            "skipped_short_source_chunk": skipped_short_chunk,
            "skipped_no_demonstrated_action": skipped_no_demo,
        }
        return tuple(examples)

    def build_split(
        self,
        trajectories: Sequence[FrozenTrajectory],
        split: EpisodeSplit,
    ) -> dict[str, tuple[TemporalExample, ...]]:
        """Build examples after applying a precomputed episode split."""

        by_id = {trajectory.episode_id: trajectory for trajectory in trajectories}
        if len(by_id) != len(trajectories):
            raise ValueError("trajectory episode identifiers must be unique")
        result: dict[str, tuple[TemporalExample, ...]] = {}
        for name, episode_ids in split.as_dict().items():
            missing = [episode_id for episode_id in episode_ids if episode_id not in by_id]
            if missing:
                raise KeyError(f"split contains unknown episodes: {missing[:3]}")
            result[name] = self.build([by_id[episode_id] for episode_id in episode_ids])
        return result

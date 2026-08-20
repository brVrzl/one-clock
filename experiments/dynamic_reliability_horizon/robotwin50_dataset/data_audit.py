"""Audit local RoboTwin v3 metadata and materialize the episode-only split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_TASKS = (
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler", "put_bottles_dustbin",
    "put_object_cabinet", "rotate_qrcode", "scan_object", "shake_bottle", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two", "stamp_seal",
    "turn_switch",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _episode_rows(root: Path) -> list[dict[str, Any]]:
    import pandas as pd

    paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError("no meta/episodes parquet shards found")
    rows: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_parquet(path)
        rows.extend(frame.to_dict(orient="records"))
    return rows


def _first_task(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return str(value[0]) if len(value) else "<missing>"
    return str(value) if value is not None else "<missing>"


def _stratification_key(row: dict[str, Any]) -> str:
    # LeRobot v3 episode metadata normally has `tasks`; older conversions may
    # expose task_index. We preserve the recorded value and never infer a
    # canonical task from natural-language wording.
    if "task_index" in row:
        return str(row["task_index"])
    if "tasks" in row:
        return _first_task(row["tasks"])
    return "<missing>"


def _split(rows: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[str]] = {}
    for row_number, row in enumerate(rows):
        episode = row.get("episode_index", row_number)
        buckets.setdefault(_stratification_key(row), []).append(str(int(episode)))
    if len(rows) > 1 and set(buckets) == {"<missing>"}:
        raise ValueError(
            "episode metadata has no recorded task_index/tasks field; refusing a non-task-stratified split"
        )
    parts = {"train": [], "validation": [], "test": []}
    names = tuple(parts)
    targets = np.asarray([0.8, 0.1, 0.1], dtype=np.float64) * len(rows)
    counts = np.zeros(3, dtype=np.float64)
    bucket_items = list(buckets.items())
    rng.shuffle(bucket_items)
    # Keep every recorded task bucket in exactly one split.  This prevents a
    # repeated task prompt from crossing the split while a deficit-weighted
    # assignment keeps the episode counts near the requested 80/10/10 ratio,
    # including when most task buckets are singletons.
    bucket_items.sort(key=lambda item: -len(item[1]))
    for _key, values in bucket_items:
        bucket_size = len(values)
        deficits = (targets - counts) / np.maximum(targets, 1.0)
        split_index = int(np.argmax(deficits))
        parts[names[split_index]].extend(values)
        counts[split_index] += bucket_size
    if not parts["train"]:
        raise ValueError("deterministic split has no training episodes")
    return parts


def audit(
    root: Path,
    *,
    output_dir: Path,
    dataset_repo: str,
    dataset_revision: str,
    seed: int = 20260820,
) -> dict[str, Any]:
    info = _read_json(root / "meta" / "info.json")
    rows = _episode_rows(root)
    split = _split(rows, seed)
    tasks_path = root / "meta" / "tasks.parquet"
    task_rows = 0
    if tasks_path.is_file():
        import pandas as pd

        task_rows = len(pd.read_parquet(tasks_path))

    manifest = {
        "status": "audited",
        "dataset_root": str(root.resolve()),
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "dataset_revision_source": "caller-pinned revision; local files do not encode the Hub commit",
        "codebase_version": info.get("codebase_version"),
        "total_episodes": int(info.get("total_episodes", len(rows))),
        "total_frames": int(info.get("total_frames", 0)),
        "total_task_rows": task_rows,
        "fps": int(info.get("fps", 0)),
        "features": info.get("features", {}),
        "canonical_tasks": list(CANONICAL_TASKS),
        "canonical_task_count": len(CANONICAL_TASKS),
        "clean_demo_provenance": "not represented in LeRobot v3 metadata; verify source subset before claiming demo_clean",
        "intended_subset": {
            "name": "canonical RoboTwin 2.0 demo_clean demonstrations",
            "expected_canonical_tasks": 50,
            "expected_demonstrations_per_task": 50,
            "selection_status": "not verified from the unified LeRobot metadata; no clean/random provenance field is present",
        },
        "canonical_task_source": {
            "path": "/home/wjq/workspace/upstreams/RoboTwin/envs",
            "task_count": len(CANONICAL_TASKS),
            "mapping_policy": "episode task prompt remains authoritative; no guessed semantic remapping",
        },
        "metadata_checksums_manifest": "metadata_checksums.json",
        "large_data": {
            "cache_root": "/home/wjq/robotwin_reliability_cache",
            "git_tracked": False,
        },
        "environment_action_contract": {
            "state_equals_action_order": info.get("features", {}).get("observation.state", {}).get("names")
            == info.get("features", {}).get("action", {}).get("names"),
            "action_order_source": "meta/info.json features.action.names",
            "action_units": "RoboTwin/LeRobot joint-space values as stored; no conversion applied",
            "verified_group_schema": "group_schema.json",
        },
        "episode_metadata_rows": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    split_manifest = {
        "status": "audited",
        "version": "robotwin_episode_split_v1",
        "seed": seed,
        "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "stratification_key": "recorded_episode_task_prompt_or_task_index",
        "task_bucket_holdout": True,
        "leakage_unit": "episode_index",
        **split,
    }
    (output_dir / "episode_split.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "manifest": manifest,
        "task_bucket_count": len({_stratification_key(row) for row in rows}),
        "split_counts": {name: len(values) for name, values in split.items()},
        "disjoint": not (
            set(split["train"]) & set(split["validation"])
            or set(split["train"]) & set(split["test"])
            or set(split["validation"]) & set(split["test"])
        ),
    }
    (output_dir / "data_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", default="lerobot/robotwin_unified")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--write-manifests", action="store_true")
    parser.add_argument("--seed", type=int, default=20260820)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.write_manifests:
        raise SystemExit("pass --write-manifests to materialize the policy-independent manifests")
    print(
        json.dumps(
            audit(
                args.dataset_root,
                output_dir=args.output_dir,
                dataset_repo=args.dataset_repo,
                dataset_revision=args.dataset_revision,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

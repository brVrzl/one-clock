#!/usr/bin/env python3
"""Download the pinned ``lerobot/libero`` snapshot with resumable atomic files.

The Hub's Xet path has been unreliable on Thor.  Direct resolve URLs are used
deliberately, while the API tree at the same commit supplies the exact file
list and expected byte sizes.  A failed file is recorded and does not abort
the remaining download work; corpus construction later refuses incomplete
tabular data, and cache generation records missing video episodes separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from dataset_common import DATASET_REPO_ID, DATASET_REVISION, atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--include-videos", action="store_true")
    parser.add_argument("--max-files", type=int, default=None, help="Bounded smoke-test limit; omit for full snapshot.")
    return parser.parse_args()


def api_tree() -> list[dict[str, object]]:
    url = f"https://huggingface.co/api/datasets/{DATASET_REPO_ID}/tree/{DATASET_REVISION}?recursive=true&expand=false&limit=1000"
    request = Request(url, headers={"User-Agent": "one-clock-libero4/1.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed public Hub URL
        entries = json.load(response)
    files = [entry for entry in entries if entry.get("type") == "file"]
    if not files:
        raise RuntimeError("Pinned Hub tree returned no files")
    return sorted(files, key=lambda item: str(item["path"]))


def selected_files(entries: list[dict[str, object]], include_videos: bool) -> list[dict[str, object]]:
    result = []
    for entry in entries:
        path = str(entry["path"])
        if path.startswith("videos/") and not include_videos:
            continue
        result.append(entry)
    return result


def download_one(dataset_root: Path, entry: dict[str, object]) -> tuple[str, int, str | None, str | None]:
    relative = Path(str(entry["path"]))
    expected_size = int(entry.get("size") or 0)
    destination = dataset_root / relative
    if destination.is_file() and (expected_size == 0 or destination.stat().st_size == expected_size):
        return str(relative), destination.stat().st_size, None, sha256_file(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.part")
    url = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/{DATASET_REVISION}/{relative.as_posix()}?download=true"
    command = [
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--retry", "5", "--retry-all-errors", "--connect-timeout", "20",
        "--max-time", "86400", "-C", "-", "-o", str(partial), url,
    ]
    last_error = None
    for attempt in range(1, 4):
        try:
            subprocess.run(command, check=True, text=True, capture_output=True)
            if expected_size and partial.stat().st_size != expected_size:
                raise RuntimeError(f"size {partial.stat().st_size} != expected {expected_size}")
            partial.replace(destination)
            return str(relative), destination.stat().st_size, None, sha256_file(destination)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            last_error = f"attempt {attempt}: {exc}"
    return str(relative), partial.stat().st_size if partial.exists() else 0, last_error, None


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = selected_files(api_tree(), args.include_videos)
    if args.max_files is not None:
        entries = entries[: max(0, args.max_files)]
    failures: list[dict[str, object]] = []
    completed_files: list[dict[str, object]] = []
    completed = 0
    for entry in entries:
        relative, size, error, digest = download_one(root, entry)
        completed += int(error is None)
        if error is not None:
            failures.append({"path": relative, "bytes_present": size, "error": error})
        else:
            completed_files.append({"path": relative, "bytes": size, "sha256": digest, "source_size": int(entry.get("size") or 0), "source_oid": entry.get("oid")})
        if completed == 1 or completed % 10 == 0 or completed + len(failures) == len(entries):
            print(f"downloaded_or_verified={completed}/{len(entries)} failures={len(failures)}", flush=True)
    listing = {
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "include_videos": bool(args.include_videos),
        "requested_files": [str(entry["path"]) for entry in entries],
        "completed_files": completed,
        "file_records": completed_files,
        "failures": failures,
    }
    atomic_write_json(root / "download_listing.json", listing)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())

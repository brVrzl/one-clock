#!/usr/bin/env python3
"""One-shot Track-B queue status without inspecting terminal success."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "track_b_manifest.json").read_text())
counts = Counter()
for cell in manifest["cells"]:
    stem = cell["cell_id"]
    metadata = ROOT / "track_b/results" / f"{stem}.json"
    predictions = ROOT / "track_b/predictions" / f"{stem}.npz"
    complete = ROOT / "track_b/markers" / f"{stem}.complete"
    failed = ROOT / "track_b/markers" / f"{stem}.technical_failed"
    if metadata.is_file() and predictions.is_file() and complete.is_file():
        counts["complete"] += 1
    elif failed.is_file():
        counts["technical_failed"] += 1
    else:
        counts["pending"] += 1
progress_path = ROOT / "track_b/progress/worker_0.json"
progress = json.loads(progress_path.read_text()) if progress_path.is_file() else None
print(json.dumps({"cells": len(manifest["cells"]), "counts": counts, "worker": progress}, indent=2))

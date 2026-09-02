#!/usr/bin/env python3
"""One-shot durable Track-A queue status; this does not inspect success outcomes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "track_a_manifest.json").read_text())
counts = Counter()
for cell in manifest["cells"]:
    stem = cell["cell_id"]
    result = ROOT / "track_a/results" / f"{stem}.json"
    complete = ROOT / "track_a/markers" / f"{stem}.complete"
    failed = ROOT / "track_a/markers" / f"{stem}.technical_failed"
    if result.is_file() and complete.is_file():
        counts["complete"] += 1
    elif failed.is_file():
        counts["technical_failed"] += 1
    else:
        counts["pending"] += 1
progress = {}
for path in sorted((ROOT / "track_a/progress").glob("worker_*.json")) if (ROOT / "track_a/progress").exists() else []:
    progress[path.stem] = json.loads(path.read_text())
print(json.dumps({"cells": len(manifest["cells"]), "counts": counts, "workers": progress}, indent=2))

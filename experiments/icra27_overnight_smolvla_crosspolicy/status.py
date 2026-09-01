#!/usr/bin/env python3
"""Print marker-driven queue status without starting work."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "queue_manifest.json").read_text())
counts = Counter()
for cell in manifest["cells"]:
    base = ROOT / "markers" / cell["phase"] / cell["cell_id"]
    status = "completed" if base.with_suffix(".complete").is_file() else (
        "technical_failed" if base.with_suffix(".technical_failed").is_file() else "pending"
    )
    counts[(cell["phase"], status)] += 1
for phase, total in manifest["expected_counts"].items():
    row = {status: counts[(phase, status)] for status in ("completed", "technical_failed", "pending")}
    row["total"] = total
    print(json.dumps({"phase": phase, **row}))


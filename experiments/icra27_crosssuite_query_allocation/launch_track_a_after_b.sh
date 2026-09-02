#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$root/orchestration/logs" "$root/orchestration/pids"
track_b_pid="$(<"$root/track_b/pids/worker_0.pid")"
while kill -0 "$track_b_pid" 2>/dev/null; do
  sleep 60
done

"/home/wjq/workspace/venvs/libero_act/bin/python" - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "track_b_manifest.json").read_text())
missing, failed = [], []
for cell in manifest["cells"]:
    stem = cell["cell_id"]
    if (root / "track_b/markers" / f"{stem}.technical_failed").is_file():
        failed.append(stem)
    if not (
        (root / "track_b/results" / f"{stem}.json").is_file()
        and (root / "track_b/predictions" / f"{stem}.npz").is_file()
        and (root / "track_b/markers" / f"{stem}.complete").is_file()
    ):
        missing.append(stem)
if missing or failed:
    (root / "orchestration/TRACK_A_NOT_LAUNCHED").write_text(
        json.dumps({"reason": "TRACK_B_INCOMPLETE_OR_TECHNICAL_FAILURE", "missing": missing, "failed": failed}, indent=2) + "\n"
    )
    raise SystemExit(2)
PY

"/home/wjq/workspace/venvs/libero_act/bin/python" "$root/validate_phase0.py"
"$root/launch_track_a.sh"
date --iso-8601=seconds > "$root/orchestration/TRACK_A_AUTOLAUNCH_COMPLETE"

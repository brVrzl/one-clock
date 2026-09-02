#!/usr/bin/env python3
"""Technical-only Track-A completion and integrity gate."""

import json
import subprocess
import sys
from pathlib import Path

TRACK = Path(__file__).resolve().parent.parent / "icra27_crosssuite_query_allocation"
manifest = json.loads((TRACK / "track_a_manifest.json").read_text())
assert manifest["preregistration_commit"] == "40549d876c0e09fad4e8033b3206f6018f53ece5"
cells = manifest["cells"]
assert len(cells) == 2700 and len({c["cell_id"] for c in cells}) == 2700
assert len({(c["block_id"], c["method"]) for c in cells}) == 2700
expected = {c["cell_id"] for c in cells}
observed = {p.name.removesuffix(".complete") for p in (TRACK / "track_a/markers").glob("*.complete")}
assert observed == expected, f"completion set mismatch missing={len(expected-observed)} extra={len(observed-expected)}"
assert not list((TRACK / "track_a/markers").glob("*.technical_failed"))
sys.path.insert(0, str(TRACK))
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("track_a_runner", TRACK / "run_track_a.py")
module = module_from_spec(spec); spec.loader.exec_module(module)
for cell in cells:
    module.validate_result(cell, TRACK / "track_a/results" / f"{cell['cell_id']}.json")
subprocess.run(["python", str(TRACK / "validate_phase0.py"), "--completed-track-a"], check=True)
print(json.dumps({"status": "PASS", "complete": 2700, "technical_failures": 0}))

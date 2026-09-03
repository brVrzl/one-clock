#!/usr/bin/env python3
"""Run the frozen Track-A analysis exactly once and freeze canonical artifacts."""

import csv, hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACK = ROOT.parent / "icra27_crosssuite_query_allocation"
marker = ROOT / "orchestration/TRACK_A_ANALYSIS_COMPLETE"
lock = ROOT / "orchestration/TRACK_A_ANALYSIS_LAUNCHED"
if marker.is_file():
    raise SystemExit(0)
lock.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
os.write(fd, f"pid={os.getpid()}\n".encode()); os.close(fd)
script = TRACK / "analyze_track_a.py"
assert hashlib.sha256(script.read_bytes()).hexdigest() == "ee5a7e91a865b41fc20305f4d6a4245b64a555a81ee68a963b4a1873151b2b08"
subprocess.run([sys.executable, str(script)], check=True, cwd=TRACK.parents[1])
data = json.loads((TRACK / "track_a/analysis.json").read_text())
assert data["status"] == "COMPLETE" and data["validated_results"] == 2700
with (TRACK / "track_a/condition_summaries.csv").open("w", newline="") as f:
    rows = data["method_summaries"]; fields = ["condition"] + list(next(iter(rows.values())))
    w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for key, value in rows.items(): w.writerow({"condition": key, **value})
with (TRACK / "track_a/contrasts.csv").open("w", newline="") as f:
    fields=["contrast","first_successes","second_successes","N","first_only","second_only","delta_percentage_points","exact_two_sided_mcnemar_p","paired_bootstrap_ci_percentage_points","task_cluster_bootstrap_ci_percentage_points"]
    w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for key,value in data["contrasts"].items(): w.writerow({k:(json.dumps(value[k]) if isinstance(value.get(k),list) else value.get(k,key if k=="contrast" else None)) for k in fields})
artifacts = [TRACK / "track_a/analysis.json", TRACK / "track_a/report.md", TRACK / "track_a/condition_summaries.csv", TRACK / "track_a/contrasts.csv", TRACK / "FIGURE_SPEC.md"]
freeze = {str(p.relative_to(TRACK.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
(TRACK / "track_a/CANONICAL_ARTIFACTS.json").write_text(json.dumps(freeze, indent=2)+"\n")
marker.write_text("COMPLETE\n")

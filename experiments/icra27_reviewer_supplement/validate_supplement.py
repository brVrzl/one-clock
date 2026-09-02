#!/usr/bin/env python3
"""Static, reuse, and terminal integrity gates; never selects on outcomes."""

import argparse, hashlib, json, subprocess
from pathlib import Path
from frozen_queue import ROOT, cells, marker_path, phase_cells, result_path
from run_queue import validate

def static() -> None:
    p=json.loads((ROOT/"protocol.json").read_text()); rows=cells()
    governing=(ROOT/"PREREGISTRATION_COMMIT").read_text().strip()
    assert len(governing)==40
    subprocess.run(["git","merge-base","--is-ancestor",governing,"HEAD"],cwd=ROOT.parents[1],check=True)
    remote=subprocess.check_output(["git","branch","-r","--contains",governing],cwd=ROOT.parents[1],text=True)
    assert "origin/exp/icra27-crosssuite-query-allocation" in remote
    governed=["PREREGISTRATION.md","protocol.json","FIGURE_SPEC.md","executors.py","frozen_queue.py","run_queue.py","run_canaries.py","analyze_supplement.py","master_pipeline.sh"]
    subprocess.run(["git","diff","--exit-code",governing,"--",*[str((ROOT/name).relative_to(ROOT.parents[1])) for name in governed],*[str((ROOT/"manifests"/f"{phase}.json").relative_to(ROOT.parents[1])) for phase in ("r1a","r1b","r1c","r1d","r2")]],cwd=ROOT.parents[1],check=True)
    assert p["outcomes_used_to_select_conditions"] is False
    assert len(rows)==p["expected_new_cells"]["all_if_r2"]
    assert {phase:len(phase_cells(phase)) for phase in ("r1a","r1b","r1c","r1d","r2")} == {"r1a":1512,"r1b":252,"r1c":280,"r1d":100,"r2":160}
    assert len({r["cell_id"] for r in rows})==len(rows)
    for r in rows:
        cp=Path(r["checkpoint"]); assert (cp/"config.json").is_file() and (cp/"model.safetensors").is_file()
    r1d=p["r1d"]
    assert hashlib.sha256((Path(r1d["checkpoint"])/"model.safetensors").read_bytes()).hexdigest()==r1d["model_sha256"]
    assert subprocess.check_output(["git","-C","/home/wjq/workspace/upstreams/lerobot","rev-parse","HEAD"],text=True).strip()==r1d["lerobot_commit"]
    for phase in ("r1a","r1b","r1c","r1d","r2"):
        frozen=json.loads((ROOT/"manifests"/f"{phase}.json").read_text()); assert frozen["cells"]==phase_cells(phase)

def terminal(phase: str) -> None:
    rows=phase_cells(phase); assert not any(marker_path(r,"technical_failed").is_file() for r in rows)
    assert all(marker_path(r).is_file() for r in rows)
    for r in rows: validate(r,result_path(r))

ap=argparse.ArgumentParser(); ap.add_argument("--static",action="store_true"); ap.add_argument("--phase")
a=ap.parse_args(); static()
if a.phase: terminal(a.phase)
print(json.dumps({"status":"PASS","phase":a.phase or "static"}))

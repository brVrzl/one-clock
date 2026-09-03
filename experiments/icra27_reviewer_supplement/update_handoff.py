#!/usr/bin/env python3
"""Write technical overnight handoff without opening scientific outcomes."""

import json, os, subprocess, time
from pathlib import Path
from frozen_queue import ROOT, phase_cells, marker_path

def git(*args): return subprocess.check_output(["git",*args],cwd=ROOT.parents[1],text=True).strip()
def status(phase):
    rows=phase_cells(phase); complete=sum(marker_path(r).is_file() for r in rows); failed=sum(marker_path(r,"technical_failed").is_file() for r in rows)
    return f"{complete}/{len(rows)} complete; {failed} unresolved technical failures"
def pids():
    rows=[]
    for path in sorted((ROOT/"orchestration/pids").glob("*.pid")):
        try:
            pid=int(path.read_text().strip())
        except Exception:
            pid=-1
        try:
            cmdline=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
            live=("master_pipeline.sh" in cmdline or "resume_master.sh" in cmdline) if path.stem == "master" else "run_queue.py" in cmdline
        except Exception:
            live=False
        rows.append(f"{path.stem}={pid} ({'active' if live else 'completed'})")
    return ", ".join(rows) or "none"
prereg=(ROOT/"PREREGISTRATION_COMMIT").read_text().strip() if (ROOT/"PREREGISTRATION_COMMIT").is_file() else "pending"
track=ROOT.parent/"icra27_crosssuite_query_allocation"
ta_complete=len(list((track/"track_a/markers").glob("*.complete"))); ta_failed=len(list((track/"track_a/markers").glob("*.technical_failed")))
attempts=sum(len(json.loads(p.read_text()).get("attempts",[])) for p in (ROOT/"attempts").rglob("*.json")) if (ROOT/"attempts").exists() else 0
resume=f"bash {ROOT/'launch_watcher.sh'} --resume"+""
failure=(ROOT/"orchestration/PIPELINE_FAILED").read_text().strip() if (ROOT/"orchestration/PIPELINE_FAILED").is_file() else "none"
start_epoch=(ROOT/"orchestration/pipeline_start_epoch").read_text().strip() if (ROOT/"orchestration/pipeline_start_epoch").is_file() else "not started"
text=f"""# Overnight ICRA handoff

Updated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}

1. Branch: `{git('branch','--show-current')}`
2. HEAD: `{git('rev-parse','HEAD')}`
3. Track-A preregistration SHA: `40549d876c0e09fad4e8033b3206f6018f53ece5`
4. Reviewer-supplement preregistration SHA: `{prereg}`
5. Track-A final count: {ta_complete}/2700
6. Track-A technical failure count: {ta_failed}
7. Track-A scientific-analysis status: {'complete' if (ROOT/'orchestration/TRACK_A_ANALYSIS_COMPLETE').is_file() else 'not complete'}
8. Track-A headline preregistered labels: {'written to canonical analysis; not copied into technical handoff' if (ROOT/'orchestration/TRACK_A_ANALYSIS_COMPLETE').is_file() else 'not analyzed'}
9. R1A status: {status('r1a')}
10. R1B status: {status('r1b')}
11. R1C status: {status('r1c')}
12. R1D status: {status('r1d')}
13. R2 eligibility decision: `R2_ENABLED_TECHNICALLY` frozen before R1 outcomes
14. R2 status: {status('r2')}
15. Active/completed PIDs: {pids()}
16. Log paths: `{ROOT/'orchestration/logs'}`, `{track/'track_a/logs'}`
17. Completion-marker counts: R1A {status('r1a')}; R1B {status('r1b')}; R1C {status('r1c')}; R1D {status('r1d')}; R2 {status('r2')}
18. Technical retries: {attempts}
19. Remaining queue: see `{ROOT/'orchestration'}` markers and master log
20. Exact resume command: `{resume}`

Pipeline failure marker: `{failure}`
Original pipeline start epoch: `{start_epoch}` (preserved on `--resume`)
"""
(ROOT/"HANDOFF.md").write_text(text)

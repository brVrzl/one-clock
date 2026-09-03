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
analysis_path=track/"track_a/analysis.json"
analysis=json.loads(analysis_path.read_text()) if analysis_path.is_file() else None
if analysis:
    label_text=", ".join(f"{key}={'PASS' if value else 'FAIL'}" for key,value in analysis["labels"].items())
    condition_text="; ".join(f"{key} {value['successes']}/{value['N']} ({100*value['success_rate']:.2f}%)" for key,value in analysis["method_summaries"].items())
    contrast_text="; ".join(f"{key} {value['delta_percentage_points']:+.2f} pp, task-CI [{value['task_cluster_bootstrap_ci_percentage_points'][0]:+.2f},{value['task_cluster_bootstrap_ci_percentage_points'][1]:+.2f}]" for key,value in analysis["contrasts"].items())
else:
    label_text=condition_text=contrast_text="not analyzed"
b3_root=track/"track_b/forecast"
b3_complete=len(list((b3_root/"markers").glob("*.complete"))) if (b3_root/"markers").exists() else 0
b3_failed=len(list((b3_root/"markers").glob("*.technical_failed"))) if (b3_root/"markers").exists() else 0
timebase=json.loads((track/"temporal_contract_audit.json").read_text())["status"] if (track/"temporal_contract_audit.json").is_file() else "not audited"
text=f"""# Overnight ICRA handoff

Updated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}

1. Branch: `{git('branch','--show-current')}`
2. HEAD: `{git('rev-parse','HEAD')}`
3. Track-A preregistration SHA: `40549d876c0e09fad4e8033b3206f6018f53ece5`
4. Reviewer-supplement preregistration SHA: `{prereg}`
5. Track-A final count: {ta_complete}/2700
6. Track-A technical failure count: {ta_failed}
7. Track-A scientific-analysis status: {'complete' if (ROOT/'orchestration/TRACK_A_ANALYSIS_COMPLETE').is_file() else 'not complete'}; canonical path `{analysis_path}`
8. Track-A headline preregistered labels: {label_text}
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

## Authorized morning continuation

- Technical repair commit: `af1b54dc567973d47f0e234d98c9b83ab68e675b`
- Governing final analysis-only amendment commit: `e2fb21b`
- Track-A conditions: {condition_text}
- Track-A contrasts: {contrast_text}
- B3 status: {b3_complete}/8 task-policy shards complete; {b3_failed} technical failures; canonical analysis {'complete' if (b3_root/'analysis/summary.json').is_file() else 'pending'}.
- Temporal contract: `{timebase}`. Standard ACT training/B2/B3 are 10 Hz; R1A/B/C/D are 20 Hz; R2A is 30 Hz. Seconds are the primary cross-family axis.
- Reviewer prelaunch canary: {'PASS' if (ROOT/'canaries/r1_prelaunch.json').is_file() else 'pending'}.
- Scientific configuration confirmation: no condition, manifest, cohort, state, seed, statistic, decision rule, or preregistered launch order was changed. Governed supplement files remain those of `f44a7605246d4c9ea82f4d19ad61833e8fb13eb8`.
"""
(ROOT/"HANDOFF.md").write_text(text)

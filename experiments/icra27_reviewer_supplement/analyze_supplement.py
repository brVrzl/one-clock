#!/usr/bin/env python3
"""Canonical preregistered supplement analysis; run once after rollout decisions."""

from __future__ import annotations

import csv, gzip, json, math, os
from collections import defaultdict
from pathlib import Path
import numpy as np

from frozen_queue import ROOT, phase_cells, protocol, result_path


def exact_mcnemar(a: int,b: int) -> float:
    n=a+b
    if not n:return 1.0
    m=min(a,b); return min(1.0,2*sum(math.comb(n,k) for k in range(m+1))/(2**n))


def normalized(phase,method,suite,task,state,row):
    return {"phase":phase,"method":method,"suite":suite,"task_id":int(task),"state_id":int(state),
            "block":f"{suite}:task{int(task)}:state{int(state)}","success":bool(row["success"]),
            "environment_steps":int(row.get("environment_steps",row.get("steps",0))),
            "policy_queries":int(row.get("policy_queries",0)),"wall_clock_seconds":float(row.get("wall_clock_seconds",row.get("episode_wall_seconds",0)))}


def new_rows(phases):
    out=[]
    for phase in phases:
        for c in phase_cells(phase): out.append(normalized(phase,c["method"],c["suite"],c["task_id"],c["state_id"],json.loads(result_path(c).read_text())))
    return out


def object_reuse():
    out=[]; root=ROOT.parent/"group_delay_factorial_act20/results"; mapping={"FRESH":"A0_G0","REVERSE20":"A20_G0","FO20":"A0_G20"}
    for path in sorted(root.glob("task_*.json")):
        data=json.loads(path.read_text()); task=int(data["task_id"])
        for old,new in mapping.items():
            for ep in data["episodes"][old]: out.append(normalized("r1_object_reuse",new,"libero_object",task,ep["requested_initial_state_id"],ep))
    return out


def r1c_reuse():
    out=[]
    for path in sorted((ROOT.parent/"cross_suite_confirmation/results").glob("libero_*.json")):
        data=json.loads(path.read_text()); suite=data["suite"]; task=int(data["task_id"])
        if suite not in {"libero_goal","libero_10"}: continue
        for ep in data["episodes"]["FRESH"]: out.append(normalized("r1c_reuse","C00",suite,task,ep["requested_initial_state_id"],ep))
    for path in sorted((ROOT.parent/"candidate1_c2_cross_suite/results").glob("libero_*.json")):
        data=json.loads(path.read_text()); suite=data["suite"]; task=int(data["task_id"])
        for ep in data["episodes"]["C2_H16_ARM_FRESH_GRIP"]: out.append(normalized("r1c_reuse","C10",suite,task,ep["requested_initial_state_id"],ep))
    return out


def r1d_reuse():
    out=[]; base=Path("/home/wjq/workspace/one-clock/experiments/gate4a2_spatial_act_generalization/episodes")
    mapping={"A_NEWEST":"A0_G0","B_FULL_OLD20":"A20_G20","C_ASYMMETRIC_FO20":"A0_G20"}
    for path in base.rglob("*.json.gz"):
        if path.stem.removesuffix(".json") not in mapping: continue
        with gzip.open(path,"rt") as f:data=json.load(f)
        run=data["run"]; summary=data["summary"]; summary={**summary,"environment_steps":summary["steps"],"wall_clock_seconds":summary["episode_wall_seconds"]}
        out.append(normalized("r1d_reuse",mapping[run["method"]],"libero_spatial",run["task_id"],run["state_id"],summary))
    return out


def contrast(rows, first, second, index):
    by=defaultdict(dict)
    for r in rows:
        if r["method"] in {first,second}: by[r["block"]][r["method"]]=r
    pairs=[v for v in by.values() if set(v)=={first,second}]
    if not pairs: raise RuntimeError(f"no complete pairs {first}-{second}")
    x=np.array([p[first]["success"] for p in pairs],float); y=np.array([p[second]["success"] for p in pairs],float); d=x-y
    a=int(np.sum((x==1)&(y==0))); b=int(np.sum((x==0)&(y==1))); draws=protocol()["statistics"]["bootstrap_draws"]
    rng=np.random.default_rng(protocol()["statistics"]["paired_seed_base"]+index); paired=d[rng.integers(0,len(d),(draws,len(d)))].mean(1)
    tasks=defaultdict(list)
    for p,value in zip(pairs,d,strict=True): tasks[(p[first]["suite"],p[first]["task_id"])].append(float(value))
    keys=sorted(tasks); vals=np.array([np.mean(tasks[k]) for k in keys]); rng=np.random.default_rng(protocol()["statistics"]["task_cluster_seed_base"]+index)
    clustered=vals[rng.integers(0,len(vals),(draws,len(vals)))].mean(1)
    return {"contrast":f"{first}-{second}","first_successes":int(x.sum()),"second_successes":int(y.sum()),"N":len(d),"first_only":a,"second_only":b,
            "delta_percentage_points":100*float(d.mean()),"exact_two_sided_mcnemar_p":exact_mcnemar(a,b),
            "paired_bootstrap_ci_percentage_points":(100*np.percentile(paired,[2.5,97.5])).tolist(),
            "task_cluster_bootstrap_ci_percentage_points":(100*np.percentile(clustered,[2.5,97.5])).tolist(),
            "per_task_delta_percentage_points":{f"{k[0]}:task{k[1]}":100*float(v) for k,v in zip(keys,vals,strict=True)},
            "loto_percentage_points":[100*float(np.delete(vals,i).mean()) for i in range(len(vals))]}


def main():
    include_r2=(ROOT/"orchestration/R2_COMPLETE").is_file()
    rows=new_rows(["r1a","r1b","r1c","r1d"]+(["r2"] if include_r2 else []))+object_reuse()+r1c_reuse()+r1d_reuse()
    families={"r1a":[r for r in rows if r["phase"] in {"r1a","r1_object_reuse"}],"r1b":[r for r in rows if r["phase"] in {"r1b","r1_object_reuse"}],
              "r1c":[r for r in rows if r["phase"] in {"r1c","r1c_reuse"}],"r1d":[r for r in rows if r["phase"] in {"r1d","r1d_reuse"}]}
    if include_r2: families["r2"]=[r for r in rows if r["phase"]=="r2"]
    specs=[]
    for d in (2,4,8,12,16,20,32): specs += [("r1a","A0_G0",f"A{d}_G0"),("r1a","A0_G0",f"A0_G{d}"),("r1a",f"A0_G{d}",f"A{d}_G0")]
    specs += [("r1b","T20_R0_G0","T0_R20_G0"),("r1b","T20_R0_G0","A0_G0"),("r1b","T0_R20_G0","A0_G0")]
    specs += [("r1c",a,b) for a,b in (("C10","C00"),("C01","C00"),("C11","C10"),("C11","C01"),("C11","C00"),("C10","C01"))]
    specs += [("r1d",a,b) for a,b in (("A0_G20","A20_G0"),("A20_G0","A0_G0"),("A0_G20","A0_G0"),("A20_G20","A0_G0"))]
    if include_r2: specs += [("r2",a,b) for a,b in (("A0_G20","A20_G0"),("A0_G20","A0_G0"),("A20_G0","A0_G0"),("A20_G20","A0_G0"))]
    contrasts=[{"family":fam,**contrast(families[fam],a,b,i)} for i,(fam,a,b) in enumerate(specs)]
    summaries=[]
    for fam,rs in families.items():
        for method in sorted({r["method"] for r in rs}):
            m=[r for r in rs if r["method"]==method]; steps=sum(r["environment_steps"] for r in m); queries=sum(r["policy_queries"] for r in m)
            summaries.append({"family":fam,"method":method,"successes":sum(r["success"] for r in m),"N":len(m),"success_rate":np.mean([r["success"] for r in m]),"environment_steps":steps,"policy_queries":queries,"query_rate":queries/steps,"wall_clock_seconds":sum(r["wall_clock_seconds"] for r in m)})
    c={r["method"]:r["success_rate"] for r in summaries if r["family"]=="r1c"}
    output={"status":"COMPLETE","r2_included":include_r2,"condition_summaries":summaries,"contrasts":contrasts,
            "r1c_risk_difference_interaction":c["C11"]-c["C10"]-c["C01"]+c["C00"],"scientific_retries":0}
    (ROOT/"analysis.json").write_text(json.dumps(output,indent=2)+"\n")
    for name,items in (("condition_summaries.csv",summaries),("contrasts.csv",contrasts)):
        with (ROOT/name).open("w",newline="") as f:
            fields=list(items[0]); w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for item in items:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in item.items()})
    (ROOT/"orchestration/SUPPLEMENT_ANALYSIS_COMPLETE").write_text("COMPLETE\n")

if __name__=="__main__": main()

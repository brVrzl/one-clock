#!/usr/bin/env python3
"""Frozen technical canaries. Canary outcomes never enter scientific analysis."""

from __future__ import annotations

import argparse, gc, json, random, sys, time
from pathlib import Path
import numpy as np

from executors import DenseExecutor
from frozen_queue import ROOT, protocol
from run_queue import Runtime, atomic_json


def r1c_canary(gpu: str) -> dict:
    cross = ROOT.parent / "cross_suite_confirmation"
    sys.path.insert(0, str(cross))
    import run_confirmation as confirmation
    p = json.loads((cross / "protocol.json").read_text())
    task = next(t for t in p["cohort"]["tasks"] if t["suite"]=="libero_goal" and t["task_id"]==4)
    sparse_runtime = confirmation.build_task_runtime(task, gpu)
    sparse = confirmation.run_episode(sparse_runtime, "HARD_H16", 0, 342400, 424242)
    sparse_actions = np.asarray([r["action"] for r in sparse["step_log"]], dtype=np.float32)
    torch=sparse_runtime["torch"]; del sparse_runtime; gc.collect(); torch.cuda.empty_cache()

    runtime = confirmation.build_task_runtime(task, gpu)
    env = confirmation.make_fresh_env(runtime, 342400); env.envs[0].init_state_id=0
    random.seed(342400); np.random.seed(342400); confirmation.reset_policy_rng(runtime["torch"],424242)
    runtime["policy"].reset(); observation,_=env.reset(seed=[342400])
    executor=DenseExecutor("C11"); dense_actions=[]; success=False; completion=None
    try:
        for t in range(300):
            chunk,_ = confirmation.query_act_chunk(observation,env,runtime)
            action,_ = executor.step(t,chunk)
            dense_actions.append(action.copy())
            observation,reward,terminated,truncated,info=env.step(action[None])
            done=bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0])
            if done:
                success=confirmation.extract_success(info,reward); completion=t+1 if success else None; break
    finally: env.close()
    dense_actions=np.asarray(dense_actions,dtype=np.float32)
    checks={"executed_actions_exact": dense_actions.shape==sparse_actions.shape and np.array_equal(dense_actions,sparse_actions),
            "episode_length_exact": len(dense_actions)==int(sparse["environment_steps"]),
            "terminal_success_exact": bool(success)==bool(sparse["success"]),
            "completion_step_exact": completion==sparse["completion_step"],
            "trajectory_identity_from_same_initial_state_and_exact_actions": dense_actions.shape==sparse_actions.shape and np.array_equal(dense_actions,sparse_actions)}
    if not all(checks.values()): raise RuntimeError(f"R1C identity canary failed: {checks}")
    return {"status":"PASS","excluded_from_scientific_analysis":True,"checks":checks}


def r1d_canary() -> dict:
    p=protocol()["r1d"]
    manifest=Path("/home/wjq/workspace/one-clock/research/audit_outputs/gate4a2_spatial_rollout_manifest.json")
    data=json.loads(manifest.read_text()); episodes=Path("/home/wjq/workspace/one-clock/experiments/gate4a2_spatial_act_generalization/episodes")
    checks={"historical_manifest_complete":data["complete"] is True and data["completed_episodes"]==500,
            "historical_episode_files":len(list(episodes.rglob("*.json.gz")))==500,
            "dense_queries_historical":data["valid_environment_steps"]==data["valid_policy_queries"],
            "checkpoint_revision":data["provenance"]["hf_checkpoint_revision"]==p["checkpoint_revision"],
            "checkpoint_model":data["provenance"]["model_sha256"]==p["model_sha256"],
            "lerobot_commit":data["provenance"]["lerobot_git_commit"]==p["lerobot_commit"]}
    synthetic=np.arange(40*100*7,dtype=np.float32).reshape(40,100,7)
    ex=DenseExecutor("A20_G0")
    for t in range(40):
        action,source=ex.step(t,synthetic[t])
        expected=synthetic[t,0] if t<20 else np.r_[synthetic[t-20,20,:6],synthetic[t,0,6]]
        if not np.array_equal(action,expected): checks["source_mapping_and_prefix"]=False; break
    else: checks["source_mapping_and_prefix"]=True
    if not all(checks.values()): raise RuntimeError(f"R1D reconstruction canary failed: {checks}")
    return {"status":"PASS","checks":checks,"outcomes_used":False}


def r2_preflight(gpu: str) -> dict:
    cell=__import__("frozen_queue").phase_cells("r2")[0]
    rt=Runtime(gpu); cfg=rt.env_cfg(cell); started=time.time(); rt.load(cell,cfg)
    result={"status":"PASS","checkpoint":cell["checkpoint"],"policy_type":rt.cfg.type,
            "chunk_size":int(rt.cfg.chunk_size),"n_action_steps":int(rt.cfg.n_action_steps),
            "environment_steps":0,"outcome_observed":False,"load_seconds":time.time()-started}
    rt.drop(); return result


ap=argparse.ArgumentParser(); ap.add_argument("--gpu",default="0"); ap.add_argument("--r2",action="store_true")
a=ap.parse_args(); out = r2_preflight(a.gpu) if a.r2 else {"r1c":r1c_canary(a.gpu),"r1d":r1d_canary(),"status":"PASS"}
path=ROOT/"canaries"/("r2_preflight.json" if a.r2 else "r1_prelaunch.json"); atomic_json(path,out)
(ROOT/"orchestration"/("R2_CANARY_PASS" if a.r2 else "R1_CANARIES_PASS")).write_text("PASS\n")
print(json.dumps(out,indent=2))

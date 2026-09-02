#!/usr/bin/env python3
"""CPU-only critical manifest, executor, resume, and gate smoke."""

import json, tempfile
from pathlib import Path
import numpy as np
from executors import DenseExecutor
from frozen_queue import cells, phase_cells
from run_queue import validate

assert len(cells())==2304 and [len(phase_cells(x)) for x in ("r1a","r1b","r1c","r1d","r2")]==[1512,252,280,100,160]
chunks=np.arange(40*100*7,dtype=np.float32).reshape(40,100,7)
for method in ("A2_G0","A0_G32","T20_R0_G0","T0_R20_G0","C01","C11","A20_G20"):
    ex=DenseExecutor(method)
    for t in range(40):
        action,source=ex.step(t,chunks[t]); assert action.shape==(7,)
        assert all(source[f"{label}_q"]+source[f"{label}_k"]==t for label in ("translation","rotation","gripper"))
cell=phase_cells("r1a")[0]; steps=2
row={key:cell[key] for key in ("cell_id","block_id","phase","policy","suite","task_id","state_id","environment_seed","method","checkpoint")}
row.update(status="COMPLETE",success=False,environment_steps=steps,resolved_max_episode_steps=280,query_steps=[0,1],policy_queries=2,executed_actions=[[0]*7]*2,sources=[{"translation_q":t,"translation_k":0,"rotation_q":t,"rotation_k":0,"gripper_q":t,"gripper_k":0} for t in range(2)])
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/"result.json"; path.write_text(json.dumps(row)); validate(cell,path); validate(cell,path)
print(json.dumps({"status":"PASS","duplicate_validation_is_idempotent":True,"manifest_cells":2304}))

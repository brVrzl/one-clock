#!/usr/bin/env python3
"""Deterministic parity test for the validated ACT temporal ensemble path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "component_temporal_reuse"))

from temporal_operators import (  # noqa: E402
    act_temporal_weights,
    aggregate_full_action,
    same_target_candidates,
)
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler  # noqa: E402


COEFFICIENT = 0.01
CHUNK_SIZE = 100
ACTION_DIM = 7
STEPS = 20
ATOL = 2e-6
RTOL = 2e-6


def deterministic_chunks() -> list[np.ndarray]:
    rng = np.random.default_rng(20270828)
    return [
        rng.normal(0.0, 0.25, size=(CHUNK_SIZE, ACTION_DIM)).astype(np.float32)
        for _ in range(STEPS)
    ]


def explicit_action(chunks: list[np.ndarray], target_step: int) -> np.ndarray:
    candidates = same_target_candidates(chunks, target_step=target_step)
    weights = act_temporal_weights(len(candidates.actions), coefficient=COEFFICIENT)
    return aggregate_full_action(candidates.actions, weights)


def main() -> None:
    chunks = deterministic_chunks()
    ensembler = ACTTemporalEnsembler(
        temporal_ensemble_coeff=COEFFICIENT,
        chunk_size=CHUNK_SIZE,
    )
    official = []
    for source_chunk in chunks:
        tensor_chunk = torch.from_numpy(source_chunk[None, ...])
        official.append(ensembler.update(tensor_chunk)[0].detach().cpu().numpy())

    explicit = [explicit_action(chunks, target_step=step) for step in range(STEPS)]
    official_array = np.asarray(official, dtype=np.float64)
    explicit_array = np.asarray(explicit, dtype=np.float64)
    errors = np.max(np.abs(official_array - explicit_array), axis=1)

    fresh_error = float(np.max(np.abs(official_array[0] - chunks[0][0].astype(np.float64))))
    t1_error = float(np.max(np.abs(official_array[1] - explicit_array[1])))
    t2_error = float(np.max(np.abs(official_array[2] - explicit_array[2])))
    result = {
        "coefficient": COEFFICIENT,
        "chunk_size": CHUNK_SIZE,
        "steps": STEPS,
        "fresh_t0_max_abs_error": fresh_error,
        "t1_explicit_formula_max_abs_error": t1_error,
        "t2_explicit_formula_max_abs_error": t2_error,
        "official_vs_custom_max_abs_error": float(errors.max()),
        "official_vs_custom_max_abs_error_by_step": errors.tolist(),
        "t0_pass": bool(np.allclose(official_array[0], chunks[0][0], atol=ATOL, rtol=RTOL)),
        "t1_pass": bool(np.allclose(official_array[1], explicit_array[1], atol=ATOL, rtol=RTOL)),
        "t2_pass": bool(np.allclose(official_array[2], explicit_array[2], atol=ATOL, rtol=RTOL)),
        "all_20_pass": bool(np.allclose(official_array, explicit_array, atol=ATOL, rtol=RTOL)),
        "pass": bool(
            np.allclose(official_array[0], chunks[0][0], atol=ATOL, rtol=RTOL)
            and np.allclose(official_array[1], explicit_array[1], atol=ATOL, rtol=RTOL)
            and np.allclose(official_array[2], explicit_array[2], atol=ATOL, rtol=RTOL)
            and np.allclose(official_array, explicit_array, atol=ATOL, rtol=RTOL)
        ),
    }
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

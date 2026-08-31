#!/usr/bin/env python3
"""Shared pure-Python/NumPy utilities for ACT and SmolVLA shards."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

try:
    from .group_memory_operators import (
        m0_hard,
        m1_shared_te,
        m2_shared_cogact,
        m3_group_cogact,
        m4_anchored_group_reliability,
    )
except ImportError:  # direct execution from an experiment shard
    from group_memory_operators import (
        m0_hard,
        m1_shared_te,
        m2_shared_cogact,
        m3_group_cogact,
        m4_anchored_group_reliability,
    )


METHODS = (
    "M0_h16",
    "M1_shared_te_h16",
    "M2_shared_cogact_h16",
    "M3_group_cogact_h16",
    "M4_anchored_group_reliability_h16",
)
RUNNABLE_METHODS = METHODS[:4]


def compose_method(
    method: str,
    candidates: Any,
    *,
    kernel_name: str,
    coefficient: float = 0.01,
    alpha: float = 0.3,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compose one method from a CandidateSet or an equivalent object."""

    actions = np.asarray(candidates.actions, dtype=np.float64)
    ages = np.asarray(candidates.ages, dtype=np.float64)
    if method == "M0_h16":
        return m0_hard(actions, ages)
    if method == "M1_shared_te_h16":
        return m1_shared_te(actions, ages, kernel_name=kernel_name, coefficient=coefficient)
    if method == "M2_shared_cogact_h16":
        return m2_shared_cogact(
            actions, ages, kernel_name=kernel_name, alpha=alpha, coefficient=coefficient
        )
    if method == "M3_group_cogact_h16":
        return m3_group_cogact(
            actions, ages, kernel_name=kernel_name, alpha=alpha, coefficient=coefficient
        )
    if method == "M4_anchored_group_reliability_h16":
        return m4_anchored_group_reliability(actions, ages, reliability=None)
    raise ValueError(f"unknown group-memory method: {method!r}")


def canonical_smolvla_query_key(task_key: str, state_id: int, env_seed: int, q: int) -> str:
    """Reuse the established method-independent SmolVLA query key."""

    return f"smolvla|{task_key}|state={int(state_id)}|env_seed={int(env_seed)}|q={int(q)}"


def smolvla_query_seed(task_key: str, state_id: int, env_seed: int, q: int) -> tuple[str, int]:
    key = canonical_smolvla_query_key(task_key, state_id, env_seed, q)
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)
    return key, seed


def reset_torch_generators(torch: Any, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def paired_counts(candidate: list[bool], reference: list[bool]) -> dict[str, Any]:
    """Return paired counts and exact two-sided McNemar probability."""

    if len(candidate) != len(reference):
        raise ValueError("paired outcome vectors must have equal length")
    both_fail = reference_only = candidate_only = both_success = 0
    for c, r in zip(candidate, reference, strict=True):
        if c and r:
            both_success += 1
        elif c:
            candidate_only += 1
        elif r:
            reference_only += 1
        else:
            both_fail += 1
    discordant = candidate_only + reference_only
    p_value = 1.0
    if discordant:
        p_value = min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(min(candidate_only, reference_only) + 1))
            / (2**discordant),
        )
    candidate_array = np.asarray(candidate, dtype=np.int8)
    reference_array = np.asarray(reference, dtype=np.int8)
    return {
        "candidate_successes": int(sum(candidate)),
        "reference_successes": int(sum(reference)),
        "paired_net": float(np.mean(candidate_array - reference_array)),
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "both_success": both_success,
        "both_fail": both_fail,
        "exact_mcnemar_two_sided_p": p_value,
    }

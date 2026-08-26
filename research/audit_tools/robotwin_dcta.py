#!/usr/bin/env python3
"""Dynamic component-wise temporal aggregation for frozen RoboTwin ACT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


ACTION_DIM = 14
CHUNK_LENGTH = 50
ACT_DECAY = 0.01
CONTEXT_DIM = 512
GROUPS = {
    "left_arm": tuple(range(0, 6)),
    "left_gripper": (6,),
    "right_arm": tuple(range(7, 13)),
    "right_gripper": (13,),
}
GROUP_NAMES = tuple(GROUPS)


def native_act_weights(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Pinned ACT weights in its oldest-to-newest candidate order."""
    positions = torch.arange(mask.shape[-1], device=mask.device, dtype=dtype)
    logits = -ACT_DECAY * positions
    logits = logits.expand(mask.shape)
    return torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)


def group_mask(group: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(ACTION_DIM, device=device, dtype=dtype)
    mask[list(GROUPS[GROUP_NAMES[group]])] = 1
    return mask


class DynamicTemporalGate(nn.Module):
    """Small residual-logit gate shared across semantic action groups."""

    def __init__(self, hidden_dim: int = 64, group_embedding_dim: int = 8):
        super().__init__()
        # lag, physical age, dispersion norm; four 14-D vectors; qpos; ACT context
        self.feature_dim = 3 + 4 * ACTION_DIM + ACTION_DIM + CONTEXT_DIM
        self.group_embedding = nn.Embedding(len(GROUPS) + 1, group_embedding_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim // 2 + group_embedding_dim, 1)
        # The initial gate is exactly the native ACT zero-residual model.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _residual_for_view(
        self,
        candidates: torch.Tensor,
        lags: torch.Tensor,
        physical_ages: torch.Tensor,
        qpos: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
        action_view: torch.Tensor,
        embedding_id: int,
    ) -> torch.Tensor:
        newest_index = mask.sum(dim=-1) - 1
        newest = candidates[
            torch.arange(candidates.shape[0], device=candidates.device), newest_index
        ]
        candidate_mean = (
            candidates * mask.unsqueeze(-1)
        ).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        masked_candidates = candidates * action_view
        differences = (candidates - newest[:, None]) * action_view
        disagreement = (candidates - candidate_mean[:, None]) * action_view
        dispersion = torch.linalg.vector_norm(disagreement, dim=-1, keepdim=True)
        view = action_view.expand_as(candidates)
        features = torch.cat(
            [
                (lags / (CHUNK_LENGTH - 1)).unsqueeze(-1),
                physical_ages.unsqueeze(-1),
                dispersion,
                masked_candidates,
                differences,
                disagreement,
                view,
                qpos[:, None].expand(-1, candidates.shape[1], -1),
                context[:, None].expand(-1, candidates.shape[1], -1),
            ],
            dim=-1,
        )
        encoded = self.encoder(features)
        group_ids = torch.full(
            encoded.shape[:-1], embedding_id, device=encoded.device, dtype=torch.long
        )
        embedding = self.group_embedding(group_ids)
        residual = self.head(torch.cat([encoded, embedding], dim=-1)).squeeze(-1)
        return residual.masked_fill(~mask, 0)

    def residual_logits(
        self,
        candidates: torch.Tensor,
        lags: torch.Tensor,
        physical_ages: torch.Tensor,
        qpos: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
        *,
        shared: bool,
    ) -> torch.Tensor:
        if shared:
            all_actions = torch.ones(
                1, 1, ACTION_DIM, device=candidates.device, dtype=candidates.dtype
            )
            residual = self._residual_for_view(
                candidates,
                lags,
                physical_ages,
                qpos,
                context,
                mask,
                all_actions,
                len(GROUPS),
            )
            return residual[:, None].expand(-1, len(GROUPS), -1)

        residuals = []
        for group in range(len(GROUPS)):
            view = group_mask(
                group, device=candidates.device, dtype=candidates.dtype
            ).view(1, 1, ACTION_DIM)
            residuals.append(
                self._residual_for_view(
                    candidates,
                    lags,
                    physical_ages,
                    qpos,
                    context,
                    mask,
                    view,
                    group,
                )
            )
        return torch.stack(residuals, dim=1)


def aggregate_candidates(
    candidates: torch.Tensor,
    mask: torch.Tensor,
    residual_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate same-decision-target candidates by semantic group."""
    baseline = native_act_weights(mask, candidates.dtype)
    if residual_logits is None:
        residual_logits = torch.zeros(
            candidates.shape[0], len(GROUPS), candidates.shape[1],
            device=candidates.device, dtype=candidates.dtype,
        )
    # All official ACT weights are positive; clamp is inactive and only avoids log(0).
    logits = torch.log(torch.clamp(baseline, min=torch.finfo(candidates.dtype).tiny))
    logits = logits[:, None] + residual_logits
    weights = torch.softmax(logits.masked_fill(~mask[:, None], -torch.inf), dim=-1)
    action = torch.empty(
        candidates.shape[0], ACTION_DIM,
        device=candidates.device, dtype=candidates.dtype,
    )
    for group, name in enumerate(GROUP_NAMES):
        indices = list(GROUPS[name])
        action[:, indices] = torch.sum(
            weights[:, group, :, None] * candidates[:, :, indices], dim=1
        )
    return action, weights


def dcta_action(
    gate: DynamicTemporalGate,
    candidates: torch.Tensor,
    lags: torch.Tensor,
    physical_ages: torch.Tensor,
    qpos: torch.Tensor,
    context: torch.Tensor,
    mask: torch.Tensor,
    *,
    shared: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    residuals = gate.residual_logits(
        candidates, lags, physical_ages, qpos, context, mask, shared=shared
    )
    return aggregate_candidates(candidates, mask, residuals)


@dataclass(frozen=True)
class DCTAStep:
    action: np.ndarray
    weights: np.ndarray
    candidate_sources: tuple[int, ...]
    candidate_offsets: tuple[int, ...]
    source_ages_seconds: tuple[float, ...]


class DCTAExecutor:
    """Episode-local history and same-decision-target DCTA composition."""

    def __init__(self, gate: DynamicTemporalGate, mode: str, device: torch.device):
        if mode not in {"NATIVE_ACT", "SHARED_DYNAMIC_AGG", "DCTA"}:
            raise ValueError(f"unsupported aggregation mode {mode}")
        self.gate = gate
        self.mode = mode
        self.device = device
        self.reset()

    def reset(self) -> None:
        self._history: list[tuple[int, float, np.ndarray]] = []
        self._next_decision = 0

    def update(
        self,
        decision: int,
        query_time_seconds: float,
        normalized_chunk: np.ndarray,
        normalized_qpos: np.ndarray,
        act_context: np.ndarray,
    ) -> DCTAStep:
        if decision != self._next_decision:
            raise ValueError("DCTA requires one ordered query per decision")
        if self._history and query_time_seconds <= self._history[-1][1]:
            raise ValueError("simulator query time must strictly increase")
        if normalized_chunk.shape != (CHUNK_LENGTH, ACTION_DIM):
            raise ValueError("expected one 50x14 ACT chunk")
        self._history.append((decision, query_time_seconds, normalized_chunk.copy()))
        self._history = self._history[-CHUNK_LENGTH:]
        self._next_decision += 1

        sources = tuple(item[0] for item in self._history)
        offsets = tuple(decision - source for source in sources)
        candidates = np.stack(
            [chunk[offset] for (_, _, chunk), offset in zip(self._history, offsets)]
        )
        ages = tuple(query_time_seconds - item[1] for item in self._history)
        count = len(candidates)
        padded = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM, device=self.device)
        padded[0, :count] = torch.as_tensor(candidates, device=self.device)
        mask = torch.zeros(1, CHUNK_LENGTH, dtype=torch.bool, device=self.device)
        mask[0, :count] = True
        lags = torch.zeros(1, CHUNK_LENGTH, device=self.device)
        lags[0, :count] = torch.as_tensor(offsets, device=self.device)
        physical_ages = torch.zeros(1, CHUNK_LENGTH, device=self.device)
        physical_ages[0, :count] = torch.as_tensor(ages, device=self.device)
        qpos = torch.as_tensor(normalized_qpos, device=self.device).float().view(1, -1)
        context = torch.as_tensor(act_context, device=self.device).float().view(1, -1)
        with torch.no_grad():
            if self.mode == "NATIVE_ACT":
                action, weights = aggregate_candidates(padded, mask)
            else:
                action, weights = dcta_action(
                    self.gate,
                    padded,
                    lags,
                    physical_ages,
                    qpos,
                    context,
                    mask,
                    shared=self.mode == "SHARED_DYNAMIC_AGG",
                )
        return DCTAStep(
            action=action[0].cpu().numpy(),
            weights=weights[0, :, :count].cpu().numpy(),
            candidate_sources=sources,
            candidate_offsets=offsets,
            source_ages_seconds=ages,
        )


def group_balanced_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = [
        torch.mean((prediction[:, list(indices)] - target[:, list(indices)]) ** 2)
        for indices in GROUPS.values()
    ]
    return torch.stack(losses).mean()


def effective_source_age(weights: np.ndarray, ages: Sequence[float]) -> np.ndarray:
    return weights @ np.asarray(ages, dtype=np.float64)

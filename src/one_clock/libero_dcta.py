"""Native ACT temporal aggregation and learned residual generalizations for LIBERO.

Candidates are ordered from oldest source query to newest source query.  This
matches LeRobot's ``ACTTemporalEnsembler``: candidate position zero receives
weight ``exp(0)`` and later candidates receive ``exp(-m * position)``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


AggregationMethod = Literal["standard_act", "shared_dynamic", "dcta"]


@dataclass(frozen=True)
class TemporalAggregationOutput:
    """Executed action and group-wise temporal diagnostics."""

    action: Tensor
    weights: Tensor
    effective_query_age: Tensor
    effective_physical_age: Tensor
    entropy: Tensor
    arm_gripper_kernel_distance: Tensor


@dataclass(frozen=True)
class TemporalTrainingExamples:
    """Padded same-target candidates and demonstration targets."""

    candidates: Tensor
    valid_mask: Tensor
    query_ages: Tensor
    physical_ages: Tensor
    robot_state: Tensor
    target_action: Tensor


@dataclass(frozen=True)
class TemporalExecutorStep:
    """One executor decision in normalized and environment action spaces."""

    normalized_action: Tensor
    environment_action: Tensor
    aggregation: TemporalAggregationOutput


class ACTEncoderContextCapture:
    """Localized hook that mean-pools the frozen ACT encoder representation."""

    def __init__(self, encoder: nn.Module) -> None:
        self._context: Tensor | None = None
        self._handle = encoder.register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        if not isinstance(output, Tensor) or output.ndim != 3:
            raise ValueError("ACT encoder hook expects output with shape [tokens, batch, dim]")
        self._context = output.mean(dim=0).detach()

    def pop(self, *, expected_batch_size: int | None = None) -> Tensor:
        if self._context is None:
            raise RuntimeError("no ACT encoder context has been captured")
        context = self._context
        self._context = None
        if expected_batch_size is not None and context.shape[0] != expected_batch_size:
            raise ValueError("captured ACT context has the wrong batch dimension")
        return context

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "ACTEncoderContextCapture":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class SameTargetChunkHistory:
    """Single-environment ACT history with exact same-target indexing."""

    def __init__(self, *, chunk_size: int = 10, action_dim: int = 7) -> None:
        if chunk_size < 1 or action_dim < 1:
            raise ValueError("chunk_size and action_dim must be positive")
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self._chunks: deque[tuple[int, int, Tensor]] = deque()

    def reset(self) -> None:
        self._chunks.clear()

    def update(
        self,
        *,
        source_step: int,
        chunk: Tensor,
        physical_source_step: int | None = None,
        current_physical_step: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Insert one query and return candidates, mask, and two age tensors.

        The returned tensors have a leading batch dimension of one and include
        only valid candidates, ordered oldest to newest.
        """

        source_step = int(source_step)
        physical_source_step = source_step if physical_source_step is None else int(physical_source_step)
        current_physical_step = (
            physical_source_step if current_physical_step is None else int(current_physical_step)
        )
        if physical_source_step > current_physical_step:
            raise ValueError("physical_source_step cannot be later than current_physical_step")
        if chunk.shape != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"chunk must have shape {(self.chunk_size, self.action_dim)}, got {tuple(chunk.shape)}"
            )
        if not torch.isfinite(chunk).all():
            raise ValueError("ACT chunk contains non-finite values")
        if self._chunks and source_step != self._chunks[-1][0] + 1:
            raise ValueError("STANDARD_ACT requires one ordered policy query per decision")

        self._chunks.append((source_step, physical_source_step, chunk.detach().clone()))
        while source_step - self._chunks[0][0] >= self.chunk_size:
            self._chunks.popleft()

        query_ages = torch.tensor(
            [source_step - query_step for query_step, _, _ in self._chunks],
            device=chunk.device,
            dtype=chunk.dtype,
        )
        physical_ages = torch.tensor(
            [current_physical_step - physical_step for _, physical_step, _ in self._chunks],
            device=chunk.device,
            dtype=chunk.dtype,
        )
        if (physical_ages < 0).any():
            raise ValueError("stored physical source steps cannot be later than the current physical step")
        candidates = torch.stack(
            [saved_chunk[int(age.item())] for age, (_, _, saved_chunk) in zip(query_ages, self._chunks)]
        ).unsqueeze(0)
        valid_mask = torch.ones((1, len(self._chunks)), dtype=torch.bool, device=chunk.device)
        return candidates, valid_mask, query_ages.unsqueeze(0), physical_ages.unsqueeze(0)


class LiberoTemporalExecutor:
    """Stateful single-environment executor shared by all three methods.

    The caller supplies one newly predicted normalized ACT chunk on every
    simulator decision. The executor performs same-target aggregation before
    applying the checkpoint-native action postprocessor.
    """

    def __init__(
        self,
        *,
        method: AggregationMethod,
        chunk_size: int = 10,
        gate: DynamicTemporalGate | None = None,
        postprocessor: Callable[[Tensor], Tensor] | None = None,
        temporal_ensemble_coeff: float = 0.01,
    ) -> None:
        if method != "standard_act" and gate is None:
            raise ValueError(f"{method} requires a DynamicTemporalGate")
        if temporal_ensemble_coeff < 0:
            raise ValueError("the canonical STANDARD_ACT coefficient must be non-negative")
        self.method = method
        self.gate = gate
        self.postprocessor = postprocessor
        self.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        self.history = SameTargetChunkHistory(chunk_size=chunk_size, action_dim=7)
        self._query_step = 0

    def reset(self) -> None:
        self.history.reset()
        self._query_step = 0

    def step(
        self,
        *,
        normalized_chunk: Tensor,
        normalized_robot_state: Tensor,
        physical_source_step: int | None = None,
        current_physical_step: int | None = None,
        act_context: Tensor | None = None,
    ) -> TemporalExecutorStep:
        """Aggregate one query's chunk and return the next environment action."""

        if normalized_robot_state.shape == (8,):
            normalized_robot_state = normalized_robot_state.unsqueeze(0)
        if normalized_robot_state.shape != (1, 8):
            raise ValueError("normalized_robot_state must have shape [8] or [1, 8]")
        if act_context is not None and act_context.ndim == 1:
            act_context = act_context.unsqueeze(0)

        candidates, valid_mask, query_ages, physical_ages = self.history.update(
            source_step=self._query_step,
            chunk=normalized_chunk,
            physical_source_step=physical_source_step,
            current_physical_step=current_physical_step,
        )
        aggregation = aggregate_same_target_predictions(
            method=self.method,
            candidates=candidates,
            valid_mask=valid_mask,
            query_ages=query_ages,
            physical_ages=physical_ages,
            robot_state=normalized_robot_state,
            gate=self.gate,
            act_context=act_context,
            temporal_ensemble_coeff=self.temporal_ensemble_coeff,
        )
        normalized_action = aggregation.action
        environment_action = (
            normalized_action if self.postprocessor is None else self.postprocessor(normalized_action)
        )
        if environment_action.shape != (1, 7):
            raise ValueError("ACT action postprocessor must return shape [1, 7]")
        if not torch.isfinite(environment_action).all():
            raise ValueError("ACT action postprocessor returned non-finite values")
        self._query_step += 1
        return TemporalExecutorStep(
            normalized_action=normalized_action,
            environment_action=environment_action,
            aggregation=aggregation,
        )


def native_act_logits(valid_mask: Tensor, coefficient: float = 0.01) -> Tensor:
    """Return native ACT logits for oldest-to-newest valid candidates.

    ``valid_mask`` has shape ``[batch, candidates]``. Invalid candidates may be
    padding anywhere; each valid candidate is assigned its rank among valid
    candidates so warmup behavior is identical to upstream ACT.
    """

    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be a bool tensor with shape [batch, candidates]")
    if not valid_mask.any(dim=1).all():
        raise ValueError("each batch item must contain at least one valid candidate")
    valid_rank = valid_mask.long().cumsum(dim=1) - 1
    return -float(coefficient) * valid_rank.to(dtype=torch.float32)


def masked_softmax(logits: Tensor, valid_mask: Tensor) -> Tensor:
    """Softmax over valid temporal candidates only."""

    if logits.shape != valid_mask.shape:
        raise ValueError("logits and valid_mask must have the same shape")
    masked_logits = logits.masked_fill(~valid_mask, -torch.inf)
    return torch.softmax(masked_logits, dim=-1)


class DynamicTemporalGate(nn.Module):
    """Lightweight per-candidate residual-logit gate.

    The gate sees full normalized ACT actions, their difference from the newest
    prediction, cross-candidate variance, query and physical ages, current
    normalized robot state, group identity, and an optional ACT context vector.
    The final layer is initialized to zero, so a newly constructed gate exactly
    starts from native ACT weights.
    """

    def __init__(
        self,
        *,
        action_dim: int = 7,
        state_dim: int = 8,
        context_dim: int = 0,
        num_groups: int = 2,
        hidden_dim: int = 64,
        group_embedding_dim: int = 4,
        max_age: int = 10,
    ) -> None:
        super().__init__()
        if action_dim < 1 or state_dim < 0 or context_dim < 0 or num_groups < 1:
            raise ValueError("feature dimensions and num_groups must be valid")
        if max_age < 1:
            raise ValueError("max_age must be positive")
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.max_age = int(max_age)
        self.group_embedding = nn.Embedding(num_groups, group_embedding_dim)
        feature_dim = 3 * action_dim + state_dim + context_dim + group_embedding_dim + 2
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        candidates: Tensor,
        valid_mask: Tensor,
        query_ages: Tensor,
        physical_ages: Tensor,
        robot_state: Tensor,
        group_id: int,
        act_context: Tensor | None = None,
    ) -> Tensor:
        """Return residual logits with shape ``[batch, candidates]``."""

        if candidates.ndim != 3 or candidates.shape[-1] != self.action_dim:
            raise ValueError("candidates must have shape [batch, candidates, action_dim]")
        batch_size, candidate_count, _ = candidates.shape
        expected_temporal_shape = (batch_size, candidate_count)
        if valid_mask.shape != expected_temporal_shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask has the wrong shape or dtype")
        if query_ages.shape != expected_temporal_shape or physical_ages.shape != expected_temporal_shape:
            raise ValueError("candidate ages must have shape [batch, candidates]")
        if robot_state.shape != (batch_size, self.state_dim):
            raise ValueError("robot_state has the wrong shape")
        if group_id < 0 or group_id >= self.group_embedding.num_embeddings:
            raise ValueError("group_id is outside the configured group range")
        if self.context_dim:
            if act_context is None or act_context.shape != (batch_size, self.context_dim):
                raise ValueError("act_context has the wrong shape")
        elif act_context is not None:
            raise ValueError("act_context was provided to a gate configured without context")

        newest_index = torch.where(
            valid_mask,
            torch.arange(candidate_count, device=candidates.device).expand(batch_size, -1),
            -1,
        ).amax(dim=1)
        newest = candidates[torch.arange(batch_size, device=candidates.device), newest_index]
        newest = newest.unsqueeze(1).expand(-1, candidate_count, -1)

        mask_float = valid_mask.unsqueeze(-1).to(candidates.dtype)
        count = mask_float.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (candidates * mask_float).sum(dim=1, keepdim=True) / count
        variance = ((candidates - mean).square() * mask_float).sum(dim=1, keepdim=True) / count
        variance = variance.expand(-1, candidate_count, -1)

        age_scale = float(self.max_age)
        features = [
            candidates,
            candidates - newest,
            variance,
            query_ages.to(candidates.dtype).unsqueeze(-1) / age_scale,
            physical_ages.to(candidates.dtype).unsqueeze(-1) / age_scale,
            robot_state.unsqueeze(1).expand(-1, candidate_count, -1),
        ]
        group_ids = torch.full(
            (batch_size, candidate_count), group_id, dtype=torch.long, device=candidates.device
        )
        features.append(self.group_embedding(group_ids))
        if act_context is not None:
            features.append(act_context.unsqueeze(1).expand(-1, candidate_count, -1))
        return self.network(torch.cat(features, dim=-1)).squeeze(-1)


def aggregate_same_target_predictions(
    *,
    method: AggregationMethod,
    candidates: Tensor,
    valid_mask: Tensor,
    query_ages: Tensor,
    physical_ages: Tensor,
    robot_state: Tensor,
    gate: DynamicTemporalGate | None = None,
    act_context: Tensor | None = None,
    temporal_ensemble_coeff: float = 0.01,
) -> TemporalAggregationOutput:
    """Aggregate same-target ACT predictions for one of the three controls."""

    if candidates.ndim != 3 or candidates.shape[-1] != 7:
        raise ValueError("LIBERO candidates must have shape [batch, candidates, 7]")
    if method not in {"standard_act", "shared_dynamic", "dcta"}:
        raise ValueError(f"unsupported aggregation method: {method!r}")
    if method != "standard_act" and gate is None:
        raise ValueError(f"{method} requires a DynamicTemporalGate")

    base_logits = native_act_logits(valid_mask, temporal_ensemble_coeff).to(
        device=candidates.device, dtype=candidates.dtype
    )
    if method == "standard_act":
        group_logits = base_logits.unsqueeze(1)
        groups = (tuple(range(7)),)
    elif method == "shared_dynamic":
        assert gate is not None
        residual = gate(
            candidates,
            valid_mask,
            query_ages,
            physical_ages,
            robot_state,
            group_id=0,
            act_context=act_context,
        )
        group_logits = (base_logits + residual).unsqueeze(1)
        groups = (tuple(range(7)),)
    else:
        assert gate is not None
        residuals = [
            gate(
                candidates,
                valid_mask,
                query_ages,
                physical_ages,
                robot_state,
                group_id=group_id,
                act_context=act_context,
            )
            for group_id in range(2)
        ]
        group_logits = base_logits.unsqueeze(1) + torch.stack(residuals, dim=1)
        groups = (tuple(range(6)), (6,))

    group_mask = valid_mask.unsqueeze(1).expand_as(group_logits)
    weights = masked_softmax(group_logits, group_mask)
    action = torch.empty(
        (candidates.shape[0], candidates.shape[-1]),
        device=candidates.device,
        dtype=candidates.dtype,
    )
    for group_index, action_indices in enumerate(groups):
        indices = list(action_indices)
        action[:, indices] = torch.sum(
            weights[:, group_index, :, None] * candidates[:, :, indices], dim=1
        )

    entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(dim=-1)
    effective_query_age = torch.sum(weights * query_ages.unsqueeze(1), dim=-1)
    effective_physical_age = torch.sum(weights * physical_ages.unsqueeze(1), dim=-1)
    if weights.shape[1] == 2:
        arm_gripper_kernel_distance = 0.5 * (weights[:, 0] - weights[:, 1]).abs().sum(dim=-1)
    else:
        arm_gripper_kernel_distance = torch.zeros(
            weights.shape[0], device=weights.device, dtype=weights.dtype
        )
    return TemporalAggregationOutput(
        action=action,
        weights=weights,
        effective_query_age=effective_query_age,
        effective_physical_age=effective_physical_age,
        entropy=entropy,
        arm_gripper_kernel_distance=arm_gripper_kernel_distance,
    )


def group_balanced_imitation_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Equal-weight arm and gripper L1 loss for 7-D LIBERO actions."""

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 7:
        raise ValueError("prediction and target must both have shape [batch, 7]")
    arm_loss = (prediction[:, :6] - target[:, :6]).abs().mean()
    gripper_loss = (prediction[:, 6] - target[:, 6]).abs().mean()
    return 0.5 * (arm_loss + gripper_loss)


def build_temporal_training_examples(
    *,
    predicted_chunks: Tensor,
    robot_states: Tensor,
    target_actions: Tensor,
    episode_ids: Tensor,
    physical_steps: Tensor | None = None,
) -> TemporalTrainingExamples:
    """Construct historical same-target examples from ordered demonstrations.

    Inputs are in frozen ACT normalized space and ordered by trajectory, then
    frame. Candidate histories never cross an episode boundary. Warmup examples
    are left padded so the newest valid candidate is always the rightmost one.
    """

    if predicted_chunks.ndim != 3 or predicted_chunks.shape[-1] != 7:
        raise ValueError("predicted_chunks must have shape [frames, chunk_size, 7]")
    frame_count, chunk_size, _ = predicted_chunks.shape
    if robot_states.shape != (frame_count, 8):
        raise ValueError("robot_states must have shape [frames, 8]")
    if target_actions.shape != (frame_count, 7):
        raise ValueError("target_actions must have shape [frames, 7]")
    if episode_ids.shape != (frame_count,):
        raise ValueError("episode_ids must have shape [frames]")
    if physical_steps is not None and physical_steps.shape != (frame_count,):
        raise ValueError("physical_steps must have shape [frames]")
    if frame_count == 0:
        raise ValueError("at least one demonstration frame is required")
    for name, value in {
        "predicted_chunks": predicted_chunks,
        "robot_states": robot_states,
        "target_actions": target_actions,
    }.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    candidates = predicted_chunks.new_zeros((frame_count, chunk_size, 7))
    valid_mask = torch.zeros(
        (frame_count, chunk_size), dtype=torch.bool, device=predicted_chunks.device
    )
    query_ages = predicted_chunks.new_zeros((frame_count, chunk_size))
    physical_ages = predicted_chunks.new_zeros((frame_count, chunk_size))

    history = SameTargetChunkHistory(chunk_size=chunk_size, action_dim=7)
    previous_episode = None
    local_query_step = 0
    local_physical_step = 0
    for frame_index in range(frame_count):
        episode = episode_ids[frame_index].item()
        if previous_episode is None or episode != previous_episode:
            history.reset()
            local_query_step = 0
            local_physical_step = 0
        if physical_steps is None:
            physical_step = local_physical_step
        else:
            physical_step = int(physical_steps[frame_index].item())
        frame_candidates, frame_valid, frame_query_ages, frame_physical_ages = history.update(
            source_step=local_query_step,
            chunk=predicted_chunks[frame_index],
            physical_source_step=physical_step,
            current_physical_step=physical_step,
        )
        candidate_count = frame_candidates.shape[1]
        start = chunk_size - candidate_count
        candidates[frame_index, start:] = frame_candidates[0]
        valid_mask[frame_index, start:] = frame_valid[0]
        query_ages[frame_index, start:] = frame_query_ages[0]
        physical_ages[frame_index, start:] = frame_physical_ages[0]
        previous_episode = episode
        local_query_step += 1
        local_physical_step += 1

    return TemporalTrainingExamples(
        candidates=candidates,
        valid_mask=valid_mask,
        query_ages=query_ages,
        physical_ages=physical_ages,
        robot_state=robot_states,
        target_action=target_actions,
    )

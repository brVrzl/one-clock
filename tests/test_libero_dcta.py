from __future__ import annotations

import torch

from one_clock.libero_dcta import (
    ACTEncoderContextCapture,
    DynamicTemporalGate,
    LiberoTemporalExecutor,
    SameTargetChunkHistory,
    aggregate_same_target_predictions,
    build_temporal_training_examples,
    group_balanced_imitation_loss,
    native_act_logits,
)


def temporal_batch(batch_size: int = 2, candidates: int = 4):
    generator = torch.Generator().manual_seed(7)
    actions = torch.randn(batch_size, candidates, 7, generator=generator)
    valid = torch.tensor([[False, True, True, True], [True, True, True, True]])
    query_ages = torch.tensor([[0, 2, 1, 0], [3, 2, 1, 0]], dtype=torch.float32)
    physical_ages = query_ages.clone()
    state = torch.randn(batch_size, 8, generator=generator)
    return actions, valid, query_ages, physical_ages, state


def test_native_act_logits_rank_only_valid_candidates_oldest_first() -> None:
    valid = torch.tensor([[False, True, True, True], [True, False, True, True]])
    logits = native_act_logits(valid, coefficient=0.01)
    torch.testing.assert_close(logits[0, 1:], torch.tensor([0.0, -0.01, -0.02]))
    torch.testing.assert_close(logits[1, [0, 2, 3]], torch.tensor([0.0, -0.01, -0.02]))


def test_same_target_history_indexes_diagonal_predictions() -> None:
    history = SameTargetChunkHistory(chunk_size=3, action_dim=7)
    for step in range(4):
        chunk = torch.stack(
            [torch.full((7,), 100.0 * step + offset) for offset in range(3)]
        )
        candidates, valid, query_ages, physical_ages = history.update(
            source_step=step,
            chunk=chunk,
            physical_source_step=step - 1,
            current_physical_step=step,
        )
    torch.testing.assert_close(candidates[0, :, 0], torch.tensor([102.0, 201.0, 300.0]))
    torch.testing.assert_close(query_ages, torch.tensor([[2.0, 1.0, 0.0]]))
    torch.testing.assert_close(physical_ages, torch.tensor([[3.0, 2.0, 1.0]]))
    assert valid.all()


def test_zero_initialized_shared_and_dcta_reproduce_standard_act() -> None:
    candidates, valid, query_ages, physical_ages, state = temporal_batch()
    standard = aggregate_same_target_predictions(
        method="standard_act",
        candidates=candidates,
        valid_mask=valid,
        query_ages=query_ages,
        physical_ages=physical_ages,
        robot_state=state,
    )
    shared_gate = DynamicTemporalGate(num_groups=1, max_age=candidates.shape[1])
    shared = aggregate_same_target_predictions(
        method="shared_dynamic",
        candidates=candidates,
        valid_mask=valid,
        query_ages=query_ages,
        physical_ages=physical_ages,
        robot_state=state,
        gate=shared_gate,
    )
    dcta_gate = DynamicTemporalGate(num_groups=2, max_age=candidates.shape[1])
    dcta = aggregate_same_target_predictions(
        method="dcta",
        candidates=candidates,
        valid_mask=valid,
        query_ages=query_ages,
        physical_ages=physical_ages,
        robot_state=state,
        gate=dcta_gate,
    )
    torch.testing.assert_close(shared.action, standard.action, rtol=0, atol=0)
    torch.testing.assert_close(dcta.action, standard.action, rtol=0, atol=0)
    torch.testing.assert_close(shared.weights[:, 0], standard.weights[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(dcta.weights[:, 0], standard.weights[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(dcta.weights[:, 1], standard.weights[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(dcta.arm_gripper_kernel_distance, torch.zeros(candidates.shape[0]))


def test_standard_act_matches_upstream_online_warmup() -> None:
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    generator = torch.Generator().manual_seed(19)
    chunks = torch.randn(7, 4, 7, generator=generator)
    upstream = ACTTemporalEnsembler(temporal_ensemble_coeff=0.01, chunk_size=4)
    history: list[tuple[int, torch.Tensor]] = []
    state = torch.zeros(1, 8)
    for step, chunk in enumerate(chunks):
        upstream_action = upstream.update(chunk.unsqueeze(0))
        history.append((step, chunk))
        history = history[-4:]
        ages = torch.tensor([[step - source for source, _ in history]], dtype=torch.float32)
        candidates = torch.stack([saved[int(age)] for age, (_, saved) in zip(ages[0], history)]).unsqueeze(0)
        valid = torch.ones(1, len(history), dtype=torch.bool)
        ours = aggregate_same_target_predictions(
            method="standard_act",
            candidates=candidates,
            valid_mask=valid,
            query_ages=ages,
            physical_ages=ages,
            robot_state=state,
        )
        torch.testing.assert_close(ours.action, upstream_action, rtol=2e-6, atol=2e-7)


def test_group_balanced_loss_does_not_sixfold_weight_arm() -> None:
    target = torch.zeros(1, 7)
    arm_error = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    gripper_error = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(
        group_balanced_imitation_loss(arm_error, target),
        group_balanced_imitation_loss(gripper_error, target),
    )


def test_training_examples_reset_history_at_episode_boundaries() -> None:
    chunks = torch.stack(
        [
            torch.stack([torch.full((7,), 100.0 * frame + offset) for offset in range(3)])
            for frame in range(4)
        ]
    )
    examples = build_temporal_training_examples(
        predicted_chunks=chunks,
        robot_states=torch.zeros(4, 8),
        target_actions=torch.zeros(4, 7),
        episode_ids=torch.tensor([5, 5, 9, 9]),
    )
    torch.testing.assert_close(examples.candidates[1, 1:, 0], torch.tensor([1.0, 100.0]))
    torch.testing.assert_close(examples.query_ages[1], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(examples.physical_ages[1], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(examples.candidates[2, 2, 0], torch.tensor(200.0))
    assert examples.valid_mask[2].tolist() == [False, False, True]


def test_executor_matches_native_act_before_affine_postprocessing() -> None:
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    generator = torch.Generator().manual_seed(31)
    chunks = torch.randn(6, 4, 7, generator=generator)
    states = torch.randn(6, 8, generator=generator)
    upstream = ACTTemporalEnsembler(temporal_ensemble_coeff=0.01, chunk_size=4)
    executor = LiberoTemporalExecutor(
        method="standard_act",
        chunk_size=4,
        postprocessor=lambda action: 2.0 * action + 1.0,
    )
    for chunk, state in zip(chunks, states):
        upstream_normalized = upstream.update(chunk.unsqueeze(0))
        step = executor.step(normalized_chunk=chunk, normalized_robot_state=state)
        torch.testing.assert_close(step.normalized_action, upstream_normalized, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(
            step.environment_action, 2.0 * upstream_normalized + 1.0, rtol=2e-6, atol=2e-7
        )


def test_act_encoder_context_hook_mean_pools_tokens() -> None:
    encoder = torch.nn.Identity()
    tokens = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    with ACTEncoderContextCapture(encoder) as capture:
        encoder(tokens)
        context = capture.pop(expected_batch_size=2)
    torch.testing.assert_close(context, tokens.mean(dim=0))

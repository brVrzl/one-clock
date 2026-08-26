#!/usr/bin/env python3
"""Train SHARED_DYNAMIC_AGG and DCTA on frozen ACT demonstration predictions."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from one_clock.libero_dcta import (  # noqa: E402
    DynamicTemporalGate,
    TemporalTrainingExamples,
    aggregate_same_target_predictions,
    build_temporal_training_examples,
    group_balanced_imitation_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--suite", choices=["spatial", "object", "goal"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def selected_episode_ids(manifest: dict, suite: str, split: str) -> torch.Tensor:
    key = f"{split}_episode_ids"
    episode_ids = [episode for task in manifest["suites"][suite]["tasks"] for episode in task[key]]
    return torch.tensor(sorted(episode_ids), dtype=torch.int64)


def subset_examples(examples: TemporalTrainingExamples, indices: torch.Tensor) -> TemporalTrainingExamples:
    return TemporalTrainingExamples(
        candidates=examples.candidates[indices],
        valid_mask=examples.valid_mask[indices],
        query_ages=examples.query_ages[indices],
        physical_ages=examples.physical_ages[indices],
        robot_state=examples.robot_state[indices],
        target_action=examples.target_action[indices],
    )


def batch_prediction(
    *,
    method: str,
    examples: TemporalTrainingExamples,
    contexts: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    gate: DynamicTemporalGate | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = aggregate_same_target_predictions(
        method=method,
        candidates=examples.candidates[indices].to(device),
        valid_mask=examples.valid_mask[indices].to(device),
        query_ages=examples.query_ages[indices].to(device),
        physical_ages=examples.physical_ages[indices].to(device),
        robot_state=examples.robot_state[indices].to(device),
        gate=gate,
        act_context=contexts[indices].to(device),
    )
    return output.action, examples.target_action[indices].to(device)


@torch.inference_mode()
def evaluate(
    *,
    method: str,
    examples: TemporalTrainingExamples,
    contexts: torch.Tensor,
    device: torch.device,
    batch_size: int,
    gate: DynamicTemporalGate | None,
) -> dict[str, float]:
    if gate is not None:
        gate.eval()
    arm_sum = 0.0
    gripper_sum = 0.0
    count = 0
    for start in range(0, examples.candidates.shape[0], batch_size):
        indices = torch.arange(start, min(start + batch_size, examples.candidates.shape[0]))
        prediction, target = batch_prediction(
            method=method,
            examples=examples,
            contexts=contexts,
            indices=indices,
            device=device,
            gate=gate,
        )
        arm_sum += (prediction[:, :6] - target[:, :6]).abs().mean(dim=1).sum().item()
        gripper_sum += (prediction[:, 6] - target[:, 6]).abs().sum().item()
        count += len(indices)
    arm_l1 = arm_sum / count
    gripper_l1 = gripper_sum / count
    return {"balanced_l1": 0.5 * (arm_l1 + gripper_l1), "arm_l1": arm_l1, "gripper_l1": gripper_l1}


def fit_gate(
    *,
    method: str,
    train_examples: TemporalTrainingExamples,
    train_contexts: torch.Tensor,
    validation_examples: TemporalTrainingExamples,
    validation_contexts: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[DynamicTemporalGate, dict]:
    device = torch.device(args.device)
    num_groups = 1 if method == "shared_dynamic" else 2
    gate = DynamicTemporalGate(
        context_dim=train_contexts.shape[1],
        num_groups=num_groups,
        hidden_dim=64,
        max_age=train_examples.candidates.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed + num_groups)
    best_state = copy.deepcopy(gate.state_dict())
    best_epoch = 0
    best_validation = evaluate(
        method=method,
        examples=validation_examples,
        contexts=validation_contexts,
        device=device,
        batch_size=args.batch_size,
        gate=gate,
    )
    epochs_without_improvement = 0
    history = [{"epoch": 0, "validation": best_validation}]

    for epoch in range(1, args.epochs + 1):
        gate.train()
        permutation = torch.randperm(train_examples.candidates.shape[0], generator=generator)
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            prediction, target = batch_prediction(
                method=method,
                examples=train_examples,
                contexts=train_contexts,
                indices=indices,
                device=device,
                gate=gate,
            )
            loss = group_balanced_imitation_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        validation = evaluate(
            method=method,
            examples=validation_examples,
            contexts=validation_contexts,
            device=device,
            batch_size=args.batch_size,
            gate=gate,
        )
        history.append({"epoch": epoch, "validation": validation})
        if validation["balanced_l1"] < best_validation["balanced_l1"]:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(gate.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    gate.load_state_dict(best_state)
    train_error = evaluate(
        method=method,
        examples=train_examples,
        contexts=train_contexts,
        device=device,
        batch_size=args.batch_size,
        gate=gate,
    )
    return gate, {
        "best_epoch": best_epoch,
        "train_error": train_error,
        "validation_error": best_validation,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, patience, and batch size must be positive")
    torch.manual_seed(args.seed)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if cache["revision"] != manifest["suites"][args.suite]["dataset_revision"]:
        raise ValueError("candidate cache and frozen split refer to different dataset revisions")

    examples = build_temporal_training_examples(
        predicted_chunks=cache["predicted_chunks"],
        robot_states=cache["normalized_robot_states"],
        target_actions=cache["normalized_target_actions"],
        episode_ids=cache["episode_ids"],
        physical_steps=cache["frame_indices"],
    )
    training_episodes = selected_episode_ids(manifest, args.suite, "training")
    validation_episodes = selected_episode_ids(manifest, args.suite, "validation")
    train_mask = torch.isin(cache["episode_ids"], training_episodes)
    validation_mask = torch.isin(cache["episode_ids"], validation_episodes)
    if (train_mask & validation_mask).any() or not (train_mask | validation_mask).all():
        raise ValueError("frozen trajectory split is overlapping or incomplete")
    train_indices = train_mask.nonzero(as_tuple=False).squeeze(1)
    validation_indices = validation_mask.nonzero(as_tuple=False).squeeze(1)
    train_examples = subset_examples(examples, train_indices)
    validation_examples = subset_examples(examples, validation_indices)
    train_contexts = cache["act_contexts"][train_indices]
    validation_contexts = cache["act_contexts"][validation_indices]
    device = torch.device(args.device)

    standard_validation = evaluate(
        method="standard_act",
        examples=validation_examples,
        contexts=validation_contexts,
        device=device,
        batch_size=args.batch_size,
        gate=None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "suite": args.suite,
        "seed": args.seed,
        "training_frames": len(train_indices),
        "validation_frames": len(validation_indices),
        "standard_act_validation_error": standard_validation,
        "methods": {},
    }
    for method in ("shared_dynamic", "dcta"):
        gate, method_result = fit_gate(
            method=method,
            train_examples=train_examples,
            train_contexts=train_contexts,
            validation_examples=validation_examples,
            validation_contexts=validation_contexts,
            args=args,
        )
        torch.save(
            {
                "method": method,
                "suite": args.suite,
                "context_dim": train_contexts.shape[1],
                "hidden_dim": 64,
                "max_age": train_examples.candidates.shape[1],
                "state_dict": gate.cpu().state_dict(),
            },
            args.output_dir / f"{method}.pt",
        )
        results["methods"][method] = method_result

    (args.output_dir / "heldout_errors.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

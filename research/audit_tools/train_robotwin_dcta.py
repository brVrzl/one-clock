#!/usr/bin/env python3
"""Train shared-dynamic and component-wise DCTA gates on demonstrations only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.audit_tools.robotwin_dcta import (
    ACTION_DIM,
    CHUNK_LENGTH,
    GROUP_NAMES,
    GROUPS,
    DynamicTemporalGate,
    aggregate_candidates,
    dcta_action,
    effective_source_age,
    group_balanced_mse,
)


TRAIN_EPISODES = frozenset(range(40))
HELDOUT_EPISODES = frozenset(range(40, 50))


class CandidateCorpus:
    def __init__(self, paths: list[Path]):
        self.tasks: dict[str, dict[str, np.ndarray]] = {}
        self.train_indices: list[tuple[str, int]] = []
        self.heldout_indices: list[tuple[str, int]] = []
        for path in paths:
            loaded = np.load(path)
            task = str(loaded["task"])
            arrays = {key: loaded[key] for key in loaded.files if key != "task"}
            self.tasks[task] = arrays
            for index, episode in enumerate(arrays["episode_id"]):
                target = (task, index)
                if int(episode) in TRAIN_EPISODES:
                    self.train_indices.append(target)
                elif int(episode) in HELDOUT_EPISODES:
                    self.heldout_indices.append(target)
                else:
                    raise RuntimeError(f"unexpected episode {episode}")

    def batch(
        self, indices: list[tuple[str, int]], device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        size = len(indices)
        candidates = np.zeros((size, CHUNK_LENGTH, ACTION_DIM), dtype=np.float32)
        masks = np.zeros((size, CHUNK_LENGTH), dtype=bool)
        lags = np.zeros((size, CHUNK_LENGTH), dtype=np.float32)
        ages = np.zeros((size, CHUNK_LENGTH), dtype=np.float32)
        qposes = np.zeros((size, ACTION_DIM), dtype=np.float32)
        contexts = np.zeros((size, 512), dtype=np.float32)
        targets = np.zeros((size, ACTION_DIM), dtype=np.float32)
        for row, (task, target_index) in enumerate(indices):
            arrays = self.tasks[task]
            decision = int(arrays["decision"][target_index])
            count = min(decision + 1, CHUNK_LENGTH)
            source_indices = np.arange(target_index - count + 1, target_index + 1)
            source_decisions = arrays["decision"][source_indices].astype(np.int64)
            if not np.array_equal(source_decisions, np.arange(decision - count + 1, decision + 1)):
                raise RuntimeError("candidate history crossed an episode boundary")
            offsets = decision - source_decisions
            candidates[row, :count] = arrays["normalized_chunk"][source_indices, offsets]
            masks[row, :count] = True
            lags[row, :count] = offsets
            ages[row, :count] = (
                arrays["query_time_seconds"][target_index]
                - arrays["query_time_seconds"][source_indices]
            )
            qposes[row] = arrays["normalized_qpos"][target_index]
            contexts[row] = arrays["act_context"][target_index]
            targets[row] = arrays["normalized_demonstrated_action"][target_index]
        return tuple(
            torch.from_numpy(value).to(device)
            for value in (candidates, masks, lags, ages, qposes, contexts, targets)
        )


def predictions(
    gate: DynamicTemporalGate | None,
    values: tuple[torch.Tensor, ...],
    *,
    shared: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates, mask, lags, ages, qpos, context, _ = values
    if gate is None:
        return aggregate_candidates(candidates, mask)
    return dcta_action(
        gate, candidates, lags, ages, qpos, context, mask, shared=shared
    )


def evaluate(
    corpus: CandidateCorpus,
    indices: list[tuple[str, int]],
    gate: DynamicTemporalGate | None,
    *,
    shared: bool,
    device: torch.device,
    batch_size: int,
    collect_weights: bool = False,
) -> dict[str, Any]:
    total = 0
    group_sse = np.zeros(len(GROUPS), dtype=np.float64)
    group_elements = np.zeros(len(GROUPS), dtype=np.int64)
    weight_sums = np.zeros((len(GROUPS), CHUNK_LENGTH), dtype=np.float64)
    weight_counts = np.zeros(CHUNK_LENGTH, dtype=np.int64)
    effective_ages = [[] for _ in GROUPS]
    effective_ages_by_decision = {
        "warmup_0_9": [[] for _ in GROUPS],
        "early_10_24": [[] for _ in GROUPS],
        "established_25_plus": [[] for _ in GROUPS],
    }
    trajectory_weights: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            values = corpus.batch(batch_indices, device)
            action, weights = predictions(gate, values, shared=shared)
            target = values[-1]
            for group, group_indices in enumerate(GROUPS.values()):
                error = action[:, list(group_indices)] - target[:, list(group_indices)]
                group_sse[group] += float(torch.sum(error * error).cpu())
                group_elements[group] += error.numel()
            total += len(batch_indices)
            if collect_weights:
                mask = values[1].cpu().numpy()
                age = values[3].cpu().numpy()
                weight_np = weights.cpu().numpy()
                for row, (task, index) in enumerate(batch_indices):
                    count = int(mask[row].sum())
                    weight_sums[:, :count] += weight_np[row, :, :count]
                    weight_counts[:count] += 1
                    for group in range(len(GROUPS)):
                        value = float(effective_source_age(weight_np[row, group, :count], age[row, :count]))
                        effective_ages[group].append(value)
                    arrays = corpus.tasks[task]
                    decision = int(arrays["decision"][index])
                    decision_bin = (
                        "warmup_0_9" if decision < 10
                        else "early_10_24" if decision < 25
                        else "established_25_plus"
                    )
                    for group in range(len(GROUPS)):
                        effective_ages_by_decision[decision_bin][group].append(
                            effective_ages[group][-1]
                        )
                    if int(arrays["episode_id"][index]) == 40 and decision in {0, 10, 25, 50, 100}:
                        trajectory_weights.append(
                            {
                                "task": task,
                                "episode": 40,
                                "decision": decision,
                                "weights": {
                                    name: weight_np[row, group, :count].tolist()
                                    for group, name in enumerate(GROUP_NAMES)
                                },
                            }
                        )
    group_mse = group_sse / group_elements
    result: dict[str, Any] = {
        "examples": total,
        "group_mse": {name: float(group_mse[group]) for group, name in enumerate(GROUP_NAMES)},
        "group_balanced_mse": float(group_mse.mean()),
    }
    if collect_weights:
        denominator = np.maximum(weight_counts, 1)
        result["mean_weight_by_group_and_candidate_position"] = {
            name: (weight_sums[group] / denominator).tolist()
            for group, name in enumerate(GROUP_NAMES)
        }
        result["effective_source_age_seconds"] = {
            name: {
                "mean": float(np.mean(effective_ages[group])),
                "median": float(np.median(effective_ages[group])),
                "std": float(np.std(effective_ages[group])),
            }
            for group, name in enumerate(GROUP_NAMES)
        }
        result["left_right_weight_mean_absolute_difference"] = {
            "arms": float(np.mean(np.abs(weight_sums[0] / denominator - weight_sums[2] / denominator))),
            "grippers": float(np.mean(np.abs(weight_sums[1] / denominator - weight_sums[3] / denominator))),
        }
        result["effective_source_age_by_decision_bin_seconds"] = {
            decision_bin: {
                name: float(np.mean(values[group]))
                for group, name in enumerate(GROUP_NAMES)
            }
            for decision_bin, values in effective_ages_by_decision.items()
        }
        result["trajectory_weight_examples"] = trajectory_weights
    return result


def train_gate(
    corpus: CandidateCorpus,
    *,
    shared: bool,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> tuple[DynamicTemporalGate, list[dict[str, float]], int]:
    gate = DynamicTemporalGate().to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = np.random.default_rng(20270826 + int(shared))
    best_loss = np.inf
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(epochs):
        gate.train()
        shuffled = [corpus.train_indices[index] for index in generator.permutation(len(corpus.train_indices))]
        running = 0.0
        seen = 0
        for start in range(0, len(shuffled), batch_size):
            values = corpus.batch(shuffled[start:start + batch_size], device)
            prediction, _ = predictions(gate, values, shared=shared)
            loss = group_balanced_mse(prediction, values[-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(values[-1])
            seen += len(values[-1])
        gate.eval()
        validation = evaluate(
            corpus,
            corpus.heldout_indices,
            gate,
            shared=shared,
            device=device,
            batch_size=batch_size,
        )["group_balanced_mse"]
        history.append({"epoch": epoch, "train_loss": running / seen, "heldout_loss": validation})
        print(
            f"mode={'shared' if shared else 'dcta'} epoch={epoch} "
            f"train={running / seen:.6f} heldout={validation:.6f}",
            flush=True,
        )
        if validation < best_loss:
            best_loss = validation
            best_state = copy.deepcopy(gate.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no DCTA checkpoint selected")
    gate.load_state_dict(best_state)
    gate.eval()
    return gate, history, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    torch.manual_seed(20270826)
    np.random.seed(20270826)
    device = torch.device(args.device)
    corpus = CandidateCorpus(args.candidates)
    native_heldout = evaluate(
        corpus,
        corpus.heldout_indices,
        None,
        shared=True,
        device=device,
        batch_size=args.batch_size,
        collect_weights=True,
    )
    shared_gate, shared_history, shared_best_epoch = train_gate(
        corpus,
        shared=True,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )
    dcta_gate, dcta_history, dcta_best_epoch = train_gate(
        corpus,
        shared=False,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )
    shared_train = evaluate(corpus, corpus.train_indices, shared_gate, shared=True, device=device, batch_size=args.batch_size)
    shared_heldout = evaluate(corpus, corpus.heldout_indices, shared_gate, shared=True, device=device, batch_size=args.batch_size, collect_weights=True)
    dcta_train = evaluate(corpus, corpus.train_indices, dcta_gate, shared=False, device=device, batch_size=args.batch_size)
    dcta_heldout = evaluate(corpus, corpus.heldout_indices, dcta_gate, shared=False, device=device, batch_size=args.batch_size, collect_weights=True)
    heldout_by_task = {}
    for task in sorted(corpus.tasks):
        indices = [item for item in corpus.heldout_indices if item[0] == task]
        heldout_by_task[task] = {
            "native_act": evaluate(corpus, indices, None, shared=True, device=device, batch_size=args.batch_size),
            "shared_dynamic": evaluate(corpus, indices, shared_gate, shared=True, device=device, batch_size=args.batch_size),
            "dcta": evaluate(corpus, indices, dcta_gate, shared=False, device=device, batch_size=args.batch_size),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": shared_gate.state_dict(), "mode": "SHARED_DYNAMIC_AGG"}, args.output_dir / "shared_dynamic_gate.pt")
    torch.save({"state_dict": dcta_gate.state_dict(), "mode": "DCTA"}, args.output_dir / "dcta_gate.pt")
    summary = {
        "method": "Dynamic Component-wise Temporal Aggregation",
        "fit_source": "official 50 demo_clean trajectories per task only",
        "train_episodes_per_task": sorted(TRAIN_EPISODES),
        "heldout_episodes_per_task": sorted(HELDOUT_EPISODES),
        "tasks": sorted(corpus.tasks),
        "act_backbone_frozen": True,
        "rollout_success_used": False,
        "gate_parameter_count": dcta_gate.parameter_count,
        "architecture": {
            "shared_parameters": True,
            "group_embedding": True,
            "act_context": "frozen ACT decoder query-0 hidden state (512-D forward hook)",
            "residual_initialization": "zero; numerically reproduces NATIVE_ACT",
        },
        "native_act_heldout": native_heldout,
        "shared_dynamic": {
            "best_epoch": shared_best_epoch,
            "history": shared_history,
            "train": shared_train,
            "heldout": shared_heldout,
        },
        "dcta": {
            "best_epoch": dcta_best_epoch,
            "history": dcta_history,
            "train": dcta_train,
            "heldout": dcta_heldout,
        },
        "heldout_by_task": heldout_by_task,
    }
    (args.output_dir / "offline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "native_heldout": native_heldout["group_balanced_mse"],
        "shared_heldout": shared_heldout["group_balanced_mse"],
        "dcta_heldout": dcta_heldout["group_balanced_mse"],
        "parameters": dcta_gate.parameter_count,
    }, indent=2))


if __name__ == "__main__":
    main()

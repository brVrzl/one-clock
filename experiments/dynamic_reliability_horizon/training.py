"""BCE training for the lightweight reliability head."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from .artifacts import PreparedReliabilityDataset
from .config import TrainingConfig
from .model import ReliabilityMLP


@dataclass(frozen=True)
class TrainingResult:
    model: object
    history: tuple[dict[str, float], ...]
    best_epoch: int
    checkpoint_path: Path | None


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - host-dependent
        raise ImportError("training requires torch") from error
    return torch


def _set_seed(seed: int) -> None:
    torch = _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mode_mask(dataset: PreparedReliabilityDataset, mode: str) -> np.ndarray:
    if mode == "combined":
        return np.ones(dataset.labels.shape, dtype=bool)
    if mode not in {"arm", "gripper"}:
        raise ValueError("mode must be combined, arm, or gripper")
    return dataset.groups == mode


def train_reliability_model(
    dataset: PreparedReliabilityDataset,
    *,
    mode: str,
    config: TrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Train one ablation on precomputed episode-split examples."""

    torch = _require_torch()
    config = config or TrainingConfig()
    config.validate()
    if dataset.split is None:
        raise ValueError("training requires an episode-level train/validation/test split")
    mode_rows = _mode_mask(dataset, mode)
    train_mask = mode_rows & (dataset.split == "train")
    validation_mask = mode_rows & (dataset.split == "validation")
    if not train_mask.any() or not validation_mask.any():
        raise ValueError(f"mode {mode!r} needs non-empty train and validation rows")

    _set_seed(config.seed)
    model = ReliabilityMLP(dataset.input_dim, hidden_dims=config.hidden_dims)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    x_train = torch.as_tensor(dataset.features[train_mask], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(dataset.labels[train_mask], dtype=torch.float32, device=device)
    x_validation = torch.as_tensor(dataset.features[validation_mask], dtype=torch.float32, device=device)
    y_validation = torch.as_tensor(dataset.labels[validation_mask], dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict[str, float]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], generator=generator)
        train_losses: list[float] = []
        for start in range(0, x_train.shape[0], config.batch_size):
            batch_indices = order[start : start + config.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.logits(x_train[batch_indices]), y_train[batch_indices])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model.logits(x_validation), y_validation)
        row = {
            "epoch": float(epoch + 1),
            "train_bce": float(np.mean(train_losses)),
            "validation_bce": float(validation_loss.detach().cpu()),
        }
        history.append(row)
        if row["validation_bce"] < best_loss:
            best_loss = row["validation_bce"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs > config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": dataset.input_dim,
                "hidden_dims": list(config.hidden_dims),
                "mode": mode,
                "config": config.as_dict(),
                "feature_names": list(dataset.feature_names),
                "best_epoch": best_epoch,
            },
            checkpoint,
        )
    return TrainingResult(model, tuple(history), best_epoch, checkpoint)


def run_ablation_suite(
    dataset: PreparedReliabilityDataset,
    *,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
    modes: tuple[str, ...] = ("combined", "arm", "gripper"),
) -> dict[str, TrainingResult]:
    output_dir = Path(output_dir)
    results: dict[str, TrainingResult] = {}
    for mode in modes:
        results[mode] = train_reliability_model(
            dataset,
            mode=mode,
            config=config,
            checkpoint_path=output_dir / f"reliability_mlp_{mode}.pt",
        )
    return results


def load_reliability_checkpoint(path: str | Path, *, device: str = "cpu") -> object:
    torch = _require_torch()
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Older Torch versions do not expose weights_only.
        payload = torch.load(path, map_location=device)
    model = ReliabilityMLP(int(payload["input_dim"]), hidden_dims=tuple(payload["hidden_dims"]))
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


def predict_scores(model: object, features: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    torch = _require_torch()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = model(tensor)
    return values.detach().cpu().numpy().astype(np.float64)

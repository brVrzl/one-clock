"""Masked BCE training for the shared vector reliability estimator.

Each row is one source observation and action group.  The output is the full
future-offset curve for that row; missing offsets are ignored through the
dataset mask.  No future observation or future action is passed to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path

import numpy as np

from .config import TrainingConfig
from .model import MonotoneSharedSurvivalMLP, SharedReliabilityMLP
from .vector_dataset import VectorReliabilityDataset


@dataclass(frozen=True)
class SharedTrainingResult:
    model: object
    history: tuple[dict[str, float], ...]
    best_epoch: int
    checkpoint_path: Path | None
    mode: str


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - host-dependent
        raise ImportError("shared reliability training requires torch") from error
    return torch


def _set_seed(seed: int) -> None:
    torch = _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mode_mask(dataset: VectorReliabilityDataset, mode: str) -> np.ndarray:
    if mode == "combined":
        return np.ones(dataset.features.shape[0], dtype=bool)
    if mode not in {"arm", "gripper"}:
        raise ValueError("mode must be combined, arm, or gripper")
    return dataset.groups == mode


def _masked_bce(logits, labels, mask):
    torch = _require_torch()
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    weighted = losses * mask
    denominator = mask.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("at least one observed target is required")
    return weighted.sum() / denominator


def _masked_probability_bce(probabilities, labels, mask):
    torch = _require_torch()
    losses = torch.nn.functional.binary_cross_entropy(
        probabilities, labels, reduction="none"
    )
    weighted = losses * mask
    denominator = mask.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("at least one observed target is required")
    return weighted.sum() / denominator


def _headline_target_mask(dataset: VectorReliabilityDataset) -> np.ndarray:
    """Mask the trivial identity event from every training loss."""

    mask = dataset.label_mask.copy()
    if mask.shape[1] > 0:
        mask[:, 0] = False
    return mask


def _train_shared_model(
    dataset: VectorReliabilityDataset,
    *,
    mode: str,
    config: TrainingConfig,
    checkpoint_path: str | Path | None,
    model_type: str,
) -> SharedTrainingResult:
    torch = _require_torch()
    if dataset.split is None:
        raise ValueError("training requires an episode-level train/validation/test split")
    mode_rows = _mode_mask(dataset, mode)
    train_rows = mode_rows & (dataset.split == "train")
    validation_rows = mode_rows & (dataset.split == "validation")
    if not train_rows.any() or not validation_rows.any():
        raise ValueError(f"mode {mode!r} needs non-empty train and validation rows")

    headline_mask = _headline_target_mask(dataset)
    train_mask = headline_mask[train_rows]
    validation_mask = headline_mask[validation_rows]
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("train and validation must contain observed offsets k>=1")

    _set_seed(config.seed)
    model_class = (
        SharedReliabilityMLP
        if model_type == "shared_reliability_mlp"
        else MonotoneSharedSurvivalMLP
    )
    model = model_class(
        dataset.input_dim,
        dataset.horizon_dim,
        hidden_dims=config.hidden_dims,
    )
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    x_train = torch.as_tensor(dataset.features[train_rows], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(dataset.labels[train_rows], dtype=torch.float32, device=device)
    m_train = torch.as_tensor(train_mask, dtype=torch.float32, device=device)
    x_validation = torch.as_tensor(
        dataset.features[validation_rows], dtype=torch.float32, device=device
    )
    y_validation = torch.as_tensor(
        dataset.labels[validation_rows], dtype=torch.float32, device=device
    )
    m_validation = torch.as_tensor(validation_mask, dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict[str, float]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    use_logits = model_type == "shared_reliability_mlp"

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], generator=generator)
        train_losses: list[float] = []
        for start in range(0, x_train.shape[0], config.batch_size):
            indices = order[start : start + config.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_logits:
                loss = _masked_bce(
                    model.logits(x_train[indices]), y_train[indices], m_train[indices]
                )
            else:
                loss = _masked_probability_bce(
                    model(x_train[indices]), y_train[indices], m_train[indices]
                )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            if use_logits:
                validation_loss = _masked_bce(
                    model.logits(x_validation), y_validation, m_validation
                )
            else:
                validation_loss = _masked_probability_bce(
                    model(x_validation), y_validation, m_validation
                )
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
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "input_dim": dataset.input_dim,
                "horizon_dim": dataset.horizon_dim,
                "hidden_dims": list(config.hidden_dims),
                "mode": mode,
                "config": config.as_dict(),
                "feature_names": list(dataset.feature_names),
                "identity_offset_masked": True,
                "best_epoch": best_epoch,
            },
            checkpoint,
        )
    return SharedTrainingResult(model, tuple(history), best_epoch, checkpoint, mode)


def train_shared_reliability_model(
    dataset: VectorReliabilityDataset,
    *,
    mode: str = "combined",
    config: TrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> SharedTrainingResult:
    """Train the independent per-offset shared MLP ablation."""

    config = config or TrainingConfig()
    config.validate()
    return _train_shared_model(
        dataset,
        mode=mode,
        config=config,
        checkpoint_path=checkpoint_path,
        model_type="shared_reliability_mlp",
    )


def train_monotone_shared_survival_model(
    dataset: VectorReliabilityDataset,
    *,
    mode: str = "combined",
    config: TrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> SharedTrainingResult:
    """Train the conditional-survival product parameterization."""

    config = config or TrainingConfig()
    config.validate()
    return _train_shared_model(
        dataset,
        mode=mode,
        config=config,
        checkpoint_path=checkpoint_path,
        model_type="monotone_shared_survival_mlp",
    )


def load_shared_checkpoint(path: str | Path, *, device: str = "cpu") -> object:
    """Load only checkpoints produced by the vector-output shared head."""

    torch = _require_torch()
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Older Torch versions do not expose weights_only.
        payload = torch.load(path, map_location=device)
    model_type = payload.get("model_type", "shared_reliability_mlp")
    model_class = (
        SharedReliabilityMLP
        if model_type == "shared_reliability_mlp"
        else MonotoneSharedSurvivalMLP
        if model_type == "monotone_shared_survival_mlp"
        else None
    )
    if model_class is None:
        raise ValueError("checkpoint is not a supported shared vector reliability model")
    model = model_class(
        int(payload["input_dim"]),
        int(payload["horizon_dim"]),
        hidden_dims=tuple(payload["hidden_dims"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


def predict_reliability_curves(
    model: object,
    features: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Return one ``[K]`` reliability curve per causal source row."""

    torch = _require_torch()
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = model(tensor)
    result = scores.detach().cpu().numpy().astype(np.float64)
    if result.ndim != 2:
        raise ValueError("shared model must return a two-dimensional curve array")
    return result


__all__ = [
    "train_monotone_shared_survival_model",
    "SharedTrainingResult",
    "load_shared_checkpoint",
    "predict_reliability_curves",
    "train_shared_reliability_model",
]

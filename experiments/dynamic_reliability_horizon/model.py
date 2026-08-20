"""Lightweight temporal reliability model interfaces.

The original row-wise ``ReliabilityMLP`` remains available for compatibility
with the preparation-stage artifact format.  The next-stage model below is a
shared head: one causal source feature vector produces the complete
``R_g(0...K-1)`` curve, so the model never receives an offset and never
directly predicts a horizon.
"""

from __future__ import annotations

from collections.abc import Sequence

from experiments.temporal_reliability_training.model import MLPBaseline


try:  # Torch stays optional for split, feature, and evaluation utilities.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depends on the host environment.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


ReliabilityMLP = MLPBaseline


if nn is not None:

    class SharedReliabilityMLP(nn.Module):
        """Small shared MLP returning a vector of reliability probabilities."""

        def __init__(
            self,
            input_dim: int,
            horizon_dim: int,
            *,
            hidden_dims: Sequence[int] = (128, 64),
        ) -> None:
            super().__init__()
            if input_dim < 1:
                raise ValueError("input_dim must be positive")
            if horizon_dim < 1:
                raise ValueError("horizon_dim must be positive")
            if not hidden_dims or any(width < 1 for width in hidden_dims):
                raise ValueError("hidden_dims must contain positive widths")
            layers: list[nn.Module] = []
            previous = input_dim
            for width in hidden_dims:
                layers.extend((nn.Linear(previous, width), nn.ReLU()))
                previous = width
            layers.append(nn.Linear(previous, horizon_dim))
            self.network = nn.Sequential(*layers)
            self.input_dim = input_dim
            self.horizon_dim = horizon_dim

        def logits(self, features: "torch.Tensor") -> "torch.Tensor":
            if features.ndim != 2 or features.shape[-1] != self.input_dim:
                raise ValueError(
                    f"features must have shape [batch, {self.input_dim}], got "
                    f"{tuple(features.shape)}"
                )
            values = self.network(features)
            if values.shape[-1] != self.horizon_dim:  # defensive contract check
                raise RuntimeError("shared reliability head returned the wrong curve width")
            return values

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            """Return ``[batch, K]`` reliability probabilities in ``[0, 1]``."""

            return torch.sigmoid(self.logits(features))


    class MonotoneSharedSurvivalMLP(nn.Module):
        """Shared conditional-survival head with a monotone product output.

        Offset zero is the identity event and is fixed to one.  The network
        predicts conditional survival probabilities for offsets ``1...K-1``;
        their cumulative product is the returned reliability curve.  This is
        the smallest structural change that guarantees
        ``R(k+1) <= R(k)`` without predicting a horizon directly.
        """

        def __init__(
            self,
            input_dim: int,
            horizon_dim: int,
            *,
            hidden_dims: Sequence[int] = (128, 64),
        ) -> None:
            super().__init__()
            if input_dim < 1:
                raise ValueError("input_dim must be positive")
            if horizon_dim < 1:
                raise ValueError("horizon_dim must be positive")
            if not hidden_dims or any(width < 1 for width in hidden_dims):
                raise ValueError("hidden_dims must contain positive widths")
            layers: list[nn.Module] = []
            previous = input_dim
            for width in hidden_dims:
                layers.extend((nn.Linear(previous, width), nn.ReLU()))
                previous = width
            # Offset zero is fixed identity; only later conditional events are learned.
            if horizon_dim > 1:
                layers.append(nn.Linear(previous, horizon_dim - 1))
            self.network = nn.Sequential(*layers) if layers else nn.Identity()
            self.input_dim = input_dim
            self.horizon_dim = horizon_dim

        def conditional_logits(self, features: "torch.Tensor") -> "torch.Tensor":
            if features.ndim != 2 or features.shape[-1] != self.input_dim:
                raise ValueError(
                    f"features must have shape [batch, {self.input_dim}], got "
                    f"{tuple(features.shape)}"
                )
            if self.horizon_dim == 1:
                return features.new_empty((features.shape[0], 0))
            return self.network(features)

        def conditional_survival(self, features: "torch.Tensor") -> "torch.Tensor":
            logits = self.conditional_logits(features)
            identity = torch.ones(
                (features.shape[0], 1), dtype=features.dtype, device=features.device
            )
            if self.horizon_dim == 1:
                return identity
            return torch.cat((identity, torch.sigmoid(logits)), dim=1)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            """Return a guaranteed-monotone ``[batch, K]`` survival curve."""

            return torch.cumprod(self.conditional_survival(features), dim=1)

else:

    class SharedReliabilityMLP:  # pragma: no cover - exercised only without Torch.
        """Placeholder that reports the optional dependency clearly."""

        def __init__(
            self,
            input_dim: int,
            horizon_dim: int,
            *,
            hidden_dims: Sequence[int] = (128, 64),
        ) -> None:
            del input_dim, horizon_dim, hidden_dims
            raise ImportError("SharedReliabilityMLP requires torch")


    class MonotoneSharedSurvivalMLP:  # pragma: no cover - exercised only without Torch.
        """Placeholder that reports the optional dependency clearly."""

        def __init__(
            self,
            input_dim: int,
            horizon_dim: int,
            *,
            hidden_dims: Sequence[int] = (128, 64),
        ) -> None:
            del input_dim, horizon_dim, hidden_dims
            raise ImportError("MonotoneSharedSurvivalMLP requires torch")


__all__ = [
    "MonotoneSharedSurvivalMLP",
    "ReliabilityMLP",
    "SharedReliabilityMLP",
]

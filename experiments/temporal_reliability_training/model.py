"""Untrained lightweight estimator interface.

Torch is optional at import time.  Constructing ``MLPBaseline`` is the first
operation that requires it; no optimizer, loss, or training loop is provided in
this preparation package.
"""

from __future__ import annotations

from collections.abc import Sequence


try:  # Keep dataset and evaluation utilities usable without Torch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depends on the host environment.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class MLPBaseline(nn.Module):
        """Small probability estimator; weights are initialized but untrained."""

        def __init__(
            self,
            input_dim: int,
            *,
            hidden_dims: Sequence[int] = (128, 64),
        ) -> None:
            super().__init__()
            if input_dim < 1:
                raise ValueError("input_dim must be positive")
            if not hidden_dims or any(width < 1 for width in hidden_dims):
                raise ValueError("hidden_dims must contain positive widths")
            layers: list[nn.Module] = []
            previous = input_dim
            for width in hidden_dims:
                layers.extend((nn.Linear(previous, width), nn.ReLU()))
                previous = width
            layers.append(nn.Linear(previous, 1))
            self.network = nn.Sequential(*layers)
            self.input_dim = input_dim

        def logits(self, features: "torch.Tensor") -> "torch.Tensor":
            if features.ndim != 2 or features.shape[-1] != self.input_dim:
                raise ValueError(
                    f"features must have shape [batch, {self.input_dim}], got "
                    f"{tuple(features.shape)}"
                )
            return self.network(features).squeeze(-1)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            """Return an uncalibrated reliability score in ``[0, 1]``."""

            return torch.sigmoid(self.logits(features))

else:

    class MLPBaseline:  # pragma: no cover - exercised only without Torch.
        """Placeholder that reports the optional dependency clearly."""

        def __init__(self, input_dim: int, *, hidden_dims: Sequence[int] = (128, 64)) -> None:
            del input_dim, hidden_dims
            raise ImportError(
                "MLPBaseline requires torch; feature and target preparation "
                "remain available without it"
            )

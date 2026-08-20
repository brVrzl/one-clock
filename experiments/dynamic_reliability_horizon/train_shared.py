"""CLI for the next-stage shared vector reliability estimator.

This command consumes a prepared vector dataset.  It does not load the robot
dataset, run ACT, or perform rollout execution; those upstream artifacts must
already exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainingConfig
from .vector_dataset import VectorReliabilityDataset
from .vector_training import (
    train_monotone_shared_survival_model,
    train_shared_reliability_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="prepared vector .npz")
    parser.add_argument("--checkpoint", type=Path, required=True, help="output .pt")
    parser.add_argument("--mode", choices=("combined", "arm", "gripper"), default="combined")
    parser.add_argument("--model", choices=("independent", "monotone"), default="independent")
    parser.add_argument("--config", type=Path, help="optional TrainingConfig JSON")
    parser.add_argument("--summary", type=Path, help="optional training summary JSON")
    args = parser.parse_args()

    config = TrainingConfig()
    if args.config is not None:
        config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    dataset = VectorReliabilityDataset.load(args.dataset)
    train_fn = (
        train_shared_reliability_model
        if args.model == "independent"
        else train_monotone_shared_survival_model
    )
    result = train_fn(
        dataset, mode=args.mode, config=config, checkpoint_path=args.checkpoint
    )
    summary = {
        "mode": result.mode,
        "model": args.model,
        "best_epoch": result.best_epoch,
        "history": list(result.history),
        "checkpoint": str(result.checkpoint_path),
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()

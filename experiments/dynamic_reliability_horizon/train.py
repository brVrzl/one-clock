"""CLI for training the adaptive reliability estimator on prepared targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import PreparedReliabilityDataset
from .config import TrainingConfig
from .training import run_ablation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="prepared .npz feature/target artifact")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    dataset = PreparedReliabilityDataset.load(args.dataset)
    config_values = {}
    if args.config:
        config_values = json.loads(args.config.read_text(encoding="utf-8"))
    config = TrainingConfig.from_dict(config_values) if config_values else TrainingConfig()
    results = run_ablation_suite(dataset, output_dir=args.output_dir, config=config)
    summary = {
        mode: {
            "checkpoint": None if result.checkpoint_path is None else str(result.checkpoint_path),
            "best_epoch": result.best_epoch,
            "history": list(result.history),
        }
        for mode, result in results.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

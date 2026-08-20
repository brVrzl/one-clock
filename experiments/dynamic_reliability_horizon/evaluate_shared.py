"""CLI for held-out evaluation of the shared vector reliability head."""

from __future__ import annotations

import argparse
from pathlib import Path

from .decoder import GroupHorizonDecoder, HorizonDecodeConfig
from .evaluation import (
    evaluate_shared_checkpoint,
    write_evaluation_artifacts,
)
from .vector_dataset import VectorReliabilityDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("combined", "arm", "gripper"), default="combined")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--min-horizon", type=int, default=1)
    parser.add_argument("--non-prefix", action="store_true")
    args = parser.parse_args()

    dataset = VectorReliabilityDataset.load(args.dataset)
    decoder = GroupHorizonDecoder(
        HorizonDecodeConfig(
            threshold_tau=args.tau,
            min_horizon=args.min_horizon,
            max_horizon=args.max_horizon,
            require_prefix=not args.non_prefix,
        )
    )
    report = evaluate_shared_checkpoint(
        dataset,
        args.checkpoint,
        mode=args.mode,
        decoder=decoder,
        n_bins=args.bins,
        device=args.device,
    )
    write_evaluation_artifacts(report, args.output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()

"""CLI for offline estimator evaluation and baseline comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import PreparedReliabilityDataset
from .decoder import GroupHorizonDecoder, HorizonDecodeConfig
from .evaluation import (
    evaluate_checkpoint,
    evaluate_horizon_sources,
    plot_calibration_curves,
    plot_reliability_diagrams,
    save_evaluation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mode", choices=("combined", "arm", "gripper"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--static-arm", type=int, default=4)
    parser.add_argument("--static-gripper", type=int, default=16)
    parser.add_argument("--global-horizon", type=int, default=4)
    args = parser.parse_args()
    dataset = PreparedReliabilityDataset.load(args.dataset)
    report = evaluate_checkpoint(
        dataset,
        args.checkpoint,
        mode=args.mode,
        n_bins=args.bins,
        device=args.device,
    )
    decoder = GroupHorizonDecoder(
        HorizonDecodeConfig(
            threshold_tau=args.tau,
            min_horizon=1,
            max_horizon=100,
            require_prefix=True,
        )
    )
    report["offline_horizon_sources"] = evaluate_horizon_sources(
        dataset,
        args.checkpoint,
        mode=args.mode,
        decoder=decoder,
        static_horizons={"arm": args.static_arm, "gripper": args.static_gripper},
        global_horizon=args.global_horizon,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_report(report, args.output_dir / f"evaluation_{args.mode}.json")
    plot_reliability_diagrams(report, args.output_dir / f"reliability_diagram_{args.mode}.png")
    plot_calibration_curves(report, args.output_dir / f"calibration_curve_{args.mode}.png")


if __name__ == "__main__":
    main()

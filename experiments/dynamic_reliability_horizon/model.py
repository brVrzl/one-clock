"""The minimal temporal reliability model used by the adaptive prototype."""

from experiments.temporal_reliability_training.model import MLPBaseline


ReliabilityMLP = MLPBaseline

__all__ = ["ReliabilityMLP"]

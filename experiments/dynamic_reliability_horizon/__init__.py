"""Group-wise temporal reliability guided adaptive execution prototype.

The package is deliberately separate from the production executor.  It can
train/evaluate an auxiliary reliability model and can emit rollout-compatible
``ExecutionDecision`` records through ``AdaptiveGroupwiseExecutor`` without
changing the frozen ACT policy or ``src/one_clock/executor.py``.
"""

from .adaptive_executor import AdaptiveGroupwiseExecutor, make_static_groupwise_executor
from .artifacts import PreparedReliabilityDataset, prepare_dataset
from .baselines import EmpiricalReliabilityPredictor, constant_prior_scores
from .config import DynamicHorizonConfig, TrainingConfig
from .decoder import GroupHorizonDecoder, HorizonDecodeConfig
from .horizon_analysis import (
    HorizonScheduleSummary,
    compare_horizon_sources,
    rows_to_curves,
    summarize_horizon_schedule,
)
from .scheduler import AdaptiveHorizonScheduler, HorizonPrediction, TorchModelScorer

__all__ = [
    "AdaptiveGroupwiseExecutor",
    "AdaptiveHorizonScheduler",
    "DynamicHorizonConfig",
    "EmpiricalReliabilityPredictor",
    "GroupHorizonDecoder",
    "HorizonDecodeConfig",
    "HorizonPrediction",
    "HorizonScheduleSummary",
    "PreparedReliabilityDataset",
    "TrainingConfig",
    "TorchModelScorer",
    "compare_horizon_sources",
    "constant_prior_scores",
    "make_static_groupwise_executor",
    "prepare_dataset",
    "rows_to_curves",
    "summarize_horizon_schedule",
]

"""Group-wise temporal reliability guided adaptive execution prototype.

The package is deliberately separate from the production executor.  It can
train/evaluate an auxiliary reliability model and can emit rollout-compatible
``ExecutionDecision`` records through ``AdaptiveGroupwiseExecutor`` without
changing the frozen ACT policy or ``src/one_clock/executor.py``.
"""

from .adaptive_executor import AdaptiveGroupwiseExecutor, make_static_groupwise_executor
from .artifacts import PreparedReliabilityDataset, prepare_dataset
from .baselines import EmpiricalReliabilityPredictor, constant_prior_scores
from .causal_features import CausalFeatureContract
from .config import DynamicHorizonConfig, TrainingConfig
from .decoder import GroupHorizonDecoder, HorizonDecodeConfig
from .evaluation import (
    evaluate_shared_checkpoint,
    evaluate_vector_horizon_regret,
    evaluate_vector_predictions,
)
from .horizon_analysis import (
    HorizonRegret,
    HorizonScheduleSummary,
    compare_horizon_sources,
    horizon_regret,
    rows_to_curves,
    summarize_horizon_schedule,
    vector_rows_to_curves,
)
from .model import SharedReliabilityMLP
from .scheduler import (
    AdaptiveHorizonScheduler,
    HorizonPrediction,
    SharedHorizonScheduler,
    SharedTorchModelScorer,
    TorchModelScorer,
)
from .split_manifest import EpisodeSplitManifest
from .vector_dataset import VectorReliabilityDataset, build_vector_dataset
from .vector_training import (
    SharedTrainingResult,
    load_shared_checkpoint,
    predict_reliability_curves,
    train_shared_reliability_model,
)

__all__ = [
    "AdaptiveGroupwiseExecutor",
    "AdaptiveHorizonScheduler",
    "CausalFeatureContract",
    "DynamicHorizonConfig",
    "EpisodeSplitManifest",
    "EmpiricalReliabilityPredictor",
    "GroupHorizonDecoder",
    "HorizonDecodeConfig",
    "HorizonPrediction",
    "HorizonRegret",
    "HorizonScheduleSummary",
    "PreparedReliabilityDataset",
    "SharedReliabilityMLP",
    "SharedHorizonScheduler",
    "SharedTrainingResult",
    "SharedTorchModelScorer",
    "TrainingConfig",
    "TorchModelScorer",
    "VectorReliabilityDataset",
    "build_vector_dataset",
    "compare_horizon_sources",
    "constant_prior_scores",
    "evaluate_vector_predictions",
    "evaluate_shared_checkpoint",
    "evaluate_vector_horizon_regret",
    "horizon_regret",
    "load_shared_checkpoint",
    "make_static_groupwise_executor",
    "prepare_dataset",
    "predict_reliability_curves",
    "rows_to_curves",
    "vector_rows_to_curves",
    "summarize_horizon_schedule",
    "train_shared_reliability_model",
]

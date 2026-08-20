"""Preparation-only components for group-wise temporal reliability studies.

This package intentionally stops at dataset construction, feature/target
materialization, model definition, and offline evaluation.  It does not load
the ACT checkpoint, train an estimator, or change execution behavior.
"""

from .config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATASET_PATH,
    ExperimentPaths,
    FeatureConfig,
    SplitConfig,
    TargetConfig,
)
from .dataset import (
    DatasetManifest,
    EpisodeManifest,
    EpisodeSplit,
    TemporalReliabilityDatasetBuilder,
    build_lerobot_manifest,
    split_episode_ids,
)
from .evaluation import (
    EvaluationResult,
    auroc,
    brier_score,
    calibration_error,
    evaluate_by_group_offset,
    expected_calibration_error,
    evaluate_reliability,
    reliability_curve,
    roc_auc,
)
from .features import FeatureBatch, FeatureEncoder, action_chunk_statistics
from .model import MLPBaseline
from .schema import (
    DEFAULT_LIBERO_GROUPS,
    FrozenTrajectory,
    GroupSpec,
    TemporalExample,
)
from .targets import TargetBatch, TemporalValidityTarget, groupwise_rms_error

__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_LIBERO_GROUPS",
    "DatasetManifest",
    "EpisodeManifest",
    "EpisodeSplit",
    "ExperimentPaths",
    "FeatureBatch",
    "FeatureConfig",
    "FeatureEncoder",
    "FrozenTrajectory",
    "GroupSpec",
    "MLPBaseline",
    "SplitConfig",
    "TargetBatch",
    "TargetConfig",
    "TemporalExample",
    "TemporalReliabilityDatasetBuilder",
    "TemporalValidityTarget",
    "action_chunk_statistics",
    "auroc",
    "brier_score",
    "build_lerobot_manifest",
    "calibration_error",
    "evaluate_by_group_offset",
    "evaluate_reliability",
    "expected_calibration_error",
    "groupwise_rms_error",
    "reliability_curve",
    "roc_auc",
    "split_episode_ids",
]

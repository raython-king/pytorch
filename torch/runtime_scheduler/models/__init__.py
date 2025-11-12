"""
ML Models for Device and Memory Decisions

Machine learning models for device placement, memory eviction,
prefetching, transfer scheduling, and runtime workload scheduling.
"""

from .device_models import (
    DevicePlacementFeatures,
    DevicePlacementModel,
    MemoryEvictionFeatures,
    MemoryEvictionModel,
    ModelManager,
    PrefetchFeatures,
    PrefetchModel,
    TransferSchedulingFeatures,
    TransferSchedulingModel,
    get_model_manager,
)

from .runtime_models import (
    RuntimeGNN,
    PriorityPredictor,
    LatencyPredictor,
    BatchingPredictor,
    OnlineLearner,
    EnsembleScheduler,
    PredictionResult,
    TrainingExample,
    FastGNNLayer,
    create_priority_predictor,
    create_latency_predictor,
    create_batching_predictor,
    create_runtime_gnn,
)

__all__ = [
    # Device Models
    "DevicePlacementFeatures",
    "DevicePlacementModel",
    "MemoryEvictionFeatures",
    "MemoryEvictionModel",
    "ModelManager",
    "PrefetchFeatures",
    "PrefetchModel",
    "TransferSchedulingFeatures",
    "TransferSchedulingModel",
    "get_model_manager",
    # Runtime Models
    "RuntimeGNN",
    "PriorityPredictor",
    "LatencyPredictor",
    "BatchingPredictor",
    "OnlineLearner",
    "EnsembleScheduler",
    "PredictionResult",
    "TrainingExample",
    "FastGNNLayer",
    "create_priority_predictor",
    "create_latency_predictor",
    "create_batching_predictor",
    "create_runtime_gnn",
]

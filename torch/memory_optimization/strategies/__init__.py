"""
Memory Optimization Strategies

Collection of various memory optimization techniques that can be applied
to PyTorch models and training loops.
"""

from .base import OptimizationStrategy, StrategyResult
from .checkpointing import GradientCheckpointingStrategy, ActivationCheckpointingStrategy
from .offloading import CPUOffloadingStrategy, LayerOffloadingStrategy
from .compression import GradientCompressionStrategy
from .precision import MixedPrecisionStrategy
from .batch_size import DynamicBatchSizeStrategy
from .optimizer import MemoryEfficientOptimizerStrategy

__all__ = [
    "OptimizationStrategy",
    "StrategyResult",
    "GradientCheckpointingStrategy",
    "ActivationCheckpointingStrategy",
    "CPUOffloadingStrategy",
    "LayerOffloadingStrategy",
    "GradientCompressionStrategy",
    "MixedPrecisionStrategy",
    "DynamicBatchSizeStrategy",
    "MemoryEfficientOptimizerStrategy",
]

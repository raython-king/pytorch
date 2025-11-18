"""
Base class for optimization strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch.nn as nn


@dataclass
class StrategyResult:
    """Result of applying an optimization strategy"""
    success: bool
    memory_saved: float  # GB
    performance_impact: float  # Relative change in throughput (-1 to 1)
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class OptimizationStrategy(ABC):
    """Base class for all optimization strategies"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = False
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def apply(self, model: nn.Module, **kwargs) -> StrategyResult:
        """Apply the optimization strategy to the model"""
        pass

    @abstractmethod
    def revert(self, model: nn.Module) -> None:
        """Revert the optimization"""
        pass

    @abstractmethod
    def estimate_memory_savings(self, model: nn.Module) -> float:
        """Estimate memory savings in GB"""
        pass

    @abstractmethod
    def estimate_performance_impact(self, model: nn.Module) -> float:
        """Estimate performance impact (-1 to 1, where 0 is neutral)"""
        pass

    def is_applicable(self, model: nn.Module, hardware_caps: Any) -> bool:
        """Check if strategy is applicable given model and hardware"""
        return True

    def get_name(self) -> str:
        """Get strategy name"""
        return self.__class__.__name__

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update strategy metrics"""
        self.metrics.update(metrics)

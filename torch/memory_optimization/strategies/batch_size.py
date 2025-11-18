"""
Dynamic Batch Size Strategy

Automatically adjusts batch size to maximize memory utilization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base import OptimizationStrategy, StrategyResult


class DynamicBatchSizeStrategy(OptimizationStrategy):
    """
    Dynamically adjusts batch size to optimize memory usage and throughput.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.initial_batch_size = config.get('initial_batch_size', 32)
        self.min_batch_size = config.get('min_batch_size', 1)
        self.max_batch_size = config.get('max_batch_size', 512)
        self.increment = config.get('batch_size_increment', 2)
        self.current_batch_size = self.initial_batch_size
        self.optimal_batch_size: Optional[int] = None

    def apply(self, model: nn.Module, **kwargs) -> StrategyResult:
        """Find optimal batch size"""
        try:
            # Binary search for optimal batch size
            optimal_size = self._find_optimal_batch_size(model, **kwargs)

            if optimal_size is None:
                return StrategyResult(
                    success=False,
                    memory_saved=0.0,
                    performance_impact=0.0,
                    metadata={},
                    error_message="Could not find optimal batch size"
                )

            self.optimal_batch_size = optimal_size
            self.current_batch_size = optimal_size
            self.enabled = True

            # Estimate memory impact
            ratio = optimal_size / self.initial_batch_size
            memory_impact = 0.0  # Neutral - just optimizing usage
            performance_impact = (ratio - 1.0) * 0.3  # Larger batch can improve throughput

            return StrategyResult(
                success=True,
                memory_saved=memory_impact,
                performance_impact=performance_impact,
                metadata={
                    'optimal_batch_size': optimal_size,
                    'initial_batch_size': self.initial_batch_size,
                    'ratio': ratio,
                }
            )
        except Exception as e:
            return StrategyResult(
                success=False,
                memory_saved=0.0,
                performance_impact=0.0,
                metadata={},
                error_message=str(e)
            )

    def _find_optimal_batch_size(self, model: nn.Module, **kwargs) -> Optional[int]:
        """Binary search for optimal batch size"""
        low = self.min_batch_size
        high = self.max_batch_size
        optimal = self.initial_batch_size

        while low <= high:
            mid = (low + high) // 2

            if self._test_batch_size(model, mid, **kwargs):
                optimal = mid
                low = mid + self.increment
            else:
                high = mid - self.increment

        return optimal if optimal >= self.min_batch_size else None

    def _test_batch_size(self, model: nn.Module, batch_size: int, **kwargs) -> bool:
        """Test if batch size fits in memory"""
        if not torch.cuda.is_available():
            return True

        try:
            # Get input shape from kwargs or use default
            input_shape = kwargs.get('input_shape', (3, 224, 224))

            # Create dummy batch
            dummy_input = torch.randn(batch_size, *input_shape, device='cuda')

            # Test forward pass
            torch.cuda.empty_cache()
            with torch.no_grad():
                _ = model(dummy_input)

            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

            # Should use < 90% of memory
            return memory_reserved < memory_total * 0.9

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return False
            raise
        except Exception:
            return False

    def get_batch_size(self) -> int:
        """Get current optimal batch size"""
        return self.current_batch_size

    def revert(self, model: nn.Module) -> None:
        """Revert to initial batch size"""
        self.current_batch_size = self.initial_batch_size
        self.optimal_batch_size = None
        self.enabled = False

    def estimate_memory_savings(self, model: nn.Module) -> float:
        """Estimate memory savings"""
        # Dynamic batch sizing optimizes memory usage but doesn't necessarily save memory
        return 0.0

    def estimate_performance_impact(self, model: nn.Module) -> float:
        """Estimate performance impact"""
        # Larger batch sizes typically improve throughput
        if self.optimal_batch_size and self.optimal_batch_size > self.initial_batch_size:
            ratio = self.optimal_batch_size / self.initial_batch_size
            return (ratio - 1.0) * 0.3
        return 0.0

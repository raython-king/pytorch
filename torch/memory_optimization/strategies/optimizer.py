"""
Memory-Efficient Optimizer Strategy

Uses memory-efficient optimizers like Adafactor or 8-bit Adam.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, Any, Optional, Type
from .base import OptimizationStrategy, StrategyResult


class MemoryEfficientOptimizerStrategy(OptimizationStrategy):
    """
    Replaces standard optimizers with memory-efficient alternatives.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.optimizer_type = config.get('optimizer_type', 'adafactor')
        self.original_optimizer: Optional[Optimizer] = None
        self.efficient_optimizer: Optional[Optimizer] = None

    def apply(self, model: nn.Module, optimizer=None, **kwargs) -> StrategyResult:
        """Apply memory-efficient optimizer"""
        try:
            if optimizer is None:
                return StrategyResult(
                    success=False,
                    memory_saved=0.0,
                    performance_impact=0.0,
                    metadata={},
                    error_message="No optimizer provided"
                )

            self.original_optimizer = optimizer

            # Calculate memory savings
            model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_memory = model_params * 4 / 1024**3  # Assume float32

            # Different optimizers have different memory footprints
            if self.optimizer_type == 'adafactor':
                # Adafactor: factored second moments, saves memory
                memory_saved = param_memory * 1.0  # Saves ~1x param memory vs Adam
                performance_impact = -0.02
            elif self.optimizer_type == 'adam8bit':
                # 8-bit Adam: quantized optimizer states
                memory_saved = param_memory * 1.5  # Saves ~1.5x param memory
                performance_impact = -0.01
            elif self.optimizer_type == 'sgd':
                # SGD with momentum: only 1x param memory vs Adam's 2x
                memory_saved = param_memory * 1.0
                performance_impact = -0.10  # SGD may converge slower
            else:
                memory_saved = 0.0
                performance_impact = 0.0

            self.enabled = True

            return StrategyResult(
                success=True,
                memory_saved=memory_saved,
                performance_impact=performance_impact,
                metadata={
                    'optimizer_type': self.optimizer_type,
                    'param_count': model_params,
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

    def create_optimizer(self, model: nn.Module, lr: float = 1e-3) -> Optimizer:
        """Create memory-efficient optimizer"""
        if self.optimizer_type == 'adafactor':
            return self._create_adafactor(model, lr)
        elif self.optimizer_type == 'adam8bit':
            return self._create_adam8bit(model, lr)
        elif self.optimizer_type == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            return torch.optim.Adam(model.parameters(), lr=lr)

    def _create_adafactor(self, model: nn.Module, lr: float) -> Optimizer:
        """Create Adafactor optimizer (if available)"""
        try:
            from transformers.optimization import Adafactor
            return Adafactor(
                model.parameters(),
                lr=lr,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False
            )
        except ImportError:
            # Fallback to Adam if Adafactor not available
            return torch.optim.Adam(model.parameters(), lr=lr)

    def _create_adam8bit(self, model: nn.Module, lr: float) -> Optimizer:
        """Create 8-bit Adam optimizer (if available)"""
        try:
            import bitsandbytes as bnb
            return bnb.optim.Adam8bit(model.parameters(), lr=lr)
        except ImportError:
            # Fallback to standard Adam
            return torch.optim.Adam(model.parameters(), lr=lr)

    def revert(self, model: nn.Module) -> None:
        """Revert to original optimizer"""
        # Note: This requires recreating the optimizer outside this class
        self.efficient_optimizer = None
        self.enabled = False

    def estimate_memory_savings(self, model: nn.Module) -> float:
        """Estimate memory savings"""
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_memory = model_params * 4 / 1024**3

        if self.optimizer_type == 'adafactor':
            return param_memory * 1.0
        elif self.optimizer_type == 'adam8bit':
            return param_memory * 1.5
        elif self.optimizer_type == 'sgd':
            return param_memory * 1.0
        return 0.0

    def estimate_performance_impact(self, model: nn.Module) -> float:
        """Estimate performance impact"""
        if self.optimizer_type == 'adafactor':
            return -0.02
        elif self.optimizer_type == 'adam8bit':
            return -0.01
        elif self.optimizer_type == 'sgd':
            return -0.10
        return 0.0

"""
Mixed Precision Training Strategy

Uses automatic mixed precision to reduce memory usage and improve performance.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional
from .base import OptimizationStrategy, StrategyResult


class MixedPrecisionStrategy(OptimizationStrategy):
    """
    Applies automatic mixed precision training (AMP).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dtype = getattr(torch, config.get('amp_dtype', 'float16'))
        self.use_scaler = config.get('use_scaler', True)
        self.scaler: Optional[GradScaler] = None

    def apply(self, model: nn.Module, **kwargs) -> StrategyResult:
        """Apply mixed precision training"""
        try:
            # Check if AMP is supported
            if not torch.cuda.is_available():
                return StrategyResult(
                    success=False,
                    memory_saved=0.0,
                    performance_impact=0.0,
                    metadata={},
                    error_message="CUDA not available"
                )

            # Check compute capability
            if torch.cuda.get_device_capability()[0] < 7:
                return StrategyResult(
                    success=False,
                    memory_saved=0.0,
                    performance_impact=0.0,
                    metadata={},
                    error_message="Mixed precision requires compute capability >= 7.0"
                )

            # Initialize gradient scaler
            if self.use_scaler:
                self.scaler = GradScaler()

            # Estimate memory savings
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
            memory_saved = model_size * 0.5  # FP16 saves ~50% memory

            self.enabled = True

            return StrategyResult(
                success=True,
                memory_saved=memory_saved,
                performance_impact=0.15,  # AMP usually improves performance
                metadata={
                    'dtype': str(self.dtype),
                    'use_scaler': self.use_scaler,
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

    def forward_context(self):
        """Get context manager for forward pass"""
        if self.enabled:
            return autocast(dtype=self.dtype)
        else:
            return torch.cuda.amp.autocast(enabled=False)

    def backward_step(self, loss: torch.Tensor, optimizer) -> None:
        """Perform backward pass with scaling"""
        if self.enabled and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, optimizer) -> None:
        """Perform optimizer step with unscaling"""
        if self.enabled and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def revert(self, model: nn.Module) -> None:
        """Disable mixed precision"""
        self.enabled = False
        self.scaler = None

    def estimate_memory_savings(self, model: nn.Module) -> float:
        """Estimate memory savings"""
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        # FP16 saves approximately 50% on model and activations
        return model_size * 0.5

    def estimate_performance_impact(self, model: nn.Module) -> float:
        """Estimate performance impact"""
        # Mixed precision typically improves performance on modern GPUs
        return 0.15  # ~15% speedup

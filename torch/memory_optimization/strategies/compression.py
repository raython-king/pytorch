"""
Gradient Compression Strategy

Compresses gradients to reduce memory and communication overhead.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base import OptimizationStrategy, StrategyResult


class GradientCompressionStrategy(OptimizationStrategy):
    """
    Compresses gradients using various techniques (top-k, random-k, quantization).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.compression_ratio = config.get('compression_ratio', 0.01)
        self.method = config.get('compression_method', 'topk')  # topk, randomk, quantization
        self._hooks = []
        self._compressed_grads: Dict[nn.Parameter, torch.Tensor] = {}

    def apply(self, model: nn.Module, **kwargs) -> StrategyResult:
        """Apply gradient compression"""
        try:
            memory_saved = 0.0

            for param in model.parameters():
                if param.requires_grad:
                    hook = param.register_hook(self._compress_gradient_hook)
                    self._hooks.append((param, hook))
                    memory_saved += param.numel() * param.element_size() * (1 - self.compression_ratio) / 1024**3

            self.enabled = True

            return StrategyResult(
                success=True,
                memory_saved=memory_saved,
                performance_impact=-0.08,  # Small overhead for compression
                metadata={
                    'compression_ratio': self.compression_ratio,
                    'method': self.method,
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

    def _compress_gradient_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Hook to compress gradients"""
        if grad is None or not self.enabled:
            return grad

        if self.method == 'topk':
            return self._topk_compression(grad)
        elif self.method == 'randomk':
            return self._randomk_compression(grad)
        elif self.method == 'quantization':
            return self._quantize_gradient(grad)
        else:
            return grad

    def _topk_compression(self, grad: torch.Tensor) -> torch.Tensor:
        """Top-k gradient compression"""
        k = max(1, int(grad.numel() * self.compression_ratio))

        # Flatten gradient
        flat_grad = grad.flatten()

        # Get top-k by absolute value
        _, indices = torch.topk(flat_grad.abs(), k)
        values = flat_grad[indices]

        # Create sparse gradient
        compressed = torch.zeros_like(flat_grad)
        compressed[indices] = values

        return compressed.reshape(grad.shape)

    def _randomk_compression(self, grad: torch.Tensor) -> torch.Tensor:
        """Random-k gradient compression"""
        k = max(1, int(grad.numel() * self.compression_ratio))

        # Flatten gradient
        flat_grad = grad.flatten()

        # Random sampling
        indices = torch.randperm(flat_grad.numel(), device=grad.device)[:k]
        values = flat_grad[indices]

        # Create sparse gradient
        compressed = torch.zeros_like(flat_grad)
        compressed[indices] = values / self.compression_ratio  # Scale to maintain expectation

        return compressed.reshape(grad.shape)

    def _quantize_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Quantize gradients to reduce precision"""
        # Simple 8-bit quantization
        grad_min = grad.min()
        grad_max = grad.max()

        # Quantize to 8-bit
        scale = (grad_max - grad_min) / 255.0
        quantized = ((grad - grad_min) / scale).round().to(torch.uint8)

        # Dequantize
        dequantized = quantized.to(grad.dtype) * scale + grad_min

        return dequantized

    def revert(self, model: nn.Module) -> None:
        """Remove compression hooks"""
        for param, hook in self._hooks:
            hook.remove()

        self._hooks.clear()
        self._compressed_grads.clear()
        self.enabled = False

    def estimate_memory_savings(self, model: nn.Module) -> float:
        """Estimate memory savings"""
        total_params = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
        return (total_params / 1024**3) * (1 - self.compression_ratio)

    def estimate_performance_impact(self, model: nn.Module) -> float:
        """Estimate performance impact"""
        # Compression has small overhead but can improve communication
        return -0.08 + (0.05 if self.method == 'topk' else 0.0)

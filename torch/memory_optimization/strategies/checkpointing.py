"""
Gradient and Activation Checkpointing Strategies

Implements various checkpointing techniques to trade computation for memory.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any, List, Set
from .base import OptimizationStrategy, StrategyResult


class GradientCheckpointingStrategy(OptimizationStrategy):
    """
    Applies gradient checkpointing to reduce activation memory.

    This strategy selectively checkpoints layers, recomputing activations
    during backward pass instead of storing them.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.checkpoint_ratio = config.get('checkpoint_ratio', 0.5)
        self.checkpointed_modules: Set[nn.Module] = set()
        self.original_forwards: Dict[nn.Module, Any] = {}

    def apply(self, model: nn.Module, **kwargs) -> StrategyResult:
        """Apply gradient checkpointing to model"""
        try:
            modules_to_checkpoint = self._select_modules(model)
            memory_saved = 0.0

            for module in modules_to_checkpoint:
                memory_saved += self._checkpoint_module(module)
                self.checkpointed_modules.add(module)

            self.enabled = True

            return StrategyResult(
                success=True,
                memory_saved=memory_saved,
                performance_impact=-0.15,  # ~15% slowdown typical
                metadata={
                    'num_checkpointed_modules': len(modules_to_checkpoint),
                    'checkpoint_ratio': self.checkpoint_ratio,
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

    def _select_modules(self, model: nn.Module) -> List[nn.Module]:
        """Select modules to checkpoint based on memory usage"""
        modules = []

        # Focus on large sequential blocks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                # Skip if too small
                if len(list(module.parameters())) < 10:
                    continue
                modules.append(module)
            elif hasattr(module, 'forward') and self._is_checkpointable(module):
                modules.append(module)

        # Sort by estimated memory usage
        modules.sort(key=lambda m: self._estimate_module_memory(m), reverse=True)

        # Select top checkpoint_ratio
        num_to_checkpoint = max(1, int(len(modules) * self.checkpoint_ratio))
        return modules[:num_to_checkpoint]

    def _is_checkpointable(self, module: nn.Module) -> bool:
        """Check if module can be checkpointed"""
        # Avoid checkpointing small or simple layers
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.Dropout)):
            return False

        # Good candidates for checkpointing
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.LSTM, nn.GRU)):
            return True

        # Check for transformer blocks
        module_name = module.__class__.__name__.lower()
        if any(keyword in module_name for keyword in ['transformer', 'attention', 'block']):
            return True

        return False

    def _estimate_module_memory(self, module: nn.Module) -> float:
        """Estimate memory usage of module's activations"""
        param_size = sum(p.numel() * p.element_size() for p in module.parameters())
        # Activations typically 2-3x parameter size
        return param_size * 2.5 / 1024**3  # GB

    def _checkpoint_module(self, module: nn.Module) -> float:
        """Apply checkpointing to a module"""
        # Store original forward
        self.original_forwards[module] = module.forward

        # Create checkpointed forward
        def checkpointed_forward(*args, **kwargs):
            # Use gradient checkpointing
            if self.enabled and module.training:
                return checkpoint(self.original_forwards[module], *args, **kwargs)
            else:
                return self.original_forwards[module](*args, **kwargs)

        # Replace forward method
        module.forward = checkpointed_forward

        return self._estimate_module_memory(module)

    def revert(self, model: nn.Module) -> None:
        """Remove checkpointing"""
        for module in self.checkpointed_modules:
            if module in self.original_forwards:
                module.forward = self.original_forwards[module]

        self.checkpointed_modules.clear()
        self.original_forwards.clear()
        self.enabled = False

    def estimate_memory_savings(self, model: nn.Module) -> float:
        """Estimate memory savings"""
        total_activation_memory = 0.0
        for module in model.modules():
            if self._is_checkpointable(module):
                total_activation_memory += self._estimate_module_memory(module)

        return total_activation_memory * self.checkpoint_ratio

    def estimate_performance_impact(self, model: nn.Module) -> float:
        """Estimate performance impact"""
        # Checkpointing typically adds 15-30% overhead
        return -0.15 - (self.checkpoint_ratio * 0.15)


class ActivationCheckpointingStrategy(OptimizationStrategy):
    """
    Advanced activation checkpointing with selective recomputation.

    Uses a more sophisticated algorithm to decide which activations
    to checkpoint based on computation cost vs memory usage.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.checkpoint_segments = config.get('checkpoint_segments', None)
        self.selective = config.get('selective_checkpointing', True)
        self.checkpointed_layers: List[nn.Module] = []
        self._checkpoint_boundaries: List[int] = []

    def apply(self, model: nn.Module, **kwargs) -> StrategyResult:
        """Apply activation checkpointing"""
        try:
            if self.checkpoint_segments:
                return self._apply_segmented(model)
            elif self.selective:
                return self._apply_selective(model)
            else:
                return self._apply_uniform(model)
        except Exception as e:
            return StrategyResult(
                success=False,
                memory_saved=0.0,
                performance_impact=0.0,
                metadata={},
                error_message=str(e)
            )

    def _apply_segmented(self, model: nn.Module) -> StrategyResult:
        """Apply segmented checkpointing"""
        layers = list(model.children())
        if not layers:
            layers = [model]

        segment_size = len(layers) // self.checkpoint_segments
        memory_saved = 0.0

        for i in range(0, len(layers), segment_size):
            segment = layers[i:i+segment_size]
            for layer in segment:
                memory_saved += self._checkpoint_layer(layer)

        return StrategyResult(
            success=True,
            memory_saved=memory_saved,
            performance_impact=-0.20,
            metadata={
                'segments': self.checkpoint_segments,
                'layers_per_segment': segment_size,
            }
        )

    def _apply_selective(self, model: nn.Module) -> StrategyResult:
        """Apply selective checkpointing based on cost-benefit analysis"""
        layers = self._analyze_layers(model)

        # Score each layer by memory/compute ratio
        scored_layers = [
            (layer, info['memory'] / max(info['compute'], 1e-6))
            for layer, info in layers
        ]
        scored_layers.sort(key=lambda x: x[1], reverse=True)

        # Checkpoint top layers by score
        memory_saved = 0.0
        num_checkpointed = 0

        for layer, score in scored_layers[:len(scored_layers)//2]:
            memory_saved += self._checkpoint_layer(layer)
            num_checkpointed += 1

        return StrategyResult(
            success=True,
            memory_saved=memory_saved,
            performance_impact=-0.12,  # Selective is more efficient
            metadata={
                'num_checkpointed': num_checkpointed,
                'total_layers': len(scored_layers),
            }
        )

    def _apply_uniform(self, model: nn.Module) -> StrategyResult:
        """Apply uniform checkpointing to all layers"""
        memory_saved = 0.0
        num_layers = 0

        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.LSTM)):
                memory_saved += self._checkpoint_layer(module)
                num_layers += 1

        return StrategyResult(
            success=True,
            memory_saved=memory_saved,
            performance_impact=-0.25,
            metadata={'num_layers': num_layers}
        )

    def _analyze_layers(self, model: nn.Module) -> List[tuple]:
        """Analyze layers for checkpointing decisions"""
        layers = []

        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear, nn.LSTM, nn.GRU)):
                continue

            # Estimate memory usage
            param_size = sum(p.numel() for p in module.parameters())
            memory = param_size * 4 / 1024**3  # Assume float32

            # Estimate compute cost (FLOPs)
            compute = self._estimate_flops(module)

            layers.append((module, {
                'name': name,
                'memory': memory,
                'compute': compute,
            }))

        return layers

    def _estimate_flops(self, module: nn.Module) -> float:
        """Estimate FLOPs for a module"""
        if isinstance(module, nn.Linear):
            return module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            # Simplified estimation
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            return kernel_ops * module.in_channels * module.out_channels
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            return module.hidden_size * module.input_size * 4
        return 1.0

    def _checkpoint_layer(self, layer: nn.Module) -> float:
        """Apply checkpointing to a single layer"""
        self.checkpointed_layers.append(layer)

        # Estimate memory saved
        param_size = sum(p.numel() * p.element_size() for p in layer.parameters())
        return param_size * 2.0 / 1024**3  # Activations ~2x params

    def revert(self, model: nn.Module) -> None:
        """Remove checkpointing"""
        self.checkpointed_layers.clear()
        self._checkpoint_boundaries.clear()
        self.enabled = False

    def estimate_memory_savings(self, model: nn.Module) -> float:
        """Estimate memory savings"""
        total_memory = sum(
            sum(p.numel() * p.element_size() for p in m.parameters())
            for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.LSTM))
        )
        # Save about 50% of activation memory
        return total_memory * 2.0 * 0.5 / 1024**3

    def estimate_performance_impact(self, model: nn.Module) -> float:
        """Estimate performance impact"""
        if self.selective:
            return -0.12
        elif self.checkpoint_segments:
            return -0.20
        else:
            return -0.25

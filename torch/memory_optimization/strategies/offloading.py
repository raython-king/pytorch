"""
CPU Offloading Strategies

Offload parts of the model or gradients to CPU memory to reduce GPU memory usage.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Set
from .base import OptimizationStrategy, StrategyResult


class CPUOffloadingStrategy(OptimizationStrategy):
    """
    Offloads optimizer states and gradients to CPU memory.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.offload_optimizer = config.get('offload_optimizer_state', True)
        self.offload_gradients = config.get('offload_gradients', False)
        self.offload_activations = config.get('offload_activations', True)
        self.offloaded_params: Set[nn.Parameter] = set()
        self._original_data: Dict[nn.Parameter, torch.Tensor] = {}

    def apply(self, model: nn.Module, optimizer=None, **kwargs) -> StrategyResult:
        """Apply CPU offloading"""
        try:
            memory_saved = 0.0

            # Offload optimizer state
            if self.offload_optimizer and optimizer is not None:
                memory_saved += self._offload_optimizer_state(optimizer)

            # Offload gradients
            if self.offload_gradients:
                memory_saved += self._setup_gradient_offloading(model)

            self.enabled = True

            performance_impact = -0.05  # Small overhead for transfers
            if self.offload_gradients:
                performance_impact -= 0.10  # More overhead

            return StrategyResult(
                success=True,
                memory_saved=memory_saved,
                performance_impact=performance_impact,
                metadata={
                    'offload_optimizer': self.offload_optimizer,
                    'offload_gradients': self.offload_gradients,
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

    def _offload_optimizer_state(self, optimizer) -> float:
        """Offload optimizer state to CPU"""
        memory_saved = 0.0

        # For each parameter group
        for group in optimizer.param_groups:
            for param in group['params']:
                if param not in optimizer.state:
                    continue

                state = optimizer.state[param]

                # Move state tensors to CPU
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        memory_saved += value.numel() * value.element_size() / 1024**3
                        state[key] = value.cpu()

        return memory_saved

    def _setup_gradient_offloading(self, model: nn.Module) -> float:
        """Setup hooks for gradient offloading"""
        memory_saved = 0.0

        for param in model.parameters():
            if param.requires_grad:
                # Register hook to offload gradients
                param.register_hook(lambda grad: grad.cpu() if grad is not None else None)
                memory_saved += param.numel() * param.element_size() / 1024**3
                self.offloaded_params.add(param)

        return memory_saved

    def revert(self, model: nn.Module, optimizer=None) -> None:
        """Revert offloading"""
        if optimizer is not None:
            # Move optimizer state back to GPU
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param not in optimizer.state:
                        continue

                    state = optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and not value.is_cuda:
                            state[key] = value.cuda()

        self.offloaded_params.clear()
        self.enabled = False

    def estimate_memory_savings(self, model: nn.Module) -> float:
        """Estimate memory savings"""
        # Optimizer state is typically 2x model parameters (for Adam)
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

        savings = 0.0
        if self.offload_optimizer:
            savings += model_size * 2  # Adam has 2 states per param

        if self.offload_gradients:
            savings += model_size  # Gradients same size as parameters

        return savings

    def estimate_performance_impact(self, model: nn.Module) -> float:
        """Estimate performance impact"""
        impact = -0.05 if self.offload_optimizer else 0.0
        if self.offload_gradients:
            impact -= 0.10  # More overhead from CPU-GPU transfers
        return impact


class LayerOffloadingStrategy(OptimizationStrategy):
    """
    Offloads entire layers to CPU, moving them to GPU only when needed.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.offload_ratio = config.get('offload_ratio', 0.3)
        self.offloaded_modules: List[nn.Module] = []
        self._forward_hooks: List[Any] = []
        self._backward_hooks: List[Any] = []

    def apply(self, model: nn.Module, **kwargs) -> StrategyResult:
        """Apply layer offloading"""
        try:
            # Select layers to offload
            layers_to_offload = self._select_layers(model)

            memory_saved = 0.0
            for layer in layers_to_offload:
                memory_saved += self._offload_layer(layer)

            self.enabled = True

            return StrategyResult(
                success=True,
                memory_saved=memory_saved,
                performance_impact=-0.20,  # Significant overhead from transfers
                metadata={
                    'num_offloaded_layers': len(layers_to_offload),
                    'offload_ratio': self.offload_ratio,
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

    def _select_layers(self, model: nn.Module) -> List[nn.Module]:
        """Select layers to offload"""
        # Get all modules with parameters
        modules_with_params = [
            module for module in model.modules()
            if len(list(module.parameters(recurse=False))) > 0
        ]

        # Sort by memory usage
        modules_with_params.sort(
            key=lambda m: sum(p.numel() for p in m.parameters(recurse=False)),
            reverse=True
        )

        # Select bottom offload_ratio (least frequently used)
        num_to_offload = max(1, int(len(modules_with_params) * self.offload_ratio))
        return modules_with_params[-num_to_offload:]

    def _offload_layer(self, layer: nn.Module) -> float:
        """Offload a layer to CPU"""
        # Calculate memory saved
        memory_saved = sum(
            p.numel() * p.element_size()
            for p in layer.parameters(recurse=False)
        ) / 1024**3

        # Move layer to CPU
        layer.cpu()

        # Register hooks to move layer to GPU during forward/backward
        def pre_forward_hook(module, input):
            module.cuda()
            return None

        def post_forward_hook(module, input, output):
            module.cpu()
            return None

        forward_hook = layer.register_forward_pre_hook(pre_forward_hook)
        backward_hook = layer.register_forward_hook(post_forward_hook)

        self._forward_hooks.append(forward_hook)
        self._backward_hooks.append(backward_hook)
        self.offloaded_modules.append(layer)

        return memory_saved

    def revert(self, model: nn.Module) -> None:
        """Revert layer offloading"""
        # Remove hooks
        for hook in self._forward_hooks + self._backward_hooks:
            hook.remove()

        # Move layers back to GPU
        for layer in self.offloaded_modules:
            layer.cuda()

        self._forward_hooks.clear()
        self._backward_hooks.clear()
        self.offloaded_modules.clear()
        self.enabled = False

    def estimate_memory_savings(self, model: nn.Module) -> float:
        """Estimate memory savings"""
        total_params = sum(p.numel() * p.element_size() for p in model.parameters())
        return (total_params / 1024**3) * self.offload_ratio

    def estimate_performance_impact(self, model: nn.Module) -> float:
        """Estimate performance impact"""
        # Layer offloading has significant overhead due to transfers
        return -0.20 - (self.offload_ratio * 0.10)

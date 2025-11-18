"""
IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

Learns vectors that scale key, value, and feed-forward activations.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from ..config import IA3Config


class IA3Layer(nn.Module):
    """
    IA3 Layer - learns scaling vectors for activations.

    Unlike LoRA which adds matrices, IA3 multiplies activations by learned vectors.
    """

    def __init__(self, dim: int, is_feedforward: bool = False):
        super().__init__()

        self.dim = dim
        self.is_feedforward = is_feedforward

        # Learned scaling vector
        self.ia3_vector = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply IA3 scaling"""
        return x * self.ia3_vector


def inject_ia3(
    model: nn.Module,
    config: Optional[IA3Config] = None,
    **kwargs
) -> nn.Module:
    """
    Inject IA3 into a model.

    Args:
        model: Model to inject IA3 into
        config: IA3 configuration

    Returns:
        Model with IA3
    """
    if config is None:
        config = IA3Config()

    modified_modules = []

    for name, module in model.named_modules():
        # Check if this module should have IA3
        is_target = any(target in name for target in config.target_modules)
        is_feedforward = any(ff in name for ff in config.feedforward_modules)

        if not is_target:
            continue

        # Determine dimension
        dim = None
        if hasattr(module, 'out_features'):
            dim = module.out_features
        elif hasattr(module, 'hidden_size'):
            dim = module.hidden_size

        if dim is None:
            continue

        # Create IA3 layer
        ia3_layer = IA3Layer(dim=dim, is_feedforward=is_feedforward)

        # Wrap module with IA3
        wrapped_module = ModuleWithIA3(module, ia3_layer)

        # Replace module
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]

        if parent_name:
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, wrapped_module)
        else:
            setattr(model, child_name, wrapped_module)

        modified_modules.append(name)

    print(f"Injected IA3 into {len(modified_modules)} modules")

    # Freeze base model
    for name, param in model.named_parameters():
        if 'ia3' not in name:
            param.requires_grad = False

    return model


class ModuleWithIA3(nn.Module):
    """Wrapper that applies IA3 after a module"""

    def __init__(self, base_module: nn.Module, ia3_layer: IA3Layer):
        super().__init__()
        self.base_module = base_module
        self.ia3_layer = ia3_layer

    def forward(self, *args, **kwargs):
        """Forward through base module then IA3 scaling"""
        output = self.base_module(*args, **kwargs)

        # Handle tuple outputs (e.g., from attention)
        if isinstance(output, tuple):
            scaled_output = self.ia3_layer(output[0])
            return (scaled_output,) + output[1:]
        else:
            return self.ia3_layer(output)

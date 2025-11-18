"""
Adapter-based Fine-tuning

Bottleneck adapters inserted after attention and feed-forward layers.
"""

import torch
import torch.nn as nn
from typing import Optional
from ..config import AdapterConfig


class AdapterLayer(nn.Module):
    """
    Bottleneck Adapter Layer.

    Adapters add a small bottleneck layer (down-project, non-linearity, up-project)
    with a residual connection.
    """

    def __init__(
        self,
        input_size: int,
        bottleneck_size: int = 64,
        non_linearity: str = "gelu",
        adapter_dropout: float = 0.1,
        adapter_layernorm_option: str = "in",
        adapter_residual_before_ln: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.adapter_residual_before_ln = adapter_residual_before_ln

        # Down projection
        self.down_project = nn.Linear(input_size, bottleneck_size)

        # Activation
        if non_linearity == "gelu":
            self.non_linear = nn.GELU()
        elif non_linearity == "relu":
            self.non_linear = nn.ReLU()
        elif non_linearity == "swish":
            self.non_linear = nn.SiLU()
        else:
            self.non_linear = nn.GELU()

        # Up projection
        self.up_project = nn.Linear(bottleneck_size, input_size)

        # Dropout
        self.dropout = nn.Dropout(adapter_dropout)

        # Layer norm
        self.adapter_layernorm_option = adapter_layernorm_option
        if adapter_layernorm_option == "in":
            self.adapter_norm_before = nn.LayerNorm(input_size)
        elif adapter_layernorm_option == "out":
            self.adapter_norm_after = nn.LayerNorm(input_size)

        # Initialize with small weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        residual = x

        # Pre-norm
        if self.adapter_layernorm_option == "in":
            x = self.adapter_norm_before(x)

        # Down-project
        x = self.down_project(x)
        x = self.non_linear(x)
        x = self.dropout(x)

        # Up-project
        x = self.up_project(x)
        x = self.dropout(x)

        # Residual connection
        if self.adapter_residual_before_ln:
            x = x + residual
        else:
            # Post-norm
            if self.adapter_layernorm_option == "out":
                x = self.adapter_norm_after(x)
            x = x + residual

        return x


def inject_adapters(
    model: nn.Module,
    config: Optional[AdapterConfig] = None,
    **kwargs
) -> nn.Module:
    """
    Inject adapter layers into a model.

    Args:
        model: Model to inject adapters into
        config: Adapter configuration
        **kwargs: Override config parameters

    Returns:
        Model with adapters injected
    """
    if config is None:
        config = AdapterConfig()

    # Override config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Find modules to add adapters to
    modified_modules = []

    for name, module in model.named_modules():
        # Check if this is a target module
        should_add_adapter = any(
            target in name for target in config.target_modules
        )

        if not should_add_adapter:
            continue

        # Determine input size
        input_size = None
        if hasattr(module, 'hidden_size'):
            input_size = module.hidden_size
        elif hasattr(module, 'out_features'):
            input_size = module.out_features
        elif hasattr(module, 'embed_dim'):
            input_size = module.embed_dim

        if input_size is None:
            continue

        # Create adapter
        adapter = AdapterLayer(
            input_size=input_size,
            bottleneck_size=config.bottleneck_size,
            non_linearity=config.non_linearity,
            adapter_dropout=config.adapter_dropout,
            adapter_layernorm_option=config.adapter_layernorm_option,
            adapter_residual_before_ln=config.adapter_residual_before_ln,
        )

        # Wrap module with adapter
        wrapped_module = ModuleWithAdapter(module, adapter)

        # Replace module
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]

        if parent_name:
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, wrapped_module)
        else:
            setattr(model, child_name, wrapped_module)

        modified_modules.append(name)

    print(f"Injected adapters into {len(modified_modules)} modules")

    # Freeze base model, keep adapters trainable
    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False

    return model


class ModuleWithAdapter(nn.Module):
    """Wrapper that adds adapter after a module"""

    def __init__(self, base_module: nn.Module, adapter: AdapterLayer):
        super().__init__()
        self.base_module = base_module
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        """Forward through base module then adapter"""
        output = self.base_module(*args, **kwargs)

        # If output is tuple (e.g., from attention), only adapt first element
        if isinstance(output, tuple):
            adapted_output = self.adapter(output[0])
            return (adapted_output,) + output[1:]
        else:
            return self.adapter(output)

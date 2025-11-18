"""
LoRA Model Utilities

Functions for injecting LoRA into models, merging, saving, and loading.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
import re
from .layers import LinearLoRA, Conv2dLoRA, EmbeddingLoRA, mark_only_lora_as_trainable
from ..config import LoRAConfig


def inject_lora(
    model: nn.Module,
    config: Optional[LoRAConfig] = None,
    **kwargs
) -> nn.Module:
    """
    Inject LoRA layers into a model.

    Args:
        model: Model to inject LoRA into
        config: LoRA configuration
        **kwargs: Override config parameters

    Returns:
        Model with LoRA layers injected
    """
    if config is None:
        config = LoRAConfig()

    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Track replacements
    replacements = []

    # Find and replace target modules
    for name, module in model.named_modules():
        # Check if module name matches target patterns
        if not _should_apply_lora(name, module, config):
            continue

        # Get parent module and child name
        parent, child_name = _get_parent_module(model, name)
        if parent is None:
            continue

        # Create LoRA version of the module
        lora_module = _create_lora_module(module, config)
        if lora_module is None:
            continue

        # Replace module
        setattr(parent, child_name, lora_module)
        replacements.append(name)

    print(f"Injected LoRA into {len(replacements)} modules: {replacements[:5]}...")

    # Freeze non-LoRA parameters
    mark_only_lora_as_trainable(model, bias=config.bias)

    return model


def _should_apply_lora(name: str, module: nn.Module, config: LoRAConfig) -> bool:
    """Check if LoRA should be applied to this module"""
    # Check module type
    module_type = module.__class__.__name__
    if module_type not in config.target_module_types:
        return False

    # Check if name matches target modules
    for target in config.target_modules:
        if target in name:
            # Check layer pattern if specified
            if config.layers_pattern:
                if not re.search(config.layers_pattern, name):
                    continue

            # Check specific layers if specified
            if config.layers_to_transform is not None:
                # Extract layer number from name
                layer_match = re.search(r'\.(\d+)\.', name)
                if layer_match:
                    layer_num = int(layer_match.group(1))
                    if layer_num not in config.layers_to_transform:
                        continue

            return True

    return False


def _get_parent_module(model: nn.Module, name: str) -> tuple:
    """Get parent module and child name"""
    if '.' not in name:
        return model, name

    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    return parent, parts[-1]


def _create_lora_module(module: nn.Module, config: LoRAConfig) -> Optional[nn.Module]:
    """Create LoRA version of a module"""
    if isinstance(module, nn.Linear):
        return LinearLoRA(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            r=config.r,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            fan_in_fan_out=config.fan_in_fan_out,
            merge_weights=config.merge_weights,
        )

    elif isinstance(module, nn.Conv2d):
        return Conv2dLoRA(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            r=config.r,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            merge_weights=config.merge_weights,
        )

    elif isinstance(module, nn.Embedding):
        return EmbeddingLoRA(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            r=config.r,
            lora_alpha=config.alpha,
            merge_weights=config.merge_weights,
        )

    return None


def merge_lora(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into base weights.

    This is useful for inference to avoid the overhead of separate LoRA computation.
    """
    for module in model.modules():
        if hasattr(module, 'merge') and callable(module.merge):
            module.merge()

    return model


def unmerge_lora(model: nn.Module) -> nn.Module:
    """Unmerge LoRA weights from base weights"""
    for module in model.modules():
        if hasattr(module, 'unmerge') and callable(module.unmerge):
            module.unmerge()

    return model


def save_lora(
    model: nn.Module,
    path: str,
    merge_before_save: bool = False
) -> None:
    """
    Save only LoRA parameters.

    Args:
        model: Model with LoRA layers
        path: Path to save to
        merge_before_save: Whether to merge LoRA weights before saving
    """
    if merge_before_save:
        merge_lora(model)

    # Get only LoRA parameters
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.cpu()

    # Save
    torch.save({
        'lora_state_dict': lora_state_dict,
        'merged': merge_before_save,
    }, path)

    print(f"Saved LoRA parameters to {path} ({len(lora_state_dict)} parameters)")

    if merge_before_save:
        unmerge_lora(model)


def load_lora(
    model: nn.Module,
    path: str,
    strict: bool = True
) -> nn.Module:
    """
    Load LoRA parameters.

    Args:
        model: Model with LoRA layers
        path: Path to load from
        strict: Whether to strictly enforce parameter names

    Returns:
        Model with loaded LoRA parameters
    """
    checkpoint = torch.load(path, map_location='cpu')

    lora_state_dict = checkpoint['lora_state_dict']
    merged = checkpoint.get('merged', False)

    # Load parameters
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)

    if strict and (missing or unexpected):
        print(f"Warning: Missing keys: {missing}")
        print(f"Warning: Unexpected keys: {unexpected}")

    print(f"Loaded LoRA parameters from {path} ({len(lora_state_dict)} parameters)")

    # Handle merged state
    if merged:
        merge_lora(model)

    return model


def get_lora_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Get statistics about LoRA parameters in the model.

    Returns:
        Dictionary with LoRA statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
            lora_modules.append(name)

    return {
        'total_parameters': total_params,
        'lora_parameters': lora_params,
        'trainable_parameters': trainable_params,
        'lora_ratio': lora_params / total_params if total_params > 0 else 0,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'num_lora_modules': len(lora_modules),
        'lora_modules': lora_modules,
    }


def print_lora_stats(model: nn.Module) -> None:
    """Print LoRA statistics"""
    stats = get_lora_stats(model)

    print("\n" + "=" * 80)
    print("LoRA Statistics")
    print("=" * 80)
    print(f"Total parameters: {stats['total_parameters']:,}")
    print(f"LoRA parameters: {stats['lora_parameters']:,}")
    print(f"Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"LoRA ratio: {stats['lora_ratio']:.2%}")
    print(f"Trainable ratio: {stats['trainable_ratio']:.2%}")
    print(f"Number of LoRA modules: {stats['num_lora_modules']}")
    print(f"\nLoRA modules: {stats['lora_modules'][:5]}...")
    print("=" * 80 + "\n")

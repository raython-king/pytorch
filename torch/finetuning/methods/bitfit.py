"""
BitFit - Bias-only Fine-tuning

Only trains bias parameters while freezing all other weights.
"""

import torch
import torch.nn as nn


def configure_bitfit(model: nn.Module) -> nn.Module:
    """
    Configure model for BitFit (bias-only fine-tuning).

    Args:
        model: Model to configure

    Returns:
        Model with only bias parameters trainable
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze bias parameters
    trainable_params = 0
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
            trainable_params += param.numel()

    print(f"BitFit: {trainable_params:,} trainable bias parameters")

    return model


def get_bitfit_stats(model: nn.Module) -> dict:
    """Get BitFit statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    bias_params = sum(p.numel() for n, p in model.named_parameters() if 'bias' in n)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'bias_parameters': bias_params,
        'trainable_parameters': trainable_params,
        'bias_ratio': bias_params / total_params if total_params > 0 else 0,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
    }

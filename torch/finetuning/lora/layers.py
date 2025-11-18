"""
LoRA Layer Implementations

Implements LoRA layers for Linear, Conv2d, and Embedding modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class LoRALayer:
    """
    Base class for LoRA layers.

    LoRA decomposes weight updates as: W = W0 + BA
    where B is (d x r) and A is (r x k), with r << min(d, k)
    """

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        merge_weights: bool = False,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x
        self.merged = False
        self.merge_weights = merge_weights

    @property
    def scaling(self) -> float:
        """Scaling factor for LoRA weights"""
        return self.lora_alpha / self.r


class LinearLoRA(nn.Linear, LoRALayer):
    """
    LoRA applied to Linear layer.

    For a linear layer with weight W ∈ R^{d×k}, LoRA adds:
    h = Wx + BAx where B ∈ R^{d×r}, A ∈ R^{r×k}
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        bias: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out

        # LoRA matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

            # Scaling for merged weights
            self.scaling = self.lora_alpha / self.r

            # Freeze the pretrained weight
            self.weight.requires_grad = False

            # Initialize LoRA weights
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """Initialize LoRA parameters"""
        if hasattr(self, 'lora_A'):
            # Kaiming uniform initialization for A
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Zero initialization for B (so ΔW = BA = 0 initially)
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """Override train to handle weight merging"""
        nn.Linear.train(self, mode)
        if mode and self.merge_weights and self.merged:
            # Unmerge weights during training
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge weights during inference
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA"""
        if self.r > 0 and not self.merged:
            # Compute base linear transformation
            result = F.linear(x, self.weight, bias=self.bias)

            # Add LoRA contribution
            lora_result = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            result = result + lora_result

            return result
        else:
            # Use merged weights or no LoRA
            return F.linear(x, self.weight, bias=self.bias)

    def merge(self):
        """Merge LoRA weights into base weights"""
        if self.r > 0 and not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from base weights"""
        if self.r > 0 and self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False


class Conv2dLoRA(nn.Conv2d, LoRALayer):
    """
    LoRA applied to Conv2d layer.

    Treats convolution as matrix multiplication for LoRA application.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        merge_weights: bool = False,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        # LoRA matrices
        if r > 0:
            # Flatten kernel dimensions
            kernel_size_flat = (
                kernel_size[0] * kernel_size[1]
                if isinstance(kernel_size, tuple)
                else kernel_size * kernel_size
            )
            in_features = in_channels * kernel_size_flat

            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_channels, r)))

            self.scaling = self.lora_alpha / self.r

            # Freeze the pretrained weight
            self.weight.requires_grad = False

            # Initialize LoRA weights
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """Initialize LoRA parameters"""
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA"""
        if self.r > 0 and not self.merged:
            # Base convolution
            result = F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )

            # LoRA contribution via convolution
            # Reshape LoRA matrices for convolution
            lora_weight = (self.lora_B @ self.lora_A).view(
                self.out_channels,
                self.in_channels,
                *self.kernel_size
            ) * self.scaling

            lora_result = F.conv2d(
                self.lora_dropout(x),
                lora_weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )

            return result + lora_result
        else:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )

    def merge(self):
        """Merge LoRA weights"""
        if self.r > 0 and not self.merged:
            lora_weight = (self.lora_B @ self.lora_A).view(
                self.out_channels,
                self.in_channels,
                *self.kernel_size
            ) * self.scaling
            self.weight.data += lora_weight
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights"""
        if self.r > 0 and self.merged:
            lora_weight = (self.lora_B @ self.lora_A).view(
                self.out_channels,
                self.in_channels,
                *self.kernel_size
            ) * self.scaling
            self.weight.data -= lora_weight
            self.merged = False


class EmbeddingLoRA(nn.Embedding, LoRALayer):
    """
    LoRA applied to Embedding layer.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 8,
        lora_alpha: int = 16,
        merge_weights: bool = False,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,  # No dropout for embeddings
            merge_weights=merge_weights,
        )

        # LoRA matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(torch.zeros((embedding_dim, r)))

            self.scaling = self.lora_alpha / self.r

            # Freeze the pretrained weight
            self.weight.requires_grad = False

            # Initialize LoRA weights
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """Initialize LoRA parameters"""
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA"""
        if self.r > 0 and not self.merged:
            # Base embedding lookup
            result = F.embedding(
                x,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse
            )

            # LoRA contribution
            # Create one-hot encoding for indexing
            after_A = F.embedding(
                x,
                self.lora_A.T,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse
            )
            lora_result = (after_A @ self.lora_B.T) * self.scaling

            return result + lora_result
        else:
            return F.embedding(
                x,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse
            )

    def merge(self):
        """Merge LoRA weights"""
        if self.r > 0 and not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights"""
        if self.r > 0 and self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    Freeze all parameters except LoRA parameters.

    Args:
        model: Model with LoRA layers
        bias: How to handle bias parameters ('none', 'all', 'lora_only')
    """
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

    # Handle bias
    if bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters from a model"""
    lora_params = []
    for n, p in model.named_parameters():
        if 'lora_' in n:
            lora_params.append(p)
    return lora_params


def get_lora_state_dict(model: nn.Module) -> dict:
    """Get state dict containing only LoRA parameters"""
    lora_state_dict = {}
    for n, p in model.named_parameters():
        if 'lora_' in n:
            lora_state_dict[n] = p
    return lora_state_dict

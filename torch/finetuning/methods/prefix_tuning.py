"""
Prefix Tuning

Adds trainable prefix tokens to the key and value of each attention layer.
"""

import torch
import torch.nn as nn
from typing import Optional
from ..config import PrefixTuningConfig


class PrefixTuningLayer(nn.Module):
    """
    Prefix Tuning Layer.

    Adds virtual tokens to the beginning of key/value in attention layers.
    """

    def __init__(
        self,
        num_virtual_tokens: int = 20,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        prefix_projection: bool = True,
        projection_hidden_size: int = 512,
        prefix_dropout: float = 0.1,
    ):
        super().__init__()

        self.num_virtual_tokens = num_virtual_tokens
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        # Prefix parameters (for keys and values)
        # Shape: (num_layers, num_virtual_tokens, 2 * hidden_size)
        # The 2x is for key and value
        if prefix_projection:
            # Use MLP reparameterization
            self.embedding = nn.Embedding(num_virtual_tokens, hidden_size)
            self.transform = nn.Sequential(
                nn.Linear(hidden_size, projection_hidden_size),
                nn.Tanh(),
                nn.Linear(projection_hidden_size, num_layers * 2 * hidden_size),
                nn.Dropout(prefix_dropout),
            )
        else:
            # Direct parameterization
            self.prefix_embeddings = nn.Parameter(
                torch.randn(num_layers, num_virtual_tokens, 2 * hidden_size)
            )

        self.prefix_projection = prefix_projection
        self.dropout = nn.Dropout(prefix_dropout)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate prefix key-value pairs.

        Args:
            batch_size: Batch size

        Returns:
            Prefix key-values of shape (num_layers, batch_size, num_virtual_tokens, 2 * hidden_size)
        """
        if self.prefix_projection:
            # Generate prefix through MLP
            prefix_tokens = torch.arange(self.num_virtual_tokens).to(self.embedding.weight.device)
            prefix_embeds = self.embedding(prefix_tokens)  # (num_virtual_tokens, hidden_size)
            prefix_kv = self.transform(prefix_embeds)  # (num_virtual_tokens, num_layers * 2 * hidden_size)

            # Reshape to (num_layers, num_virtual_tokens, 2 * hidden_size)
            prefix_kv = prefix_kv.view(
                self.num_virtual_tokens,
                self.num_layers,
                2 * self.hidden_size
            ).permute(1, 0, 2)
        else:
            prefix_kv = self.prefix_embeddings

        # Expand for batch
        # (num_layers, num_virtual_tokens, 2 * hidden_size) -> (num_layers, batch_size, num_virtual_tokens, 2 * hidden_size)
        prefix_kv = prefix_kv.unsqueeze(1).expand(-1, batch_size, -1, -1)

        return self.dropout(prefix_kv)


def inject_prefix_tuning(
    model: nn.Module,
    config: Optional[PrefixTuningConfig] = None,
    **kwargs
) -> nn.Module:
    """
    Inject prefix tuning into a model.

    Note: This requires model-specific implementation to properly inject
    into attention layers. This is a simplified version.

    Args:
        model: Model to inject prefix tuning into
        config: Prefix tuning configuration

    Returns:
        Model with prefix tuning
    """
    if config is None:
        config = PrefixTuningConfig()

    # This is model-specific and would need to be customized
    # For demonstration, we'll add a prefix module to the model
    if not hasattr(model, 'prefix_tuning'):
        # Try to detect model architecture
        num_layers = getattr(model.config, 'num_hidden_layers', 12) if hasattr(model, 'config') else 12
        num_heads = getattr(model.config, 'num_attention_heads', 12) if hasattr(model, 'config') else 12
        hidden_size = getattr(model.config, 'hidden_size', 768) if hasattr(model, 'config') else 768

        model.prefix_tuning = PrefixTuningLayer(
            num_virtual_tokens=config.num_virtual_tokens,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
            prefix_projection=config.prefix_projection,
            projection_hidden_size=config.projection_hidden_size,
            prefix_dropout=config.prefix_dropout,
        )

        # Freeze base model
        for name, param in model.named_parameters():
            if 'prefix' not in name:
                param.requires_grad = False

        print(f"Injected prefix tuning with {config.num_virtual_tokens} virtual tokens")

    return model

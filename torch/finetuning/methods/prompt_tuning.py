"""
Prompt Tuning

Adds trainable soft prompts to the input embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional
from ..config import PromptTuningConfig


class PromptTuningEmbedding(nn.Module):
    """
    Prompt Tuning - prepends trainable soft prompts to input embeddings.
    """

    def __init__(
        self,
        num_virtual_tokens: int = 8,
        embedding_dim: int = 768,
        init_text: Optional[str] = None,
    ):
        super().__init__()

        self.num_virtual_tokens = num_virtual_tokens
        self.embedding_dim = embedding_dim

        # Soft prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_virtual_tokens, embedding_dim)
        )

        # Initialize
        self.reset_parameters(init_text)

    def reset_parameters(self, init_text: Optional[str] = None):
        """Initialize prompt embeddings"""
        if init_text is None:
            # Random initialization with small values
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
        else:
            # Would initialize from text embeddings if tokenizer provided
            # For now, use random initialization
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        Prepend soft prompts to input embeddings.

        Args:
            input_embeds: Input embeddings of shape (batch_size, seq_len, embedding_dim)

        Returns:
            Embeddings with prompts prepended (batch_size, num_virtual_tokens + seq_len, embedding_dim)
        """
        batch_size = input_embeds.size(0)

        # Expand prompts for batch
        prompts = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate prompts with input embeddings
        prompt_embeds = torch.cat([prompts, input_embeds], dim=1)

        return prompt_embeds


def inject_prompt_tuning(
    model: nn.Module,
    config: Optional[PromptTuningConfig] = None,
    **kwargs
) -> nn.Module:
    """
    Inject prompt tuning into a model.

    Args:
        model: Model to inject prompt tuning into
        config: Prompt tuning configuration

    Returns:
        Model with prompt tuning
    """
    if config is None:
        config = PromptTuningConfig()

    # Detect embedding dimension
    embedding_dim = 768  # Default
    if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        embedding_dim = model.config.hidden_size

    # Create prompt tuning layer
    model.prompt_tuning = PromptTuningEmbedding(
        num_virtual_tokens=config.num_virtual_tokens,
        embedding_dim=embedding_dim,
        init_text=config.prompt_tuning_init_text,
    )

    # Freeze base model
    for name, param in model.named_parameters():
        if 'prompt' not in name:
            param.requires_grad = False

    print(f"Injected prompt tuning with {config.num_virtual_tokens} virtual tokens")

    return model

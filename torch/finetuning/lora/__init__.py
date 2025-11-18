"""
LoRA (Low-Rank Adaptation) Implementation

LoRA is a parameter-efficient fine-tuning method that freezes pretrained model weights
and injects trainable rank decomposition matrices into each layer.
"""

from .layers import LoRALayer, LinearLoRA, Conv2dLoRA, EmbeddingLoRA
from .model import inject_lora, merge_lora, save_lora, load_lora
from ..config import LoRAConfig

__all__ = [
    "LoRALayer",
    "LinearLoRA",
    "Conv2dLoRA",
    "EmbeddingLoRA",
    "inject_lora",
    "merge_lora",
    "save_lora",
    "load_lora",
    "LoRAConfig",
]

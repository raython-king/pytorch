"""
Parameter-Efficient Fine-tuning Methods

Collection of various fine-tuning methods beyond LoRA.
"""

from .adapter import AdapterLayer, inject_adapters
from .prefix_tuning import PrefixTuningLayer, inject_prefix_tuning
from .prompt_tuning import PromptTuningEmbedding, inject_prompt_tuning
from .ia3 import IA3Layer, inject_ia3
from .bitfit import configure_bitfit

__all__ = [
    "AdapterLayer",
    "inject_adapters",
    "PrefixTuningLayer",
    "inject_prefix_tuning",
    "PromptTuningEmbedding",
    "inject_prompt_tuning",
    "IA3Layer",
    "inject_ia3",
    "configure_bitfit",
]

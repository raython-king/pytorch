"""
Multi-Agent Fine-tuning System for PyTorch

This module provides a comprehensive fine-tuning system with multiple parameter-efficient
methods including LoRA, QLoRA, Adapters, Prefix tuning, and more. It uses a multi-agent
architecture to automatically select and configure the best fine-tuning approach for your
model and hardware.

Key Features:
- Multiple fine-tuning methods (LoRA, QLoRA, Adapters, Prefix tuning, etc.)
- Multi-agent architecture for automatic method selection
- Hardware-aware configuration
- Seamless integration with distributed training
- Support for model merging and serving
- Memory-efficient quantization support

Supported Methods:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA with 4-bit/8-bit)
- Adapters (Bottleneck adapters)
- Prefix Tuning
- Prompt Tuning
- IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- BitFit (Bias-only fine-tuning)

Usage:
    >>> from torch.finetuning import FineTuner
    >>>
    >>> # Create fine-tuner with automatic method selection
    >>> finetuner = FineTuner(auto_detect=True)
    >>>
    >>> # Apply fine-tuning to a model
    >>> model = finetuner.prepare_model(model, method='lora')
    >>>
    >>> # Train with fine-tuning
    >>> for batch in dataloader:
    >>>     loss = model(batch)
    >>>     loss.backward()
    >>>     optimizer.step()
"""

from .orchestrator import FineTuningOrchestrator, FineTuner
from .config import FineTuningConfig, FineTuningMethod
from .lora import LoRAConfig, LoRALayer

__all__ = [
    "FineTuningOrchestrator",
    "FineTuner",
    "FineTuningConfig",
    "FineTuningMethod",
    "LoRAConfig",
    "LoRALayer",
]

__version__ = "1.0.0"

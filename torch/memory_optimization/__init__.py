"""
Multi-Agent Memory Optimization System for PyTorch

This module provides an adaptive memory optimization system that uses multiple
AI agents to automatically diagnose hardware capabilities and select the best
memory optimization strategies for maximizing training speed.

Key Features:
- Multi-agent architecture with specialized optimization agents
- Hardware profiling and automatic diagnostics
- Adaptive strategy selection using ML models
- Support for various optimization techniques:
  * Gradient checkpointing
  * Activation checkpointing
  * CPU offloading
  * Gradient compression
  * Mixed precision training
  * Dynamic batch sizing
  * Memory-efficient optimizers
- Real-time performance monitoring and adaptation
- Integration with existing PyTorch training workflows

Usage:
    >>> from torch.memory_optimization import MemoryOptimizer
    >>>
    >>> # Create optimizer with automatic hardware detection
    >>> optimizer = MemoryOptimizer(auto_detect=True)
    >>>
    >>> # Optimize a model
    >>> model = optimizer.optimize_model(model)
    >>>
    >>> # Train with automatic memory management
    >>> for batch in dataloader:
    >>>     with optimizer.optimize_step():
    >>>         loss = model(batch)
    >>>         loss.backward()
    >>>         optimizer_step.step()
"""

from .orchestrator import MemoryOptimizationOrchestrator, MemoryOptimizer
from .diagnostics import HardwareDiagnostics, MemoryProfiler
from .config import MemoryOptimizationConfig

__all__ = [
    "MemoryOptimizationOrchestrator",
    "MemoryOptimizer",
    "HardwareDiagnostics",
    "MemoryProfiler",
    "MemoryOptimizationConfig",
]

__version__ = "1.0.0"

"""
ML-Based Scheduler for PyTorch Inductor

This package provides machine learning models and infrastructure for 
optimizing IR graph scheduling decisions in PyTorch Inductor.

Main Components:
- MLSchedulerOrchestrator: Main entry point for ML-based scheduling
- Feature extractors: Extract features from IR graphs
- Model ensemble: GNN, Transformer, and RL models
- Training infrastructure: Supervised, RL, and imitation learning
- Safety and fallback mechanisms
"""

from .orchestrator import MLSchedulerOrchestrator
from .config import MLSchedulerConfig

__all__ = [
    'MLSchedulerOrchestrator',
    'MLSchedulerConfig',
]

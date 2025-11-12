"""
Configuration for ML Scheduler
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MLSchedulerConfig:
    """Configuration for ML-based scheduler."""
    
    # Model settings
    model_path: Optional[str] = None
    model_type: str = "hybrid"  # "gnn", "transformer", "rl", "hybrid"
    
    # Inference settings
    confidence_threshold: float = 0.75
    cache_predictions: bool = True
    
    # Feedback collection
    collect_feedback: bool = True
    feedback_buffer_size: int = 1000
    
    # Graph size thresholds
    min_nodes_for_ml: int = 5
    max_nodes_for_ml: int = 1000
    
    # Feature extraction
    node_feature_dim: int = 64
    edge_feature_dim: int = 32
    graph_feature_dim: int = 32
    
    # Safety
    fallback_on_error: bool = True
    validate_fusion_plan: bool = True
    
    # Performance
    max_inference_time_ms: float = 50.0
    
    def __repr__(self):
        return (
            f"MLSchedulerConfig(model_type={self.model_type}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"model_path={self.model_path})"
        )

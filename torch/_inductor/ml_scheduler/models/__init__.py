"""
ML models for IR graph scheduling.

This package contains various neural network architectures for learning
optimal scheduling and fusion decisions:

- GNN models: Graph Neural Networks for local fusion and scheduling
- Transformer models: Sequence-based models for global patterns
- RL agents: Reinforcement learning for adaptive optimization
- Ensemble models: Hybrid architectures combining multiple approaches
"""

from typing import Optional

# Check torch_geometric availability
try:
    from .gnn_model import FusionGNN, SchedulingGNN
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    FusionGNN = None
    SchedulingGNN = None

# Transformer models (only requires PyTorch)
from .transformer_model import (
    SchedulingTransformer,
    GraphTransformer,
    PositionalEncoding,
)

# RL agent
from .rl_agent import (
    PPOAgent,
    PolicyNetwork,
    ValueNetwork,
    StateEncoder,
    RolloutBuffer,
    compute_reward,
)

# Ensemble model
from .ensemble import HybridScheduler, EnsembleMode


__all__ = [
    # GNN models
    "FusionGNN",
    "SchedulingGNN",
    "GNN_AVAILABLE",

    # Transformer models
    "SchedulingTransformer",
    "GraphTransformer",
    "PositionalEncoding",

    # RL agent
    "PPOAgent",
    "PolicyNetwork",
    "ValueNetwork",
    "StateEncoder",
    "RolloutBuffer",
    "compute_reward",

    # Ensemble
    "HybridScheduler",
    "EnsembleMode",
]


def get_model(
    model_type: str,
    **kwargs,
) -> Optional[object]:
    """
    Factory function to create models by name.

    Args:
        model_type: Model type string. One of:
            - "fusion_gnn": FusionGNN for pairwise fusion decisions
            - "scheduling_gnn": SchedulingGNN for priority/scheduling
            - "transformer": SchedulingTransformer
            - "graph_transformer": GraphTransformer
            - "ppo_agent": PPO reinforcement learning agent
            - "hybrid": HybridScheduler ensemble
        **kwargs: Model-specific arguments

    Returns:
        Initialized model or None if model_type not recognized

    Raises:
        ImportError: If required dependencies not available
        ValueError: If model_type not recognized

    Examples:
        >>> model = get_model("fusion_gnn", hidden_dim=128, num_layers=4)
        >>> model = get_model("transformer", d_model=256, num_layers=6)
        >>> model = get_model("hybrid", enable_gnn=True, enable_transformer=True)
    """
    model_type = model_type.lower()

    if model_type == "fusion_gnn":
        if not GNN_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install with: pip install torch_geometric"
            )
        return FusionGNN(**kwargs)

    elif model_type == "scheduling_gnn":
        if not GNN_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install with: pip install torch_geometric"
            )
        return SchedulingGNN(**kwargs)

    elif model_type == "transformer":
        return SchedulingTransformer(**kwargs)

    elif model_type == "graph_transformer":
        return GraphTransformer(**kwargs)

    elif model_type == "ppo_agent":
        return PPOAgent(**kwargs)

    elif model_type == "hybrid":
        return HybridScheduler(**kwargs)

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Valid options: fusion_gnn, scheduling_gnn, transformer, "
            f"graph_transformer, ppo_agent, hybrid"
        )


def list_available_models() -> dict:
    """
    List all available models and their dependencies.

    Returns:
        Dictionary mapping model names to availability status

    Example:
        >>> available = list_available_models()
        >>> print(available)
        {
            'fusion_gnn': True,
            'scheduling_gnn': True,
            'transformer': True,
            'ppo_agent': True,
            'hybrid': True
        }
    """
    return {
        'fusion_gnn': GNN_AVAILABLE,
        'scheduling_gnn': GNN_AVAILABLE,
        'transformer': True,  # Only requires PyTorch
        'graph_transformer': True,
        'ppo_agent': True,
        'hybrid': True,  # Can work without GNN
    }


def print_model_info(model_type: str):
    """
    Print detailed information about a model.

    Args:
        model_type: Model type string

    Example:
        >>> print_model_info("fusion_gnn")
        Model: FusionGNN
        Description: GNN for predicting fusion decisions between node pairs
        ...
    """
    info = {
        'fusion_gnn': {
            'name': 'FusionGNN',
            'description': 'Graph Neural Network for predicting fusion decisions between node pairs',
            'architecture': 'GAT-based message passing with pairwise decoder',
            'use_case': 'Local fusion decisions, graph structure understanding',
            'requires': 'torch_geometric',
        },
        'scheduling_gnn': {
            'name': 'SchedulingGNN',
            'description': 'Graph Neural Network for scheduling and priority prediction',
            'architecture': 'GAT-based message passing with multi-head prediction',
            'use_case': 'Execution order, partition assignment, memory planning',
            'requires': 'torch_geometric',
        },
        'transformer': {
            'name': 'SchedulingTransformer',
            'description': 'Transformer for sequential scheduling decisions',
            'architecture': '6-layer encoder with multi-head attention',
            'use_case': 'Global patterns, long-range dependencies',
            'requires': 'PyTorch only',
        },
        'graph_transformer': {
            'name': 'GraphTransformer',
            'description': 'Hybrid architecture combining graph structure with self-attention',
            'architecture': 'Graph-aware transformer layers',
            'use_case': 'Combining local topology with global attention',
            'requires': 'PyTorch only',
        },
        'ppo_agent': {
            'name': 'PPOAgent',
            'description': 'Proximal Policy Optimization agent for adaptive scheduling',
            'architecture': 'Actor-Critic with state encoder',
            'use_case': 'Learning from performance feedback, adaptation',
            'requires': 'PyTorch only',
        },
        'hybrid': {
            'name': 'HybridScheduler',
            'description': 'Ensemble combining GNN, Transformer, and RL',
            'architecture': 'Configurable ensemble with meta-learner',
            'use_case': 'Production deployment with fallback strategies',
            'requires': 'PyTorch (torch_geometric optional)',
        },
    }

    model_type = model_type.lower()
    if model_type not in info:
        print(f"Unknown model type: {model_type}")
        print(f"Available models: {', '.join(info.keys())}")
        return

    model_info = info[model_type]
    print(f"\n{'='*60}")
    print(f"Model: {model_info['name']}")
    print(f"{'='*60}")
    print(f"\nDescription:")
    print(f"  {model_info['description']}")
    print(f"\nArchitecture:")
    print(f"  {model_info['architecture']}")
    print(f"\nUse Case:")
    print(f"  {model_info['use_case']}")
    print(f"\nRequirements:")
    print(f"  {model_info['requires']}")
    print(f"\nAvailable: {list_available_models().get(model_type, False)}")
    print(f"{'='*60}\n")

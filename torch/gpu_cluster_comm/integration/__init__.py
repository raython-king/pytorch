"""
GPU Cluster Communication Integration Module

This module provides integration with PyTorch's distributed training systems.
"""

from .pytorch_integration import (
    TorchDistributedIntegration,
    TransparentOptimization,
    IntegrationMode,
    enable_optimization,
    disable_optimization,
    get_integration,
)

from .backward_compatibility import (
    BackwardCompatibility,
    CompatibilityError,
    CompatibilityWarning,
    get_compatibility,
    check_compatibility,
    print_compatibility_report,
)

from .deployment import (
    ClusterConfig,
    ClusterSetup,
    EnvironmentDetector,
    create_config_template,
    quick_setup,
)

__all__ = [
    # PyTorch Integration
    'TorchDistributedIntegration',
    'TransparentOptimization',
    'IntegrationMode',
    'enable_optimization',
    'disable_optimization',
    'get_integration',

    # Backward Compatibility
    'BackwardCompatibility',
    'CompatibilityError',
    'CompatibilityWarning',
    'get_compatibility',
    'check_compatibility',
    'print_compatibility_report',

    # Deployment
    'ClusterConfig',
    'ClusterSetup',
    'EnvironmentDetector',
    'create_config_template',
    'quick_setup',
]

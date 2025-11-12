"""
Adaptive Flow Control for PyTorch

Production-ready adaptive flow control system for optimizing data transfers
in PyTorch. Provides transparent integration with advanced congestion control,
adaptive policies, and comprehensive monitoring.

Quick Start:
    >>> from torch.adaptive_flow import enable_adaptive_flow, ConfigPresets
    >>> enable_adaptive_flow(ConfigPresets.low_latency())
    >>> # Your existing PyTorch code now has adaptive flow control

Features:
    - Advanced congestion control (BBR, Vegas, DCTCP, TIMELY)
    - Adaptive policies (latency, throughput, fairness, energy)
    - Transparent PyTorch integration
    - Real-time monitoring and visualization
    - Production-ready with extensive testing
"""

# Configuration API
from .config import (
    AdaptiveFlowConfig,
    ConfigPresets,
    ConfigManager,
    PolicyType,
    CongestionControlAlgorithm,
    TargetObjective,
    MonitoringLevel,
    get_config,
    set_config,
    update_config,
    reset_config,
    load_config,
    save_config,
)

# Integration API
from .integration import (
    AdaptiveFlowIntegration,
    enable_adaptive_flow,
    disable_adaptive_flow,
    is_adaptive_flow_enabled,
    get_flow_stats,
)

# Monitoring API
from .flow_monitor import (
    FlowMetricsCollector,
    LinkUtilizationTracker,
    BottleneckDetector,
    PerformanceAnalyzer,
    FlowMetrics,
    LinkMetrics,
    BottleneckInfo,
    MetricType,
)

# Policy API
from .policy_engine import (
    Policy,
    LatencyPolicy,
    ThroughputPolicy,
    FairnessPolicy,
    EnergyPolicy,
    AdaptivePolicy,
    PolicyEngine,
    PolicyObjective,
    PolicyDecision,
    PolicyContext,
    PolicyAction,
)

# Congestion Control API
from .advanced_congestion import (
    CongestionController,
    BBR_Controller,
    Vegas_Controller,
    DCTCP_Controller,
    TIMELY_Controller,
    CongestionState,
    CongestionMetrics,
    RTTSample,
    create_controller,
)

# Visualization API
from .visualization import (
    FlowDashboard,
    start_dashboard,
    stop_dashboard,
    TraceExporter,
    export_chrome_trace,
    export_tensorboard,
)

# Version
__version__ = '1.0.0'

# Export public API
__all__ = [
    # Configuration
    'AdaptiveFlowConfig',
    'ConfigPresets',
    'ConfigManager',
    'PolicyType',
    'CongestionControlAlgorithm',
    'TargetObjective',
    'MonitoringLevel',
    'get_config',
    'set_config',
    'update_config',
    'reset_config',
    'load_config',
    'save_config',

    # Integration
    'AdaptiveFlowIntegration',
    'enable_adaptive_flow',
    'disable_adaptive_flow',
    'is_adaptive_flow_enabled',
    'get_flow_stats',

    # Monitoring
    'FlowMetricsCollector',
    'LinkUtilizationTracker',
    'BottleneckDetector',
    'PerformanceAnalyzer',
    'FlowMetrics',
    'LinkMetrics',
    'BottleneckInfo',
    'MetricType',

    # Policies
    'Policy',
    'LatencyPolicy',
    'ThroughputPolicy',
    'FairnessPolicy',
    'EnergyPolicy',
    'AdaptivePolicy',
    'PolicyEngine',
    'PolicyObjective',
    'PolicyDecision',
    'PolicyContext',
    'PolicyAction',

    # Congestion Control
    'CongestionController',
    'BBR_Controller',
    'Vegas_Controller',
    'DCTCP_Controller',
    'TIMELY_Controller',
    'CongestionState',
    'CongestionMetrics',
    'RTTSample',
    'create_controller',

    # Visualization
    'FlowDashboard',
    'start_dashboard',
    'stop_dashboard',
    'TraceExporter',
    'export_chrome_trace',
    'export_tensorboard',

    # Version
    '__version__',
]

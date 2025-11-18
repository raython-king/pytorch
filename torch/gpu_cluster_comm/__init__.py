"""
GPU Cluster Communication Optimization
GPU集群通讯优化

This package provides advanced communication optimization for distributed
GPU training, including:

- Topology-aware algorithm selection
- Compute-communication overlap
- Message coalescing and compression
- Performance profiling and load balancing

本包提供分布式GPU训练的高级通讯优化，包括：
- 拓扑感知的算法选择
- 计算-通讯重叠
- 消息聚合和压缩
- 性能分析和负载均衡

Example usage:

    from torch.gpu_cluster_comm import get_optimizer

    # Get global optimizer
    optimizer = get_optimizer()

    # Enable auto-optimization
    optimizer.enable_auto_optimization()

    # Optimize AllReduce
    tensor = torch.randn(1000, 1000, device='cuda')
    result = optimizer.optimize_allreduce(tensor)

    # At end of training iteration
    optimizer.step()

    # Print statistics
    optimizer.print_summary()
"""

__version__ = "0.1.0"

# Core components
from .comm_optimizer import (
    GPUClusterCommOptimizer,
    get_optimizer,
    set_optimizer,
    reset_optimizer,
)

# Configuration
from .config import (
    GPUClusterCommConfig,
    TopologyConfig,
    CollectiveConfig,
    OverlapConfig,
    CompressionConfig,
    CoalescingConfig,
    ProfilingConfig,
    LoadBalancingConfig,
    get_config,
    set_config,
    reset_config,
)

# Types
from .types import (
    # Enums
    CollectiveAlgorithm,
    CollectiveOperation,
    ReductionOp,
    InterconnectType,
    CompressionStrategy,
    OperationType,

    # Data structures
    GPUDevice,
    Link,
    TopologyNode,
    CommunicationPlan,
    CommunicationStep,
    CommMetrics,
    CompressedTensor,
    Message,
    CoalescedMessage,
    WorkloadStats,
    Bottleneck,
    CommunicationPattern,
)

# Component classes
from .topology_manager import (
    GPUTopology,
    TopologyManager,
    TopologyDiscovery,
)

from .collective_optimizer import (
    AdaptiveCollectiveOptimizer,
    AlgorithmCostModel,
)

from .overlap_scheduler import (
    OverlapScheduler,
    DependencyGraph,
    GradientBucketing,
)

from .message_coalescing import (
    MessageCoalescer,
    SmartCoalescer,
)

from .compression_manager import (
    CompressionManager,
)

from .communication_profiler import (
    CommunicationProfiler,
    PerformanceMonitor,
)

from .load_balancer import (
    LoadBalancer,
    DynamicLoadBalancer,
)

# Utilities
from .utils import (
    Timer,
    SynchronizedTimer,
    get_tensor_size_bytes,
    compute_bandwidth_gbps,
    estimate_communication_time,
    format_bytes,
    format_bandwidth,
    format_time,
)


__all__ = [
    # Version
    '__version__',

    # Main API
    'GPUClusterCommOptimizer',
    'get_optimizer',
    'set_optimizer',
    'reset_optimizer',

    # Configuration
    'GPUClusterCommConfig',
    'TopologyConfig',
    'CollectiveConfig',
    'OverlapConfig',
    'CompressionConfig',
    'CoalescingConfig',
    'ProfilingConfig',
    'LoadBalancingConfig',
    'get_config',
    'set_config',
    'reset_config',

    # Enums
    'CollectiveAlgorithm',
    'CollectiveOperation',
    'ReductionOp',
    'InterconnectType',
    'CompressionStrategy',
    'OperationType',

    # Data structures
    'GPUDevice',
    'Link',
    'TopologyNode',
    'CommunicationPlan',
    'CommunicationStep',
    'CommMetrics',
    'CompressedTensor',
    'Message',
    'CoalescedMessage',
    'WorkloadStats',
    'Bottleneck',
    'CommunicationPattern',

    # Components
    'GPUTopology',
    'TopologyManager',
    'TopologyDiscovery',
    'AdaptiveCollectiveOptimizer',
    'AlgorithmCostModel',
    'OverlapScheduler',
    'DependencyGraph',
    'GradientBucketing',
    'MessageCoalescer',
    'SmartCoalescer',
    'CompressionManager',
    'CommunicationProfiler',
    'PerformanceMonitor',
    'LoadBalancer',
    'DynamicLoadBalancer',

    # Utilities
    'Timer',
    'SynchronizedTimer',
    'get_tensor_size_bytes',
    'compute_bandwidth_gbps',
    'estimate_communication_time',
    'format_bytes',
    'format_bandwidth',
    'format_time',
]

"""
Multi-Agent Dynamic Data Pipeline for PyTorch

A sophisticated data loading system that implements multi-level caching and
intelligent prefetching across different storage tiers:

Disk → Memory → Redis → GPU

Key Features:
-------------
- **Multi-Agent Architecture**: Specialized agents for each storage layer
- **Intelligent Caching**: Adaptive cache policies (LRU, LFU, ARC)
- **Dynamic Prefetching**: ML-based prediction and pattern detection
- **Multi-Level Pipeline**: Disk → Memory → Redis → GPU
- **Performance Optimization**: Automatic tuning and adaptation
- **CUDA Integration**: Stream management and non-blocking transfers
- **Distributed Support**: Redis-based cross-node caching

Architecture:
------------
The system consists of multiple cooperating agents:

1. **Disk Reader Agent**: Manages disk I/O with async operations
2. **Memory Cache Agent**: In-memory LRU/LFU/ARC caching
3. **Redis Cache Agent**: Distributed caching for multi-node training
4. **GPU Transfer Agent**: CUDA stream management and prefetching
5. **Orchestrator**: Coordinates all agents and manages data flow

Usage:
------
Basic usage:

    >>> from torch.data_pipeline import DataPipelineDataLoader
    >>> from torch.data_pipeline.config import get_default_config
    >>>
    >>> # Create dataloader with pipeline
    >>> config = get_default_config()
    >>> loader = DataPipelineDataLoader(
    ...     dataset=my_dataset,
    ...     config=config,
    ...     batch_size=32,
    ...     shuffle=True
    ... )
    >>>
    >>> # Train with automatic multi-level caching
    >>> for batch in loader:
    ...     output = model(batch)
    ...     loss = criterion(output, target)
    ...     loss.backward()
    ...     optimizer.step()
    >>>
    >>> # Get performance statistics
    >>> stats = loader.get_statistics()
    >>> print(f"Memory hit rate: {stats['memory']['hit_rate']:.2%}")
    >>> print(f"Average latency: {stats['latency_ms']['mean']:.2f}ms")

Advanced usage with custom configuration:

    >>> from torch.data_pipeline.config import DataPipelineConfig
    >>>
    >>> # Create custom configuration
    >>> config = DataPipelineConfig()
    >>> config.memory.max_size_gb = 16.0
    >>> config.redis.enabled = True
    >>> config.redis.host = "redis-server"
    >>> config.gpu.prefetch_queue_size = 4
    >>>
    >>> loader = DataPipelineDataLoader(dataset, config=config)

Preset configurations:

    >>> from torch.data_pipeline.config import (
    ...     get_high_performance_config,
    ...     get_memory_constrained_config,
    ...     get_distributed_config
    ... )
    >>>
    >>> # High-performance configuration for large-scale training
    >>> config = get_high_performance_config()
    >>>
    >>> # Memory-constrained configuration
    >>> config = get_memory_constrained_config()
    >>>
    >>> # Distributed training configuration with Redis
    >>> config = get_distributed_config()

Performance Benefits:
--------------------
- **Reduced I/O Wait**: Multi-level caching reduces disk reads
- **GPU Utilization**: Prefetching keeps GPU busy
- **Automatic Tuning**: Adapts to access patterns
- **Distributed Efficiency**: Shared cache across nodes

Monitoring and Statistics:
-------------------------
The pipeline provides detailed statistics:

    >>> stats = loader.get_statistics()
    >>> print(stats)
    {
        'total_requests': 10000,
        'memory': {
            'hit_rate': 0.85,
            'cache_size_mb': 2048.0,
            'evictions': 125
        },
        'gpu': {
            'prefetch_hit_rate': 0.92,
            'bandwidth_mbps': 12500.0
        },
        'latency_ms': {
            'mean': 1.2,
            'median': 0.8,
            'max': 15.3
        }
    }

Integration with Existing Code:
------------------------------
The DataPipelineDataLoader is designed as a drop-in replacement
for PyTorch's DataLoader:

    >>> # Before: Standard DataLoader
    >>> loader = torch.utils.data.DataLoader(
    ...     dataset, batch_size=32, shuffle=True
    ... )
    >>>
    >>> # After: Pipeline DataLoader (same interface)
    >>> from torch.data_pipeline import DataPipelineDataLoader
    >>> loader = DataPipelineDataLoader(
    ...     dataset, batch_size=32, shuffle=True
    ... )

Configuration Options:
---------------------
See `torch.data_pipeline.config` for full configuration options:

- Disk layer: I/O workers, read-ahead, compression
- Memory layer: Cache size, eviction policy, NUMA awareness
- Redis layer: Connection settings, TTL, cluster mode
- GPU layer: Stream count, prefetch queue, unified memory
- Prefetch: Strategy selection, pattern detection, ML prediction
- Monitoring: Statistics tracking, profiling, visualization

Best Practices:
--------------
1. **Start with defaults**: Use `get_default_config()` initially
2. **Monitor statistics**: Check hit rates and latencies
3. **Tune for workload**: Adjust cache sizes based on dataset
4. **Enable Redis**: For distributed training across nodes
5. **Use GPU prefetch**: Set appropriate queue size for model
6. **Profile patterns**: Let the system learn access patterns

Limitations:
-----------
- Redis requires redis-py package and running Redis server
- GPU features require CUDA-capable device
- Memory usage scales with cache size configuration
- Distributed caching requires network connectivity

See Also:
---------
- `torch.data_pipeline.config`: Configuration classes
- `torch.data_pipeline.orchestrator`: Core orchestration
- `torch.data_pipeline.agents`: Agent implementations
"""

from .config import (
    DataPipelineConfig,
    CachePolicy,
    PrefetchStrategy,
    get_default_config,
    get_high_performance_config,
    get_memory_constrained_config,
    get_distributed_config,
)

from .orchestrator import (
    DataPipelineOrchestrator,
    DataPipelineDataLoader,
)

from .agents.base_agent import (
    AgentRole,
    AgentAction,
    DataRequest,
    DataItem,
)

__all__ = [
    # Main classes
    "DataPipelineDataLoader",
    "DataPipelineOrchestrator",

    # Configuration
    "DataPipelineConfig",
    "CachePolicy",
    "PrefetchStrategy",
    "get_default_config",
    "get_high_performance_config",
    "get_memory_constrained_config",
    "get_distributed_config",

    # Agent types
    "AgentRole",
    "AgentAction",
    "DataRequest",
    "DataItem",
]

__version__ = "1.0.0"

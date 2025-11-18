# GPU Cluster Communication Optimization
# GPU集群通讯优化

Advanced communication optimization for distributed GPU training in PyTorch.

PyTorch分布式GPU训练的高级通讯优化。

## Overview / 概述

This module provides comprehensive optimization for collective communication operations in GPU clusters. It includes:

本模块为GPU集群中的集合通讯操作提供全面优化。包括：

- **Topology-aware algorithm selection** - Automatically selects the best collective algorithm based on GPU topology, message size, and number of ranks
- **Compute-communication overlap** - Overlaps gradient communication with backward computation to hide latency
- **Message coalescing** - Combines small messages to reduce protocol overhead
- **Gradient compression** - Reduces communication volume using FP16, INT8, or sparsification
- **Performance profiling** - Detailed profiling and bottleneck identification
- **Load balancing** - Detects and mitigates stragglers

## Architecture / 架构

```
GPUClusterCommOptimizer (Main Coordinator)
├── TopologyManager - GPU topology discovery and management
├── AdaptiveCollectiveOptimizer - Algorithm selection and planning
├── OverlapScheduler - Compute-communication overlap scheduling
├── MessageCoalescer - Small message aggregation
├── CompressionManager - Gradient compression strategies
├── CommunicationProfiler - Performance monitoring and analysis
└── LoadBalancer - Workload distribution and straggler mitigation
```

## Quick Start / 快速开始

### Basic Usage

```python
import torch
from torch.gpu_cluster_comm import get_optimizer

# Get the global optimizer instance
optimizer = get_optimizer()

# Enable auto-optimization
optimizer.enable_auto_optimization()

# Use in your training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(batch)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Optimize gradient all-reduce
        for param in model.parameters():
            if param.grad is not None:
                param.grad = optimizer.optimize_allreduce(param.grad)

        # Update parameters
        optimizer_step()

        # Step the comm optimizer (for auto-tuning)
        optimizer.step()

# Print optimization statistics
optimizer.print_summary()
```

## Core Components / 核心组件

### 1. TopologyManager - 拓扑管理器

Discovers and manages GPU cluster topology.

### 2. AdaptiveCollectiveOptimizer - 自适应集合优化器

Selects optimal algorithms for collective operations based on message size and topology.

### 3. OverlapScheduler - 重叠调度器

Schedules compute and communication for maximum overlap efficiency.

### 4. CompressionManager - 压缩管理器

Compresses gradients using FP16, INT8, or sparsification strategies.

### 5. CommunicationProfiler - 通讯分析器

Profiles communication performance and identifies bottlenecks.

### 6. LoadBalancer - 负载均衡器

Detects stragglers and rebalances workload for optimal performance.

## API Reference / API参考

### Main API

- `get_optimizer()` - Get global optimizer instance
- `optimize_allreduce(tensor)` - Optimize AllReduce operation
- `optimize_allgather(tensor)` - Optimize AllGather operation
- `optimize_reduce_scatter(tensor)` - Optimize ReduceScatter operation
- `enable_auto_optimization()` - Enable automatic tuning
- `print_summary()` - Print optimization statistics

## Configuration / 配置

See `config.py` for detailed configuration options.

## Performance Tips / 性能建议

1. **Enable bucketing** - Use 25-50 MB buckets for best performance
2. **Use compression wisely** - FP16 is safe for most models
3. **Enable overlap** - Always enable compute-communication overlap
4. **Profile regularly** - Use profiling to identify bottlenecks
5. **Monitor stragglers** - Enable load balancing in heterogeneous clusters

## Documentation / 文档

- `README.md` - This file
- `ALGORITHMS.md` - Detailed algorithm descriptions
- `config.py` - Configuration reference

## License

BSD 3-Clause License (same as PyTorch)

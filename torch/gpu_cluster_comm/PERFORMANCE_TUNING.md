# Performance Tuning Guide

## Overview

This guide provides strategies for maximizing the performance of GPU cluster communication optimization.

## Quick Wins

### 1. Enable All Optimizations

```python
from torch.gpu_cluster_comm.integration import ClusterConfig

config = ClusterConfig(
    num_nodes=nodes,
    num_gpus_per_node=gpus_per_node,
    master_addr=addr,
    master_port=port,
    enable_optimization=True,
    optimization_mode='enabled',
    use_nvlink=True,              # Enable NVLink
    use_hierarchical=True,        # Enable hierarchical communication
    ml_algorithm_selection=True,  # Enable ML-based algorithm selection
    enable_overlap=True,          # Enable computation-communication overlap
)
```

### 2. Optimize Bucket Size

For DDP, bucket size significantly affects performance:

**Small clusters (< 8 GPUs):**
```python
config.bucket_size_mb = 25  # Default
```

**Medium clusters (8-32 GPUs):**
```python
config.bucket_size_mb = 50
```

**Large clusters (> 32 GPUs):**
```python
config.bucket_size_mb = 100
```

### 3. Use Shadow Mode First

Always validate before going to production:

```python
# Week 1: Shadow mode
enable_optimization(mode='shadow')
# Monitor metrics, verify correctness

# Week 2: If all looks good, switch
enable_optimization(mode='enabled')
```

## Performance Analysis

### Measure Current Performance

```python
from torch.gpu_cluster_comm.benchmarks import CollectiveBenchmark

benchmark = CollectiveBenchmark(rank, world_size)
results = benchmark.benchmark_all_operations()
```

### Identify Bottlenecks

```python
from torch.gpu_cluster_comm.benchmarks import profiling_tools

profiler = profiling_tools.CommunicationProfiler()
profiler.profile_communication_pattern(model, dataloader)
profiler.generate_report()  # HTML report with bottleneck analysis
```

## Optimization Strategies by Scenario

### Scenario 1: Small Messages (< 1 MB)

**Problem:** High latency, low throughput

**Solutions:**
1. Enable message coalescing:
```python
config.enable_compression = True
```

2. Use ring algorithm for small messages:
```python
# Automatically selected by ML algorithm selector
config.ml_algorithm_selection = True
```

### Scenario 2: Large Messages (> 100 MB)

**Problem:** Bandwidth not fully utilized

**Solutions:**
1. Increase bucket size:
```python
config.bucket_size_mb = 100
```

2. Enable hierarchical communication:
```python
config.use_hierarchical = True
```

3. Use NVLink for intra-node:
```python
config.use_nvlink = True
```

### Scenario 3: Multi-Node Cluster

**Problem:** Inter-node communication slow

**Solutions:**
1. Enable hierarchical communication:
```python
config.use_hierarchical = True
```

2. Optimize network topology:
- Use InfiniBand if available
- Ensure proper network configuration
- Check for network congestion

### Scenario 4: High Computation/Communication Ratio

**Problem:** Communication time is small but still blocking

**Solution:**
Enable aggressive overlap:
```python
config.enable_overlap = True

# For DDP
integration.integrate_with_ddp(
    model,
    optimize_overlap=True
)
```

### Scenario 5: Heterogeneous Network

**Problem:** Mixed NVLink and Ethernet/IB

**Solution:**
Use hierarchical with topology awareness:
```python
config.use_hierarchical = True
config.use_nvlink = True

# Optimization automatically detects topology
# and uses NVLink for intra-node, IB for inter-node
```

## Advanced Tuning

### 1. Algorithm Selection

Different algorithms for different scenarios:

```python
from torch.gpu_cluster_comm.optimizer import CollectiveOptimizer

optimizer = CollectiveOptimizer()

# Force specific algorithm (override ML selection)
optimizer.set_algorithm('allreduce', 'ring')     # For small messages
optimizer.set_algorithm('allreduce', 'tree')     # For medium messages
optimizer.set_algorithm('allreduce', 'hierarchical')  # For large clusters
```

### 2. Custom Communication Patterns

For custom models with specific communication patterns:

```python
from torch.gpu_cluster_comm.optimizer import CommunicationPattern

# Define custom pattern
pattern = CommunicationPattern(
    operation='allreduce',
    size_range=(1024*1024, 100*1024*1024),
    algorithm='ring',
    chunk_size=1024*1024,
)

optimizer.register_pattern(pattern)
```

### 3. Profiling-Guided Optimization

Use profiling data to guide optimization:

```python
from torch.gpu_cluster_comm.benchmarks import profiling_tools

profiler = profiling_tools.CommunicationProfiler()

# Profile your workload
profiler.profile_communication_pattern(model, dataloader)

# Get recommendations
recommendations = profiler.get_recommendations()

# Apply recommendations
for rec in recommendations:
    print(f"Recommendation: {rec}")
```

## Performance Targets

### Expected Speedups

| Cluster Size | Message Size | Native Time | Optimized Time | Speedup |
|-------------|-------------|-------------|----------------|---------|
| 4 GPUs      | 10 MB       | 2.5 ms      | 1.8 ms         | 1.4x    |
| 8 GPUs      | 10 MB       | 3.2 ms      | 2.1 ms         | 1.5x    |
| 16 GPUs     | 10 MB       | 4.5 ms      | 2.8 ms         | 1.6x    |
| 32 GPUs     | 10 MB       | 6.0 ms      | 3.5 ms         | 1.7x    |

### Bandwidth Utilization

Target bandwidth utilization:
- **NVLink (intra-node):** > 80% of theoretical (200+ GB/s for V100)
- **InfiniBand (inter-node):** > 70% of theoretical (12.5 GB/s for EDR)
- **Ethernet (inter-node):** > 60% of theoretical (1.25 GB/s for 10GbE)

## Monitoring and Debugging

### Real-time Monitoring

```python
integration = TransparentOptimization.get_instance()

# Get metrics during training
summary = integration.metrics.get_summary()
print(f"Average speedup: {summary['comparison']['avg_speedup']:.2f}x")
```

### Detailed Profiling

```python
from torch.gpu_cluster_comm.benchmarks import profiling_tools

profiler = profiling_tools.CommunicationProfiler()

with profiler:
    # Your training code
    train_epoch(model, dataloader)

profiler.visualize_timeline()  # Shows communication timeline
```

## Common Performance Issues

### Issue 1: No Speedup on Small Clusters

**Diagnosis:**
```python
benchmark = CollectiveBenchmark(rank, world_size)
results = benchmark.benchmark_allreduce()

# Check if message size is too small
for size, metrics in results.items():
    if metrics['avg_time'] < 0.001:  # < 1ms
        print(f"Message {size} is too small for optimization overhead")
```

**Solution:** Increase bucket size or disable optimization for small messages

### Issue 2: Degraded Performance Under Load

**Diagnosis:**
```python
validator = PerformanceValidator(rank, world_size)
validator.stress_test(duration_seconds=300)  # 5 minutes
```

**Solution:** May indicate memory issues or network congestion

### Issue 3: Inconsistent Performance

**Diagnosis:**
```python
# Check variance in measurements
results = benchmark.benchmark_allreduce()
for size, metrics in results.items():
    variance = metrics['std_time'] / metrics['avg_time']
    if variance > 0.1:  # > 10% variance
        print(f"High variance for size {size}: {variance:.2%}")
```

**Solution:** May indicate:
- Network congestion
- Thermal throttling
- Background processes

## Best Practices Checklist

- [ ] Started with shadow mode for validation
- [ ] Measured baseline performance
- [ ] Tuned bucket size for cluster size
- [ ] Enabled hierarchical for multi-node
- [ ] Enabled NVLink detection
- [ ] Enabled ML algorithm selection
- [ ] Profiled communication patterns
- [ ] Validated correctness
- [ ] Verified minimum 20% speedup
- [ ] Monitored overhead (< 1%)
- [ ] Tested under load
- [ ] Documented configuration

## Performance Optimization Workflow

1. **Baseline Measurement**
   ```bash
   torchrun --nproc_per_node=8 train.py --no-optimization
   # Record: throughput, time per epoch
   ```

2. **Enable Shadow Mode**
   ```bash
   torchrun --nproc_per_node=8 train.py --mode=shadow
   # Compare metrics
   ```

3. **Analyze Results**
   - Check speedup for each operation
   - Identify bottlenecks
   - Verify correctness

4. **Tune Configuration**
   - Adjust bucket size
   - Enable/disable specific optimizations
   - Test different algorithms

5. **Production Deployment**
   ```bash
   torchrun --nproc_per_node=8 train.py --mode=enabled
   # Monitor performance
   ```

6. **Continuous Monitoring**
   - Track metrics over time
   - Watch for regressions
   - Adjust configuration as needed

## Additional Resources

- See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for setup
- See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for expected performance
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

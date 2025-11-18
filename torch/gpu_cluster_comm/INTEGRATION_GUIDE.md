# GPU Cluster Communication Optimization - Integration Guide

## Overview

This guide explains how to integrate GPU cluster communication optimization into your PyTorch distributed training workflows.

## Quick Start

### 1. Zero-Code-Change Integration

The simplest way to use the optimization:

```python
import torch.distributed as dist
from torch.gpu_cluster_comm.integration import TransparentOptimization

# Initialize distributed
dist.init_process_group(backend='nccl', ...)

# Enable optimization - that's it!
TransparentOptimization.enable_auto_optimization()

# Your existing distributed code works unchanged
dist.all_reduce(tensor)
```

### 2. DDP Integration

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.gpu_cluster_comm.integration import enable_optimization

# Enable optimization
enable_optimization(mode='shadow')  # Start with shadow mode for safety

# Create your model
model = nn.Sequential(
    nn.Linear(1000, 2000),
    nn.ReLU(),
    nn.Linear(2000, 1000)
).cuda()

# Wrap with DDP - optimization is automatic
ddp_model = DDP(model)

# Train as usual - gradient synchronization is automatically optimized
for data in dataloader:
    output = ddp_model(data)
    loss = criterion(output, target)
    loss.backward()  # Communication happens here, optimized!
    optimizer.step()
```

## Integration Modes

The optimization supports three modes:

### 1. SHADOW Mode (Recommended for Initial Deployment)

```python
from torch.gpu_cluster_comm.integration import IntegrationMode

TransparentOptimization.enable_auto_optimization(mode=IntegrationMode.SHADOW)
```

- Runs both native and optimized implementations in parallel
- Verifies correctness automatically
- Collects performance metrics for comparison
- Always returns native results (safe)
- Use this mode to validate optimization in production

### 2. ENABLED Mode (For Production)

```python
TransparentOptimization.enable_auto_optimization(mode=IntegrationMode.ENABLED)
```

- Uses only optimized implementation
- Maximum performance
- Includes automatic fallback on errors
- Use after validating with shadow mode

### 3. DISABLED Mode

```python
TransparentOptimization.enable_auto_optimization(mode=IntegrationMode.DISABLED)
```

- No optimization, uses native PyTorch
- Useful for debugging or comparison

## Advanced Integration

### 1. Selective Operation Optimization

You can selectively optimize specific operations:

```python
from torch.gpu_cluster_comm.integration import TorchDistributedIntegration

integration = TorchDistributedIntegration()

# Only optimize allreduce
integration.hook_allreduce()

# Don't hook other operations
```

### 2. Custom Configuration

```python
from torch.gpu_cluster_comm.integration import ClusterConfig, ClusterSetup

config = ClusterConfig(
    num_nodes=4,
    num_gpus_per_node=8,
    master_addr='192.168.1.100',
    master_port=29500,
    enable_optimization=True,
    optimization_mode='shadow',
    bucket_size_mb=50,  # Larger buckets for large clusters
    use_hierarchical=True,  # Enable hierarchical communication
)

setup = ClusterSetup(config)
setup.initialize_process_group(rank, world_size)
setup.apply_optimizations()
```

### 3. DDP with Custom Hooks

```python
from torch.gpu_cluster_comm.integration import TorchDistributedIntegration

integration = TorchDistributedIntegration()

# Optimize DDP specifically
integration.integrate_with_ddp(
    ddp_model,
    optimize_bucket_size=True,
    optimize_overlap=True
)
```

## Best Practices

### 1. Start with Shadow Mode

Always start with shadow mode in production:

```python
# Phase 1: Validation
enable_optimization(mode='shadow')
# Train for a few epochs, monitor metrics

# Phase 2: If validation passes, switch to enabled
disable_optimization()
enable_optimization(mode='enabled')
```

### 2. Monitor Metrics

```python
integration = TransparentOptimization.get_instance()

# During or after training
if integration:
    integration.print_metrics()
```

Output:
```
======================================================================
GPU Cluster Communication Optimization Metrics
======================================================================

ALLREDUCE:
  Count: 1000
  Avg Time: 5.23 ms
  Total Time: 5.23 s

COMPARISON (Shadow Mode):
  Num Comparisons: 1000
  Avg Speedup: 1.45x
  Min Speedup: 1.12x
  Max Speedup: 2.31x
======================================================================
```

### 3. Validate Correctness

Use the validation tools:

```python
from torch.gpu_cluster_comm.validation import validate_system

# Run comprehensive validation
success = validate_system(rank, world_size)

if success:
    print("All validations passed!")
```

### 4. Performance Tuning

For large clusters, tune these parameters:

```python
config = ClusterConfig(
    # Increase bucket size for large clusters
    bucket_size_mb=100 if world_size > 32 else 25,

    # Enable hierarchical for multi-node
    use_hierarchical=True if num_nodes > 1 else False,

    # Enable ML-based algorithm selection
    ml_algorithm_selection=True,
)
```

## Troubleshooting

### Issue: No Speedup Observed

**Possible causes:**
1. Network is already saturated (check bandwidth)
2. Message sizes are too small (< 1MB)
3. Computation time dominates communication time

**Solutions:**
- Check network utilization
- Increase bucket size
- Enable overlap optimization

### Issue: Numerical Differences

**Solution:**
- This should not happen - report as a bug
- Use shadow mode to identify the operation
- Check tolerance settings

### Issue: Slowdown Instead of Speedup

**Possible causes:**
1. Overhead of optimization layer
2. Suboptimal algorithm selection

**Solutions:**
- Check overhead validation: `validator.validate_overhead()`
- Disable ML algorithm selection temporarily
- Profile with: `torch.gpu_cluster_comm.benchmarks.profiling_tools`

## Environment-Specific Setup

### SLURM

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

srun python train.py --enable-optimization --mode=shadow
```

### Kubernetes

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: training
    image: pytorch:latest
    env:
    - name: GPU_CLUSTER_OPTIMIZATION
      value: "enabled"
    - name: OPTIMIZATION_MODE
      value: "shadow"
```

### Docker

```dockerfile
FROM pytorch/pytorch:latest

RUN pip install torch-gpu-cluster-comm

ENV GPU_CLUSTER_OPTIMIZATION=enabled
ENV OPTIMIZATION_MODE=shadow

CMD ["torchrun", "--nproc_per_node=8", "train.py"]
```

## API Reference

### Main Functions

- `enable_optimization(mode='shadow')`: Enable optimization
- `disable_optimization()`: Disable optimization
- `get_integration()`: Get current integration instance

### Classes

- `TorchDistributedIntegration`: Main integration class
- `IntegrationMode`: Enum for modes (DISABLED, SHADOW, ENABLED)
- `ClusterConfig`: Configuration for cluster setup
- `ClusterSetup`: Setup and initialization utilities

## Examples

See `examples/gpu_cluster_comm/` for complete examples:
- `ddp_example.py`: DDP training
- `fsdp_example.py`: FSDP training
- `pipeline_parallel_example.py`: Pipeline parallelism
- `custom_optimization.py`: Custom optimization strategies

## Performance Targets

Expected performance improvements:

| Message Size | Operation | Expected Speedup |
|-------------|-----------|------------------|
| < 1 MB      | AllReduce | 1.1x - 1.3x     |
| 1-100 MB    | AllReduce | 1.3x - 1.8x     |
| > 100 MB    | AllReduce | 1.5x - 2.0x     |
| Any         | AllGather | 1.2x - 1.5x     |

Actual speedup depends on:
- Network topology (NVLink, InfiniBand, etc.)
- Cluster size
- Message size distribution
- Hardware capabilities

## Support

For issues, questions, or feature requests:
- GitHub: https://github.com/pytorch/pytorch
- Docs: https://pytorch.org/docs/stable/gpu_cluster_comm.html
- Forum: https://discuss.pytorch.org/

## Next Steps

1. Read [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) for optimization tips
2. Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
3. Check [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for expected performance

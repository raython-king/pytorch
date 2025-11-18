# GPU Cluster Communication Optimization Examples

This directory contains examples demonstrating how to use GPU cluster communication optimization in various scenarios.

## Examples

### 1. simple_example.py
The simplest possible example - just one line to enable optimization.

```bash
torchrun --nproc_per_node=2 simple_example.py
```

**What it demonstrates:**
- Basic setup with one-line optimization
- AllReduce operation
- Minimal code changes

### 2. ddp_example.py
Complete DDP training example with optimization.

```bash
torchrun --nproc_per_node=4 ddp_example.py
```

**What it demonstrates:**
- DistributedDataParallel integration
- Training loop with gradient synchronization
- Shadow mode for validation
- Metrics collection

**Options:**
```bash
# Disable optimization (baseline)
torchrun --nproc_per_node=4 ddp_example.py --no-optimization

# Use enabled mode (maximum performance)
torchrun --nproc_per_node=4 ddp_example.py --mode=enabled

# Use shadow mode (validation)
torchrun --nproc_per_node=4 ddp_example.py --mode=shadow
```

### 3. fsdp_example.py
FSDP (Fully Sharded Data Parallel) training with optimization.

```bash
torchrun --nproc_per_node=8 fsdp_example.py
```

**What it demonstrates:**
- FSDP integration
- Large model training
- Sharded parameter synchronization

### 4. pipeline_parallel_example.py
Pipeline parallelism with optimized communication.

```bash
torchrun --nproc_per_node=8 pipeline_parallel_example.py
```

**What it demonstrates:**
- Pipeline parallel training
- Inter-stage communication optimization
- Multi-stage model splitting

### 5. custom_optimization.py
Advanced example with custom optimization strategies.

```bash
torchrun --nproc_per_node=4 custom_optimization.py
```

**What it demonstrates:**
- Selective operation optimization
- Custom configuration
- Fine-grained control
- Profiling and metrics

## Quick Start

1. **Install PyTorch** (if not already installed):
```bash
pip install torch
```

2. **Run an example**:
```bash
cd /path/to/pytorch/examples/gpu_cluster_comm
torchrun --nproc_per_node=2 simple_example.py
```

3. **Modify for your use case**:
- Copy an example
- Replace model with your model
- Adjust configuration
- Run and monitor metrics

## Common Patterns

### Pattern 1: Zero-Change Integration

```python
import torch.distributed as dist
from torch.gpu_cluster_comm import enable_optimization

dist.init_process_group(backend='nccl')
enable_optimization()  # That's it!

# Your existing code...
```

### Pattern 2: Safe Validation

```python
# Start with shadow mode
enable_optimization(mode='shadow')

# Train for a few epochs, check metrics
train_loop()

# If good, switch to enabled mode
disable_optimization()
enable_optimization(mode='enabled')
```

### Pattern 3: Custom Configuration

```python
from torch.gpu_cluster_comm import ClusterConfig, ClusterSetup

config = ClusterConfig(
    num_nodes=4,
    num_gpus_per_node=8,
    enable_optimization=True,
    optimization_mode='shadow',
)

setup = ClusterSetup(config)
setup.initialize_process_group(rank, world_size)
setup.apply_optimizations()
```

## Performance Tips

1. **Start with shadow mode** - Validate before production
2. **Monitor metrics** - Check speedup and correctness
3. **Tune bucket size** - Larger for big clusters
4. **Enable hierarchical** - For multi-node setups
5. **Profile communication** - Identify bottlenecks

## Troubleshooting

### No GPU available
```
RuntimeError: CUDA not available
```
**Solution:** These examples require CUDA-capable GPUs.

### NCCL errors
```
RuntimeError: NCCL error
```
**Solution:** Check NCCL installation and network configuration.

### Import errors
```
ImportError: cannot import gpu_cluster_comm
```
**Solution:** Ensure you're using PyTorch with this module installed.

## Next Steps

- Read [INTEGRATION_GUIDE.md](../../torch/gpu_cluster_comm/INTEGRATION_GUIDE.md)
- Check [PERFORMANCE_TUNING.md](../../torch/gpu_cluster_comm/PERFORMANCE_TUNING.md)
- Run benchmarks to measure your performance
- Customize for your specific workload

## Support

For questions or issues:
- PyTorch Forums: https://discuss.pytorch.org/
- GitHub Issues: https://github.com/pytorch/pytorch/issues

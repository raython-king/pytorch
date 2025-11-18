># Multi-Agent Memory Optimization Examples

This directory contains examples demonstrating the Multi-Agent Memory Optimization system for PyTorch.

## Overview

The Multi-Agent Memory Optimization system provides automatic, adaptive memory optimization for PyTorch models across different hardware configurations. It uses multiple AI agents to:

- **Diagnose** hardware capabilities automatically
- **Select** optimal memory optimization strategies
- **Monitor** runtime performance and memory usage
- **Adapt** strategies dynamically during training

## Examples

### 1. Basic Example (`basic_example.py`)

Demonstrates the simplest usage of the memory optimization system:

```bash
python basic_example.py
```

**Features shown:**
- Auto-detection of hardware capabilities
- Automatic strategy selection
- Integration with standard PyTorch training loop
- Real-time adaptation based on metrics

**Key code:**
```python
from torch.memory_optimization import MemoryOptimizer

# Initialize with auto-detection
optimizer = MemoryOptimizer(auto_detect=True)

# Optimize model
model = optimizer.optimize_model(model, optimizer)

# Training with optimization
with optimizer.optimize_step():
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### 2. Distributed Example (`distributed_example.py`)

Shows memory optimization in distributed training:

```bash
# Single machine, multiple GPUs
python distributed_example.py

# Or with torchrun
torchrun --nproc_per_node=4 distributed_example.py
```

**Features shown:**
- Integration with DistributedDataParallel (DDP)
- ZeRO optimization stages
- Gradient compression
- Multi-GPU memory optimization

## Available Strategies

The system can automatically apply these strategies:

1. **Mixed Precision Training** - Uses FP16/BF16 to reduce memory usage
2. **Gradient Checkpointing** - Trades computation for memory
3. **Activation Checkpointing** - Selective recomputation of activations
4. **CPU Offloading** - Offloads optimizer states to CPU memory
5. **Gradient Compression** - Compresses gradients (top-k, random-k, quantization)
6. **Dynamic Batch Sizing** - Automatically finds optimal batch size
7. **Memory-Efficient Optimizers** - Uses Adafactor or 8-bit Adam

## Configuration

### Auto-Detection Mode (Recommended)

```python
from torch.memory_optimization import MemoryOptimizer

# Automatically detect hardware and select strategies
optimizer = MemoryOptimizer(auto_detect=True)
```

### Manual Configuration

```python
from torch.memory_optimization import MemoryOptimizationConfig, MemoryOptimizer

config = MemoryOptimizationConfig()

# Enable specific strategies
config.strategy_preferences = {
    'mixed_precision': 1.0,
    'gradient_checkpointing': 0.9,
    'cpu_offloading': 0.8,
}

# Set thresholds
config.memory_usage_threshold = 0.85
config.target_memory_usage = 0.75

# Enable online learning
config.online_learning = True

optimizer = MemoryOptimizer(config=config)
```

### Hardware-Specific Configuration

```python
from torch.memory_optimization import MemoryOptimizationConfig

# Optimize for specific GPU memory
config = MemoryOptimizationConfig.for_hardware(
    gpu_memory_gb=8.0,  # GPU memory in GB
    num_gpus=4          # Number of GPUs
)
```

## Integration with Existing Systems

### Runtime Scheduler Integration

```python
from torch.memory_optimization.integration import RuntimeSchedulerIntegration

integration = RuntimeSchedulerIntegration(config)
model = integration.setup(model, optimizer)
```

### GPU Cluster Communication Integration

```python
from torch.memory_optimization.integration import GPUClusterCommIntegration

integration = GPUClusterCommIntegration(config)
model = integration.setup(model, optimizer)
```

## Monitoring and Metrics

Get optimization summary:

```python
summary = memory_optimizer.summary()

print(f"Active strategies: {summary['active_strategies']}")
print(f"Memory saved: {summary['total_memory_saved_gb']:.2f} GB")
print(f"Agent weights: {summary['agent_weights']}")
print(f"Strategy rankings: {summary['strategy_rankings']}")
```

## Advanced Usage

### Custom Adaptation

```python
# Provide custom metrics for adaptation
memory_optimizer.adapt({
    'iteration_time': 0.15,
    'throughput': 200.0,
    'memory_usage': 0.82,
    'batch_size': 64,
})
```

### Manual Strategy Control

```python
from torch.memory_optimization.orchestrator import MemoryOptimizationOrchestrator

orchestrator = MemoryOptimizationOrchestrator(config)

# Apply specific strategy
orchestrator._apply_strategy('gradient_checkpointing', model)

# Check active strategies
print(orchestrator.active_strategies)
```

## Performance Tips

1. **Enable auto-detection** for best results across different hardware
2. **Use mixed precision** on GPUs with Tensor Cores (Volta, Turing, Ampere, etc.)
3. **Enable online learning** to improve strategy selection over time
4. **Monitor metrics** and provide feedback for better adaptation
5. **Integrate with existing systems** (runtime scheduler, GPU comm) for coordinated optimization

## Troubleshooting

### High Memory Usage Despite Optimization

- Check if strategies are actually being applied: `optimizer.summary()['active_strategies']`
- Lower `memory_usage_threshold` in config to trigger more aggressive optimization
- Enable CPU offloading for optimizer states

### Performance Degradation

- Some strategies trade compute for memory (e.g., checkpointing)
- Adjust `strategy_preferences` to prefer performance-oriented strategies
- Monitor `performance_impact` in strategy results

### OOM Errors

- Enable emergency mode: set `critical_memory_threshold` lower
- Use `MemoryOptimizationConfig.for_hardware()` with actual GPU memory
- Enable all aggressive strategies: checkpointing, offloading, compression

## Requirements

- PyTorch >= 2.0
- CUDA >= 11.0 (for GPU features)
- Python >= 3.8

Optional dependencies for advanced features:
- `transformers` (for Adafactor optimizer)
- `bitsandbytes` (for 8-bit optimizers)

## Learn More

- See `torch/memory_optimization/README.md` for implementation details
- Check `test/test_memory_optimization.py` for more usage patterns
- Read `torch/memory_optimization/orchestrator.py` for architecture overview

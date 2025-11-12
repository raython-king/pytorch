# Monitoring, Profiling, and Integration Guide

This guide covers the monitoring, profiling, and integration features of the PyTorch Runtime Scheduler.

## Performance Monitoring

The performance monitoring system provides real-time tracking with minimal overhead (< 0.1%).

### Quick Start

```python
from torch.runtime_scheduler import PerformanceMonitor, TimedOperation

monitor = PerformanceMonitor()
monitor.start()

# Automatic timing
with TimedOperation(monitor, "my_operation", "cuda:0"):
    result = expensive_computation()

# Get summary
summary = monitor.get_summary()
print(f"Total operations: {summary['total_operations']}")

monitor.stop()
```

## Profiling

Detailed profiling and analysis:

```python
from torch.runtime_scheduler import RuntimeSchedulerProfiler

profiler = RuntimeSchedulerProfiler(enabled=True)
profiler.start()

# Your code
model(input_data)

profiler.stop()
profiler.export_chrome_trace("trace.json")
```

## Integration with PyTorch

Seamless hooks into PyTorch runtime:

```python
from torch.runtime_scheduler.integration import RuntimeSchedulerHooks, HookMode

hooks = RuntimeSchedulerHooks(mode=HookMode.ENABLED)
hooks.enable()

# Your PyTorch code
model(input_data)

hooks.disable()
```

See the full documentation for detailed API reference and examples.

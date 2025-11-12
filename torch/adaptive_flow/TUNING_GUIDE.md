# Performance Tuning Guide

This guide provides recommendations for optimizing adaptive flow control performance for different workloads and scenarios.

## Quick Tuning Recommendations

### For Low Latency Workloads

**Use Case**: Inference, real-time applications, interactive workloads

**Configuration**:
```python
from torch.adaptive_flow import ConfigPresets, enable_adaptive_flow

enable_adaptive_flow(ConfigPresets.low_latency())
```

**Additional Tuning**:
```python
from torch.adaptive_flow import update_config

update_config(
    latency_target_ms=0.5,  # Aggressive latency target
    enable_priority_queuing=True,  # Prioritize latency-sensitive flows
    congestion_control='timely',  # RTT-based, fast reaction
)
```

**Expected Results**:
- P99 latency reduction: 30-50%
- May sacrifice some throughput
- Best for small transfers

### For High Throughput Workloads

**Use Case**: Batch training, large data transfers, bulk operations

**Configuration**:
```python
enable_adaptive_flow(ConfigPresets.high_throughput())
```

**Additional Tuning**:
```python
update_config(
    policy='throughput',
    congestion_control='bbr',  # Bandwidth-optimal
    cc_initial_cwnd=20,  # Start aggressive
    enable_adaptive_routing=True,  # Use multiple paths
)
```

**Expected Results**:
- 20-40% throughput improvement
- Higher utilization of network links
- Better for large transfers

### For Multi-Tenant/Fair Sharing

**Use Case**: Shared resources, multiple users, fairness required

**Configuration**:
```python
enable_adaptive_flow(ConfigPresets.fair_sharing())
```

**Additional Tuning**:
```python
update_config(
    policy='fairness',
    fairness_threshold=0.8,  # Strict fairness
    congestion_control='dctcp',  # ECN-based for datacenter
    enable_fairness_enforcement=True,
)
```

**Expected Results**:
- Jain's fairness index > 0.8
- All flows get fair share
- Reduced variance in performance

### For Energy Efficiency

**Use Case**: Power-constrained environments, green computing

**Configuration**:
```python
enable_adaptive_flow(ConfigPresets.energy_efficient())
```

**Additional Tuning**:
```python
update_config(
    policy='energy',
    energy_power_cap_w=50.0,  # Set power budget
    congestion_control='vegas',  # Conservative, low power
)
```

### For Distributed Training

**Use Case**: Multi-GPU, multi-node training

**Configuration**:
```python
enable_adaptive_flow(ConfigPresets.distributed_training())
```

**Additional Tuning**:
```python
update_config(
    policy='adaptive',  # Dynamic adaptation
    enable_bottleneck_detection=True,
    enable_fairness_enforcement=True,
    enable_priority_queuing=True,
    congestion_control='dctcp',  # Best for datacenter networks
)
```

## Congestion Control Algorithm Selection

### BBR (Recommended Default)

**Best For**:
- High bandwidth-delay product networks
- Long-distance transfers
- Variable network conditions

**Pros**:
- Achieves high throughput
- Robust to packet loss
- Adapts to varying bandwidth

**Cons**:
- Can be aggressive in datacenter networks
- May cause some latency variance

**Configuration**:
```python
update_config(congestion_control='bbr')
```

### Vegas

**Best For**:
- Low latency requirements
- Stable network conditions
- Energy efficiency

**Pros**:
- Proactive congestion avoidance
- Low packet loss
- Stable behavior

**Cons**:
- Can be conservative
- May underutilize high-BDP links
- Sensitive to RTT measurement accuracy

**Configuration**:
```python
update_config(congestion_control='vegas')
```

### DCTCP

**Best For**:
- Datacenter networks
- Networks with ECN support
- Bursty workloads

**Pros**:
- Low latency in datacenters
- High throughput
- Good burst tolerance

**Cons**:
- Requires ECN support
- Less effective on WAN

**Configuration**:
```python
update_config(congestion_control='dctcp')
```

### TIMELY

**Best For**:
- RDMA networks
- Ultra-low latency requirements
- Datacenter environments

**Pros**:
- Very fast reaction to congestion
- RTT-gradient based (proactive)
- Good for datacenter

**Cons**:
- Requires accurate RTT measurements
- May be too aggressive for some networks

**Configuration**:
```python
update_config(
    congestion_control='timely',
    cc_min_rtt_us=20.0  # Set based on your network
)
```

## Policy Selection

### Adaptive Policy (Recommended Default)

Automatically selects the best strategy based on network state.

**When to Use**:
- General workloads
- Unknown or variable conditions
- When you want hands-off optimization

**Configuration**:
```python
update_config(policy='adaptive')
```

### Latency Policy

Optimizes for minimum latency.

**When to Use**:
- Inference workloads
- Real-time applications
- Interactive systems

**Tuning Parameters**:
```python
update_config(
    policy='latency',
    latency_target_ms=1.0,  # Target latency
    # Lower = more aggressive rate reduction on latency increase
)
```

### Throughput Policy

Maximizes aggregate throughput.

**When to Use**:
- Batch training
- Large file transfers
- Bulk data movement

**Tuning Parameters**:
```python
update_config(
    policy='throughput',
    throughput_target_gbps=10.0  # Optional throughput target
)
```

### Fairness Policy

Ensures fair resource allocation.

**When to Use**:
- Multi-tenant environments
- Shared resources
- When fairness is critical

**Tuning Parameters**:
```python
update_config(
    policy='fairness',
    fairness_threshold=0.7  # Jain's index threshold (0-1)
    # Higher = stricter fairness requirement
)
```

## Monitoring Level Selection

### None

Disables monitoring for minimal overhead.

**When to Use**:
- Production with extreme performance requirements
- After validation in shadow mode

**Overhead**: ~0.01%

### Basic

Collects essential metrics only.

**When to Use**:
- Production deployments
- When detailed metrics not needed

**Overhead**: ~0.1%

### Standard (Recommended Default)

Balanced monitoring with percentiles.

**When to Use**:
- Most production scenarios
- Normal operation

**Overhead**: ~0.3%

### Detailed

Comprehensive metrics with history.

**When to Use**:
- Performance analysis
- Troubleshooting
- Development

**Overhead**: ~0.5%

### Debug

Full instrumentation with traces.

**When to Use**:
- Development
- Debugging issues
- Performance investigation

**Overhead**: ~1-2%

**Configuration**:
```python
update_config(monitoring_level='standard')
```

## Parameter Tuning

### Latency Target

Controls how aggressively the system reacts to latency increases.

```python
update_config(
    latency_target_ms=1.0  # Default
    # Lower = stricter latency requirement
    # Higher = more tolerance for latency
)
```

**Recommendations**:
- Inference: 0.5 - 1.0 ms
- Training: 1.0 - 5.0 ms
- Bulk transfers: 10+ ms

### Fairness Threshold

Minimum acceptable Jain's fairness index.

```python
update_config(
    fairness_threshold=0.7  # Default
    # 1.0 = perfect fairness (may hurt throughput)
    # 0.5 = loose fairness (better throughput)
)
```

**Recommendations**:
- Strict fairness: 0.8 - 0.9
- Balanced: 0.7 - 0.8
- Loose fairness: 0.5 - 0.7

### Congestion Window

Initial congestion window size (in segments).

```python
update_config(
    cc_initial_cwnd=10  # Default (RFC 6928)
    # Higher = faster ramp-up, more aggressive
    # Lower = slower ramp-up, more conservative
)
```

**Recommendations**:
- LAN: 20 - 50
- WAN: 10 - 20
- High-latency: 5 - 10

### Update Interval

How often to update metrics and make decisions.

```python
update_config(
    update_interval=1.0  # seconds
    # Lower = more responsive, higher overhead
    # Higher = less responsive, lower overhead
)
```

**Recommendations**:
- Low latency workloads: 0.1 - 0.5 s
- Normal workloads: 1.0 s
- Batch workloads: 2.0 - 5.0 s

## Shadow Mode for Validation

Test adaptive flow control without affecting behavior:

```python
from torch.adaptive_flow import enable_adaptive_flow, AdaptiveFlowConfig

config = AdaptiveFlowConfig(
    shadow_mode=True,  # Observe only, don't control
    monitoring_level='detailed'  # Collect detailed metrics
)
enable_adaptive_flow(config)

# Run your workload...

# Analyze results
from torch.adaptive_flow import get_performance_report
report = get_performance_report()

# If results look good, disable shadow mode
from torch.adaptive_flow import update_config
update_config(shadow_mode=False)
```

## Troubleshooting Performance Issues

### Issue: High Latency

**Symptoms**:
- P99 latency > target
- Queueing detected

**Solutions**:
```python
# 1. Switch to latency policy
update_config(policy='latency', latency_target_ms=0.5)

# 2. Use TIMELY for fast reaction
update_config(congestion_control='timely')

# 3. Enable priority queuing
update_config(enable_priority_queuing=True)

# 4. Check for bottlenecks
from torch.adaptive_flow import get_performance_report
report = get_performance_report()
if report['bottleneck_count'] > 0:
    update_config(enable_adaptive_routing=True)
```

### Issue: Low Throughput

**Symptoms**:
- Low link utilization
- Underutilization in report

**Solutions**:
```python
# 1. Switch to throughput policy
update_config(policy='throughput')

# 2. Use BBR for bandwidth optimization
update_config(congestion_control='bbr')

# 3. Increase initial window
update_config(cc_initial_cwnd=20)

# 4. Enable adaptive routing
update_config(enable_adaptive_routing=True)
```

### Issue: Poor Fairness

**Symptoms**:
- Jain's fairness index < 0.7
- Large variance in flow throughputs

**Solutions**:
```python
# 1. Switch to fairness policy
update_config(policy='fairness')

# 2. Enable fairness enforcement
update_config(enable_fairness_enforcement=True)

# 3. Raise fairness threshold
update_config(fairness_threshold=0.8)

# 4. Use DCTCP for better fairness
update_config(congestion_control='dctcp')
```

### Issue: High Overhead

**Symptoms**:
- Noticeable performance impact
- High CPU usage from monitoring

**Solutions**:
```python
# 1. Reduce monitoring level
update_config(monitoring_level='basic')

# 2. Increase update interval
update_config(update_interval=2.0)

# 3. Reduce history size
update_config(metrics_history_size=100)

# 4. Disable visualization if running
from torch.adaptive_flow.visualization import stop_dashboard
stop_dashboard()
```

## Performance Benchmarking

### Measure Overhead

```python
import time
import torch
from torch.adaptive_flow import enable_adaptive_flow, disable_adaptive_flow

# Baseline (no adaptive flow)
disable_adaptive_flow()
x = torch.randn(10000, 10000).cuda()

start = time.time()
for i in range(100):
    y = x.to('cpu')
    z = y.to('cuda')
baseline_time = time.time() - start

# With adaptive flow
enable_adaptive_flow()

start = time.time()
for i in range(100):
    y = x.to('cpu')
    z = y.to('cuda')
adaptive_time = time.time() - start

overhead = (adaptive_time - baseline_time) / baseline_time * 100
print(f"Overhead: {overhead:.2f}%")
```

### Measure Latency Improvement

```python
from torch.adaptive_flow import get_performance_report, ConfigPresets
import numpy as np

# Run workload
# ...

report = get_performance_report()
print(f"P99 latency: {report['flow_summary']['p99_latency']*1000:.2f} ms")
print(f"Average latency: {report['flow_summary']['average_latency']*1000:.2f} ms")
```

### Measure Fairness

```python
report = get_performance_report()
print(f"Fairness index: {report['fairness_index']:.3f}")

# Good: > 0.8
# Acceptable: 0.6 - 0.8
# Poor: < 0.6
```

## Best Practices

1. **Start with presets**: Use ConfigPresets for your use case
2. **Use shadow mode first**: Validate before enabling control
3. **Monitor in production**: Use 'standard' monitoring level
4. **Tune incrementally**: Change one parameter at a time
5. **Measure impact**: Use get_performance_report() to validate changes
6. **Enable visualization during development**: Dashboard helps understand behavior
7. **Disable visualization in production**: Unless needed for monitoring
8. **Set realistic targets**: Based on your network capabilities
9. **Consider network topology**: Different settings for different interconnects
10. **Review recommendations**: The system provides tuning suggestions

## Network-Specific Tuning

### NVLink (GPU-to-GPU)

```python
update_config(
    policy='throughput',  # High bandwidth available
    congestion_control='bbr',
    cc_initial_cwnd=50,  # Fast ramp-up
)
```

### PCIe (CPU-GPU)

```python
update_config(
    policy='balanced',
    congestion_control='bbr',
    cc_initial_cwnd=20,
)
```

### Ethernet (Distributed)

```python
update_config(
    policy='adaptive',
    congestion_control='dctcp',  # If ECN available
    enable_adaptive_routing=True,
)
```

### InfiniBand (HPC)

```python
update_config(
    policy='throughput',
    congestion_control='timely',  # RDMA-friendly
    enable_adaptive_routing=True,
)
```

# Adaptive Flow Control API Reference

## Configuration API

### AdaptiveFlowConfig

Main configuration dataclass for the adaptive flow control system.

```python
from torch.adaptive_flow import AdaptiveFlowConfig

config = AdaptiveFlowConfig(
    enabled: bool = True,
    policy: str = "adaptive",
    target: str = "balanced",
    congestion_control: str = "bbr",
    monitoring_level: str = "standard",
    shadow_mode: bool = False,
    max_flows: int = 1000,
    metrics_history_size: int = 1000,
    update_interval: float = 1.0,
    enable_visualization: bool = False,
    visualization_port: int = 8080,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_bottleneck_detection: bool = True,
    enable_fairness_enforcement: bool = True,
    enable_adaptive_routing: bool = True,
    enable_priority_queuing: bool = False,
    latency_target_ms: float = 1.0,
    throughput_target_gbps: Optional[float] = None,
    fairness_threshold: float = 0.7,
    energy_power_cap_w: Optional[float] = None,
    cc_initial_cwnd: int = 10,
    cc_mss: int = 1500,
    cc_min_rtt_us: float = 20.0,
    custom_params: Dict[str, Any] = {}
)
```

**Methods**:
- `to_dict() -> dict`: Convert to dictionary
- `to_json() -> str`: Convert to JSON string
- `save(path)`: Save to file
- `load(path) -> AdaptiveFlowConfig`: Load from file
- `from_dict(data) -> AdaptiveFlowConfig`: Create from dictionary
- `update(**kwargs)`: Update parameters

### ConfigPresets

Pre-defined configurations for common scenarios.

```python
from torch.adaptive_flow import ConfigPresets

# Available presets
config = ConfigPresets.low_latency()
config = ConfigPresets.high_throughput()
config = ConfigPresets.fair_sharing()
config = ConfigPresets.energy_efficient()
config = ConfigPresets.balanced()
config = ConfigPresets.distributed_training()
config = ConfigPresets.debug()

# Or get by name
config = ConfigPresets.get_preset('low_latency')
```

### Configuration Functions

```python
from torch.adaptive_flow import (
    get_config,
    set_config,
    update_config,
    reset_config,
    load_config,
    save_config
)

# Get current configuration
config = get_config()

# Set new configuration
set_config(new_config)

# Update specific parameters
update_config(policy='latency', latency_target_ms=0.5)

# Reset to default
reset_config()

# Load/save configuration
config = load_config('my_config.json')
save_config('my_config.json', config)
```

## Integration API

### enable_adaptive_flow

Enable adaptive flow control globally.

```python
from torch.adaptive_flow import enable_adaptive_flow, AdaptiveFlowConfig

# Enable with default config
enable_adaptive_flow()

# Enable with custom config
config = AdaptiveFlowConfig(policy='latency')
enable_adaptive_flow(config)

# Enable with preset
from torch.adaptive_flow import ConfigPresets
enable_adaptive_flow(ConfigPresets.low_latency())
```

**Parameters**:
- `config` (optional): Configuration dictionary or AdaptiveFlowConfig instance

### disable_adaptive_flow

Disable adaptive flow control.

```python
from torch.adaptive_flow import disable_adaptive_flow

disable_adaptive_flow()
```

### is_adaptive_flow_enabled

Check if adaptive flow control is enabled.

```python
from torch.adaptive_flow import is_adaptive_flow_enabled

if is_adaptive_flow_enabled():
    print("Adaptive flow is active")
```

**Returns**: `bool` - True if enabled

### get_flow_stats

Get flow control statistics.

```python
from torch.adaptive_flow import get_flow_stats

stats = get_flow_stats()
# Returns:
# {
#     'enabled': bool,
#     'shadow_mode': bool,
#     'total_transfers': int,
#     'total_bytes': int,
#     'total_time': float,
#     'average_bandwidth': float,
#     'active_transfers': int,
#     'flow_count': int,
#     'link_count': int
# }
```

### get_performance_report

Get comprehensive performance analysis.

```python
from torch.adaptive_flow import get_performance_report

report = get_performance_report()
# Returns:
# {
#     'timestamp': float,
#     'flow_summary': {
#         'total_flows': int,
#         'total_bytes_sent': int,
#         'total_bytes_lost': int,
#         'overall_loss_rate': float,
#         'average_throughput_bps': float,
#         'average_latency': float,
#         'p95_latency': float,
#         'p99_latency': float
#     },
#     'fairness_index': float,
#     'average_link_utilization': float,
#     'bottleneck_count': int,
#     'issues': [
#         {
#             'type': str,
#             'severity': str,
#             'message': str,
#             'recommendation': str (optional)
#         }
#     ],
#     'recommendations': [str]
# }
```

## Congestion Control API

### CongestionController

Base class for congestion control algorithms.

```python
from torch.adaptive_flow.advanced_congestion import BBR_Controller

controller = BBR_Controller(initial_cwnd=10, mss=1500)

# Process acknowledgment
controller.on_ack(bytes_acked=1500, rtt=0.001)

# Handle loss
controller.on_loss(bytes_lost=1500)

# Get current window
cwnd = controller.get_cwnd()

# Get pacing rate
rate = controller.get_pacing_rate()
```

### create_controller

Factory function for creating controllers.

```python
from torch.adaptive_flow.advanced_congestion import create_controller

# Create BBR controller
controller = create_controller('bbr', initial_cwnd=10, mss=1500)

# Create Vegas controller
controller = create_controller('vegas', initial_cwnd=10, mss=1500)

# Create DCTCP controller
controller = create_controller('dctcp', initial_cwnd=10, mss=1500)

# Create TIMELY controller
controller = create_controller('timely', initial_rate=10e9, mss=1500)
```

## Monitoring API

### FlowMetricsCollector

Collects per-flow metrics.

```python
from torch.adaptive_flow.flow_monitor import FlowMetricsCollector

collector = FlowMetricsCollector(history_size=1000)

# Register flow
collector.register_flow('flow1', 'cpu', 'cuda:0')

# Record transfer
collector.record_transfer(
    flow_id='flow1',
    bytes_sent=1024*1024,
    latency=0.001,
    success=True,
    queue_depth=5
)

# Get flow metrics
metrics = collector.get_flow_metrics('flow1')
# Returns FlowMetrics object with:
# - bytes_sent, bytes_received, bytes_lost
# - throughput_bps
# - latency_mean, latency_p50, latency_p95, latency_p99
# - loss_rate
# - queue_depth_mean, queue_depth_max
# - transfers_completed, transfers_failed

# Get all flows
flows = collector.get_all_flows()

# Get summary
summary = collector.get_summary()
```

### LinkUtilizationTracker

Tracks link bandwidth utilization.

```python
from torch.adaptive_flow.flow_monitor import LinkUtilizationTracker

tracker = LinkUtilizationTracker(update_interval=1.0)

# Register link
tracker.register_link('link1', 'cpu', 'cuda:0', capacity_bps=10e9)

# Record transmission
tracker.record_transmission('link1', bytes_transmitted=1024*1024, flow_id='flow1')

# Get utilization
util = tracker.get_link_utilization('link1')

# Get congested links
congested = tracker.get_congested_links(threshold=0.8)

# Get utilization history
history = tracker.get_link_history('link1', duration=60.0)
```

### BottleneckDetector

Identifies network bottlenecks.

```python
from torch.adaptive_flow.flow_monitor import BottleneckDetector

detector = BottleneckDetector(link_tracker, flow_collector)

# Detect bottlenecks
bottlenecks = detector.detect_bottlenecks()
# Returns list of BottleneckInfo:
# - link_id, src_device, dst_device
# - utilization
# - affected_flows
# - severity ('low', 'medium', 'high', 'critical')

# Get worst bottleneck
worst = detector.get_worst_bottleneck()

# Get affected flows
flows = detector.get_bottleneck_flows('link1')
```

### PerformanceAnalyzer

System-wide performance analysis.

```python
from torch.adaptive_flow.flow_monitor import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(flow_collector, link_tracker, bottleneck_detector)

# Compute fairness
fairness = analyzer.compute_jains_fairness_index()

# Analyze performance
analysis = analyzer.analyze_performance()

# Export metrics
json_export = analyzer.export_metrics(format='json')
csv_export = analyzer.export_metrics(format='csv')
```

## Policy API

### Policy Classes

```python
from torch.adaptive_flow.policy_engine import (
    LatencyPolicy,
    ThroughputPolicy,
    FairnessPolicy,
    EnergyPolicy,
    AdaptivePolicy,
    PolicyEngine,
    PolicyContext
)

# Create policies
latency_policy = LatencyPolicy(target_latency=0.001, tolerance=0.1)
throughput_policy = ThroughputPolicy(target_utilization=0.9)
fairness_policy = FairnessPolicy(fairness_threshold=0.7)
energy_policy = EnergyPolicy(power_cap=100.0)
adaptive_policy = AdaptivePolicy(flow_collector, link_tracker, bottleneck_detector)

# Create policy context
context = PolicyContext(
    flow_id='flow1',
    current_rate=1e9,
    latency=0.001,
    loss_rate=0.0,
    queue_depth=5,
    link_utilization=0.7,
    competing_flows=3,
    bottleneck_detected=False
)

# Evaluate policy
action = policy.evaluate(context)
# Returns PolicyAction with:
# - decision (INCREASE_RATE, DECREASE_RATE, MAINTAIN_RATE, etc.)
# - parameters (new_rate, etc.)
# - priority
# - reason
# - confidence

# Learn from feedback
outcome = {'success': True, 'latency_improved': True}
policy.learn_from_feedback(action, outcome)
```

### PolicyEngine

Manages multiple policies.

```python
engine = PolicyEngine(flow_collector, link_tracker, bottleneck_detector)

# Set active policy
engine.set_active_policy('latency')

# Set policy chain
engine.set_policy_chain(['latency', 'fairness'])

# Make decision
action = engine.make_decision(context)

# Provide feedback
engine.provide_feedback(action, outcome)

# Get statistics
stats = engine.get_policy_stats()
```

## Visualization API

### Dashboard

```python
from torch.adaptive_flow.visualization import start_dashboard, stop_dashboard

# Start dashboard
dashboard = start_dashboard(
    flow_collector,
    link_tracker,
    performance_analyzer,
    port=8080
)

# Dashboard available at http://localhost:8080

# Stop dashboard
stop_dashboard()
```

### Trace Export

```python
from torch.adaptive_flow.visualization import export_chrome_trace, export_tensorboard

# Chrome trace (open in chrome://tracing)
export_chrome_trace(flow_collector, link_tracker, 'trace.json')

# TensorBoard logs
export_tensorboard(flow_collector, link_tracker, 'runs/adaptive_flow')
# Then run: tensorboard --logdir=runs
```

### TraceExporter

```python
from torch.adaptive_flow.visualization import TraceExporter

exporter = TraceExporter(flow_collector, link_tracker)

# Record events
exporter.record_transfer_event('flow1', start_time, end_time, bytes_transferred)
exporter.record_congestion_event('link1', timestamp, utilization)
exporter.record_policy_decision('flow1', timestamp, 'DECREASE_RATE', params)

# Export
exporter.export_chrome_trace('trace.json')
exporter.export_tensorboard('runs/logs')
exporter.export_csv('trace.csv')
exporter.export_json('trace_full.json')
exporter.export_all('output_dir', prefix='trace')

# Clear events
exporter.clear()
```

## Constants and Enums

```python
from torch.adaptive_flow.policy_engine import PolicyObjective, PolicyDecision
from torch.adaptive_flow.advanced_congestion import CongestionState
from torch.adaptive_flow.config import PolicyType, CongestionControlAlgorithm

# Policy objectives
PolicyObjective.LATENCY
PolicyObjective.THROUGHPUT
PolicyObjective.FAIRNESS
PolicyObjective.ENERGY
PolicyObjective.BALANCED

# Policy decisions
PolicyDecision.INCREASE_RATE
PolicyDecision.DECREASE_RATE
PolicyDecision.MAINTAIN_RATE
PolicyDecision.REROUTE
PolicyDecision.PRIORITIZE
PolicyDecision.DEPRIORITIZE
PolicyDecision.SPLIT_FLOW

# Congestion states
CongestionState.STARTUP
CongestionState.DRAIN
CongestionState.PROBE_BW
CongestionState.PROBE_RTT
CongestionState.STEADY
CongestionState.CONGESTION_AVOIDANCE
CongestionState.FAST_RECOVERY
```

## Error Handling

All functions raise standard Python exceptions:

- `ValueError`: Invalid parameter values
- `FileNotFoundError`: Configuration file not found
- `RuntimeError`: System state errors

Errors are logged using Python's logging framework.

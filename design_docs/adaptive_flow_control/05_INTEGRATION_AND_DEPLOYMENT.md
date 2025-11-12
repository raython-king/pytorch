# Adaptive Flow Control - Integration and Deployment Plan

## Integration with Existing Runtime Scheduler

### Architecture Integration

```
┌──────────────────────────────────────────────────────────────┐
│                 PyTorch User Application                      │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────┐
│              PyTorch Runtime Scheduler Hooks                  │
│         (torch/runtime_scheduler/integration/pytorch_hooks.py)│
└────────────────────────┬─────────────────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │                                 │
┌────────▼──────────┐       ┌─────────────▼──────────┐
│ Adaptive Flow     │       │ Existing Runtime       │
│ Control (NEW)     │◄─────►│ Scheduler Components   │
│                   │       │                        │
│ - FlowController  │       │ - TransferOptimizer    │
│ - BandwidthMonitor│       │ - StreamManager        │
│ - CongestionDetect│       │ - WorkloadScheduler    │
│ - TrafficShaper   │       │                        │
│ - QoSManager      │       │                        │
└───────────────────┘       └────────────────────────┘
         │                                 │
         └────────────────┬────────────────┘
                          │
         ┌────────────────▼───────────────┐
         │  CUDA / Network Transport Layer │
         └────────────────────────────────┘
```

### Hook Points

**1. Pre-Transfer Hook (Before Transfer Optimizer)**

```python
# In: torch/runtime_scheduler/integration/pytorch_hooks.py

def pre_transfer_hook(tensor, device, **kwargs):
    """
    Called before tensor.to(device) transfer.
    
    Performs:
    - Flow registration
    - Admission control
    - QoS classification
    - Congestion check
    """
    if not is_adaptive_flow_enabled():
        return None  # Pass through
    
    flow_controller = get_global_flow_controller()
    
    # Extract transfer information
    src_device = tensor.device
    dst_device = torch.device(device)
    size_gb = tensor.element_size() * tensor.nelement() / (1024**3)
    
    # Register flow
    flow_id = f"transfer_{uuid.uuid4()}"
    registration = flow_controller.register_flow(
        flow_id=flow_id,
        src_device=src_device,
        dst_device=dst_device,
        size_gb=size_gb,
        qos_class=get_current_qos_class(),  # From context
        qos_contract=get_current_qos_contract()
    )
    
    if not registration.admitted:
        # Handle rejection (log, fallback to default behavior)
        logger.warning(f"Flow admission rejected: {registration.reason}")
        return None
    
    # Store flow metadata for post-hook
    store_flow_metadata(flow_id, registration.flow_descriptor)
    
    return {
        'flow_id': flow_id,
        'path': registration.flow_descriptor.path,
        'priority': registration.flow_descriptor.priority
    }


def transfer_routing_hook(transfer_request, pre_hook_result):
    """
    Called during path selection in TransferOptimizer.
    
    Overrides default path selection with flow control decisions.
    """
    if not pre_hook_result:
        return None  # Use default routing
    
    flow_controller = get_global_flow_controller()
    flow_id = pre_hook_result['flow_id']
    
    # Schedule transfer
    schedule_result = flow_controller.schedule_transfer(flow_id)
    
    if not schedule_result.allowed:
        # Apply backpressure
        if schedule_result.wait_time_ms > 0:
            time.sleep(schedule_result.wait_time_ms / 1000.0)
            # Retry
            schedule_result = flow_controller.schedule_transfer(flow_id)
    
    return {
        'path': schedule_result.path,
        'stream_id': schedule_result.stream_id,
        'bandwidth': schedule_result.allocated_bandwidth_gbps
    }


def post_transfer_hook(flow_id, latency_ms, throughput_gbps, success):
    """
    Called after transfer completes.
    
    Records performance and updates ML models.
    """
    if not is_adaptive_flow_enabled():
        return
    
    flow_controller = get_global_flow_controller()
    flow_controller.complete_transfer(
        flow_id=flow_id,
        actual_latency_ms=latency_ms,
        actual_throughput_gbps=throughput_gbps,
        success=success
    )
```

**2. Integration with WorkloadScheduler**

```python
# Modify: torch/runtime_scheduler/workload_scheduler.py

class WorkloadScheduler:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add flow controller
        if config.enable_adaptive_flow:
            self.flow_controller = FlowController(config.flow_control_config)
        else:
            self.flow_controller = None
    
    def submit_operation(self, op_type, ...):
        # ... existing code ...
        
        # Register with flow controller if transfer operation
        if self.flow_controller and self._is_transfer_op(op_type):
            flow_info = self.flow_controller.register_flow(...)
            operation.flow_info = flow_info
        
        # ... rest of existing code ...
```

---

## Phased Rollout Strategy

### Phase 1: Monitoring Only (Weeks 1-2)

**Goals:**
- Collect baseline metrics
- Validate monitoring infrastructure
- No impact on existing behavior

**Activities:**
1. Deploy BandwidthMonitor in passive mode
2. Deploy CongestionDetector in monitoring mode
3. Collect training data for ML models
4. Validate overhead < 0.3%

**Success Criteria:**
- Monitor runs stable for 7 days
- Data collection complete
- Overhead verified < 0.3%

### Phase 2: Shadow Mode (Weeks 3-4)

**Goals:**
- Test decision-making without affecting traffic
- Compare decisions with actual outcomes
- Tune algorithms

**Activities:**
1. Enable FlowController in shadow mode
2. Make scheduling decisions but don't apply
3. Log what decisions would have been made
4. Compare with actual transfer performance

**Success Criteria:**
- Shadow decisions logged for all transfers
- Decision latency < 0.2ms
- ML models trained and validated

### Phase 3: Limited Rollout (Weeks 5-6)

**Goals:**
- Apply flow control to subset of traffic
- Validate benefits
- Monitor for issues

**Activities:**
1. Enable flow control for 10% of transfers (via sampling)
2. Monitor performance improvements
3. A/B test different policies
4. Gradually increase to 25%, 50%

**Success Criteria:**
- No performance regressions
- Measurable improvements in:
  - Tail latency reduction (> 20%)
  - Throughput increase (> 5%)
  - Fairness improvement (JFI > 0.95)

### Phase 4: Full Deployment (Weeks 7-8)

**Goals:**
- Enable for all traffic
- Continuous monitoring and tuning

**Activities:**
1. Enable flow control for 100% of traffic
2. Monitor system-wide metrics
3. Enable online learning
4. Fine-tune policies based on workload

**Success Criteria:**
- System stable at 100% rollout
- Performance targets met:
  - Aggregate throughput > 95% of peak
  - P99 latency < 1.5x baseline
  - Fairness index > 0.95
  - Overhead < 1%

---

## Configuration Management

### Default Configuration

```python
# torch/adaptive_flow/default_config.py

DEFAULT_FLOW_CONTROL_CONFIG = FlowControlConfig(
    # Enable/disable
    enabled=False,  # Default: disabled for backward compatibility
    mode='ml',  # 'disabled', 'heuristic', 'ml', 'hybrid'
    
    # Monitoring
    monitoring_interval_ms=100.0,
    enable_detailed_monitoring=False,
    
    # Congestion control
    congestion_threshold=0.8,
    congestion_detection_method='hybrid',  # 'explicit', 'implicit', 'ml', 'hybrid'
    enable_congestion_prediction=True,
    
    # Traffic shaping
    shaping_algorithm='token_bucket',  # 'token_bucket', 'leaky_bucket', 'htb'
    default_rate_limit_gbps=None,  # None = no limit
    enable_burst_control=True,
    
    # QoS
    qos_enabled=True,
    enable_admission_control=True,
    enable_priority_enforcement=True,
    enable_preemption=False,  # Conservative default
    
    # Scheduling
    scheduling_policy='ml_guided',  # 'fifo', 'sjf', 'wfq', 'edf', 'ml_guided'
    fairness_weight=0.5,
    latency_weight=0.3,
    throughput_weight=0.2,
    
    # Routing
    routing_policy='load_aware',  # 'shortest_path', 'load_aware', 'ml_optimized'
    enable_multipath=True,
    enable_rerouting=True,
    
    # ML
    use_ml_prediction=True,
    enable_online_learning=True,
    model_update_interval=1000,  # operations
    ml_inference_timeout_ms=0.2,
    
    # Performance
    max_overhead_percent=1.0,
    enable_batched_inference=True,
    enable_model_quantization=True,
    
    # Debugging
    shadow_mode=False,
    log_decisions=False,
    enable_profiling=False,
)
```

### Environment Variables

```python
# Configuration via environment variables

TORCH_ADAPTIVE_FLOW_ENABLED=1  # Enable/disable
TORCH_ADAPTIVE_FLOW_MODE=ml  # Mode
TORCH_ADAPTIVE_FLOW_MONITORING_INTERVAL_MS=100
TORCH_ADAPTIVE_FLOW_CONGESTION_THRESHOLD=0.8
TORCH_ADAPTIVE_FLOW_QOS_ENABLED=1
TORCH_ADAPTIVE_FLOW_SCHEDULING_POLICY=ml_guided
TORCH_ADAPTIVE_FLOW_ML_ENABLED=1
TORCH_ADAPTIVE_FLOW_SHADOW_MODE=0
TORCH_ADAPTIVE_FLOW_CONFIG_FILE=/path/to/config.json
```

---

## Testing Strategy

### Unit Tests

```python
# test/test_adaptive_flow_control.py

class TestFlowController(TestCase):
    def test_flow_registration(self):
        """Test flow registration and admission control."""
        controller = FlowController(test_config)
        
        result = controller.register_flow(
            flow_id="test_001",
            src_device=torch.device("cuda:0"),
            dst_device=torch.device("cuda:1"),
            size_gb=10.0
        )
        
        self.assertTrue(result.admitted)
        self.assertIsNotNone(result.flow_descriptor)
    
    def test_qos_admission_control(self):
        """Test QoS-based admission control."""
        controller = FlowController(test_config)
        
        # Fill capacity with high-priority flows
        for i in range(10):
            controller.register_flow(
                flow_id=f"high_pri_{i}",
                src_device=torch.device("cuda:0"),
                dst_device=torch.device("cuda:1"),
                size_gb=10.0,
                qos_class=QoSClass.GUARANTEED,
                qos_contract=QoSContract(min_bandwidth_gbps=10.0)
            )
        
        # Try to register another guaranteed flow
        result = controller.register_flow(
            flow_id="test_overflow",
            src_device=torch.device("cuda:0"),
            dst_device=torch.device("cuda:1"),
            size_gb=10.0,
            qos_class=QoSClass.GUARANTEED,
            qos_contract=QoSContract(min_bandwidth_gbps=10.0)
        )
        
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "insufficient_bandwidth")
    
    def test_traffic_shaping(self):
        """Test traffic shaping rate limits."""
        config = FlowControlConfig(
            shaping_algorithm='token_bucket',
            default_rate_limit_gbps=10.0
        )
        controller = FlowController(config)
        
        # Register and schedule transfers
        # Verify rate limiting is applied
        pass
```

### Integration Tests

```python
class TestFlowControlIntegration(TestCase):
    def test_tensor_transfer_with_flow_control(self):
        """Test actual tensor transfer with flow control."""
        from torch.adaptive_flow import enable_adaptive_flow
        
        # Enable flow control
        controller = enable_adaptive_flow(
            mode='heuristic',
            policy='max_throughput'
        )
        
        # Create tensors
        tensor = torch.randn(1000, 1000, device='cuda:0')
        
        # Transfer with QoS
        with controller.qos_context(
            qos_class=QoSClass.HIGH,
            min_bandwidth_gbps=5.0
        ):
            tensor_gpu1 = tensor.to('cuda:1')
        
        # Verify transfer succeeded
        self.assertTrue(torch.allclose(tensor.cpu(), tensor_gpu1.cpu()))
        
        # Check statistics
        stats = controller.get_statistics()
        self.assertEqual(stats.completed_flows, 1)
```

### Performance Tests

```python
class TestFlowControlPerformance(TestCase):
    def test_overhead(self):
        """Verify overhead is < 1%."""
        # Baseline: transfer without flow control
        baseline_time = measure_transfer_time(enable_flow_control=False)
        
        # With flow control
        with_fc_time = measure_transfer_time(enable_flow_control=True)
        
        overhead = (with_fc_time - baseline_time) / baseline_time
        
        self.assertLess(overhead, 0.01, f"Overhead {overhead*100:.2f}% > 1%")
    
    def test_ml_inference_latency(self):
        """Verify ML inference is < 0.2ms."""
        predictor = LatencyPredictor()
        features = torch.randn(64)
        
        # Warm up
        for _ in range(100):
            predictor(features)
        
        # Measure
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            predictor(features)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        p99_latency = np.percentile(times, 99)
        self.assertLess(p99_latency, 0.2, f"P99 latency {p99_latency:.3f}ms > 0.2ms")
```

---

## Monitoring and Observability

### Metrics to Track

```python
# Exported metrics (Prometheus format)

METRICS = {
    # Throughput
    'adaptive_flow_throughput_gbps': Gauge(
        'Current aggregate throughput in GB/s'
    ),
    'adaptive_flow_total_bytes': Counter(
        'Total bytes transferred'
    ),
    
    # Latency
    'adaptive_flow_latency_ms': Histogram(
        'Transfer latency in milliseconds',
        buckets=[1, 5, 10, 50, 100, 500, 1000]
    ),
    
    # Congestion
    'adaptive_flow_congestion_events': Counter(
        'Number of congestion events detected'
    ),
    'adaptive_flow_utilization': Gauge(
        'Link utilization (0-1)'
    ),
    
    # QoS
    'adaptive_flow_sla_violations': Counter(
        'Number of SLA violations'
    ),
    'adaptive_flow_admission_rejections': Counter(
        'Number of flows rejected by admission control'
    ),
    
    # Fairness
    'adaptive_flow_fairness_index': Gauge(
        "Jain's fairness index (0-1)"
    ),
    
    # Overhead
    'adaptive_flow_decision_latency_us': Histogram(
        'Time to make scheduling decision in microseconds',
        buckets=[10, 50, 100, 200, 500, 1000]
    ),
}
```

### Logging

```python
# Structured logging

logger = logging.getLogger('torch.adaptive_flow')

# Log levels
logger.debug("Flow registered: flow_id={}, size={} GB", flow_id, size_gb)
logger.info("Congestion detected on link: link_id={}, util={:.2f}", link_id, util)
logger.warning("QoS SLA violation: flow_id={}, latency={:.2f}ms > {:.2f}ms",
               flow_id, actual, sla)
logger.error("Flow admission failed: flow_id={}, reason={}", flow_id, reason)
```

---

## Summary

This integration plan provides:

1. **Clear integration points** with existing runtime scheduler
2. **Phased rollout** to minimize risk
3. **Comprehensive testing** at unit, integration, and performance levels
4. **Monitoring and observability** for production deployment
5. **Configuration management** for flexibility

Next steps:
1. Implement core components
2. Add hooks to runtime scheduler
3. Execute phased rollout
4. Monitor and iterate

See design documents for detailed component specifications.

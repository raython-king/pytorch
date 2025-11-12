# Adaptive Flow Control - Quick Reference Guide

## Document Map

| Document | Size | Purpose | Audience |
|----------|------|---------|----------|
| [00_EXECUTIVE_SUMMARY.md](00_EXECUTIVE_SUMMARY.md) | 10KB | High-level overview | Everyone |
| [01_ARCHITECTURE_OVERVIEW.md](01_ARCHITECTURE_OVERVIEW.md) | 18KB | System architecture | Engineers, Architects |
| [02_ALGORITHMS.md](02_ALGORITHMS.md) | 45KB | Detailed algorithms | Algorithm engineers |
| [03_ML_MODELS.md](03_ML_MODELS.md) | 45KB | ML model specs | ML engineers |
| [04_COMPONENT_INTERFACES.md](04_COMPONENT_INTERFACES.md) | 18KB | API specifications | Implementation engineers |
| [05_INTEGRATION_AND_DEPLOYMENT.md](05_INTEGRATION_AND_DEPLOYMENT.md) | 17KB | Integration plan | DevOps, SRE |
| [06_VISUAL_DIAGRAMS.md](06_VISUAL_DIAGRAMS.md) | 37KB | Visual flows | Everyone |
| [README.md](README.md) | 17KB | Navigation guide | Everyone |

**Total:** 207KB of design documentation

## Key Components at a Glance

```
FlowController          - Central manager
├─ BandwidthMonitor    - Real-time bandwidth tracking
├─ CongestionDetector  - Congestion identification
├─ TrafficShaper       - Rate limiting & queuing
├─ AdaptiveScheduler   - Flow scheduling
├─ QoSManager          - QoS enforcement
└─ PathRouter          - Path selection
```

## Performance Targets

| Metric | Target | Current Baseline |
|--------|--------|------------------|
| Aggregate Throughput | > 95% of peak | ~85% |
| P99 Latency | < 1.5x baseline | 1.0x (baseline) |
| Fairness Index | > 0.95 | ~0.7 |
| System Overhead | < 1% | 0% (none) |
| QoS SLA Compliance | > 99.9% | N/A (no QoS) |

## Implementation Checklist

### Phase 1: Monitoring (Weeks 1-2)
- [ ] Implement BandwidthMonitor
- [ ] Implement CongestionDetector (monitoring mode)
- [ ] Add metrics collection hooks
- [ ] Validate < 0.3% overhead
- [ ] Collect training data

### Phase 2: Shadow Mode (Weeks 3-4)
- [ ] Implement FlowController (shadow mode)
- [ ] Implement TrafficShaper (decisions only)
- [ ] Train initial ML models
- [ ] Validate decision latency < 0.2ms
- [ ] Compare shadow decisions with actual

### Phase 3: Limited Rollout (Weeks 5-6)
- [ ] Enable for 10% of traffic
- [ ] Monitor for regressions
- [ ] A/B test policies
- [ ] Increase to 25%, then 50%
- [ ] Tune parameters

### Phase 4: Full Deployment (Weeks 7-8)
- [ ] Enable for 100% of traffic
- [ ] Enable online learning
- [ ] Continuous monitoring
- [ ] Measure against targets
- [ ] Document results

## API Quick Start

### Basic Usage
```python
from torch.adaptive_flow import enable_adaptive_flow

# Enable flow control
controller = enable_adaptive_flow(
    mode='ml',
    policy='max_throughput'
)

# Your code - flow control is automatic
tensor = tensor.to('cuda:1')

# Check stats
print(controller.get_statistics())
```

### With QoS
```python
# Transfer with guarantees
with controller.qos_context(
    qos_class=QoSClass.GUARANTEED,
    min_bandwidth_gbps=10.0,
    max_latency_ms=50.0
):
    tensor = tensor.to('cuda:1')
```

### Configuration
```python
# Custom configuration
config = FlowControlConfig(
    mode='ml',
    congestion_threshold=0.85,
    scheduling_policy='ml_guided',
    enable_online_learning=True
)
controller = FlowController(config)
controller.enable()
```

## Algorithm Cheat Sheet

### Congestion Control
- **AIMD:** Additive increase, multiplicative decrease
- **ECN:** Explicit congestion notification
- **ML:** Learned optimal rates

### Bandwidth Allocation
- **Max-Min Fair:** Maximize minimum allocation
- **Weighted Fair:** Priority-based allocation
- **Hierarchical:** Multi-level allocation

### Scheduling
- **SJF:** Shortest job first
- **WFQ:** Weighted fair queuing
- **EDF:** Earliest deadline first
- **ML-Guided:** Learned priorities

### Traffic Shaping
- **Token Bucket:** Allow bursts
- **Leaky Bucket:** Smooth traffic
- **HTB:** Hierarchical token bucket

## ML Models Reference

| Model | Architecture | Purpose | Latency | Size |
|-------|-------------|---------|---------|------|
| Bandwidth Predictor | LSTM + Attention | Predict bandwidth | 0.15ms | 2MB |
| Congestion Predictor | XGBoost | Predict congestion | 0.05ms | 1MB |
| Routing Optimizer | GNN | Optimize paths | 0.18ms | 3MB |
| Latency Predictor | MLP + Quantile | Predict latency | 0.08ms | 2MB |
| Flow Size Estimator | Ensemble | Estimate size | 0.10ms | 1.5MB |
| Priority Predictor | MLP | Assign priority | 0.06ms | 0.5MB |
| **Total** | | | **< 0.2ms** | **< 10MB** |

## Configuration Options

### Mode
- `disabled`: Flow control off
- `heuristic`: Rule-based control
- `ml`: ML-guided control
- `hybrid`: Mix of heuristic + ML

### Policy
- `max_throughput`: Maximize aggregate throughput
- `min_latency`: Minimize average latency
- `fair`: Maximize fairness
- `balanced`: Balance all objectives

### QoS Classes
- `GUARANTEED`: Hard guarantees
- `PREMIUM`: Soft guarantees
- `BEST_EFFORT`: No guarantees
- `SCAVENGER`: Use spare capacity

## Monitoring Metrics

### Throughput
- `adaptive_flow_throughput_gbps` - Current aggregate throughput
- `adaptive_flow_total_bytes` - Total bytes transferred

### Latency
- `adaptive_flow_latency_ms` - Transfer latency histogram
- P50, P90, P99 percentiles

### Congestion
- `adaptive_flow_congestion_events` - Congestion event count
- `adaptive_flow_utilization` - Link utilization

### QoS
- `adaptive_flow_sla_violations` - SLA violation count
- `adaptive_flow_admission_rejections` - Rejected flows

### Overhead
- `adaptive_flow_decision_latency_us` - Decision time

## Testing Commands

```bash
# Unit tests
python -m pytest test/test_adaptive_flow_control.py

# Integration tests
python -m pytest test/test_flow_control_integration.py

# Performance tests
python test/benchmark_adaptive_flow.py

# Stress test
python test/stress_test_flow_control.py --duration=3600
```

## Troubleshooting

### High Overhead (> 1%)
- Check monitoring interval (increase if needed)
- Disable detailed monitoring
- Enable model quantization
- Reduce ML inference frequency

### Low Throughput (< 90%)
- Check congestion threshold (may be too low)
- Verify bandwidth allocation fairness
- Check for traffic shaping bottlenecks
- Review path selection decisions

### SLA Violations
- Increase QoS reservation for guaranteed flows
- Enable preemption for critical flows
- Check admission control settings
- Verify latency predictions accuracy

### Fairness Issues (JFI < 0.95)
- Switch to max-min fair allocation
- Increase fairness weight in policy
- Check for priority imbalance
- Disable preemption

## Common Patterns

### Pattern 1: High-Priority Transfer
```python
# Ensure high-priority transfer gets resources
with controller.qos_context(qos_class=QoSClass.PREMIUM):
    critical_data = critical_data.to('cuda:1')
```

### Pattern 2: Bulk Background Transfer
```python
# Mark as low-priority to not impact training
with controller.qos_context(qos_class=QoSClass.SCAVENGER):
    checkpoint_data = checkpoint_data.to('cpu')
```

### Pattern 3: Latency-Sensitive Transfer
```python
# Guarantee low latency for interactive workload
with controller.qos_context(
    qos_class=QoSClass.GUARANTEED,
    max_latency_ms=10.0
):
    result = model(input_data.to('cuda:1'))
```

### Pattern 4: Bandwidth-Intensive Transfer
```python
# Reserve bandwidth for large transfer
with controller.qos_context(
    qos_class=QoSClass.GUARANTEED,
    min_bandwidth_gbps=50.0
):
    large_tensor = large_tensor.to('cuda:1')
```

## Performance Optimization Tips

1. **Enable batched inference** for ML models
2. **Use quantized models** in production
3. **Tune monitoring interval** based on workload
4. **Enable online learning** for adaptation
5. **Use hierarchical token buckets** for multi-level control
6. **Enable multi-path routing** for large transfers
7. **Configure appropriate QoS classes** for different workloads

## Links to Detailed Sections

- **Architecture:** [01_ARCHITECTURE_OVERVIEW.md § System Architecture](01_ARCHITECTURE_OVERVIEW.md#system-architecture)
- **Congestion Control:** [02_ALGORITHMS.md § Congestion Control](02_ALGORITHMS.md#congestion-control)
- **ML Models:** [03_ML_MODELS.md § Model Architectures](03_ML_MODELS.md#model-architectures)
- **APIs:** [04_COMPONENT_INTERFACES.md § FlowController](04_COMPONENT_INTERFACES.md#1-flowcontroller)
- **Integration:** [05_INTEGRATION_AND_DEPLOYMENT.md § Hook Points](05_INTEGRATION_AND_DEPLOYMENT.md#hook-points)
- **Visuals:** [06_VISUAL_DIAGRAMS.md § Complete Transfer Flow](06_VISUAL_DIAGRAMS.md#1-complete-transfer-flow)

## Contact & Feedback

**Design Team:** [Your team/email]  
**Issue Tracker:** [GitHub Issues]  
**Discussion:** [Forum/Slack]  
**Documentation:** This directory

---

**Last Updated:** 2025-11-12  
**Version:** 1.0  
**Status:** Design Phase

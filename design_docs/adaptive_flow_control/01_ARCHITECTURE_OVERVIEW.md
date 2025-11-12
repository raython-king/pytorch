# PyTorch Adaptive Flow Control System - Architecture Overview

## Executive Summary

The Adaptive Flow Control System provides intelligent, ML-guided traffic management for
PyTorch data transfers between devices, maximizing throughput while minimizing latency
and ensuring Quality of Service (QoS) guarantees with < 1% performance overhead.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Design Principles](#design-principles)
4. [Integration Strategy](#integration-strategy)
5. [Performance Objectives](#performance-objectives)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PyTorch Application Layer                         │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────────────┐
│                    Adaptive Flow Control System                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      FlowController (Central Manager)               │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │ │
│  │  │ QoS Manager │  │ Flow Router  │  │  Policy Engine (ML)    │  │ │
│  │  └─────────────┘  └──────────────┘  └─────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Bandwidth   │  │ Congestion   │  │   Traffic    │  │  Adaptive  │ │
│  │   Monitor    │  │  Detector    │  │   Shaper     │  │ Scheduler  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                  │                  │                 │        │
│  ┌──────┴──────────────────┴──────────────────┴─────────────────┴─────┐ │
│  │                   ML Intelligence Layer                             │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────┐ │ │
│  │  │ Bandwidth  │  │ Congestion │  │  Routing   │  │  Latency    │ │ │
│  │  │ Predictor  │  │ Predictor  │  │ Optimizer  │  │ Predictor   │ │ │
│  │  └────────────┘  └────────────┘  └────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────────────┐
│                  Existing Runtime Scheduler Infrastructure               │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────────────┐ │
│  │   Transfer     │  │     Stream     │  │    Workload Scheduler     │ │
│  │   Optimizer    │  │    Manager     │  │   (ML-Guided Priority)    │ │
│  └────────────────┘  └────────────────┘  └───────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────────────┐
│                     Hardware Transport Layer                             │
│     PCIe   │   NVLink   │   Network (IB/RoCE)   │   Fabric (NVSwitch)  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. FlowController (Central Manager)

**Responsibilities:**
- Coordinate all flow control operations
- Manage global flow policies
- Enforce QoS guarantees
- Handle flow admission control
- Coordinate with existing runtime scheduler

**Key Features:**
- Multi-tenant flow isolation
- Priority-based flow classification
- Dynamic policy adaptation
- Resource reservation system

**State Management:**
```
FlowController State:
├── Active Flows: Map[FlowID -> FlowDescriptor]
├── Flow Policies: Map[FlowClass -> Policy]
├── QoS Guarantees: Map[FlowID -> QoSContract]
├── Resource Allocations: Map[Device -> BandwidthAllocation]
└── Performance Metrics: RealTimeMetrics
```

### 2. BandwidthMonitor

**Responsibilities:**
- Real-time bandwidth measurement
- Per-link utilization tracking
- Historical bandwidth analysis
- Bandwidth prediction (ML-based)

**Monitoring Granularity:**
- Link-level: Per physical connection
- Flow-level: Per data transfer flow
- Device-level: Aggregate per device
- System-level: Global throughput

**Metrics Collected:**
```
BandwidthMetrics:
├── Instantaneous: Current transfer rate (GB/s)
├── Moving Average: Short-term (100ms), Long-term (1s, 10s)
├── Peak: Maximum observed bandwidth
├── Utilization: Percentage of theoretical peak
├── Variance: Statistical variance over time
└── Trends: Increasing/decreasing/stable
```

### 3. CongestionDetector

**Responsibilities:**
- Identify congestion events
- Predict congestion before occurrence
- Classify congestion severity
- Trigger congestion avoidance

**Detection Methods:**

1. **Explicit Indicators:**
   - Queue depth monitoring
   - Buffer occupancy
   - Packet/transfer loss
   - Timeout events

2. **Implicit Indicators:**
   - Latency inflation (RTT increase)
   - Throughput degradation
   - Bandwidth variance
   - Scheduler queue length

3. **ML-Based Prediction:**
   - Pattern recognition
   - Time-series forecasting
   - Anomaly detection

**Congestion Levels:**
```
CongestionLevel:
├── NONE:     < 60% utilization, normal operation
├── LOW:      60-75% utilization, proactive management
├── MODERATE: 75-85% utilization, active control
├── HIGH:     85-95% utilization, aggressive mitigation
└── CRITICAL: > 95% utilization, emergency measures
```

### 4. TrafficShaper

**Responsibilities:**
- Rate limiting and pacing
- Flow scheduling and prioritization
- Burst control
- Fair bandwidth allocation

**Shaping Algorithms:**

1. **Token Bucket:**
   - Allows burst traffic up to bucket size
   - Refill rate controls average bandwidth
   - Per-flow or per-class buckets

2. **Leaky Bucket:**
   - Smooth traffic at constant rate
   - No bursts allowed
   - Good for latency-sensitive flows

3. **Hierarchical Shaping:**
   - Multiple levels: System -> Device -> Flow
   - Nested token buckets
   - Fair sharing with max-min fairness

**Priority Queues:**
```
Priority Queue Structure:
├── CRITICAL (P0): System-critical, latency-sensitive
├── HIGH (P1):     User-facing, interactive
├── NORMAL (P2):   Background compute, ML training
└── LOW (P3):      Batch transfers, checkpointing
```

### 5. AdaptiveScheduler

**Responsibilities:**
- Dynamic flow scheduling
- Load-aware path selection
- Deadline-aware scheduling
- ML-guided optimization

**Scheduling Policies:**

1. **Shortest Job First (SJF):**
   - Minimize average completion time
   - Prevent small transfer starvation
   - Dynamic size estimation

2. **Weighted Fair Queuing (WFQ):**
   - Fair bandwidth sharing
   - Weight-based allocation
   - Work-conserving

3. **Earliest Deadline First (EDF):**
   - Deadline-aware scheduling
   - Critical for latency-sensitive flows
   - Admission control

4. **ML-Guided Priority:**
   - Learn from historical patterns
   - Predict optimal scheduling
   - Adapt to workload characteristics

### 6. QoSManager

**Responsibilities:**
- QoS policy enforcement
- SLA monitoring and enforcement
- Resource reservation
- Admission control

**QoS Classes:**
```
QoS Classes:
├── GUARANTEED: Hard bandwidth/latency guarantees
├── PREMIUM:    Soft guarantees, best effort with priority
├── BEST_EFFORT: No guarantees, opportunistic
└── SCAVENGER:  Use spare capacity only
```

**QoS Contracts:**
```python
@dataclass
class QoSContract:
    flow_id: str
    qos_class: QoSClass
    
    # Bandwidth guarantees
    min_bandwidth_gbps: float  # Minimum guaranteed
    max_bandwidth_gbps: float  # Maximum allowed
    avg_bandwidth_gbps: float  # Target average
    
    # Latency guarantees
    max_latency_ms: float      # P99 latency SLA
    avg_latency_ms: float      # Target average
    
    # Priority
    priority: int              # 0 (highest) - 10 (lowest)
    preemption_allowed: bool   # Can preempt lower priority
    
    # Duration
    start_time: float
    duration_sec: float
    deadline: Optional[float]  # Hard deadline
```

---

## Design Principles

### 1. Low Overhead (< 1%)

**Strategies:**
- Fast-path optimization for common cases
- Amortize overhead across batches
- Lazy evaluation and caching
- Lock-free data structures where possible
- Sampling-based monitoring (not per-operation)

**Overhead Budget:**
```
Total Overhead Budget: 1% of transfer time
├── Monitoring:      0.3% (bandwidth, congestion)
├── ML Inference:    0.2% (predictions, decisions)
├── Scheduling:      0.2% (flow scheduling, routing)
├── Traffic Shaping: 0.2% (rate limiting, queueing)
└── Bookkeeping:     0.1% (stats, logging)
```

### 2. Adaptive to Dynamic Conditions

**Adaptation Mechanisms:**
- Online learning from observations
- Feedback-based control loops
- Periodic policy re-evaluation
- Automatic parameter tuning

**Adaptation Triggers:**
- Congestion detection
- Workload changes
- Performance degradation
- SLA violations

### 3. Fair and Efficient

**Fairness Criteria:**
- Max-min fairness for equal priority
- Weighted fairness for different priorities
- Starvation prevention (aging)
- Isolation between tenants/users

**Efficiency Metrics:**
- Maximize aggregate throughput
- Minimize average latency
- Minimize wasted bandwidth
- High link utilization (> 90%)

### 4. Backward Compatible

**Compatibility Guarantees:**
- Default behavior unchanged when disabled
- Gradual rollout with feature flags
- A/B testing support
- Fallback to traditional scheduling

### 5. Modular and Extensible

**Extension Points:**
- Pluggable scheduling policies
- Custom congestion detectors
- User-defined QoS classes
- Custom ML models

---

## Integration Strategy

### Integration with Existing Runtime Scheduler

**Layered Architecture:**
```
User Code
    ↓
PyTorch API (tensor.to(), distributed ops)
    ↓
Runtime Scheduler Hooks (pytorch_hooks.py)
    ↓
┌─────────────────────────────────────────┐
│   Adaptive Flow Control (NEW)          │
│   - Flow admission control              │
│   - QoS policy enforcement              │
│   - Congestion detection                │
│   - Traffic shaping                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│   Existing Runtime Scheduler            │
│   - Transfer Optimizer                  │
│   - Stream Manager                      │
│   - Workload Scheduler                  │
└─────────────────────────────────────────┘
    ↓
CUDA/Network Transport
```

**Hook Points:**

1. **Pre-Transfer Hook:**
   - Flow admission control
   - QoS classification
   - Congestion check
   - Rate limiting decision

2. **Transfer Routing Hook:**
   - Path selection (PCIe vs NVLink vs Network)
   - Load balancing
   - Congestion avoidance

3. **Post-Transfer Hook:**
   - Performance measurement
   - ML model update
   - Congestion state update

### API Design

**User-Facing API:**
```python
from torch.adaptive_flow import (
    enable_adaptive_flow,
    FlowController,
    QoSClass,
    FlowPolicy
)

# Enable with one line
flow_controller = enable_adaptive_flow(
    mode='ml',              # 'disabled', 'heuristic', 'ml'
    policy='max_throughput',  # 'max_throughput', 'min_latency', 'fair'
    monitoring=True,
    qos_enabled=True
)

# Optional: Configure QoS for specific transfers
with flow_controller.qos_context(
    qos_class=QoSClass.GUARANTEED,
    min_bandwidth_gbps=10.0,
    max_latency_ms=5.0
):
    # Transfers in this context get QoS guarantees
    data = data.to('cuda:1')
    
# Optional: Set custom policy
flow_controller.set_policy(
    FlowPolicy(
        congestion_threshold=0.8,
        fairness_weight=0.5,
        latency_weight=0.3,
        throughput_weight=0.2
    )
)

# Monitor performance
stats = flow_controller.get_statistics()
print(f"Throughput: {stats['aggregate_throughput_gbps']:.2f} GB/s")
print(f"P99 Latency: {stats['p99_latency_ms']:.2f} ms")
print(f"Fairness Index: {stats['jain_fairness_index']:.3f}")
```

---

## Performance Objectives

### Throughput Objectives

```
Target Metrics:
├── Aggregate Throughput: > 95% of hardware peak
├── Per-Flow Throughput:  > 90% of fair share
├── Small Transfer Efficiency: > 80% of peak
└── Large Transfer Efficiency: > 98% of peak
```

### Latency Objectives

```
Latency Targets:
├── P50 Latency: < 1.2x baseline (unmanaged)
├── P99 Latency: < 1.5x baseline
├── P999 Latency: < 2.0x baseline
└── Tail Latency Reduction: -50% compared to unmanaged
```

### Fairness Objectives

```
Fairness Metrics:
├── Jain's Fairness Index: > 0.95
├── Max-Min Fairness Deviation: < 10%
├── Starvation Prevention: 100% (no flow starved > 100ms)
└── QoS SLA Compliance: > 99.9%
```

### Overhead Objectives

```
Overhead Constraints:
├── Total Overhead: < 1% of transfer time
├── Monitoring Overhead: < 0.3%
├── ML Inference Overhead: < 0.2%
├── Scheduling Overhead: < 0.2%
└── Memory Overhead: < 50 MB per device
```

### Adaptation Speed

```
Adaptation Targets:
├── Congestion Detection Latency: < 10ms
├── Policy Adaptation Time: < 100ms
├── ML Model Update Interval: 1-10 seconds
└── Feedback Loop Delay: < 50ms
```

---

## System Assumptions

1. **Hardware:**
   - Multi-GPU system with PCIe, NVLink, or network interconnects
   - CUDA-capable devices (compute capability >= 7.0)
   - Support for CUDA streams and events

2. **Workload:**
   - Mixed workload: compute, communication, memory transfers
   - Variable transfer sizes: 1KB - 10GB
   - Variable transfer patterns: point-to-point, broadcast, all-reduce

3. **Environment:**
   - Shared infrastructure (multi-tenant possible)
   - Dynamic workload (arrival/departure of flows)
   - Network conditions may vary

---

## Next Steps

1. **Component Design:** Detailed design of each component
2. **Algorithms:** Flow control, congestion control, scheduling algorithms
3. **ML Models:** Architecture and training procedures
4. **Integration Plan:** Step-by-step integration with existing scheduler
5. **Performance Analysis:** Theoretical analysis and simulations
6. **Implementation Plan:** Phased rollout strategy

See subsequent design documents for details.

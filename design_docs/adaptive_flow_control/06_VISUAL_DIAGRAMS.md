# Adaptive Flow Control - Visual Diagrams and Data Flows

## Data Flow Diagrams

### 1. Complete Transfer Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Application                              │
│                                                                       │
│   tensor = torch.randn(1000, 1000, device='cuda:0')                 │
│   tensor_gpu1 = tensor.to('cuda:1')  ◄─── User initiates transfer   │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  PyTorch Hook (tensor.to)                            │
│                                                                       │
│  1. Extract metadata: src=cuda:0, dst=cuda:1, size=4MB              │
│  2. Generate flow_id                                                 │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│              FlowController.register_flow()                          │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Step 1: QoS Classification                                │     │
│  │  - Check current QoS context                               │     │
│  │  - Default: BEST_EFFORT                                    │     │
│  │  Result: qos_class = BEST_EFFORT                           │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Step 2: Admission Control (QoSManager)                    │     │
│  │  - Check bandwidth availability                            │     │
│  │  - Predict latency                                         │     │
│  │  - Verify can meet guarantees                              │     │
│  │  Result: ADMITTED (priority=5)                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Step 3: Path Selection (PathRouter)                       │     │
│  │  - Identify available paths:                               │     │
│  │    * Direct NVLink: cuda:0 → cuda:1 (300 GB/s)            │     │
│  │    * Via CPU: cuda:0 → CPU → cuda:1 (16 GB/s)             │     │
│  │  - Check congestion on each path                           │     │
│  │  - ML-based scoring                                        │     │
│  │  Result: path = [NVLink_0_1], score=0.95                   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Step 4: Register with TrafficShaper                        │     │
│  │  - Add to token bucket for NVLink                          │     │
│  │  - Current rate: 250 GB/s (83% util)                       │     │
│  │  Result: flow_registered                                   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  Return: FlowDescriptor(flow_id, path, priority, est_latency)       │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│             FlowController.schedule_transfer()                       │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Step 1: Congestion Check                                  │     │
│  │  - Monitor NVLink utilization: 83%                         │     │
│  │  - Check queue depth: 3 pending transfers                  │     │
│  │  - ML prediction: congestion_prob = 0.15 (LOW)             │     │
│  │  Result: PROCEED (no rerouting needed)                     │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Step 2: Traffic Shaping (Token Bucket)                    │     │
│  │  - Transfer size: 4 MB = 0.004 GB                          │     │
│  │  - Tokens available: 50 GB                                 │     │
│  │  - Consume 0.004 GB tokens                                 │     │
│  │  Result: ALLOWED (no wait)                                 │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Step 3: Adaptive Scheduling                               │     │
│  │  - Select CUDA stream for transfer                         │     │
│  │  - Allocate bandwidth: 50 GB/s (fair share)                │     │
│  │  - Set priority: 5                                         │     │
│  │  Result: stream_id=2, bandwidth=50 GB/s                    │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  Return: TransferScheduleResult(allowed=True, stream=2, bw=50)      │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Execute Transfer (CUDA Runtime)                     │
│                                                                       │
│  - Launch DtoD copy on stream 2                                      │
│  - Actual throughput: 48 GB/s                                        │
│  - Actual latency: 0.083 ms                                          │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│           FlowController.complete_transfer()                         │
│                                                                       │
│  1. Release QoS reservation                                          │
│  2. Update traffic shaper statistics                                 │
│  3. Record performance: latency=0.083ms, throughput=48 GB/s          │
│  4. Update ML models (online learning)                               │
│  5. Update fairness metrics                                          │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Return to User Application                         │
│                                                                       │
│   tensor_gpu1 is ready on cuda:1                                     │
└──────────────────────────────────────────────────────────────────────┘
```

### 2. Congestion Detection Flow

```
┌────────────────────────────────────────────────────────────────┐
│              BandwidthMonitor (Background Thread)               │
│              Runs every 100ms                                   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Collect Measurements                                │     │
│  │  - Per-link bandwidth (CUDA event timing)            │     │
│  │  - Queue depths                                      │     │
│  │  - Active transfer count                             │     │
│  └────────────────┬─────────────────────────────────────┘     │
│                   │                                             │
│                   ▼                                             │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Update Statistics                                   │     │
│  │  - Moving average (1s, 10s)                          │     │
│  │  - Variance calculation                              │     │
│  │  - Peak tracking                                     │     │
│  └────────────────┬─────────────────────────────────────┘     │
└───────────────────┼─────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────┐
│                  CongestionDetector                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Step 1: Explicit Indicators                         │     │
│  │  - Utilization > 80% ?                               │     │
│  │  - Queue depth > 10 ?                                │     │
│  │  - Loss rate > 0.01% ?                               │     │
│  └────────────────┬─────────────────────────────────────┘     │
│                   │                                             │
│                   ▼                                             │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Step 2: Implicit Indicators                         │     │
│  │  - Latency inflation > 20% ?                         │     │
│  │  - Throughput degradation > 15% ?                    │     │
│  │  - Bandwidth variance increasing ?                   │     │
│  └────────────────┬─────────────────────────────────────┘     │
│                   │                                             │
│                   ▼                                             │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Step 3: ML Prediction                               │     │
│  │  - Extract features (10 features)                    │     │
│  │  - XGBoost inference (< 0.1ms)                       │     │
│  │  - Output: congestion_probability                    │     │
│  └────────────────┬─────────────────────────────────────┘     │
│                   │                                             │
│                   ▼                                             │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Step 4: Determine Congestion Level                  │     │
│  │                                                       │     │
│  │  IF utilization < 60%:                               │     │
│  │      level = NONE                                    │     │
│  │  ELIF utilization < 75%:                             │     │
│  │      level = LOW                                     │     │
│  │  ELIF utilization < 85%:                             │     │
│  │      level = MODERATE                                │     │
│  │  ELIF utilization < 95%:                             │     │
│  │      level = HIGH                                    │     │
│  │  ELSE:                                               │     │
│  │      level = CRITICAL                                │     │
│  └────────────────┬─────────────────────────────────────┘     │
└───────────────────┼─────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────┐
│               Congestion Response (if level >= MODERATE)        │
│                                                                 │
│  HIGH Priority:                                                 │
│    - Find alternative paths                                    │
│    - Reroute new flows                                         │
│    - Increase rate limiting                                    │
│                                                                 │
│  CRITICAL Priority:                                             │
│    - Drop low-priority flows                                   │
│    - Preempt background transfers                              │
│    - Emergency throttling                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Token Bucket Traffic Shaping

```
                        Token Bucket State
┌──────────────────────────────────────────────────────────┐
│                                                           │
│  Bucket Capacity: 100 GB                                 │
│  Current Tokens:   75 GB   ████████████████░░░░░░░       │
│  Refill Rate:      10 GB/s                               │
│  Last Refill:      0.5s ago                              │
│                                                           │
└──────────────────────────────────────────────────────────┘

Time: T0
─────────────────────────────────────────────────────────────
Action: Refill
  - Elapsed since last refill: 0.5s
  - Tokens to add: 0.5s × 10 GB/s = 5 GB
  - New token count: 75 + 5 = 80 GB (capped at 100 GB)

State: [Tokens: 80 GB]  ████████████████████░░░

─────────────────────────────────────────────────────────────
Time: T1
─────────────────────────────────────────────────────────────
Request: Transfer 10 GB
  - Required tokens: 10 GB
  - Available tokens: 80 GB
  - Decision: ALLOWED
  - New token count: 80 - 10 = 70 GB

State: [Tokens: 70 GB]  ██████████████░░░░░░░░

─────────────────────────────────────────────────────────────
Time: T2 (immediately after T1)
─────────────────────────────────────────────────────────────
Request: Transfer 80 GB (burst)
  - Required tokens: 80 GB
  - Available tokens: 70 GB
  - Decision: NOT ALLOWED
  - Wait time: (80 - 70) GB / 10 GB/s = 1.0 second

State: [Tokens: 70 GB]  ██████████████░░░░░░░░

Options:
  1. Block for 1.0 second
  2. Return to caller with wait_time
  3. Reduce transfer size to 70 GB

─────────────────────────────────────────────────────────────
Time: T3 (1.0s after T2)
─────────────────────────────────────────────────────────────
Action: Refill
  - Elapsed: 1.0s
  - Tokens to add: 1.0s × 10 GB/s = 10 GB
  - New token count: 70 + 10 = 80 GB

State: [Tokens: 80 GB]  ████████████████████░░░

Request: Transfer 80 GB (retry)
  - Required tokens: 80 GB
  - Available tokens: 80 GB
  - Decision: ALLOWED
  - New token count: 80 - 80 = 0 GB

State: [Tokens:  0 GB]  ░░░░░░░░░░░░░░░░░░░░░░

─────────────────────────────────────────────────────────────
```

### 4. Max-Min Fair Bandwidth Allocation

```
Scenario: 3 flows, 100 GB/s total bandwidth

Flow 1: Demands  20 GB/s  [Small transfer]
Flow 2: Demands  80 GB/s  [Large transfer]
Flow 3: Demands 100 GB/s  [Large transfer]

Step-by-Step Allocation:
═══════════════════════════════════════════════════════════

Iteration 1:
───────────
  Remaining flows: {1, 2, 3}
  Remaining bandwidth: 100 GB/s
  Fair share: 100 / 3 = 33.33 GB/s
  
  Check demands:
    Flow 1: 20 GB/s < 33.33 GB/s  ✓ Satisfied
    Flow 2: 80 GB/s > 33.33 GB/s  ✗ Not satisfied
    Flow 3: 100 GB/s > 33.33 GB/s ✗ Not satisfied
  
  Allocations:
    Flow 1: 20 GB/s  [FINAL]
  
  Update:
    Remaining flows: {2, 3}
    Remaining bandwidth: 100 - 20 = 80 GB/s

Iteration 2:
───────────
  Remaining flows: {2, 3}
  Remaining bandwidth: 80 GB/s
  Fair share: 80 / 2 = 40 GB/s
  
  Check demands:
    Flow 2: 80 GB/s > 40 GB/s  ✗ Not satisfied
    Flow 3: 100 GB/s > 40 GB/s ✗ Not satisfied
  
  All remaining flows need more than fair share
  → Give each flow the fair share
  
  Allocations:
    Flow 2: 40 GB/s  [FINAL]
    Flow 3: 40 GB/s  [FINAL]

Final Allocation:
═════════════════
  Flow 1:  20 GB/s  ████░░░░░░░░░░░░░░░░
  Flow 2:  40 GB/s  ████████░░░░░░░░░░░░
  Flow 3:  40 GB/s  ████████░░░░░░░░░░░░
                    ────────────────────
  Total:  100 GB/s  ████████████████████

Properties:
  ✓ Max-min fair: Flow 1 got full demand
  ✓ Equal sharing: Flows 2 & 3 share remaining equally
  ✓ No waste: All 100 GB/s allocated
```

### 5. ML-Based Latency Prediction

```
┌─────────────────────────────────────────────────────────────┐
│                   Input: Transfer Request                    │
│                                                              │
│  src_device: cuda:0                                          │
│  dst_device: cuda:1                                          │
│  size: 10 GB                                                 │
│  priority: 5                                                 │
│  path: [NVLink_0_1]                                          │
│  dependencies: [transfer_123]                                │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Extraction (64 features)                │
│                                                              │
│  Transfer Features:                                          │
│    - log_size = log(10) = 2.30                              │
│    - priority_norm = 5/10 = 0.5                             │
│    - num_dependencies = 1                                   │
│                                                              │
│  Path Features:                                              │
│    - num_hops = 1                                           │
│    - bottleneck_bw = 300 GB/s                               │
│    - base_latency = 0.01 ms                                 │
│    - max_util = 0.83                                        │
│                                                              │
│  System Features:                                            │
│    - system_util = 0.75                                     │
│    - num_active_flows = 12                                  │
│    - queue_depth = 3                                        │
│    - bw_variance = 15.2                                     │
│                                                              │
│  Historical Features:                                        │
│    - mean_latency_nvlink = 0.15 ms                          │
│    - std_latency_nvlink = 0.05 ms                           │
│    - p99_latency_nvlink = 0.25 ms                           │
│                                                              │
│  + 49 more features (padded)...                             │
│                                                              │
│  Output: feature_vector[64] = [2.30, 0.5, 1.0, ...]        │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Neural Network (MLP)                        │
│                                                              │
│  Layer 1: Linear(64 → 128) + ReLU + BatchNorm + Dropout     │
│    Hidden[128] = [0.23, 0.45, 0.12, ...]                   │
│                                                              │
│  Layer 2: Linear(128 → 128) + ReLU + BatchNorm              │
│    Hidden[128] = [0.34, 0.56, 0.89, ...]                   │
│                                                              │
│  Quantile Heads (Quantile Regression):                      │
│    ├─ P50 Head: Linear(128 → 1) → 0.18 ms                  │
│    ├─ P90 Head: Linear(128 → 1) → 0.24 ms                  │
│    └─ P99 Head: Linear(128 → 1) → 0.31 ms                  │
│                                                              │
│  Uncertainty Heads:                                          │
│    ├─ Mean Head: Linear(128 → 1) → 0.19 ms                 │
│    └─ Std Head:  Linear(128 → 1) + Softplus → 0.06 ms      │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      Prediction Output                       │
│                                                              │
│  Predicted Latency Distribution:                            │
│    P50 = 0.18 ms  (median)                                  │
│    P90 = 0.24 ms  (90th percentile)                         │
│    P99 = 0.31 ms  (99th percentile)  ← Use for conservative │
│    Mean = 0.19 ms                                           │
│    Std = 0.06 ms  (uncertainty)                             │
│                                                              │
│  Visualization:                                              │
│                                                              │
│   Probability                                                │
│       │                                                      │
│   1.0 │                                                      │
│       │                ╱╲                                    │
│   0.8 │               ╱  ╲                                   │
│       │              ╱    ╲                                  │
│   0.6 │             ╱      ╲                                 │
│       │            ╱        ╲                                │
│   0.4 │           ╱          ╲                               │
│       │          ╱            ╲                              │
│   0.2 │         ╱              ╲___                          │
│       │    ____╱                   ╲____                     │
│   0.0 └────────────────────────────────────► Latency (ms)   │
│        0.0   0.1   0.2   0.3   0.4   0.5                    │
│              ↑     ↑     ↑                                   │
│              P50   P90   P99                                 │
│                                                              │
│  Decision: Use P99 (0.31 ms) for scheduling                 │
└──────────────────────────────────────────────────────────────┘

Inference Time: 0.15 ms  ✓ (< 0.2ms target)
```

### 6. QoS Flow Lifecycle

```
Time: T0 - Flow Registration
═══════════════════════════════════════════════════════════════

User Request:
  QoS Class: GUARANTEED
  Contract:
    - min_bandwidth: 10 GB/s
    - max_latency: 50 ms
    - duration: 5 seconds

QoS Manager State:
  Total Bandwidth: 100 GB/s
  Reserved: 60 GB/s (6 flows)
  Available: 40 GB/s

Admission Decision:
  ✓ Bandwidth check: 10 GB/s < 40 GB/s available
  ✓ Latency check: predicted 30 ms < 50 ms max
  ✓ Result: ADMITTED (priority=1, reservation_id=789)

Updated State:
  Total Bandwidth: 100 GB/s
  Reserved: 70 GB/s (7 flows)
  Available: 30 GB/s

─────────────────────────────────────────────────────────────

Time: T1 - Transfer Execution
═══════════════════════════════════════════════════════════════

Bandwidth Allocation (Hierarchical):
  
  System (100 GB/s)
  ├─ GUARANTEED (30 GB/s reserved)
  │  ├─ Flow 789: 10 GB/s ◄── Our flow
  │  └─ Other flows: 20 GB/s
  ├─ PREMIUM (20 GB/s reserved)
  │  └─ Flows: 20 GB/s
  └─ BEST_EFFORT (50 GB/s opportunistic)
     └─ Flows: 50 GB/s

Actual Performance:
  Bandwidth: 10.5 GB/s  ✓ (>= 10 GB/s minimum)
  Latency: 42 ms        ✓ (< 50 ms maximum)

SLA Status: COMPLIANT

─────────────────────────────────────────────────────────────

Time: T2 - Congestion Event
═══════════════════════════════════════════════════════════════

New Flow Arrives:
  QoS Class: CRITICAL
  Requires: 25 GB/s
  
Current Allocation:
  Reserved: 70 GB/s
  Available: 30 GB/s
  Needed: 25 GB/s
  Deficit: 0 GB/s  ✓ Enough capacity

Preemption Check:
  - Critical priority > Guaranteed priority
  - Preemption allowed for CRITICAL
  - No preemption needed (sufficient capacity)

Decision: Admit new flow, no preemption

─────────────────────────────────────────────────────────────

Time: T3 - Another Flow (Would Cause Preemption)
═══════════════════════════════════════════════════════════════

New Flow Arrives:
  QoS Class: CRITICAL
  Requires: 40 GB/s
  
Current Allocation:
  Reserved: 95 GB/s (includes previous flows)
  Available: 5 GB/s
  Needed: 40 GB/s
  Deficit: 35 GB/s  ✗ Insufficient capacity

Preemption Decision:
  1. Identify preemption candidates:
     - BEST_EFFORT flows (priority 3): 50 GB/s
     - Can reclaim up to 35 GB/s
  
  2. Select flows to preempt:
     - Largest BEST_EFFORT flows first
     - Preempt 3 flows totaling 35 GB/s
  
  3. Execute preemption:
     - Pause preempted flows
     - Reallocate bandwidth
     - Admit CRITICAL flow

Result: CRITICAL flow admitted, 3 BEST_EFFORT flows preempted

─────────────────────────────────────────────────────────────

Time: T4 - Flow Completion
═══════════════════════════════════════════════════════════════

Flow 789 completes:
  Duration: 4.8 seconds
  Total data: 50 GB
  Avg bandwidth: 10.4 GB/s
  Avg latency: 41 ms
  SLA violations: 0

QoS Manager:
  1. Release reservation: 10 GB/s
  2. Update statistics
  3. Resume preempted flows (if any)
  4. Record SLA compliance

Final Stats:
  ✓ Bandwidth SLA: 100% compliance
  ✓ Latency SLA: 100% compliance
  ✓ Duration: 96% of requested (4.8/5.0s)
```

## Summary

These diagrams illustrate:

1. **Complete transfer flow** from user code to hardware
2. **Congestion detection** with multi-level indicators
3. **Token bucket** traffic shaping mechanics
4. **Max-min fair allocation** algorithm execution
5. **ML latency prediction** with quantile regression
6. **QoS lifecycle** including admission control and preemption

All diagrams show actual system behavior with realistic values and decision points.

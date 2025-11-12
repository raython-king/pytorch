# PyTorch Runtime Scheduler Architecture

## Executive Summary

This document describes a comprehensive **Runtime Scheduling System** for PyTorch that makes dynamic decisions during model execution. The system uses machine learning models to optimize operation scheduling, device placement, stream assignment, and memory management in real-time.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Integration Points](#integration-points)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PyTorch Application Layer                     │
│                     (User Model + torch.compile)                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Runtime Scheduler System                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Scheduling Decision Engine (SDE)                 │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │  │
│  │  │ Workload │  │  Device  │  │  Stream  │  │   Memory    │  │  │
│  │  │Scheduler │  │  Manager │  │  Manager │  │  Scheduler  │  │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘  │  │
│  │       │             │              │                │          │  │
│  │       └─────────────┴──────────────┴────────────────┘          │  │
│  │                             │                                   │  │
│  │                             ▼                                   │  │
│  │           ┌──────────────────────────────────┐                 │  │
│  │           │    ML-Based Decision Models      │                 │  │
│  │           │  ┌────────┐  ┌────────┐  ┌────┐ │                 │  │
│  │           │  │  GNN   │  │Transf. │  │ RL │ │                 │  │
│  │           │  │ Model  │  │ Model  │  │ Ag.│ │                 │  │
│  │           │  └────────┘  └────────┘  └────┘ │                 │  │
│  │           └──────────────────────────────────┘                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    State Manager                              │  │
│  │  - System State (device util, memory, queue lengths)          │  │
│  │  - Operation Metadata (compute, memory, dependencies)         │  │
│  │  - Historical Traces (execution history)                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────┬─────────────────────────────────────────┬─────────────┘
             │                                         │
             ▼                                         ▼
┌────────────────────────┐                ┌───────────────────────┐
│   PyTorch Dispatcher   │                │   CUDA Runtime        │
│   (c10::Dispatcher)    │                │   - Streams           │
│   - Operation dispatch │                │   - Memory allocator  │
│   - Kernel selection   │                │   - Device management │
└────────────────────────┘                └───────────────────────┘
```

### System Layers

The runtime scheduler operates at three levels:

1. **Decision Layer**: ML models make scheduling decisions
2. **Orchestration Layer**: Coordinators execute decisions
3. **Execution Layer**: PyTorch runtime executes operations

---

## Core Components

### 1. Scheduling Decision Engine (SDE)

**Purpose**: Central coordinator that receives execution requests and makes scheduling decisions.

**Responsibilities**:
- Receive operation dispatch requests
- Collect system state and operation metadata
- Invoke ML models for decision making
- Coordinate between sub-schedulers
- Enforce correctness constraints

**Key Characteristics**:
- Low latency (< 1ms per decision)
- Thread-safe and lock-free where possible
- Supports batched decision making
- Caches recent decisions

### 2. WorkloadScheduler

**Purpose**: Decides which operations to execute next and in what order.

**Decision Points**:
- Operation priorities
- Execution ordering
- Batching opportunities
- Fusion opportunities

**ML Model Inputs**:
```python
{
    'op_features': [
        compute_intensity,      # FLOPs
        memory_intensity,       # bytes read/written
        data_dependencies,      # dependency graph
        execution_time_estimate # predicted time
    ],
    'system_state': [
        pending_ops_count,      # queue depth
        device_utilization,     # % busy
        memory_pressure         # available memory
    ]
}
```

**ML Model Outputs**:
```python
{
    'priorities': List[float],        # Priority scores for each op
    'schedule_order': List[int],      # Recommended execution order
    'batch_groups': List[List[int]]   # Operations to batch together
}
```

### 3. DeviceManager

**Purpose**: Manages multi-device execution and device placement decisions.

**Architecture**:
```
DeviceManager
│
├── DeviceState (per device)
│   ├── utilization: float (0-1)
│   ├── memory_available: int (bytes)
│   ├── active_streams: int
│   ├── pending_ops: Queue
│   └── temperature: float (for throttling)
│
├── LoadBalancer
│   ├── strategy: "ml" | "round_robin" | "least_loaded"
│   └── ml_model: DevicePlacementModel
│
└── DeviceTopology
    ├── p2p_bandwidth: Dict[device_pair, bandwidth]
    ├── nvlink_connections: Set[device_pair]
    └── numa_affinity: Dict[device, cpu_node]
```

**Placement Decision Factors**:
- Device utilization and memory
- Data locality (minimize transfers)
- Communication costs
- Load balancing across devices

### 4. StreamManager

**Purpose**: Manages CUDA streams for parallelism and dependency satisfaction.

**Stream Pool Architecture**:
```
StreamManager (per device)
│
├── DefaultStream (stream 0)
│
├── ComputeStreamPool
│   ├── HighPriorityStreams (4 streams)
│   └── NormalPriorityStreams (28 streams)
│
├── CopyStreamPool (dedicated for D2D, H2D, D2H)
│   ├── H2D_stream (host to device)
│   ├── D2H_stream (device to host)
│   └── D2D_streams (device to device)
│
└── StreamDependencyTracker
    ├── stream_sync_graph: DAG of stream dependencies
    └── event_pool: Pool of reusable cudaEvent_t
```

**Stream Assignment Decisions**:
```python
def assign_stream(op: Operation) -> Stream:
    """
    ML model decides stream assignment based on:
    - Operation dependencies (data hazards)
    - Available parallelism
    - Stream utilization
    - Communication/computation overlap opportunities
    """
    features = {
        'op_type': op.type,
        'dependencies': extract_deps(op),
        'stream_utilization': get_stream_states(),
        'overlap_opportunity': estimate_overlap_benefit(op)
    }
    
    stream_scores = ml_model.predict(features)
    return streams[argmax(stream_scores)]
```

### 5. MemoryScheduler

**Purpose**: Dynamic memory allocation, eviction, and prefetching decisions.

**Memory Management Architecture**:
```
MemoryScheduler
│
├── AllocationPolicy
│   ├── strategy: "ml" | "best_fit" | "caching"
│   └── ml_model: MemoryAllocationModel
│
├── EvictionPolicy
│   ├── strategy: "ml" | "lru" | "lfu"
│   ├── ml_model: EvictionModel
│   └── eviction_candidates: PriorityQueue
│
├── PrefetchEngine
│   ├── prefetch_predictor: ML model
│   ├── prefetch_queue: Queue
│   └── prefetch_stream: dedicated CUDA stream
│
└── MemoryPoolManager
    ├── pools: Dict[device, MemoryPool]
    └── fragmentation_monitor: Monitor
```

**Memory Decisions**:

1. **Allocation**: When to allocate, how much, where
2. **Eviction**: What to evict when out of memory
3. **Prefetching**: What to prefetch, when, to which device
4. **Reuse**: Tensor memory reuse opportunities

**ML Model for Memory**:
```python
class MemoryDecisionModel:
    def predict_allocation_size(
        self,
        requested_size: int,
        op_metadata: Dict,
        memory_state: MemoryState
    ) -> int:
        """
        Predicts optimal allocation size (may be larger for reuse).
        """
        pass
    
    def predict_eviction_candidates(
        self,
        tensors: List[Tensor],
        upcoming_ops: List[Operation]
    ) -> List[Tensor]:
        """
        Predicts which tensors to evict.
        Uses Belady's algorithm with ML predictions.
        """
        pass
    
    def predict_prefetch_targets(
        self,
        current_op: Operation,
        lookahead: int = 5
    ) -> List[Tuple[Tensor, Device]]:
        """
        Predicts what to prefetch and where.
        """
        pass
```

### 6. LoadBalancer

**Purpose**: Balances work across devices and detects stragglers.

**Metrics**:
- Per-device throughput
- Queue depths
- Straggler detection
- Dynamic work stealing

---

## Data Flow

### Operation Execution Flow with Runtime Scheduler

```
┌──────────────────┐
│  User calls op   │
│  x = y + z       │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────────┐
│  PyTorch Dispatcher                    │
│  Dispatcher::call<>(op, args...)       │
└────────┬───────────────────────────────┘
         │
         │ Hook Point 1: Pre-dispatch
         ▼
┌────────────────────────────────────────┐
│  Runtime Scheduler Hook                │
│  - Capture operation metadata          │
│  - Extract features                    │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│  Scheduling Decision Engine            │
│                                        │
│  1. WorkloadScheduler.schedule(op)    │
│     ├─> Priority: 0.85                │
│     └─> Queue: high_priority          │
│                                        │
│  2. DeviceManager.assign_device(op)   │
│     ├─> Device: cuda:1 (less loaded)  │
│     └─> Reason: load_balancing        │
│                                        │
│  3. StreamManager.assign_stream(op)   │
│     ├─> Stream: compute_stream_3      │
│     └─> Dependencies: [stream_1]      │
│                                        │
│  4. MemoryScheduler.prepare_memory()  │
│     ├─> Allocate: output tensor       │
│     ├─> Prefetch: input from CPU      │
│     └─> Evict: unused_tensor_x        │
└────────┬───────────────────────────────┘
         │
         │ Scheduling decisions made
         ▼
┌────────────────────────────────────────┐
│  Apply Scheduling Decisions            │
│  - Set device (cudaSetDevice)          │
│  - Set stream (setCurrentCUDAStream)   │
│  - Allocate memory if needed           │
│  - Insert stream dependencies (events) │
└────────┬───────────────────────────────┘
         │
         │ Hook Point 2: Post-decision
         ▼
┌────────────────────────────────────────┐
│  Kernel Dispatch                       │
│  - kernel.call(device, stream, args)   │
└────────┬───────────────────────────────┘
         │
         │ Hook Point 3: Post-dispatch
         ▼
┌────────────────────────────────────────┐
│  State Update                          │
│  - Update device utilization           │
│  - Update stream state                 │
│  - Record timing (for learning)        │
│  - Update dependency graph             │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│  CUDA Execution                        │
│  - Kernel runs on GPU                  │
│  - Async execution                     │
└────────────────────────────────────────┘
```

### Feedback Loop for Learning

```
┌────────────────┐
│  Operation     │
│  Execution     │
└────────┬───────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Profiling & Measurement            │
│  - Actual execution time            │
│  - Memory usage                     │
│  - Achieved parallelism             │
│  - Cache hit rates                  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Feedback Collector                 │
│  - Compare predicted vs actual      │
│  - Compute reward signal            │
│  - Store in replay buffer           │
└────────┬────────────────────────────┘
         │
         │ Periodic (every N ops)
         ▼
┌─────────────────────────────────────┐
│  Online Learning (Optional)         │
│  - Update model parameters          │
│  - Adjust scheduling strategies     │
│  - A/B testing of policies          │
└─────────────────────────────────────┘
```

---

## Integration Points

### 1. Dispatcher Integration

**Hook Location**: `/home/user/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h`

**Integration Strategy**:
```cpp
// In Dispatcher::call()
template <class Return, class... Args>
Return Dispatcher::call(
    const TypedOperatorHandle<Return(Args...)>& op,
    Args... args) const {
  
  // HOOK POINT 1: Pre-dispatch scheduling decision
  if (RuntimeScheduler::isEnabled()) {
    auto decision = RuntimeScheduler::singleton().schedule(op, args...);
    
    // Apply device placement
    if (decision.device != getCurrentDevice()) {
      c10::cuda::set_device(decision.device);
    }
    
    // Apply stream assignment
    if (decision.stream != getCurrentCUDAStream()) {
      c10::cuda::setCurrentCUDAStream(decision.stream);
    }
    
    // Handle dependencies (insert events)
    if (!decision.dependencies.empty()) {
      decision.insertStreamDependencies();
    }
  }
  
  // Original dispatch logic continues...
  auto dispatchKeySet = /* ... */;
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
  
  auto result = kernel.template call<Return, Args...>(
      op, dispatchKeySet, std::forward<Args>(args)...);
  
  // HOOK POINT 2: Post-dispatch state update
  if (RuntimeScheduler::isEnabled()) {
    RuntimeScheduler::singleton().recordExecution(op, decision);
  }
  
  return result;
}
```

### 2. Memory Allocator Integration

**Hook Location**: `/home/user/pytorch/c10/cuda/CUDACachingAllocator.cpp`

**Integration Strategy**:
```cpp
DataPtr CUDACachingAllocator::allocate(size_t size) {
  // HOOK: Query runtime scheduler for allocation decision
  if (RuntimeScheduler::isEnabled()) {
    auto decision = RuntimeScheduler::singleton()
        .memoryScheduler()
        .allocate_decision(size, getCurrentDevice());
    
    // Check if we should evict first
    if (decision.should_evict && !decision.evict_candidates.empty()) {
      for (auto& candidate : decision.evict_candidates) {
        evict(candidate);
      }
    }
    
    // Use predicted allocation size (may be larger for reuse)
    size = decision.allocation_size;
  }
  
  // Continue with normal allocation logic...
  Block* block = /* find free block or allocate new */;
  return DataPtr(/* ... */);
}
```

### 3. Stream Management Integration

**Hook Location**: `/home/user/pytorch/c10/cuda/CUDAStream.cpp`

**Integration Strategy**:
```cpp
CUDAStream getStreamFromPool(bool isHighPriority, DeviceIndex device) {
  // HOOK: Let runtime scheduler decide stream
  if (RuntimeScheduler::isEnabled()) {
    auto stream = RuntimeScheduler::singleton()
        .streamManager()
        .assign_stream(getCurrentOp(), device);
    return stream;
  }
  
  // Fall back to default round-robin
  return default_stream_pool.get_stream(isHighPriority, device);
}
```

### 4. Autograd Engine Integration

**Hook Location**: `/home/user/pytorch/torch/csrc/autograd/engine.cpp`

**Integration Strategy**:
```cpp
void Engine::evaluate_function(
    std::shared_ptr<GraphTask>& graph_task,
    Node* func,
    InputBuffer& inputs,
    const std::shared_ptr<ReadyQueue>& cpu_ready_queue) {
  
  // HOOK: Runtime scheduler can reorder autograd nodes
  if (RuntimeScheduler::isEnabled()) {
    auto schedule = RuntimeScheduler::singleton()
        .workloadScheduler()
        .schedule_autograd_node(func, graph_task);
    
    // Apply scheduling decisions
    apply_schedule(schedule);
  }
  
  // Continue with normal execution...
}
```

---

## Performance Characteristics

### Overhead Analysis

| Component | Overhead per Operation | Mitigation Strategy |
|-----------|----------------------|-------------------|
| Feature Extraction | 10-50 μs | Cache features, batch extraction |
| ML Model Inference | 50-500 μs | Batch inference, use fast models (MLP) |
| Decision Application | 5-20 μs | Optimize hot path, inline |
| State Update | 5-10 μs | Lock-free data structures |
| **Total** | **70-580 μs** | **< 1ms target** |

### When to Use Runtime Scheduler

Enable runtime scheduler when:
- Model has > 100 operations
- Multi-GPU training
- Memory-bound workloads
- Irregular execution patterns
- Dynamic shapes

Disable for:
- Simple models (< 50 ops)
- Single GPU inference
- Tiny batch sizes
- Heavily optimized models (already fused)

---

## Next Sections

- [02_component_apis.md](./02_component_apis.md) - Detailed API specifications
- [03_ml_models.md](./03_ml_models.md) - ML model architectures
- [04_integration_guide.md](./04_integration_guide.md) - Implementation guide
- [05_performance_analysis.md](./05_performance_analysis.md) - Performance deep dive
- [06_safety_mechanisms.md](./06_safety_mechanisms.md) - Safety and validation


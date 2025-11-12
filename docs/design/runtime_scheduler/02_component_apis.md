# Runtime Scheduler Component APIs

## Overview

This document specifies the detailed APIs for each component of the Runtime Scheduler system.

---

## Table of Contents

1. [Core Interfaces](#core-interfaces)
2. [Scheduling Decision Engine](#scheduling-decision-engine)
3. [WorkloadScheduler](#workloadscheduler)
4. [DeviceManager](#devicemanager)
5. [StreamManager](#streammanager)
6. [MemoryScheduler](#memoryscheduler)
7. [State Manager](#state-manager)

---

## Core Interfaces

### SchedulingDecision

```cpp
namespace torch::runtime_scheduler {

// Represents a complete scheduling decision for an operation
struct SchedulingDecision {
  // Device placement
  c10::Device device;
  
  // Stream assignment
  c10::cuda::CUDAStream stream;
  
  // Stream dependencies (operations that must complete first)
  std::vector<c10::cuda::CUDAEvent> dependencies;
  
  // Memory decisions
  struct MemoryDecisions {
    bool prefetch_inputs = false;
    std::vector<at::Tensor> evict_candidates;
    size_t allocation_size = 0;  // 0 means use requested size
  } memory;
  
  // Priority score (higher = more urgent)
  float priority = 0.5f;
  
  // Batching hint
  std::optional<int> batch_group_id;
  
  // Metadata
  std::chrono::microseconds decision_time;
  float confidence = 1.0f;  // Model confidence [0, 1]
  bool use_fallback = false;  // Whether fallback heuristic was used
  
  // Apply this decision to current execution context
  void apply() const;
  
  // Insert stream dependencies using CUDA events
  void insertStreamDependencies() const;
};

// Operation metadata captured at dispatch time
struct OperationMetadata {
  // Operation identification
  c10::OperatorHandle op_handle;
  std::string op_name;
  
  // Operation characteristics
  int64_t estimated_flops = 0;
  int64_t estimated_memory_bytes = 0;
  std::vector<c10::Device> input_devices;
  std::vector<c10::ScalarType> input_dtypes;
  std::vector<c10::IntArrayRef> input_shapes;
  
  // Dependencies
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  
  // Context
  c10::Device current_device;
  c10::cuda::CUDAStream current_stream;
  int64_t sequence_number = 0;
  
  // Extract from operation arguments
  template<typename... Args>
  static OperationMetadata extract(
      const c10::OperatorHandle& op,
      Args&&... args);
};

} // namespace torch::runtime_scheduler
```

---

## Scheduling Decision Engine

### RuntimeScheduler (Singleton)

```cpp
namespace torch::runtime_scheduler {

class TORCH_API RuntimeScheduler {
 public:
  // Get singleton instance
  static RuntimeScheduler& singleton();
  
  // Configuration
  struct Config {
    bool enabled = false;
    bool use_ml_models = true;
    bool collect_feedback = true;
    float confidence_threshold = 0.75f;
    int batch_decision_size = 32;  // Batch up to N operations
    std::string model_path = "";
    
    // Performance knobs
    int max_decision_time_us = 500;  // Max time for decision
    int state_update_interval_ops = 10;  // Update state every N ops
    bool enable_async_decisions = true;  // Make decisions in background
  };
  
  // Initialize with configuration
  void initialize(const Config& config);
  
  // Check if runtime scheduler is enabled
  static bool isEnabled();
  
  // Main scheduling entry point
  SchedulingDecision schedule(
      const c10::OperatorHandle& op,
      const OperationMetadata& metadata);
  
  // Batch scheduling for multiple operations
  std::vector<SchedulingDecision> scheduleBatch(
      const std::vector<std::pair<c10::OperatorHandle, OperationMetadata>>& ops);
  
  // Record execution results (for feedback)
  void recordExecution(
      const c10::OperatorHandle& op,
      const SchedulingDecision& decision,
      std::chrono::microseconds actual_time);
  
  // Access sub-schedulers
  WorkloadScheduler& workloadScheduler();
  DeviceManager& deviceManager();
  StreamManager& streamManager();
  MemoryScheduler& memoryScheduler();
  StateManager& stateManager();
  
  // Control
  void enable();
  void disable();
  void reset();
  
  // Statistics
  struct Stats {
    uint64_t total_decisions = 0;
    uint64_t ml_decisions = 0;
    uint64_t fallback_decisions = 0;
    uint64_t cached_decisions = 0;
    std::chrono::microseconds total_decision_time{0};
    std::chrono::microseconds avg_decision_time{0};
  };
  
  Stats getStats() const;
  void resetStats();
  
 private:
  RuntimeScheduler();
  ~RuntimeScheduler();
  
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace torch::runtime_scheduler
```

---

## WorkloadScheduler

### Interface

```cpp
namespace torch::runtime_scheduler {

class TORCH_API WorkloadScheduler {
 public:
  // Priority-based scheduling
  struct ScheduleResult {
    float priority;  // Urgency score [0, 1]
    int queue_assignment;  // Which queue to use (0=default, 1=high, 2=low)
    std::optional<int> batch_group;  // Group ID for batching
    bool should_defer;  // Whether to defer execution
  };
  
  ScheduleResult schedule(
      const OperationMetadata& op,
      const SystemState& state);
  
  // Batch scheduling
  std::vector<ScheduleResult> scheduleBatch(
      const std::vector<OperationMetadata>& ops,
      const SystemState& state);
  
  // Fusion opportunities
  struct FusionOpportunity {
    std::vector<int> op_indices;  // Operations to fuse
    float expected_speedup;  // Predicted speedup
    float confidence;  // Confidence in prediction
  };
  
  std::vector<FusionOpportunity> findFusionOpportunities(
      const std::vector<OperationMetadata>& ops);
  
  // Query scheduling state
  int getPendingOpCount() const;
  float getAveragePriority() const;
  
 private:
  class MLSchedulingModel* ml_model_;
  std::priority_queue</* ... */> pending_ops_;
};

} // namespace torch::runtime_scheduler
```

### ML Model Interface

```cpp
namespace torch::runtime_scheduler {

class MLSchedulingModel {
 public:
  virtual ~MLSchedulingModel() = default;
  
  // Feature extraction
  struct OpFeatures {
    torch::Tensor compute_features;  // [compute_dim]
    torch::Tensor memory_features;   // [memory_dim]
    torch::Tensor dependency_features;  // [dep_dim]
  };
  
  OpFeatures extractFeatures(const OperationMetadata& op);
  
  // Predictions
  virtual float predictPriority(
      const OpFeatures& op_features,
      const torch::Tensor& state_features) = 0;
  
  virtual torch::Tensor predictBatchSchedule(
      const std::vector<OpFeatures>& ops,
      const torch::Tensor& state_features) = 0;
  
  // Load/save model
  virtual void load(const std::string& path) = 0;
  virtual void save(const std::string& path) = 0;
  
  // Online learning (optional)
  virtual void update(
      const OpFeatures& features,
      float reward) {}
};

// Concrete implementations
class GNNSchedulingModel : public MLSchedulingModel { /* ... */ };
class TransformerSchedulingModel : public MLSchedulingModel { /* ... */ };
class RLSchedulingModel : public MLSchedulingModel { /* ... */ };

} // namespace torch::runtime_scheduler
```

---

## DeviceManager

### Interface

```cpp
namespace torch::runtime_scheduler {

class TORCH_API DeviceManager {
 public:
  // Device selection
  c10::Device selectDevice(
      const OperationMetadata& op,
      const SystemState& state);
  
  // Multi-device scheduling
  struct MultiDeviceSchedule {
    std::vector<c10::Device> devices;  // One per operation
    std::vector<std::pair<int, int>> transfers;  // (from_op, to_op) pairs
    float estimated_time;  // Total time estimate
  };
  
  MultiDeviceSchedule scheduleMultiDevice(
      const std::vector<OperationMetadata>& ops);
  
  // Device state
  struct DeviceState {
    c10::Device device;
    float utilization;  // [0, 1]
    size_t available_memory;
    size_t total_memory;
    int active_streams;
    int pending_ops;
    float temperature;  // For thermal throttling
    
    // NVLINK/PCIe bandwidth to other devices
    std::unordered_map<c10::Device, float> p2p_bandwidth;
  };
  
  DeviceState getDeviceState(c10::Device device) const;
  std::vector<DeviceState> getAllDeviceStates() const;
  
  // Load balancing
  struct LoadBalanceResult {
    std::vector<int> op_to_device;  // Device assignment per op
    float load_variance;  // Lower is better balanced
  };
  
  LoadBalanceResult balanceLoad(
      const std::vector<OperationMetadata>& ops);
  
  // Device topology
  struct Topology {
    int num_devices;
    std::vector<std::vector<float>> bandwidth_matrix;  // [device][device]
    std::vector<std::vector<bool>> nvlink_matrix;  // Has NVLink?
    std::vector<int> numa_nodes;  // NUMA node per device
  };
  
  const Topology& getTopology() const;
  
 private:
  std::vector<DeviceState> device_states_;
  Topology topology_;
  class DevicePlacementModel* ml_model_;
};

} // namespace torch::runtime_scheduler
```

---

## StreamManager

### Interface

```cpp
namespace torch::runtime_scheduler {

class TORCH_API StreamManager {
 public:
  // Stream assignment
  c10::cuda::CUDAStream assignStream(
      const OperationMetadata& op,
      c10::Device device);
  
  // Dependency management
  struct StreamDependency {
    c10::cuda::CUDAStream src_stream;
    c10::cuda::CUDAStream dst_stream;
    c10::cuda::CUDAEvent event;  // Event to wait on
  };
  
  std::vector<StreamDependency> computeDependencies(
      const OperationMetadata& op,
      c10::cuda::CUDAStream assigned_stream);
  
  // Insert dependencies (cudaStreamWaitEvent)
  void insertDependencies(
      const std::vector<StreamDependency>& deps);
  
  // Stream state
  struct StreamState {
    c10::cuda::CUDAStream stream;
    int priority;  // CUDA stream priority
    float utilization;  // Estimated utilization [0, 1]
    int pending_ops;
    bool is_busy;
    std::chrono::microseconds last_used;
  };
  
  std::vector<StreamState> getStreamStates(c10::Device device) const;
  
  // Stream pool management
  struct StreamPoolConfig {
    int num_compute_streams = 32;
    int num_high_priority = 4;
    int num_copy_streams = 3;  // H2D, D2H, D2D
    bool enable_stream_capture = true;  // For CUDA graphs
  };
  
  void configureStreamPool(c10::Device device, const StreamPoolConfig& config);
  
  // Overlap analysis
  struct OverlapOpportunity {
    c10::cuda::CUDAStream stream1;
    c10::cuda::CUDAStream stream2;
    std::vector<int> ops_on_stream1;
    std::vector<int> ops_on_stream2;
    float expected_overlap;  // Time overlap [0, 1]
  };
  
  std::vector<OverlapOpportunity> findOverlapOpportunities(
      const std::vector<OperationMetadata>& ops);
  
 private:
  // Per-device stream pools
  std::unordered_map<c10::Device, std::vector<c10::cuda::CUDAStream>> stream_pools_;
  
  // Dependency tracker
  class StreamDependencyGraph* dep_graph_;
  
  // ML model for stream assignment
  class StreamAssignmentModel* ml_model_;
};

} // namespace torch::runtime_scheduler
```

---

## MemoryScheduler

### Interface

```cpp
namespace torch::runtime_scheduler {

class TORCH_API MemoryScheduler {
 public:
  // Allocation decisions
  struct AllocationDecision {
    size_t allocation_size;  // May be larger than requested
    bool should_evict;
    std::vector<at::Tensor> evict_candidates;
    bool use_prefetch;
    c10::Device source_device;  // For prefetch
  };
  
  AllocationDecision decideAllocation(
      size_t requested_size,
      c10::Device device,
      const OperationMetadata& op);
  
  // Eviction policy
  struct EvictionCandidate {
    at::Tensor tensor;
    float eviction_score;  // Higher = better to evict
    std::chrono::microseconds last_used;
    std::chrono::microseconds predicted_next_use;
  };
  
  std::vector<EvictionCandidate> selectEvictionCandidates(
      size_t required_bytes,
      c10::Device device);
  
  // Prefetching
  struct PrefetchDecision {
    at::Tensor tensor;
    c10::Device target_device;
    c10::cuda::CUDAStream prefetch_stream;
    int prefetch_ahead_ops;  // How many ops ahead
    float confidence;
  };
  
  std::vector<PrefetchDecision> decidePrefetch(
      const std::vector<OperationMetadata>& upcoming_ops,
      int lookahead = 5);
  
  void executePrefetch(const PrefetchDecision& decision);
  
  // Memory state
  struct MemoryState {
    c10::Device device;
    size_t total_memory;
    size_t allocated_memory;
    size_t reserved_memory;  // By caching allocator
    size_t active_memory;  // Actually in use
    float fragmentation;  // [0, 1], higher is worse
    int num_allocations;
    int num_cached_blocks;
  };
  
  MemoryState getMemoryState(c10::Device device) const;
  
  // Memory pool management
  void releaseCache(c10::Device device);
  void consolidateFragmentation(c10::Device device);
  
  // Tensor reuse opportunities
  struct ReuseOpportunity {
    at::Tensor source_tensor;  // Tensor to reuse from
    std::string target_name;  // Tensor to create
    bool requires_copy;
    float memory_saved;
  };
  
  std::vector<ReuseOpportunity> findReuseOpportunities(
      const std::vector<OperationMetadata>& ops);
  
 private:
  // Memory state tracking
  std::unordered_map<c10::Device, MemoryState> memory_states_;
  
  // Tensor access tracking
  struct TensorAccessInfo {
    at::Tensor tensor;
    std::vector<std::chrono::steady_clock::time_point> access_times;
    int access_count;
  };
  std::unordered_map<std::string, TensorAccessInfo> tensor_access_history_;
  
  // ML models
  class EvictionModel* eviction_model_;
  class PrefetchModel* prefetch_model_;
};

} // namespace torch::runtime_scheduler
```

---

## State Manager

### Interface

```cpp
namespace torch::runtime_scheduler {

class TORCH_API StateManager {
 public:
  // System state snapshot
  struct SystemState {
    // Time
    std::chrono::steady_clock::time_point timestamp;
    
    // Device states
    std::vector<DeviceManager::DeviceState> device_states;
    
    // Stream states
    std::unordered_map<c10::Device, std::vector<StreamManager::StreamState>> stream_states;
    
    // Memory states
    std::unordered_map<c10::Device, MemoryScheduler::MemoryState> memory_states;
    
    // Workload state
    int total_pending_ops;
    int total_executing_ops;
    float avg_queue_depth;
    
    // Convert to tensor for ML models
    torch::Tensor toTensor() const;
  };
  
  // Get current system state
  SystemState getSystemState() const;
  
  // Update system state (called after each operation)
  void updateState(const OperationMetadata& op, 
                   const SchedulingDecision& decision,
                   std::chrono::microseconds actual_time);
  
  // Historical traces
  struct ExecutionTrace {
    OperationMetadata op;
    SchedulingDecision decision;
    std::chrono::microseconds actual_time;
    SystemState state_before;
    SystemState state_after;
  };
  
  void recordTrace(const ExecutionTrace& trace);
  std::vector<ExecutionTrace> getRecentTraces(int count = 100) const;
  void exportTraces(const std::string& path) const;
  
  // Statistics
  struct Statistics {
    // Throughput
    float ops_per_second;
    float achieved_tflops;
    
    // Utilization
    std::unordered_map<c10::Device, float> device_utilization;
    std::unordered_map<c10::Device, float> memory_utilization;
    
    // Scheduling quality
    float avg_decision_time_us;
    float decision_hit_rate;  // Cache hit rate
    float model_confidence_avg;
    
    // Performance
    std::chrono::microseconds avg_op_time;
    std::chrono::microseconds e2e_latency;
  };
  
  Statistics getStatistics() const;
  void resetStatistics();
  
 private:
  // Current state
  SystemState current_state_;
  std::mutex state_mutex_;
  
  // Historical data
  std::deque<ExecutionTrace> trace_buffer_;
  size_t max_trace_buffer_size_ = 10000;
  
  // Statistics accumulation
  Statistics stats_;
};

} // namespace torch::runtime_scheduler
```

---

## Python Bindings

### PyTorch Python API

```python
# torch/runtime_scheduler/__init__.py

import torch
from typing import Optional, Dict, Any

class RuntimeSchedulerConfig:
    """Configuration for runtime scheduler."""
    enabled: bool = False
    use_ml_models: bool = True
    collect_feedback: bool = True
    confidence_threshold: float = 0.75
    model_path: Optional[str] = None
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def enable_runtime_scheduler(
    config: Optional[RuntimeSchedulerConfig] = None
) -> None:
    """
    Enable runtime scheduler with optional configuration.
    
    Example:
        >>> import torch
        >>> from torch.runtime_scheduler import enable_runtime_scheduler, RuntimeSchedulerConfig
        >>> 
        >>> config = RuntimeSchedulerConfig(
        ...     enabled=True,
        ...     use_ml_models=True,
        ...     model_path="/path/to/model.pt"
        ... )
        >>> enable_runtime_scheduler(config)
    """
    pass

def disable_runtime_scheduler() -> None:
    """Disable runtime scheduler."""
    pass

def get_runtime_scheduler_stats() -> Dict[str, Any]:
    """
    Get runtime scheduler statistics.
    
    Returns:
        Dictionary with statistics:
        - total_decisions: Total number of scheduling decisions
        - ml_decisions: Decisions made by ML models
        - fallback_decisions: Decisions made by fallback heuristics
        - avg_decision_time_us: Average decision time in microseconds
        - device_utilization: Per-device utilization
        - memory_utilization: Per-device memory utilization
    """
    pass

def reset_runtime_scheduler() -> None:
    """Reset runtime scheduler state and statistics."""
    pass

class RuntimeSchedulerContext:
    """Context manager for runtime scheduler."""
    
    def __init__(self, config: Optional[RuntimeSchedulerConfig] = None):
        self.config = config or RuntimeSchedulerConfig()
    
    def __enter__(self):
        enable_runtime_scheduler(self.config)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        disable_runtime_scheduler()
        return False

# Example usage
if __name__ == "__main__":
    # Enable globally
    enable_runtime_scheduler()
    
    # Or use context manager
    with RuntimeSchedulerContext() as scheduler:
        model = torch.nn.Linear(1000, 1000).cuda()
        x = torch.randn(128, 1000).cuda()
        y = model(x)
    
    # Get statistics
    stats = get_runtime_scheduler_stats()
    print(f"Total decisions: {stats['total_decisions']}")
    print(f"Average decision time: {stats['avg_decision_time_us']} μs")
```

---

## Next Steps

See:
- [03_ml_models.md](./03_ml_models.md) for ML model architectures
- [04_integration_guide.md](./04_integration_guide.md) for implementation details


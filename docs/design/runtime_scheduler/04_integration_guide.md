# Runtime Scheduler Integration Guide

## Overview

This document provides step-by-step implementation guidance for integrating the Runtime Scheduler into PyTorch.

---

## Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-4)
- [ ] Implement base scheduler interfaces
- [ ] Add hooks to Dispatcher
- [ ] Create state management system
- [ ] Implement fallback mechanisms

### Phase 2: Device & Stream Management (Weeks 5-8)
- [ ] Implement DeviceManager
- [ ] Implement StreamManager
- [ ] Add CUDA stream dependency tracking
- [ ] Implement multi-GPU support

### Phase 3: Memory Scheduling (Weeks 9-12)
- [ ] Integrate with CachingAllocator
- [ ] Implement eviction policies
- [ ] Add prefetching support
- [ ] Memory reuse optimization

### Phase 4: ML Models (Weeks 13-16)
- [ ] Implement fast path MLP
- [ ] Add GNN model
- [ ] Implement ensemble
- [ ] Training pipeline

### Phase 5: Testing & Optimization (Weeks 17-20)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Production readiness

---

## Detailed Implementation Steps

### Step 1: Core Scheduler Infrastructure

#### 1.1 Create Base Classes

```cpp
// File: torch/csrc/runtime_scheduler/RuntimeScheduler.h

#pragma once

#include <torch/csrc/Export.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <memory>
#include <chrono>
#include <vector>

namespace torch {
namespace runtime_scheduler {

// Forward declarations
class WorkloadScheduler;
class DeviceManager;
class StreamManager;
class MemoryScheduler;
class StateManager;

// Core scheduler implementation
class TORCH_API RuntimeScheduler {
 public:
  // Singleton access
  static RuntimeScheduler& singleton();
  
  // Configuration
  struct Config {
    bool enabled = false;
    bool use_ml_models = true;
    bool collect_feedback = true;
    float confidence_threshold = 0.75f;
    std::string model_path = "";
  };
  
  void initialize(const Config& config);
  void shutdown();
  
  // Main scheduling API
  SchedulingDecision schedule(
      const c10::OperatorHandle& op,
      const OperationMetadata& metadata);
  
  // State management
  void recordExecution(
      const c10::OperatorHandle& op,
      const SchedulingDecision& decision,
      std::chrono::microseconds actual_time);
  
  // Access sub-components
  WorkloadScheduler& workloadScheduler() { return *workload_scheduler_; }
  DeviceManager& deviceManager() { return *device_manager_; }
  StreamManager& streamManager() { return *stream_manager_; }
  MemoryScheduler& memoryScheduler() { return *memory_scheduler_; }
  StateManager& stateManager() { return *state_manager_; }
  
 private:
  RuntimeScheduler();
  ~RuntimeScheduler();
  
  // Sub-components
  std::unique_ptr<WorkloadScheduler> workload_scheduler_;
  std::unique_ptr<DeviceManager> device_manager_;
  std::unique_ptr<StreamManager> stream_manager_;
  std::unique_ptr<MemoryScheduler> memory_scheduler_;
  std::unique_ptr<StateManager> state_manager_;
  
  Config config_;
  bool initialized_ = false;
  
  // Thread safety
  mutable std::mutex mutex_;
};

} // namespace runtime_scheduler
} // namespace torch
```

```cpp
// File: torch/csrc/runtime_scheduler/RuntimeScheduler.cpp

#include <torch/csrc/runtime_scheduler/RuntimeScheduler.h>
#include <torch/csrc/runtime_scheduler/WorkloadScheduler.h>
#include <torch/csrc/runtime_scheduler/DeviceManager.h>
#include <torch/csrc/runtime_scheduler/StreamManager.h>
#include <torch/csrc/runtime_scheduler/MemoryScheduler.h>
#include <torch/csrc/runtime_scheduler/StateManager.h>

namespace torch {
namespace runtime_scheduler {

RuntimeScheduler& RuntimeScheduler::singleton() {
  static RuntimeScheduler instance;
  return instance;
}

RuntimeScheduler::RuntimeScheduler() {
  // Initialize sub-components
  workload_scheduler_ = std::make_unique<WorkloadScheduler>();
  device_manager_ = std::make_unique<DeviceManager>();
  stream_manager_ = std::make_unique<StreamManager>();
  memory_scheduler_ = std::make_unique<MemoryScheduler>();
  state_manager_ = std::make_unique<StateManager>();
}

RuntimeScheduler::~RuntimeScheduler() {
  shutdown();
}

void RuntimeScheduler::initialize(const Config& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (initialized_) {
    TORCH_WARN("RuntimeScheduler already initialized");
    return;
  }
  
  config_ = config;
  
  // Initialize sub-components
  workload_scheduler_->initialize();
  device_manager_->initialize();
  stream_manager_->initialize();
  memory_scheduler_->initialize();
  state_manager_->initialize();
  
  // Load ML models if enabled
  if (config_.use_ml_models && !config_.model_path.empty()) {
    // Load models (implementation in next section)
    loadMLModels(config_.model_path);
  }
  
  initialized_ = true;
  TORCH_INFO("RuntimeScheduler initialized successfully");
}

void RuntimeScheduler::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (!initialized_) {
    return;
  }
  
  // Cleanup sub-components
  state_manager_->shutdown();
  memory_scheduler_->shutdown();
  stream_manager_->shutdown();
  device_manager_->shutdown();
  workload_scheduler_->shutdown();
  
  initialized_ = false;
}

SchedulingDecision RuntimeScheduler::schedule(
    const c10::OperatorHandle& op,
    const OperationMetadata& metadata) {
  
  if (!initialized_ || !config_.enabled) {
    // Return default decision
    return SchedulingDecision::createDefault(metadata);
  }
  
  auto start = std::chrono::steady_clock::now();
  
  // Get current system state
  auto state = state_manager_->getSystemState();
  
  // Make scheduling decisions
  SchedulingDecision decision;
  
  // 1. Device placement
  decision.device = device_manager_->selectDevice(metadata, state);
  
  // 2. Stream assignment
  decision.stream = stream_manager_->assignStream(metadata, decision.device);
  
  // 3. Memory decisions
  decision.memory = memory_scheduler_->decideAllocation(
      metadata, decision.device);
  
  // 4. Priority and dependencies
  auto schedule_result = workload_scheduler_->schedule(metadata, state);
  decision.priority = schedule_result.priority;
  decision.dependencies = stream_manager_->computeDependencies(
      metadata, decision.stream);
  
  // Record decision time
  auto end = std::chrono::steady_clock::now();
  decision.decision_time = std::chrono::duration_cast<std::chrono::microseconds>(
      end - start);
  
  return decision;
}

void RuntimeScheduler::recordExecution(
    const c10::OperatorHandle& op,
    const SchedulingDecision& decision,
    std::chrono::microseconds actual_time) {
  
  if (!initialized_ || !config_.collect_feedback) {
    return;
  }
  
  // Record in state manager for learning
  state_manager_->recordExecution(op, decision, actual_time);
}

} // namespace runtime_scheduler
} // namespace torch
```

#### 1.2 Add Dispatcher Hooks

```cpp
// File: aten/src/ATen/core/dispatch/Dispatcher.h
// Add to existing Dispatcher class:

namespace c10 {

class TORCH_API Dispatcher final {
 public:
  // ... existing methods ...
  
  // Runtime scheduler hook points
  static void setRuntimeSchedulerEnabled(bool enabled);
  static bool isRuntimeSchedulerEnabled();
  
 private:
  // ... existing members ...
  
  static std::atomic<bool> runtime_scheduler_enabled_;
};

} // namespace c10
```

```cpp
// File: aten/src/ATen/core/dispatch/Dispatcher.cpp
// Modify the call() method:

template <class Return, class... Args>
Return Dispatcher::call(
    const TypedOperatorHandle<Return(Args...)>& op,
    Args... args) const {
  
  // HOOK POINT 1: Pre-dispatch scheduling
  if (runtime_scheduler_enabled_.load(std::memory_order_relaxed)) {
    auto& scheduler = torch::runtime_scheduler::RuntimeScheduler::singleton();
    
    // Extract operation metadata
    auto metadata = torch::runtime_scheduler::OperationMetadata::extract(
        op, std::forward<Args>(args)...);
    
    // Get scheduling decision
    auto decision = scheduler.schedule(op, metadata);
    
    // Apply decision
    decision.apply();
    
    // Continue with dispatch
    auto result = dispatchImpl<Return, Args...>(
        op, std::forward<Args>(args)...);
    
    // HOOK POINT 2: Post-dispatch recording
    auto exec_time = /* measure time */;
    scheduler.recordExecution(op, decision, exec_time);
    
    return result;
  }
  
  // Original dispatch path
  return dispatchImpl<Return, Args...>(op, std::forward<Args>(args)...);
}
```

### Step 2: Device Manager Implementation

```cpp
// File: torch/csrc/runtime_scheduler/DeviceManager.cpp

namespace torch {
namespace runtime_scheduler {

c10::Device DeviceManager::selectDevice(
    const OperationMetadata& op,
    const SystemState& state) {
  
  // Fast path: use current device if no better option
  if (!shouldConsiderDevicePlacement(op)) {
    return op.current_device;
  }
  
  // Get available devices
  auto available_devices = getAvailableDevices();
  
  if (available_devices.size() == 1) {
    return available_devices[0];
  }
  
  // ML-based device selection
  if (ml_model_ && ml_model_->isLoaded()) {
    auto features = extractDevicePlacementFeatures(op, state);
    auto device_scores = ml_model_->predict(features);
    
    // Select device with highest score
    int best_device_idx = argmax(device_scores);
    return available_devices[best_device_idx];
  }
  
  // Fallback: load balancing heuristic
  return selectDeviceByLoadBalancing(available_devices, state);
}

bool DeviceManager::shouldConsiderDevicePlacement(
    const OperationMetadata& op) const {
  // Don't move small ops
  if (op.estimated_flops < 1e6) {
    return false;
  }
  
  // Don't move if inputs are already on device
  bool all_inputs_on_device = true;
  for (const auto& device : op.input_devices) {
    if (device != op.current_device) {
      all_inputs_on_device = false;
      break;
    }
  }
  
  if (all_inputs_on_device) {
    return false;
  }
  
  return true;
}

c10::Device DeviceManager::selectDeviceByLoadBalancing(
    const std::vector<c10::Device>& devices,
    const SystemState& state) const {
  
  // Select least loaded device
  c10::Device best_device = devices[0];
  float min_utilization = 1.0f;
  
  for (const auto& device : devices) {
    auto device_state = getDeviceState(device);
    if (device_state.utilization < min_utilization) {
      min_utilization = device_state.utilization;
      best_device = device;
    }
  }
  
  return best_device;
}

} // namespace runtime_scheduler
} // namespace torch
```

### Step 3: Stream Manager Implementation

```cpp
// File: torch/csrc/runtime_scheduler/StreamManager.cpp

namespace torch {
namespace runtime_scheduler {

c10::cuda::CUDAStream StreamManager::assignStream(
    const OperationMetadata& op,
    c10::Device device) {
  
  // Get device's stream pool
  auto& pool = getStreamPool(device);
  
  // Check if op can use default stream
  if (canUseDefaultStream(op)) {
    return c10::cuda::getDefaultCUDAStream(device.index());
  }
  
  // ML-based stream selection
  if (ml_model_ && ml_model_->isLoaded()) {
    auto features = extractStreamAssignmentFeatures(op, device);
    auto stream_scores = ml_model_->predict(features);
    
    int best_stream_idx = argmax(stream_scores);
    return pool.streams[best_stream_idx];
  }
  
  // Fallback: round-robin
  return pool.getNextStream();
}

std::vector<StreamDependency> StreamManager::computeDependencies(
    const OperationMetadata& op,
    c10::cuda::CUDAStream assigned_stream) {
  
  std::vector<StreamDependency> dependencies;
  
  // Find operations that produce inputs for this op
  for (const auto& input_name : op.input_tensor_names) {
    if (tensor_to_stream_.count(input_name)) {
      auto producer_stream = tensor_to_stream_[input_name];
      
      // If different stream, add dependency
      if (producer_stream != assigned_stream) {
        StreamDependency dep;
        dep.src_stream = producer_stream;
        dep.dst_stream = assigned_stream;
        dep.event = getOrCreateEvent(producer_stream);
        
        dependencies.push_back(dep);
      }
    }
  }
  
  return dependencies;
}

void StreamManager::insertDependencies(
    const std::vector<StreamDependency>& deps) {
  
  for (const auto& dep : deps) {
    // Record event on source stream
    C10_CUDA_CHECK(cudaEventRecord(
        dep.event,
        dep.src_stream
    ));
    
    // Wait on destination stream
    C10_CUDA_CHECK(cudaStreamWaitEvent(
        dep.dst_stream,
        dep.event,
        0
    ));
  }
}

} // namespace runtime_scheduler
} // namespace torch
```

### Step 4: Memory Scheduler Integration

```cpp
// File: torch/csrc/runtime_scheduler/MemoryScheduler.cpp

namespace torch {
namespace runtime_scheduler {

AllocationDecision MemoryScheduler::decideAllocation(
    const OperationMetadata& op,
    c10::Device device) {
  
  AllocationDecision decision;
  
  // Estimate output size
  size_t output_size = estimateOutputSize(op);
  decision.allocation_size = output_size;
  
  // Check if we have enough memory
  auto memory_state = getMemoryState(device);
  bool has_memory = (memory_state.available_memory > output_size);
  
  if (!has_memory) {
    // Need to evict
    decision.should_evict = true;
    decision.evict_candidates = selectEvictionCandidates(
        output_size, device);
  }
  
  // Check if we should prefetch inputs
  decision.use_prefetch = shouldPrefetch(op, device);
  
  return decision;
}

std::vector<at::Tensor> MemoryScheduler::selectEvictionCandidates(
    size_t required_bytes,
    c10::Device device) {
  
  std::vector<at::Tensor> candidates;
  
  // ML-based eviction
  if (ml_model_ && ml_model_->isLoaded()) {
    auto features = extractEvictionFeatures(device);
    auto scores = ml_model_->predictEvictionScores(features);
    
    // Sort tensors by eviction score
    auto sorted_tensors = sortByScore(getAllTensors(device), scores);
    
    // Select tensors until we have enough space
    size_t freed_bytes = 0;
    for (const auto& tensor : sorted_tensors) {
      candidates.push_back(tensor);
      freed_bytes += tensor.nbytes();
      
      if (freed_bytes >= required_bytes) {
        break;
      }
    }
  } else {
    // Fallback: LRU
    candidates = selectLRUCandidates(required_bytes, device);
  }
  
  return candidates;
}

} // namespace runtime_scheduler
} // namespace torch
```

### Step 5: Python Bindings

```cpp
// File: torch/csrc/runtime_scheduler/python_bindings.cpp

#include <torch/csrc/runtime_scheduler/RuntimeScheduler.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace runtime_scheduler {

void initRuntimeSchedulerBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  
  auto runtime_scheduler = m.def_submodule("_runtime_scheduler");
  
  // Config
  py::class_<RuntimeScheduler::Config>(runtime_scheduler, "Config")
      .def(py::init<>())
      .def_readwrite("enabled", &RuntimeScheduler::Config::enabled)
      .def_readwrite("use_ml_models", &RuntimeScheduler::Config::use_ml_models)
      .def_readwrite("collect_feedback", &RuntimeScheduler::Config::collect_feedback)
      .def_readwrite("confidence_threshold", &RuntimeScheduler::Config::confidence_threshold)
      .def_readwrite("model_path", &RuntimeScheduler::Config::model_path);
  
  // Functions
  runtime_scheduler.def(
      "enable",
      [](const RuntimeScheduler::Config& config) {
        RuntimeScheduler::singleton().initialize(config);
        c10::Dispatcher::setRuntimeSchedulerEnabled(true);
      },
      py::arg("config") = RuntimeScheduler::Config());
  
  runtime_scheduler.def(
      "disable",
      []() {
        c10::Dispatcher::setRuntimeSchedulerEnabled(false);
        RuntimeScheduler::singleton().shutdown();
      });
  
  runtime_scheduler.def(
      "is_enabled",
      []() {
        return c10::Dispatcher::isRuntimeSchedulerEnabled();
      });
  
  runtime_scheduler.def(
      "get_stats",
      []() {
        auto stats = RuntimeScheduler::singleton().getStats();
        py::dict result;
        result["total_decisions"] = stats.total_decisions;
        result["ml_decisions"] = stats.ml_decisions;
        result["fallback_decisions"] = stats.fallback_decisions;
        result["avg_decision_time_us"] = stats.avg_decision_time.count();
        return result;
      });
}

} // namespace runtime_scheduler
} // namespace torch
```

```python
# File: torch/runtime_scheduler/__init__.py

from torch._C import _runtime_scheduler
from typing import Optional

class RuntimeSchedulerConfig:
    """Configuration for runtime scheduler."""
    
    def __init__(
        self,
        enabled: bool = False,
        use_ml_models: bool = True,
        collect_feedback: bool = True,
        confidence_threshold: float = 0.75,
        model_path: Optional[str] = None
    ):
        self._config = _runtime_scheduler.Config()
        self._config.enabled = enabled
        self._config.use_ml_models = use_ml_models
        self._config.collect_feedback = collect_feedback
        self._config.confidence_threshold = confidence_threshold
        if model_path:
            self._config.model_path = model_path

def enable(config: Optional[RuntimeSchedulerConfig] = None):
    """Enable runtime scheduler."""
    if config is None:
        config = RuntimeSchedulerConfig(enabled=True)
    _runtime_scheduler.enable(config._config)

def disable():
    """Disable runtime scheduler."""
    _runtime_scheduler.disable()

def is_enabled() -> bool:
    """Check if runtime scheduler is enabled."""
    return _runtime_scheduler.is_enabled()

def get_stats() -> dict:
    """Get runtime scheduler statistics."""
    return _runtime_scheduler.get_stats()
```

---

## Testing Strategy

### Unit Tests

```python
# File: test/test_runtime_scheduler.py

import torch
import torch.runtime_scheduler as rs
import unittest

class TestRuntimeScheduler(unittest.TestCase):
    
    def setUp(self):
        rs.enable()
    
    def tearDown(self):
        rs.disable()
    
    def test_basic_operation(self):
        """Test that runtime scheduler doesn't break basic ops."""
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        z = x + y
        
        # Should execute successfully
        self.assertEqual(z.shape, (100, 100))
    
    def test_multi_device(self):
        """Test multi-device scheduling."""
        if torch.cuda.device_count() < 2:
            self.skipTest("Need 2+ GPUs")
        
        x = torch.randn(1000, 1000, device='cuda:0')
        y = torch.randn(1000, 1000, device='cuda:1')
        
        # Runtime scheduler should handle this
        z = x.to('cuda:1') + y
        self.assertEqual(z.device.index, 1)
    
    def test_stats_collection(self):
        """Test that stats are collected."""
        x = torch.randn(100, 100, device='cuda')
        y = x + 1
        
        stats = rs.get_stats()
        self.assertGreater(stats['total_decisions'], 0)
```

### Integration Tests

```python
# File: test/test_runtime_scheduler_integration.py

import torch
import torch.nn as nn
import torch.runtime_scheduler as rs

class TestRuntimeSchedulerIntegration(unittest.TestCase):
    
    def test_simple_model(self):
        """Test with a simple model."""
        rs.enable()
        
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).cuda()
        
        x = torch.randn(32, 100, device='cuda')
        y = model(x)
        
        self.assertEqual(y.shape, (32, 10))
        
        stats = rs.get_stats()
        self.assertGreater(stats['total_decisions'], 0)
        
        rs.disable()
    
    def test_resnet_training(self):
        """Test with ResNet training."""
        import torchvision.models as models
        
        rs.enable()
        
        model = models.resnet18().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Training loop
        for _ in range(10):
            x = torch.randn(4, 3, 224, 224, device='cuda')
            target = torch.randint(0, 1000, (4,), device='cuda')
            
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        rs.disable()
```

---

## Deployment Checklist

- [ ] Code Review
  - [ ] Core infrastructure reviewed
  - [ ] Performance impact assessed
  - [ ] Memory safety verified

- [ ] Testing
  - [ ] Unit tests passing
  - [ ] Integration tests passing
  - [ ] Performance benchmarks run
  - [ ] Memory leak tests passing

- [ ] Documentation
  - [ ] API documentation complete
  - [ ] User guide written
  - [ ] Example code provided

- [ ] Performance
  - [ ] Overhead < 1ms per operation
  - [ ] No regression on existing workloads
  - [ ] Speedup demonstrated on target workloads

- [ ] Safety
  - [ ] Correctness validation passing
  - [ ] Fallback mechanisms tested
  - [ ] Error handling comprehensive

---

## Next Steps

- [05_performance_analysis.md](./05_performance_analysis.md) - Performance benchmarks
- [06_safety_mechanisms.md](./06_safety_mechanisms.md) - Safety validation


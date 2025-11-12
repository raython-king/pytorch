# Runtime Scheduler Safety Mechanisms

## Overview

This document describes safety mechanisms, validation strategies, and error handling for the Runtime Scheduler to ensure correctness and reliability.

---

## Safety Principles

### Core Principles

1. **Correctness First**: Never sacrifice correctness for performance
2. **Fail Safe**: When in doubt, fall back to safe defaults
3. **Transparent**: User should be able to disable scheduler at any time
4. **Debuggable**: Provide clear diagnostics when issues occur
5. **Incremental**: Can be enabled/disabled per-component

---

## Validation Mechanisms

### 1. Decision Validation

```cpp
namespace torch {
namespace runtime_scheduler {

class DecisionValidator {
 public:
  // Validate scheduling decision before applying
  struct ValidationResult {
    bool is_valid;
    std::string error_message;
    std::vector<std::string> warnings;
  };
  
  ValidationResult validate(
      const SchedulingDecision& decision,
      const OperationMetadata& op) {
    
    ValidationResult result{true, "", {}};
    
    // 1. Device validation
    if (!validateDevice(decision, op)) {
      result.is_valid = false;
      result.error_message = "Invalid device placement";
      return result;
    }
    
    // 2. Stream validation
    if (!validateStream(decision, op)) {
      result.is_valid = false;
      result.error_message = "Invalid stream assignment";
      return result;
    }
    
    // 3. Memory validation
    if (!validateMemory(decision, op)) {
      result.is_valid = false;
      result.error_message = "Invalid memory decision";
      return result;
    }
    
    // 4. Dependency validation
    if (!validateDependencies(decision, op)) {
      result.is_valid = false;
      result.error_message = "Invalid dependencies (potential deadlock)";
      return result;
    }
    
    return result;
  }
  
 private:
  bool validateDevice(
      const SchedulingDecision& decision,
      const OperationMetadata& op) {
    // Check device exists
    if (decision.device.index() >= torch::cuda::device_count()) {
      return false;
    }
    
    // Check device is available
    if (!isDeviceAvailable(decision.device)) {
      return false;
    }
    
    // Check memory availability
    auto memory_info = getCUDAMemoryInfo(decision.device);
    if (memory_info.available < decision.memory.allocation_size) {
      return false;
    }
    
    return true;
  }
  
  bool validateStream(
      const SchedulingDecision& decision,
      const OperationMetadata& op) {
    // Check stream is valid for device
    if (decision.stream.device() != decision.device) {
      return false;
    }
    
    // Check stream is not destroyed
    if (!isStreamValid(decision.stream)) {
      return false;
    }
    
    return true;
  }
  
  bool validateMemory(
      const SchedulingDecision& decision,
      const OperationMetadata& op) {
    // Check eviction candidates exist
    for (const auto& tensor : decision.memory.evict_candidates) {
      if (!tensor.defined()) {
        return false;
      }
    }
    
    // Check allocation size is reasonable
    if (decision.memory.allocation_size > getMaxAllocationSize()) {
      return false;
    }
    
    return true;
  }
  
  bool validateDependencies(
      const SchedulingDecision& decision,
      const OperationMetadata& op) {
    // Check for circular dependencies
    if (hasCycle(decision.dependencies)) {
      return false;
    }
    
    // Check all dependency events are valid
    for (const auto& event : decision.dependencies) {
      if (!isEventValid(event)) {
        return false;
      }
    }
    
    return true;
  }
  
  bool hasCycle(const std::vector<c10::cuda::CUDAEvent>& deps) {
    // Use DFS to detect cycles in stream dependency graph
    // ... implementation
    return false;
  }
};

} // namespace runtime_scheduler
} // namespace torch
```

### 2. Correctness Verification

```cpp
class CorrectnessVerifier {
 public:
  // Verify operation produces correct result
  bool verifyOperation(
      const c10::OperatorHandle& op,
      const SchedulingDecision& decision,
      const at::Tensor& result) {
    
    // Shadow execution: run operation with default scheduler
    auto reference_result = executeWithDefaultScheduler(op);
    
    // Compare results
    return tensorEquals(result, reference_result);
  }
  
 private:
  bool tensorEquals(
      const at::Tensor& a,
      const at::Tensor& b,
      float rtol = 1e-5,
      float atol = 1e-8) {
    
    if (a.sizes() != b.sizes()) return false;
    if (a.dtype() != b.dtype()) return false;
    
    // Numerical comparison
    auto diff = (a - b).abs();
    auto max_diff = diff.max().item<float>();
    auto max_val = a.abs().max().item<float>();
    
    return max_diff <= (atol + rtol * max_val);
  }
};
```

---

## Fallback Mechanisms

### 1. Multi-Level Fallback

```
Decision Flow with Fallback:

┌─────────────────────────────────────┐
│  1. Try ML Model                    │
│     └─> If confidence > threshold   │
│         Use ML decision ✓           │
└──────────────┬──────────────────────┘
               │
               │ Confidence too low
               ▼
┌─────────────────────────────────────┐
│  2. Try Heuristic                   │
│     └─> Use rule-based heuristic    │
└──────────────┬──────────────────────┘
               │
               │ Heuristic uncertain
               ▼
┌─────────────────────────────────────┐
│  3. Use Safe Default                │
│     └─> Use current device/stream   │
│         No optimization             │
└─────────────────────────────────────┘
```

```cpp
class FallbackDecider {
 public:
  SchedulingDecision decideWithFallback(
      const OperationMetadata& op,
      const SystemState& state) {
    
    // Level 1: Try ML model
    if (ml_model_ && ml_model_->isLoaded()) {
      auto decision = ml_model_->decide(op, state);
      
      if (decision.confidence > confidence_threshold_) {
        decision.use_fallback = false;
        return decision;
      }
    }
    
    // Level 2: Try heuristic
    auto heuristic_decision = heuristic_.decide(op, state);
    if (heuristic_decision.is_confident) {
      heuristic_decision.use_fallback = true;
      return heuristic_decision;
    }
    
    // Level 3: Safe default
    return SchedulingDecision::createDefault(op);
  }
};
```

### 2. Gradual Rollback

```cpp
class GradualRollback {
 public:
  // If errors occur, gradually disable components
  void handleError(const std::string& component, const std::exception& e) {
    error_count_[component]++;
    
    if (error_count_[component] > max_errors_) {
      LOG(WARNING) << "Disabling component " << component 
                   << " due to repeated errors: " << e.what();
      
      disableComponent(component);
    }
  }
  
 private:
  void disableComponent(const std::string& component) {
    if (component == "ml_model") {
      // Disable ML, use heuristics
      RuntimeScheduler::singleton().disableMLModels();
    } else if (component == "device_placement") {
      // Disable device placement
      RuntimeScheduler::singleton().deviceManager().disable();
    }
    // ... etc
  }
  
  std::unordered_map<std::string, int> error_count_;
  int max_errors_ = 10;
};
```

---

## Error Handling

### 1. Exception Handling

```cpp
SchedulingDecision RuntimeScheduler::schedule(
    const c10::OperatorHandle& op,
    const OperationMetadata& metadata) {
  
  try {
    // Normal scheduling path
    return scheduleImpl(op, metadata);
    
  } catch (const c10::Error& e) {
    // PyTorch error
    LOG(ERROR) << "PyTorch error in scheduler: " << e.what();
    rollback_manager_.handleError("scheduler", e);
    return SchedulingDecision::createDefault(metadata);
    
  } catch (const std::exception& e) {
    // Standard exception
    LOG(ERROR) << "Exception in scheduler: " << e.what();
    rollback_manager_.handleError("scheduler", e);
    return SchedulingDecision::createDefault(metadata);
    
  } catch (...) {
    // Unknown exception
    LOG(ERROR) << "Unknown exception in scheduler";
    return SchedulingDecision::createDefault(metadata);
  }
}
```

### 2. Timeout Protection

```cpp
class TimeoutProtection {
 public:
  template<typename Func, typename... Args>
  auto executeWithTimeout(
      std::chrono::microseconds timeout,
      Func&& func,
      Args&&... args) -> std::optional<decltype(func(args...))> {
    
    std::promise<decltype(func(args...))> promise;
    auto future = promise.get_future();
    
    // Execute in separate thread
    std::thread worker([&]() {
      try {
        auto result = func(std::forward<Args>(args)...);
        promise.set_value(std::move(result));
      } catch (...) {
        promise.set_exception(std::current_exception());
      }
    });
    
    // Wait with timeout
    if (future.wait_for(timeout) == std::future_status::timeout) {
      // Timeout occurred
      worker.detach();  // Let thread finish in background
      return std::nullopt;
    }
    
    worker.join();
    
    try {
      return future.get();
    } catch (...) {
      return std::nullopt;
    }
  }
};

// Usage
auto decision = timeout_protection_.executeWithTimeout(
    std::chrono::microseconds(500),  // 500 μs timeout
    [&]() { return ml_model_->predict(features); }
);

if (!decision.has_value()) {
  // Timeout: fall back to heuristic
  return heuristic_.decide(op, state);
}
```

---

## Deadlock Prevention

### 1. Stream Dependency DAG

```cpp
class StreamDependencyTracker {
 public:
  // Check if adding dependency would create cycle
  bool wouldCreateCycle(
      c10::cuda::CUDAStream from,
      c10::cuda::CUDAStream to) {
    
    // Build dependency graph
    std::unordered_map<c10::cuda::CUDAStream, 
                       std::vector<c10::cuda::CUDAStream>> graph;
    
    for (const auto& [src, dst] : dependencies_) {
      graph[src].push_back(dst);
    }
    
    // Add proposed edge
    graph[from].push_back(to);
    
    // Check for cycle using DFS
    return hasCycle(graph);
  }
  
  void addDependency(
      c10::cuda::CUDAStream from,
      c10::cuda::CUDAStream to) {
    
    if (wouldCreateCycle(from, to)) {
      throw std::runtime_error(
          "Cannot add dependency: would create cycle (deadlock)");
    }
    
    dependencies_.emplace_back(from, to);
  }
  
 private:
  bool hasCycle(
      const std::unordered_map<c10::cuda::CUDAStream,
                                std::vector<c10::cuda::CUDAStream>>& graph) {
    
    std::unordered_set<c10::cuda::CUDAStream> visited;
    std::unordered_set<c10::cuda::CUDAStream> rec_stack;
    
    for (const auto& [node, _] : graph) {
      if (hasCycleUtil(node, graph, visited, rec_stack)) {
        return true;
      }
    }
    
    return false;
  }
  
  bool hasCycleUtil(
      c10::cuda::CUDAStream node,
      const std::unordered_map<c10::cuda::CUDAStream,
                                std::vector<c10::cuda::CUDAStream>>& graph,
      std::unordered_set<c10::cuda::CUDAStream>& visited,
      std::unordered_set<c10::cuda::CUDAStream>& rec_stack) {
    
    if (!visited.count(node)) {
      visited.insert(node);
      rec_stack.insert(node);
      
      if (graph.count(node)) {
        for (const auto& neighbor : graph.at(node)) {
          if (!visited.count(neighbor) && 
              hasCycleUtil(neighbor, graph, visited, rec_stack)) {
            return true;
          } else if (rec_stack.count(neighbor)) {
            return true;
          }
        }
      }
    }
    
    rec_stack.erase(node);
    return false;
  }
  
  std::vector<std::pair<c10::cuda::CUDAStream, c10::cuda::CUDAStream>> dependencies_;
};
```

### 2. Resource Deadlock Prevention

```cpp
class ResourceDeadlockPrevention {
 public:
  // Prevent memory deadlock (all devices full)
  void preventMemoryDeadlock(
      const SchedulingDecision& decision,
      const OperationMetadata& op) {
    
    // Check if allocation would succeed
    auto memory_state = getMemoryState(decision.device);
    
    if (memory_state.available_memory < decision.memory.allocation_size) {
      // Need to free memory
      if (decision.memory.evict_candidates.empty()) {
        // No eviction candidates: this is a problem
        throw std::runtime_error(
            "Memory deadlock: cannot allocate and no eviction candidates");
      }
      
      // Ensure eviction would free enough memory
      size_t eviction_bytes = 0;
      for (const auto& tensor : decision.memory.evict_candidates) {
        eviction_bytes += tensor.nbytes();
      }
      
      if (eviction_bytes < decision.memory.allocation_size) {
        throw std::runtime_error(
            "Memory deadlock: eviction would not free enough memory");
      }
    }
  }
};
```

---

## Testing and Validation

### 1. Shadow Mode

```cpp
class ShadowMode {
 public:
  // Run scheduler in shadow mode (compare with baseline)
  void runShadowMode(const c10::OperatorHandle& op,
                     const OperationMetadata& metadata) {
    
    // Get scheduler decision
    auto scheduler_decision = scheduler_.schedule(op, metadata);
    
    // Execute with scheduler
    auto scheduler_result = executeWithScheduler(op, scheduler_decision);
    
    // Execute with default (baseline)
    auto baseline_result = executeWithDefault(op);
    
    // Compare results
    if (!tensorEquals(scheduler_result, baseline_result)) {
      LOG(ERROR) << "Shadow mode: result mismatch for op " << op.operator_name();
      
      // Log details
      logMismatch(op, scheduler_decision, scheduler_result, baseline_result);
      
      // Optionally disable scheduler
      if (config_.strict_mode) {
        scheduler_.disable();
      }
    }
  }
};
```

### 2. Stress Testing

```cpp
class StressTest {
 public:
  // Test scheduler under heavy load
  void testUnderLoad() {
    const int num_threads = 16;
    const int ops_per_thread = 10000;
    
    std::vector<std::thread> threads;
    std::atomic<int> errors{0};
    
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&, i]() {
        for (int j = 0; j < ops_per_thread; ++j) {
          try {
            // Random operation
            auto op = generateRandomOp();
            auto decision = scheduler_.schedule(op);
            decision.apply();
          } catch (...) {
            errors++;
          }
        }
      });
    }
    
    for (auto& thread : threads) {
      thread.join();
    }
    
    double error_rate = double(errors) / (num_threads * ops_per_thread);
    EXPECT_LT(error_rate, 0.001) << "Error rate too high under load";
  }
  
  // Test memory pressure
  void testMemoryPressure() {
    // Allocate tensors until near OOM
    std::vector<at::Tensor> tensors;
    
    while (true) {
      try {
        auto tensor = torch::randn({1000, 1000}, torch::kCUDA);
        tensors.push_back(tensor);
        
        // Run scheduler operation
        auto op = generateRandomOp();
        auto decision = scheduler_.schedule(op);
        decision.apply();
        
      } catch (const c10::Error& e) {
        // Expected: out of memory
        break;
      }
    }
    
    // Verify no crashes
    EXPECT_GT(tensors.size(), 0);
  }
};
```

### 3. Correctness Tests

```python
# File: test/test_runtime_scheduler_correctness.py

import torch
import torch.runtime_scheduler as rs
import unittest

class TestSchedulerCorrectness(unittest.TestCase):
    
    def test_numerical_correctness(self):
        """Test that scheduler produces correct numerical results."""
        rs.enable()
        
        # Test various operations
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        
        # Compute with scheduler
        z_scheduler = x @ y + x
        
        # Compute baseline
        rs.disable()
        z_baseline = x @ y + x
        
        # Compare
        torch.testing.assert_close(z_scheduler, z_baseline, rtol=1e-5, atol=1e-8)
    
    def test_gradient_correctness(self):
        """Test that gradients are correct with scheduler."""
        rs.enable()
        
        x = torch.randn(100, 100, device='cuda', requires_grad=True)
        y = torch.randn(100, 100, device='cuda', requires_grad=True)
        
        # Forward + backward with scheduler
        z = (x @ y).sum()
        z.backward()
        
        grad_x_scheduler = x.grad.clone()
        grad_y_scheduler = y.grad.clone()
        
        # Reset
        x.grad = None
        y.grad = None
        
        # Forward + backward baseline
        rs.disable()
        z = (x @ y).sum()
        z.backward()
        
        grad_x_baseline = x.grad
        grad_y_baseline = y.grad
        
        # Compare gradients
        torch.testing.assert_close(grad_x_scheduler, grad_x_baseline)
        torch.testing.assert_close(grad_y_scheduler, grad_y_baseline)
    
    def test_multi_device_correctness(self):
        """Test correctness with multi-device operations."""
        if torch.cuda.device_count() < 2:
            self.skipTest("Need 2+ GPUs")
        
        rs.enable()
        
        x = torch.randn(100, 100, device='cuda:0')
        y = torch.randn(100, 100, device='cuda:1')
        
        # Operation crossing devices
        z = x.to('cuda:1') + y
        
        # Should be correct
        self.assertEqual(z.device.index, 1)
        self.assertEqual(z.shape, (100, 100))
```

---

## Monitoring and Debugging

### 1. Logging

```cpp
// Comprehensive logging for debugging
class SchedulerLogger {
 public:
  void logDecision(
      const c10::OperatorHandle& op,
      const SchedulingDecision& decision) {
    
    if (log_level_ >= LogLevel::DEBUG) {
      LOG(INFO) << "Scheduling decision for " << op.operator_name() << ":"
                << "\n  Device: " << decision.device
                << "\n  Stream: " << decision.stream.id()
                << "\n  Priority: " << decision.priority
                << "\n  Confidence: " << decision.confidence
                << "\n  Decision time: " << decision.decision_time.count() << " μs"
                << "\n  Fallback: " << (decision.use_fallback ? "yes" : "no");
    }
  }
  
  void logError(
      const c10::OperatorHandle& op,
      const std::exception& e) {
    
    LOG(ERROR) << "Error scheduling " << op.operator_name() << ": " << e.what();
    
    // Log system state for debugging
    auto state = state_manager_->getSystemState();
    LOG(ERROR) << "System state at error:"
               << "\n  Device utilization: " << state.device_utilization
               << "\n  Memory available: " << state.memory_available
               << "\n  Pending ops: " << state.pending_ops;
  }
};
```

### 2. Metrics

```cpp
class SchedulerMetrics {
 public:
  void recordDecision(const SchedulingDecision& decision) {
    total_decisions_++;
    
    if (decision.use_fallback) {
      fallback_decisions_++;
    } else {
      ml_decisions_++;
    }
    
    decision_time_sum_ += decision.decision_time;
    
    // Histogram
    decision_time_histogram_[getBucket(decision.decision_time)]++;
  }
  
  void report() {
    LOG(INFO) << "Scheduler Metrics:"
              << "\n  Total decisions: " << total_decisions_
              << "\n  ML decisions: " << ml_decisions_
              << "\n  Fallback decisions: " << fallback_decisions_
              << "\n  Avg decision time: " 
              << (decision_time_sum_ / total_decisions_).count() << " μs"
              << "\n  ML rate: " 
              << (100.0 * ml_decisions_ / total_decisions_) << "%";
  }
};
```

---

## Summary

### Safety Checklist

- [x] Decision validation before application
- [x] Correctness verification (shadow mode)
- [x] Multi-level fallback mechanism
- [x] Exception handling and recovery
- [x] Timeout protection
- [x] Deadlock prevention
- [x] Resource leak prevention
- [x] Comprehensive logging
- [x] Metrics and monitoring
- [x] Stress testing
- [x] Correctness tests

### Key Takeaways

1. **Never trust ML models blindly** - always validate
2. **Fail gracefully** - fall back to safe defaults
3. **Test extensively** - correctness, performance, stress
4. **Monitor in production** - metrics, logging, alerts
5. **Make it debuggable** - comprehensive logging and diagnostics


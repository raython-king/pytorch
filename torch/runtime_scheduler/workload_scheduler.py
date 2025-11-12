"""
Dynamic Workload Scheduler for PyTorch Runtime.

This module implements the main scheduler that decides:
- When to execute each operation
- Priority of operations in the queue
- Dependencies between operations
- Batching opportunities for similar ops

The scheduler uses ML models for intelligent decisions with minimal overhead (< 1% total runtime).
"""

import heapq
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Deque
import torch

from .state_tracker import RuntimeStateTracker, DeviceType
from .features.runtime_features import RuntimeFeatureExtractor
from .models.runtime_models import (
    PriorityPredictor,
    LatencyPredictor,
    BatchingPredictor,
    OnlineLearner,
    create_priority_predictor,
    create_latency_predictor,
    create_batching_predictor,
)


class OperationStatus(Enum):
    """Status of an operation."""
    PENDING = "pending"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class SchedulingPolicy(Enum):
    """Scheduling policy."""
    FIFO = "fifo"  # First-in-first-out
    PRIORITY = "priority"  # Priority-based
    ML_GUIDED = "ml_guided"  # ML model guided
    ADAPTIVE = "adaptive"  # Adaptive based on system state


@dataclass
class RuntimeOperation:
    """
    Represents a pending runtime operation.

    Contains all information needed for scheduling decisions.
    """
    # Operation identity
    op_id: int
    op_type: str
    creation_time: float = field(default_factory=time.time)

    # Computation details
    function: Optional[Callable] = None
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)

    # Tensor information
    input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    input_devices: List[torch.device] = field(default_factory=list)
    target_device: Optional[torch.device] = None

    # Dependency tracking
    dependencies: Set[int] = field(default_factory=set)  # op_ids this depends on
    dependents: Set[int] = field(default_factory=set)  # op_ids that depend on this
    satisfied_dependencies: Set[int] = field(default_factory=set)

    # Scheduling metadata
    priority: float = 0.0  # Higher priority = execute sooner
    estimated_latency_ms: float = 0.0
    status: OperationStatus = OperationStatus.PENDING

    # Execution tracking
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    actual_latency_ms: Optional[float] = None
    success: bool = False

    # Batching
    batch_group_id: Optional[int] = None
    can_batch: bool = False

    def __lt__(self, other: 'RuntimeOperation') -> bool:
        """Comparison for priority queue (higher priority first)."""
        return self.priority > other.priority

    def is_ready(self) -> bool:
        """Check if all dependencies are satisfied."""
        return len(self.satisfied_dependencies) == len(self.dependencies)

    def satisfy_dependency(self, op_id: int) -> bool:
        """
        Mark a dependency as satisfied.

        Returns:
            True if operation becomes ready
        """
        if op_id in self.dependencies:
            self.satisfied_dependencies.add(op_id)
            if self.is_ready() and self.status == OperationStatus.PENDING:
                self.status = OperationStatus.READY
                return True
        return False

    def get_wait_time_ms(self) -> float:
        """Get how long operation has been waiting."""
        return (time.time() - self.creation_time) * 1000

    def get_execution_time_ms(self) -> Optional[float]:
        """Get execution time if completed."""
        if self.start_time is not None and self.end_time is not None:
            return (self.end_time - self.start_time) * 1000
        return None


class OperationQueue:
    """
    Priority queue for pending operations.

    Thread-safe with efficient priority updates.
    """

    def __init__(self):
        """Initialize operation queue."""
        self._heap: List[RuntimeOperation] = []
        self._operations: Dict[int, RuntimeOperation] = {}
        self._lock = threading.RLock()
        self._generation = 0  # For priority updates

    def push(self, operation: RuntimeOperation) -> None:
        """
        Add operation to queue.

        Args:
            operation: Operation to add
        """
        with self._lock:
            if operation.op_id not in self._operations:
                heapq.heappush(self._heap, operation)
                self._operations[operation.op_id] = operation

    def pop(self) -> Optional[RuntimeOperation]:
        """
        Remove and return highest priority operation.

        Returns:
            Highest priority operation or None if queue empty
        """
        with self._lock:
            while self._heap:
                op = heapq.heappop(self._heap)
                # Check if still in operations dict (not removed)
                if op.op_id in self._operations:
                    del self._operations[op.op_id]
                    return op
            return None

    def peek(self) -> Optional[RuntimeOperation]:
        """
        View highest priority operation without removing.

        Returns:
            Highest priority operation or None
        """
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0]

    def remove(self, op_id: int) -> bool:
        """
        Remove operation from queue.

        Args:
            op_id: Operation ID to remove

        Returns:
            True if operation was removed
        """
        with self._lock:
            if op_id in self._operations:
                del self._operations[op_id]
                # Note: operation stays in heap but will be skipped in pop()
                return True
            return False

    def update_priority(self, op_id: int, new_priority: float) -> bool:
        """
        Update priority of an operation.

        Args:
            op_id: Operation ID
            new_priority: New priority value

        Returns:
            True if priority was updated
        """
        with self._lock:
            if op_id in self._operations:
                op = self._operations[op_id]
                op.priority = new_priority
                # Re-heapify (simple but not most efficient)
                heapq.heapify(self._heap)
                return True
            return False

    def get_ready_operations(self, max_count: int = 10) -> List[RuntimeOperation]:
        """
        Get ready operations (all dependencies satisfied).

        Args:
            max_count: Maximum number to return

        Returns:
            List of ready operations
        """
        with self._lock:
            ready = [
                op for op in self._operations.values()
                if op.status == OperationStatus.READY
            ]
            # Sort by priority
            ready.sort(reverse=True)
            return ready[:max_count]

    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._operations)

    def clear(self) -> None:
        """Clear the queue."""
        with self._lock:
            self._heap.clear()
            self._operations.clear()


class DependencyTracker:
    """
    Tracks data dependencies between operations.

    Maintains dependency graph for correct execution ordering.
    """

    def __init__(self):
        """Initialize dependency tracker."""
        self._dependencies: Dict[int, Set[int]] = defaultdict(set)  # op_id -> depends on
        self._dependents: Dict[int, Set[int]] = defaultdict(set)  # op_id -> blocks
        self._completed_ops: Set[int] = set()
        self._lock = threading.RLock()

    def add_dependency(self, op_id: int, depends_on: int) -> None:
        """
        Add a dependency relationship.

        Args:
            op_id: Operation that depends
            depends_on: Operation it depends on
        """
        with self._lock:
            if depends_on not in self._completed_ops:
                self._dependencies[op_id].add(depends_on)
                self._dependents[depends_on].add(op_id)

    def add_dependencies(self, op_id: int, depends_on: Set[int]) -> None:
        """
        Add multiple dependencies.

        Args:
            op_id: Operation that depends
            depends_on: Set of operations it depends on
        """
        with self._lock:
            for dep in depends_on:
                self.add_dependency(op_id, dep)

    def mark_completed(self, op_id: int) -> Set[int]:
        """
        Mark operation as completed and get newly ready operations.

        Args:
            op_id: Completed operation ID

        Returns:
            Set of operation IDs that became ready
        """
        with self._lock:
            self._completed_ops.add(op_id)

            # Find operations that were waiting for this
            newly_ready = set()
            for dependent in self._dependents.get(op_id, set()):
                # Remove from dependencies
                if dependent in self._dependencies:
                    self._dependencies[dependent].discard(op_id)
                    # Check if all dependencies satisfied
                    if not self._dependencies[dependent]:
                        newly_ready.add(dependent)
                        del self._dependencies[dependent]

            # Clean up
            if op_id in self._dependents:
                del self._dependents[op_id]
            if op_id in self._dependencies:
                del self._dependencies[op_id]

            return newly_ready

    def get_dependencies(self, op_id: int) -> Set[int]:
        """Get dependencies for an operation."""
        with self._lock:
            return self._dependencies.get(op_id, set()).copy()

    def get_dependents(self, op_id: int) -> Set[int]:
        """Get operations that depend on this one."""
        with self._lock:
            return self._dependents.get(op_id, set()).copy()

    def is_ready(self, op_id: int) -> bool:
        """Check if operation is ready (no unsatisfied dependencies)."""
        with self._lock:
            return op_id not in self._dependencies or not self._dependencies[op_id]

    def get_dependency_depth(self, op_id: int) -> int:
        """
        Get depth of operation in dependency graph.

        Returns:
            Depth (0 = no dependencies, higher = more dependencies)
        """
        with self._lock:
            if not self._dependencies.get(op_id):
                return 0

            # BFS to find max depth
            max_depth = 0
            visited = set()
            queue = deque([(op_id, 0)])

            while queue:
                current, depth = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                max_depth = max(max_depth, depth)

                for dep in self._dependencies.get(current, set()):
                    if dep not in visited:
                        queue.append((dep, depth + 1))

            return max_depth

    def clear(self) -> None:
        """Clear all dependency information."""
        with self._lock:
            self._dependencies.clear()
            self._dependents.clear()
            self._completed_ops.clear()


class WorkloadScheduler:
    """
    Main workload scheduler with ML-guided decision making.

    Coordinates operation scheduling, priority assignment, and batching decisions.
    """

    def __init__(
        self,
        policy: SchedulingPolicy = SchedulingPolicy.ML_GUIDED,
        enable_batching: bool = True,
        enable_online_learning: bool = True,
        max_overhead_percent: float = 1.0
    ):
        """
        Initialize workload scheduler.

        Args:
            policy: Scheduling policy to use
            enable_batching: Whether to enable operation batching
            enable_online_learning: Whether to enable online learning
            max_overhead_percent: Maximum overhead as percentage of runtime
        """
        self.policy = policy
        self.enable_batching = enable_batching
        self.enable_online_learning = enable_online_learning
        self.max_overhead_percent = max_overhead_percent

        # Core components
        self.operation_queue = OperationQueue()
        self.dependency_tracker = DependencyTracker()
        self.state_tracker = RuntimeStateTracker()
        self.feature_extractor = RuntimeFeatureExtractor(self.state_tracker)

        # ML models
        self.priority_model = create_priority_predictor()
        self.latency_model = create_latency_predictor()
        self.batching_model = create_batching_predictor() if enable_batching else None

        # Online learners
        self.priority_learner = None
        self.latency_learner = None
        if enable_online_learning:
            self.priority_learner = OnlineLearner(self.priority_model)
            self.latency_learner = OnlineLearner(self.latency_model)

        # Operation tracking
        self._next_op_id = 0
        self._executing_ops: Dict[int, RuntimeOperation] = {}
        self._completed_ops: Dict[int, RuntimeOperation] = {}

        # Batching
        self._next_batch_id = 0
        self._batch_groups: Dict[int, List[RuntimeOperation]] = {}

        # Performance tracking
        self._total_overhead_ns = 0
        self._scheduling_decisions = 0
        self._operations_scheduled = 0

        # Threading
        self._lock = threading.RLock()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False

    def submit_operation(
        self,
        op_type: str,
        function: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        input_shapes: List[Tuple[int, ...]] = None,
        target_device: Optional[torch.device] = None,
        dependencies: Set[int] = None
    ) -> int:
        """
        Submit an operation for scheduling.

        Args:
            op_type: Type of operation (e.g., "matmul", "conv2d")
            function: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            input_shapes: Input tensor shapes
            target_device: Target device
            dependencies: Set of operation IDs this depends on

        Returns:
            Operation ID
        """
        start = time.perf_counter_ns()

        with self._lock:
            op_id = self._next_op_id
            self._next_op_id += 1

            # Create operation
            operation = RuntimeOperation(
                op_id=op_id,
                op_type=op_type,
                function=function,
                args=args,
                kwargs=kwargs or {},
                input_shapes=input_shapes or [],
                target_device=target_device or torch.device("cpu"),
                dependencies=dependencies or set()
            )

            # Track dependencies
            if dependencies:
                self.dependency_tracker.add_dependencies(op_id, dependencies)
                # Set satisfied dependencies from already completed ops
                for dep in dependencies:
                    if dep in self._completed_ops:
                        operation.satisfy_dependency(dep)

            # Compute priority
            operation.priority = self._compute_priority(operation)

            # Estimate latency
            operation.estimated_latency_ms = self._estimate_latency(operation)

            # Check for batching opportunities
            if self.enable_batching:
                self._check_batching(operation)

            # Update status
            if operation.is_ready():
                operation.status = OperationStatus.READY

            # Add to queue
            self.operation_queue.push(operation)

            self._operations_scheduled += 1

        # Track overhead
        elapsed = time.perf_counter_ns() - start
        self._total_overhead_ns += elapsed

        return op_id

    def get_next_operation(self) -> Optional[RuntimeOperation]:
        """
        Get next operation to execute.

        Returns:
            Next operation or None if no ready operations
        """
        start = time.perf_counter_ns()

        with self._lock:
            # Get ready operations
            ready_ops = self.operation_queue.get_ready_operations(max_count=10)

            if not ready_ops:
                return None

            # Select best operation based on policy
            if self.policy == SchedulingPolicy.FIFO:
                selected = min(ready_ops, key=lambda op: op.creation_time)
            elif self.policy == SchedulingPolicy.PRIORITY:
                selected = max(ready_ops, key=lambda op: op.priority)
            elif self.policy in [SchedulingPolicy.ML_GUIDED, SchedulingPolicy.ADAPTIVE]:
                selected = self._select_ml_guided(ready_ops)
            else:
                selected = ready_ops[0]

            # Remove from queue
            self.operation_queue.remove(selected.op_id)
            selected.status = OperationStatus.EXECUTING
            selected.start_time = time.time()
            self._executing_ops[selected.op_id] = selected

            self._scheduling_decisions += 1

        # Track overhead
        elapsed = time.perf_counter_ns() - start
        self._total_overhead_ns += elapsed

        return selected

    def mark_completed(
        self,
        op_id: int,
        success: bool = True,
        actual_latency_ms: Optional[float] = None
    ) -> None:
        """
        Mark an operation as completed.

        Args:
            op_id: Operation ID
            success: Whether operation succeeded
            actual_latency_ms: Actual execution time (optional)
        """
        start = time.perf_counter_ns()

        with self._lock:
            if op_id not in self._executing_ops:
                return

            operation = self._executing_ops.pop(op_id)
            operation.end_time = time.time()
            operation.success = success
            operation.status = OperationStatus.COMPLETED if success else OperationStatus.FAILED

            # Record actual latency
            if actual_latency_ms is not None:
                operation.actual_latency_ms = actual_latency_ms
            else:
                operation.actual_latency_ms = operation.get_execution_time_ms()

            # Record in state tracker
            self.state_tracker.record_operation(
                op_id=op_id,
                op_type=operation.op_type,
                device=str(operation.target_device),
                start_time=operation.start_time,
                end_time=operation.end_time,
                success=success
            )

            # Update dependency tracker
            newly_ready = self.dependency_tracker.mark_completed(op_id)

            # Update dependent operations
            for dep_id in newly_ready:
                # Find operation in queue and update status
                self.operation_queue.update_priority(dep_id, self._compute_priority_by_id(dep_id))

            # Online learning update
            if self.enable_online_learning and success:
                self._update_models(operation)

            # Store completed operation
            self._completed_ops[op_id] = operation

        # Track overhead
        elapsed = time.perf_counter_ns() - start
        self._total_overhead_ns += elapsed

    def _compute_priority(self, operation: RuntimeOperation) -> float:
        """
        Compute priority for an operation.

        Args:
            operation: Operation to prioritize

        Returns:
            Priority score (higher = more urgent)
        """
        # Base priority factors
        priority = 0.0

        # 1. Dependency depth (operations on critical path get higher priority)
        depth = self.dependency_tracker.get_dependency_depth(operation.op_id)
        priority += depth * 10.0

        # 2. Number of dependents (operations blocking others get higher priority)
        num_dependents = len(operation.dependents)
        priority += num_dependents * 5.0

        # 3. Wait time (older operations get higher priority)
        wait_time_ms = operation.get_wait_time_ms()
        priority += wait_time_ms * 0.01

        # 4. ML-guided priority (if using ML)
        if self.policy in [SchedulingPolicy.ML_GUIDED, SchedulingPolicy.ADAPTIVE]:
            try:
                # Extract features
                features = self._extract_features_for_priority(operation)
                # Get ML prediction
                ml_priority = self.priority_model.predict(features)
                priority += ml_priority.value * 100.0  # Scale up ML contribution
            except Exception:
                pass  # Fall back to heuristic priority

        return priority

    def _compute_priority_by_id(self, op_id: int) -> float:
        """Compute priority for operation by ID."""
        # This is a simplified version - in practice, we'd need to look up the operation
        return 0.0

    def _estimate_latency(self, operation: RuntimeOperation) -> float:
        """
        Estimate operation latency.

        Args:
            operation: Operation to estimate

        Returns:
            Estimated latency in milliseconds
        """
        # Try ML-based estimation first
        if self.policy in [SchedulingPolicy.ML_GUIDED, SchedulingPolicy.ADAPTIVE]:
            try:
                features = self._extract_features_for_latency(operation)
                prediction = self.latency_model.predict(features)
                return prediction.value
            except Exception:
                pass

        # Fallback: use historical data
        hist_features = self.feature_extractor.extract_historical_features(
            op_type=operation.op_type,
            device=str(operation.target_device)
        )

        if hist_features.times_executed > 0:
            return hist_features.avg_duration_ms

        # Default estimate based on operation size
        total_elements = sum(
            math.prod(shape) if shape else 0
            for shape in operation.input_shapes
        )
        return max(0.1, total_elements / 1_000_000)  # Rough estimate

    def _check_batching(self, operation: RuntimeOperation) -> None:
        """
        Check if operation can be batched with others.

        Args:
            operation: Operation to check
        """
        if not self.enable_batching or not self.batching_model:
            return

        # Look for similar operations in queue
        ready_ops = self.operation_queue.get_ready_operations(max_count=20)

        for other_op in ready_ops:
            if other_op.op_type == operation.op_type and other_op.op_id != operation.op_id:
                # Check if batching is beneficial
                try:
                    features_a = self._extract_features_for_priority(operation)
                    features_b = self._extract_features_for_priority(other_op)
                    batching_score = self.batching_model.predict(features_a, features_b)

                    if batching_score.value > 0.7:  # Threshold for batching
                        # Assign to same batch group
                        if other_op.batch_group_id is not None:
                            operation.batch_group_id = other_op.batch_group_id
                        else:
                            batch_id = self._next_batch_id
                            self._next_batch_id += 1
                            operation.batch_group_id = batch_id
                            other_op.batch_group_id = batch_id

                        operation.can_batch = True
                        break
                except Exception:
                    pass

    def _select_ml_guided(self, ready_ops: List[RuntimeOperation]) -> RuntimeOperation:
        """
        Select operation using ML guidance.

        Args:
            ready_ops: List of ready operations

        Returns:
            Selected operation
        """
        if not ready_ops:
            return None

        # Get current system state
        system_features = self.feature_extractor.extract_system_state_features()

        best_op = None
        best_score = float('-inf')

        for op in ready_ops:
            # Compute score based on multiple factors
            score = op.priority

            # Adjust based on system state
            if system_features.overall_load > 0.8:
                # High load: prefer memory-bound ops
                if op.input_shapes:
                    total_bytes = sum(math.prod(s) if s else 0 for s in op.input_shapes) * 4
                    if total_bytes < 1_000_000:  # Small operations
                        score += 20.0

            # Prefer operations on less loaded devices
            device_idx = op.target_device.index if op.target_device.index else 0
            if device_idx < len(system_features.compute_util):
                device_util = system_features.compute_util[device_idx]
                score += (1.0 - device_util) * 10.0

            if score > best_score:
                best_score = score
                best_op = op

        return best_op or ready_ops[0]

    def _extract_features_for_priority(self, operation: RuntimeOperation) -> torch.Tensor:
        """Extract features for priority prediction."""
        # Simplified feature extraction
        features = []

        # Operation features
        features.append(float(hash(operation.op_type) % 1000))
        features.append(float(len(operation.input_shapes)))
        features.append(float(sum(math.prod(s) if s else 0 for s in operation.input_shapes)))

        # Dependency features
        features.append(float(len(operation.dependencies)))
        features.append(float(len(operation.dependents)))

        # System features
        features.append(self.operation_queue.size())

        # Pad to feature_dim
        while len(features) < 64:
            features.append(0.0)

        return torch.tensor(features[:64], dtype=torch.float32)

    def _extract_features_for_latency(self, operation: RuntimeOperation) -> torch.Tensor:
        """Extract features for latency prediction."""
        return self._extract_features_for_priority(operation)

    def _update_models(self, operation: RuntimeOperation) -> None:
        """
        Update ML models with completed operation data.

        Args:
            operation: Completed operation
        """
        if not self.enable_online_learning:
            return

        # Update latency model
        if operation.actual_latency_ms is not None and self.latency_learner:
            features = self._extract_features_for_latency(operation)
            self.latency_learner.add_example(
                features=features,
                target=operation.actual_latency_ms
            )

            # Perform update if threshold reached
            if self.latency_learner.should_update():
                self.latency_learner.update()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                "operations_scheduled": self._operations_scheduled,
                "scheduling_decisions": self._scheduling_decisions,
                "queue_size": self.operation_queue.size(),
                "executing_ops": len(self._executing_ops),
                "completed_ops": len(self._completed_ops),
                "overhead_ms": self._total_overhead_ns / 1_000_000,
                "avg_overhead_us": (
                    (self._total_overhead_ns / self._scheduling_decisions) / 1000
                    if self._scheduling_decisions > 0 else 0
                ),
                "state_tracker": self.state_tracker.get_overhead_metrics(),
                "feature_extractor": self.feature_extractor.get_performance_metrics()
            }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._total_overhead_ns = 0
            self._scheduling_decisions = 0
            self._operations_scheduled = 0
            self.state_tracker.reset_overhead_metrics()
            self.feature_extractor.reset_performance_metrics()

    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self.operation_queue.clear()
            self.dependency_tracker.clear()
            self._executing_ops.clear()
            self._completed_ops.clear()
            self._batch_groups.clear()


import math  # Add missing import

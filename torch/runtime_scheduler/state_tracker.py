"""
Runtime State Tracker for PyTorch Dynamic Workload Scheduler.

This module tracks the runtime system state including:
- Device utilization (GPU/CPU usage)
- Memory state (allocated, free, cached)
- Queue lengths and pending operations
- Stream status
- Recent execution history

All components are thread-safe with minimal overhead.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Deque
import torch


class DeviceType(Enum):
    """Device type enumeration."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    XLA = "xla"


@dataclass
class DeviceUtilization:
    """Track device utilization metrics."""
    device_type: DeviceType
    device_id: int
    compute_utilization: float  # 0.0 to 1.0
    memory_utilization: float  # 0.0 to 1.0
    active_streams: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryState:
    """Track memory state for a device."""
    device_type: DeviceType
    device_id: int
    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int
    cached_bytes: int
    peak_allocated_bytes: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class OperationRecord:
    """Record of a completed operation for history tracking."""
    op_id: int
    op_type: str
    device: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    memory_allocated: int = 0
    tensor_shapes: Optional[List[Tuple[int, ...]]] = None


@dataclass
class StreamState:
    """Track state of a CUDA stream."""
    stream_id: int
    device_id: int
    is_default: bool
    queued_ops: int
    last_activity: float = field(default_factory=time.time)


class RuntimeStateTracker:
    """
    Thread-safe tracker for runtime system state.

    Tracks device utilization, memory state, queue status, and execution history
    with minimal overhead. Uses lock-free data structures where possible.
    """

    def __init__(
        self,
        history_size: int = 1000,
        state_update_interval_ms: float = 10.0,
        enable_detailed_tracking: bool = True
    ):
        """
        Initialize the state tracker.

        Args:
            history_size: Number of operation records to keep in history
            state_update_interval_ms: Minimum interval between state updates (ms)
            enable_detailed_tracking: Whether to track detailed metrics
        """
        self.history_size = history_size
        self.state_update_interval_ms = state_update_interval_ms
        self.enable_detailed_tracking = enable_detailed_tracking

        # Thread-safe locks
        self._device_lock = threading.RLock()
        self._memory_lock = threading.RLock()
        self._history_lock = threading.RLock()
        self._stream_lock = threading.RLock()

        # Device utilization tracking
        self._device_utilization: Dict[Tuple[DeviceType, int], DeviceUtilization] = {}
        self._last_device_update: Dict[Tuple[DeviceType, int], float] = {}

        # Memory state tracking
        self._memory_state: Dict[Tuple[DeviceType, int], MemoryState] = {}
        self._last_memory_update: Dict[Tuple[DeviceType, int], float] = {}

        # Operation history (circular buffer for efficiency)
        self._operation_history: Deque[OperationRecord] = deque(maxlen=history_size)
        self._total_operations = 0
        self._failed_operations = 0

        # Stream tracking
        self._stream_states: Dict[int, StreamState] = {}

        # Queue metrics
        self._queue_length = 0
        self._pending_ops_by_device: Dict[str, int] = {}

        # Performance metrics
        self._total_overhead_ns = 0
        self._update_count = 0

        # Initialize device states
        self._initialize_device_states()

    def _initialize_device_states(self) -> None:
        """Initialize tracking for available devices."""
        # Track CPU
        cpu_key = (DeviceType.CPU, 0)
        self._device_utilization[cpu_key] = DeviceUtilization(
            device_type=DeviceType.CPU,
            device_id=0,
            compute_utilization=0.0,
            memory_utilization=0.0,
            active_streams=0
        )

        # Track CUDA devices if available
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                cuda_key = (DeviceType.CUDA, device_id)
                self._device_utilization[cuda_key] = DeviceUtilization(
                    device_type=DeviceType.CUDA,
                    device_id=device_id,
                    compute_utilization=0.0,
                    memory_utilization=0.0,
                    active_streams=0
                )
                self._memory_state[cuda_key] = self._get_cuda_memory_state(device_id)

    def _get_cuda_memory_state(self, device_id: int) -> MemoryState:
        """Get current CUDA memory state for a device."""
        try:
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            cached = reserved - allocated

            # Get total memory
            props = torch.cuda.get_device_properties(device_id)
            total_memory = props.total_memory
            free = total_memory - reserved

            peak = torch.cuda.max_memory_allocated(device_id)

            return MemoryState(
                device_type=DeviceType.CUDA,
                device_id=device_id,
                allocated_bytes=allocated,
                reserved_bytes=reserved,
                free_bytes=free,
                cached_bytes=cached,
                peak_allocated_bytes=peak
            )
        except Exception:
            # Fallback for errors
            return MemoryState(
                device_type=DeviceType.CUDA,
                device_id=device_id,
                allocated_bytes=0,
                reserved_bytes=0,
                free_bytes=0,
                cached_bytes=0,
                peak_allocated_bytes=0
            )

    def update_device_utilization(
        self,
        device_type: DeviceType,
        device_id: int,
        compute_util: float,
        memory_util: float,
        active_streams: int
    ) -> None:
        """
        Update device utilization metrics.

        Args:
            device_type: Type of device (CPU/CUDA/etc)
            device_id: Device index
            compute_util: Compute utilization (0.0 to 1.0)
            memory_util: Memory utilization (0.0 to 1.0)
            active_streams: Number of active streams
        """
        start = time.perf_counter_ns()

        key = (device_type, device_id)
        current_time = time.time()

        # Rate limiting: only update if enough time has passed
        with self._device_lock:
            last_update = self._last_device_update.get(key, 0.0)
            if (current_time - last_update) * 1000 < self.state_update_interval_ms:
                return

            self._device_utilization[key] = DeviceUtilization(
                device_type=device_type,
                device_id=device_id,
                compute_utilization=max(0.0, min(1.0, compute_util)),
                memory_utilization=max(0.0, min(1.0, memory_util)),
                active_streams=active_streams,
                timestamp=current_time
            )
            self._last_device_update[key] = current_time

        # Track overhead
        elapsed = time.perf_counter_ns() - start
        self._total_overhead_ns += elapsed
        self._update_count += 1

    def update_memory_state(self, device_type: DeviceType, device_id: int) -> None:
        """
        Update memory state for a device.

        Args:
            device_type: Type of device
            device_id: Device index
        """
        start = time.perf_counter_ns()

        key = (device_type, device_id)
        current_time = time.time()

        # Rate limiting
        with self._memory_lock:
            last_update = self._last_memory_update.get(key, 0.0)
            if (current_time - last_update) * 1000 < self.state_update_interval_ms:
                return

            if device_type == DeviceType.CUDA and torch.cuda.is_available():
                self._memory_state[key] = self._get_cuda_memory_state(device_id)

            self._last_memory_update[key] = current_time

        # Track overhead
        elapsed = time.perf_counter_ns() - start
        self._total_overhead_ns += elapsed
        self._update_count += 1

    def record_operation(
        self,
        op_id: int,
        op_type: str,
        device: str,
        start_time: float,
        end_time: float,
        success: bool,
        memory_allocated: int = 0,
        tensor_shapes: Optional[List[Tuple[int, ...]]] = None
    ) -> None:
        """
        Record a completed operation in history.

        Args:
            op_id: Unique operation identifier
            op_type: Type of operation (e.g., "matmul", "conv2d")
            device: Device where operation executed
            start_time: Operation start timestamp
            end_time: Operation end timestamp
            success: Whether operation completed successfully
            memory_allocated: Memory allocated during operation (bytes)
            tensor_shapes: Input tensor shapes
        """
        start = time.perf_counter_ns()

        duration_ms = (end_time - start_time) * 1000.0

        record = OperationRecord(
            op_id=op_id,
            op_type=op_type,
            device=device,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success=success,
            memory_allocated=memory_allocated,
            tensor_shapes=tensor_shapes if self.enable_detailed_tracking else None
        )

        with self._history_lock:
            self._operation_history.append(record)
            self._total_operations += 1
            if not success:
                self._failed_operations += 1

        # Track overhead
        elapsed = time.perf_counter_ns() - start
        self._total_overhead_ns += elapsed
        self._update_count += 1

    def update_stream_state(
        self,
        stream_id: int,
        device_id: int,
        is_default: bool,
        queued_ops: int
    ) -> None:
        """
        Update state of a CUDA stream.

        Args:
            stream_id: Stream identifier
            device_id: Device the stream is on
            is_default: Whether this is the default stream
            queued_ops: Number of operations queued on stream
        """
        with self._stream_lock:
            self._stream_states[stream_id] = StreamState(
                stream_id=stream_id,
                device_id=device_id,
                is_default=is_default,
                queued_ops=queued_ops
            )

    def update_queue_metrics(
        self,
        total_queue_length: int,
        pending_by_device: Dict[str, int]
    ) -> None:
        """
        Update queue metrics.

        Args:
            total_queue_length: Total number of operations in queue
            pending_by_device: Number of pending operations per device
        """
        self._queue_length = total_queue_length
        self._pending_ops_by_device = pending_by_device.copy()

    def get_device_utilization(
        self,
        device_type: DeviceType,
        device_id: int
    ) -> Optional[DeviceUtilization]:
        """Get device utilization metrics."""
        with self._device_lock:
            return self._device_utilization.get((device_type, device_id))

    def get_memory_state(
        self,
        device_type: DeviceType,
        device_id: int
    ) -> Optional[MemoryState]:
        """Get memory state for a device."""
        with self._memory_lock:
            # Update if needed
            key = (device_type, device_id)
            if key not in self._memory_state and device_type == DeviceType.CUDA:
                if torch.cuda.is_available():
                    self._memory_state[key] = self._get_cuda_memory_state(device_id)
            return self._memory_state.get(key)

    def get_recent_operations(
        self,
        count: int = 100,
        op_type: Optional[str] = None,
        device: Optional[str] = None
    ) -> List[OperationRecord]:
        """
        Get recent operation records.

        Args:
            count: Maximum number of records to return
            op_type: Filter by operation type (optional)
            device: Filter by device (optional)

        Returns:
            List of operation records (most recent first)
        """
        with self._history_lock:
            records = list(self._operation_history)

        # Apply filters
        if op_type:
            records = [r for r in records if r.op_type == op_type]
        if device:
            records = [r for r in records if r.device == device]

        # Return most recent first
        return list(reversed(records[-count:]))

    def get_operation_stats(
        self,
        op_type: Optional[str] = None,
        window_size: int = 100
    ) -> Dict[str, float]:
        """
        Get statistics for recent operations.

        Args:
            op_type: Filter by operation type (optional)
            window_size: Number of recent operations to analyze

        Returns:
            Dictionary with statistics (avg_duration_ms, success_rate, etc.)
        """
        records = self.get_recent_operations(count=window_size, op_type=op_type)

        if not records:
            return {
                "count": 0,
                "avg_duration_ms": 0.0,
                "min_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "success_rate": 0.0
            }

        durations = [r.duration_ms for r in records]
        successes = sum(1 for r in records if r.success)

        return {
            "count": len(records),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "success_rate": successes / len(records) if records else 0.0
        }

    def get_stream_states(self) -> Dict[int, StreamState]:
        """Get all stream states."""
        with self._stream_lock:
            return self._stream_states.copy()

    def get_queue_metrics(self) -> Tuple[int, Dict[str, int]]:
        """
        Get current queue metrics.

        Returns:
            Tuple of (total_queue_length, pending_by_device)
        """
        return self._queue_length, self._pending_ops_by_device.copy()

    def get_state_snapshot(self) -> Dict:
        """
        Get a complete snapshot of the current runtime state.

        Returns:
            Dictionary containing all state information
        """
        with self._device_lock, self._memory_lock, self._history_lock, self._stream_lock:
            return {
                "timestamp": time.time(),
                "devices": {
                    f"{dt.value}:{did}": {
                        "compute_util": util.compute_utilization,
                        "memory_util": util.memory_utilization,
                        "active_streams": util.active_streams
                    }
                    for (dt, did), util in self._device_utilization.items()
                },
                "memory": {
                    f"{dt.value}:{did}": {
                        "allocated_mb": mem.allocated_bytes / (1024 * 1024),
                        "reserved_mb": mem.reserved_bytes / (1024 * 1024),
                        "free_mb": mem.free_bytes / (1024 * 1024),
                        "cached_mb": mem.cached_bytes / (1024 * 1024)
                    }
                    for (dt, did), mem in self._memory_state.items()
                },
                "queues": {
                    "total_length": self._queue_length,
                    "by_device": self._pending_ops_by_device
                },
                "streams": {
                    sid: {
                        "device_id": state.device_id,
                        "queued_ops": state.queued_ops,
                        "is_default": state.is_default
                    }
                    for sid, state in self._stream_states.items()
                },
                "history": {
                    "total_operations": self._total_operations,
                    "failed_operations": self._failed_operations,
                    "failure_rate": (
                        self._failed_operations / self._total_operations
                        if self._total_operations > 0 else 0.0
                    )
                }
            }

    def get_overhead_metrics(self) -> Dict[str, float]:
        """
        Get performance overhead metrics.

        Returns:
            Dictionary with overhead statistics
        """
        if self._update_count == 0:
            return {
                "total_overhead_ms": 0.0,
                "avg_overhead_us": 0.0,
                "update_count": 0
            }

        return {
            "total_overhead_ms": self._total_overhead_ns / 1_000_000,
            "avg_overhead_us": (self._total_overhead_ns / self._update_count) / 1000,
            "update_count": self._update_count
        }

    def reset_overhead_metrics(self) -> None:
        """Reset overhead tracking metrics."""
        self._total_overhead_ns = 0
        self._update_count = 0

    def clear_history(self) -> None:
        """Clear operation history."""
        with self._history_lock:
            self._operation_history.clear()
            self._total_operations = 0
            self._failed_operations = 0

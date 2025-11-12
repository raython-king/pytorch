"""
Stream Manager for Runtime Scheduling

CUDA stream management with dynamic stream creation,
dependency-aware assignment, and compute-communication overlap.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch


class StreamPriority(Enum):
    """Stream priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2


class OperationType(Enum):
    """Operation type for stream assignment"""
    COMPUTE = "compute"
    COMMUNICATION = "communication"
    MEMORY = "memory"


@dataclass
class OperationInfo:
    """Operation metadata"""
    op_id: int
    op_type: OperationType
    device: torch.device
    dependencies: Set[int] = field(default_factory=set)
    dependents: Set[int] = field(default_factory=set)
    estimated_time: float = 0.0
    priority: StreamPriority = StreamPriority.NORMAL

    # Execution tracking
    stream_id: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    completed: bool = False


class StreamInfo:
    """Track stream state and operations"""

    def __init__(
        self,
        stream_id: int,
        device: torch.device,
        priority: StreamPriority = StreamPriority.NORMAL,
    ):
        self.stream_id = stream_id
        self.device = device
        self.priority = priority

        # Create CUDA stream
        if device.type == "cuda":
            with torch.cuda.device(device):
                if priority == StreamPriority.HIGH:
                    # High priority stream
                    self.stream = torch.cuda.Stream(priority=-1)
                elif priority == StreamPriority.LOW:
                    # Low priority stream
                    self.stream = torch.cuda.Stream(priority=1)
                else:
                    # Normal priority stream
                    self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        # Operation tracking
        self.queued_ops: deque = deque()
        self.active_ops: Set[int] = set()
        self.completed_ops: List[int] = []

        # Statistics
        self.total_ops = 0
        self.total_time = 0.0
        self.utilization = 0.0
        self.last_activity = time.time()

        # Lock for thread safety
        self.lock = threading.Lock()

    def enqueue_op(self, op_id: int):
        """Enqueue operation to stream"""
        with self.lock:
            self.queued_ops.append(op_id)

    def start_op(self, op_id: int):
        """Mark operation as started"""
        with self.lock:
            if op_id in self.queued_ops:
                self.queued_ops.remove(op_id)
            self.active_ops.add(op_id)
            self.last_activity = time.time()

    def complete_op(self, op_id: int, duration: float):
        """Mark operation as completed"""
        with self.lock:
            self.active_ops.discard(op_id)
            self.completed_ops.append(op_id)
            self.total_ops += 1
            self.total_time += duration
            self.last_activity = time.time()

            # Update utilization
            if self.total_ops > 0:
                self.utilization = self.total_time / (
                    time.time() - self.last_activity + self.total_time
                )

    def get_load(self) -> float:
        """Get current load (0-1)"""
        with self.lock:
            # Load = queued + active operations
            return min(1.0, (len(self.queued_ops) + len(self.active_ops)) / 10.0)

    def is_idle(self) -> bool:
        """Check if stream is idle"""
        with self.lock:
            return len(self.queued_ops) == 0 and len(self.active_ops) == 0

    def synchronize(self):
        """Synchronize stream"""
        if self.stream:
            self.stream.synchronize()


class StreamAssigner:
    """Assign operations to streams"""

    def __init__(self, streams: Dict[int, StreamInfo]):
        self.streams = streams
        self.lock = threading.Lock()

        # Assignment strategies
        self.strategy = "dependency_aware"  # round_robin, least_loaded, dependency_aware

        # Operation type to preferred stream mapping
        self.type_affinity: Dict[OperationType, List[int]] = defaultdict(list)

    def assign_stream(
        self,
        op: OperationInfo,
        available_streams: List[int],
    ) -> int:
        """
        Assign operation to a stream.

        Returns:
            Stream ID
        """
        with self.lock:
            if self.strategy == "round_robin":
                return self._assign_round_robin(available_streams)

            elif self.strategy == "least_loaded":
                return self._assign_least_loaded(available_streams)

            elif self.strategy == "dependency_aware":
                return self._assign_dependency_aware(op, available_streams)

            else:
                # Default: least loaded
                return self._assign_least_loaded(available_streams)

    def _assign_round_robin(self, available_streams: List[int]) -> int:
        """Round-robin assignment"""
        if not available_streams:
            return 0
        # Simple round-robin
        return available_streams[0]

    def _assign_least_loaded(self, available_streams: List[int]) -> int:
        """Assign to least loaded stream"""
        if not available_streams:
            return 0

        # Find stream with least load
        best_stream = available_streams[0]
        best_load = float("inf")

        for stream_id in available_streams:
            if stream_id in self.streams:
                load = self.streams[stream_id].get_load()
                if load < best_load:
                    best_load = load
                    best_stream = stream_id

        return best_stream

    def _assign_dependency_aware(
        self, op: OperationInfo, available_streams: List[int]
    ) -> int:
        """Dependency-aware assignment"""
        if not available_streams:
            return 0

        # If operation has dependencies, try to assign to same stream as dependencies
        if op.dependencies:
            # Find streams where dependencies are executing
            dep_streams = set()
            for stream_id, stream_info in self.streams.items():
                for dep_id in op.dependencies:
                    if (
                        dep_id in stream_info.active_ops
                        or dep_id in stream_info.queued_ops
                    ):
                        dep_streams.add(stream_id)

            # Prefer stream with dependencies
            for stream_id in dep_streams:
                if stream_id in available_streams:
                    return stream_id

        # If no dependencies or dependencies not found, use least loaded
        return self._assign_least_loaded(available_streams)


class StreamSynchronizer:
    """Manage dependencies between streams"""

    def __init__(self):
        self.lock = threading.Lock()

        # Dependency graph
        self.dependencies: Dict[int, Set[int]] = defaultdict(set)  # op_id -> deps
        self.dependents: Dict[int, Set[int]] = defaultdict(set)  # op_id -> dependents

        # Stream events for synchronization
        self.events: Dict[Tuple[torch.device, int], torch.cuda.Event] = {}

    def add_dependency(self, op_id: int, depends_on: int):
        """Add dependency between operations"""
        with self.lock:
            self.dependencies[op_id].add(depends_on)
            self.dependents[depends_on].add(op_id)

    def get_dependencies(self, op_id: int) -> Set[int]:
        """Get operation dependencies"""
        with self.lock:
            return self.dependencies.get(op_id, set()).copy()

    def check_ready(self, op_id: int, completed_ops: Set[int]) -> bool:
        """Check if operation is ready to execute"""
        with self.lock:
            deps = self.dependencies.get(op_id, set())
            return deps.issubset(completed_ops)

    def synchronize_streams(
        self,
        src_stream: StreamInfo,
        dst_stream: StreamInfo,
    ):
        """Synchronize dst_stream with src_stream"""
        if src_stream.device.type != "cuda" or dst_stream.device.type != "cuda":
            return

        # Use CUDA event for synchronization
        event_key = (src_stream.device, src_stream.stream_id)

        with self.lock:
            if event_key not in self.events:
                with torch.cuda.device(src_stream.device):
                    self.events[event_key] = torch.cuda.Event()

            event = self.events[event_key]

        # Record event on source stream
        event.record(src_stream.stream)

        # Wait for event on destination stream
        dst_stream.stream.wait_event(event)

    def clear_completed(self, op_id: int):
        """Clear completed operation from dependency graph"""
        with self.lock:
            # Remove from dependencies
            if op_id in self.dependencies:
                del self.dependencies[op_id]

            # Remove from dependents
            for dependents in self.dependents.values():
                dependents.discard(op_id)

            if op_id in self.dependents:
                del self.dependents[op_id]


class StreamManager:
    """Coordinate all streams"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.lock = threading.RLock()

        # Streams per device
        self.streams: Dict[torch.device, Dict[int, StreamInfo]] = defaultdict(dict)
        self.next_stream_id = 0

        # Initialize default streams
        self._initialize_streams()

        # Components
        self.assigner = StreamAssigner(self._get_all_streams())
        self.synchronizer = StreamSynchronizer()

        # Operation registry
        self.operations: Dict[int, OperationInfo] = {}
        self.next_op_id = 0
        self.completed_ops: Set[int] = set()

        # Stream pool for reuse
        self.stream_pool: Dict[torch.device, List[StreamInfo]] = defaultdict(list)

        # Configuration
        self.max_streams_per_device = 16
        self.enable_compute_comm_overlap = True

    def _initialize_streams(self):
        """Initialize default streams for each device"""
        # CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")

                # Create default streams
                # 1 high priority, 2 normal, 1 low priority
                self._create_stream(device, StreamPriority.HIGH)
                self._create_stream(device, StreamPriority.NORMAL)
                self._create_stream(device, StreamPriority.NORMAL)
                self._create_stream(device, StreamPriority.LOW)

    def _create_stream(
        self, device: torch.device, priority: StreamPriority = StreamPriority.NORMAL
    ) -> StreamInfo:
        """Create a new stream"""
        with self.lock:
            stream_id = self.next_stream_id
            self.next_stream_id += 1

            stream_info = StreamInfo(stream_id, device, priority)
            self.streams[device][stream_id] = stream_info

            return stream_info

    def _get_all_streams(self) -> Dict[int, StreamInfo]:
        """Get all streams across all devices"""
        all_streams = {}
        for device_streams in self.streams.values():
            all_streams.update(device_streams)
        return all_streams

    def register_operation(
        self,
        op_type: OperationType,
        device: torch.device,
        dependencies: Optional[Set[int]] = None,
        estimated_time: float = 0.0,
        priority: StreamPriority = StreamPriority.NORMAL,
    ) -> int:
        """Register an operation"""
        with self.lock:
            op_id = self.next_op_id
            self.next_op_id += 1

            op = OperationInfo(
                op_id=op_id,
                op_type=op_type,
                device=device,
                dependencies=dependencies or set(),
                estimated_time=estimated_time,
                priority=priority,
            )

            self.operations[op_id] = op

            # Add dependencies to synchronizer
            for dep_id in op.dependencies:
                self.synchronizer.add_dependency(op_id, dep_id)

            return op_id

    def schedule_operation(self, op_id: int) -> Optional[StreamInfo]:
        """
        Schedule operation to a stream.

        Returns:
            StreamInfo where operation was scheduled
        """
        with self.lock:
            if op_id not in self.operations:
                return None

            op = self.operations[op_id]

            # Check if dependencies are satisfied
            if not self.synchronizer.check_ready(op_id, self.completed_ops):
                return None  # Not ready yet

            # Get available streams for this device
            device_streams = self.streams.get(op.device, {})
            available_stream_ids = list(device_streams.keys())

            if not available_stream_ids:
                # Create a new stream
                stream_info = self._create_stream(op.device, op.priority)
                available_stream_ids = [stream_info.stream_id]

            # Assign to stream
            stream_id = self.assigner.assign_stream(op, available_stream_ids)
            stream_info = device_streams[stream_id]

            # Enqueue operation
            stream_info.enqueue_op(op_id)
            op.stream_id = stream_id

            # Synchronize with dependency streams if needed
            for dep_id in op.dependencies:
                if dep_id in self.operations:
                    dep_op = self.operations[dep_id]
                    if dep_op.stream_id and dep_op.stream_id != stream_id:
                        dep_stream = device_streams.get(dep_op.stream_id)
                        if dep_stream:
                            self.synchronizer.synchronize_streams(
                                dep_stream, stream_info
                            )

            return stream_info

    def start_operation(self, op_id: int):
        """Mark operation as started"""
        with self.lock:
            if op_id not in self.operations:
                return

            op = self.operations[op_id]
            op.start_time = time.time()

            # Update stream
            if op.stream_id:
                device_streams = self.streams.get(op.device, {})
                stream_info = device_streams.get(op.stream_id)
                if stream_info:
                    stream_info.start_op(op_id)

    def complete_operation(self, op_id: int):
        """Mark operation as completed"""
        with self.lock:
            if op_id not in self.operations:
                return

            op = self.operations[op_id]
            op.end_time = time.time()
            op.completed = True

            duration = op.end_time - (op.start_time or op.end_time)

            # Update stream
            if op.stream_id:
                device_streams = self.streams.get(op.device, {})
                stream_info = device_streams.get(op.stream_id)
                if stream_info:
                    stream_info.complete_op(op_id, duration)

            # Mark as completed
            self.completed_ops.add(op_id)

            # Clear from dependency graph
            self.synchronizer.clear_completed(op_id)

    def get_stream(self, device: torch.device, stream_id: int) -> Optional[StreamInfo]:
        """Get stream by ID"""
        with self.lock:
            return self.streams.get(device, {}).get(stream_id)

    def synchronize_device(self, device: torch.device):
        """Synchronize all streams on device"""
        with self.lock:
            device_streams = self.streams.get(device, {})
            for stream_info in device_streams.values():
                stream_info.synchronize()

    def synchronize_all(self):
        """Synchronize all streams"""
        with self.lock:
            for device_streams in self.streams.values():
                for stream_info in device_streams.values():
                    stream_info.synchronize()

    def get_stats(self) -> Dict[str, any]:
        """Get stream manager statistics"""
        with self.lock:
            stats = {
                "total_streams": sum(len(s) for s in self.streams.values()),
                "total_operations": len(self.operations),
                "completed_operations": len(self.completed_ops),
                "devices": {},
            }

            for device, device_streams in self.streams.items():
                device_stats = {
                    "streams": len(device_streams),
                    "stream_details": [
                        {
                            "stream_id": s.stream_id,
                            "priority": s.priority.name,
                            "load": s.get_load(),
                            "utilization": s.utilization,
                            "queued_ops": len(s.queued_ops),
                            "active_ops": len(s.active_ops),
                            "total_ops": s.total_ops,
                        }
                        for s in device_streams.values()
                    ],
                }
                stats["devices"][str(device)] = device_stats

            return stats

    def cleanup_idle_streams(self):
        """Cleanup idle streams to free resources"""
        with self.lock:
            for device, device_streams in list(self.streams.items()):
                # Keep at least 2 streams per device
                if len(device_streams) <= 2:
                    continue

                # Find idle streams
                for stream_id, stream_info in list(device_streams.items()):
                    if stream_info.is_idle():
                        # Move to pool for reuse
                        self.stream_pool[device].append(stream_info)
                        del device_streams[stream_id]

                        # Keep pool size limited
                        if len(self.stream_pool[device]) > 10:
                            self.stream_pool[device].pop(0)


# Global singleton instance
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """Get global stream manager instance"""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager

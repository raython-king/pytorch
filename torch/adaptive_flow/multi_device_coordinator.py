"""
Multi-Device Coordinator for Adaptive Flow Control.

Coordinates cross-device transfers, distributed communication patterns,
and collective operations across multiple devices.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import torch

from .topology_manager import NetworkTopology
from .routing_engine import RoutingEngine, RoutingStrategy
from .qos_manager import FlowDescriptor, QoSRequirements, QoSClass

logger = logging.getLogger(__name__)


class CommunicationPattern(Enum):
    """Types of communication patterns."""
    POINT_TO_POINT = "point_to_point"
    BROADCAST = "broadcast"
    SCATTER = "scatter"
    GATHER = "gather"
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    PIPELINE = "pipeline"


class TransferStage(Enum):
    """Stages of a transfer."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TransferRequest:
    """
    Request for data transfer between devices.

    Attributes:
        transfer_id: Unique transfer identifier
        src_device: Source device ID
        dst_device: Destination device ID
        size_bytes: Size of data to transfer
        priority: Transfer priority (0-7)
        qos_requirements: QoS requirements
        callback: Optional callback on completion
        timeout: Transfer timeout in seconds
        metadata: Additional metadata
    """
    transfer_id: int
    src_device: int
    dst_device: int
    size_bytes: int
    priority: int = 3
    qos_requirements: Optional[QoSRequirements] = None
    callback: Optional[Callable] = None
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransferStatus:
    """Status of an ongoing or completed transfer."""
    transfer_id: int
    stage: TransferStage
    src_device: int
    dst_device: int
    size_bytes: int
    bytes_transferred: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    route: Optional[List[int]] = None

    def get_progress(self) -> float:
        """Get transfer progress (0.0-1.0)."""
        if self.size_bytes == 0:
            return 1.0
        return self.bytes_transferred / self.size_bytes

    def get_duration(self) -> Optional[float]:
        """Get transfer duration in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def get_throughput_gbps(self) -> Optional[float]:
        """Get transfer throughput in Gbps."""
        duration = self.get_duration()
        if duration is None or duration == 0:
            return None
        bytes_per_sec = self.bytes_transferred / duration
        return (bytes_per_sec * 8) / (1024 ** 3)


@dataclass
class CollectiveOperation:
    """
    Descriptor for a collective communication operation.

    Attributes:
        op_id: Operation identifier
        pattern: Communication pattern
        devices: Participating devices
        data_size_bytes: Size of data per device
        root_device: Root device (for rooted operations)
        priority: Operation priority
        qos_requirements: QoS requirements
    """
    op_id: int
    pattern: CommunicationPattern
    devices: List[int]
    data_size_bytes: int
    root_device: Optional[int] = None
    priority: int = 3
    qos_requirements: Optional[QoSRequirements] = None


class StagingBuffer:
    """
    Manage staging through intermediate devices.

    When direct transfer is suboptimal, stage data through intermediate
    devices to improve overall performance.
    """

    def __init__(self, topology: NetworkTopology):
        """
        Initialize staging buffer.

        Args:
            topology: Network topology
        """
        self.topology = topology
        self._buffers: Dict[int, Set[int]] = defaultdict(set)  # device -> staging transfer IDs
        logger.info("StagingBuffer initialized")

    def should_stage(
        self,
        src: int,
        dst: int,
        size_bytes: int
    ) -> Optional[int]:
        """
        Determine if transfer should be staged through intermediate device.

        Args:
            src: Source device
            dst: Destination device
            size_bytes: Transfer size

        Returns:
            Intermediate device ID if staging recommended, None otherwise
        """
        # Get direct link
        direct_link = self.topology.get_link(src, dst)
        if not direct_link:
            return None

        direct_bw = direct_link.get_effective_bandwidth()

        # Check if any two-hop path has better aggregate bandwidth
        for intermediate in self.topology.get_neighbors(src):
            if intermediate == dst:
                continue

            link1 = self.topology.get_link(src, intermediate)
            link2 = self.topology.get_link(intermediate, dst)

            if not link1 or not link2:
                continue

            # Calculate effective bandwidth for staged transfer
            # Using harmonic mean of the two links
            bw1 = link1.get_effective_bandwidth()
            bw2 = link2.get_effective_bandwidth()

            if bw1 <= 0 or bw2 <= 0:
                continue

            staged_bw = 2 * bw1 * bw2 / (bw1 + bw2)

            # Use staging if significantly better
            if staged_bw > direct_bw * 1.3:
                logger.info(
                    f"Staging recommended: {src} -> {intermediate} -> {dst} "
                    f"(staged_bw={staged_bw:.2f} vs direct_bw={direct_bw:.2f} Gbps)"
                )
                return intermediate

        return None

    def reserve_buffer(self, device: int, transfer_id: int) -> bool:
        """Reserve staging buffer on device."""
        # Simplified: assume we can always reserve
        self._buffers[device].add(transfer_id)
        return True

    def release_buffer(self, device: int, transfer_id: int) -> None:
        """Release staging buffer."""
        self._buffers[device].discard(transfer_id)


class TransferScheduler:
    """
    Schedule transfers to optimize network utilization.

    Implements various scheduling policies:
    - FIFO: First-in-first-out
    - Priority: Priority-based scheduling
    - SJF: Shortest job first
    - Deadline: Earliest deadline first
    """

    def __init__(self):
        """Initialize transfer scheduler."""
        self._pending_queue: List[TransferRequest] = []
        self._active_transfers: Dict[int, TransferStatus] = {}
        self._completed_transfers: List[TransferStatus] = []
        self._lock = threading.RLock()
        self._next_transfer_id = 0
        logger.info("TransferScheduler initialized")

    def submit_transfer(self, request: TransferRequest) -> int:
        """
        Submit transfer request.

        Args:
            request: Transfer request

        Returns:
            Transfer ID
        """
        with self._lock:
            if request.transfer_id == 0:
                request.transfer_id = self._get_next_id()

            self._pending_queue.append(request)
            logger.debug(
                f"Transfer {request.transfer_id} submitted: "
                f"{request.src_device} -> {request.dst_device}, "
                f"size={request.size_bytes / (1024**2):.2f} MB"
            )
            return request.transfer_id

    def get_next_transfer(self, policy: str = "priority") -> Optional[TransferRequest]:
        """
        Get next transfer to execute.

        Args:
            policy: Scheduling policy ("fifo", "priority", "sjf", "deadline")

        Returns:
            Transfer request or None
        """
        with self._lock:
            if not self._pending_queue:
                return None

            if policy == "fifo":
                return self._pending_queue.pop(0)
            elif policy == "priority":
                return self._get_highest_priority()
            elif policy == "sjf":
                return self._get_shortest_job()
            elif policy == "deadline":
                return self._get_earliest_deadline()
            else:
                return self._pending_queue.pop(0)

    def start_transfer(self, transfer_id: int, route: List[int]) -> None:
        """Mark transfer as started."""
        with self._lock:
            status = TransferStatus(
                transfer_id=transfer_id,
                stage=TransferStage.IN_PROGRESS,
                src_device=route[0] if route else 0,
                dst_device=route[-1] if route else 0,
                size_bytes=0,  # Would be filled from request
                start_time=time.time(),
                route=route
            )
            self._active_transfers[transfer_id] = status

    def complete_transfer(self, transfer_id: int, success: bool = True, error: Optional[str] = None) -> None:
        """Mark transfer as completed."""
        with self._lock:
            if transfer_id in self._active_transfers:
                status = self._active_transfers[transfer_id]
                status.stage = TransferStage.COMPLETED if success else TransferStage.FAILED
                status.end_time = time.time()
                status.error = error
                self._completed_transfers.append(status)
                del self._active_transfers[transfer_id]
                logger.debug(f"Transfer {transfer_id} completed (success={success})")

    def get_status(self, transfer_id: int) -> Optional[TransferStatus]:
        """Get transfer status."""
        with self._lock:
            if transfer_id in self._active_transfers:
                return self._active_transfers[transfer_id]
            for status in self._completed_transfers:
                if status.transfer_id == transfer_id:
                    return status
            return None

    def _get_next_id(self) -> int:
        """Get next transfer ID."""
        self._next_transfer_id += 1
        return self._next_transfer_id

    def _get_highest_priority(self) -> Optional[TransferRequest]:
        """Get highest priority transfer."""
        if not self._pending_queue:
            return None
        self._pending_queue.sort(key=lambda x: x.priority, reverse=True)
        return self._pending_queue.pop(0)

    def _get_shortest_job(self) -> Optional[TransferRequest]:
        """Get shortest transfer."""
        if not self._pending_queue:
            return None
        self._pending_queue.sort(key=lambda x: x.size_bytes)
        return self._pending_queue.pop(0)

    def _get_earliest_deadline(self) -> Optional[TransferRequest]:
        """Get transfer with earliest deadline."""
        if not self._pending_queue:
            return None

        # Filter transfers with deadlines
        with_deadline = [t for t in self._pending_queue if t.timeout is not None]
        if not with_deadline:
            return self._pending_queue.pop(0)

        # Sort by deadline
        with_deadline.sort(key=lambda x: x.timeout)
        earliest = with_deadline[0]
        self._pending_queue.remove(earliest)
        return earliest


class MultiDeviceCoordinator:
    """
    Main coordinator for multi-device communication.

    Manages transfers, collectives, and optimizations across multiple devices.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        routing_engine: RoutingEngine
    ):
        """
        Initialize multi-device coordinator.

        Args:
            topology: Network topology
            routing_engine: Routing engine
        """
        self.topology = topology
        self.routing_engine = routing_engine
        self.scheduler = TransferScheduler()
        self.staging_buffer = StagingBuffer(topology)

        self._coordinator_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info("MultiDeviceCoordinator initialized")

    def transfer_data(
        self,
        src_device: int,
        dst_device: int,
        size_bytes: int,
        priority: int = 3,
        qos_requirements: Optional[QoSRequirements] = None,
        callback: Optional[Callable] = None
    ) -> int:
        """
        Initiate data transfer between devices.

        Args:
            src_device: Source device
            dst_device: Destination device
            size_bytes: Size of data
            priority: Transfer priority
            qos_requirements: QoS requirements
            callback: Completion callback

        Returns:
            Transfer ID
        """
        request = TransferRequest(
            transfer_id=0,  # Will be assigned
            src_device=src_device,
            dst_device=dst_device,
            size_bytes=size_bytes,
            priority=priority,
            qos_requirements=qos_requirements,
            callback=callback
        )

        transfer_id = self.scheduler.submit_transfer(request)
        logger.info(f"Transfer {transfer_id} initiated: {src_device} -> {dst_device}")

        return transfer_id

    def execute_collective(
        self,
        operation: CollectiveOperation
    ) -> List[int]:
        """
        Execute collective communication operation.

        Args:
            operation: Collective operation descriptor

        Returns:
            List of transfer IDs for the collective
        """
        logger.info(f"Executing collective {operation.pattern.value} with {len(operation.devices)} devices")

        if operation.pattern == CommunicationPattern.BROADCAST:
            return self._execute_broadcast(operation)
        elif operation.pattern == CommunicationPattern.SCATTER:
            return self._execute_scatter(operation)
        elif operation.pattern == CommunicationPattern.GATHER:
            return self._execute_gather(operation)
        elif operation.pattern == CommunicationPattern.ALL_REDUCE:
            return self._execute_all_reduce(operation)
        elif operation.pattern == CommunicationPattern.ALL_GATHER:
            return self._execute_all_gather(operation)
        elif operation.pattern == CommunicationPattern.ALL_TO_ALL:
            return self._execute_all_to_all(operation)
        else:
            logger.error(f"Unsupported collective pattern: {operation.pattern}")
            return []

    def _execute_broadcast(self, operation: CollectiveOperation) -> List[int]:
        """Execute broadcast operation."""
        transfer_ids = []
        root = operation.root_device or operation.devices[0]

        # Simple broadcast: root sends to all others
        for device in operation.devices:
            if device == root:
                continue

            transfer_id = self.transfer_data(
                src_device=root,
                dst_device=device,
                size_bytes=operation.data_size_bytes,
                priority=operation.priority,
                qos_requirements=operation.qos_requirements
            )
            transfer_ids.append(transfer_id)

        return transfer_ids

    def _execute_scatter(self, operation: CollectiveOperation) -> List[int]:
        """Execute scatter operation."""
        transfer_ids = []
        root = operation.root_device or operation.devices[0]

        # Root sends different chunks to each device
        for i, device in enumerate(operation.devices):
            if device == root:
                continue

            transfer_id = self.transfer_data(
                src_device=root,
                dst_device=device,
                size_bytes=operation.data_size_bytes,
                priority=operation.priority,
                qos_requirements=operation.qos_requirements
            )
            transfer_ids.append(transfer_id)

        return transfer_ids

    def _execute_gather(self, operation: CollectiveOperation) -> List[int]:
        """Execute gather operation."""
        transfer_ids = []
        root = operation.root_device or operation.devices[0]

        # All devices send to root
        for device in operation.devices:
            if device == root:
                continue

            transfer_id = self.transfer_data(
                src_device=device,
                dst_device=root,
                size_bytes=operation.data_size_bytes,
                priority=operation.priority,
                qos_requirements=operation.qos_requirements
            )
            transfer_ids.append(transfer_id)

        return transfer_ids

    def _execute_all_reduce(self, operation: CollectiveOperation) -> List[int]:
        """Execute all-reduce operation using ring algorithm."""
        transfer_ids = []
        n_devices = len(operation.devices)

        if n_devices < 2:
            return transfer_ids

        # Ring all-reduce: each device sends to next in ring
        for i in range(n_devices):
            src = operation.devices[i]
            dst = operation.devices[(i + 1) % n_devices]

            # Multiple phases in ring all-reduce
            for phase in range(n_devices - 1):
                transfer_id = self.transfer_data(
                    src_device=src,
                    dst_device=dst,
                    size_bytes=operation.data_size_bytes // n_devices,
                    priority=operation.priority,
                    qos_requirements=operation.qos_requirements
                )
                transfer_ids.append(transfer_id)

        return transfer_ids

    def _execute_all_gather(self, operation: CollectiveOperation) -> List[int]:
        """Execute all-gather operation."""
        transfer_ids = []

        # All-to-all communication pattern
        for src in operation.devices:
            for dst in operation.devices:
                if src == dst:
                    continue

                transfer_id = self.transfer_data(
                    src_device=src,
                    dst_device=dst,
                    size_bytes=operation.data_size_bytes,
                    priority=operation.priority,
                    qos_requirements=operation.qos_requirements
                )
                transfer_ids.append(transfer_id)

        return transfer_ids

    def _execute_all_to_all(self, operation: CollectiveOperation) -> List[int]:
        """Execute all-to-all operation."""
        transfer_ids = []

        for src in operation.devices:
            for dst in operation.devices:
                if src == dst:
                    continue

                transfer_id = self.transfer_data(
                    src_device=src,
                    dst_device=dst,
                    size_bytes=operation.data_size_bytes,
                    priority=operation.priority,
                    qos_requirements=operation.qos_requirements
                )
                transfer_ids.append(transfer_id)

        return transfer_ids

    def get_transfer_status(self, transfer_id: int) -> Optional[TransferStatus]:
        """Get status of a transfer."""
        return self.scheduler.get_status(transfer_id)

    def cancel_transfer(self, transfer_id: int) -> bool:
        """Cancel a pending or active transfer."""
        # Implementation would cancel the transfer
        logger.info(f"Cancelling transfer {transfer_id}")
        return True

    def optimize_pipeline_communication(
        self,
        pipeline_stages: List[int],
        micro_batch_size: int
    ) -> Dict[str, Any]:
        """
        Optimize communication for pipeline parallelism.

        Args:
            pipeline_stages: Device IDs for each pipeline stage
            micro_batch_size: Size of micro-batches

        Returns:
            Optimization recommendations
        """
        recommendations = {
            "overlap_compute_comm": True,
            "double_buffering": True,
            "optimal_micro_batches": 4,
            "stage_assignments": {}
        }

        # Analyze pipeline structure
        n_stages = len(pipeline_stages)

        # Recommend optimal micro-batch count
        # More micro-batches = better overlap, but more overhead
        recommendations["optimal_micro_batches"] = max(4, n_stages * 2)

        # Check link bandwidths between stages
        bottleneck_bw = float('inf')
        for i in range(n_stages - 1):
            src = pipeline_stages[i]
            dst = pipeline_stages[i + 1]
            link = self.topology.get_link(src, dst)
            if link:
                bottleneck_bw = min(bottleneck_bw, link.get_effective_bandwidth())

        recommendations["bottleneck_bandwidth_gbps"] = bottleneck_bw

        logger.info(f"Pipeline optimization: {n_stages} stages, bottleneck_bw={bottleneck_bw:.2f} Gbps")

        return recommendations

    def start(self) -> None:
        """Start coordinator background thread."""
        if self._running:
            return

        self._running = True
        self._coordinator_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True,
            name="MultiDeviceCoordinator"
        )
        self._coordinator_thread.start()
        logger.info("MultiDeviceCoordinator started")

    def stop(self) -> None:
        """Stop coordinator."""
        if not self._running:
            return

        self._running = False
        if self._coordinator_thread:
            self._coordinator_thread.join(timeout=5.0)
        logger.info("MultiDeviceCoordinator stopped")

    def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self._running:
            try:
                # Process pending transfers
                self._process_transfers()
                time.sleep(0.01)  # 10ms interval
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}", exc_info=True)

    def _process_transfers(self) -> None:
        """Process pending transfers."""
        # Get next transfer
        transfer = self.scheduler.get_next_transfer(policy="priority")
        if not transfer:
            return

        # Compute route
        route = self.routing_engine.compute_route(
            transfer.src_device,
            transfer.dst_device,
            strategy=RoutingStrategy.LEAST_CONGESTED
        )

        if not route:
            logger.warning(f"No route found for transfer {transfer.transfer_id}")
            self.scheduler.complete_transfer(transfer.transfer_id, success=False, error="No route")
            return

        # Start transfer
        self.scheduler.start_transfer(transfer.transfer_id, route.path)

        # Execute callback if provided
        if transfer.callback:
            try:
                transfer.callback(transfer.transfer_id, route)
            except Exception as e:
                logger.error(f"Error in transfer callback: {e}", exc_info=True)


__all__ = [
    "CommunicationPattern",
    "TransferStage",
    "TransferRequest",
    "TransferStatus",
    "CollectiveOperation",
    "StagingBuffer",
    "TransferScheduler",
    "MultiDeviceCoordinator",
]

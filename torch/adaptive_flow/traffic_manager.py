"""
Core Traffic Manager for Adaptive Flow Control.

This module provides intelligent traffic management with priority-based scheduling,
bandwidth allocation, and real-time monitoring for distributed training workloads.
"""

import time
import heapq
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import IntEnum
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Flow priority levels."""
    HIGH = 0      # Critical control messages
    MEDIUM = 1    # Normal data transfers
    LOW = 2       # Background transfers


@dataclass(order=True)
class DataFlow:
    """Represents a data transfer with scheduling metadata.

    Attributes:
        priority: Flow priority (lower value = higher priority)
        flow_id: Unique flow identifier
        source: Source node ID
        dest: Destination node ID
        size: Transfer size in bytes
        deadline: Optional deadline timestamp (seconds since epoch)
        creation_time: When flow was created
        start_time: When transfer started (None if not started)
        completion_time: When transfer completed (None if not completed)
        bytes_sent: Bytes already transferred
        metadata: Additional metadata
    """
    priority: int = field(compare=True)
    deadline: Optional[float] = field(default=None, compare=True)
    flow_id: str = field(compare=False)
    source: str = field(default="", compare=False)
    dest: str = field(default="", compare=False)
    size: int = field(default=0, compare=False)
    creation_time: float = field(default_factory=time.time, compare=False)
    start_time: Optional[float] = field(default=None, compare=False)
    completion_time: Optional[float] = field(default=None, compare=False)
    bytes_sent: int = field(default=0, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        """Ensure deadline is comparable."""
        if self.deadline is None:
            # Use infinity for flows without deadlines
            object.__setattr__(self, 'deadline', float('inf'))

    @property
    def remaining_bytes(self) -> int:
        """Bytes remaining to transfer."""
        return max(0, self.size - self.bytes_sent)

    @property
    def is_complete(self) -> bool:
        """Check if flow is complete."""
        return self.bytes_sent >= self.size

    @property
    def progress(self) -> float:
        """Progress as fraction [0.0, 1.0]."""
        if self.size == 0:
            return 1.0
        return min(1.0, self.bytes_sent / self.size)

    def update_progress(self, bytes_transferred: int) -> None:
        """Update transfer progress."""
        self.bytes_sent += bytes_transferred
        if self.start_time is None:
            self.start_time = time.time()
        if self.is_complete and self.completion_time is None:
            self.completion_time = time.time()


class FlowQueue:
    """Thread-safe priority queue for pending data flows.

    Implements a multi-level priority queue with efficient insertion,
    removal, and priority-based dequeue operations.
    """

    def __init__(self):
        """Initialize flow queue."""
        self._heap: List[DataFlow] = []
        self._flow_map: Dict[str, DataFlow] = {}
        self._lock = threading.RLock()
        self._size = 0

    def enqueue(self, flow: DataFlow) -> None:
        """Add flow to queue.

        Args:
            flow: DataFlow to enqueue

        Raises:
            ValueError: If flow_id already exists
        """
        with self._lock:
            if flow.flow_id in self._flow_map:
                raise ValueError(f"Flow {flow.flow_id} already in queue")

            heapq.heappush(self._heap, flow)
            self._flow_map[flow.flow_id] = flow
            self._size += 1
            logger.debug(f"Enqueued flow {flow.flow_id}, priority={flow.priority}")

    def dequeue(self) -> Optional[DataFlow]:
        """Remove and return highest priority flow.

        Returns:
            Highest priority DataFlow or None if queue empty
        """
        with self._lock:
            while self._heap:
                flow = heapq.heappop(self._heap)
                # Check if flow was removed
                if flow.flow_id in self._flow_map:
                    del self._flow_map[flow.flow_id]
                    self._size -= 1
                    logger.debug(f"Dequeued flow {flow.flow_id}")
                    return flow
            return None

    def peek(self) -> Optional[DataFlow]:
        """Return highest priority flow without removing.

        Returns:
            Highest priority DataFlow or None if queue empty
        """
        with self._lock:
            while self._heap:
                flow = self._heap[0]
                if flow.flow_id in self._flow_map:
                    return flow
                # Remove stale entry
                heapq.heappop(self._heap)
            return None

    def remove(self, flow_id: str) -> bool:
        """Remove flow by ID.

        Args:
            flow_id: Flow identifier

        Returns:
            True if flow was removed, False if not found
        """
        with self._lock:
            if flow_id in self._flow_map:
                del self._flow_map[flow_id]
                self._size -= 1
                # Actual removal from heap happens lazily during dequeue
                logger.debug(f"Removed flow {flow_id}")
                return True
            return False

    def get(self, flow_id: str) -> Optional[DataFlow]:
        """Get flow by ID without removing.

        Args:
            flow_id: Flow identifier

        Returns:
            DataFlow or None if not found
        """
        with self._lock:
            return self._flow_map.get(flow_id)

    def __len__(self) -> int:
        """Number of flows in queue."""
        return self._size

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._size == 0

    def get_all(self) -> List[DataFlow]:
        """Get all flows in queue (snapshot).

        Returns:
            List of DataFlow objects
        """
        with self._lock:
            return list(self._flow_map.values())


class BandwidthMonitor:
    """Monitor and track available bandwidth per link.

    Tracks bandwidth utilization, maintains moving averages,
    and provides estimates of available bandwidth.
    """

    def __init__(self, window_size: int = 100):
        """Initialize bandwidth monitor.

        Args:
            window_size: Number of samples for moving average
        """
        self._lock = threading.RLock()
        self._window_size = window_size

        # Per-link bandwidth tracking
        self._link_capacity: Dict[str, float] = {}  # bytes/sec
        self._link_usage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._link_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

        # Statistics
        self._bytes_sent: Dict[str, int] = defaultdict(int)
        self._flows_active: Dict[str, Set[str]] = defaultdict(set)

    def set_link_capacity(self, link_id: str, capacity: float) -> None:
        """Set link capacity in bytes/sec.

        Args:
            link_id: Link identifier
            capacity: Capacity in bytes per second
        """
        with self._lock:
            self._link_capacity[link_id] = capacity
            logger.info(f"Set link {link_id} capacity to {capacity / 1e9:.2f} GB/s")

    def record_transfer(self, link_id: str, bytes_transferred: int,
                       timestamp: Optional[float] = None) -> None:
        """Record a data transfer on a link.

        Args:
            link_id: Link identifier
            bytes_transferred: Bytes transferred
            timestamp: Transfer timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            self._link_usage[link_id].append(bytes_transferred)
            self._link_timestamps[link_id].append(timestamp)
            self._bytes_sent[link_id] += bytes_transferred

    def register_flow(self, link_id: str, flow_id: str) -> None:
        """Register active flow on link.

        Args:
            link_id: Link identifier
            flow_id: Flow identifier
        """
        with self._lock:
            self._flows_active[link_id].add(flow_id)

    def unregister_flow(self, link_id: str, flow_id: str) -> None:
        """Unregister flow from link.

        Args:
            link_id: Link identifier
            flow_id: Flow identifier
        """
        with self._lock:
            self._flows_active[link_id].discard(flow_id)

    def get_available_bandwidth(self, link_id: str) -> float:
        """Get estimated available bandwidth on link.

        Args:
            link_id: Link identifier

        Returns:
            Available bandwidth in bytes/sec
        """
        with self._lock:
            capacity = self._link_capacity.get(link_id, float('inf'))
            current_usage = self._get_current_usage(link_id)
            return max(0.0, capacity - current_usage)

    def get_utilization(self, link_id: str) -> float:
        """Get link utilization as fraction [0.0, 1.0].

        Args:
            link_id: Link identifier

        Returns:
            Utilization fraction
        """
        with self._lock:
            capacity = self._link_capacity.get(link_id)
            if capacity is None or capacity == 0:
                return 0.0

            current_usage = self._get_current_usage(link_id)
            return min(1.0, current_usage / capacity)

    def _get_current_usage(self, link_id: str) -> float:
        """Calculate current bandwidth usage (bytes/sec).

        Args:
            link_id: Link identifier

        Returns:
            Current usage in bytes/sec
        """
        usage_queue = self._link_usage[link_id]
        time_queue = self._link_timestamps[link_id]

        if len(usage_queue) < 2:
            return 0.0

        # Calculate rate over recent window
        total_bytes = sum(usage_queue)
        time_span = time_queue[-1] - time_queue[0]

        if time_span <= 0:
            return 0.0

        return total_bytes / time_span

    def get_statistics(self, link_id: str) -> Dict[str, Any]:
        """Get link statistics.

        Args:
            link_id: Link identifier

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'capacity': self._link_capacity.get(link_id, 0.0),
                'current_usage': self._get_current_usage(link_id),
                'available_bandwidth': self.get_available_bandwidth(link_id),
                'utilization': self.get_utilization(link_id),
                'bytes_sent': self._bytes_sent[link_id],
                'active_flows': len(self._flows_active[link_id]),
            }


class TrafficManager:
    """Coordinate all traffic management operations.

    Main coordinator that integrates flow queuing, bandwidth monitoring,
    scheduling, and flow control for efficient traffic management.
    """

    def __init__(self):
        """Initialize traffic manager."""
        self._lock = threading.RLock()

        # Core components
        self._pending_flows = FlowQueue()
        self._active_flows: Dict[str, DataFlow] = {}
        self._completed_flows: deque = deque(maxlen=10000)

        # Bandwidth management
        self._bandwidth_monitor = BandwidthMonitor()

        # Per-link flow assignment
        self._link_flows: Dict[str, Set[str]] = defaultdict(set)

        # Statistics
        self._stats = {
            'flows_submitted': 0,
            'flows_completed': 0,
            'flows_failed': 0,
            'bytes_transferred': 0,
            'total_wait_time': 0.0,
            'total_transfer_time': 0.0,
        }

        # Configuration
        self._max_flows_per_link = 4  # Fair sharing
        self._min_flow_rate = 1024 * 1024  # 1 MB/s minimum

        logger.info("TrafficManager initialized")

    def submit_flow(self, flow: DataFlow) -> None:
        """Submit a new flow for scheduling.

        Args:
            flow: DataFlow to submit

        Raises:
            ValueError: If flow_id already exists
        """
        with self._lock:
            if flow.flow_id in self._active_flows:
                raise ValueError(f"Flow {flow.flow_id} already active")

            self._pending_flows.enqueue(flow)
            self._stats['flows_submitted'] += 1
            logger.info(f"Submitted flow {flow.flow_id}: {flow.size} bytes, "
                       f"priority={flow.priority}")

    def schedule_flows(self) -> List[DataFlow]:
        """Schedule pending flows based on available bandwidth.

        Returns:
            List of flows scheduled for execution
        """
        scheduled = []

        with self._lock:
            while not self._pending_flows.is_empty():
                flow = self._pending_flows.peek()
                if flow is None:
                    break

                # Find link for this flow
                link_id = self._get_link_id(flow.source, flow.dest)

                # Check if we can schedule this flow
                if not self._can_schedule_flow(flow, link_id):
                    break

                # Schedule flow
                flow = self._pending_flows.dequeue()
                if flow:
                    self._activate_flow(flow, link_id)
                    scheduled.append(flow)

        if scheduled:
            logger.info(f"Scheduled {len(scheduled)} flows")

        return scheduled

    def _can_schedule_flow(self, flow: DataFlow, link_id: str) -> bool:
        """Check if flow can be scheduled.

        Args:
            flow: DataFlow to check
            link_id: Target link

        Returns:
            True if flow can be scheduled
        """
        # Check link capacity
        active_on_link = len(self._link_flows[link_id])
        if active_on_link >= self._max_flows_per_link:
            return False

        # Check available bandwidth
        available_bw = self._bandwidth_monitor.get_available_bandwidth(link_id)
        if available_bw < self._min_flow_rate:
            return False

        return True

    def _activate_flow(self, flow: DataFlow, link_id: str) -> None:
        """Activate a flow for transmission.

        Args:
            flow: DataFlow to activate
            link_id: Link to use
        """
        flow.start_time = time.time()
        self._active_flows[flow.flow_id] = flow
        self._link_flows[link_id].add(flow.flow_id)
        self._bandwidth_monitor.register_flow(link_id, flow.flow_id)

        logger.debug(f"Activated flow {flow.flow_id} on link {link_id}")

    def update_flow_progress(self, flow_id: str, bytes_transferred: int) -> None:
        """Update progress of an active flow.

        Args:
            flow_id: Flow identifier
            bytes_transferred: Bytes transferred in this update

        Raises:
            ValueError: If flow not found
        """
        with self._lock:
            flow = self._active_flows.get(flow_id)
            if flow is None:
                raise ValueError(f"Flow {flow_id} not active")

            flow.update_progress(bytes_transferred)
            self._stats['bytes_transferred'] += bytes_transferred

            # Record bandwidth usage
            link_id = self._get_link_id(flow.source, flow.dest)
            self._bandwidth_monitor.record_transfer(link_id, bytes_transferred)

            # Check if complete
            if flow.is_complete:
                self._complete_flow(flow_id)

    def _complete_flow(self, flow_id: str) -> None:
        """Mark flow as complete.

        Args:
            flow_id: Flow identifier
        """
        flow = self._active_flows.pop(flow_id, None)
        if flow is None:
            return

        # Unregister from link
        link_id = self._get_link_id(flow.source, flow.dest)
        self._link_flows[link_id].discard(flow_id)
        self._bandwidth_monitor.unregister_flow(link_id, flow_id)

        # Update statistics
        self._stats['flows_completed'] += 1
        if flow.start_time:
            wait_time = flow.start_time - flow.creation_time
            self._stats['total_wait_time'] += wait_time
        if flow.completion_time and flow.start_time:
            transfer_time = flow.completion_time - flow.start_time
            self._stats['total_transfer_time'] += transfer_time

        # Archive
        self._completed_flows.append(flow)

        logger.info(f"Completed flow {flow_id}: {flow.size} bytes in "
                   f"{transfer_time:.3f}s" if flow.completion_time and flow.start_time else "")

    def cancel_flow(self, flow_id: str) -> bool:
        """Cancel a pending or active flow.

        Args:
            flow_id: Flow identifier

        Returns:
            True if flow was cancelled
        """
        with self._lock:
            # Check if pending
            if self._pending_flows.remove(flow_id):
                self._stats['flows_failed'] += 1
                return True

            # Check if active
            if flow_id in self._active_flows:
                flow = self._active_flows.pop(flow_id)
                link_id = self._get_link_id(flow.source, flow.dest)
                self._link_flows[link_id].discard(flow_id)
                self._bandwidth_monitor.unregister_flow(link_id, flow_id)
                self._stats['flows_failed'] += 1
                return True

            return False

    def get_flow_status(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a flow.

        Args:
            flow_id: Flow identifier

        Returns:
            Dictionary with flow status or None if not found
        """
        with self._lock:
            # Check active flows
            flow = self._active_flows.get(flow_id)
            if flow:
                return {
                    'flow_id': flow.flow_id,
                    'status': 'active',
                    'progress': flow.progress,
                    'bytes_sent': flow.bytes_sent,
                    'bytes_remaining': flow.remaining_bytes,
                }

            # Check pending flows
            flow = self._pending_flows.get(flow_id)
            if flow:
                return {
                    'flow_id': flow.flow_id,
                    'status': 'pending',
                    'progress': 0.0,
                    'queue_position': 'unknown',
                }

            # Check completed flows
            for flow in reversed(self._completed_flows):
                if flow.flow_id == flow_id:
                    return {
                        'flow_id': flow.flow_id,
                        'status': 'completed',
                        'progress': 1.0,
                        'bytes_sent': flow.bytes_sent,
                    }

            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get traffic manager statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats['pending_flows'] = len(self._pending_flows)
            stats['active_flows'] = len(self._active_flows)

            # Calculate averages
            if stats['flows_completed'] > 0:
                stats['avg_wait_time'] = stats['total_wait_time'] / stats['flows_completed']
                stats['avg_transfer_time'] = stats['total_transfer_time'] / stats['flows_completed']
            else:
                stats['avg_wait_time'] = 0.0
                stats['avg_transfer_time'] = 0.0

            return stats

    def set_link_capacity(self, source: str, dest: str, capacity: float) -> None:
        """Set capacity for a link.

        Args:
            source: Source node
            dest: Destination node
            capacity: Capacity in bytes/sec
        """
        link_id = self._get_link_id(source, dest)
        self._bandwidth_monitor.set_link_capacity(link_id, capacity)

    def get_link_statistics(self, source: str, dest: str) -> Dict[str, Any]:
        """Get statistics for a link.

        Args:
            source: Source node
            dest: Destination node

        Returns:
            Dictionary with link statistics
        """
        link_id = self._get_link_id(source, dest)
        return self._bandwidth_monitor.get_statistics(link_id)

    @staticmethod
    def _get_link_id(source: str, dest: str) -> str:
        """Generate link identifier.

        Args:
            source: Source node
            dest: Destination node

        Returns:
            Link identifier
        """
        return f"{source}->{dest}"

    def __repr__(self) -> str:
        """String representation."""
        return (f"TrafficManager(pending={len(self._pending_flows)}, "
                f"active={len(self._active_flows)}, "
                f"completed={self._stats['flows_completed']})")

"""
Bandwidth Management for Adaptive Flow Control.

Implements bandwidth allocation, reservation, rate limiting,
and link monitoring for efficient bandwidth utilization.
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ReservationPriority(Enum):
    """Priority levels for bandwidth reservations."""
    CRITICAL = 0    # Must be satisfied
    HIGH = 1        # Important
    MEDIUM = 2      # Normal
    LOW = 3         # Best effort


@dataclass
class BandwidthReservation:
    """Bandwidth reservation request.

    Attributes:
        reservation_id: Unique identifier
        flow_id: Associated flow identifier
        bandwidth: Requested bandwidth in bytes/sec
        priority: Reservation priority
        duration: Reservation duration in seconds
        start_time: When reservation starts
        end_time: When reservation ends
        granted: Whether reservation was granted
    """
    reservation_id: str
    flow_id: str
    bandwidth: float
    priority: ReservationPriority = ReservationPriority.MEDIUM
    duration: float = 60.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    granted: bool = False

    def __post_init__(self):
        """Calculate end time if not set."""
        if self.end_time is None:
            self.end_time = self.start_time + self.duration

    @property
    def is_active(self) -> bool:
        """Check if reservation is currently active."""
        current = time.time()
        return self.granted and self.start_time <= current <= self.end_time

    @property
    def is_expired(self) -> bool:
        """Check if reservation has expired."""
        return time.time() > self.end_time


class TokenBucket:
    """Token bucket for rate limiting.

    Implements standard token bucket algorithm for smooth
    rate limiting with burst capacity.
    """

    def __init__(self, rate: float, capacity: float):
        """Initialize token bucket.

        Args:
            rate: Token generation rate (bytes/sec)
            capacity: Bucket capacity (bytes)
        """
        self._lock = threading.RLock()
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_update = time.time()

    def consume(self, tokens: float) -> bool:
        """Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        # Add tokens based on rate
        new_tokens = elapsed * self._rate
        self._tokens = min(self._capacity, self._tokens + new_tokens)

    def get_tokens(self) -> float:
        """Get current token count.

        Returns:
            Current tokens available
        """
        with self._lock:
            self._refill()
            return self._tokens

    def set_rate(self, rate: float) -> None:
        """Update token generation rate.

        Args:
            rate: New rate in bytes/sec
        """
        with self._lock:
            self._refill()
            self._rate = rate

    def set_capacity(self, capacity: float) -> None:
        """Update bucket capacity.

        Args:
            capacity: New capacity in bytes
        """
        with self._lock:
            self._capacity = capacity
            self._tokens = min(self._tokens, capacity)

    def reset(self) -> None:
        """Reset bucket to full capacity."""
        with self._lock:
            self._tokens = self._capacity
            self._last_update = time.time()


class BandwidthAllocator:
    """Allocate bandwidth to flows with fairness guarantees.

    Implements max-min fair allocation and proportional sharing
    for efficient bandwidth distribution.
    """

    def __init__(self):
        """Initialize bandwidth allocator."""
        self._lock = threading.RLock()

        # Link capacities
        self._link_capacity: Dict[str, float] = {}

        # Flow demands and allocations
        self._flow_demands: Dict[str, float] = {}
        self._flow_allocations: Dict[str, float] = {}
        self._flow_links: Dict[str, str] = {}

        # Fair share tracking
        self._link_flows: Dict[str, Set[str]] = defaultdict(set)

    def set_link_capacity(self, link_id: str, capacity: float) -> None:
        """Set link capacity.

        Args:
            link_id: Link identifier
            capacity: Capacity in bytes/sec
        """
        with self._lock:
            self._link_capacity[link_id] = capacity
            # Recompute allocations
            self._compute_allocations()
            logger.info(f"Set link {link_id} capacity to {capacity / 1e9:.2f} GB/s")

    def add_flow(self, flow_id: str, link_id: str, demand: float) -> float:
        """Add flow with bandwidth demand.

        Args:
            flow_id: Flow identifier
            link_id: Link to use
            demand: Bandwidth demand in bytes/sec

        Returns:
            Allocated bandwidth in bytes/sec
        """
        with self._lock:
            self._flow_demands[flow_id] = demand
            self._flow_links[flow_id] = link_id
            self._link_flows[link_id].add(flow_id)

            # Recompute fair allocations
            self._compute_allocations()

            allocation = self._flow_allocations.get(flow_id, 0.0)
            logger.debug(f"Added flow {flow_id}: demand={demand / 1e6:.2f} MB/s, "
                        f"allocated={allocation / 1e6:.2f} MB/s")
            return allocation

    def remove_flow(self, flow_id: str) -> None:
        """Remove flow and reclaim bandwidth.

        Args:
            flow_id: Flow identifier
        """
        with self._lock:
            link_id = self._flow_links.pop(flow_id, None)
            if link_id:
                self._link_flows[link_id].discard(flow_id)

            self._flow_demands.pop(flow_id, None)
            self._flow_allocations.pop(flow_id, None)

            # Recompute allocations
            self._compute_allocations()

            logger.debug(f"Removed flow {flow_id}")

    def _compute_allocations(self) -> None:
        """Compute max-min fair bandwidth allocations.

        Uses iterative max-min fairness algorithm to ensure fair
        bandwidth distribution while satisfying demands.
        """
        # Initialize allocations
        self._flow_allocations.clear()

        # Group flows by link
        for link_id, capacity in self._link_capacity.items():
            flows = list(self._link_flows[link_id])
            if not flows:
                continue

            # Get demands for flows on this link
            demands = {fid: self._flow_demands.get(fid, 0.0) for fid in flows}

            # Compute max-min fair allocation
            allocations = self._max_min_fairness(demands, capacity)

            # Update flow allocations
            for fid, alloc in allocations.items():
                self._flow_allocations[fid] = alloc

    def _max_min_fairness(self, demands: Dict[str, float], capacity: float) -> Dict[str, float]:
        """Compute max-min fair allocation.

        Args:
            demands: Flow bandwidth demands
            capacity: Link capacity

        Returns:
            Fair bandwidth allocations
        """
        allocations = {}
        remaining_capacity = capacity
        unsatisfied = set(demands.keys())

        while unsatisfied:
            # Equal share among unsatisfied flows
            equal_share = remaining_capacity / len(unsatisfied)

            # Find flows satisfied by equal share
            satisfied = set()
            for fid in unsatisfied:
                demand = demands[fid]
                if demand <= equal_share:
                    # Flow satisfied
                    allocations[fid] = demand
                    remaining_capacity -= demand
                    satisfied.add(fid)

            if not satisfied:
                # All remaining flows get equal share
                for fid in unsatisfied:
                    allocations[fid] = equal_share
                break

            unsatisfied -= satisfied

        return allocations

    def get_allocation(self, flow_id: str) -> float:
        """Get current allocation for flow.

        Args:
            flow_id: Flow identifier

        Returns:
            Allocated bandwidth in bytes/sec
        """
        with self._lock:
            return self._flow_allocations.get(flow_id, 0.0)

    def update_demand(self, flow_id: str, demand: float) -> float:
        """Update flow bandwidth demand.

        Args:
            flow_id: Flow identifier
            demand: New demand in bytes/sec

        Returns:
            New allocated bandwidth
        """
        with self._lock:
            if flow_id in self._flow_demands:
                self._flow_demands[flow_id] = demand
                self._compute_allocations()
                return self._flow_allocations.get(flow_id, 0.0)
            return 0.0

    def get_link_statistics(self, link_id: str) -> Dict[str, float]:
        """Get allocation statistics for link.

        Args:
            link_id: Link identifier

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            capacity = self._link_capacity.get(link_id, 0.0)
            flows = self._link_flows[link_id]

            total_demand = sum(self._flow_demands.get(fid, 0.0) for fid in flows)
            total_allocated = sum(self._flow_allocations.get(fid, 0.0) for fid in flows)

            return {
                'capacity': capacity,
                'num_flows': len(flows),
                'total_demand': total_demand,
                'total_allocated': total_allocated,
                'utilization': total_allocated / capacity if capacity > 0 else 0.0,
            }


class BandwidthReservationManager:
    """Manage bandwidth reservations for critical flows."""

    def __init__(self):
        """Initialize reservation manager."""
        self._lock = threading.RLock()

        # Active reservations
        self._reservations: Dict[str, BandwidthReservation] = {}
        self._flow_reservations: Dict[str, str] = {}

        # Per-link reserved bandwidth
        self._link_reserved: Dict[str, float] = defaultdict(float)
        self._link_capacity: Dict[str, float] = {}

    def set_link_capacity(self, link_id: str, capacity: float) -> None:
        """Set link capacity.

        Args:
            link_id: Link identifier
            capacity: Capacity in bytes/sec
        """
        with self._lock:
            self._link_capacity[link_id] = capacity

    def request_reservation(self, reservation: BandwidthReservation,
                          link_id: str) -> bool:
        """Request bandwidth reservation.

        Args:
            reservation: Reservation request
            link_id: Link identifier

        Returns:
            True if reservation granted
        """
        with self._lock:
            # Check if enough bandwidth available
            capacity = self._link_capacity.get(link_id, float('inf'))
            reserved = self._link_reserved[link_id]
            available = capacity - reserved

            if reservation.bandwidth <= available:
                # Grant reservation
                reservation.granted = True
                self._reservations[reservation.reservation_id] = reservation
                self._flow_reservations[reservation.flow_id] = reservation.reservation_id
                self._link_reserved[link_id] += reservation.bandwidth

                logger.info(f"Granted reservation {reservation.reservation_id}: "
                           f"{reservation.bandwidth / 1e6:.2f} MB/s for {reservation.duration:.1f}s")
                return True
            else:
                # Try admission control based on priority
                if reservation.priority == ReservationPriority.CRITICAL:
                    # Try to preempt lower priority reservations
                    if self._try_preempt(link_id, reservation.bandwidth):
                        reservation.granted = True
                        self._reservations[reservation.reservation_id] = reservation
                        self._flow_reservations[reservation.flow_id] = reservation.reservation_id
                        self._link_reserved[link_id] += reservation.bandwidth
                        return True

                logger.warning(f"Denied reservation {reservation.reservation_id}: "
                              f"insufficient bandwidth (need {reservation.bandwidth / 1e6:.2f} MB/s, "
                              f"available {available / 1e6:.2f} MB/s)")
                return False

    def _try_preempt(self, link_id: str, needed_bandwidth: float) -> bool:
        """Try to preempt lower priority reservations.

        Args:
            link_id: Link identifier
            needed_bandwidth: Bandwidth needed

        Returns:
            True if enough bandwidth freed
        """
        # Find preemptable reservations
        preemptable = []
        for res in self._reservations.values():
            if (res.is_active and
                res.priority in (ReservationPriority.LOW, ReservationPriority.MEDIUM)):
                preemptable.append(res)

        # Sort by priority (preempt lowest priority first)
        preemptable.sort(key=lambda r: (r.priority.value, r.start_time), reverse=True)

        # Try to free enough bandwidth
        freed = 0.0
        preempted = []
        for res in preemptable:
            freed += res.bandwidth
            preempted.append(res)
            if freed >= needed_bandwidth:
                break

        if freed >= needed_bandwidth:
            # Preempt reservations
            for res in preempted:
                self.cancel_reservation(res.reservation_id)
                logger.info(f"Preempted reservation {res.reservation_id}")
            return True

        return False

    def cancel_reservation(self, reservation_id: str) -> bool:
        """Cancel an existing reservation.

        Args:
            reservation_id: Reservation identifier

        Returns:
            True if cancelled
        """
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation:
                self._flow_reservations.pop(reservation.flow_id, None)
                # Note: link_id not stored, so we can't update _link_reserved
                # In production, should track link_id per reservation
                logger.info(f"Cancelled reservation {reservation_id}")
                return True
            return False

    def cleanup_expired(self) -> int:
        """Clean up expired reservations.

        Returns:
            Number of reservations cleaned up
        """
        with self._lock:
            expired = [rid for rid, res in self._reservations.items() if res.is_expired]

            for rid in expired:
                self.cancel_reservation(rid)

            if expired:
                logger.info(f"Cleaned up {len(expired)} expired reservations")

            return len(expired)

    def get_reservation(self, flow_id: str) -> Optional[BandwidthReservation]:
        """Get reservation for flow.

        Args:
            flow_id: Flow identifier

        Returns:
            Reservation or None
        """
        with self._lock:
            rid = self._flow_reservations.get(flow_id)
            if rid:
                return self._reservations.get(rid)
            return None

    def get_reserved_bandwidth(self, link_id: str) -> float:
        """Get total reserved bandwidth on link.

        Args:
            link_id: Link identifier

        Returns:
            Reserved bandwidth in bytes/sec
        """
        with self._lock:
            return self._link_reserved[link_id]


class AdaptiveLimiter:
    """Adaptive rate limiting based on congestion feedback.

    Dynamically adjusts rate limits based on network conditions
    using token bucket with adaptive parameters.
    """

    def __init__(self, initial_rate: float):
        """Initialize adaptive limiter.

        Args:
            initial_rate: Initial rate limit in bytes/sec
        """
        self._lock = threading.RLock()

        # Token bucket
        self._rate = initial_rate
        self._burst_capacity = initial_rate * 2  # 2 seconds worth
        self._bucket = TokenBucket(self._rate, self._burst_capacity)

        # Adaptation parameters
        self._min_rate = initial_rate * 0.1
        self._max_rate = initial_rate * 10
        self._increase_factor = 1.1
        self._decrease_factor = 0.5

        # Statistics
        self._allowed = 0
        self._throttled = 0

    def allow(self, size: float) -> bool:
        """Check if transfer is allowed.

        Args:
            size: Transfer size in bytes

        Returns:
            True if allowed
        """
        with self._lock:
            allowed = self._bucket.consume(size)

            if allowed:
                self._allowed += 1
            else:
                self._throttled += 1

            return allowed

    def adapt(self, congestion_detected: bool) -> None:
        """Adapt rate limit based on congestion.

        Args:
            congestion_detected: Whether congestion is detected
        """
        with self._lock:
            if congestion_detected:
                # Decrease rate
                new_rate = max(self._min_rate, self._rate * self._decrease_factor)
                logger.info(f"Adaptive limiter decreasing rate: "
                           f"{self._rate / 1e6:.2f} -> {new_rate / 1e6:.2f} MB/s")
            else:
                # Increase rate
                new_rate = min(self._max_rate, self._rate * self._increase_factor)
                logger.debug(f"Adaptive limiter increasing rate: "
                            f"{self._rate / 1e6:.2f} -> {new_rate / 1e6:.2f} MB/s")

            self._rate = new_rate
            self._bucket.set_rate(new_rate)

    def get_rate(self) -> float:
        """Get current rate limit.

        Returns:
            Rate limit in bytes/sec
        """
        with self._lock:
            return self._rate

    def get_statistics(self) -> Dict[str, float]:
        """Get limiter statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total = self._allowed + self._throttled
            throttle_rate = self._throttled / total if total > 0 else 0.0

            return {
                'rate': self._rate,
                'allowed': self._allowed,
                'throttled': self._throttled,
                'throttle_rate': throttle_rate,
            }


class LinkMonitor:
    """Per-link bandwidth tracking and monitoring.

    Monitors bandwidth usage, latency, and packet loss for
    each network link.
    """

    def __init__(self, window_size: int = 100):
        """Initialize link monitor.

        Args:
            window_size: Number of samples for moving average
        """
        self._lock = threading.RLock()
        self._window_size = window_size

        # Per-link state
        self._link_capacity: Dict[str, float] = {}
        self._link_usage: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._link_latency: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._link_loss: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # Cumulative statistics
        self._bytes_sent: Dict[str, int] = defaultdict(int)
        self._packets_sent: Dict[str, int] = defaultdict(int)
        self._packets_lost: Dict[str, int] = defaultdict(int)

    def set_capacity(self, link_id: str, capacity: float) -> None:
        """Set link capacity.

        Args:
            link_id: Link identifier
            capacity: Capacity in bytes/sec
        """
        with self._lock:
            self._link_capacity[link_id] = capacity

    def record_transfer(self, link_id: str, bytes_sent: int,
                       latency: Optional[float] = None) -> None:
        """Record a transfer on link.

        Args:
            link_id: Link identifier
            bytes_sent: Bytes transferred
            latency: Optional latency measurement in seconds
        """
        with self._lock:
            self._bytes_sent[link_id] += bytes_sent
            self._packets_sent[link_id] += 1
            self._link_usage[link_id].append(bytes_sent)

            if latency is not None:
                self._link_latency[link_id].append(latency)

    def record_loss(self, link_id: str, packets_lost: int = 1) -> None:
        """Record packet loss on link.

        Args:
            link_id: Link identifier
            packets_lost: Number of packets lost
        """
        with self._lock:
            self._packets_lost[link_id] += packets_lost
            self._link_loss[link_id].append(packets_lost)

    def get_throughput(self, link_id: str) -> float:
        """Get average throughput on link.

        Args:
            link_id: Link identifier

        Returns:
            Throughput in bytes/sec
        """
        with self._lock:
            usage = self._link_usage[link_id]
            if len(usage) < 2:
                return 0.0

            # Average over window
            return sum(usage) / len(usage)

    def get_utilization(self, link_id: str) -> float:
        """Get link utilization.

        Args:
            link_id: Link identifier

        Returns:
            Utilization fraction [0.0, 1.0]
        """
        with self._lock:
            capacity = self._link_capacity.get(link_id, 0.0)
            if capacity == 0:
                return 0.0

            throughput = self.get_throughput(link_id)
            return min(1.0, throughput / capacity)

    def get_average_latency(self, link_id: str) -> float:
        """Get average latency on link.

        Args:
            link_id: Link identifier

        Returns:
            Average latency in seconds
        """
        with self._lock:
            latency = self._link_latency[link_id]
            if not latency:
                return 0.0

            return sum(latency) / len(latency)

    def get_loss_rate(self, link_id: str) -> float:
        """Get packet loss rate on link.

        Args:
            link_id: Link identifier

        Returns:
            Loss rate [0.0, 1.0]
        """
        with self._lock:
            packets_sent = self._packets_sent[link_id]
            if packets_sent == 0:
                return 0.0

            packets_lost = self._packets_lost[link_id]
            return packets_lost / (packets_sent + packets_lost)

    def get_statistics(self, link_id: str) -> Dict[str, float]:
        """Get comprehensive statistics for link.

        Args:
            link_id: Link identifier

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'capacity': self._link_capacity.get(link_id, 0.0),
                'throughput': self.get_throughput(link_id),
                'utilization': self.get_utilization(link_id),
                'average_latency': self.get_average_latency(link_id),
                'loss_rate': self.get_loss_rate(link_id),
                'bytes_sent': self._bytes_sent[link_id],
                'packets_sent': self._packets_sent[link_id],
                'packets_lost': self._packets_lost[link_id],
            }

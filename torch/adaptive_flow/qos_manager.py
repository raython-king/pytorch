"""
Quality of Service (QoS) Manager for Adaptive Flow Control.

Provides QoS classification, enforcement, admission control, and SLA monitoring
for network flows.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from .topology_manager import NetworkTopology

logger = logging.getLogger(__name__)


class QoSClass(Enum):
    """QoS classes with different service guarantees."""
    LATENCY_SENSITIVE = "latency_sensitive"  # Interactive, low latency
    THROUGHPUT_ORIENTED = "throughput_oriented"  # Bulk transfers
    BEST_EFFORT = "best_effort"  # No guarantees
    BACKGROUND = "background"  # Lowest priority


@dataclass
class QoSRequirements:
    """
    QoS requirements for a flow.

    Attributes:
        qos_class: QoS class
        min_bandwidth_gbps: Minimum bandwidth requirement
        max_latency_us: Maximum latency tolerance
        max_jitter_us: Maximum jitter tolerance
        max_packet_loss: Maximum packet loss rate
        priority: Priority level (0-7, higher is better)
        deadline_ms: Deadline for completion (optional)
    """
    qos_class: QoSClass
    min_bandwidth_gbps: float = 0.0
    max_latency_us: float = float('inf')
    max_jitter_us: float = float('inf')
    max_packet_loss: float = 1.0
    priority: int = 0
    deadline_ms: Optional[float] = None

    def is_satisfied(
        self,
        bandwidth: float,
        latency: float,
        jitter: float,
        packet_loss: float
    ) -> bool:
        """Check if requirements are satisfied."""
        return (
            bandwidth >= self.min_bandwidth_gbps and
            latency <= self.max_latency_us and
            jitter <= self.max_jitter_us and
            packet_loss <= self.max_packet_loss
        )


@dataclass
class FlowDescriptor:
    """
    Descriptor for a network flow.

    Attributes:
        flow_id: Unique flow identifier
        src_device: Source device
        dst_device: Destination device
        size_bytes: Total size of transfer
        qos_requirements: QoS requirements
        creation_time: When flow was created
        deadline: Absolute deadline (if any)
        priority: Flow priority
        metadata: Additional metadata
    """
    flow_id: int
    src_device: int
    dst_device: int
    size_bytes: int
    qos_requirements: QoSRequirements
    creation_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize deadline from requirements."""
        if self.deadline is None and self.qos_requirements.deadline_ms:
            self.deadline = self.creation_time + (self.qos_requirements.deadline_ms / 1000.0)


@dataclass
class SLAMetrics:
    """
    Service Level Agreement metrics.

    Tracks actual performance vs. requirements.
    """
    flow_id: int
    required_bandwidth: float
    actual_bandwidth: float
    required_latency: float
    actual_latency: float
    packet_loss: float
    violations: int = 0
    total_checks: int = 0
    last_check: float = field(default_factory=time.time)

    def check_violation(self) -> bool:
        """Check if SLA is violated."""
        self.total_checks += 1
        violated = (
            self.actual_bandwidth < self.required_bandwidth * 0.9 or  # Allow 10% margin
            self.actual_latency > self.required_latency * 1.1 or
            self.packet_loss > 0.01
        )
        if violated:
            self.violations += 1
        self.last_check = time.time()
        return violated

    def get_violation_rate(self) -> float:
        """Get SLA violation rate."""
        return self.violations / self.total_checks if self.total_checks > 0 else 0.0


class QoSClassifier:
    """
    Classify flows into QoS classes based on characteristics.

    Uses heuristics and ML models to determine appropriate QoS class.
    """

    def __init__(self):
        """Initialize QoS classifier."""
        self._classification_rules: List[tuple] = []
        self._setup_default_rules()
        logger.info("QoSClassifier initialized")

    def _setup_default_rules(self) -> None:
        """Setup default classification rules."""
        # Rule format: (predicate_func, qos_class, priority)
        self._classification_rules = [
            # Small, time-sensitive transfers
            (lambda f: f.size_bytes < 1024 * 1024 and f.qos_requirements.max_latency_us < 1000,
             QoSClass.LATENCY_SENSITIVE, 7),

            # Large transfers with deadline
            (lambda f: f.size_bytes > 100 * 1024 * 1024 and f.deadline is not None,
             QoSClass.THROUGHPUT_ORIENTED, 5),

            # Large background transfers
            (lambda f: f.size_bytes > 1024 * 1024 * 1024,
             QoSClass.BACKGROUND, 1),

            # Default to best effort
            (lambda f: True, QoSClass.BEST_EFFORT, 3),
        ]

    def classify(self, flow: FlowDescriptor) -> tuple[QoSClass, int]:
        """
        Classify flow into QoS class and priority.

        Args:
            flow: Flow descriptor

        Returns:
            Tuple of (QoS class, priority)
        """
        # If QoS class is already specified, use it
        if flow.qos_requirements.qos_class != QoSClass.BEST_EFFORT:
            return flow.qos_requirements.qos_class, flow.qos_requirements.priority

        # Apply classification rules
        for predicate, qos_class, priority in self._classification_rules:
            try:
                if predicate(flow):
                    logger.debug(f"Flow {flow.flow_id} classified as {qos_class.value}")
                    return qos_class, priority
            except Exception as e:
                logger.warning(f"Error applying classification rule: {e}")
                continue

        # Fallback
        return QoSClass.BEST_EFFORT, 3

    def add_rule(self, predicate: callable, qos_class: QoSClass, priority: int) -> None:
        """
        Add custom classification rule.

        Args:
            predicate: Function that takes FlowDescriptor and returns bool
            qos_class: QoS class to assign
            priority: Priority to assign
        """
        self._classification_rules.insert(0, (predicate, qos_class, priority))


class QoSEnforcer:
    """
    Enforce QoS policies by managing bandwidth allocation and prioritization.

    Implements traffic shaping, rate limiting, and priority scheduling.
    """

    def __init__(self, topology: NetworkTopology):
        """
        Initialize QoS enforcer.

        Args:
            topology: Network topology
        """
        self.topology = topology
        self._active_flows: Dict[int, FlowDescriptor] = {}
        self._bandwidth_allocations: Dict[int, float] = {}
        self._lock = threading.RLock()
        logger.info("QoSEnforcer initialized")

    def admit_flow(self, flow: FlowDescriptor) -> bool:
        """
        Admit a flow and allocate resources.

        Args:
            flow: Flow descriptor

        Returns:
            True if flow is admitted, False otherwise
        """
        with self._lock:
            # Check if we can allocate required bandwidth
            if not self._can_allocate_bandwidth(flow):
                logger.warning(f"Cannot admit flow {flow.flow_id}: insufficient bandwidth")
                return False

            # Admit flow
            self._active_flows[flow.flow_id] = flow

            # Allocate bandwidth
            allocated_bw = self._allocate_bandwidth(flow)
            self._bandwidth_allocations[flow.flow_id] = allocated_bw

            logger.info(
                f"Admitted flow {flow.flow_id}: "
                f"{flow.src_device} -> {flow.dst_device}, "
                f"size={flow.size_bytes / (1024**2):.2f} MB, "
                f"allocated_bw={allocated_bw:.2f} Gbps"
            )

            return True

    def release_flow(self, flow_id: int) -> None:
        """Release resources for a completed flow."""
        with self._lock:
            if flow_id in self._active_flows:
                flow = self._active_flows[flow_id]
                del self._active_flows[flow_id]
                del self._bandwidth_allocations[flow_id]
                logger.info(f"Released flow {flow_id}")

    def get_bandwidth_allocation(self, flow_id: int) -> Optional[float]:
        """Get bandwidth allocation for a flow."""
        with self._lock:
            return self._bandwidth_allocations.get(flow_id)

    def _can_allocate_bandwidth(self, flow: FlowDescriptor) -> bool:
        """Check if we can allocate required bandwidth for flow."""
        # Get available bandwidth on path
        # This is simplified; real implementation would check entire path
        link = self.topology.get_link(flow.src_device, flow.dst_device)
        if not link:
            return False

        # Calculate currently allocated bandwidth
        allocated = sum(
            self._bandwidth_allocations.get(fid, 0.0)
            for fid, f in self._active_flows.items()
            if f.src_device == flow.src_device and f.dst_device == flow.dst_device
        )

        available = link.get_effective_bandwidth() - allocated
        return available >= flow.qos_requirements.min_bandwidth_gbps

    def _allocate_bandwidth(self, flow: FlowDescriptor) -> float:
        """Allocate bandwidth for a flow."""
        # Get link
        link = self.topology.get_link(flow.src_device, flow.dst_device)
        if not link:
            return 0.0

        # Calculate available bandwidth
        allocated = sum(
            self._bandwidth_allocations.get(fid, 0.0)
            for fid, f in self._active_flows.items()
            if f.src_device == flow.src_device and f.dst_device == flow.dst_device
        )

        available = link.get_effective_bandwidth() - allocated

        # Allocate based on QoS class
        if flow.qos_requirements.qos_class == QoSClass.LATENCY_SENSITIVE:
            # Allocate minimum + some headroom
            return min(flow.qos_requirements.min_bandwidth_gbps * 1.2, available)
        elif flow.qos_requirements.qos_class == QoSClass.THROUGHPUT_ORIENTED:
            # Allocate as much as available
            return max(flow.qos_requirements.min_bandwidth_gbps, available * 0.8)
        elif flow.qos_requirements.qos_class == QoSClass.BACKGROUND:
            # Allocate minimum or less
            return min(flow.qos_requirements.min_bandwidth_gbps, available * 0.3)
        else:
            # Best effort: fair share
            return min(flow.qos_requirements.min_bandwidth_gbps, available * 0.5)

    def rebalance_allocations(self) -> None:
        """Rebalance bandwidth allocations among active flows."""
        with self._lock:
            # Group flows by (src, dst) pair
            flow_groups = defaultdict(list)
            for flow_id, flow in self._active_flows.items():
                flow_groups[(flow.src_device, flow.dst_device)].append((flow_id, flow))

            # Rebalance each group
            for (src, dst), flows in flow_groups.items():
                link = self.topology.get_link(src, dst)
                if not link:
                    continue

                available_bw = link.get_effective_bandwidth()

                # Sort by priority
                flows.sort(key=lambda x: x[1].priority, reverse=True)

                # Allocate bandwidth by priority
                remaining_bw = available_bw
                for flow_id, flow in flows:
                    if remaining_bw <= 0:
                        self._bandwidth_allocations[flow_id] = 0.0
                        continue

                    # Allocate based on QoS class
                    if flow.qos_requirements.qos_class == QoSClass.LATENCY_SENSITIVE:
                        allocated = min(flow.qos_requirements.min_bandwidth_gbps * 1.2, remaining_bw * 0.5)
                    elif flow.qos_requirements.qos_class == QoSClass.THROUGHPUT_ORIENTED:
                        allocated = min(flow.qos_requirements.min_bandwidth_gbps, remaining_bw * 0.7)
                    else:
                        allocated = min(flow.qos_requirements.min_bandwidth_gbps, remaining_bw * 0.3)

                    self._bandwidth_allocations[flow_id] = allocated
                    remaining_bw -= allocated

                logger.debug(f"Rebalanced allocations for {src} -> {dst}: {len(flows)} flows")


class AdmissionControl:
    """
    Admission control for network flows.

    Decides whether to admit new flows based on available resources
    and QoS requirements.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        enforcer: QoSEnforcer,
        max_utilization: float = 0.9
    ):
        """
        Initialize admission control.

        Args:
            topology: Network topology
            enforcer: QoS enforcer
            max_utilization: Maximum link utilization to allow
        """
        self.topology = topology
        self.enforcer = enforcer
        self.max_utilization = max_utilization
        self._admission_stats = {
            "total_requests": 0,
            "admitted": 0,
            "rejected": 0,
            "rejection_reasons": defaultdict(int)
        }
        logger.info(f"AdmissionControl initialized (max_util={max_utilization})")

    def admit(self, flow: FlowDescriptor) -> tuple[bool, Optional[str]]:
        """
        Decide whether to admit a flow.

        Args:
            flow: Flow descriptor

        Returns:
            Tuple of (admitted, rejection_reason)
        """
        self._admission_stats["total_requests"] += 1

        # Check topology
        if flow.src_device not in self.topology.get_all_devices():
            reason = f"Unknown source device: {flow.src_device}"
            self._reject(reason)
            return False, reason

        if flow.dst_device not in self.topology.get_all_devices():
            reason = f"Unknown destination device: {flow.dst_device}"
            self._reject(reason)
            return False, reason

        # Check if path exists
        link = self.topology.get_link(flow.src_device, flow.dst_device)
        if not link:
            reason = "No path available"
            self._reject(reason)
            return False, reason

        # Check link health
        if not link.is_healthy():
            reason = f"Link unhealthy: {link.status.value}"
            self._reject(reason)
            return False, reason

        # Check utilization
        if link.get_avg_utilization() > self.max_utilization:
            reason = f"Link over-utilized: {link.get_avg_utilization():.2%}"
            self._reject(reason)
            return False, reason

        # Check if QoS requirements can be met
        if not self._can_meet_requirements(flow, link):
            reason = "Cannot meet QoS requirements"
            self._reject(reason)
            return False, reason

        # Try to admit through enforcer
        if not self.enforcer.admit_flow(flow):
            reason = "Resource allocation failed"
            self._reject(reason)
            return False, reason

        self._admission_stats["admitted"] += 1
        return True, None

    def _can_meet_requirements(self, flow: FlowDescriptor, link) -> bool:
        """Check if link can meet flow requirements."""
        qos = flow.qos_requirements

        # Check bandwidth
        if qos.min_bandwidth_gbps > 0:
            if link.get_effective_bandwidth() < qos.min_bandwidth_gbps:
                return False

        # Check latency
        if qos.max_latency_us < float('inf'):
            if link.latency_us > qos.max_latency_us:
                return False

        # Check packet loss
        if qos.max_packet_loss < 1.0:
            if link.packet_loss_rate > qos.max_packet_loss:
                return False

        return True

    def _reject(self, reason: str) -> None:
        """Record rejection."""
        self._admission_stats["rejected"] += 1
        self._admission_stats["rejection_reasons"][reason] += 1
        logger.debug(f"Flow rejected: {reason}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get admission control statistics."""
        stats = self._admission_stats.copy()
        stats["rejection_reasons"] = dict(stats["rejection_reasons"])
        return stats


class SLAMonitor:
    """
    Monitor Service Level Agreement compliance.

    Tracks performance metrics and detects SLA violations.
    """

    def __init__(self, check_interval: float = 1.0):
        """
        Initialize SLA monitor.

        Args:
            check_interval: Interval for checking SLAs (seconds)
        """
        self.check_interval = check_interval
        self._sla_metrics: Dict[int, SLAMetrics] = {}
        self._violations: deque = deque(maxlen=1000)
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        logger.info(f"SLAMonitor initialized (interval={check_interval}s)")

    def register_flow(
        self,
        flow_id: int,
        required_bandwidth: float,
        required_latency: float
    ) -> None:
        """Register a flow for SLA monitoring."""
        with self._lock:
            self._sla_metrics[flow_id] = SLAMetrics(
                flow_id=flow_id,
                required_bandwidth=required_bandwidth,
                actual_bandwidth=0.0,
                required_latency=required_latency,
                actual_latency=0.0,
                packet_loss=0.0
            )
            logger.debug(f"Registered flow {flow_id} for SLA monitoring")

    def unregister_flow(self, flow_id: int) -> None:
        """Unregister a flow from monitoring."""
        with self._lock:
            if flow_id in self._sla_metrics:
                del self._sla_metrics[flow_id]
                logger.debug(f"Unregistered flow {flow_id} from SLA monitoring")

    def update_metrics(
        self,
        flow_id: int,
        actual_bandwidth: float,
        actual_latency: float,
        packet_loss: float
    ) -> None:
        """Update actual performance metrics for a flow."""
        with self._lock:
            metrics = self._sla_metrics.get(flow_id)
            if metrics:
                metrics.actual_bandwidth = actual_bandwidth
                metrics.actual_latency = actual_latency
                metrics.packet_loss = packet_loss

    def check_sla(self, flow_id: int) -> bool:
        """
        Check if SLA is being met for a flow.

        Returns:
            True if SLA is met, False if violated
        """
        with self._lock:
            metrics = self._sla_metrics.get(flow_id)
            if not metrics:
                return True

            violated = metrics.check_violation()
            if violated:
                self._violations.append({
                    "flow_id": flow_id,
                    "timestamp": time.time(),
                    "metrics": {
                        "required_bw": metrics.required_bandwidth,
                        "actual_bw": metrics.actual_bandwidth,
                        "required_latency": metrics.required_latency,
                        "actual_latency": metrics.actual_latency,
                        "packet_loss": metrics.packet_loss
                    }
                })
                logger.warning(f"SLA violation detected for flow {flow_id}")

            return not violated

    def start(self) -> None:
        """Start SLA monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="SLAMonitor"
        )
        self._monitor_thread.start()
        logger.info("SLA monitoring started")

    def stop(self) -> None:
        """Stop SLA monitoring."""
        if not self._running:
            return

        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("SLA monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_all_slas()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in SLA monitor loop: {e}", exc_info=True)

    def _check_all_slas(self) -> None:
        """Check all registered SLAs."""
        with self._lock:
            for flow_id in list(self._sla_metrics.keys()):
                self.check_sla(flow_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get SLA monitoring statistics."""
        with self._lock:
            total_flows = len(self._sla_metrics)
            total_violations = len(self._violations)

            violation_rates = {
                flow_id: metrics.get_violation_rate()
                for flow_id, metrics in self._sla_metrics.items()
            }

            return {
                "monitored_flows": total_flows,
                "total_violations": total_violations,
                "recent_violations": list(self._violations)[-10:],
                "violation_rates": violation_rates
            }


__all__ = [
    "QoSClass",
    "QoSRequirements",
    "FlowDescriptor",
    "SLAMetrics",
    "QoSClassifier",
    "QoSEnforcer",
    "AdmissionControl",
    "SLAMonitor",
]

"""
Flow Performance Monitoring and Analytics

This module provides comprehensive monitoring and analysis of network flows,
link utilization, bottleneck detection, and performance metrics.

Components:
- FlowMetricsCollector: Per-flow metrics collection
- LinkUtilizationTracker: Link bandwidth tracking
- BottleneckDetector: Network bottleneck identification
- PerformanceAnalyzer: System-wide performance analysis
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Set, Deque
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import statistics
import json

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    LOSS_RATE = "loss_rate"
    QUEUE_DEPTH = "queue_depth"
    UTILIZATION = "utilization"
    FAIRNESS = "fairness"


@dataclass
class FlowMetrics:
    """Metrics for a single flow"""
    flow_id: str
    src_device: str
    dst_device: str

    # Throughput metrics
    bytes_sent: int = 0
    bytes_received: int = 0
    bytes_lost: int = 0
    throughput_bps: float = 0.0  # bits per second

    # Latency metrics
    latency_samples: List[float] = field(default_factory=list)
    latency_mean: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_min: float = float('inf')
    latency_max: float = 0.0

    # Loss metrics
    loss_rate: float = 0.0  # fraction of packets lost
    loss_events: int = 0

    # Queue metrics
    queue_depth_samples: List[int] = field(default_factory=list)
    queue_depth_mean: float = 0.0
    queue_depth_max: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    duration: float = 0.0

    # Transfer info
    transfers_completed: int = 0
    transfers_failed: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Exclude large sample lists from dict representation
        data['latency_samples'] = f"<{len(self.latency_samples)} samples>"
        data['queue_depth_samples'] = f"<{len(self.queue_depth_samples)} samples>"
        return data


@dataclass
class LinkMetrics:
    """Metrics for a network link"""
    link_id: str
    src_device: str
    dst_device: str

    # Capacity
    capacity_bps: float  # Link capacity in bits per second

    # Utilization
    bytes_transmitted: int = 0
    utilization: float = 0.0  # fraction of capacity used
    peak_utilization: float = 0.0

    # Active flows
    active_flows: Set[str] = field(default_factory=set)
    flow_count: int = 0

    # Timing
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['active_flows'] = list(self.active_flows)
        return data


@dataclass
class BottleneckInfo:
    """Information about detected bottleneck"""
    link_id: str
    src_device: str
    dst_device: str
    utilization: float
    affected_flows: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    detected_at: float = field(default_factory=time.time)


class FlowMetricsCollector:
    """
    Collects and maintains per-flow metrics

    Tracks throughput, latency, loss rate, and queue depths for each flow.
    Computes percentiles and statistics.
    """

    def __init__(self, history_size: int = 1000):
        """
        Initialize metrics collector

        Args:
            history_size: Number of samples to keep for statistical analysis
        """
        self.history_size = history_size
        self.flows: Dict[str, FlowMetrics] = {}
        self.lock = threading.RLock()

        logger.info("FlowMetricsCollector initialized")

    def register_flow(self, flow_id: str, src_device: str, dst_device: str) -> None:
        """Register a new flow for monitoring"""
        with self.lock:
            if flow_id not in self.flows:
                self.flows[flow_id] = FlowMetrics(
                    flow_id=flow_id,
                    src_device=src_device,
                    dst_device=dst_device
                )
                logger.debug(f"Registered flow {flow_id}: {src_device} -> {dst_device}")

    def record_transfer(self, flow_id: str, bytes_sent: int, latency: float,
                       success: bool = True, queue_depth: int = 0) -> None:
        """
        Record a transfer event

        Args:
            flow_id: Flow identifier
            bytes_sent: Number of bytes transferred
            latency: Transfer latency in seconds
            success: Whether transfer succeeded
            queue_depth: Current queue depth
        """
        with self.lock:
            if flow_id not in self.flows:
                logger.warning(f"Recording transfer for unregistered flow {flow_id}")
                return

            metrics = self.flows[flow_id]
            now = time.time()

            # Update transfer counts
            if success:
                metrics.transfers_completed += 1
                metrics.bytes_sent += bytes_sent
                metrics.bytes_received += bytes_sent
            else:
                metrics.transfers_failed += 1
                metrics.bytes_lost += bytes_sent
                metrics.loss_events += 1

            # Update latency statistics
            if success and latency > 0:
                metrics.latency_samples.append(latency)
                # Keep only recent samples
                if len(metrics.latency_samples) > self.history_size:
                    metrics.latency_samples.pop(0)

                metrics.latency_min = min(metrics.latency_min, latency)
                metrics.latency_max = max(metrics.latency_max, latency)

                # Compute percentiles
                self._update_latency_percentiles(metrics)

            # Update queue depth
            if queue_depth >= 0:
                metrics.queue_depth_samples.append(queue_depth)
                if len(metrics.queue_depth_samples) > self.history_size:
                    metrics.queue_depth_samples.pop(0)

                metrics.queue_depth_max = max(metrics.queue_depth_max, queue_depth)
                if metrics.queue_depth_samples:
                    metrics.queue_depth_mean = statistics.mean(metrics.queue_depth_samples)

            # Update timing
            metrics.last_update = now
            metrics.duration = now - metrics.start_time

            # Update throughput
            if metrics.duration > 0:
                metrics.throughput_bps = (metrics.bytes_sent * 8) / metrics.duration

            # Update loss rate
            total_bytes = metrics.bytes_sent + metrics.bytes_lost
            if total_bytes > 0:
                metrics.loss_rate = metrics.bytes_lost / total_bytes

    def _update_latency_percentiles(self, metrics: FlowMetrics) -> None:
        """Update latency percentile statistics"""
        if not metrics.latency_samples:
            return

        sorted_samples = sorted(metrics.latency_samples)
        n = len(sorted_samples)

        metrics.latency_mean = statistics.mean(sorted_samples)
        metrics.latency_p50 = sorted_samples[int(0.50 * n)]
        metrics.latency_p95 = sorted_samples[int(0.95 * n)]
        metrics.latency_p99 = sorted_samples[min(int(0.99 * n), n - 1)]

    def get_flow_metrics(self, flow_id: str) -> Optional[FlowMetrics]:
        """Get metrics for a specific flow"""
        with self.lock:
            return self.flows.get(flow_id)

    def get_all_flows(self) -> Dict[str, FlowMetrics]:
        """Get metrics for all flows"""
        with self.lock:
            return dict(self.flows)

    def get_summary(self) -> dict:
        """Get summary statistics across all flows"""
        with self.lock:
            if not self.flows:
                return {}

            total_bytes = sum(m.bytes_sent for m in self.flows.values())
            total_lost = sum(m.bytes_lost for m in self.flows.values())
            avg_throughput = statistics.mean(m.throughput_bps for m in self.flows.values())

            # Collect all latency samples
            all_latencies = []
            for m in self.flows.values():
                all_latencies.extend(m.latency_samples)

            return {
                'total_flows': len(self.flows),
                'total_bytes_sent': total_bytes,
                'total_bytes_lost': total_lost,
                'overall_loss_rate': total_lost / (total_bytes + total_lost) if (total_bytes + total_lost) > 0 else 0.0,
                'average_throughput_bps': avg_throughput,
                'average_latency': statistics.mean(all_latencies) if all_latencies else 0.0,
                'p95_latency': sorted(all_latencies)[int(0.95 * len(all_latencies))] if all_latencies else 0.0,
                'p99_latency': sorted(all_latencies)[int(0.99 * len(all_latencies))] if all_latencies else 0.0,
            }

    def reset_flow(self, flow_id: str) -> None:
        """Reset metrics for a flow"""
        with self.lock:
            if flow_id in self.flows:
                old_metrics = self.flows[flow_id]
                self.flows[flow_id] = FlowMetrics(
                    flow_id=flow_id,
                    src_device=old_metrics.src_device,
                    dst_device=old_metrics.dst_device
                )


class LinkUtilizationTracker:
    """
    Tracks bandwidth utilization on network links

    Monitors link capacity usage, identifies congested links, and tracks
    flows using each link.
    """

    def __init__(self, update_interval: float = 1.0):
        """
        Initialize link utilization tracker

        Args:
            update_interval: How often to update utilization metrics (seconds)
        """
        self.update_interval = update_interval
        self.links: Dict[str, LinkMetrics] = {}
        self.lock = threading.RLock()

        # Historical tracking
        self.utilization_history: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        logger.info("LinkUtilizationTracker initialized")

    def register_link(self, link_id: str, src_device: str, dst_device: str,
                     capacity_bps: float) -> None:
        """
        Register a network link for monitoring

        Args:
            link_id: Link identifier
            src_device: Source device
            dst_device: Destination device
            capacity_bps: Link capacity in bits per second
        """
        with self.lock:
            if link_id not in self.links:
                self.links[link_id] = LinkMetrics(
                    link_id=link_id,
                    src_device=src_device,
                    dst_device=dst_device,
                    capacity_bps=capacity_bps
                )
                logger.debug(f"Registered link {link_id}: {src_device} -> {dst_device} "
                           f"(capacity: {capacity_bps/1e9:.2f} Gbps)")

    def record_transmission(self, link_id: str, bytes_transmitted: int,
                           flow_id: Optional[str] = None) -> None:
        """
        Record a transmission on a link

        Args:
            link_id: Link identifier
            bytes_transmitted: Number of bytes transmitted
            flow_id: Optional flow identifier using this link
        """
        with self.lock:
            if link_id not in self.links:
                logger.warning(f"Recording transmission on unregistered link {link_id}")
                return

            link = self.links[link_id]
            link.bytes_transmitted += bytes_transmitted

            # Track active flow
            if flow_id:
                link.active_flows.add(flow_id)
                link.flow_count = len(link.active_flows)

            # Update utilization
            self._update_utilization(link_id)

    def _update_utilization(self, link_id: str) -> None:
        """Update link utilization metrics"""
        link = self.links[link_id]
        now = time.time()
        elapsed = now - link.last_update

        if elapsed >= self.update_interval and link.capacity_bps > 0:
            # Calculate utilization over the interval
            bits_transmitted = link.bytes_transmitted * 8
            utilization = (bits_transmitted / elapsed) / link.capacity_bps
            utilization = min(utilization, 1.0)  # Cap at 100%

            link.utilization = utilization
            link.peak_utilization = max(link.peak_utilization, utilization)
            link.last_update = now

            # Store in history
            self.utilization_history[link_id].append((now, utilization))

            # Reset counter for next interval
            link.bytes_transmitted = 0

            logger.debug(f"Link {link_id} utilization: {utilization*100:.1f}%")

    def get_link_utilization(self, link_id: str) -> float:
        """Get current utilization for a link"""
        with self.lock:
            if link_id in self.links:
                return self.links[link_id].utilization
            return 0.0

    def get_congested_links(self, threshold: float = 0.8) -> List[LinkMetrics]:
        """
        Get links with utilization above threshold

        Args:
            threshold: Utilization threshold (0.0 to 1.0)

        Returns:
            List of congested links
        """
        with self.lock:
            return [
                link for link in self.links.values()
                if link.utilization >= threshold
            ]

    def get_link_history(self, link_id: str, duration: float = 60.0) -> List[Tuple[float, float]]:
        """
        Get utilization history for a link

        Args:
            link_id: Link identifier
            duration: How far back to look (seconds)

        Returns:
            List of (timestamp, utilization) tuples
        """
        with self.lock:
            if link_id not in self.utilization_history:
                return []

            now = time.time()
            cutoff = now - duration

            return [
                (ts, util) for ts, util in self.utilization_history[link_id]
                if ts >= cutoff
            ]

    def get_all_links(self) -> Dict[str, LinkMetrics]:
        """Get metrics for all links"""
        with self.lock:
            return dict(self.links)


class BottleneckDetector:
    """
    Identifies bottleneck links in the network

    Analyzes link utilization, flow patterns, and performance degradation
    to identify bottlenecks and affected flows.
    """

    def __init__(self, utilization_tracker: LinkUtilizationTracker,
                 flow_collector: FlowMetricsCollector):
        """
        Initialize bottleneck detector

        Args:
            utilization_tracker: Link utilization tracker
            flow_collector: Flow metrics collector
        """
        self.utilization_tracker = utilization_tracker
        self.flow_collector = flow_collector
        self.bottlenecks: Dict[str, BottleneckInfo] = {}
        self.lock = threading.RLock()

        logger.info("BottleneckDetector initialized")

    def detect_bottlenecks(self) -> List[BottleneckInfo]:
        """
        Detect current bottlenecks in the network

        Returns:
            List of detected bottlenecks
        """
        with self.lock:
            bottlenecks = []

            # Check each link
            for link_id, link in self.utilization_tracker.get_all_links().items():
                # High utilization indicates potential bottleneck
                if link.utilization >= 0.9:
                    severity = 'critical'
                elif link.utilization >= 0.8:
                    severity = 'high'
                elif link.utilization >= 0.7:
                    severity = 'medium'
                else:
                    continue

                # Find affected flows
                affected_flows = list(link.active_flows)

                bottleneck = BottleneckInfo(
                    link_id=link_id,
                    src_device=link.src_device,
                    dst_device=link.dst_device,
                    utilization=link.utilization,
                    affected_flows=affected_flows,
                    severity=severity
                )

                bottlenecks.append(bottleneck)
                self.bottlenecks[link_id] = bottleneck

                logger.warning(f"Bottleneck detected on link {link_id}: "
                             f"utilization={link.utilization*100:.1f}%, "
                             f"severity={severity}, "
                             f"affected_flows={len(affected_flows)}")

            return bottlenecks

    def get_bottleneck_flows(self, link_id: str) -> List[FlowMetrics]:
        """Get flows affected by a bottleneck"""
        with self.lock:
            if link_id not in self.bottlenecks:
                return []

            bottleneck = self.bottlenecks[link_id]
            flows = []

            for flow_id in bottleneck.affected_flows:
                metrics = self.flow_collector.get_flow_metrics(flow_id)
                if metrics:
                    flows.append(metrics)

            return flows

    def get_worst_bottleneck(self) -> Optional[BottleneckInfo]:
        """Get the most severe bottleneck"""
        with self.lock:
            if not self.bottlenecks:
                return None

            # Sort by utilization
            sorted_bottlenecks = sorted(
                self.bottlenecks.values(),
                key=lambda b: b.utilization,
                reverse=True
            )

            return sorted_bottlenecks[0] if sorted_bottlenecks else None


class PerformanceAnalyzer:
    """
    Analyzes system-wide performance

    Computes fairness metrics, identifies performance issues, and provides
    recommendations for optimization.
    """

    def __init__(self, flow_collector: FlowMetricsCollector,
                 utilization_tracker: LinkUtilizationTracker,
                 bottleneck_detector: BottleneckDetector):
        """Initialize performance analyzer"""
        self.flow_collector = flow_collector
        self.utilization_tracker = utilization_tracker
        self.bottleneck_detector = bottleneck_detector
        self.lock = threading.RLock()

        logger.info("PerformanceAnalyzer initialized")

    def compute_jains_fairness_index(self) -> float:
        """
        Compute Jain's fairness index for flow throughputs

        Returns value between 0 (unfair) and 1 (perfectly fair)
        """
        with self.lock:
            flows = self.flow_collector.get_all_flows()
            if not flows:
                return 1.0

            throughputs = [m.throughput_bps for m in flows.values() if m.throughput_bps > 0]
            if not throughputs:
                return 1.0

            n = len(throughputs)
            sum_x = sum(throughputs)
            sum_x_squared = sum(x * x for x in throughputs)

            if sum_x_squared == 0:
                return 1.0

            fairness = (sum_x * sum_x) / (n * sum_x_squared)
            return fairness

    def analyze_performance(self) -> dict:
        """
        Comprehensive performance analysis

        Returns:
            Dictionary with analysis results
        """
        with self.lock:
            # Collect metrics
            flow_summary = self.flow_collector.get_summary()
            bottlenecks = self.bottleneck_detector.detect_bottlenecks()
            fairness = self.compute_jains_fairness_index()

            # Analyze links
            links = self.utilization_tracker.get_all_links()
            avg_utilization = statistics.mean(
                l.utilization for l in links.values()
            ) if links else 0.0

            # Identify issues
            issues = []

            # Check for bottlenecks
            if bottlenecks:
                issues.append({
                    'type': 'bottleneck',
                    'severity': 'high' if any(b.severity == 'critical' for b in bottlenecks) else 'medium',
                    'message': f"{len(bottlenecks)} bottleneck(s) detected",
                    'details': [asdict(b) for b in bottlenecks]
                })

            # Check fairness
            if fairness < 0.7:
                issues.append({
                    'type': 'fairness',
                    'severity': 'medium' if fairness < 0.5 else 'low',
                    'message': f"Poor fairness index: {fairness:.3f}",
                    'recommendation': 'Consider enabling fairness-aware scheduling'
                })

            # Check loss rate
            if flow_summary.get('overall_loss_rate', 0) > 0.01:
                issues.append({
                    'type': 'packet_loss',
                    'severity': 'high',
                    'message': f"High packet loss: {flow_summary['overall_loss_rate']*100:.2f}%",
                    'recommendation': 'Reduce congestion or increase capacity'
                })

            # Check latency
            p99_latency = flow_summary.get('p99_latency', 0)
            if p99_latency > 0.1:  # 100ms
                issues.append({
                    'type': 'high_latency',
                    'severity': 'medium',
                    'message': f"High P99 latency: {p99_latency*1000:.1f}ms",
                    'recommendation': 'Enable latency-optimized scheduling'
                })

            return {
                'timestamp': time.time(),
                'flow_summary': flow_summary,
                'fairness_index': fairness,
                'average_link_utilization': avg_utilization,
                'bottleneck_count': len(bottlenecks),
                'issues': issues,
                'recommendations': self._generate_recommendations(issues)
            }

    def _generate_recommendations(self, issues: List[dict]) -> List[str]:
        """Generate optimization recommendations based on issues"""
        recommendations = []

        issue_types = {issue['type'] for issue in issues}

        if 'bottleneck' in issue_types:
            recommendations.append("Enable adaptive routing to avoid bottlenecks")
            recommendations.append("Consider using BBR or TIMELY congestion control")

        if 'fairness' in issue_types:
            recommendations.append("Enable fairness-aware scheduling policy")
            recommendations.append("Use max-min fair bandwidth allocation")

        if 'packet_loss' in issue_types:
            recommendations.append("Increase buffer sizes or reduce send rate")
            recommendations.append("Enable ECN if using DCTCP")

        if 'high_latency' in issue_types:
            recommendations.append("Use latency-optimized scheduling")
            recommendations.append("Reduce queue depths")
            recommendations.append("Enable priority queuing for latency-sensitive flows")

        if not recommendations:
            recommendations.append("System is performing well - no changes needed")

        return recommendations

    def export_metrics(self, format: str = 'json') -> str:
        """
        Export all metrics in specified format

        Args:
            format: Export format ('json', 'csv')

        Returns:
            Formatted metrics string
        """
        analysis = self.analyze_performance()

        if format == 'json':
            return json.dumps(analysis, indent=2, default=str)
        elif format == 'csv':
            # Simplified CSV export
            lines = ['metric,value']
            lines.append(f"fairness_index,{analysis['fairness_index']}")
            lines.append(f"avg_utilization,{analysis['average_link_utilization']}")
            lines.append(f"bottleneck_count,{analysis['bottleneck_count']}")
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")


__all__ = [
    'FlowMetricsCollector',
    'LinkUtilizationTracker',
    'BottleneckDetector',
    'PerformanceAnalyzer',
    'FlowMetrics',
    'LinkMetrics',
    'BottleneckInfo',
    'MetricType',
]

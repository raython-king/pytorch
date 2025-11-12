"""
Congestion Control for Adaptive Flow Control.

Implements multiple congestion control algorithms including AIMD, Vegas,
and BBR for efficient bandwidth utilization and congestion avoidance.
"""

import time
import math
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque, defaultdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class CongestionState(Enum):
    """Congestion state classification."""
    NORMAL = "normal"           # No congestion
    WARNING = "warning"         # Early warning
    CONGESTED = "congested"     # Active congestion
    SEVERE = "severe"           # Severe congestion


@dataclass
class CongestionMetrics:
    """Metrics for congestion detection.

    Attributes:
        queue_length: Current queue length
        packet_loss_rate: Packet loss rate [0.0, 1.0]
        rtt: Round-trip time in seconds
        throughput: Current throughput in bytes/sec
        timestamp: Measurement timestamp
    """
    queue_length: int = 0
    packet_loss_rate: float = 0.0
    rtt: float = 0.0
    throughput: float = 0.0
    timestamp: float = field(default_factory=time.time)


class CongestionDetector:
    """Detect congestion via multiple signals.

    Monitors queue lengths, packet loss, RTT, and throughput to
    detect and classify congestion levels.
    """

    def __init__(self,
                 queue_threshold: int = 100,
                 loss_threshold: float = 0.01,
                 rtt_threshold: float = 0.1):
        """Initialize congestion detector.

        Args:
            queue_threshold: Queue length threshold for congestion
            loss_threshold: Packet loss rate threshold
            rtt_threshold: RTT threshold in seconds
        """
        self._lock = threading.RLock()

        # Thresholds
        self._queue_threshold = queue_threshold
        self._loss_threshold = loss_threshold
        self._rtt_threshold = rtt_threshold

        # Historical metrics
        self._metrics_history: Dict[str, Deque[CongestionMetrics]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Current state
        self._congestion_state: Dict[str, CongestionState] = {}
        self._baseline_rtt: Dict[str, float] = {}

    def update_metrics(self, link_id: str, metrics: CongestionMetrics) -> None:
        """Update congestion metrics for a link.

        Args:
            link_id: Link identifier
            metrics: Current metrics
        """
        with self._lock:
            self._metrics_history[link_id].append(metrics)

            # Update baseline RTT
            if link_id not in self._baseline_rtt and metrics.rtt > 0:
                self._baseline_rtt[link_id] = metrics.rtt

            # Detect congestion
            state = self._detect_congestion(link_id, metrics)
            old_state = self._congestion_state.get(link_id, CongestionState.NORMAL)

            if state != old_state:
                logger.info(f"Link {link_id} congestion state: {old_state.value} -> {state.value}")
                self._congestion_state[link_id] = state

    def _detect_congestion(self, link_id: str, metrics: CongestionMetrics) -> CongestionState:
        """Detect congestion state based on metrics.

        Args:
            link_id: Link identifier
            metrics: Current metrics

        Returns:
            Detected congestion state
        """
        indicators = []

        # Queue-based detection
        if metrics.queue_length > self._queue_threshold * 2:
            indicators.append(3)  # Severe
        elif metrics.queue_length > self._queue_threshold:
            indicators.append(2)  # Congested

        # Loss-based detection
        if metrics.packet_loss_rate > self._loss_threshold * 5:
            indicators.append(3)  # Severe
        elif metrics.packet_loss_rate > self._loss_threshold:
            indicators.append(2)  # Congested

        # RTT-based detection (Vegas-style)
        baseline_rtt = self._baseline_rtt.get(link_id, metrics.rtt)
        if baseline_rtt > 0:
            rtt_increase = (metrics.rtt - baseline_rtt) / baseline_rtt

            if rtt_increase > 0.5:
                indicators.append(3)  # Severe
            elif rtt_increase > 0.25:
                indicators.append(2)  # Congested
            elif rtt_increase > 0.1:
                indicators.append(1)  # Warning

        # Aggregate indicators
        if not indicators:
            return CongestionState.NORMAL

        max_indicator = max(indicators)
        if max_indicator >= 3:
            return CongestionState.SEVERE
        elif max_indicator >= 2:
            return CongestionState.CONGESTED
        elif max_indicator >= 1:
            return CongestionState.WARNING
        else:
            return CongestionState.NORMAL

    def get_congestion_state(self, link_id: str) -> CongestionState:
        """Get current congestion state for link.

        Args:
            link_id: Link identifier

        Returns:
            Current congestion state
        """
        with self._lock:
            return self._congestion_state.get(link_id, CongestionState.NORMAL)

    def get_metrics(self, link_id: str) -> Optional[CongestionMetrics]:
        """Get latest metrics for link.

        Args:
            link_id: Link identifier

        Returns:
            Latest metrics or None
        """
        with self._lock:
            history = self._metrics_history.get(link_id)
            if history:
                return history[-1]
            return None

    def is_congested(self, link_id: str) -> bool:
        """Check if link is congested.

        Args:
            link_id: Link identifier

        Returns:
            True if congested or severe
        """
        state = self.get_congestion_state(link_id)
        return state in (CongestionState.CONGESTED, CongestionState.SEVERE)


class CongestionController:
    """Adaptive rate control with multiple algorithms.

    Implements AIMD, Vegas, and BBR-style congestion control
    for adaptive bandwidth allocation.
    """

    def __init__(self, algorithm: str = "aimd"):
        """Initialize congestion controller.

        Args:
            algorithm: Control algorithm ("aimd", "vegas", "bbr")
        """
        self._lock = threading.RLock()
        self._algorithm = algorithm.lower()

        # Per-flow state
        self._flow_rates: Dict[str, float] = {}  # Current sending rate (bytes/sec)
        self._flow_cwnd: Dict[str, float] = {}   # Congestion window (bytes)
        self._flow_ssthresh: Dict[str, float] = {}  # Slow start threshold

        # AIMD parameters
        self._aimd_alpha = 1024 * 1024  # Additive increase: 1 MB/s
        self._aimd_beta = 0.5           # Multiplicative decrease factor

        # Vegas parameters
        self._vegas_alpha = 2           # Lower bound for extra packets
        self._vegas_beta = 4            # Upper bound for extra packets
        self._vegas_gamma = 1           # Slow start threshold parameter

        # BBR state
        self._bbr_state: Dict[str, str] = {}  # "startup", "drain", "probe_bw", "probe_rtt"
        self._bbr_max_bw: Dict[str, float] = {}
        self._bbr_min_rtt: Dict[str, float] = {}

        # Configuration
        self._min_rate = 1024 * 1024     # 1 MB/s minimum
        self._max_rate = 10 * 1024**3    # 10 GB/s maximum
        self._initial_rate = 10 * 1024 * 1024  # 10 MB/s

        logger.info(f"CongestionController initialized with {self._algorithm} algorithm")

    def initialize_flow(self, flow_id: str, initial_rate: Optional[float] = None) -> None:
        """Initialize congestion control state for a flow.

        Args:
            flow_id: Flow identifier
            initial_rate: Initial sending rate (default: configured default)
        """
        with self._lock:
            rate = initial_rate or self._initial_rate
            self._flow_rates[flow_id] = rate
            self._flow_cwnd[flow_id] = rate  # 1 second worth of data
            self._flow_ssthresh[flow_id] = float('inf')

            if self._algorithm == "bbr":
                self._bbr_state[flow_id] = "startup"
                self._bbr_max_bw[flow_id] = 0.0
                self._bbr_min_rtt[flow_id] = float('inf')

            logger.debug(f"Initialized flow {flow_id} with rate {rate / 1e6:.2f} MB/s")

    def update_rate(self, flow_id: str, metrics: CongestionMetrics,
                   congestion_state: CongestionState) -> float:
        """Update sending rate based on feedback.

        Args:
            flow_id: Flow identifier
            metrics: Current congestion metrics
            congestion_state: Current congestion state

        Returns:
            New sending rate in bytes/sec
        """
        with self._lock:
            if flow_id not in self._flow_rates:
                self.initialize_flow(flow_id)

            if self._algorithm == "aimd":
                new_rate = self._update_aimd(flow_id, congestion_state)
            elif self._algorithm == "vegas":
                new_rate = self._update_vegas(flow_id, metrics)
            elif self._algorithm == "bbr":
                new_rate = self._update_bbr(flow_id, metrics)
            else:
                new_rate = self._flow_rates[flow_id]

            # Clamp to valid range
            new_rate = max(self._min_rate, min(self._max_rate, new_rate))
            self._flow_rates[flow_id] = new_rate

            return new_rate

    def _update_aimd(self, flow_id: str, state: CongestionState) -> float:
        """Update rate using AIMD algorithm.

        Args:
            flow_id: Flow identifier
            state: Congestion state

        Returns:
            New sending rate
        """
        current_rate = self._flow_rates[flow_id]
        cwnd = self._flow_cwnd[flow_id]
        ssthresh = self._flow_ssthresh[flow_id]

        if state in (CongestionState.CONGESTED, CongestionState.SEVERE):
            # Multiplicative decrease
            new_cwnd = cwnd * self._aimd_beta
            self._flow_ssthresh[flow_id] = new_cwnd
            logger.debug(f"AIMD decrease for {flow_id}: {cwnd / 1e6:.2f} -> {new_cwnd / 1e6:.2f} MB")
        elif state == CongestionState.WARNING:
            # Cautious increase
            new_cwnd = cwnd + self._aimd_alpha * 0.5
        else:
            # Slow start or congestion avoidance
            if cwnd < ssthresh:
                # Slow start: exponential increase
                new_cwnd = cwnd * 2
            else:
                # Congestion avoidance: additive increase
                new_cwnd = cwnd + self._aimd_alpha

        self._flow_cwnd[flow_id] = new_cwnd
        return new_cwnd  # Rate equals window size for 1-second RTT assumption

    def _update_vegas(self, flow_id: str, metrics: CongestionMetrics) -> float:
        """Update rate using Vegas algorithm (delay-based).

        Args:
            flow_id: Flow identifier
            metrics: Congestion metrics

        Returns:
            New sending rate
        """
        current_rate = self._flow_rates[flow_id]
        cwnd = self._flow_cwnd[flow_id]

        if metrics.rtt <= 0:
            return current_rate

        # Calculate expected throughput based on baseline RTT
        baseline_rtt = metrics.rtt * 0.9  # Estimate of minimum RTT
        expected_throughput = cwnd / baseline_rtt
        actual_throughput = metrics.throughput

        # Calculate diff (extra packets in network)
        diff = (expected_throughput - actual_throughput) * baseline_rtt

        # Update window based on diff
        if diff < self._vegas_alpha:
            # Increase window (no congestion)
            new_cwnd = cwnd + self._aimd_alpha
        elif diff > self._vegas_beta:
            # Decrease window (congestion detected)
            new_cwnd = cwnd - self._aimd_alpha
        else:
            # Maintain window (optimal region)
            new_cwnd = cwnd

        self._flow_cwnd[flow_id] = max(self._min_rate, new_cwnd)
        return new_cwnd

    def _update_bbr(self, flow_id: str, metrics: CongestionMetrics) -> float:
        """Update rate using BBR algorithm.

        Args:
            flow_id: Flow identifier
            metrics: Congestion metrics

        Returns:
            New sending rate
        """
        state = self._bbr_state[flow_id]
        max_bw = self._bbr_max_bw[flow_id]
        min_rtt = self._bbr_min_rtt[flow_id]

        # Update max bandwidth and min RTT
        if metrics.throughput > max_bw:
            self._bbr_max_bw[flow_id] = metrics.throughput
            max_bw = metrics.throughput

        if metrics.rtt > 0 and metrics.rtt < min_rtt:
            self._bbr_min_rtt[flow_id] = metrics.rtt
            min_rtt = metrics.rtt

        # State machine
        current_rate = self._flow_rates[flow_id]

        if state == "startup":
            # Exponential growth until loss
            new_rate = current_rate * 2
            if metrics.packet_loss_rate > 0:
                self._bbr_state[flow_id] = "drain"
                logger.debug(f"BBR {flow_id}: startup -> drain")

        elif state == "drain":
            # Drain queue to min_rtt
            new_rate = current_rate * 0.8
            if min_rtt < float('inf') and metrics.rtt <= min_rtt * 1.25:
                self._bbr_state[flow_id] = "probe_bw"
                logger.debug(f"BBR {flow_id}: drain -> probe_bw")

        elif state == "probe_bw":
            # Probe for bandwidth changes
            if max_bw > 0:
                new_rate = max_bw * 1.25  # Pacing gain
            else:
                new_rate = current_rate

            # Periodically probe RTT
            if time.time() % 10 < 0.2:  # 200ms every 10 seconds
                self._bbr_state[flow_id] = "probe_rtt"
                logger.debug(f"BBR {flow_id}: probe_bw -> probe_rtt")

        else:  # probe_rtt
            # Reduce rate to probe min RTT
            new_rate = current_rate * 0.5
            if metrics.rtt > 0 and metrics.rtt <= min_rtt * 1.1:
                self._bbr_state[flow_id] = "probe_bw"
                logger.debug(f"BBR {flow_id}: probe_rtt -> probe_bw")

        return new_rate

    def get_flow_rate(self, flow_id: str) -> float:
        """Get current sending rate for flow.

        Args:
            flow_id: Flow identifier

        Returns:
            Current rate in bytes/sec
        """
        with self._lock:
            return self._flow_rates.get(flow_id, self._initial_rate)

    def remove_flow(self, flow_id: str) -> None:
        """Remove flow state.

        Args:
            flow_id: Flow identifier
        """
        with self._lock:
            self._flow_rates.pop(flow_id, None)
            self._flow_cwnd.pop(flow_id, None)
            self._flow_ssthresh.pop(flow_id, None)

            if self._algorithm == "bbr":
                self._bbr_state.pop(flow_id, None)
                self._bbr_max_bw.pop(flow_id, None)
                self._bbr_min_rtt.pop(flow_id, None)


class BackpressureManager:
    """Apply backpressure to senders when congestion detected.

    Manages flow control signals to prevent overwhelming receivers
    and network links during congestion.
    """

    def __init__(self):
        """Initialize backpressure manager."""
        self._lock = threading.RLock()

        # Per-flow backpressure state
        self._flow_paused: Dict[str, bool] = {}
        self._flow_rate_limits: Dict[str, float] = {}

        # Callbacks
        self._pause_callbacks: List = []
        self._resume_callbacks: List = []

    def apply_backpressure(self, flow_id: str, rate_limit: Optional[float] = None) -> None:
        """Apply backpressure to a flow.

        Args:
            flow_id: Flow identifier
            rate_limit: Optional rate limit (bytes/sec), or None to pause
        """
        with self._lock:
            if rate_limit is None:
                # Pause flow
                if not self._flow_paused.get(flow_id, False):
                    self._flow_paused[flow_id] = True
                    logger.info(f"Paused flow {flow_id}")
                    self._notify_pause(flow_id)
            else:
                # Rate limit flow
                self._flow_rate_limits[flow_id] = rate_limit
                logger.info(f"Rate limited flow {flow_id} to {rate_limit / 1e6:.2f} MB/s")

    def release_backpressure(self, flow_id: str) -> None:
        """Release backpressure from a flow.

        Args:
            flow_id: Flow identifier
        """
        with self._lock:
            was_paused = self._flow_paused.pop(flow_id, False)
            self._flow_rate_limits.pop(flow_id, None)

            if was_paused:
                logger.info(f"Resumed flow {flow_id}")
                self._notify_resume(flow_id)

    def is_paused(self, flow_id: str) -> bool:
        """Check if flow is paused.

        Args:
            flow_id: Flow identifier

        Returns:
            True if paused
        """
        with self._lock:
            return self._flow_paused.get(flow_id, False)

    def get_rate_limit(self, flow_id: str) -> Optional[float]:
        """Get rate limit for flow.

        Args:
            flow_id: Flow identifier

        Returns:
            Rate limit in bytes/sec or None if unlimited
        """
        with self._lock:
            return self._flow_rate_limits.get(flow_id)

    def register_pause_callback(self, callback) -> None:
        """Register callback for pause events.

        Args:
            callback: Function(flow_id: str) -> None
        """
        with self._lock:
            self._pause_callbacks.append(callback)

    def register_resume_callback(self, callback) -> None:
        """Register callback for resume events.

        Args:
            callback: Function(flow_id: str) -> None
        """
        with self._lock:
            self._resume_callbacks.append(callback)

    def _notify_pause(self, flow_id: str) -> None:
        """Notify pause callbacks."""
        for callback in self._pause_callbacks:
            try:
                callback(flow_id)
            except Exception as e:
                logger.error(f"Error in pause callback: {e}")

    def _notify_resume(self, flow_id: str) -> None:
        """Notify resume callbacks."""
        for callback in self._resume_callbacks:
            try:
                callback(flow_id)
            except Exception as e:
                logger.error(f"Error in resume callback: {e}")


class ExplicitCongestionNotification:
    """ECN-style congestion signaling.

    Provides explicit congestion notifications to enable proactive
    congestion avoidance without packet loss.
    """

    def __init__(self):
        """Initialize ECN manager."""
        self._lock = threading.RLock()

        # ECN marking thresholds
        self._marking_threshold = 0.8  # Mark at 80% capacity
        self._severe_threshold = 0.95  # Severe at 95% capacity

        # Per-link ECN state
        self._link_marks: Dict[str, int] = defaultdict(int)
        self._link_packets: Dict[str, int] = defaultdict(int)

    def mark_packet(self, link_id: str, utilization: float) -> bool:
        """Mark packet with ECN if congestion detected.

        Args:
            link_id: Link identifier
            utilization: Current link utilization [0.0, 1.0]

        Returns:
            True if packet should be marked
        """
        with self._lock:
            self._link_packets[link_id] += 1

            # Probabilistic marking based on utilization
            if utilization >= self._severe_threshold:
                # Always mark in severe congestion
                should_mark = True
            elif utilization >= self._marking_threshold:
                # Probabilistic marking
                prob = (utilization - self._marking_threshold) / (1.0 - self._marking_threshold)
                should_mark = np.random.random() < prob
            else:
                should_mark = False

            if should_mark:
                self._link_marks[link_id] += 1

            return should_mark

    def get_marking_rate(self, link_id: str) -> float:
        """Get ECN marking rate for link.

        Args:
            link_id: Link identifier

        Returns:
            Marking rate [0.0, 1.0]
        """
        with self._lock:
            packets = self._link_packets.get(link_id, 0)
            if packets == 0:
                return 0.0

            marks = self._link_marks.get(link_id, 0)
            return marks / packets

    def reset_statistics(self, link_id: str) -> None:
        """Reset ECN statistics for link.

        Args:
            link_id: Link identifier
        """
        with self._lock:
            self._link_marks[link_id] = 0
            self._link_packets[link_id] = 0

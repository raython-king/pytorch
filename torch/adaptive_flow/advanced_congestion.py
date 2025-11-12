"""
Advanced Congestion Control Algorithms for Adaptive Flow Control

This module implements production-ready congestion control algorithms:
- BBR: Bottleneck Bandwidth and RTT-based control
- Vegas: TCP Vegas delay-based control
- DCTCP: Data Center TCP with ECN support
- TIMELY: RTT-based congestion control for datacenters

Each algorithm adapts to network conditions to optimize throughput while
minimizing latency and packet loss.
"""

import time
import math
import logging
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CongestionState(Enum):
    """Congestion control states"""
    STARTUP = "startup"
    DRAIN = "drain"
    PROBE_BW = "probe_bw"
    PROBE_RTT = "probe_rtt"
    STEADY = "steady"
    CONGESTION_AVOIDANCE = "congestion_avoidance"
    FAST_RECOVERY = "fast_recovery"


@dataclass
class RTTSample:
    """RTT measurement sample"""
    timestamp: float
    rtt: float  # Round-trip time in seconds
    bytes_sent: int
    bytes_acked: int


@dataclass
class CongestionMetrics:
    """Metrics tracked by congestion control"""
    bandwidth_estimate: float = 0.0  # bytes per second
    min_rtt: float = float('inf')  # minimum RTT observed
    current_rtt: float = 0.0  # current RTT
    rtt_variance: float = 0.0  # RTT variance
    cwnd: int = 10  # congestion window (segments)
    ssthresh: int = 65535  # slow start threshold
    pacing_rate: float = 0.0  # pacing rate (bytes per second)
    in_flight: int = 0  # bytes in flight
    delivered: int = 0  # total bytes delivered
    lost: int = 0  # total bytes lost
    bdp: float = 0.0  # bandwidth-delay product


class CongestionController:
    """Base class for congestion control algorithms"""

    def __init__(self, initial_cwnd: int = 10, mss: int = 1500):
        """
        Initialize congestion controller

        Args:
            initial_cwnd: Initial congestion window size (in segments)
            mss: Maximum segment size in bytes
        """
        self.mss = mss
        self.metrics = CongestionMetrics(cwnd=initial_cwnd)
        self.state = CongestionState.STARTUP
        self.rtt_samples: Deque[RTTSample] = deque(maxlen=100)
        self.lock = threading.RLock()
        self.start_time = time.time()

    def on_ack(self, bytes_acked: int, rtt: float) -> None:
        """
        Called when acknowledgment is received

        Args:
            bytes_acked: Number of bytes acknowledged
            rtt: Round-trip time for this acknowledgment
        """
        raise NotImplementedError

    def on_loss(self, bytes_lost: int) -> None:
        """
        Called when packet loss is detected

        Args:
            bytes_lost: Number of bytes lost
        """
        raise NotImplementedError

    def get_cwnd(self) -> int:
        """Get current congestion window size in bytes"""
        with self.lock:
            return self.metrics.cwnd * self.mss

    def get_pacing_rate(self) -> float:
        """Get current pacing rate in bytes per second"""
        with self.lock:
            return self.metrics.pacing_rate

    def update_rtt(self, rtt: float, bytes_sent: int, bytes_acked: int) -> None:
        """Update RTT measurements"""
        with self.lock:
            sample = RTTSample(
                timestamp=time.time(),
                rtt=rtt,
                bytes_sent=bytes_sent,
                bytes_acked=bytes_acked
            )
            self.rtt_samples.append(sample)

            self.metrics.current_rtt = rtt
            self.metrics.min_rtt = min(self.metrics.min_rtt, rtt)

            # Calculate RTT variance (exponential moving average)
            if self.metrics.rtt_variance == 0.0:
                self.metrics.rtt_variance = rtt / 2
            else:
                diff = abs(rtt - self.metrics.current_rtt)
                self.metrics.rtt_variance = 0.75 * self.metrics.rtt_variance + 0.25 * diff


class BBR_Controller(CongestionController):
    """
    BBR (Bottleneck Bandwidth and RTT) Congestion Control

    BBR creates explicit model of the network path by continuously measuring
    bottleneck bandwidth and round-trip propagation time. It uses this model
    to control both sending rate (pacing) and the maximum volume of data
    in flight.

    Reference: "BBR: Congestion-Based Congestion Control" (ACM Queue 2016)
    """

    # BBR parameters
    PROBE_BW_GAIN_CYCLE = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    PROBE_BW_CYCLE_LEN = 8
    PROBE_RTT_DURATION = 0.2  # 200ms
    STARTUP_GAIN = 2.89
    DRAIN_GAIN = 1.0 / STARTUP_GAIN

    def __init__(self, initial_cwnd: int = 10, mss: int = 1500):
        super().__init__(initial_cwnd, mss)
        self.state = CongestionState.STARTUP

        # BBR state
        self.btlbw = 0.0  # bottleneck bandwidth estimate
        self.rtprop = float('inf')  # round-trip propagation time
        self.rtprop_stamp = time.time()
        self.cycle_index = 0
        self.cycle_stamp = time.time()
        self.probe_rtt_done_stamp = 0.0
        self.prior_cwnd = 0
        self.packet_conservation = False

        # Bandwidth estimation
        self.bw_samples: Deque[Tuple[float, float]] = deque(maxlen=10)  # (timestamp, bw)

        logger.info("BBR congestion control initialized")

    def on_ack(self, bytes_acked: int, rtt: float) -> None:
        """Process acknowledgment and update BBR state"""
        with self.lock:
            self.update_rtt(rtt, 0, bytes_acked)
            self.metrics.delivered += bytes_acked

            # Update bandwidth estimate
            self._update_bandwidth(bytes_acked, rtt)

            # Update RTT propagation estimate
            self._update_rtprop(rtt)

            # Update model and control parameters
            self._update_model()

            # State machine
            self._update_state()

            # Set pacing rate and congestion window
            self._set_pacing_rate()
            self._set_cwnd()

    def on_loss(self, bytes_lost: int) -> None:
        """Handle packet loss"""
        with self.lock:
            self.metrics.lost += bytes_lost

            # BBR doesn't immediately react to loss like traditional TCP
            # It relies on bandwidth and RTT measurements
            logger.debug(f"BBR: Packet loss detected: {bytes_lost} bytes")

    def _update_bandwidth(self, bytes_acked: int, rtt: float) -> None:
        """Update bottleneck bandwidth estimate"""
        if rtt <= 0:
            return

        # Calculate delivery rate
        delivery_rate = bytes_acked / rtt

        # Store sample
        now = time.time()
        self.bw_samples.append((now, delivery_rate))

        # Take windowed maximum over last 10 RTTs
        self.btlbw = max(bw for _, bw in self.bw_samples) if self.bw_samples else delivery_rate
        self.metrics.bandwidth_estimate = self.btlbw

    def _update_rtprop(self, rtt: float) -> None:
        """Update round-trip propagation time estimate"""
        now = time.time()

        # Update minimum RTT
        if rtt < self.rtprop:
            self.rtprop = rtt
            self.rtprop_stamp = now

        # Expire old rtprop estimate after 10 seconds
        if now > self.rtprop_stamp + 10.0:
            self.rtprop = rtt
            self.rtprop_stamp = now

    def _update_model(self) -> None:
        """Update BBR model (BDP estimate)"""
        if self.rtprop != float('inf') and self.btlbw > 0:
            self.metrics.bdp = self.btlbw * self.rtprop
        else:
            self.metrics.bdp = self.metrics.cwnd * self.mss

    def _update_state(self) -> None:
        """Update BBR state machine"""
        now = time.time()

        if self.state == CongestionState.STARTUP:
            # Exit startup if bandwidth plateaus
            if len(self.bw_samples) >= 3:
                recent_bw = [bw for _, bw in list(self.bw_samples)[-3:]]
                if max(recent_bw) < 1.25 * min(recent_bw):
                    self.state = CongestionState.DRAIN
                    self.cycle_stamp = now
                    logger.debug("BBR: Transitioning to DRAIN state")

        elif self.state == CongestionState.DRAIN:
            # Exit drain when in-flight <= BDP
            if self.metrics.in_flight <= self._get_target_cwnd():
                self.state = CongestionState.PROBE_BW
                self.cycle_index = 0
                self.cycle_stamp = now
                logger.debug("BBR: Transitioning to PROBE_BW state")

        elif self.state == CongestionState.PROBE_BW:
            # Cycle through gain values
            if now > self.cycle_stamp + self.metrics.min_rtt:
                self.cycle_index = (self.cycle_index + 1) % self.PROBE_BW_CYCLE_LEN
                self.cycle_stamp = now

            # Enter PROBE_RTT if rtprop estimate is old
            if now > self.rtprop_stamp + 10.0:
                self.state = CongestionState.PROBE_RTT
                self.probe_rtt_done_stamp = 0.0
                self.prior_cwnd = self.metrics.cwnd
                logger.debug("BBR: Transitioning to PROBE_RTT state")

        elif self.state == CongestionState.PROBE_RTT:
            # Reduce cwnd to min
            if self.probe_rtt_done_stamp == 0.0:
                if self.metrics.in_flight <= 4 * self.mss:
                    self.probe_rtt_done_stamp = now + self.PROBE_RTT_DURATION
            elif now > self.probe_rtt_done_stamp:
                self.rtprop_stamp = now
                self.state = CongestionState.PROBE_BW
                self.cycle_index = 0
                self.cycle_stamp = now
                logger.debug("BBR: Transitioning back to PROBE_BW state")

    def _get_pacing_gain(self) -> float:
        """Get pacing gain based on current state"""
        if self.state == CongestionState.STARTUP:
            return self.STARTUP_GAIN
        elif self.state == CongestionState.DRAIN:
            return self.DRAIN_GAIN
        elif self.state == CongestionState.PROBE_BW:
            return self.PROBE_BW_GAIN_CYCLE[self.cycle_index]
        elif self.state == CongestionState.PROBE_RTT:
            return 1.0
        return 1.0

    def _get_cwnd_gain(self) -> float:
        """Get cwnd gain based on current state"""
        if self.state == CongestionState.STARTUP:
            return self.STARTUP_GAIN
        elif self.state == CongestionState.DRAIN:
            return self.DRAIN_GAIN
        elif self.state == CongestionState.PROBE_BW:
            return 2.0
        elif self.state == CongestionState.PROBE_RTT:
            return 1.0
        return 1.0

    def _set_pacing_rate(self) -> None:
        """Set pacing rate based on BBR model"""
        gain = self._get_pacing_gain()
        if self.btlbw > 0:
            self.metrics.pacing_rate = gain * self.btlbw
        else:
            # Default to high rate in startup
            self.metrics.pacing_rate = self.STARTUP_GAIN * 10 * 1024 * 1024  # 10 MB/s

    def _get_target_cwnd(self) -> int:
        """Get target congestion window"""
        gain = self._get_cwnd_gain()
        if self.metrics.bdp > 0:
            cwnd = gain * self.metrics.bdp
        else:
            cwnd = self.metrics.cwnd * self.mss

        # Minimum cwnd
        return max(int(cwnd), 4 * self.mss)

    def _set_cwnd(self) -> None:
        """Set congestion window based on BBR model"""
        if self.state == CongestionState.PROBE_RTT:
            self.metrics.cwnd = 4  # Minimum window
        else:
            target = self._get_target_cwnd()
            self.metrics.cwnd = max(target // self.mss, 4)


class Vegas_Controller(CongestionController):
    """
    TCP Vegas: Delay-based Congestion Control

    Vegas uses changes in RTT to detect congestion before packet loss occurs.
    It compares expected throughput (cwnd/baseRTT) with actual throughput
    (cwnd/RTT) and adjusts cwnd accordingly.

    Reference: "TCP Vegas: End to End Congestion Avoidance on a Global Internet" (IEEE JSAC 1995)
    """

    # Vegas parameters
    ALPHA = 2  # Minimum number of extra packets in network
    BETA = 4   # Maximum number of extra packets in network
    GAMMA = 1  # Parameter for slow start

    def __init__(self, initial_cwnd: int = 10, mss: int = 1500):
        super().__init__(initial_cwnd, mss)
        self.state = CongestionState.STARTUP

        # Vegas state
        self.base_rtt = float('inf')
        self.doing_vegas_now = False
        self.vegas_enabled = False
        self.expected_throughput = 0.0
        self.actual_throughput = 0.0
        self.diff = 0.0

        logger.info("TCP Vegas congestion control initialized")

    def on_ack(self, bytes_acked: int, rtt: float) -> None:
        """Process acknowledgment using Vegas algorithm"""
        with self.lock:
            self.update_rtt(rtt, 0, bytes_acked)
            self.metrics.delivered += bytes_acked

            # Update base RTT (minimum observed)
            if rtt < self.base_rtt:
                self.base_rtt = rtt
                self.metrics.min_rtt = rtt

            # Enable Vegas after collecting base RTT samples
            if len(self.rtt_samples) >= 2 and not self.vegas_enabled:
                self.vegas_enabled = True
                self.state = CongestionState.CONGESTION_AVOIDANCE
                logger.debug("Vegas: Enabled congestion avoidance")

            if not self.vegas_enabled:
                # Traditional slow start
                self._slow_start(bytes_acked)
            else:
                # Vegas congestion avoidance
                self._vegas_congestion_avoidance(rtt)

            # Update pacing rate
            self._update_pacing_rate()

    def on_loss(self, bytes_lost: int) -> None:
        """Handle packet loss - reduce cwnd"""
        with self.lock:
            self.metrics.lost += bytes_lost

            # Multiplicative decrease
            self.metrics.cwnd = max(self.metrics.cwnd // 2, 2)
            self.metrics.ssthresh = self.metrics.cwnd
            self.state = CongestionState.FAST_RECOVERY

            logger.debug(f"Vegas: Loss detected, cwnd reduced to {self.metrics.cwnd}")

    def _slow_start(self, bytes_acked: int) -> None:
        """Traditional slow start phase"""
        segments_acked = bytes_acked // self.mss
        self.metrics.cwnd += segments_acked

        if self.metrics.cwnd >= self.metrics.ssthresh:
            self.state = CongestionState.CONGESTION_AVOIDANCE

    def _vegas_congestion_avoidance(self, rtt: float) -> None:
        """Vegas congestion avoidance algorithm"""
        if self.base_rtt == float('inf') or rtt == 0:
            return

        # Calculate expected throughput: cwnd / base_rtt
        cwnd_bytes = self.metrics.cwnd * self.mss
        self.expected_throughput = cwnd_bytes / self.base_rtt

        # Calculate actual throughput: cwnd / rtt
        self.actual_throughput = cwnd_bytes / rtt

        # Calculate diff in segments
        diff_bytes = (self.expected_throughput - self.actual_throughput) * self.base_rtt
        self.diff = diff_bytes / self.mss

        # Adjust cwnd based on diff
        if self.diff < self.ALPHA:
            # Too few packets in network, increase cwnd
            self.metrics.cwnd += 1
            logger.debug(f"Vegas: Increasing cwnd to {self.metrics.cwnd} (diff={self.diff:.2f})")
        elif self.diff > self.BETA:
            # Too many packets in network, decrease cwnd
            self.metrics.cwnd = max(self.metrics.cwnd - 1, 2)
            logger.debug(f"Vegas: Decreasing cwnd to {self.metrics.cwnd} (diff={self.diff:.2f})")
        else:
            # In the sweet spot, keep cwnd stable
            logger.debug(f"Vegas: Maintaining cwnd at {self.metrics.cwnd} (diff={self.diff:.2f})")

        # Estimate bandwidth
        self.metrics.bandwidth_estimate = self.actual_throughput

    def _update_pacing_rate(self) -> None:
        """Update pacing rate based on current cwnd and RTT"""
        if self.metrics.current_rtt > 0:
            self.metrics.pacing_rate = (self.metrics.cwnd * self.mss) / self.metrics.current_rtt
        else:
            self.metrics.pacing_rate = self.metrics.cwnd * self.mss * 100  # Default high rate


class DCTCP_Controller(CongestionController):
    """
    DCTCP (Data Center TCP): ECN-based Congestion Control

    DCTCP uses Explicit Congestion Notification (ECN) marks to react to
    congestion in proportion to its extent. It maintains high burst tolerance
    and low buffer occupancy simultaneously.

    Reference: "DCTCP: Efficient Packet Transport for the Commoditized Data Center" (SIGCOMM 2010)
    """

    # DCTCP parameters
    DCTCP_G = 0.0625  # Weight for EWMA (1/16)

    def __init__(self, initial_cwnd: int = 10, mss: int = 1500):
        super().__init__(initial_cwnd, mss)
        self.state = CongestionState.STARTUP

        # DCTCP state
        self.alpha = 0.0  # Fraction of packets marked
        self.acked_bytes_ecn = 0  # Bytes acked with ECN mark
        self.acked_bytes_total = 0  # Total bytes acked
        self.prior_rcv_nxt = 0
        self.ce_state = False  # Current CE state
        self.delayed_ack_reserved = False

        # Slow start parameters
        self.slow_start_threshold = 65535

        logger.info("DCTCP congestion control initialized")

    def on_ack(self, bytes_acked: int, rtt: float, ecn_marked: bool = False) -> None:
        """
        Process acknowledgment with ECN information

        Args:
            bytes_acked: Number of bytes acknowledged
            rtt: Round-trip time
            ecn_marked: Whether this ACK indicates ECN marking
        """
        with self.lock:
            self.update_rtt(rtt, 0, bytes_acked)
            self.metrics.delivered += bytes_acked

            # Track ECN marks
            self.acked_bytes_total += bytes_acked
            if ecn_marked:
                self.acked_bytes_ecn += bytes_acked

            # Update alpha (fraction of marked packets)
            if self.acked_bytes_total > 0:
                measured_alpha = self.acked_bytes_ecn / self.acked_bytes_total
                self.alpha = (1 - self.DCTCP_G) * self.alpha + self.DCTCP_G * measured_alpha

            # Congestion window update
            if self.state == CongestionState.STARTUP:
                self._slow_start(bytes_acked)
            else:
                self._congestion_avoidance(bytes_acked)

            # Update pacing rate
            self._update_pacing_rate()

    def on_loss(self, bytes_lost: int) -> None:
        """Handle packet loss - DCTCP style reduction"""
        with self.lock:
            self.metrics.lost += bytes_lost

            # DCTCP reduces cwnd based on alpha
            reduction_factor = (1.0 - self.alpha / 2.0)
            self.metrics.cwnd = max(int(self.metrics.cwnd * reduction_factor), 2)
            self.metrics.ssthresh = self.metrics.cwnd
            self.state = CongestionState.FAST_RECOVERY

            logger.debug(f"DCTCP: Loss detected, cwnd reduced to {self.metrics.cwnd} (alpha={self.alpha:.4f})")

            # Reset ECN counters for new measurement window
            self.acked_bytes_total = 0
            self.acked_bytes_ecn = 0

    def _slow_start(self, bytes_acked: int) -> None:
        """Slow start phase"""
        segments_acked = bytes_acked // self.mss
        self.metrics.cwnd += segments_acked

        if self.metrics.cwnd >= self.metrics.ssthresh:
            self.state = CongestionState.CONGESTION_AVOIDANCE
            logger.debug("DCTCP: Transitioning to congestion avoidance")

    def _congestion_avoidance(self, bytes_acked: int) -> None:
        """Congestion avoidance phase"""
        # Additive increase
        segments_acked = bytes_acked // self.mss
        cwnd_increase = segments_acked / self.metrics.cwnd
        self.metrics.cwnd += int(cwnd_increase)

    def _update_pacing_rate(self) -> None:
        """Update pacing rate"""
        if self.metrics.current_rtt > 0:
            self.metrics.pacing_rate = (self.metrics.cwnd * self.mss) / self.metrics.current_rtt
        else:
            self.metrics.pacing_rate = self.metrics.cwnd * self.mss * 100

    def get_alpha(self) -> float:
        """Get current alpha value (fraction of marked packets)"""
        with self.lock:
            return self.alpha


class TIMELY_Controller(CongestionController):
    """
    TIMELY: RTT-based Congestion Control for Datacenters

    TIMELY uses RTT gradient to detect congestion and adjust sending rate.
    It's designed for RDMA networks in datacenters where RTT is a reliable
    congestion signal.

    Reference: "TIMELY: RTT-based Congestion Control for the Datacenter" (SIGCOMM 2015)
    """

    # TIMELY parameters
    T_LOW = 0.000050  # 50 microseconds
    T_HIGH = 0.000500  # 500 microseconds
    MIN_RTT = 0.000020  # 20 microseconds
    ALPHA = 0.875  # EWMA weight for RTT
    BETA = 0.8  # EWMA weight for RTT gradient
    DELTA = 0.5  # Additive increment step (Mbps)
    HAI_THRESH = 5  # Hyperactive increase threshold

    def __init__(self, initial_rate: float = 10.0 * 1024 * 1024, mss: int = 1500):
        """
        Initialize TIMELY controller

        Args:
            initial_rate: Initial sending rate in bytes per second
            mss: Maximum segment size in bytes
        """
        super().__init__(10, mss)

        # TIMELY state
        self.rate = initial_rate  # Sending rate in bytes/sec
        self.prev_rtt = 0.0
        self.rtt_diff = 0.0  # RTT gradient
        self.smoothed_rtt = 0.0
        self.smoothed_gradient = 0.0
        self.hai_counter = 0  # Hyperactive increase counter

        logger.info(f"TIMELY congestion control initialized with rate {initial_rate/1e6:.2f} MB/s")

    def on_ack(self, bytes_acked: int, rtt: float) -> None:
        """Process acknowledgment and update rate using TIMELY algorithm"""
        with self.lock:
            self.update_rtt(rtt, 0, bytes_acked)
            self.metrics.delivered += bytes_acked

            # Initialize smoothed RTT
            if self.smoothed_rtt == 0.0:
                self.smoothed_rtt = rtt
                self.prev_rtt = rtt

            # Calculate RTT gradient
            if self.prev_rtt > 0:
                self.rtt_diff = rtt - self.prev_rtt

                # Smooth the gradient
                if self.smoothed_gradient == 0.0:
                    self.smoothed_gradient = self.rtt_diff
                else:
                    self.smoothed_gradient = (self.BETA * self.smoothed_gradient +
                                             (1 - self.BETA) * self.rtt_diff)

            # Update smoothed RTT
            self.smoothed_rtt = self.ALPHA * self.smoothed_rtt + (1 - self.ALPHA) * rtt

            # Rate update algorithm
            self._update_rate(rtt)

            # Save for next iteration
            self.prev_rtt = rtt

            # Update metrics
            self.metrics.pacing_rate = self.rate
            if self.smoothed_rtt > 0:
                self.metrics.cwnd = int((self.rate * self.smoothed_rtt) / self.mss)

    def on_loss(self, bytes_lost: int) -> None:
        """Handle packet loss - reduce rate significantly"""
        with self.lock:
            self.metrics.lost += bytes_lost

            # Aggressive rate reduction on loss
            self.rate = self.rate * 0.5
            self.hai_counter = 0

            logger.debug(f"TIMELY: Loss detected, rate reduced to {self.rate/1e6:.2f} MB/s")

    def _update_rate(self, rtt: float) -> None:
        """Update sending rate based on RTT measurements"""
        # Normalize RTT difference
        rtt_diff_normalized = self.rtt_diff / self.MIN_RTT if self.MIN_RTT > 0 else 0

        # Compute new rate
        if rtt < self.T_LOW:
            # RTT is low, increase rate additively
            delta_bytes = self.DELTA * 1024 * 1024  # Convert Mbps to bytes/sec
            new_rate = self.rate + delta_bytes
            logger.debug(f"TIMELY: RTT low ({rtt*1e6:.1f}us), increasing rate")

        elif rtt > self.T_HIGH:
            # RTT is high, decrease rate multiplicatively
            # Use gradient to determine decrease
            if self.smoothed_gradient > 0:
                # RTT is increasing, decrease more aggressively
                beta = (1.0 - self.T_LOW / rtt) * (self.smoothed_gradient / (self.MIN_RTT if self.MIN_RTT > 0 else 1e-6))
                beta = min(max(beta, 0.0), 1.0)
                new_rate = self.rate * (1.0 - beta / 2.0)
                logger.debug(f"TIMELY: RTT high ({rtt*1e6:.1f}us) and increasing, decreasing rate (beta={beta:.4f})")
            else:
                # RTT is high but decreasing, small decrease
                new_rate = self.rate * 0.95
                logger.debug(f"TIMELY: RTT high ({rtt*1e6:.1f}us) but decreasing, small decrease")

        else:
            # RTT is in the target range
            if self.smoothed_gradient <= 0:
                # RTT is stable or decreasing, additive increase
                delta_bytes = self.DELTA * 1024 * 1024
                new_rate = self.rate + delta_bytes

                # Hyperactive increase
                self.hai_counter += 1
                if self.hai_counter >= self.HAI_THRESH:
                    new_rate = self.rate + 5 * delta_bytes
                    self.hai_counter = 0
                    logger.debug("TIMELY: Hyperactive increase")
            else:
                # RTT is increasing, multiplicative decrease
                gradient_normalized = self.smoothed_gradient / (self.MIN_RTT if self.MIN_RTT > 0 else 1e-6)
                beta = min(gradient_normalized, 1.0)
                new_rate = self.rate * (1.0 - beta / 2.0)
                self.hai_counter = 0
                logger.debug(f"TIMELY: RTT increasing, decreasing rate (gradient={self.smoothed_gradient*1e6:.1f}us)")

        # Apply new rate with minimum
        min_rate = 1.0 * 1024 * 1024  # 1 MB/s minimum
        self.rate = max(new_rate, min_rate)

    def get_rate(self) -> float:
        """Get current sending rate in bytes per second"""
        with self.lock:
            return self.rate

    def get_gradient(self) -> float:
        """Get current RTT gradient"""
        with self.lock:
            return self.smoothed_gradient


def create_controller(algorithm: str, **kwargs) -> CongestionController:
    """
    Factory function to create congestion controller

    Args:
        algorithm: Algorithm name ('bbr', 'vegas', 'dctcp', 'timely')
        **kwargs: Algorithm-specific parameters

    Returns:
        CongestionController instance

    Raises:
        ValueError: If algorithm is not supported
    """
    algorithm = algorithm.lower()

    if algorithm == 'bbr':
        return BBR_Controller(**kwargs)
    elif algorithm == 'vegas':
        return Vegas_Controller(**kwargs)
    elif algorithm == 'dctcp':
        return DCTCP_Controller(**kwargs)
    elif algorithm == 'timely':
        return TIMELY_Controller(**kwargs)
    else:
        raise ValueError(f"Unsupported congestion control algorithm: {algorithm}")


__all__ = [
    'CongestionController',
    'BBR_Controller',
    'Vegas_Controller',
    'DCTCP_Controller',
    'TIMELY_Controller',
    'CongestionState',
    'CongestionMetrics',
    'RTTSample',
    'create_controller',
]

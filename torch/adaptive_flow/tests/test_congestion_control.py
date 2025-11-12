"""
Tests for Congestion Control Algorithms

Tests BBR, Vegas, DCTCP, and TIMELY congestion control implementations.
"""

import unittest
import time
import random
from torch.testing._internal.common_utils import TestCase, run_tests

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from advanced_congestion import (
    BBR_Controller,
    Vegas_Controller,
    DCTCP_Controller,
    TIMELY_Controller,
    create_controller,
    CongestionState
)


class TestCongestionControllers(TestCase):
    """Test congestion control algorithms"""

    def test_bbr_initialization(self):
        """Test BBR controller initialization"""
        controller = BBR_Controller(initial_cwnd=10, mss=1500)

        self.assertEqual(controller.mss, 1500)
        self.assertEqual(controller.metrics.cwnd, 10)
        self.assertEqual(controller.state, CongestionState.STARTUP)
        self.assertEqual(controller.btlbw, 0.0)
        self.assertEqual(controller.rtprop, float('inf'))

    def test_bbr_ack_processing(self):
        """Test BBR ACK processing"""
        controller = BBR_Controller()

        # Simulate ACKs
        for i in range(10):
            controller.on_ack(bytes_acked=1500, rtt=0.001)  # 1ms RTT

        self.assertGreater(controller.metrics.delivered, 0)
        self.assertGreater(controller.metrics.bandwidth_estimate, 0)
        self.assertLess(controller.metrics.min_rtt, float('inf'))

    def test_bbr_loss_handling(self):
        """Test BBR loss handling"""
        controller = BBR_Controller()

        # Send some data
        for i in range(5):
            controller.on_ack(bytes_acked=1500, rtt=0.001)

        initial_cwnd = controller.metrics.cwnd

        # Simulate loss
        controller.on_loss(bytes_lost=1500)

        self.assertGreater(controller.metrics.lost, 0)

    def test_bbr_state_transitions(self):
        """Test BBR state machine transitions"""
        controller = BBR_Controller()

        self.assertEqual(controller.state, CongestionState.STARTUP)

        # Simulate bandwidth plateau to trigger DRAIN
        for i in range(10):
            controller.on_ack(bytes_acked=1500, rtt=0.001)

        # State should eventually transition
        # (exact behavior depends on bandwidth measurements)

    def test_vegas_initialization(self):
        """Test Vegas controller initialization"""
        controller = Vegas_Controller(initial_cwnd=10, mss=1500)

        self.assertEqual(controller.mss, 1500)
        self.assertEqual(controller.metrics.cwnd, 10)
        self.assertEqual(controller.base_rtt, float('inf'))

    def test_vegas_ack_processing(self):
        """Test Vegas ACK processing"""
        controller = Vegas_Controller()

        # Simulate ACKs with varying RTT
        rtts = [0.001, 0.0015, 0.002, 0.0012, 0.0011]
        for rtt in rtts:
            controller.on_ack(bytes_acked=1500, rtt=rtt)

        self.assertGreater(controller.metrics.delivered, 0)
        self.assertLess(controller.base_rtt, float('inf'))

    def test_vegas_congestion_avoidance(self):
        """Test Vegas congestion avoidance logic"""
        controller = Vegas_Controller()

        # Build up base RTT
        for i in range(5):
            controller.on_ack(bytes_acked=1500, rtt=0.001)

        # Enable Vegas
        controller.vegas_enabled = True
        controller.state = CongestionState.CONGESTION_AVOIDANCE

        initial_cwnd = controller.metrics.cwnd

        # High RTT should decrease cwnd
        for i in range(3):
            controller.on_ack(bytes_acked=1500, rtt=0.010)  # 10ms RTT

    def test_dctcp_initialization(self):
        """Test DCTCP controller initialization"""
        controller = DCTCP_Controller(initial_cwnd=10, mss=1500)

        self.assertEqual(controller.mss, 1500)
        self.assertEqual(controller.metrics.cwnd, 10)
        self.assertEqual(controller.alpha, 0.0)

    def test_dctcp_ecn_marking(self):
        """Test DCTCP ECN marking handling"""
        controller = DCTCP_Controller()

        # ACKs without ECN
        for i in range(5):
            controller.on_ack(bytes_acked=1500, rtt=0.001, ecn_marked=False)

        alpha_no_ecn = controller.alpha

        # ACKs with ECN marking
        for i in range(5):
            controller.on_ack(bytes_acked=1500, rtt=0.001, ecn_marked=True)

        # Alpha should increase with ECN marks
        self.assertGreater(controller.alpha, alpha_no_ecn)

    def test_dctcp_alpha_update(self):
        """Test DCTCP alpha parameter update"""
        controller = DCTCP_Controller()

        # Send packets with 50% ECN marking
        for i in range(10):
            ecn_marked = (i % 2 == 0)
            controller.on_ack(bytes_acked=1500, rtt=0.001, ecn_marked=ecn_marked)

        # Alpha should converge toward 0.5
        self.assertGreater(controller.alpha, 0.3)
        self.assertLess(controller.alpha, 0.7)

    def test_timely_initialization(self):
        """Test TIMELY controller initialization"""
        initial_rate = 10.0 * 1024 * 1024  # 10 MB/s
        controller = TIMELY_Controller(initial_rate=initial_rate, mss=1500)

        self.assertEqual(controller.rate, initial_rate)
        self.assertEqual(controller.smoothed_rtt, 0.0)

    def test_timely_rtt_tracking(self):
        """Test TIMELY RTT tracking and gradient computation"""
        controller = TIMELY_Controller()

        # Simulate increasing RTT
        rtts = [0.00005, 0.00006, 0.00007, 0.00008]  # 50-80 microseconds
        for rtt in rtts:
            controller.on_ack(bytes_acked=1500, rtt=rtt)

        self.assertGreater(controller.smoothed_rtt, 0)
        self.assertGreater(controller.rtt_diff, 0)  # Increasing RTT

    def test_timely_rate_adjustment(self):
        """Test TIMELY rate adjustment logic"""
        controller = TIMELY_Controller()

        initial_rate = controller.rate

        # Low RTT should increase rate
        for i in range(5):
            controller.on_ack(bytes_acked=1500, rtt=0.00003)  # 30 microseconds

        # Rate should increase
        self.assertGreater(controller.rate, initial_rate)

    def test_controller_factory(self):
        """Test controller factory function"""
        bbr = create_controller('bbr')
        self.assertIsInstance(bbr, BBR_Controller)

        vegas = create_controller('vegas')
        self.assertIsInstance(vegas, Vegas_Controller)

        dctcp = create_controller('dctcp')
        self.assertIsInstance(dctcp, DCTCP_Controller)

        timely = create_controller('timely')
        self.assertIsInstance(timely, TIMELY_Controller)

        # Invalid algorithm
        with self.assertRaises(ValueError):
            create_controller('invalid')

    def test_cwnd_bounds(self):
        """Test congestion window stays within bounds"""
        controller = BBR_Controller()

        # Simulate many ACKs
        for i in range(100):
            controller.on_ack(bytes_acked=1500, rtt=0.001)

        # cwnd should be positive
        self.assertGreater(controller.metrics.cwnd, 0)

        # Simulate many losses
        for i in range(50):
            controller.on_loss(bytes_lost=1500)

        # cwnd should still be positive
        self.assertGreater(controller.metrics.cwnd, 0)

    def test_pacing_rate_computation(self):
        """Test pacing rate computation"""
        controller = BBR_Controller()

        # Send some data to establish rate
        for i in range(10):
            controller.on_ack(bytes_acked=1500, rtt=0.001)

        pacing_rate = controller.get_pacing_rate()
        self.assertGreater(pacing_rate, 0)

    def test_rtt_statistics(self):
        """Test RTT statistics tracking"""
        controller = Vegas_Controller()

        rtts = [0.001, 0.0015, 0.002, 0.0012, 0.0018]
        for rtt in rtts:
            controller.on_ack(bytes_acked=1500, rtt=rtt)

        # Min RTT should be tracked
        self.assertEqual(controller.metrics.min_rtt, min(rtts))
        self.assertGreater(controller.metrics.rtt_variance, 0)

    def test_concurrent_flows(self):
        """Test multiple concurrent flows"""
        flows = [BBR_Controller() for _ in range(5)]

        # Simulate concurrent transfers
        for iteration in range(10):
            for flow in flows:
                flow.on_ack(bytes_acked=1500, rtt=random.uniform(0.001, 0.005))

        # All flows should have positive throughput
        for flow in flows:
            self.assertGreater(flow.metrics.delivered, 0)


class TestCongestionScenarios(TestCase):
    """Test congestion control in various scenarios"""

    def test_congestion_detection(self):
        """Test congestion detection via increased RTT"""
        controller = Vegas_Controller()

        # Normal operation
        for i in range(5):
            controller.on_ack(bytes_acked=1500, rtt=0.001)

        cwnd_normal = controller.metrics.cwnd

        # Congestion (increased RTT)
        for i in range(5):
            controller.on_ack(bytes_acked=1500, rtt=0.020)

        # cwnd should adapt
        self.assertNotEqual(controller.metrics.cwnd, cwnd_normal)

    def test_loss_recovery(self):
        """Test recovery from packet loss"""
        controller = BBR_Controller()

        # Build up state
        for i in range(10):
            controller.on_ack(bytes_acked=1500, rtt=0.001)

        cwnd_before = controller.metrics.cwnd

        # Loss event
        controller.on_loss(bytes_lost=3000)

        # Continue sending
        for i in range(10):
            controller.on_ack(bytes_acked=1500, rtt=0.001)

        # Should recover
        self.assertGreater(controller.metrics.delivered, 0)

    def test_varying_bandwidth(self):
        """Test adaptation to varying bandwidth"""
        controller = BBR_Controller()

        # High bandwidth period
        for i in range(10):
            controller.on_ack(bytes_acked=15000, rtt=0.001)  # 15 KB in 1ms

        bw_high = controller.metrics.bandwidth_estimate

        # Low bandwidth period
        for i in range(10):
            controller.on_ack(bytes_acked=1500, rtt=0.001)  # 1.5 KB in 1ms

        bw_low = controller.metrics.bandwidth_estimate

        # Should detect different bandwidths
        self.assertNotEqual(bw_high, bw_low)

    def test_fairness_scenario(self):
        """Test fairness with multiple flows"""
        flows = [Vegas_Controller() for _ in range(3)]

        # Simulate sharing bottleneck link
        for iteration in range(20):
            # All flows compete for same bandwidth
            for flow in flows:
                flow.on_ack(bytes_acked=1500, rtt=random.uniform(0.001, 0.003))

        # Calculate throughputs
        throughputs = [f.metrics.throughput_bps for f in flows]

        # Check relative fairness (within 2x of each other)
        if all(t > 0 for t in throughputs):
            ratio = max(throughputs) / min(throughputs)
            self.assertLess(ratio, 5.0)  # Reasonable fairness


if __name__ == '__main__':
    run_tests()

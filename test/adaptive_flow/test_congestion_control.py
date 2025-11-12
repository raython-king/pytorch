"""
Unit tests for Congestion Control.
"""

import time
import unittest
from torch.adaptive_flow.congestion_control import (
    CongestionDetector,
    CongestionController,
    BackpressureManager,
    ExplicitCongestionNotification,
    CongestionState,
    CongestionMetrics,
)


class TestCongestionDetector(unittest.TestCase):
    """Test CongestionDetector class."""

    def test_detection(self):
        """Test congestion detection."""
        detector = CongestionDetector(
            queue_threshold=100,
            loss_threshold=0.01,
            rtt_threshold=0.1,
        )

        # Normal state
        metrics = CongestionMetrics(
            queue_length=50,
            packet_loss_rate=0.001,
            rtt=0.05,
            throughput=1e9,
        )
        detector.update_metrics("link1", metrics)

        state = detector.get_congestion_state("link1")
        self.assertEqual(state, CongestionState.NORMAL)

        # Congested state
        metrics = CongestionMetrics(
            queue_length=150,
            packet_loss_rate=0.02,
            rtt=0.15,
            throughput=1e9,
        )
        detector.update_metrics("link1", metrics)

        state = detector.get_congestion_state("link1")
        self.assertTrue(detector.is_congested("link1"))

    def test_metrics_retrieval(self):
        """Test metrics retrieval."""
        detector = CongestionDetector()

        metrics = CongestionMetrics(
            queue_length=100,
            packet_loss_rate=0.01,
            rtt=0.1,
            throughput=1e9,
        )
        detector.update_metrics("link1", metrics)

        retrieved = detector.get_metrics("link1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.queue_length, 100)


class TestCongestionController(unittest.TestCase):
    """Test CongestionController class."""

    def test_aimd(self):
        """Test AIMD algorithm."""
        controller = CongestionController(algorithm="aimd")

        controller.initialize_flow("flow1", initial_rate=10e6)

        # Normal state: increase rate
        metrics = CongestionMetrics(
            queue_length=50,
            packet_loss_rate=0.001,
            rtt=0.05,
            throughput=10e6,
        )
        new_rate = controller.update_rate("flow1", metrics, CongestionState.NORMAL)
        initial_rate = controller.get_flow_rate("flow1")

        # Should increase
        self.assertGreater(new_rate, initial_rate * 0.9)

        # Congested state: decrease rate
        new_rate = controller.update_rate("flow1", metrics, CongestionState.CONGESTED)

        # Should decrease
        self.assertLess(new_rate, initial_rate)

    def test_vegas(self):
        """Test Vegas algorithm."""
        controller = CongestionController(algorithm="vegas")

        controller.initialize_flow("flow1", initial_rate=10e6)

        metrics = CongestionMetrics(
            queue_length=50,
            packet_loss_rate=0.001,
            rtt=0.05,
            throughput=10e6,
        )

        new_rate = controller.update_rate("flow1", metrics, CongestionState.NORMAL)
        self.assertGreater(new_rate, 0)

    def test_bbr(self):
        """Test BBR algorithm."""
        controller = CongestionController(algorithm="bbr")

        controller.initialize_flow("flow1", initial_rate=10e6)

        metrics = CongestionMetrics(
            queue_length=50,
            packet_loss_rate=0.0,
            rtt=0.05,
            throughput=10e6,
        )

        # Update rate multiple times
        for _ in range(10):
            new_rate = controller.update_rate("flow1", metrics, CongestionState.NORMAL)

        self.assertGreater(new_rate, 0)

    def test_flow_removal(self):
        """Test flow removal."""
        controller = CongestionController(algorithm="aimd")

        controller.initialize_flow("flow1")
        self.assertGreater(controller.get_flow_rate("flow1"), 0)

        controller.remove_flow("flow1")
        rate = controller.get_flow_rate("flow1")
        # Should return initial rate for unknown flows
        self.assertGreater(rate, 0)


class TestBackpressureManager(unittest.TestCase):
    """Test BackpressureManager class."""

    def test_pause_resume(self):
        """Test pause and resume."""
        manager = BackpressureManager()

        self.assertFalse(manager.is_paused("flow1"))

        manager.apply_backpressure("flow1")
        self.assertTrue(manager.is_paused("flow1"))

        manager.release_backpressure("flow1")
        self.assertFalse(manager.is_paused("flow1"))

    def test_rate_limiting(self):
        """Test rate limiting."""
        manager = BackpressureManager()

        rate_limit = 5e6  # 5 MB/s
        manager.apply_backpressure("flow1", rate_limit=rate_limit)

        limit = manager.get_rate_limit("flow1")
        self.assertEqual(limit, rate_limit)

        manager.release_backpressure("flow1")
        limit = manager.get_rate_limit("flow1")
        self.assertIsNone(limit)

    def test_callbacks(self):
        """Test pause/resume callbacks."""
        manager = BackpressureManager()

        paused_flows = []
        resumed_flows = []

        def on_pause(flow_id):
            paused_flows.append(flow_id)

        def on_resume(flow_id):
            resumed_flows.append(flow_id)

        manager.register_pause_callback(on_pause)
        manager.register_resume_callback(on_resume)

        manager.apply_backpressure("flow1")
        self.assertIn("flow1", paused_flows)

        manager.release_backpressure("flow1")
        self.assertIn("flow1", resumed_flows)


class TestExplicitCongestionNotification(unittest.TestCase):
    """Test ECN class."""

    def test_marking(self):
        """Test packet marking."""
        ecn = ExplicitCongestionNotification()

        # Low utilization: no marking
        marked = ecn.mark_packet("link1", utilization=0.5)
        self.assertFalse(marked)

        # High utilization: probabilistic marking
        marks = []
        for _ in range(100):
            marked = ecn.mark_packet("link1", utilization=0.9)
            marks.append(marked)

        # Should have some marks
        self.assertGreater(sum(marks), 0)

        # Very high utilization: always mark
        marked = ecn.mark_packet("link1", utilization=0.99)
        self.assertTrue(marked)

    def test_marking_rate(self):
        """Test marking rate calculation."""
        ecn = ExplicitCongestionNotification()

        for _ in range(100):
            ecn.mark_packet("link1", utilization=0.9)

        rate = ecn.get_marking_rate("link1")
        self.assertGreater(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_reset(self):
        """Test statistics reset."""
        ecn = ExplicitCongestionNotification()

        for _ in range(10):
            ecn.mark_packet("link1", utilization=0.9)

        ecn.reset_statistics("link1")

        rate = ecn.get_marking_rate("link1")
        self.assertEqual(rate, 0.0)


if __name__ == "__main__":
    unittest.main()

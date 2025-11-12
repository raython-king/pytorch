"""
Tests for performance monitoring.
"""

import time
import unittest
import tempfile
import shutil
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase

from torch.runtime_scheduler.monitor import (
    MetricsCollector,
    PerformanceMonitor,
    Visualizer,
    Alerter,
    TimedOperation,
    PerformanceStats,
    MetricType,
)


class TestMetricsCollector(TestCase):
    """Test MetricsCollector functionality."""

    def setUp(self):
        self.collector = MetricsCollector(max_samples=1000)

    def test_record_metric(self):
        """Test recording metrics."""
        self.collector.record("test_metric", 1.0)
        self.collector.record("test_metric", 2.0)
        self.collector.record("test_metric", 3.0)

        stats = self.collector.get_stats("test_metric")
        self.assertIsNotNone(stats)
        self.assertEqual(stats.count, 3)
        self.assertEqual(stats.sum, 6.0)
        self.assertEqual(stats.min, 1.0)
        self.assertEqual(stats.max, 3.0)

    def test_get_recent_values(self):
        """Test getting recent metric values."""
        for i in range(10):
            self.collector.record("test_metric", float(i))
            time.sleep(0.01)

        recent = self.collector.get_recent_values("test_metric", window_seconds=0.5)
        self.assertGreater(len(recent), 0)

    def test_max_samples(self):
        """Test max samples limit."""
        collector = MetricsCollector(max_samples=10)

        for i in range(20):
            collector.record("test_metric", float(i))

        values = collector.get_recent_values("test_metric")
        self.assertEqual(len(values), 10)

    def test_enable_disable(self):
        """Test enabling/disabling collection."""
        self.collector.disable()
        self.collector.record("test_metric", 1.0)

        stats = self.collector.get_stats("test_metric")
        self.assertIsNone(stats)

        self.collector.enable()
        self.collector.record("test_metric", 2.0)

        stats = self.collector.get_stats("test_metric")
        self.assertIsNotNone(stats)

    def test_reset(self):
        """Test resetting metrics."""
        self.collector.record("test_metric", 1.0)
        self.collector.reset()

        stats = self.collector.get_stats("test_metric")
        self.assertIsNone(stats)


class TestPerformanceMonitor(TestCase):
    """Test PerformanceMonitor functionality."""

    def setUp(self):
        self.monitor = PerformanceMonitor()

    def tearDown(self):
        self.monitor.stop()

    def test_record_operation(self):
        """Test recording operations."""
        self.monitor.record_operation(
            op_name="test_op",
            device="cuda:0",
            latency=0.001,
            memory_delta=1024
        )

        summary = self.monitor.get_summary()
        self.assertEqual(summary["operation_counts"]["test_op"], 1)

    def test_record_device_utilization(self):
        """Test recording device utilization."""
        self.monitor.record_device_utilization(
            device="cuda:0",
            compute_util=0.8,
            memory_util=0.6
        )

        summary = self.monitor.get_summary()
        self.assertIn("utilization.compute.cuda:0", summary["metrics"])

    def test_record_transfer(self):
        """Test recording transfers."""
        self.monitor.record_transfer(
            src_device="cuda:0",
            dst_device="cuda:1",
            bytes_transferred=1024 * 1024,
            duration=0.001
        )

        summary = self.monitor.get_summary()
        self.assertIn("bandwidth.cuda:0_to_cuda:1", summary["metrics"])

    def test_timed_operation(self):
        """Test TimedOperation context manager."""
        with TimedOperation(self.monitor, "test_op", "cuda:0"):
            time.sleep(0.01)

        summary = self.monitor.get_summary()
        self.assertEqual(summary["operation_counts"]["test_op"], 1)

    def test_reset(self):
        """Test resetting monitor."""
        self.monitor.record_operation("test_op", "cuda:0", 0.001)
        self.monitor.reset()

        summary = self.monitor.get_summary()
        self.assertEqual(summary["total_operations"], 0)


class TestVisualizer(TestCase):
    """Test Visualizer functionality."""

    def setUp(self):
        self.monitor = PerformanceMonitor()
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = Visualizer(self.monitor, log_dir=self.temp_dir)

    def tearDown(self):
        self.visualizer.close()
        self.monitor.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update(self):
        """Test visualization update."""
        self.monitor.record_operation("test_op", "cuda:0", 0.001)
        self.visualizer.update()

        # Check that log directory exists
        self.assertTrue(Path(self.temp_dir).exists())

    def test_print_dashboard(self):
        """Test printing dashboard."""
        self.monitor.record_operation("test_op", "cuda:0", 0.001)

        # Should not raise
        self.visualizer.print_dashboard()


class TestAlerter(TestCase):
    """Test Alerter functionality."""

    def setUp(self):
        self.monitor = PerformanceMonitor()
        self.alerts_received = []

        def alert_callback(metric_name, alert_info):
            self.alerts_received.append((metric_name, alert_info))

        self.alerter = Alerter(self.monitor, alert_callback=alert_callback)

    def tearDown(self):
        self.monitor.stop()

    def test_threshold_alert(self):
        """Test threshold-based alerting."""
        # Add threshold
        self.alerter.add_threshold("latency", 10.0, ">")

        # Record high latency
        self.monitor.record_operation("test_op", "cuda:0", 0.015)  # 15 ms

        # Check alerts
        alerts = self.alerter.check_alerts()

        # Should trigger alert
        self.assertGreater(len(self.alerts_received), 0)

    def test_no_alert_below_threshold(self):
        """Test no alert when below threshold."""
        self.alerter.add_threshold("latency", 10.0, ">")

        # Record low latency
        self.monitor.record_operation("test_op", "cuda:0", 0.001)  # 1 ms

        # Check alerts
        alerts = self.alerter.check_alerts()

        # Should not trigger alert
        self.assertEqual(len(self.alerts_received), 0)


class TestPerformanceStats(TestCase):
    """Test PerformanceStats functionality."""

    def test_update_stats(self):
        """Test updating statistics."""
        stats = PerformanceStats()

        for i in range(10):
            stats.update(float(i))

        self.assertEqual(stats.count, 10)
        self.assertEqual(stats.sum, 45.0)
        self.assertEqual(stats.min, 0.0)
        self.assertEqual(stats.max, 9.0)
        self.assertEqual(stats.mean, 4.5)

    def test_compute_percentiles(self):
        """Test percentile computation."""
        try:
            import numpy as np

            stats = PerformanceStats()
            values = list(range(100))

            for v in values:
                stats.update(float(v))

            stats.compute_percentiles(values)

            self.assertAlmostEqual(stats.p50, 49.5, delta=1.0)
            self.assertAlmostEqual(stats.p90, 89.5, delta=1.0)

        except ImportError:
            self.skipTest("NumPy not available")


if __name__ == "__main__":
    run_tests()

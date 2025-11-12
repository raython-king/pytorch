"""
Unit tests for Traffic Manager.
"""

import time
import unittest
import threading
from torch.adaptive_flow import (
    TrafficManager,
    DataFlow,
    FlowQueue,
    BandwidthMonitor,
    Priority,
)


class TestDataFlow(unittest.TestCase):
    """Test DataFlow class."""

    def test_creation(self):
        """Test flow creation."""
        flow = DataFlow(
            priority=Priority.HIGH,
            flow_id="test_flow",
            source="node1",
            dest="node2",
            size=1024 * 1024,
        )

        self.assertEqual(flow.flow_id, "test_flow")
        self.assertEqual(flow.size, 1024 * 1024)
        self.assertEqual(flow.remaining_bytes, 1024 * 1024)
        self.assertFalse(flow.is_complete)

    def test_progress_update(self):
        """Test progress tracking."""
        flow = DataFlow(
            priority=Priority.MEDIUM,
            flow_id="test_flow",
            size=1000,
        )

        flow.update_progress(500)
        self.assertEqual(flow.bytes_sent, 500)
        self.assertEqual(flow.remaining_bytes, 500)
        self.assertAlmostEqual(flow.progress, 0.5)
        self.assertFalse(flow.is_complete)

        flow.update_progress(500)
        self.assertEqual(flow.bytes_sent, 1000)
        self.assertTrue(flow.is_complete)

    def test_deadline_ordering(self):
        """Test flows are ordered by priority and deadline."""
        flow1 = DataFlow(priority=Priority.HIGH, flow_id="f1", deadline=100.0)
        flow2 = DataFlow(priority=Priority.HIGH, flow_id="f2", deadline=50.0)
        flow3 = DataFlow(priority=Priority.MEDIUM, flow_id="f3", deadline=10.0)

        # Lower priority value comes first
        self.assertLess(flow1, flow3)

        # Same priority, earlier deadline comes first
        self.assertLess(flow2, flow1)


class TestFlowQueue(unittest.TestCase):
    """Test FlowQueue class."""

    def test_enqueue_dequeue(self):
        """Test basic enqueue and dequeue."""
        queue = FlowQueue()

        flow1 = DataFlow(priority=Priority.MEDIUM, flow_id="f1", size=1000)
        flow2 = DataFlow(priority=Priority.HIGH, flow_id="f2", size=2000)

        queue.enqueue(flow1)
        queue.enqueue(flow2)

        self.assertEqual(len(queue), 2)

        # High priority should come out first
        dequeued = queue.dequeue()
        self.assertEqual(dequeued.flow_id, "f2")

        dequeued = queue.dequeue()
        self.assertEqual(dequeued.flow_id, "f1")

        self.assertTrue(queue.is_empty())

    def test_peek(self):
        """Test peek operation."""
        queue = FlowQueue()

        flow = DataFlow(priority=Priority.LOW, flow_id="f1", size=1000)
        queue.enqueue(flow)

        peeked = queue.peek()
        self.assertEqual(peeked.flow_id, "f1")
        self.assertEqual(len(queue), 1)  # Should not remove

    def test_remove(self):
        """Test remove operation."""
        queue = FlowQueue()

        flow1 = DataFlow(priority=Priority.MEDIUM, flow_id="f1", size=1000)
        flow2 = DataFlow(priority=Priority.HIGH, flow_id="f2", size=2000)

        queue.enqueue(flow1)
        queue.enqueue(flow2)

        self.assertTrue(queue.remove("f1"))
        self.assertEqual(len(queue), 1)

        dequeued = queue.dequeue()
        self.assertEqual(dequeued.flow_id, "f2")

    def test_thread_safety(self):
        """Test concurrent access."""
        queue = FlowQueue()
        errors = []

        def enqueue_flows():
            try:
                for i in range(100):
                    flow = DataFlow(priority=Priority.MEDIUM, flow_id=f"f{i}", size=1000)
                    queue.enqueue(flow)
            except Exception as e:
                errors.append(e)

        def dequeue_flows():
            try:
                for _ in range(50):
                    queue.dequeue()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=enqueue_flows),
            threading.Thread(target=dequeue_flows),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestBandwidthMonitor(unittest.TestCase):
    """Test BandwidthMonitor class."""

    def test_link_capacity(self):
        """Test setting link capacity."""
        monitor = BandwidthMonitor()

        capacity = 1e9  # 1 GB/s
        monitor.set_link_capacity("link1", capacity)

        available = monitor.get_available_bandwidth("link1")
        self.assertEqual(available, capacity)

    def test_bandwidth_tracking(self):
        """Test bandwidth usage tracking."""
        monitor = BandwidthMonitor()

        capacity = 1e9
        monitor.set_link_capacity("link1", capacity)

        # Record transfers
        for _ in range(10):
            monitor.record_transfer("link1", 1024 * 1024)
            time.sleep(0.01)

        utilization = monitor.get_utilization("link1")
        self.assertGreater(utilization, 0.0)
        self.assertLessEqual(utilization, 1.0)

    def test_flow_registration(self):
        """Test flow registration."""
        monitor = BandwidthMonitor()

        monitor.register_flow("link1", "flow1")
        monitor.register_flow("link1", "flow2")

        stats = monitor.get_statistics("link1")
        self.assertEqual(stats['active_flows'], 2)

        monitor.unregister_flow("link1", "flow1")
        stats = monitor.get_statistics("link1")
        self.assertEqual(stats['active_flows'], 1)


class TestTrafficManager(unittest.TestCase):
    """Test TrafficManager class."""

    def test_flow_submission(self):
        """Test flow submission."""
        manager = TrafficManager()

        flow = DataFlow(
            priority=Priority.MEDIUM,
            flow_id="f1",
            source="node1",
            dest="node2",
            size=1024 * 1024,
        )

        manager.submit_flow(flow)

        stats = manager.get_statistics()
        self.assertEqual(stats['flows_submitted'], 1)
        self.assertEqual(stats['pending_flows'], 1)

    def test_flow_scheduling(self):
        """Test flow scheduling."""
        manager = TrafficManager()

        # Set link capacity
        manager.set_link_capacity("node1", "node2", 1e9)

        # Submit flows
        for i in range(5):
            flow = DataFlow(
                priority=Priority.MEDIUM,
                flow_id=f"f{i}",
                source="node1",
                dest="node2",
                size=1024 * 1024,
            )
            manager.submit_flow(flow)

        # Schedule flows
        scheduled = manager.schedule_flows()

        self.assertGreater(len(scheduled), 0)
        stats = manager.get_statistics()
        self.assertGreater(stats['active_flows'], 0)

    def test_flow_progress(self):
        """Test flow progress tracking."""
        manager = TrafficManager()

        manager.set_link_capacity("node1", "node2", 1e9)

        flow = DataFlow(
            priority=Priority.MEDIUM,
            flow_id="f1",
            source="node1",
            dest="node2",
            size=1000,
        )

        manager.submit_flow(flow)
        manager.schedule_flows()

        # Update progress
        manager.update_flow_progress("f1", 500)

        status = manager.get_flow_status("f1")
        self.assertEqual(status['status'], 'active')
        self.assertEqual(status['bytes_sent'], 500)

        # Complete flow
        manager.update_flow_progress("f1", 500)

        status = manager.get_flow_status("f1")
        self.assertEqual(status['status'], 'completed')

        stats = manager.get_statistics()
        self.assertEqual(stats['flows_completed'], 1)

    def test_flow_cancellation(self):
        """Test flow cancellation."""
        manager = TrafficManager()

        flow = DataFlow(
            priority=Priority.MEDIUM,
            flow_id="f1",
            source="node1",
            dest="node2",
            size=1000,
        )

        manager.submit_flow(flow)
        result = manager.cancel_flow("f1")

        self.assertTrue(result)

        stats = manager.get_statistics()
        self.assertEqual(stats['flows_failed'], 1)

    def test_link_statistics(self):
        """Test link statistics."""
        manager = TrafficManager()

        capacity = 1e9
        manager.set_link_capacity("node1", "node2", capacity)

        stats = manager.get_link_statistics("node1", "node2")

        self.assertEqual(stats['capacity'], capacity)
        self.assertIn('utilization', stats)
        self.assertIn('active_flows', stats)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for Bandwidth Manager.
"""

import time
import unittest
from torch.adaptive_flow.bandwidth_manager import (
    BandwidthAllocator,
    BandwidthReservationManager,
    BandwidthReservation,
    ReservationPriority,
    AdaptiveLimiter,
    LinkMonitor,
    TokenBucket,
)


class TestTokenBucket(unittest.TestCase):
    """Test TokenBucket class."""

    def test_token_consumption(self):
        """Test token consumption."""
        rate = 1000  # 1000 bytes/sec
        capacity = 2000  # 2000 bytes
        bucket = TokenBucket(rate, capacity)

        # Should have full capacity
        self.assertEqual(bucket.get_tokens(), capacity)

        # Consume tokens
        result = bucket.consume(1000)
        self.assertTrue(result)
        self.assertEqual(bucket.get_tokens(), 1000)

        # Try to consume more than available
        result = bucket.consume(2000)
        self.assertFalse(result)

    def test_token_refill(self):
        """Test token refill."""
        rate = 1000  # 1000 bytes/sec
        capacity = 2000
        bucket = TokenBucket(rate, capacity)

        # Consume all tokens
        bucket.consume(2000)

        # Wait and check refill
        time.sleep(0.5)
        tokens = bucket.get_tokens()
        self.assertGreater(tokens, 0)

    def test_rate_update(self):
        """Test rate update."""
        bucket = TokenBucket(1000, 2000)

        bucket.set_rate(2000)
        time.sleep(0.5)

        tokens = bucket.get_tokens()
        # Should refill faster with new rate
        self.assertGreater(tokens, 0)

    def test_reset(self):
        """Test bucket reset."""
        bucket = TokenBucket(1000, 2000)

        bucket.consume(1000)
        bucket.reset()

        self.assertEqual(bucket.get_tokens(), 2000)


class TestBandwidthAllocator(unittest.TestCase):
    """Test BandwidthAllocator class."""

    def test_link_capacity(self):
        """Test setting link capacity."""
        allocator = BandwidthAllocator()

        capacity = 1e9  # 1 GB/s
        allocator.set_link_capacity("link1", capacity)

        stats = allocator.get_link_statistics("link1")
        self.assertEqual(stats['capacity'], capacity)

    def test_flow_addition(self):
        """Test adding flows."""
        allocator = BandwidthAllocator()

        capacity = 1e9
        allocator.set_link_capacity("link1", capacity)

        # Add flows
        demand = 0.5e9  # 500 MB/s
        allocation = allocator.add_flow("flow1", "link1", demand)

        self.assertGreater(allocation, 0)
        self.assertLessEqual(allocation, demand)

    def test_fair_allocation(self):
        """Test max-min fair allocation."""
        allocator = BandwidthAllocator()

        capacity = 1e9
        allocator.set_link_capacity("link1", capacity)

        # Add multiple flows with equal demands
        for i in range(4):
            allocator.add_flow(f"flow{i}", "link1", 0.5e9)

        # Each should get equal share (250 MB/s)
        alloc1 = allocator.get_allocation("flow0")
        alloc2 = allocator.get_allocation("flow1")

        self.assertAlmostEqual(alloc1, alloc2, delta=1e6)

    def test_flow_removal(self):
        """Test flow removal."""
        allocator = BandwidthAllocator()

        capacity = 1e9
        allocator.set_link_capacity("link1", capacity)

        allocator.add_flow("flow1", "link1", 0.5e9)
        allocator.add_flow("flow2", "link1", 0.5e9)

        allocator.remove_flow("flow1")

        stats = allocator.get_link_statistics("link1")
        self.assertEqual(stats['num_flows'], 1)

    def test_demand_update(self):
        """Test demand update."""
        allocator = BandwidthAllocator()

        capacity = 1e9
        allocator.set_link_capacity("link1", capacity)

        allocator.add_flow("flow1", "link1", 0.5e9)

        new_allocation = allocator.update_demand("flow1", 0.8e9)
        self.assertGreater(new_allocation, 0)


class TestBandwidthReservationManager(unittest.TestCase):
    """Test BandwidthReservationManager class."""

    def test_reservation_request(self):
        """Test reservation request."""
        manager = BandwidthReservationManager()

        capacity = 1e9
        manager.set_link_capacity("link1", capacity)

        reservation = BandwidthReservation(
            reservation_id="res1",
            flow_id="flow1",
            bandwidth=0.5e9,
            priority=ReservationPriority.MEDIUM,
            duration=60.0,
        )

        granted = manager.request_reservation(reservation, "link1")
        self.assertTrue(granted)

    def test_insufficient_bandwidth(self):
        """Test reservation denial."""
        manager = BandwidthReservationManager()

        capacity = 1e9
        manager.set_link_capacity("link1", capacity)

        # Reserve most bandwidth
        res1 = BandwidthReservation(
            reservation_id="res1",
            flow_id="flow1",
            bandwidth=0.9e9,
            priority=ReservationPriority.MEDIUM,
        )
        manager.request_reservation(res1, "link1")

        # Try to reserve more
        res2 = BandwidthReservation(
            reservation_id="res2",
            flow_id="flow2",
            bandwidth=0.5e9,
            priority=ReservationPriority.MEDIUM,
        )
        granted = manager.request_reservation(res2, "link1")

        self.assertFalse(granted)

    def test_critical_preemption(self):
        """Test preemption for critical flows."""
        manager = BandwidthReservationManager()

        capacity = 1e9
        manager.set_link_capacity("link1", capacity)

        # Reserve with low priority
        res1 = BandwidthReservation(
            reservation_id="res1",
            flow_id="flow1",
            bandwidth=0.6e9,
            priority=ReservationPriority.LOW,
        )
        manager.request_reservation(res1, "link1")

        # Critical reservation should preempt
        res2 = BandwidthReservation(
            reservation_id="res2",
            flow_id="flow2",
            bandwidth=0.6e9,
            priority=ReservationPriority.CRITICAL,
        )
        granted = manager.request_reservation(res2, "link1")

        # Should succeed by preempting low priority
        self.assertTrue(granted)

    def test_cancellation(self):
        """Test reservation cancellation."""
        manager = BandwidthReservationManager()

        manager.set_link_capacity("link1", 1e9)

        reservation = BandwidthReservation(
            reservation_id="res1",
            flow_id="flow1",
            bandwidth=0.5e9,
        )
        manager.request_reservation(reservation, "link1")

        result = manager.cancel_reservation("res1")
        self.assertTrue(result)

    def test_expiration_cleanup(self):
        """Test cleanup of expired reservations."""
        manager = BandwidthReservationManager()

        manager.set_link_capacity("link1", 1e9)

        # Create expired reservation
        reservation = BandwidthReservation(
            reservation_id="res1",
            flow_id="flow1",
            bandwidth=0.5e9,
            duration=0.001,  # Very short
        )
        manager.request_reservation(reservation, "link1")

        time.sleep(0.01)

        count = manager.cleanup_expired()
        self.assertEqual(count, 1)


class TestAdaptiveLimiter(unittest.TestCase):
    """Test AdaptiveLimiter class."""

    def test_allow(self):
        """Test transfer allow/deny."""
        rate = 1e6  # 1 MB/s
        limiter = AdaptiveLimiter(initial_rate=rate)

        # Should allow small transfer
        allowed = limiter.allow(1000)
        self.assertTrue(allowed)

        # Should deny very large transfer
        allowed = limiter.allow(1e9)
        self.assertFalse(allowed)

    def test_adaptation(self):
        """Test rate adaptation."""
        limiter = AdaptiveLimiter(initial_rate=1e6)

        initial_rate = limiter.get_rate()

        # Adapt to congestion
        limiter.adapt(congestion_detected=True)
        new_rate = limiter.get_rate()
        self.assertLess(new_rate, initial_rate)

        # Adapt to no congestion
        limiter.adapt(congestion_detected=False)
        new_rate = limiter.get_rate()
        self.assertGreater(new_rate, initial_rate * 0.5)

    def test_statistics(self):
        """Test statistics collection."""
        limiter = AdaptiveLimiter(initial_rate=1e6)

        for _ in range(10):
            limiter.allow(1000)

        stats = limiter.get_statistics()

        self.assertIn('rate', stats)
        self.assertIn('allowed', stats)
        self.assertIn('throttled', stats)


class TestLinkMonitor(unittest.TestCase):
    """Test LinkMonitor class."""

    def test_transfer_recording(self):
        """Test transfer recording."""
        monitor = LinkMonitor()

        capacity = 1e9
        monitor.set_capacity("link1", capacity)

        # Record transfers
        for _ in range(10):
            monitor.record_transfer("link1", 1024 * 1024)

        throughput = monitor.get_throughput("link1")
        self.assertGreater(throughput, 0)

    def test_utilization(self):
        """Test utilization calculation."""
        monitor = LinkMonitor()

        capacity = 1e9
        monitor.set_capacity("link1", capacity)

        # Record some transfers
        for _ in range(5):
            monitor.record_transfer("link1", 100 * 1024 * 1024)

        utilization = monitor.get_utilization("link1")
        self.assertGreaterEqual(utilization, 0.0)
        self.assertLessEqual(utilization, 1.0)

    def test_latency_tracking(self):
        """Test latency tracking."""
        monitor = LinkMonitor()

        # Record transfers with latency
        for i in range(10):
            latency = 0.001 * (i + 1)  # 1-10ms
            monitor.record_transfer("link1", 1024, latency=latency)

        avg_latency = monitor.get_average_latency("link1")
        self.assertGreater(avg_latency, 0)

    def test_loss_tracking(self):
        """Test packet loss tracking."""
        monitor = LinkMonitor()

        # Record transfers and losses
        for _ in range(100):
            monitor.record_transfer("link1", 1024)

        for _ in range(5):
            monitor.record_loss("link1", packets_lost=1)

        loss_rate = monitor.get_loss_rate("link1")
        self.assertGreater(loss_rate, 0)
        self.assertLess(loss_rate, 1.0)

    def test_statistics(self):
        """Test comprehensive statistics."""
        monitor = LinkMonitor()

        capacity = 1e9
        monitor.set_capacity("link1", capacity)

        # Record some activity
        for _ in range(10):
            monitor.record_transfer("link1", 1024 * 1024, latency=0.001)

        stats = monitor.get_statistics("link1")

        self.assertIn('capacity', stats)
        self.assertIn('throughput', stats)
        self.assertIn('utilization', stats)
        self.assertIn('average_latency', stats)
        self.assertIn('loss_rate', stats)
        self.assertIn('bytes_sent', stats)
        self.assertIn('packets_sent', stats)


if __name__ == "__main__":
    unittest.main()

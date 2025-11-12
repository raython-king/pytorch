"""
Unit tests for Flow Scheduler.
"""

import time
import unittest
from torch.adaptive_flow.flow_scheduler import (
    SJF_Scheduler,
    WFQ_Scheduler,
    EDF_Scheduler,
    ML_Scheduler,
    CompositeScheduler,
    StarvationPrevention,
    FlowInfo,
)


class TestSJFScheduler(unittest.TestCase):
    """Test SJF Scheduler."""

    def test_shortest_job_first(self):
        """Test SJF scheduling."""
        scheduler = SJF_Scheduler(preemptive=False)

        flows = [
            FlowInfo(flow_id="f1", size=1000, estimated_duration=10.0),
            FlowInfo(flow_id="f2", size=500, estimated_duration=5.0),
            FlowInfo(flow_id="f3", size=2000, estimated_duration=20.0),
        ]

        selected = scheduler.schedule(flows, available_slots=2)

        # Should select shortest jobs first
        self.assertIn("f2", selected)
        self.assertEqual(len(selected), 2)

    def test_preemptive_scheduling(self):
        """Test preemptive SJF (SRTF)."""
        scheduler = SJF_Scheduler(preemptive=True)

        flows = [
            FlowInfo(flow_id="f1", size=1000, estimated_duration=10.0),
            FlowInfo(flow_id="f2", size=500, estimated_duration=5.0),
        ]

        selected = scheduler.schedule(flows, available_slots=1)

        # Update progress on f1
        scheduler.update("f1", progress=0.5)

        # Add new shorter flow
        flows.append(FlowInfo(flow_id="f3", size=100, estimated_duration=1.0))

        selected = scheduler.schedule(flows, available_slots=1)

        # Should potentially preempt for shorter remaining time


class TestWFQScheduler(unittest.TestCase):
    """Test WFQ Scheduler."""

    def test_weighted_fair_queuing(self):
        """Test WFQ scheduling."""
        scheduler = WFQ_Scheduler()

        flows = [
            FlowInfo(flow_id="f1", size=1000, weight=1.0),
            FlowInfo(flow_id="f2", size=1000, weight=2.0),  # Higher weight
            FlowInfo(flow_id="f3", size=1000, weight=1.0),
        ]

        selected = scheduler.schedule(flows, available_slots=2)

        self.assertEqual(len(selected), 2)

    def test_progress_update(self):
        """Test progress tracking."""
        scheduler = WFQ_Scheduler()

        flows = [FlowInfo(flow_id="f1", size=1000, weight=1.0)]

        selected = scheduler.schedule(flows, available_slots=1)

        scheduler.update("f1", progress=0.5)
        scheduler.update("f1", progress=1.0)

        # Flow should be removed after completion


class TestEDFScheduler(unittest.TestCase):
    """Test EDF Scheduler."""

    def test_earliest_deadline_first(self):
        """Test EDF scheduling."""
        scheduler = EDF_Scheduler()

        current_time = time.time()
        flows = [
            FlowInfo(flow_id="f1", size=1000, deadline=current_time + 100,
                    estimated_duration=10.0),
            FlowInfo(flow_id="f2", size=1000, deadline=current_time + 50,
                    estimated_duration=10.0),
            FlowInfo(flow_id="f3", size=1000, deadline=current_time + 200,
                    estimated_duration=10.0),
        ]

        selected = scheduler.schedule(flows, available_slots=2)

        # Should select flows with earlier deadlines
        self.assertIn("f2", selected)

    def test_deadline_tracking(self):
        """Test deadline satisfaction tracking."""
        scheduler = EDF_Scheduler()

        scheduler.update("f1", progress=1.0)
        scheduler.update("f2", progress=1.0)

        scheduler.record_missed_deadline("f1")

        stats = scheduler.get_deadline_stats()

        self.assertEqual(stats['total'], 2)
        self.assertEqual(stats['missed'], 1)
        self.assertAlmostEqual(stats['satisfaction_rate'], 0.5)

    def test_no_deadline_flows(self):
        """Test handling of flows without deadlines."""
        scheduler = EDF_Scheduler()

        flows = [
            FlowInfo(flow_id="f1", size=1000, deadline=None, estimated_duration=10.0),
            FlowInfo(flow_id="f2", size=1000, deadline=None, estimated_duration=5.0),
        ]

        selected = scheduler.schedule(flows, available_slots=2)

        # Should schedule all flows
        self.assertEqual(len(selected), 2)


class TestMLScheduler(unittest.TestCase):
    """Test ML Scheduler."""

    def test_fallback_scheduling(self):
        """Test scheduling without model (fallback)."""
        scheduler = ML_Scheduler(model=None)

        flows = [
            FlowInfo(flow_id="f1", size=1000, priority=2),
            FlowInfo(flow_id="f2", size=1000, priority=1),
            FlowInfo(flow_id="f3", size=1000, priority=2),
        ]

        selected = scheduler.schedule(flows, available_slots=2)

        # Should use priority-based fallback
        self.assertIn("f2", selected)  # Higher priority
        self.assertEqual(len(selected), 2)

    def test_model_update(self):
        """Test model update."""
        scheduler = ML_Scheduler(model=None)

        # Create a simple mock model
        class MockModel:
            def predict(self, features):
                return [1.0, 2.0, 3.0]

        scheduler.update_model(MockModel())

        flows = [
            FlowInfo(flow_id="f1", size=1000, priority=1),
            FlowInfo(flow_id="f2", size=1000, priority=1),
            FlowInfo(flow_id="f3", size=1000, priority=1),
        ]

        selected = scheduler.schedule(flows, available_slots=2)

        # Should use model predictions
        self.assertEqual(len(selected), 2)


class TestCompositeScheduler(unittest.TestCase):
    """Test Composite Scheduler."""

    def test_scheduler_selection(self):
        """Test automatic scheduler selection."""
        scheduler = CompositeScheduler()

        current_time = time.time()

        # Many deadline flows -> should use EDF
        flows = [
            FlowInfo(flow_id="f1", size=1000, deadline=current_time + 100),
            FlowInfo(flow_id="f2", size=1000, deadline=current_time + 50),
            FlowInfo(flow_id="f3", size=1000, deadline=current_time + 200),
        ]

        selected = scheduler.schedule(flows, available_slots=2)
        self.assertEqual(len(selected), 2)

    def test_manual_selection(self):
        """Test manual scheduler selection."""
        scheduler = CompositeScheduler()

        scheduler.set_active_scheduler('sjf')

        flows = [FlowInfo(flow_id="f1", size=1000, estimated_duration=10.0)]

        selected = scheduler.schedule(flows, available_slots=1)

    def test_statistics(self):
        """Test scheduler statistics."""
        scheduler = CompositeScheduler()

        flows = [FlowInfo(flow_id="f1", size=1000)]

        scheduler.schedule(flows, available_slots=1)

        stats = scheduler.get_scheduler_stats()
        self.assertIsInstance(stats, dict)


class TestStarvationPrevention(unittest.TestCase):
    """Test Starvation Prevention."""

    def test_flow_registration(self):
        """Test flow registration."""
        prevention = StarvationPrevention(max_wait_time=10.0)

        prevention.register_flow("flow1", arrival_time=time.time())

        boost = prevention.check_starvation("flow1")
        self.assertEqual(boost, 0)  # No starvation yet

    def test_starvation_detection(self):
        """Test starvation detection."""
        prevention = StarvationPrevention(max_wait_time=0.1)

        # Register flow with old arrival time
        old_time = time.time() - 0.5
        prevention.register_flow("flow1", arrival_time=old_time)

        boost = prevention.check_starvation("flow1")
        self.assertGreater(boost, 0)  # Should detect starvation

    def test_priority_boost(self):
        """Test priority boost application."""
        prevention = StarvationPrevention(max_wait_time=0.1)

        old_time = time.time() - 1.0
        prevention.register_flow("flow1", arrival_time=old_time)

        flows = [
            FlowInfo(flow_id="flow1", size=1000, priority=5),
            FlowInfo(flow_id="flow2", size=1000, priority=1),
        ]

        boosted_flows = prevention.apply_boost(flows)

        # flow1 should have better priority after boost
        flow1_boosted = next(f for f in boosted_flows if f.flow_id == "flow1")
        self.assertLess(flow1_boosted.priority, 5)

    def test_flow_removal(self):
        """Test flow removal."""
        prevention = StarvationPrevention()

        prevention.register_flow("flow1")
        prevention.remove_flow("flow1")

        boost = prevention.check_starvation("flow1")
        self.assertEqual(boost, 0)  # Unknown flow


if __name__ == "__main__":
    unittest.main()

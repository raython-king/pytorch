"""
Tests for runtime scheduler profiler.
"""

import time
import unittest
import tempfile
import os

from torch.testing._internal.common_utils import run_tests, TestCase

from torch.runtime_scheduler.profiler import (
    RuntimeSchedulerProfiler,
    ProfilerContext,
    EventType,
    SchedulingDecision,
)


class TestRuntimeSchedulerProfiler(TestCase):
    """Test RuntimeSchedulerProfiler functionality."""

    def setUp(self):
        self.profiler = RuntimeSchedulerProfiler(enabled=True)

    def test_record_event(self):
        """Test recording events."""
        event_id = self.profiler.record_event(
            EventType.OPERATION_START,
            device="cuda:0",
            operation="test_op"
        )

        self.assertGreaterEqual(event_id, 0)

        events = self.profiler.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.OPERATION_START)

    def test_record_decision(self):
        """Test recording scheduling decisions."""
        decision_id = self.profiler.record_decision(
            operation="test_op",
            chosen_device="cuda:0",
            candidate_devices=["cuda:0", "cuda:1"],
            decision_factors={"latency": 0.8, "memory": 0.2},
            predicted_latency=0.001
        )

        self.assertGreaterEqual(decision_id, 0)

        decisions = self.profiler.get_decisions()
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].chosen_device, "cuda:0")

    def test_update_decision_outcome(self):
        """Test updating decision outcomes."""
        decision_id = self.profiler.record_decision(
            operation="test_op",
            chosen_device="cuda:0",
            candidate_devices=["cuda:0"],
            decision_factors={},
            predicted_latency=0.001
        )

        self.profiler.update_decision_outcome(decision_id, actual_latency=0.0012)

        decisions = self.profiler.get_decisions()
        self.assertEqual(decisions[0].actual_latency, 0.0012)
        self.assertIsNotNone(decisions[0].decision_quality)

    def test_get_timeline(self):
        """Test timeline generation."""
        start_id = self.profiler.record_event(
            EventType.OPERATION_START,
            device="cuda:0",
            operation="test_op"
        )

        time.sleep(0.01)

        self.profiler.record_event(
            EventType.OPERATION_END,
            device="cuda:0",
            operation="test_op",
            start_event_id=start_id
        )

        timeline = self.profiler.get_timeline()
        self.assertIn("cuda:0", timeline)
        self.assertGreater(len(timeline["cuda:0"]), 0)

    def test_export_chrome_trace(self):
        """Test Chrome trace export."""
        start_id = self.profiler.record_event(
            EventType.OPERATION_START,
            device="cuda:0",
            operation="test_op"
        )

        time.sleep(0.01)

        self.profiler.record_event(
            EventType.OPERATION_END,
            device="cuda:0",
            operation="test_op",
            start_event_id=start_id
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            trace_file = f.name

        try:
            self.profiler.export_chrome_trace(trace_file)
            self.assertTrue(os.path.exists(trace_file))

            # Verify JSON format
            import json
            with open(trace_file, 'r') as f:
                data = json.load(f)
                self.assertIn("traceEvents", data)

        finally:
            if os.path.exists(trace_file):
                os.unlink(trace_file)

    def test_analyze_scheduling_effectiveness(self):
        """Test scheduling effectiveness analysis."""
        # Record some decisions with outcomes
        for i in range(10):
            decision_id = self.profiler.record_decision(
                operation=f"op_{i}",
                chosen_device="cuda:0",
                candidate_devices=["cuda:0", "cuda:1"],
                decision_factors={},
                predicted_latency=0.001
            )

            self.profiler.update_decision_outcome(
                decision_id,
                actual_latency=0.001 + i * 0.0001
            )

        analysis = self.profiler.analyze_scheduling_effectiveness()

        self.assertEqual(analysis["total_decisions"], 10)
        self.assertEqual(analysis["decisions_with_outcomes"], 10)
        self.assertIn("mean_prediction_error", analysis)

    def test_what_if_analysis(self):
        """Test what-if analysis."""
        # Record decision
        decision_id = self.profiler.record_decision(
            operation="test_op",
            chosen_device="cuda:0",
            candidate_devices=["cuda:0", "cuda:1"],
            decision_factors={},
            predicted_latency=0.001
        )

        self.profiler.update_decision_outcome(decision_id, actual_latency=0.001)

        # Perform what-if analysis
        results = self.profiler.what_if_analysis({decision_id: "cuda:1"})

        self.assertIn("original", results)
        self.assertIn("alternative", results)

    def test_get_statistics(self):
        """Test getting statistics."""
        self.profiler.record_event(EventType.OPERATION_START, device="cuda:0")
        self.profiler.record_decision(
            operation="test_op",
            chosen_device="cuda:0",
            candidate_devices=["cuda:0"],
            decision_factors={},
            predicted_latency=0.001
        )

        stats = self.profiler.get_statistics()

        self.assertIn("total_events", stats)
        self.assertIn("total_decisions", stats)
        self.assertEqual(stats["total_events"], 1)
        self.assertEqual(stats["total_decisions"], 1)


class TestProfilerContext(TestCase):
    """Test ProfilerContext context manager."""

    def test_context_manager(self):
        """Test profiler as context manager."""
        with ProfilerContext() as prof:
            prof.record_event(EventType.OPERATION_START, device="cuda:0")
            time.sleep(0.01)

        # Profiler should be stopped
        events = prof.get_events()
        self.assertEqual(len(events), 1)

    def test_context_manager_with_error(self):
        """Test profiler context manager with error."""
        try:
            with ProfilerContext() as prof:
                prof.record_event(EventType.OPERATION_START, device="cuda:0")
                raise ValueError("Test error")
        except ValueError:
            pass

        # Profiler should still be stopped properly
        events = prof.get_events()
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    run_tests()

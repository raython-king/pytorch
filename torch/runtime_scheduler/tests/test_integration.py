"""
Integration tests for runtime scheduler.

Tests the full system working together.
"""

import time
import unittest

from torch.testing._internal.common_utils import run_tests, TestCase

from torch.runtime_scheduler.monitor import PerformanceMonitor, MetricsCollector
from torch.runtime_scheduler.profiler import RuntimeSchedulerProfiler
from torch.runtime_scheduler.config import (
    SchedulerConfig,
    RuntimeSchedulerManager,
    SchedulingMode,
)
from torch.runtime_scheduler.integration.pytorch_hooks import (
    RuntimeSchedulerHooks,
    HookMode,
    get_global_hooks,
)
from torch.runtime_scheduler.training.data_collector import RuntimeDataCollector
from torch.runtime_scheduler.training.replay_buffer import ReplayBuffer


class TestMonitorProfilerIntegration(TestCase):
    """Test integration between monitor and profiler."""

    def test_monitor_and_profiler_together(self):
        """Test using monitor and profiler together."""
        monitor = PerformanceMonitor()
        profiler = RuntimeSchedulerProfiler(enabled=True)

        # Start both
        monitor.start()
        profiler.start()

        # Simulate some operations
        for i in range(10):
            # Record in monitor
            monitor.record_operation(
                op_name=f"op_{i}",
                device="cuda:0",
                latency=0.001 * (i + 1)
            )

            # Record in profiler
            decision_id = profiler.record_decision(
                operation=f"op_{i}",
                chosen_device="cuda:0",
                candidate_devices=["cuda:0", "cuda:1"],
                decision_factors={"latency": 0.8},
                predicted_latency=0.001 * (i + 1)
            )

            profiler.update_decision_outcome(
                decision_id,
                actual_latency=0.001 * (i + 1)
            )

        # Stop both
        profiler.stop()
        monitor.stop()

        # Verify data collection
        monitor_summary = monitor.get_summary()
        profiler_stats = profiler.get_statistics()

        self.assertEqual(monitor_summary["total_operations"], 10)
        self.assertEqual(profiler_stats["total_decisions"], 10)


class TestFullSystemIntegration(TestCase):
    """Test full system integration."""

    def test_complete_workflow(self):
        """Test complete workflow from config to execution."""
        # Create configuration
        config = SchedulerConfig(
            mode=SchedulingMode.HEURISTIC,
            monitoring_enabled=True,
            profiling_enabled=False,
        )

        # Create manager
        manager = RuntimeSchedulerManager(config)

        # Start system
        manager.start()

        try:
            # Simulate some workload
            monitor = manager.get_monitor()
            if monitor:
                for i in range(5):
                    monitor.record_operation(
                        op_name="test_op",
                        device="cuda:0",
                        latency=0.001
                    )
                    time.sleep(0.01)

            # Get statistics
            stats = manager.get_stats()

            self.assertIn("monitor", stats)

        finally:
            manager.stop()


class TestHooksIntegration(TestCase):
    """Test hooks integration with the system."""

    def test_hooks_with_monitor(self):
        """Test hooks integration with monitoring."""
        hooks = RuntimeSchedulerHooks(mode=HookMode.SHADOW)
        monitor = PerformanceMonitor()

        # Register callback
        operations_intercepted = []

        def operation_callback(op_info):
            operations_intercepted.append(op_info)
            return op_info

        hooks.register_operation_dispatch_callback(operation_callback)

        # Enable hooks
        hooks.enable()

        try:
            # Simulate operation (simplified)
            # In real usage, this would intercept actual PyTorch operations

            # Verify hook state
            self.assertEqual(hooks.mode, HookMode.SHADOW)

        finally:
            hooks.disable()

        monitor.stop()


class TestDataCollectionIntegration(TestCase):
    """Test data collection integration."""

    def test_data_collector_with_monitor(self):
        """Test data collector working with monitor."""
        collector = RuntimeDataCollector(
            max_traces=100,
            sampling_rate=1.0,
            auto_save=False
        )

        monitor = PerformanceMonitor()

        # Collect traces
        for i in range(20):
            # Record in monitor
            monitor.record_operation(
                op_name=f"op_{i % 5}",
                device="cuda:0",
                latency=0.001 * (i + 1)
            )

            # Collect trace
            collector.collect(
                operation=f"op_{i % 5}",
                input_shapes=[(2, 3, 224, 224)],
                input_dtypes=["float32"],
                device="cuda:0",
                actual_latency=0.001 * (i + 1),
                memory_allocated=1024 * (i + 1)
            )

        # Verify collection
        stats = collector.get_statistics()
        self.assertEqual(stats["total_collected"], 20)

        # Create training examples
        examples = collector.create_training_examples()
        self.assertGreater(len(examples), 0)

        monitor.stop()


class TestReplayBufferIntegration(TestCase):
    """Test replay buffer integration."""

    def test_replay_buffer_with_collector(self):
        """Test replay buffer with data collector."""
        buffer = ReplayBuffer(capacity=100, prioritized=True)

        # Add some experiences
        for i in range(50):
            experience = {
                "state": {"op": f"op_{i}"},
                "action": "cuda:0",
                "reward": float(i),
                "next_state": {"op": f"op_{i+1}"},
            }
            buffer.add(experience, priority=float(i + 1))

        # Sample batch
        batch, indices = buffer.sample(10, return_indices=True)

        self.assertEqual(len(batch), 10)
        self.assertEqual(len(indices), 10)

        # Update priorities
        new_priorities = [float(idx + 100) for idx in indices]
        buffer.update_priorities(indices, new_priorities)

        # Verify buffer state
        stats = buffer.get_statistics()
        self.assertEqual(stats["current_size"], 50)


class TestEndToEndScenario(TestCase):
    """Test end-to-end scenarios."""

    def test_basic_scheduling_scenario(self):
        """Test basic scheduling scenario."""
        # Setup
        config = SchedulerConfig(
            mode=SchedulingMode.HEURISTIC,
            monitoring_enabled=True,
        )

        with RuntimeSchedulerManager(config) as manager:
            monitor = manager.get_monitor()

            # Simulate workload on multiple devices
            devices = ["cuda:0", "cuda:1"]
            operations = ["conv2d", "matmul", "relu"]

            for _ in range(10):
                for op in operations:
                    for device in devices:
                        if monitor:
                            monitor.record_operation(
                                op_name=op,
                                device=device,
                                latency=0.001
                            )

            # Get final statistics
            stats = manager.get_stats()

            if "monitor" in stats:
                self.assertGreater(stats["monitor"]["total_operations"], 0)

    def test_profiling_scenario(self):
        """Test profiling scenario."""
        config = SchedulerConfig(
            mode=SchedulingMode.HEURISTIC,
            profiling_enabled=True,
        )

        with RuntimeSchedulerManager(config) as manager:
            profiler = manager.get_profiler()

            if profiler:
                # Record some scheduling decisions
                for i in range(5):
                    decision_id = profiler.record_decision(
                        operation=f"op_{i}",
                        chosen_device="cuda:0",
                        candidate_devices=["cuda:0", "cuda:1"],
                        decision_factors={"latency": 0.7, "memory": 0.3},
                        predicted_latency=0.001
                    )

                    profiler.update_decision_outcome(
                        decision_id,
                        actual_latency=0.0011
                    )

                # Analyze effectiveness
                analysis = profiler.analyze_scheduling_effectiveness()
                self.assertIn("total_decisions", analysis)


if __name__ == "__main__":
    run_tests()

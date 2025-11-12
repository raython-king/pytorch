"""
Tests for runtime scheduler configuration.
"""

import tempfile
import os
import unittest

from torch.testing._internal.common_utils import run_tests, TestCase

from torch.runtime_scheduler.config import (
    SchedulerConfig,
    RuntimeSchedulerManager,
    SchedulingMode,
    OptimizationTarget,
    MonitoringLevel,
    enable_runtime_scheduler,
    disable_runtime_scheduler,
    SchedulerContext,
    load_config_from_env,
)


class TestSchedulerConfig(TestCase):
    """Test SchedulerConfig functionality."""

    def test_default_config(self):
        """Test default configuration."""
        config = SchedulerConfig()

        self.assertEqual(config.mode, SchedulingMode.DISABLED)
        self.assertEqual(config.target, OptimizationTarget.LATENCY)
        self.assertTrue(config.monitoring_enabled)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = SchedulerConfig(
            mode=SchedulingMode.ML,
            target=OptimizationTarget.THROUGHPUT
        )

        data = config.to_dict()

        self.assertEqual(data["mode"], "ml")
        self.assertEqual(data["target"], "throughput")

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "mode": "ml",
            "target": "latency",
            "devices": ["cuda:0", "cuda:1"],
            "monitoring_enabled": True,
        }

        config = SchedulerConfig.from_dict(data)

        self.assertEqual(config.mode, SchedulingMode.ML)
        self.assertEqual(config.target, OptimizationTarget.LATENCY)
        self.assertEqual(config.devices, ["cuda:0", "cuda:1"])

    def test_config_save_load(self):
        """Test saving and loading configuration."""
        config = SchedulerConfig(
            mode=SchedulingMode.HYBRID,
            target=OptimizationTarget.MEMORY
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name

        try:
            config.save(config_file)
            loaded_config = SchedulerConfig.load(config_file)

            self.assertEqual(loaded_config.mode, config.mode)
            self.assertEqual(loaded_config.target, config.target)

        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = SchedulerConfig()
        errors = config.validate()
        self.assertEqual(len(errors), 0)

        # Invalid weights
        config = SchedulerConfig(
            load_balance_weight=0.5,
            latency_weight=0.5,
            memory_weight=0.5  # Sum > 1.0
        )
        errors = config.validate()
        self.assertGreater(len(errors), 0)

        # Empty devices
        config = SchedulerConfig(devices=[])
        errors = config.validate()
        self.assertGreater(len(errors), 0)


class TestRuntimeSchedulerManager(TestCase):
    """Test RuntimeSchedulerManager functionality."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        config = SchedulerConfig(mode=SchedulingMode.HEURISTIC)
        manager = RuntimeSchedulerManager(config)

        self.assertEqual(manager.config.mode, SchedulingMode.HEURISTIC)
        self.assertFalse(manager._started)

    def test_manager_start_stop(self):
        """Test starting and stopping manager."""
        config = SchedulerConfig(mode=SchedulingMode.DISABLED)
        manager = RuntimeSchedulerManager(config)

        manager.start()
        self.assertTrue(manager._started)

        manager.stop()
        self.assertFalse(manager._started)

    def test_manager_context_manager(self):
        """Test manager as context manager."""
        config = SchedulerConfig(mode=SchedulingMode.DISABLED)
        manager = RuntimeSchedulerManager(config)

        with manager:
            self.assertTrue(manager._started)

        self.assertFalse(manager._started)

    def test_manager_get_stats(self):
        """Test getting statistics."""
        config = SchedulerConfig(mode=SchedulingMode.DISABLED)
        manager = RuntimeSchedulerManager(config)
        manager.start()

        stats = manager.get_stats()

        self.assertIn("start_time", stats)
        self.assertIn("operations_scheduled", stats)

        manager.stop()

    def test_manager_update_config(self):
        """Test updating configuration."""
        manager = RuntimeSchedulerManager()

        manager.update_config(
            monitoring_enabled=False,
            batch_size=64
        )

        self.assertFalse(manager.config.monitoring_enabled)


class TestGlobalFunctions(TestCase):
    """Test global configuration functions."""

    def test_enable_disable_scheduler(self):
        """Test enabling and disabling scheduler."""
        manager = enable_runtime_scheduler(
            mode='heuristic',
            target='latency',
            monitoring=True
        )

        self.assertIsNotNone(manager)
        self.assertTrue(manager._started)

        disable_runtime_scheduler()

    def test_scheduler_context(self):
        """Test SchedulerContext."""
        with SchedulerContext(mode='ml', target='throughput') as manager:
            self.assertTrue(manager._started)
            self.assertEqual(manager.config.mode, SchedulingMode.ML)

    def test_load_config_from_env(self):
        """Test loading configuration from environment."""
        # Set environment variables
        os.environ['TORCH_SCHEDULER_MODE'] = 'ml'
        os.environ['TORCH_SCHEDULER_TARGET'] = 'throughput'
        os.environ['TORCH_SCHEDULER_DEVICES'] = 'cuda:0,cuda:1'

        try:
            config = load_config_from_env()

            self.assertEqual(config.mode, SchedulingMode.ML)
            self.assertEqual(config.target, OptimizationTarget.THROUGHPUT)
            self.assertEqual(config.devices, ['cuda:0', 'cuda:1'])

        finally:
            # Clean up
            for key in ['TORCH_SCHEDULER_MODE', 'TORCH_SCHEDULER_TARGET', 'TORCH_SCHEDULER_DEVICES']:
                if key in os.environ:
                    del os.environ[key]


if __name__ == "__main__":
    run_tests()

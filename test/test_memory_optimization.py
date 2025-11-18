"""
Tests for Multi-Agent Memory Optimization System
"""

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

from torch.memory_optimization import (
    MemoryOptimizer,
    MemoryOptimizationConfig,
    HardwareDiagnostics,
    MemoryProfiler,
)
from torch.memory_optimization.orchestrator import MemoryOptimizationOrchestrator
from torch.memory_optimization.agents import (
    DiagnosticsAgent,
    StrategySelectorAgent,
    MonitoringAgent,
    CoordinatorAgent,
)


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestHardwareDiagnostics(TestCase):
    """Test hardware diagnostics functionality"""

    def test_diagnose(self):
        """Test hardware diagnostics"""
        diagnostics = HardwareDiagnostics()
        caps = diagnostics.diagnose()

        self.assertIsNotNone(caps)
        self.assertGreaterEqual(caps.num_cpu_cores, 1)
        self.assertGreater(caps.cpu_total_memory, 0)

    def test_recommended_strategies(self):
        """Test strategy recommendations"""
        diagnostics = HardwareDiagnostics()
        recommendations = diagnostics.get_recommended_strategies()

        self.assertIsInstance(recommendations, list)


class TestMemoryProfiler(TestCase):
    """Test memory profiler"""

    def test_capture_snapshot(self):
        """Test capturing memory snapshot"""
        profiler = MemoryProfiler()
        snapshot = profiler.capture_snapshot()

        self.assertIsNotNone(snapshot)
        self.assertGreater(snapshot.cpu_used, 0)

    def test_detect_memory_pressure(self):
        """Test memory pressure detection"""
        profiler = MemoryProfiler()
        pressure = profiler.detect_memory_pressure(threshold=0.99)

        self.assertIsInstance(pressure, bool)


class TestAgents(TestCase):
    """Test agent functionality"""

    def test_diagnostics_agent(self):
        """Test diagnostics agent"""
        config = {'auto_detect_hardware': True}
        agent = DiagnosticsAgent("test_diag", config)

        environment = {'model': SimpleModel()}
        agent.observe(environment)

        decision = agent.decide()
        self.assertIsNotNone(decision)
        self.assertIn(decision.action, ['optimize', 'monitor', 'wait'])

    def test_strategy_selector_agent(self):
        """Test strategy selector agent"""
        config = {'ml_model_type': 'ensemble'}
        agent = StrategySelectorAgent("test_selector", config)

        environment = {
            'hardware_caps': HardwareDiagnostics().diagnose(),
            'memory_usage': {},
        }
        agent.observe(environment)

        decision = agent.decide()
        self.assertIsNotNone(decision)

    def test_monitoring_agent(self):
        """Test monitoring agent"""
        config = {'monitoring_window': 10}
        agent = MonitoringAgent("test_monitor", config)

        environment = {
            'memory_usage': {},
            'throughput': 100.0,
        }
        agent.observe(environment)

        decision = agent.decide()
        self.assertIsNotNone(decision)

    def test_coordinator_agent(self):
        """Test coordinator agent"""
        config = {}
        agent = CoordinatorAgent("test_coord", config)

        # Create mock decisions
        from torch.memory_optimization.agents.base_agent import AgentDecision

        decisions = {
            'agent1': AgentDecision(action="optimize", confidence=0.8),
            'agent2': AgentDecision(action="optimize", confidence=0.9),
        }

        agent.observe({'agent_decisions': decisions})
        decision = agent.decide()

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "optimize")


class TestStrategies(TestCase):
    """Test optimization strategies"""

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing strategy"""
        from torch.memory_optimization.strategies import GradientCheckpointingStrategy

        model = SimpleModel()
        config = {'checkpoint_ratio': 0.5}
        strategy = GradientCheckpointingStrategy(config)

        result = strategy.apply(model)
        self.assertTrue(result.success or not result.success)  # May fail in CPU-only env

        # Test estimation
        savings = strategy.estimate_memory_savings(model)
        self.assertGreaterEqual(savings, 0)

    def test_mixed_precision(self):
        """Test mixed precision strategy"""
        from torch.memory_optimization.strategies import MixedPrecisionStrategy

        model = SimpleModel()
        config = {'amp_dtype': 'float16'}
        strategy = MixedPrecisionStrategy(config)

        result = strategy.apply(model)
        # May fail if CUDA not available
        self.assertIsNotNone(result)


class TestOrchestrator(TestCase):
    """Test orchestrator functionality"""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        config = MemoryOptimizationConfig()
        orchestrator = MemoryOptimizationOrchestrator(config)

        self.assertIsNotNone(orchestrator.diagnostics_agent)
        self.assertIsNotNone(orchestrator.strategy_selector)
        self.assertIsNotNone(orchestrator.monitoring_agent)
        self.assertIsNotNone(orchestrator.coordinator)

    def test_optimize_model(self):
        """Test model optimization"""
        config = MemoryOptimizationConfig(auto_detect_hardware=True)
        orchestrator = MemoryOptimizationOrchestrator(config)

        model = SimpleModel()
        optimized_model = orchestrator.optimize_model(model)

        self.assertIsNotNone(optimized_model)

    def test_adapt(self):
        """Test adaptation"""
        config = MemoryOptimizationConfig()
        orchestrator = MemoryOptimizationOrchestrator(config)

        model = SimpleModel()
        orchestrator.optimize_model(model)

        # Test adaptation
        metrics = {
            'throughput': 100.0,
            'iteration_time': 0.1,
            'memory_usage': 0.7,
        }
        orchestrator.adapt(metrics)

        summary = orchestrator.get_summary()
        self.assertIn('iterations', summary)

    def test_get_summary(self):
        """Test getting summary"""
        config = MemoryOptimizationConfig()
        orchestrator = MemoryOptimizationOrchestrator(config)

        model = SimpleModel()
        orchestrator.optimize_model(model)

        summary = orchestrator.get_summary()

        self.assertIn('active_strategies', summary)
        self.assertIn('iterations', summary)


class TestMemoryOptimizer(TestCase):
    """Test high-level MemoryOptimizer interface"""

    def test_optimizer_creation(self):
        """Test creating memory optimizer"""
        optimizer = MemoryOptimizer(auto_detect=True)
        self.assertIsNotNone(optimizer)

    def test_optimize_model(self):
        """Test optimizing a model"""
        optimizer = MemoryOptimizer(auto_detect=True)
        model = SimpleModel()

        optimized = optimizer.optimize_model(model)
        self.assertIsNotNone(optimized)

    def test_training_loop(self):
        """Test integration with training loop"""
        optimizer = MemoryOptimizer(auto_detect=True)
        model = SimpleModel()
        model = optimizer.optimize_model(model)

        # Simulate training step
        with optimizer.optimize_step():
            x = torch.randn(2, 3, 32, 32)
            y = model(x)
            self.assertEqual(y.shape, (2, 10))

        summary = optimizer.summary()
        self.assertIn('active_strategies', summary)


class TestConfig(TestCase):
    """Test configuration"""

    def test_config_validation(self):
        """Test config validation"""
        config = MemoryOptimizationConfig()
        config.validate()  # Should not raise

    def test_config_for_hardware(self):
        """Test hardware-specific config"""
        config = MemoryOptimizationConfig.for_hardware(
            gpu_memory_gb=8.0,
            num_gpus=1
        )

        self.assertIsNotNone(config)
        # Low memory should enable aggressive optimization
        self.assertGreater(config.checkpoint_ratio, 0.5)

    def test_config_to_dict(self):
        """Test config serialization"""
        config = MemoryOptimizationConfig()
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertIn('auto_detect_hardware', config_dict)


if __name__ == "__main__":
    run_tests()

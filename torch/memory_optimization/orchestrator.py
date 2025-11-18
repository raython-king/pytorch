"""
Multi-Agent Memory Optimization Orchestrator

Central orchestrator that coordinates all agents and strategies to
optimize memory usage adaptively across different hardware configurations.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any, List, Optional, Set
from contextlib import contextmanager

from .config import MemoryOptimizationConfig, OptimizationStrategy as StrategyEnum
from .diagnostics import HardwareDiagnostics, MemoryProfiler, HardwareCapabilities
from .agents import (
    DiagnosticsAgent,
    StrategySelectorAgent,
    MonitoringAgent,
    CoordinatorAgent,
    AgentDecision,
)
from .strategies import (
    GradientCheckpointingStrategy,
    ActivationCheckpointingStrategy,
    CPUOffloadingStrategy,
    MixedPrecisionStrategy,
    GradientCompressionStrategy,
    DynamicBatchSizeStrategy,
    MemoryEfficientOptimizerStrategy,
)


class MemoryOptimizationOrchestrator:
    """
    Main orchestrator for multi-agent memory optimization.

    Coordinates multiple specialized agents to adaptively select and
    apply memory optimization strategies based on hardware capabilities
    and runtime conditions.
    """

    def __init__(self, config: Optional[MemoryOptimizationConfig] = None):
        self.config = config or MemoryOptimizationConfig()
        self.config.validate()

        # Initialize agents
        self.diagnostics_agent = DiagnosticsAgent(
            "diagnostics_001",
            self.config.to_dict()
        )
        self.strategy_selector = StrategySelectorAgent(
            "selector_001",
            self.config.to_dict()
        )
        self.monitoring_agent = MonitoringAgent(
            "monitor_001",
            self.config.to_dict()
        )
        self.coordinator = CoordinatorAgent(
            "coordinator_001",
            self.config.to_dict()
        )

        # Initialize strategy instances
        self.strategies: Dict[str, Any] = {
            'gradient_checkpointing': GradientCheckpointingStrategy(self.config.to_dict()),
            'activation_checkpointing': ActivationCheckpointingStrategy(self.config.to_dict()),
            'cpu_offloading': CPUOffloadingStrategy(self.config.to_dict()),
            'mixed_precision': MixedPrecisionStrategy(self.config.to_dict()),
            'gradient_compression': GradientCompressionStrategy(self.config.to_dict()),
            'dynamic_batch_size': DynamicBatchSizeStrategy(self.config.to_dict()),
            'memory_efficient_optimizer': MemoryEfficientOptimizerStrategy(self.config.to_dict()),
        }

        # State tracking
        self.active_strategies: Set[str] = set()
        self.hardware_caps: Optional[HardwareCapabilities] = None
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.iteration_count = 0
        self.last_adaptation = 0

        # Performance metrics
        self.metrics_history = []
        self.total_memory_saved = 0.0
        self.throughput_improvement = 0.0

        # Integration with other systems
        self.runtime_scheduler = None
        self.gpu_comm_optimizer = None
        self.adaptive_flow = None

        if self.config.integrate_with_runtime_scheduler:
            self._setup_runtime_scheduler_integration()
        if self.config.integrate_with_gpu_cluster_comm:
            self._setup_gpu_comm_integration()
        if self.config.integrate_with_adaptive_flow:
            self._setup_adaptive_flow_integration()

    def _setup_runtime_scheduler_integration(self):
        """Setup integration with runtime scheduler"""
        try:
            from torch.runtime_scheduler import RuntimeScheduler
            self.runtime_scheduler = RuntimeScheduler()
        except ImportError:
            pass

    def _setup_gpu_comm_integration(self):
        """Setup integration with GPU cluster communication optimizer"""
        try:
            from torch.gpu_cluster_comm import GPUCommOptimizer
            self.gpu_comm_optimizer = GPUCommOptimizer()
        except ImportError:
            pass

    def _setup_adaptive_flow_integration(self):
        """Setup integration with adaptive flow control"""
        try:
            from torch.adaptive_flow import AdaptiveFlowController
            self.adaptive_flow = AdaptiveFlowController()
        except ImportError:
            pass

    def optimize_model(
        self,
        model: nn.Module,
        optimizer=None,
        **kwargs
    ) -> nn.Module:
        """
        Optimize model with automatic strategy selection.

        Args:
            model: PyTorch model to optimize
            optimizer: Optional optimizer (for optimizer-related strategies)
            **kwargs: Additional arguments

        Returns:
            Optimized model
        """
        self.model = model
        self.optimizer = optimizer

        # Diagnose hardware
        if self.config.auto_detect_hardware:
            self.hardware_caps = self.diagnostics_agent.get_hardware_capabilities()

        # Get initial observations
        environment = self._gather_environment()

        # Let agents observe
        self.diagnostics_agent.observe(environment)
        self.strategy_selector.observe(environment)
        self.monitoring_agent.observe(environment)

        # Get agent decisions
        agent_decisions = {
            'diagnostics': self.diagnostics_agent.decide(),
            'strategy_selector': self.strategy_selector.decide(),
            'monitoring': self.monitoring_agent.decide(),
        }

        # Coordinator makes final decision
        self.coordinator.observe({'agent_decisions': agent_decisions})
        final_decision = self.coordinator.decide()

        # Apply decision
        if final_decision.action == "apply_strategy" and final_decision.strategy:
            self._apply_strategy(final_decision.strategy, model, optimizer)
        elif final_decision.action == "optimize":
            # Apply multiple recommended strategies
            recommended = final_decision.parameters.get('recommended_strategies', [])
            for strategy_name in recommended:
                self._apply_strategy(strategy_name, model, optimizer)

        return model

    def _gather_environment(self) -> Dict[str, Any]:
        """Gather current environment state"""
        # Memory snapshot
        profiler = MemoryProfiler()
        snapshot = profiler.capture_snapshot()

        # Calculate memory usage dict
        memory_usage = {
            'gpu_utilization': snapshot.gpu_utilization,
            'gpu_allocated': snapshot.gpu_allocated,
            'cpu_percent': snapshot.cpu_percent,
        }

        return {
            'model': self.model,
            'optimizer': self.optimizer,
            'hardware_caps': self.hardware_caps,
            'memory_usage': memory_usage,
            'current_strategies': list(self.active_strategies),
            'iteration': self.iteration_count,
        }

    def _apply_strategy(
        self,
        strategy_name: str,
        model: nn.Module,
        optimizer=None
    ) -> bool:
        """Apply a specific optimization strategy"""
        if strategy_name in self.active_strategies:
            return True  # Already applied

        if strategy_name not in self.strategies:
            return False  # Unknown strategy

        strategy = self.strategies[strategy_name]

        # Apply strategy
        result = strategy.apply(model, optimizer=optimizer)

        if result.success:
            self.active_strategies.add(strategy_name)
            self.total_memory_saved += result.memory_saved
            return True

        return False

    def adapt(self, metrics: Dict[str, float]) -> None:
        """
        Adapt optimization strategies based on runtime metrics.

        Args:
            metrics: Dictionary containing performance metrics
        """
        self.iteration_count += 1

        # Check if it's time to adapt
        if self.iteration_count - self.last_adaptation < self.config.adaptation_interval:
            return

        # Update metrics history
        self.metrics_history.append({
            'iteration': self.iteration_count,
            'timestamp': time.time(),
            **metrics
        })

        # Gather current environment
        environment = self._gather_environment()
        environment.update({
            'performance_metrics': metrics,
            'throughput': metrics.get('throughput', 0.0),
            'iteration_time': metrics.get('iteration_time', 0.0),
        })

        # Let agents observe
        self.diagnostics_agent.observe(environment)
        self.strategy_selector.observe(environment)
        self.monitoring_agent.observe(environment)

        # Get decisions
        agent_decisions = {
            'diagnostics': self.diagnostics_agent.decide(),
            'strategy_selector': self.strategy_selector.decide(),
            'monitoring': self.monitoring_agent.decide(),
        }

        # Coordinator decides
        self.coordinator.observe({'agent_decisions': agent_decisions})
        decision = self.coordinator.decide()

        # Calculate reward for learning
        reward = self._calculate_reward(metrics)

        # Agents learn
        next_state = environment
        self.diagnostics_agent.learn(reward, next_state)
        self.strategy_selector.learn(reward, next_state)
        self.monitoring_agent.learn(reward, next_state)
        self.coordinator.learn(reward, next_state)

        # Record decisions
        self.diagnostics_agent.record_decision(agent_decisions['diagnostics'])
        self.strategy_selector.record_decision(agent_decisions['strategy_selector'])
        self.monitoring_agent.record_decision(agent_decisions['monitoring'])
        self.coordinator.record_decision(decision)

        # Apply new strategies if recommended
        if decision.action == "apply_strategy" and decision.strategy:
            if self.model:
                self._apply_strategy(decision.strategy, self.model, self.optimizer)

        self.last_adaptation = self.iteration_count

    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate reward signal for agents"""
        reward = 0.0

        # Reward for throughput improvement
        if 'throughput' in metrics and len(self.metrics_history) > 0:
            baseline = sum(m.get('throughput', 0) for m in self.metrics_history[-10:]) / 10
            if baseline > 0:
                improvement = (metrics['throughput'] - baseline) / baseline
                reward += improvement * 0.5

        # Reward for memory efficiency
        if 'memory_usage' in metrics:
            memory_util = metrics.get('memory_usage', 1.0)
            # Reward efficient memory use (not too high, not too low)
            if 0.6 < memory_util < 0.85:
                reward += 0.3
            elif memory_util > 0.95:
                reward -= 0.5  # Penalize high memory pressure

        # Reward for stability
        if len(self.metrics_history) >= 10:
            recent_throughputs = [m.get('throughput', 0) for m in self.metrics_history[-10:]]
            if recent_throughputs:
                import numpy as np
                stability = 1.0 - (np.std(recent_throughputs) / (np.mean(recent_throughputs) + 1e-6))
                reward += stability * 0.2

        return reward

    @contextmanager
    def optimize_step(self):
        """Context manager for optimized training step"""
        start_time = time.time()

        # Use mixed precision if enabled
        precision_strategy = self.strategies.get('mixed_precision')
        if precision_strategy and precision_strategy.enabled:
            with precision_strategy.forward_context():
                yield self
        else:
            yield self

        # Record iteration time
        iteration_time = time.time() - start_time
        self.adapt({'iteration_time': iteration_time})

    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        return {
            'active_strategies': list(self.active_strategies),
            'total_memory_saved_gb': self.total_memory_saved,
            'iterations': self.iteration_count,
            'adaptations': self.last_adaptation,
            'agent_weights': self.coordinator.get_agent_weights(),
            'strategy_rankings': self.strategy_selector.get_strategy_rankings(),
            'monitoring_stats': self.monitoring_agent.get_statistics(),
        }

    def reset(self):
        """Reset optimizer state"""
        # Revert all strategies
        for strategy_name in list(self.active_strategies):
            strategy = self.strategies.get(strategy_name)
            if strategy and self.model:
                strategy.revert(self.model)

        self.active_strategies.clear()
        self.iteration_count = 0
        self.last_adaptation = 0
        self.metrics_history.clear()


# Convenience class for easier usage
class MemoryOptimizer:
    """Simplified interface for memory optimization"""

    def __init__(self, auto_detect: bool = True, config: Optional[MemoryOptimizationConfig] = None):
        if config is None:
            config = MemoryOptimizationConfig()
            config.auto_detect_hardware = auto_detect

        self.orchestrator = MemoryOptimizationOrchestrator(config)

    def optimize_model(self, model: nn.Module, optimizer=None):
        """Optimize a model"""
        return self.orchestrator.optimize_model(model, optimizer)

    def optimize_step(self):
        """Get context manager for training step"""
        return self.orchestrator.optimize_step()

    def adapt(self, metrics: Dict[str, float]):
        """Adapt based on metrics"""
        self.orchestrator.adapt(metrics)

    def summary(self) -> Dict[str, Any]:
        """Get summary"""
        return self.orchestrator.get_summary()

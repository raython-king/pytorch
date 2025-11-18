"""
Diagnostics Agent

Responsible for diagnosing hardware capabilities and workload characteristics.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_agent import BaseAgent, AgentDecision
from ..diagnostics import HardwareDiagnostics, MemoryProfiler


class DiagnosticsAgent(BaseAgent):
    """
    Agent specialized in hardware diagnostics and workload analysis.

    This agent profiles the hardware and analyzes the workload to
    inform optimization decisions.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.hardware_diagnostics = HardwareDiagnostics()
        self.memory_profiler = MemoryProfiler()
        self.hardware_caps = None
        self.workload_profile = {}

    def observe(self, environment: Dict[str, Any]) -> None:
        """Observe hardware and workload"""
        # Diagnose hardware if not done
        if self.hardware_caps is None:
            self.hardware_caps = self.hardware_diagnostics.diagnose()

        # Capture memory snapshot
        snapshot = self.memory_profiler.capture_snapshot()

        # Update state
        self.state = {
            'hardware': self.hardware_caps,
            'memory_snapshot': snapshot,
            'model': environment.get('model'),
            'batch_size': environment.get('batch_size'),
            'oom_risk': self.memory_profiler.predict_oom_risk(),
        }

    def decide(self) -> AgentDecision:
        """Analyze system and recommend actions"""
        if not self.state:
            return AgentDecision(
                action="wait",
                confidence=1.0,
                reasoning="No observations yet"
            )

        # Check for memory pressure
        oom_risk = self.state.get('oom_risk', 0.0)
        snapshot = self.state.get('memory_snapshot')

        if oom_risk > 0.7:
            # High OOM risk - recommend aggressive optimization
            return AgentDecision(
                action="optimize",
                confidence=0.9,
                parameters={
                    'urgency': 'high',
                    'recommended_strategies': self._get_recommended_strategies(),
                },
                reasoning=f"High OOM risk detected: {oom_risk:.2f}"
            )

        elif snapshot and snapshot.gpu_utilization:
            max_util = max(snapshot.gpu_utilization)
            if max_util > 85:
                return AgentDecision(
                    action="optimize",
                    confidence=0.7,
                    parameters={
                        'urgency': 'medium',
                        'recommended_strategies': self._get_recommended_strategies(),
                    },
                    reasoning=f"High GPU memory utilization: {max_util:.1f}%"
                )

        return AgentDecision(
            action="monitor",
            confidence=0.8,
            reasoning="System stable, continue monitoring"
        )

    def _get_recommended_strategies(self) -> list:
        """Get recommended strategies based on hardware"""
        if not self.hardware_caps:
            return []

        recommendations = []

        # Mixed precision if supported
        if self.hardware_caps.supports_amp:
            recommendations.append('mixed_precision')

        # Checkpointing for limited memory
        if self.hardware_caps.gpu_total_memory:
            avg_memory = sum(self.hardware_caps.gpu_total_memory) / len(self.hardware_caps.gpu_total_memory)
            if avg_memory < 16:
                recommendations.append('gradient_checkpointing')

        # CPU offloading if good CPU memory
        if self.hardware_caps.cpu_total_memory > 32:
            recommendations.append('cpu_offloading')

        # Gradient compression for multi-GPU
        if self.hardware_caps.num_gpus > 1:
            recommendations.append('gradient_compression')

        return recommendations

    def learn(self, reward: float, next_state: Dict[str, Any]) -> None:
        """Update diagnostics based on feedback"""
        # Record workload characteristics
        if 'throughput' in next_state:
            self.workload_profile['throughput'] = next_state['throughput']
        if 'memory_efficiency' in next_state:
            self.workload_profile['memory_efficiency'] = next_state['memory_efficiency']

    def get_hardware_capabilities(self):
        """Get hardware capabilities"""
        if self.hardware_caps is None:
            self.hardware_caps = self.hardware_diagnostics.diagnose()
        return self.hardware_caps

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return self.memory_profiler.get_statistics()

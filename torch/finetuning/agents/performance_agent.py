"""
Performance Monitoring Agent
"""

import time
from typing import Dict, Any, List
from collections import deque
import numpy as np
from .base_agent import BaseFineTuningAgent, FineTuningDecision


class PerformanceMonitoringAgent(BaseFineTuningAgent):
    """Monitors fine-tuning performance"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.metrics_history = deque(maxlen=100)

    def observe(self, environment: Dict[str, Any]) -> None:
        """Observe performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'loss': environment.get('loss', 0.0),
            'accuracy': environment.get('accuracy', 0.0),
            'throughput': environment.get('throughput', 0.0),
            'memory_usage': environment.get('memory_usage', 0.0),
        }

        self.metrics_history.append(metrics)
        self.state = {'current_metrics': metrics}

    def decide(self) -> FineTuningDecision:
        """Analyze performance trends"""
        if len(self.metrics_history) < 10:
            return FineTuningDecision(
                method='current',
                confidence=1.0,
                config={},
                reasoning="Collecting baseline metrics"
            )

        # Check for performance issues
        recent_losses = [m['loss'] for m in self.metrics_history if m['loss'] > 0]

        if not recent_losses:
            return FineTuningDecision(
                method='current',
                confidence=1.0,
                config={},
                reasoning="No loss data yet"
            )

        # Check if loss is plateauing or increasing
        if len(recent_losses) >= 20:
            recent_trend = np.mean(np.diff(recent_losses[-20:]))

            if recent_trend > 0:  # Loss increasing
                return FineTuningDecision(
                    method='adjust',
                    confidence=0.7,
                    config={'action': 'reduce_lr'},
                    reasoning="Loss increasing, may need adjustment"
                )

        return FineTuningDecision(
            method='current',
            confidence=0.9,
            config={},
            reasoning="Performance stable"
        )

    def learn(self, reward: float, next_state: Dict[str, Any]) -> None:
        """Learn from performance outcomes"""
        pass

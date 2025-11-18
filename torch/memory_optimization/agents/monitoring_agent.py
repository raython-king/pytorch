"""
Monitoring Agent

Continuously monitors system performance and memory usage.
"""

import time
import torch
from typing import Dict, Any, List
from collections import deque
import numpy as np
from .base_agent import BaseAgent, AgentDecision


class MonitoringAgent(BaseAgent):
    """
    Agent responsible for continuous monitoring of system metrics.

    Tracks memory usage, throughput, and identifies anomalies or
    performance degradation.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.window_size = config.get('monitoring_window', 100)
        self.metrics_history = deque(maxlen=self.window_size)
        self.alert_threshold = config.get('alert_threshold', 0.90)
        self.baseline_metrics = {}

    def observe(self, environment: Dict[str, Any]) -> None:
        """Observe current metrics"""
        metrics = {
            'timestamp': time.time(),
            'memory_usage': environment.get('memory_usage', {}),
            'throughput': environment.get('throughput', 0.0),
            'iteration_time': environment.get('iteration_time', 0.0),
            'batch_size': environment.get('batch_size', 0),
        }

        self.metrics_history.append(metrics)
        self.state = {
            'current_metrics': metrics,
            'history_length': len(self.metrics_history),
        }

    def decide(self) -> AgentDecision:
        """Analyze metrics and decide on actions"""
        if len(self.metrics_history) < 2:
            return AgentDecision(
                action="monitor",
                confidence=1.0,
                reasoning="Collecting baseline metrics"
            )

        # Check for anomalies
        anomalies = self._detect_anomalies()

        if anomalies:
            return AgentDecision(
                action="alert",
                confidence=0.8,
                parameters={
                    'anomalies': anomalies,
                    'recommendations': self._get_recommendations(anomalies),
                },
                reasoning=f"Detected {len(anomalies)} anomalies"
            )

        # Check for performance degradation
        degradation = self._check_degradation()

        if degradation:
            return AgentDecision(
                action="optimize",
                confidence=0.7,
                parameters={
                    'degradation': degradation,
                },
                reasoning=f"Performance degradation detected: {degradation}"
            )

        return AgentDecision(
            action="monitor",
            confidence=0.9,
            reasoning="System performing normally"
        )

    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics"""
        anomalies = []

        if not self.metrics_history:
            return anomalies

        current = self.metrics_history[-1]

        # Check memory usage
        memory_usage = current.get('memory_usage', {})
        if memory_usage:
            gpu_util = memory_usage.get('gpu_utilization', [])
            if gpu_util and max(gpu_util) > self.alert_threshold * 100:
                anomalies.append({
                    'type': 'high_memory',
                    'severity': 'high',
                    'value': max(gpu_util),
                    'threshold': self.alert_threshold * 100,
                })

        # Check throughput drop
        if len(self.metrics_history) >= 10:
            recent_throughput = [m['throughput'] for m in list(self.metrics_history)[-10:]]
            if recent_throughput and sum(recent_throughput) > 0:
                avg_throughput = np.mean(recent_throughput)
                current_throughput = current.get('throughput', 0)

                if current_throughput < avg_throughput * 0.7:
                    anomalies.append({
                        'type': 'throughput_drop',
                        'severity': 'medium',
                        'value': current_throughput,
                        'baseline': avg_throughput,
                    })

        # Check iteration time spike
        if len(self.metrics_history) >= 10:
            recent_times = [m['iteration_time'] for m in list(self.metrics_history)[-10:]]
            if recent_times and sum(recent_times) > 0:
                avg_time = np.mean(recent_times)
                current_time = current.get('iteration_time', 0)

                if current_time > avg_time * 1.5:
                    anomalies.append({
                        'type': 'slow_iteration',
                        'severity': 'low',
                        'value': current_time,
                        'baseline': avg_time,
                    })

        return anomalies

    def _check_degradation(self) -> Optional[str]:
        """Check for performance degradation"""
        if len(self.metrics_history) < 50:
            return None

        # Compare first half vs second half
        half = len(self.metrics_history) // 2
        first_half = list(self.metrics_history)[:half]
        second_half = list(self.metrics_history)[half:]

        first_throughput = np.mean([m['throughput'] for m in first_half if m['throughput'] > 0])
        second_throughput = np.mean([m['throughput'] for m in second_half if m['throughput'] > 0])

        if first_throughput > 0 and second_throughput < first_throughput * 0.9:
            return f"Throughput degraded from {first_throughput:.2f} to {second_throughput:.2f}"

        return None

    def _get_recommendations(self, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Get recommendations based on anomalies"""
        recommendations = []

        for anomaly in anomalies:
            if anomaly['type'] == 'high_memory':
                recommendations.append("Consider enabling gradient checkpointing")
                recommendations.append("Try CPU offloading for optimizer states")
            elif anomaly['type'] == 'throughput_drop':
                recommendations.append("Check for memory fragmentation")
                recommendations.append("Consider adjusting batch size")
            elif anomaly['type'] == 'slow_iteration':
                recommendations.append("Profile computation bottlenecks")
                recommendations.append("Check data loading pipeline")

        return list(set(recommendations))

    def learn(self, reward: float, next_state: Dict[str, Any]) -> None:
        """Update baseline metrics"""
        if len(self.metrics_history) >= 20:
            recent_metrics = list(self.metrics_history)[-20:]
            self.baseline_metrics = {
                'avg_throughput': np.mean([m['throughput'] for m in recent_metrics if m['throughput'] > 0]),
                'avg_iteration_time': np.mean([m['iteration_time'] for m in recent_metrics if m['iteration_time'] > 0]),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        if not self.metrics_history:
            return {}

        metrics_list = list(self.metrics_history)

        throughputs = [m['throughput'] for m in metrics_list if m['throughput'] > 0]
        iteration_times = [m['iteration_time'] for m in metrics_list if m['iteration_time'] > 0]

        stats = {
            'num_samples': len(metrics_list),
        }

        if throughputs:
            stats['throughput'] = {
                'mean': np.mean(throughputs),
                'std': np.std(throughputs),
                'min': np.min(throughputs),
                'max': np.max(throughputs),
            }

        if iteration_times:
            stats['iteration_time'] = {
                'mean': np.mean(iteration_times),
                'std': np.std(iteration_times),
                'min': np.min(iteration_times),
                'max': np.max(iteration_times),
            }

        return stats

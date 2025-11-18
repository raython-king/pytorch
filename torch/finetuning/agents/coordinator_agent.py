"""
Fine-tuning Coordinator Agent
"""

from typing import Dict, Any, List
import numpy as np
from .base_agent import BaseFineTuningAgent, FineTuningDecision


class FineTuningCoordinator(BaseFineTuningAgent):
    """Coordinates decisions from multiple fine-tuning agents"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.agent_decisions: Dict[str, FineTuningDecision] = {}
        self.agent_weights = {
            'method_selector': 0.5,
            'hardware_analysis': 0.3,
            'performance_monitoring': 0.2,
        }

    def observe(self, environment: Dict[str, Any]) -> None:
        """Receive decisions from other agents"""
        self.agent_decisions = environment.get('agent_decisions', {})
        self.state = {'num_agents': len(self.agent_decisions)}

    def decide(self) -> FineTuningDecision:
        """Make final decision based on agent inputs"""
        if not self.agent_decisions:
            return FineTuningDecision(
                method='lora',
                confidence=0.5,
                config={'r': 8},
                reasoning="No agent decisions, using default"
            )

        # Weighted voting for methods
        method_votes: Dict[str, float] = {}
        method_configs: Dict[str, List[Dict]] = {}

        for agent_name, decision in self.agent_decisions.items():
            agent_type = self._get_agent_type(agent_name)
            weight = self.agent_weights.get(agent_type, 0.2)

            vote_strength = decision.confidence * weight
            method = decision.method

            if method not in method_votes:
                method_votes[method] = 0.0
                method_configs[method] = []

            method_votes[method] += vote_strength
            method_configs[method].append(decision.config)

        # Select best method
        best_method = max(method_votes.items(), key=lambda x: x[1])
        selected_method = best_method[0]
        confidence = best_method[1] / sum(method_votes.values()) if method_votes else 0.5

        # Merge configurations
        configs = method_configs.get(selected_method, [{}])
        merged_config = self._merge_configs(configs)

        return FineTuningDecision(
            method=selected_method,
            confidence=confidence,
            config=merged_config,
            reasoning=f"Consensus: {selected_method} ({confidence:.2f})"
        )

    def _get_agent_type(self, agent_name: str) -> str:
        """Extract agent type from name"""
        if 'selector' in agent_name.lower() or 'method' in agent_name.lower():
            return 'method_selector'
        elif 'hardware' in agent_name.lower():
            return 'hardware_analysis'
        elif 'performance' in agent_name.lower() or 'monitor' in agent_name.lower():
            return 'performance_monitoring'
        return 'other'

    def _merge_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple configurations"""
        if not configs:
            return {}

        merged = {}
        for config in configs:
            for key, value in config.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(value, (int, float)):
                    # Average numeric values
                    merged[key] = (merged[key] + value) / 2

        return merged

    def learn(self, reward: float, next_state: Dict[str, Any]) -> None:
        """Adjust agent weights based on outcomes"""
        if not self.history:
            return

        # Update agent weights based on performance
        for agent_name in self.agent_decisions.keys():
            agent_type = self._get_agent_type(agent_name)
            if agent_type in self.agent_weights:
                learning_rate = 0.05
                delta = learning_rate * reward

                self.agent_weights[agent_type] = np.clip(
                    self.agent_weights[agent_type] + delta,
                    0.1, 0.7
                )

        # Normalize weights
        total = sum(self.agent_weights.values())
        if total > 0:
            self.agent_weights = {k: v/total for k, v in self.agent_weights.items()}

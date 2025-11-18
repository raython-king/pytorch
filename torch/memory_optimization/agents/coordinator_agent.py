"""
Coordinator Agent

Coordinates decisions from multiple agents and resolves conflicts.
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentDecision
import numpy as np


class CoordinatorAgent(BaseAgent):
    """
    Agent that coordinates decisions from multiple specialized agents.

    This agent receives recommendations from other agents and makes
    the final decision on which actions to take, resolving conflicts
    and balancing trade-offs.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.agent_decisions: Dict[str, AgentDecision] = {}
        self.agent_weights = {
            'diagnostics': 0.3,
            'strategy_selector': 0.4,
            'monitoring': 0.3,
        }

    def observe(self, environment: Dict[str, Any]) -> None:
        """Receive decisions from other agents"""
        self.agent_decisions = environment.get('agent_decisions', {})
        self.state = {
            'num_agents': len(self.agent_decisions),
            'consensus': self._check_consensus(),
        }

    def decide(self) -> AgentDecision:
        """Make final decision based on agent inputs"""
        if not self.agent_decisions:
            return AgentDecision(
                action="wait",
                confidence=1.0,
                reasoning="No agent decisions received"
            )

        # Check for unanimous decision
        consensus = self._check_consensus()
        if consensus:
            return AgentDecision(
                action=consensus['action'],
                confidence=consensus['confidence'],
                strategy=consensus.get('strategy'),
                parameters=consensus.get('parameters', {}),
                reasoning=f"Unanimous decision from {len(self.agent_decisions)} agents"
            )

        # Resolve conflicts using weighted voting
        final_decision = self._weighted_voting()

        return final_decision

    def _check_consensus(self) -> Optional[Dict[str, Any]]:
        """Check if all agents agree"""
        if not self.agent_decisions:
            return None

        actions = [d.action for d in self.agent_decisions.values()]
        if len(set(actions)) == 1:
            # All agents agree on action
            first_decision = list(self.agent_decisions.values())[0]
            avg_confidence = np.mean([d.confidence for d in self.agent_decisions.values()])

            return {
                'action': first_decision.action,
                'confidence': avg_confidence,
                'strategy': first_decision.strategy,
                'parameters': first_decision.parameters,
            }

        return None

    def _weighted_voting(self) -> AgentDecision:
        """Resolve conflicts using weighted voting"""
        # Collect weighted votes for each action
        action_votes: Dict[str, float] = {}
        action_decisions: Dict[str, List[AgentDecision]] = {}

        for agent_name, decision in self.agent_decisions.items():
            # Get agent type from name
            agent_type = self._get_agent_type(agent_name)
            weight = self.agent_weights.get(agent_type, 0.2)

            # Weighted vote
            vote_strength = decision.confidence * weight

            if decision.action not in action_votes:
                action_votes[decision.action] = 0.0
                action_decisions[decision.action] = []

            action_votes[decision.action] += vote_strength
            action_decisions[decision.action].append(decision)

        # Select action with highest vote
        if not action_votes:
            return AgentDecision(
                action="wait",
                confidence=0.5,
                reasoning="No valid actions proposed"
            )

        best_action = max(action_votes.items(), key=lambda x: x[1])
        action = best_action[0]
        vote_sum = best_action[1]

        # Get decisions for this action
        decisions = action_decisions[action]

        # Merge parameters and strategies
        merged_params = {}
        strategy = None
        for decision in decisions:
            if decision.parameters:
                merged_params.update(decision.parameters)
            if decision.strategy and not strategy:
                strategy = decision.strategy

        # Calculate confidence (normalized vote strength)
        total_weight = sum(action_votes.values())
        confidence = vote_sum / total_weight if total_weight > 0 else 0.5

        return AgentDecision(
            action=action,
            confidence=confidence,
            strategy=strategy,
            parameters=merged_params,
            reasoning=f"Weighted voting: {action} ({vote_sum:.2f}/{total_weight:.2f})"
        )

    def _get_agent_type(self, agent_name: str) -> str:
        """Extract agent type from agent name"""
        if 'diagnostic' in agent_name.lower():
            return 'diagnostics'
        elif 'selector' in agent_name.lower() or 'strategy' in agent_name.lower():
            return 'strategy_selector'
        elif 'monitor' in agent_name.lower():
            return 'monitoring'
        else:
            return 'other'

    def learn(self, reward: float, next_state: Dict[str, Any]) -> None:
        """Adjust agent weights based on outcomes"""
        # Identify which agents' recommendations were followed
        last_decision = self.history[-1] if self.history else None
        if not last_decision:
            return

        # Update weights based on reward
        for agent_name, decision in self.agent_decisions.items():
            agent_type = self._get_agent_type(agent_name)

            if agent_type not in self.agent_weights:
                continue

            # If agent's recommendation aligned with final decision, update weight
            if decision.action == last_decision.action:
                # Positive reward -> increase weight
                # Negative reward -> decrease weight
                learning_rate = 0.05
                delta = learning_rate * reward * decision.confidence

                self.agent_weights[agent_type] = np.clip(
                    self.agent_weights[agent_type] + delta,
                    0.1,  # min weight
                    0.6   # max weight
                )

        # Normalize weights
        total = sum(self.agent_weights.values())
        if total > 0:
            self.agent_weights = {
                k: v / total for k, v in self.agent_weights.items()
            }

    def get_agent_weights(self) -> Dict[str, float]:
        """Get current agent weights"""
        return self.agent_weights.copy()

    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of recent decisions"""
        if not self.history:
            return {}

        recent = self.history[-10:]

        actions = [d.action for d in recent]
        confidences = [d.confidence for d in recent]

        return {
            'total_decisions': len(self.history),
            'recent_decisions': len(recent),
            'action_distribution': {
                action: actions.count(action) for action in set(actions)
            },
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
        }

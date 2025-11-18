"""
Strategy Selector Agent

Uses ML models to select the best optimization strategies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .base_agent import BaseAgent, AgentDecision
from ..config import OptimizationStrategy


class StrategySelectorAgent(BaseAgent):
    """
    Agent that uses ML to select optimal optimization strategies.

    Uses various ML models (RL, GNN, Transformer) to predict which
    combination of strategies will yield the best results.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.model_type = config.get('ml_model_type', 'ensemble')
        self.use_rl = config.get('use_rl', False)
        self.online_learning = config.get('online_learning', True)

        # Strategy performance tracking
        self.strategy_performance: Dict[str, List[float]] = {}
        self.strategy_counts: Dict[str, int] = {}

        # ML model (simplified for now)
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize ML model for strategy selection"""
        # For simplicity, we'll use a rule-based system with learned weights
        # In practice, this could be a neural network
        self.strategy_weights = {
            'mixed_precision': 1.0,
            'gradient_checkpointing': 0.9,
            'activation_checkpointing': 0.85,
            'gradient_compression': 0.8,
            'cpu_offloading': 0.7,
            'dynamic_batch_size': 0.75,
            'memory_efficient_optimizer': 0.9,
        }

    def observe(self, environment: Dict[str, Any]) -> None:
        """Observe current state"""
        self.state = {
            'hardware_caps': environment.get('hardware_caps'),
            'memory_usage': environment.get('memory_usage'),
            'current_strategies': environment.get('current_strategies', []),
            'performance_metrics': environment.get('performance_metrics', {}),
            'constraints': environment.get('constraints', {}),
        }

    def decide(self) -> AgentDecision:
        """Select best optimization strategies"""
        if not self.state:
            return AgentDecision(
                action="wait",
                confidence=1.0,
                reasoning="No observations yet"
            )

        # Get candidate strategies
        candidates = self._generate_candidates()

        # Score each strategy
        scored_strategies = []
        for strategy in candidates:
            score = self._score_strategy(strategy)
            scored_strategies.append((strategy, score))

        # Sort by score
        scored_strategies.sort(key=lambda x: x[1], reverse=True)

        if not scored_strategies:
            return AgentDecision(
                action="wait",
                confidence=0.5,
                reasoning="No viable strategies found"
            )

        # Select top strategy
        best_strategy, best_score = scored_strategies[0]

        return AgentDecision(
            action="apply_strategy",
            confidence=min(best_score, 1.0),
            strategy=best_strategy,
            parameters={
                'alternatives': [s for s, _ in scored_strategies[1:4]],
                'scores': {s: sc for s, sc in scored_strategies[:5]},
            },
            reasoning=f"Selected {best_strategy} with score {best_score:.3f}"
        )

    def _generate_candidates(self) -> List[str]:
        """Generate candidate strategies"""
        hardware_caps = self.state.get('hardware_caps')
        current_strategies = self.state.get('current_strategies', [])

        candidates = []

        # Check hardware compatibility
        if hardware_caps:
            if hardware_caps.supports_amp and 'mixed_precision' not in current_strategies:
                candidates.append('mixed_precision')

            if 'gradient_checkpointing' not in current_strategies:
                candidates.append('gradient_checkpointing')

            if hardware_caps.cpu_total_memory > 16 and 'cpu_offloading' not in current_strategies:
                candidates.append('cpu_offloading')

            if hardware_caps.num_gpus > 1 and 'gradient_compression' not in current_strategies:
                candidates.append('gradient_compression')

        # Always consider these
        if 'memory_efficient_optimizer' not in current_strategies:
            candidates.append('memory_efficient_optimizer')

        if 'dynamic_batch_size' not in current_strategies:
            candidates.append('dynamic_batch_size')

        return candidates

    def _score_strategy(self, strategy: str) -> float:
        """Score a strategy based on expected performance"""
        # Base score from learned weights
        base_score = self.strategy_weights.get(strategy, 0.5)

        # Adjust based on historical performance
        if strategy in self.strategy_performance and self.strategy_performance[strategy]:
            avg_performance = np.mean(self.strategy_performance[strategy])
            base_score = base_score * 0.6 + avg_performance * 0.4

        # Adjust based on current context
        memory_usage = self.state.get('memory_usage', {})
        if memory_usage:
            gpu_util = memory_usage.get('gpu_utilization', [0])
            max_util = max(gpu_util) if gpu_util else 0

            # Boost aggressive strategies for high memory usage
            if max_util > 90:
                if strategy in ['gradient_checkpointing', 'cpu_offloading']:
                    base_score *= 1.2
            elif max_util < 70:
                # Prefer performance-oriented strategies
                if strategy == 'mixed_precision':
                    base_score *= 1.1

        # Exploration bonus (epsilon-greedy)
        count = self.strategy_counts.get(strategy, 0)
        if count < 5:
            exploration_bonus = 0.1 / (count + 1)
            base_score += exploration_bonus

        return base_score

    def learn(self, reward: float, next_state: Dict[str, Any]) -> None:
        """Learn from strategy outcomes"""
        if not self.online_learning:
            return

        # Get the strategy that was applied
        last_decision = self.history[-1] if self.history else None
        if not last_decision or not last_decision.strategy:
            return

        strategy = last_decision.strategy

        # Record performance
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        self.strategy_performance[strategy].append(reward)

        # Update count
        self.strategy_counts[strategy] = self.strategy_counts.get(strategy, 0) + 1

        # Update weights using simple moving average
        if len(self.strategy_performance[strategy]) > 0:
            avg_reward = np.mean(self.strategy_performance[strategy][-10:])
            # Update weight (learning rate = 0.1)
            self.strategy_weights[strategy] = (
                self.strategy_weights.get(strategy, 0.5) * 0.9 + avg_reward * 0.1
            )

    def get_strategy_rankings(self) -> List[Tuple[str, float]]:
        """Get current strategy rankings"""
        rankings = [(s, w) for s, w in self.strategy_weights.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies"""
        summary = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                summary[strategy] = {
                    'count': len(performances),
                    'mean': np.mean(performances),
                    'std': np.std(performances),
                    'min': np.min(performances),
                    'max': np.max(performances),
                    'weight': self.strategy_weights.get(strategy, 0.5),
                }
        return summary

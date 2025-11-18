"""
Method Selector Agent

Selects the best fine-tuning method based on model, task, and constraints.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from .base_agent import BaseFineTuningAgent, FineTuningDecision
from ..config import FineTuningMethod


class MethodSelectorAgent(BaseFineTuningAgent):
    """
    Agent that selects the best fine-tuning method.

    Considers:
    - Model size and architecture
    - Available memory
    - Training speed requirements
    - Accuracy requirements
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)

        # Method performance tracking
        self.method_scores = {
            'lora': 0.9,
            'qlora': 0.85,
            'adapter': 0.8,
            'prefix_tuning': 0.75,
            'prompt_tuning': 0.7,
            'ia3': 0.75,
            'bitfit': 0.6,
            'full': 0.5,
        }

        # Historical performance
        self.method_performance: Dict[str, List[float]] = {}

    def observe(self, environment: Dict[str, Any]) -> None:
        """Observe model and constraints"""
        self.state = {
            'model_size': environment.get('model_size', 0),
            'available_memory': environment.get('available_memory', 0),
            'model_type': environment.get('model_type', 'unknown'),
            'task': environment.get('task', 'unknown'),
            'prefer_memory': environment.get('prefer_memory_efficiency', 0.5),
            'prefer_speed': environment.get('prefer_training_speed', 0.3),
            'prefer_accuracy': environment.get('prefer_accuracy', 0.7),
            'max_trainable_ratio': environment.get('max_trainable_params_ratio', 0.1),
        }

    def decide(self) -> FineTuningDecision:
        """Select best fine-tuning method"""
        if not self.state:
            return FineTuningDecision(
                method='lora',
                confidence=0.5,
                config={'r': 8, 'alpha': 16},
                reasoning="No observations, defaulting to LoRA"
            )

        # Calculate scores for each method
        method_scores = {}

        for method in ['lora', 'qlora', 'adapter', 'prefix_tuning', 'prompt_tuning', 'ia3', 'bitfit']:
            score = self._score_method(method)
            method_scores[method] = score

        # Select best method
        best_method = max(method_scores.items(), key=lambda x: x[1])
        method_name = best_method[0]
        confidence = min(best_method[1], 1.0)

        # Generate configuration for selected method
        method_config = self._generate_config(method_name)

        # Estimate trainable parameters
        trainable_ratio = self._estimate_trainable_ratio(method_name, method_config)

        # Estimate memory usage
        memory_usage = self._estimate_memory(method_name, method_config)

        return FineTuningDecision(
            method=method_name,
            confidence=confidence,
            config=method_config,
            reasoning=f"Selected {method_name} with score {confidence:.3f}",
            expected_trainable_ratio=trainable_ratio,
            expected_memory_usage=memory_usage,
        )

    def _score_method(self, method: str) -> float:
        """Score a fine-tuning method based on current context"""
        # Base score
        base_score = self.method_scores.get(method, 0.5)

        # Adjust based on historical performance
        if method in self.method_performance and self.method_performance[method]:
            avg_perf = np.mean(self.method_performance[method][-5:])
            base_score = base_score * 0.6 + avg_perf * 0.4

        # Adjust based on preferences
        model_size = self.state.get('model_size', 0)
        available_memory = self.state.get('available_memory', float('inf'))
        prefer_memory = self.state.get('prefer_memory', 0.5)
        prefer_speed = self.state.get('prefer_speed', 0.3)
        prefer_accuracy = self.state.get('prefer_accuracy', 0.7)

        # Memory efficiency bonus
        memory_efficiency = {
            'qlora': 1.0,  # Most memory efficient
            'lora': 0.9,
            'ia3': 0.85,
            'bitfit': 0.8,
            'prefix_tuning': 0.75,
            'prompt_tuning': 0.7,
            'adapter': 0.65,
            'full': 0.0,
        }
        score = base_score * (1 + prefer_memory * memory_efficiency.get(method, 0.5))

        # Speed bonus
        speed_score = {
            'bitfit': 1.0,  # Fastest
            'ia3': 0.95,
            'lora': 0.9,
            'prompt_tuning': 0.85,
            'prefix_tuning': 0.8,
            'adapter': 0.75,
            'qlora': 0.7,  # Slower due to quantization overhead
            'full': 0.5,
        }
        score = score * (1 + prefer_speed * speed_score.get(method, 0.5) * 0.5)

        # Accuracy bonus
        accuracy_score = {
            'full': 1.0,  # Best accuracy
            'lora': 0.95,
            'qlora': 0.90,
            'adapter': 0.85,
            'ia3': 0.80,
            'prefix_tuning': 0.78,
            'prompt_tuning': 0.75,
            'bitfit': 0.70,
        }
        score = score * (1 + prefer_accuracy * accuracy_score.get(method, 0.7) * 0.3)

        # Memory constraint penalty
        if model_size > 0 and available_memory < float('inf'):
            memory_ratio = available_memory / model_size
            if memory_ratio < 2 and method not in ['qlora', 'lora', 'bitfit']:
                score *= 0.5  # Heavy penalty for memory-intensive methods

        return score

    def _generate_config(self, method: str) -> Dict[str, Any]:
        """Generate configuration for a method"""
        model_size = self.state.get('model_size', 0)
        available_memory = self.state.get('available_memory', float('inf'))

        if method == 'lora':
            # Adjust rank based on available memory
            if available_memory < model_size * 2:
                r = 4
            elif available_memory < model_size * 4:
                r = 8
            else:
                r = 16

            return {
                'r': r,
                'alpha': r * 2,
                'dropout': 0.1,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            }

        elif method == 'qlora':
            return {
                'r': 8,
                'alpha': 16,
                'dropout': 0.1,
                'load_in_4bit': True,
                'bnb_4bit_compute_dtype': 'float16',
            }

        elif method == 'adapter':
            bottleneck = 64 if available_memory > model_size * 3 else 32
            return {
                'bottleneck_size': bottleneck,
                'non_linearity': 'gelu',
                'adapter_dropout': 0.1,
            }

        elif method == 'prefix_tuning':
            return {
                'num_virtual_tokens': 20,
                'prefix_projection': True,
                'projection_hidden_size': 512,
            }

        elif method == 'prompt_tuning':
            return {
                'num_virtual_tokens': 8,
                'prompt_tuning_init': 'random',
            }

        elif method == 'ia3':
            return {
                'target_modules': ['k_proj', 'v_proj', 'down_proj'],
            }

        elif method == 'bitfit':
            return {}

        return {}

    def _estimate_trainable_ratio(self, method: str, config: Dict[str, Any]) -> float:
        """Estimate trainable parameter ratio"""
        ratios = {
            'bitfit': 0.001,  # ~0.1% (only biases)
            'ia3': 0.001,     # ~0.1%
            'prompt_tuning': 0.001,  # ~0.1%
            'prefix_tuning': 0.005,  # ~0.5%
            'lora': 0.01,     # ~1% (depends on rank)
            'qlora': 0.01,    # ~1%
            'adapter': 0.02,  # ~2%
            'full': 1.0,      # 100%
        }

        base_ratio = ratios.get(method, 0.01)

        # Adjust for configuration
        if method == 'lora' and 'r' in config:
            # Higher rank = more parameters
            r = config['r']
            base_ratio *= (r / 8)  # Scale based on rank

        return base_ratio

    def _estimate_memory(self, method: str, config: Dict[str, Any]) -> float:
        """Estimate memory usage"""
        model_size = self.state.get('model_size', 1.0)

        # Memory multipliers
        multipliers = {
            'qlora': 0.3,      # 4-bit quantization saves a lot
            'lora': 1.05,      # Slight increase for LoRA matrices
            'ia3': 1.01,       # Minimal increase
            'bitfit': 1.0,     # No additional memory
            'prompt_tuning': 1.01,
            'prefix_tuning': 1.05,
            'adapter': 1.10,
            'full': 1.5,       # Full fine-tuning needs more memory
        }

        return model_size * multipliers.get(method, 1.0)

    def learn(self, reward: float, next_state: Dict[str, Any]) -> None:
        """Learn from fine-tuning outcomes"""
        if not self.history:
            return

        last_decision = self.history[-1]
        method = last_decision.method

        # Record performance
        if method not in self.method_performance:
            self.method_performance[method] = []
        self.method_performance[method].append(reward)

        # Update method scores
        if len(self.method_performance[method]) > 0:
            avg_reward = np.mean(self.method_performance[method][-10:])
            # Exponential moving average
            self.method_scores[method] = self.method_scores.get(method, 0.5) * 0.9 + avg_reward * 0.1

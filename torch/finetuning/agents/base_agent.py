"""
Base Agent for Fine-tuning System
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import torch.nn as nn


@dataclass
class FineTuningDecision:
    """Decision made by a fine-tuning agent"""
    method: str  # Which fine-tuning method to use
    confidence: float  # Confidence in decision (0-1)
    config: Dict[str, Any]  # Configuration for the method
    reasoning: str = ""  # Explanation
    expected_trainable_ratio: float = 0.0  # Expected % of trainable params
    expected_memory_usage: float = 0.0  # Expected memory in GB


class BaseFineTuningAgent(ABC):
    """Base class for fine-tuning agents"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.state: Dict[str, Any] = {}
        self.history: List[FineTuningDecision] = []

    @abstractmethod
    def observe(self, environment: Dict[str, Any]) -> None:
        """Observe the current environment"""
        pass

    @abstractmethod
    def decide(self) -> FineTuningDecision:
        """Make a decision based on observations"""
        pass

    @abstractmethod
    def learn(self, reward: float, next_state: Dict[str, Any]) -> None:
        """Learn from outcomes"""
        pass

    def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update agent state"""
        self.state.update(new_state)

    def record_decision(self, decision: FineTuningDecision) -> None:
        """Record decision in history"""
        self.history.append(decision)

    def get_name(self) -> str:
        """Get agent name"""
        return self.__class__.__name__

    def reset(self) -> None:
        """Reset agent state"""
        self.state = {}
        self.history = []

"""
Base Agent Class

Defines the interface for all optimization agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import torch.nn as nn


@dataclass
class AgentDecision:
    """Decision made by an agent"""
    action: str  # Action to take (e.g., "apply_strategy", "monitor", "adjust")
    confidence: float  # Confidence in decision (0-1)
    strategy: Optional[str] = None  # Strategy to apply
    parameters: Dict[str, Any] = None  # Parameters for the action
    reasoning: str = ""  # Explanation of the decision

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class BaseAgent(ABC):
    """
    Base class for all optimization agents.

    Each agent is responsible for a specific aspect of memory optimization
    and makes decisions based on its observations and learned policies.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.state: Dict[str, Any] = {}
        self.history: List[AgentDecision] = []

    @abstractmethod
    def observe(self, environment: Dict[str, Any]) -> None:
        """Observe the current environment state"""
        pass

    @abstractmethod
    def decide(self) -> AgentDecision:
        """Make a decision based on observations"""
        pass

    @abstractmethod
    def learn(self, reward: float, next_state: Dict[str, Any]) -> None:
        """Learn from the outcome of a decision"""
        pass

    def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update agent's internal state"""
        self.state.update(new_state)

    def record_decision(self, decision: AgentDecision) -> None:
        """Record a decision in history"""
        self.history.append(decision)

    def get_name(self) -> str:
        """Get agent name"""
        return self.__class__.__name__

    def reset(self) -> None:
        """Reset agent state"""
        self.state = {}
        self.history = []

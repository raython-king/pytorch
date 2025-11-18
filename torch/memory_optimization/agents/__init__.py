"""
Multi-Agent System for Memory Optimization

Collection of specialized agents that collaborate to optimize memory usage.
"""

from .base_agent import BaseAgent, AgentDecision
from .diagnostics_agent import DiagnosticsAgent
from .strategy_selector_agent import StrategySelectorAgent
from .monitoring_agent import MonitoringAgent
from .coordinator_agent import CoordinatorAgent

__all__ = [
    "BaseAgent",
    "AgentDecision",
    "DiagnosticsAgent",
    "StrategySelectorAgent",
    "MonitoringAgent",
    "CoordinatorAgent",
]

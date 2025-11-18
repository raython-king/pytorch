"""
Base Agent Classes for Data Pipeline System

Defines the interface for all data pipeline agents that manage
the multi-level caching and data transfer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import time
import threading


class AgentRole(Enum):
    """Roles for different pipeline agents"""
    DISK_READER = "disk_reader"
    MEMORY_MANAGER = "memory_manager"
    REDIS_MANAGER = "redis_manager"
    GPU_TRANSFER = "gpu_transfer"
    PREFETCH_COORDINATOR = "prefetch_coordinator"
    ORCHESTRATOR = "orchestrator"


class AgentAction(Enum):
    """Actions that agents can perform"""
    LOAD_FROM_DISK = "load_from_disk"
    CACHE_TO_MEMORY = "cache_to_memory"
    CACHE_TO_REDIS = "cache_to_redis"
    TRANSFER_TO_GPU = "transfer_to_gpu"
    EVICT_FROM_CACHE = "evict_from_cache"
    PREFETCH_DATA = "prefetch_data"
    ADJUST_STRATEGY = "adjust_strategy"
    NO_ACTION = "no_action"


@dataclass
class DataRequest:
    """Request for data from pipeline"""
    sample_id: Any  # Sample identifier
    priority: int = 0  # Higher priority = processed first
    timestamp: float = 0.0  # Request timestamp
    requester: str = ""  # Agent that made the request
    metadata: Dict[str, Any] = None  # Additional metadata

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DataItem:
    """Data item in the pipeline"""
    sample_id: Any
    data: Any  # The actual data
    size_bytes: int  # Size in bytes
    location: str  # Current location (disk, memory, redis, gpu)
    access_count: int = 0  # Number of times accessed
    last_access_time: float = 0.0
    load_time: float = 0.0  # Time to load
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_access_time == 0.0:
            self.last_access_time = time.time()


@dataclass
class AgentDecision:
    """Decision made by an agent"""
    action: AgentAction
    confidence: float  # Confidence in decision (0-1)
    target_data: Optional[List[Any]] = None  # Data IDs to act on
    parameters: Dict[str, Any] = None  # Parameters for the action
    reasoning: str = ""  # Explanation of the decision
    expected_benefit: float = 0.0  # Expected performance benefit
    estimated_cost: float = 0.0  # Estimated resource cost

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.target_data is None:
            self.target_data = []


@dataclass
class AgentState:
    """State information for an agent"""
    agent_id: str
    role: AgentRole
    active: bool = True
    current_load: float = 0.0  # Current workload (0-1)
    performance_score: float = 1.0  # Performance metric
    error_count: int = 0
    success_count: int = 0
    last_update_time: float = 0.0

    # Resource usage
    memory_usage_bytes: int = 0
    cache_hit_rate: float = 0.0
    average_latency_ms: float = 0.0

    # Statistics
    total_requests: int = 0
    total_bytes_processed: int = 0

    def __post_init__(self):
        if self.last_update_time == 0.0:
            self.last_update_time = time.time()


@dataclass
class PipelineEnvironment:
    """Environment state observed by agents"""
    timestamp: float
    pending_requests: List[DataRequest]
    cached_items: Dict[str, List[DataItem]]  # location -> items

    # System state
    memory_available_bytes: int
    memory_pressure: float  # 0-1, higher = more pressure
    gpu_memory_available_bytes: int
    disk_io_bandwidth_mbps: float
    network_bandwidth_mbps: float

    # Performance metrics
    cache_hit_rates: Dict[str, float]  # location -> hit rate
    average_latencies: Dict[str, float]  # location -> latency (ms)
    throughput_mbps: float

    # Access patterns
    recent_access_sequence: List[Any]  # Recent sample IDs
    access_frequency: Dict[Any, int]  # sample_id -> count

    def __post_init__(self):
        if not hasattr(self, 'timestamp') or self.timestamp == 0.0:
            self.timestamp = time.time()


class BaseDataPipelineAgent(ABC):
    """
    Base class for all data pipeline agents.

    Each agent is responsible for managing a specific layer of the
    data pipeline or a specific aspect of data management.
    """

    def __init__(self, agent_id: str, role: AgentRole, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.role = role
        self.config = config

        # Agent state
        self.state = AgentState(agent_id=agent_id, role=role)
        self.history: List[Tuple[PipelineEnvironment, AgentDecision]] = []

        # Thread safety
        self.lock = threading.RLock()

        # Communication
        self.message_queue: List[Dict[str, Any]] = []

        # Learning
        self.reward_history: List[float] = []
        self.experience_buffer: List[Tuple[Any, Any, float, Any]] = []

    @abstractmethod
    def observe(self, environment: PipelineEnvironment) -> None:
        """
        Observe the current pipeline environment state.

        Args:
            environment: Current state of the pipeline
        """
        pass

    @abstractmethod
    def decide(self, request: Optional[DataRequest] = None) -> AgentDecision:
        """
        Make a decision based on current observations.

        Args:
            request: Optional specific data request to handle

        Returns:
            AgentDecision with action to take
        """
        pass

    @abstractmethod
    def execute(self, decision: AgentDecision) -> Tuple[bool, Any]:
        """
        Execute a decision.

        Args:
            decision: The decision to execute

        Returns:
            Tuple of (success, result)
        """
        pass

    @abstractmethod
    def learn(self, reward: float, next_environment: PipelineEnvironment) -> None:
        """
        Learn from the outcome of a decision.

        Args:
            reward: Reward signal for the last action
            next_environment: Environment state after action
        """
        pass

    def update_state(self, **kwargs) -> None:
        """Update agent's internal state"""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            self.state.last_update_time = time.time()

    def record_decision(
        self,
        environment: PipelineEnvironment,
        decision: AgentDecision
    ) -> None:
        """Record a decision in history"""
        with self.lock:
            self.history.append((environment, decision))

            # Keep history bounded
            max_history = self.config.get("max_history_size", 10000)
            if len(self.history) > max_history:
                self.history = self.history[-max_history:]

    def get_state(self) -> AgentState:
        """Get current agent state"""
        with self.lock:
            return self.state

    def send_message(self, recipient_id: str, message: Dict[str, Any]) -> None:
        """Send message to another agent"""
        with self.lock:
            self.message_queue.append({
                "from": self.agent_id,
                "to": recipient_id,
                "message": message,
                "timestamp": time.time()
            })

    def receive_messages(self) -> List[Dict[str, Any]]:
        """Receive pending messages"""
        with self.lock:
            messages = self.message_queue.copy()
            self.message_queue.clear()
            return messages

    def calculate_reward(
        self,
        decision: AgentDecision,
        outcome: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for a decision based on outcome.

        Args:
            decision: The decision that was made
            outcome: The outcome of executing the decision

        Returns:
            Reward value
        """
        reward = 0.0

        # Reward for successful execution
        if outcome.get("success", False):
            reward += 1.0
        else:
            reward -= 1.0

        # Reward for performance improvement
        latency = outcome.get("latency_ms", 0.0)
        if latency > 0:
            # Lower latency = higher reward
            reward += max(0, 1.0 - latency / 100.0)

        # Reward for cache hits
        if outcome.get("cache_hit", False):
            reward += 0.5

        # Penalty for resource usage
        resource_cost = decision.estimated_cost
        reward -= resource_cost * 0.1

        return reward

    def reset(self) -> None:
        """Reset agent state"""
        with self.lock:
            self.state = AgentState(agent_id=self.agent_id, role=self.role)
            self.history.clear()
            self.message_queue.clear()
            self.reward_history.clear()
            self.experience_buffer.clear()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        with self.lock:
            total_requests = self.state.total_requests
            if total_requests == 0:
                return {
                    "success_rate": 0.0,
                    "error_rate": 0.0,
                    "average_latency_ms": 0.0,
                    "cache_hit_rate": 0.0,
                    "throughput_mbps": 0.0,
                }

            return {
                "success_rate": self.state.success_count / total_requests,
                "error_rate": self.state.error_count / total_requests,
                "average_latency_ms": self.state.average_latency_ms,
                "cache_hit_rate": self.state.cache_hit_rate,
                "throughput_mbps": (
                    self.state.total_bytes_processed / (1024 * 1024)
                ) / max(1, total_requests),
                "total_requests": total_requests,
                "performance_score": self.state.performance_score,
            }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id}, "
            f"role={self.role.value}, "
            f"active={self.state.active})"
        )

"""
Memory Cache Manager Agent

Manages in-memory caching with intelligent eviction policies.
"""

import threading
import time
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Tuple

from .base_agent import (
    BaseDataPipelineAgent,
    AgentRole,
    AgentAction,
    AgentDecision,
    DataRequest,
    DataItem,
    PipelineEnvironment,
)


class MemoryCacheAgent(BaseDataPipelineAgent):
    """
    Agent responsible for managing in-memory data cache.

    Features:
    - Multiple cache eviction policies (LRU, LFU, ARC)
    - Adaptive cache sizing
    - Prefetch coordination
    - Memory pressure awareness
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentRole.MEMORY_MANAGER, config)

        self.memory_config = config.get("memory", {})
        self.max_size_bytes = int(self.memory_config.get("max_size_gb", 8.0) * 1024 ** 3)

        # Cache storage
        self.cache: OrderedDict[Any, DataItem] = OrderedDict()
        self.cache_lock = threading.RLock()

        # Cache statistics
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Access frequency (for LFU)
        self.access_frequency: Dict[Any, int] = {}

        # Memory pressure monitoring
        self.memory_pressure_threshold = 0.9  # 90% full

    def observe(self, environment: PipelineEnvironment) -> None:
        """Observe environment and update state"""
        memory_hit_rate = environment.cache_hit_rates.get("memory", 0.0)
        memory_latency = environment.average_latencies.get("memory", 0.0)

        # Update memory pressure based on system state
        memory_pressure = environment.memory_pressure

        self.update_state(
            cache_hit_rate=memory_hit_rate,
            average_latency_ms=memory_latency,
            current_load=self.current_size_bytes / self.max_size_bytes,
            memory_usage_bytes=self.current_size_bytes,
        )

        # Adjust cache size if under memory pressure
        if memory_pressure > self.memory_pressure_threshold:
            self._reduce_cache_size(factor=0.9)

    def decide(self, request: Optional[DataRequest] = None) -> AgentDecision:
        """Decide on cache management actions"""

        # Check if we need to evict items
        if self.current_size_bytes > self.max_size_bytes * 0.95:
            return AgentDecision(
                action=AgentAction.EVICT_FROM_CACHE,
                confidence=1.0,
                reasoning="Cache near capacity, need eviction",
                expected_benefit=0.3,
            )

        # Check if we have cache hits
        if request is not None:
            with self.cache_lock:
                if request.sample_id in self.cache:
                    # Cache hit!
                    return AgentDecision(
                        action=AgentAction.NO_ACTION,
                        confidence=1.0,
                        target_data=[request.sample_id],
                        reasoning="Cache hit",
                    )
                else:
                    # Cache miss - need to load
                    return AgentDecision(
                        action=AgentAction.CACHE_TO_MEMORY,
                        confidence=1.0,
                        target_data=[request.sample_id],
                        reasoning="Cache miss, need to load",
                    )

        return AgentDecision(
            action=AgentAction.NO_ACTION,
            confidence=1.0,
        )

    def execute(self, decision: AgentDecision) -> Tuple[bool, Any]:
        """Execute cache management decision"""
        try:
            if decision.action == AgentAction.CACHE_TO_MEMORY:
                # This is handled externally (by orchestrator)
                # Just record the request
                return True, None

            elif decision.action == AgentAction.EVICT_FROM_CACHE:
                evicted = self._evict_items()
                self.evictions += len(evicted)
                return True, evicted

            return True, None

        except Exception as e:
            self.state.error_count += 1
            return False, str(e)

    def learn(self, reward: float, next_environment: PipelineEnvironment) -> None:
        """Learn from outcomes and adapt strategy"""
        self.reward_history.append(reward)

        # Keep history bounded
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

        # Adapt eviction policy based on performance
        avg_reward = sum(self.reward_history[-100:]) / 100 if len(self.reward_history) >= 100 else 0

        # If performance is poor, try different eviction policy
        if avg_reward < 0.4:
            current_policy = self.memory_config.get("cache_policy", "ARC")
            policies = ["LRU", "LFU", "ARC"]

            # Try next policy
            if current_policy in policies:
                idx = policies.index(current_policy)
                next_policy = policies[(idx + 1) % len(policies)]
                self.memory_config["cache_policy"] = next_policy

    def add_to_cache(self, data_item: DataItem) -> bool:
        """Add item to cache"""
        with self.cache_lock:
            # Check if we have space
            if self.current_size_bytes + data_item.size_bytes > self.max_size_bytes:
                # Need to evict
                self._evict_items(target_size=data_item.size_bytes)

            # Add to cache
            self.cache[data_item.sample_id] = data_item
            self.current_size_bytes += data_item.size_bytes
            self.access_frequency[data_item.sample_id] = 1

            self.state.total_requests += 1
            self.state.total_bytes_processed += data_item.size_bytes

            return True

    def get_from_cache(self, sample_id: Any) -> Optional[DataItem]:
        """Get item from cache"""
        with self.cache_lock:
            if sample_id in self.cache:
                item = self.cache[sample_id]

                # Update access info
                item.access_count += 1
                item.last_access_time = time.time()

                # Update frequency
                self.access_frequency[sample_id] = self.access_frequency.get(sample_id, 0) + 1

                # Move to end for LRU
                self.cache.move_to_end(sample_id)

                self.hits += 1
                return item
            else:
                self.misses += 1
                return None

    def _evict_items(self, target_size: int = 0) -> List[DataItem]:
        """Evict items based on policy"""
        evicted = []
        policy = self.memory_config.get("cache_policy", "ARC")

        with self.cache_lock:
            if policy == "LRU":
                # Evict least recently used
                while (
                    self.cache and
                    (self.current_size_bytes + target_size > self.max_size_bytes or
                     self.current_size_bytes > self.max_size_bytes * 0.9)
                ):
                    sample_id, item = self.cache.popitem(last=False)
                    self.current_size_bytes -= item.size_bytes
                    evicted.append(item)
                    if sample_id in self.access_frequency:
                        del self.access_frequency[sample_id]

            elif policy == "LFU":
                # Evict least frequently used
                while (
                    self.cache and
                    (self.current_size_bytes + target_size > self.max_size_bytes or
                     self.current_size_bytes > self.max_size_bytes * 0.9)
                ):
                    # Find least frequently used
                    lfu_id = min(
                        self.cache.keys(),
                        key=lambda k: self.access_frequency.get(k, 0)
                    )
                    item = self.cache.pop(lfu_id)
                    self.current_size_bytes -= item.size_bytes
                    evicted.append(item)
                    if lfu_id in self.access_frequency:
                        del self.access_frequency[lfu_id]

            else:  # ARC (Adaptive Replacement Cache) - simplified version
                # Use combination of recency and frequency
                while (
                    self.cache and
                    (self.current_size_bytes + target_size > self.max_size_bytes or
                     self.current_size_bytes > self.max_size_bytes * 0.9)
                ):
                    # Score based on both recency and frequency
                    now = time.time()
                    scored_items = [
                        (
                            sample_id,
                            item,
                            (now - item.last_access_time) / max(1, self.access_frequency.get(sample_id, 1))
                        )
                        for sample_id, item in self.cache.items()
                    ]

                    # Evict item with highest score (old and infrequent)
                    if scored_items:
                        to_evict = max(scored_items, key=lambda x: x[2])
                        sample_id = to_evict[0]
                        item = self.cache.pop(sample_id)
                        self.current_size_bytes -= item.size_bytes
                        evicted.append(item)
                        if sample_id in self.access_frequency:
                            del self.access_frequency[sample_id]

        return evicted

    def _reduce_cache_size(self, factor: float = 0.9) -> None:
        """Reduce cache size by given factor"""
        target_size = int(self.max_size_bytes * factor)

        with self.cache_lock:
            while self.current_size_bytes > target_size and self.cache:
                self._evict_items()

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses

        return {
            "cache_size_mb": self.current_size_bytes / (1024 ** 2),
            "cache_items": len(self.cache),
            "hit_rate": self.hits / total_requests if total_requests > 0 else 0.0,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "utilization": self.current_size_bytes / self.max_size_bytes,
        }

    def clear(self) -> None:
        """Clear the cache"""
        with self.cache_lock:
            self.cache.clear()
            self.access_frequency.clear()
            self.current_size_bytes = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0

"""
Redis Cache Manager Agent

Manages distributed caching using Redis for multi-node training.
"""

import pickle
import threading
import time
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


class RedisCacheAgent(BaseDataPipelineAgent):
    """
    Agent responsible for managing Redis distributed cache.

    Features:
    - Distributed caching across multiple nodes
    - TTL-based expiration
    - Compression support
    - Cluster awareness
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentRole.REDIS_MANAGER, config)

        self.redis_config = config.get("redis", {})
        self.enabled = self.redis_config.get("enabled", False)

        # Redis client (lazy initialization)
        self.redis_client = None
        self.redis_lock = threading.Lock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.errors = 0

        if self.enabled:
            self._initialize_redis()

    def _initialize_redis(self) -> None:
        """Initialize Redis connection"""
        try:
            # Try to import redis
            import redis

            host = self.redis_config.get("host", "localhost")
            port = self.redis_config.get("port", 6379)
            password = self.redis_config.get("password")
            db = self.redis_config.get("db", 0)

            max_connections = self.redis_config.get("max_connections", 50)
            socket_timeout = self.redis_config.get("socket_timeout", 5.0)

            # Create connection pool
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                password=password,
                db=db,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
            )

            self.redis_client = redis.Redis(connection_pool=pool)

            # Test connection
            self.redis_client.ping()

            print(f"Redis agent {self.agent_id} connected to {host}:{port}")

        except ImportError:
            print("Warning: redis-py not installed. Redis caching disabled.")
            self.enabled = False
            self.redis_client = None

        except Exception as e:
            print(f"Warning: Failed to connect to Redis: {e}. Redis caching disabled.")
            self.enabled = False
            self.redis_client = None

    def observe(self, environment: PipelineEnvironment) -> None:
        """Observe environment and update state"""
        if not self.enabled:
            return

        redis_hit_rate = environment.cache_hit_rates.get("redis", 0.0)
        redis_latency = environment.average_latencies.get("redis", 0.0)

        self.update_state(
            cache_hit_rate=redis_hit_rate,
            average_latency_ms=redis_latency,
            current_load=0.5,  # Placeholder
        )

    def decide(self, request: Optional[DataRequest] = None) -> AgentDecision:
        """Decide on Redis cache operations"""
        if not self.enabled:
            return AgentDecision(
                action=AgentAction.NO_ACTION,
                confidence=1.0,
                reasoning="Redis disabled",
            )

        if request is not None:
            # Check if item in Redis
            key = self._make_key(request.sample_id)

            try:
                if self.redis_client and self.redis_client.exists(key):
                    return AgentDecision(
                        action=AgentAction.NO_ACTION,  # Cache hit
                        confidence=1.0,
                        target_data=[request.sample_id],
                        reasoning="Redis cache hit",
                    )
                else:
                    return AgentDecision(
                        action=AgentAction.CACHE_TO_REDIS,
                        confidence=0.8,
                        target_data=[request.sample_id],
                        reasoning="Redis cache miss",
                    )
            except Exception:
                pass

        return AgentDecision(
            action=AgentAction.NO_ACTION,
            confidence=1.0,
        )

    def execute(self, decision: AgentDecision) -> Tuple[bool, Any]:
        """Execute Redis cache decision"""
        if not self.enabled or not self.redis_client:
            return False, "Redis not available"

        try:
            if decision.action == AgentAction.CACHE_TO_REDIS:
                # This is handled by set_to_cache method
                return True, None

            return True, None

        except Exception as e:
            self.errors += 1
            self.state.error_count += 1
            return False, str(e)

    def learn(self, reward: float, next_environment: PipelineEnvironment) -> None:
        """Learn from outcomes"""
        self.reward_history.append(reward)

        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

        # Adapt TTL based on hit rate
        total_requests = self.hits + self.misses
        if total_requests > 100:
            hit_rate = self.hits / total_requests

            current_ttl = self.redis_config.get("ttl", 3600)

            if hit_rate < 0.3:
                # Low hit rate - increase TTL
                self.redis_config["ttl"] = min(7200, int(current_ttl * 1.2))
            elif hit_rate > 0.8:
                # High hit rate - can reduce TTL to save memory
                self.redis_config["ttl"] = max(1800, int(current_ttl * 0.9))

    def set_to_cache(self, data_item: DataItem) -> bool:
        """Store item in Redis"""
        if not self.enabled or not self.redis_client:
            return False

        try:
            key = self._make_key(data_item.sample_id)

            # Serialize data
            data_bytes = self._serialize(data_item.data)

            # Optionally compress
            if self.redis_config.get("compression", True):
                data_bytes = self._compress(data_bytes)

            # Store in Redis with TTL
            ttl = self.redis_config.get("ttl", 3600)
            self.redis_client.setex(key, ttl, data_bytes)

            self.sets += 1
            self.state.total_requests += 1
            self.state.total_bytes_processed += len(data_bytes)

            return True

        except Exception as e:
            self.errors += 1
            self.state.error_count += 1
            print(f"Error storing to Redis: {e}")
            return False

    def get_from_cache(self, sample_id: Any) -> Optional[DataItem]:
        """Retrieve item from Redis"""
        if not self.enabled or not self.redis_client:
            return None

        try:
            key = self._make_key(sample_id)

            # Get from Redis
            data_bytes = self.redis_client.get(key)

            if data_bytes is None:
                self.misses += 1
                return None

            # Decompress if needed
            if self.redis_config.get("compression", True):
                data_bytes = self._decompress(data_bytes)

            # Deserialize
            data = self._deserialize(data_bytes)

            # Create data item
            data_item = DataItem(
                sample_id=sample_id,
                data=data,
                size_bytes=len(data_bytes),
                location="redis",
                access_count=1,
                last_access_time=time.time(),
            )

            self.hits += 1
            return data_item

        except Exception as e:
            self.errors += 1
            print(f"Error retrieving from Redis: {e}")
            return None

    def _make_key(self, sample_id: Any) -> str:
        """Create Redis key from sample ID"""
        return f"pytorch_data:{sample_id}"

    def _serialize(self, data: Any) -> bytes:
        """Serialize data"""
        serialization = self.redis_config.get("serialization", "pickle")

        if serialization == "pickle":
            return pickle.dumps(data)
        else:
            # Fallback to pickle
            return pickle.dumps(data)

    def _deserialize(self, data_bytes: bytes) -> Any:
        """Deserialize data"""
        serialization = self.redis_config.get("serialization", "pickle")

        if serialization == "pickle":
            return pickle.loads(data_bytes)
        else:
            return pickle.loads(data_bytes)

    def _compress(self, data: bytes) -> bytes:
        """Compress data"""
        try:
            import zlib
            return zlib.compress(data, level=self.redis_config.get("compression_level", 6))
        except ImportError:
            return data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data"""
        try:
            import zlib
            return zlib.decompress(data)
        except ImportError:
            return data

    def get_statistics(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        total_requests = self.hits + self.misses

        stats = {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "errors": self.errors,
            "hit_rate": self.hits / total_requests if total_requests > 0 else 0.0,
        }

        # Get Redis server stats if available
        if self.enabled and self.redis_client:
            try:
                info = self.redis_client.info("memory")
                stats["redis_memory_used_mb"] = info.get("used_memory", 0) / (1024 ** 2)
                stats["redis_memory_peak_mb"] = info.get("used_memory_peak", 0) / (1024 ** 2)
            except Exception:
                pass

        return stats

    def clear(self) -> None:
        """Clear Redis cache"""
        if self.enabled and self.redis_client:
            try:
                # Clear only our keys
                pattern = "pytorch_data:*"
                cursor = 0

                while True:
                    cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)

                    if keys:
                        self.redis_client.delete(*keys)

                    if cursor == 0:
                        break

                self.hits = 0
                self.misses = 0
                self.sets = 0
                self.errors = 0

            except Exception as e:
                print(f"Error clearing Redis cache: {e}")

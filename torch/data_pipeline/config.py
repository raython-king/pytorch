"""
Configuration for Multi-Agent Dynamic Data Pipeline System

Manages settings for multi-level data caching and transfer:
Disk → Memory → Redis → GPU
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class CachePolicy(Enum):
    """Cache replacement policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    ARC = "arc"  # Adaptive Replacement Cache
    ML_BASED = "ml_based"  # Machine Learning based prediction


class PrefetchStrategy(Enum):
    """Data prefetching strategies"""
    SEQUENTIAL = "sequential"  # Sequential prefetch
    PATTERN_BASED = "pattern_based"  # Based on access patterns
    ML_PREDICTION = "ml_prediction"  # ML-based prediction
    ADAPTIVE = "adaptive"  # Adaptive strategy
    NONE = "none"  # No prefetching


@dataclass
class DiskLayerConfig:
    """Configuration for disk storage layer"""
    enabled: bool = True
    cache_dir: Optional[str] = None  # Default: /tmp/pytorch_data_cache
    max_workers: int = 4  # Number of disk I/O workers
    read_ahead_size: int = 64 * 1024 * 1024  # 64MB
    use_mmap: bool = True  # Use memory-mapped files
    compression: bool = False  # Compress data on disk
    compression_level: int = 3  # Compression level (1-9)

    # Performance tuning
    io_scheduler: str = "cfq"  # I/O scheduler (cfq, deadline, noop)
    direct_io: bool = False  # Use O_DIRECT for I/O
    async_io: bool = True  # Use asynchronous I/O


@dataclass
class MemoryLayerConfig:
    """Configuration for in-memory caching layer"""
    enabled: bool = True
    max_size_gb: float = 8.0  # Maximum memory cache size in GB
    cache_policy: CachePolicy = CachePolicy.ARC
    prefetch_size: int = 32  # Number of samples to prefetch
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    shared_memory: bool = True  # Use shared memory for multi-process

    # NUMA awareness
    numa_aware: bool = True
    numa_node: Optional[int] = None  # Specific NUMA node to use

    # Memory pool
    use_memory_pool: bool = True
    pool_size_multiplier: float = 1.5  # Pool size = max_size * multiplier


@dataclass
class RedisLayerConfig:
    """Configuration for Redis intermediate caching layer"""
    enabled: bool = False  # Disabled by default
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_size_gb: float = 4.0  # Maximum Redis cache size
    ttl: int = 3600  # Time to live in seconds

    # Connection pool
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0

    # Serialization
    serialization: str = "pickle"  # pickle, msgpack, json
    compression: bool = True  # Compress data in Redis

    # Cluster support
    cluster_mode: bool = False
    cluster_nodes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GPULayerConfig:
    """Configuration for GPU data transfer and management"""
    enabled: bool = True
    device: str = "cuda:0"  # Target GPU device
    prefetch_to_gpu: bool = True  # Prefetch data to GPU
    prefetch_queue_size: int = 2  # GPU prefetch queue size
    non_blocking: bool = True  # Non-blocking GPU transfer

    # Stream management
    use_streams: bool = True
    num_streams: int = 2  # Number of CUDA streams

    # Memory management
    pin_memory: bool = True
    use_unified_memory: bool = False  # Use CUDA unified memory

    # GPU Direct Storage (GDS)
    use_gds: bool = False  # Use GPUDirect Storage if available
    gds_batch_size: int = 64


@dataclass
class PrefetchConfig:
    """Configuration for intelligent prefetching"""
    strategy: PrefetchStrategy = PrefetchStrategy.ADAPTIVE
    prefetch_factor: int = 2  # How many batches to prefetch
    num_workers: int = 4  # Number of prefetch workers

    # Pattern detection
    pattern_window: int = 100  # Window size for pattern detection
    min_pattern_length: int = 3  # Minimum pattern length

    # ML-based prediction
    use_ml_predictor: bool = True
    predictor_model: str = "lstm"  # lstm, transformer, gru
    prediction_horizon: int = 10  # How many steps to predict

    # Adaptive strategy
    adaptation_interval: int = 100  # Adapt every N batches
    performance_threshold: float = 0.8  # Min performance to keep strategy


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    learning_rate: float = 0.001
    exploration_rate: float = 0.1  # Epsilon for epsilon-greedy
    discount_factor: float = 0.95  # Gamma for Q-learning

    # Communication
    communication_interval: int = 10  # How often agents communicate
    coordination_strategy: str = "hierarchical"  # hierarchical, peer_to_peer

    # Learning
    experience_replay: bool = True
    replay_buffer_size: int = 10000
    batch_size: int = 32
    update_frequency: int = 4  # Update model every N steps


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and profiling"""
    enabled: bool = True
    log_interval: int = 100  # Log stats every N batches
    detailed_profiling: bool = False  # Detailed performance profiling

    # Metrics to track
    track_hit_rates: bool = True
    track_latency: bool = True
    track_bandwidth: bool = True
    track_memory_usage: bool = True

    # Visualization
    enable_visualization: bool = False
    visualization_port: int = 8888


@dataclass
class DataPipelineConfig:
    """Main configuration for the multi-agent data pipeline system"""

    # Layer configurations
    disk: DiskLayerConfig = field(default_factory=DiskLayerConfig)
    memory: MemoryLayerConfig = field(default_factory=MemoryLayerConfig)
    redis: RedisLayerConfig = field(default_factory=RedisLayerConfig)
    gpu: GPULayerConfig = field(default_factory=GPULayerConfig)

    # Prefetch and caching
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)

    # Agent system
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Global settings
    auto_tune: bool = True  # Automatically tune parameters
    num_epochs: int = 100  # For tuning reference
    warmup_batches: int = 10  # Warmup period

    # Safety and fallback
    fallback_to_default: bool = True  # Fallback to default DataLoader on error
    error_retry_count: int = 3
    timeout: float = 300.0  # Timeout for operations (seconds)

    def validate(self) -> bool:
        """Validate configuration"""
        if self.memory.max_size_gb <= 0:
            raise ValueError("memory.max_size_gb must be positive")

        if self.redis.enabled and not self.redis.host:
            raise ValueError("Redis host must be specified when Redis is enabled")

        if self.gpu.prefetch_queue_size < 1:
            raise ValueError("gpu.prefetch_queue_size must be at least 1")

        if self.prefetch.num_workers < 0:
            raise ValueError("prefetch.num_workers must be non-negative")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "disk": self.disk.__dict__,
            "memory": self.memory.__dict__,
            "redis": self.redis.__dict__,
            "gpu": self.gpu.__dict__,
            "prefetch": self.prefetch.__dict__,
            "agent": self.agent.__dict__,
            "monitoring": self.monitoring.__dict__,
            "auto_tune": self.auto_tune,
            "num_epochs": self.num_epochs,
            "warmup_batches": self.warmup_batches,
            "fallback_to_default": self.fallback_to_default,
            "error_retry_count": self.error_retry_count,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataPipelineConfig":
        """Create config from dictionary"""
        config = cls()

        # Update layer configs
        if "disk" in config_dict:
            for k, v in config_dict["disk"].items():
                setattr(config.disk, k, v)

        if "memory" in config_dict:
            for k, v in config_dict["memory"].items():
                setattr(config.memory, k, v)

        if "redis" in config_dict:
            for k, v in config_dict["redis"].items():
                setattr(config.redis, k, v)

        if "gpu" in config_dict:
            for k, v in config_dict["gpu"].items():
                setattr(config.gpu, k, v)

        if "prefetch" in config_dict:
            for k, v in config_dict["prefetch"].items():
                setattr(config.prefetch, k, v)

        if "agent" in config_dict:
            for k, v in config_dict["agent"].items():
                setattr(config.agent, k, v)

        if "monitoring" in config_dict:
            for k, v in config_dict["monitoring"].items():
                setattr(config.monitoring, k, v)

        # Update global settings
        for key in ["auto_tune", "num_epochs", "warmup_batches",
                    "fallback_to_default", "error_retry_count", "timeout"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config


# Preset configurations for common use cases
def get_default_config() -> DataPipelineConfig:
    """Get default configuration"""
    return DataPipelineConfig()


def get_high_performance_config() -> DataPipelineConfig:
    """Get high-performance configuration for large-scale training"""
    config = DataPipelineConfig()

    # Maximize memory usage
    config.memory.max_size_gb = 32.0
    config.memory.prefetch_size = 64

    # Enable Redis for distributed caching
    config.redis.enabled = True
    config.redis.max_size_gb = 16.0

    # Aggressive prefetching
    config.prefetch.strategy = PrefetchStrategy.ML_PREDICTION
    config.prefetch.prefetch_factor = 4
    config.prefetch.num_workers = 8

    # GPU optimization
    config.gpu.prefetch_queue_size = 4
    config.gpu.num_streams = 4

    return config


def get_memory_constrained_config() -> DataPipelineConfig:
    """Get configuration for memory-constrained environments"""
    config = DataPipelineConfig()

    # Reduce memory usage
    config.memory.max_size_gb = 2.0
    config.memory.prefetch_size = 8

    # Disable Redis
    config.redis.enabled = False

    # Conservative prefetching
    config.prefetch.strategy = PrefetchStrategy.SEQUENTIAL
    config.prefetch.prefetch_factor = 1
    config.prefetch.num_workers = 2

    # Minimal GPU prefetch
    config.gpu.prefetch_queue_size = 1
    config.gpu.num_streams = 1

    return config


def get_distributed_config() -> DataPipelineConfig:
    """Get configuration for distributed training"""
    config = DataPipelineConfig()

    # Enable Redis for cross-node caching
    config.redis.enabled = True
    config.redis.max_size_gb = 8.0
    config.redis.cluster_mode = True

    # Shared memory for local processes
    config.memory.shared_memory = True
    config.memory.max_size_gb = 16.0

    # Pattern-based prefetching works well for distributed
    config.prefetch.strategy = PrefetchStrategy.PATTERN_BASED
    config.prefetch.num_workers = 6

    return config

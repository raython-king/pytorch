"""
GPU Cluster Communication Configuration
GPU集群通讯配置管理

This module provides configuration management for the GPU cluster communication system.

本模块提供GPU集群通讯系统的配置管理。
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import logging

from .types import CompressionStrategy, CollectiveAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class TopologyConfig:
    """
    Configuration for topology discovery and management.
    拓扑发现和管理的配置。
    """
    # Auto-discover topology
    auto_discover: bool = True

    # Topology file path (if not auto-discovering)
    topology_file: Optional[str] = None

    # Refresh interval for topology updates (seconds)
    refresh_interval: float = 60.0

    # Enable NVLink detection
    detect_nvlink: bool = True

    # Enable InfiniBand detection
    detect_infiniband: bool = True


@dataclass
class CollectiveConfig:
    """
    Configuration for collective communication optimization.
    集合通讯优化的配置。
    """
    # Default algorithm (None for auto-selection)
    default_algorithm: Optional[CollectiveAlgorithm] = None

    # Enable adaptive algorithm selection
    adaptive_selection: bool = True

    # Message size thresholds for algorithm selection (bytes)
    small_message_threshold: int = 32 * 1024  # 32 KB
    large_message_threshold: int = 1 * 1024 * 1024  # 1 MB

    # Enable hierarchical algorithms for multi-node
    enable_hierarchical: bool = True

    # Number of chunks for pipelined algorithms
    num_chunks: int = 4

    # Auto-tune chunk count based on message size
    auto_tune_chunks: bool = True


@dataclass
class OverlapConfig:
    """
    Configuration for compute-communication overlap.
    计算-通讯重叠的配置。
    """
    # Enable overlap optimization
    enable_overlap: bool = True

    # Gradient bucketing size (MB)
    bucket_size_mb: float = 25.0

    # Maximum number of buckets
    max_buckets: int = 128

    # Enable async communication
    enable_async: bool = True

    # Pipeline stages for forward/backward passes
    pipeline_stages: int = 1


@dataclass
class CompressionConfig:
    """
    Configuration for communication compression.
    通讯压缩的配置。
    """
    # Enable compression
    enable_compression: bool = False

    # Default compression strategy
    default_strategy: CompressionStrategy = CompressionStrategy.NONE

    # Adaptive compression based on bandwidth
    adaptive_compression: bool = False

    # Compression ratio target (0.0 to 1.0)
    target_ratio: float = 0.5

    # Enable error feedback for compression
    enable_error_feedback: bool = True

    # Minimum message size for compression (bytes)
    min_message_size: int = 1 * 1024 * 1024  # 1 MB


@dataclass
class CoalescingConfig:
    """
    Configuration for message coalescing.
    消息聚合的配置。
    """
    # Enable message coalescing
    enable_coalescing: bool = True

    # Coalescing threshold (KB)
    threshold_kb: float = 64.0

    # Maximum coalesced message size (MB)
    max_coalesced_size_mb: float = 256.0

    # Timeout for waiting for more messages (microseconds)
    timeout_us: float = 100.0


@dataclass
class ProfilingConfig:
    """
    Configuration for performance profiling.
    性能分析的配置。
    """
    # Enable profiling
    enable_profiling: bool = False

    # Profile interval (iterations)
    profile_interval: int = 10

    # Enable Chrome trace export
    enable_trace: bool = False

    # Trace output directory
    trace_output_dir: str = "/tmp/gpu_comm_traces"

    # Enable detailed metrics collection
    detailed_metrics: bool = False

    # Log profiling results
    log_results: bool = True


@dataclass
class LoadBalancingConfig:
    """
    Configuration for load balancing.
    负载均衡的配置。
    """
    # Enable load balancing
    enable_load_balancing: bool = False

    # Straggler detection threshold (efficiency < threshold)
    straggler_threshold: float = 0.8

    # Rebalancing interval (iterations)
    rebalance_interval: int = 100

    # Enable adaptive batch sizing
    adaptive_batch_size: bool = False


@dataclass
class GPUClusterCommConfig:
    """
    Main configuration for GPU cluster communication optimization.
    GPU集群通讯优化的主配置。
    """
    # Sub-configurations
    topology: TopologyConfig = TopologyConfig()
    collective: CollectiveConfig = CollectiveConfig()
    overlap: OverlapConfig = OverlapConfig()
    compression: CompressionConfig = CompressionConfig()
    coalescing: CoalescingConfig = CoalescingConfig()
    profiling: ProfilingConfig = ProfilingConfig()
    load_balancing: LoadBalancingConfig = LoadBalancingConfig()

    # Global settings
    log_level: str = "INFO"
    enable_debug: bool = False

    def __post_init__(self):
        """Initialize sub-configurations if they're dicts"""
        if isinstance(self.topology, dict):
            self.topology = TopologyConfig(**self.topology)
        if isinstance(self.collective, dict):
            self.collective = CollectiveConfig(**self.collective)
        if isinstance(self.overlap, dict):
            self.overlap = OverlapConfig(**self.overlap)
        if isinstance(self.compression, dict):
            self.compression = CompressionConfig(**self.compression)
        if isinstance(self.coalescing, dict):
            self.coalescing = CoalescingConfig(**self.coalescing)
        if isinstance(self.profiling, dict):
            self.profiling = ProfilingConfig(**self.profiling)
        if isinstance(self.load_balancing, dict):
            self.load_balancing = LoadBalancingConfig(**self.load_balancing)

    def validate(self) -> None:
        """
        Validate configuration parameters.
        验证配置参数。

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate topology config
        if self.topology.refresh_interval <= 0:
            raise ValueError("Topology refresh interval must be positive")

        # Validate collective config
        if self.collective.small_message_threshold < 0:
            raise ValueError("Small message threshold cannot be negative")
        if self.collective.large_message_threshold < self.collective.small_message_threshold:
            raise ValueError(
                "Large message threshold must be >= small message threshold"
            )
        if self.collective.num_chunks <= 0:
            raise ValueError("Number of chunks must be positive")

        # Validate overlap config
        if self.overlap.bucket_size_mb <= 0:
            raise ValueError("Bucket size must be positive")
        if self.overlap.max_buckets <= 0:
            raise ValueError("Max buckets must be positive")
        if self.overlap.pipeline_stages <= 0:
            raise ValueError("Pipeline stages must be positive")

        # Validate compression config
        if not 0.0 <= self.compression.target_ratio <= 1.0:
            raise ValueError("Compression target ratio must be in [0, 1]")
        if self.compression.min_message_size < 0:
            raise ValueError("Min message size cannot be negative")

        # Validate coalescing config
        if self.coalescing.threshold_kb < 0:
            raise ValueError("Coalescing threshold cannot be negative")
        if self.coalescing.max_coalesced_size_mb <= 0:
            raise ValueError("Max coalesced size must be positive")
        if self.coalescing.timeout_us < 0:
            raise ValueError("Coalescing timeout cannot be negative")

        # Validate profiling config
        if self.profiling.profile_interval <= 0:
            raise ValueError("Profile interval must be positive")

        # Validate load balancing config
        if not 0.0 <= self.load_balancing.straggler_threshold <= 1.0:
            raise ValueError("Straggler threshold must be in [0, 1]")
        if self.load_balancing.rebalance_interval <= 0:
            raise ValueError("Rebalance interval must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        将配置转换为字典。

        Returns:
            Configuration as dictionary
        """
        def convert_value(val):
            """Convert values to JSON-serializable types"""
            if isinstance(val, (CompressionStrategy, CollectiveAlgorithm)):
                return val.value
            return val

        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                result[key] = {k: convert_value(v) for k, v in value.items()}
            else:
                result[key] = convert_value(value)

        return result

    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        将配置保存到JSON文件。

        Args:
            filepath: Path to output file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'GPUClusterCommConfig':
        """
        Load configuration from JSON file.
        从JSON文件加载配置。

        Args:
            filepath: Path to configuration file

        Returns:
            Loaded configuration
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert string enum values back to enums
        if 'compression' in data and 'default_strategy' in data['compression']:
            strategy_str = data['compression']['default_strategy']
            data['compression']['default_strategy'] = CompressionStrategy(strategy_str)

        if 'collective' in data and 'default_algorithm' in data['collective']:
            if data['collective']['default_algorithm'] is not None:
                algo_str = data['collective']['default_algorithm']
                data['collective']['default_algorithm'] = CollectiveAlgorithm(algo_str)

        config = cls(**data)
        config.validate()
        logger.info(f"Configuration loaded from {filepath}")
        return config

    @classmethod
    def from_env(cls) -> 'GPUClusterCommConfig':
        """
        Create configuration from environment variables.
        从环境变量创建配置。

        Environment variables:
            GPU_COMM_ENABLE_COMPRESSION: Enable compression
            GPU_COMM_COMPRESSION_STRATEGY: Compression strategy
            GPU_COMM_BUCKET_SIZE_MB: Bucket size in MB
            GPU_COMM_ENABLE_PROFILING: Enable profiling
            ... (and more)

        Returns:
            Configuration from environment
        """
        config = cls()

        # Compression settings
        if os.getenv('GPU_COMM_ENABLE_COMPRESSION'):
            config.compression.enable_compression = (
                os.getenv('GPU_COMM_ENABLE_COMPRESSION', 'false').lower() == 'true'
            )

        if os.getenv('GPU_COMM_COMPRESSION_STRATEGY'):
            strategy_str = os.getenv('GPU_COMM_COMPRESSION_STRATEGY', 'none')
            config.compression.default_strategy = CompressionStrategy(strategy_str)

        # Overlap settings
        if os.getenv('GPU_COMM_BUCKET_SIZE_MB'):
            config.overlap.bucket_size_mb = float(os.getenv('GPU_COMM_BUCKET_SIZE_MB', '25.0'))

        if os.getenv('GPU_COMM_ENABLE_OVERLAP'):
            config.overlap.enable_overlap = (
                os.getenv('GPU_COMM_ENABLE_OVERLAP', 'true').lower() == 'true'
            )

        # Profiling settings
        if os.getenv('GPU_COMM_ENABLE_PROFILING'):
            config.profiling.enable_profiling = (
                os.getenv('GPU_COMM_ENABLE_PROFILING', 'false').lower() == 'true'
            )

        if os.getenv('GPU_COMM_PROFILE_INTERVAL'):
            config.profiling.profile_interval = int(os.getenv('GPU_COMM_PROFILE_INTERVAL', '10'))

        # Coalescing settings
        if os.getenv('GPU_COMM_ENABLE_COALESCING'):
            config.coalescing.enable_coalescing = (
                os.getenv('GPU_COMM_ENABLE_COALESCING', 'true').lower() == 'true'
            )

        if os.getenv('GPU_COMM_COALESCING_THRESHOLD_KB'):
            config.coalescing.threshold_kb = float(
                os.getenv('GPU_COMM_COALESCING_THRESHOLD_KB', '64.0')
            )

        # Log level
        if os.getenv('GPU_COMM_LOG_LEVEL'):
            config.log_level = os.getenv('GPU_COMM_LOG_LEVEL', 'INFO')

        if os.getenv('GPU_COMM_DEBUG'):
            config.enable_debug = os.getenv('GPU_COMM_DEBUG', 'false').lower() == 'true'

        config.validate()
        return config


# Global default configuration
_default_config: Optional[GPUClusterCommConfig] = None


def get_config() -> GPUClusterCommConfig:
    """
    Get the global configuration.
    获取全局配置。

    Returns:
        Global configuration instance
    """
    global _default_config
    if _default_config is None:
        _default_config = GPUClusterCommConfig()
    return _default_config


def set_config(config: GPUClusterCommConfig) -> None:
    """
    Set the global configuration.
    设置全局配置。

    Args:
        config: Configuration to set as global
    """
    global _default_config
    config.validate()
    _default_config = config


def reset_config() -> None:
    """
    Reset configuration to default.
    重置配置为默认值。
    """
    global _default_config
    _default_config = GPUClusterCommConfig()

"""
Configuration for Multi-Agent Memory Optimization System
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class OptimizationStrategy(Enum):
    """Available memory optimization strategies"""
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    ACTIVATION_CHECKPOINTING = "activation_checkpointing"
    CPU_OFFLOADING = "cpu_offloading"
    GRADIENT_COMPRESSION = "gradient_compression"
    MIXED_PRECISION = "mixed_precision"
    DYNAMIC_BATCH_SIZE = "dynamic_batch_size"
    MEMORY_EFFICIENT_OPTIMIZER = "memory_efficient_optimizer"
    TENSOR_PARALLELISM = "tensor_parallelism"
    PIPELINE_PARALLELISM = "pipeline_parallelism"
    ZERO_OPTIMIZATION = "zero_optimization"
    ADAPTIVE_RECOMPUTATION = "adaptive_recomputation"
    SMART_CACHING = "smart_caching"


class AgentType(Enum):
    """Types of optimization agents"""
    DIAGNOSTICS_AGENT = "diagnostics"
    CHECKPOINTING_AGENT = "checkpointing"
    OFFLOADING_AGENT = "offloading"
    COMPRESSION_AGENT = "compression"
    PRECISION_AGENT = "precision"
    BATCH_SIZE_AGENT = "batch_size"
    COORDINATOR_AGENT = "coordinator"
    MONITORING_AGENT = "monitoring"


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization system"""

    # Auto-detection and adaptation
    auto_detect_hardware: bool = True
    auto_select_strategies: bool = True
    adaptive_mode: bool = True

    # Strategy preferences (higher values = higher priority)
    strategy_preferences: Dict[OptimizationStrategy, float] = field(default_factory=lambda: {
        OptimizationStrategy.MIXED_PRECISION: 1.0,
        OptimizationStrategy.GRADIENT_CHECKPOINTING: 0.9,
        OptimizationStrategy.ACTIVATION_CHECKPOINTING: 0.85,
        OptimizationStrategy.GRADIENT_COMPRESSION: 0.8,
        OptimizationStrategy.CPU_OFFLOADING: 0.7,
        OptimizationStrategy.DYNAMIC_BATCH_SIZE: 0.75,
        OptimizationStrategy.MEMORY_EFFICIENT_OPTIMIZER: 0.9,
        OptimizationStrategy.ZERO_OPTIMIZATION: 0.85,
        OptimizationStrategy.ADAPTIVE_RECOMPUTATION: 0.8,
        OptimizationStrategy.SMART_CACHING: 0.85,
    })

    # Memory thresholds
    memory_usage_threshold: float = 0.85  # Trigger optimization at 85% usage
    critical_memory_threshold: float = 0.95  # Critical threshold
    target_memory_usage: float = 0.75  # Target after optimization

    # Performance targets
    min_throughput_improvement: float = 0.05  # 5% minimum improvement
    max_latency_overhead: float = 0.10  # 10% max overhead allowed

    # Agent configuration
    num_diagnostic_iterations: int = 10
    strategy_evaluation_steps: int = 100
    adaptation_interval: int = 50  # Steps between adaptations

    # ML model configuration
    use_ml_selection: bool = True
    ml_model_type: str = "ensemble"  # ensemble, gnn, transformer, rl
    model_checkpoint_path: Optional[str] = None
    online_learning: bool = True

    # Hardware constraints
    min_gpu_memory_gb: Optional[float] = None
    min_cpu_memory_gb: Optional[float] = None
    max_cpu_offload_ratio: float = 0.5  # Max 50% to CPU

    # Gradient checkpointing
    checkpoint_ratio: float = 0.5  # Checkpoint 50% of layers
    checkpoint_segments: Optional[int] = None
    selective_checkpointing: bool = True

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16, bfloat16
    amp_opt_level: str = "O2"

    # Gradient compression
    compression_ratio: float = 0.01  # 1% of gradients
    compression_method: str = "topk"  # topk, randomk, threshold

    # CPU offloading
    offload_optimizer_state: bool = True
    offload_gradients: bool = False
    offload_parameters: bool = False
    offload_activations: bool = True

    # Dynamic batch sizing
    initial_batch_size: Optional[int] = None
    min_batch_size: int = 1
    max_batch_size: int = 512
    batch_size_increment: int = 2

    # ZeRO optimization (if using distributed training)
    zero_stage: int = 2  # 0, 1, 2, or 3
    zero_offload: bool = True

    # Monitoring and logging
    enable_monitoring: bool = True
    log_interval: int = 10
    detailed_profiling: bool = False
    export_metrics: bool = True
    metrics_path: str = "./memory_optimization_metrics"

    # Safety mechanisms
    enable_fallback: bool = True
    max_optimization_attempts: int = 5
    rollback_on_failure: bool = True

    # Integration with existing systems
    integrate_with_runtime_scheduler: bool = True
    integrate_with_gpu_cluster_comm: bool = True
    integrate_with_adaptive_flow: bool = True

    # Advanced features
    enable_memory_defragmentation: bool = True
    enable_tensor_lifecycle_tracking: bool = True
    enable_predictive_allocation: bool = True

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 0.0 < self.memory_usage_threshold <= 1.0
        assert 0.0 < self.critical_memory_threshold <= 1.0
        assert 0.0 < self.target_memory_usage <= 1.0
        assert self.target_memory_usage < self.memory_usage_threshold
        assert self.memory_usage_threshold < self.critical_memory_threshold
        assert 0.0 <= self.max_cpu_offload_ratio <= 1.0
        assert 0.0 < self.checkpoint_ratio <= 1.0
        assert 0.0 < self.compression_ratio <= 1.0
        assert self.zero_stage in [0, 1, 2, 3]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MemoryOptimizationConfig':
        """Create config from dictionary"""
        return cls(**config_dict)

    @classmethod
    def for_hardware(cls, gpu_memory_gb: float, num_gpus: int = 1) -> 'MemoryOptimizationConfig':
        """Create config optimized for specific hardware"""
        config = cls()

        # Adjust based on GPU memory
        if gpu_memory_gb < 8:
            # Low memory - aggressive optimization
            config.memory_usage_threshold = 0.75
            config.checkpoint_ratio = 0.7
            config.offload_optimizer_state = True
            config.offload_activations = True
            config.compression_ratio = 0.05
        elif gpu_memory_gb < 16:
            # Medium memory - moderate optimization
            config.memory_usage_threshold = 0.80
            config.checkpoint_ratio = 0.5
            config.offload_optimizer_state = True
        else:
            # High memory - light optimization
            config.memory_usage_threshold = 0.85
            config.checkpoint_ratio = 0.3

        # Adjust for multi-GPU
        if num_gpus > 1:
            config.zero_stage = 2 if num_gpus <= 8 else 3
            config.integrate_with_gpu_cluster_comm = True

        config.validate()
        return config

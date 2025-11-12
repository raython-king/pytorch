"""
Configuration and control for runtime scheduler.

Provides:
- Enable/disable runtime scheduler
- Configure scheduling strategies
- Set performance targets
- Control monitoring level
- Runtime configuration management
"""

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import warnings


class SchedulingMode(Enum):
    """Scheduling mode for runtime scheduler."""
    DISABLED = "disabled"  # Runtime scheduler disabled
    HEURISTIC = "heuristic"  # Use heuristic-based scheduling
    ML = "ml"  # Use machine learning models
    HYBRID = "hybrid"  # Hybrid heuristic + ML


class OptimizationTarget(Enum):
    """Optimization target for scheduling."""
    LATENCY = "latency"  # Minimize latency
    THROUGHPUT = "throughput"  # Maximize throughput
    MEMORY = "memory"  # Minimize memory usage
    POWER = "power"  # Minimize power consumption
    BALANCED = "balanced"  # Balance multiple objectives


class MonitoringLevel(Enum):
    """Level of monitoring detail."""
    NONE = "none"  # No monitoring
    BASIC = "basic"  # Basic metrics only
    DETAILED = "detailed"  # Detailed per-operation metrics
    FULL = "full"  # Full profiling including traces


@dataclass
class SchedulerConfig:
    """
    Configuration for runtime scheduler.

    This class holds all configuration parameters for the runtime scheduler.
    """

    # Mode and target
    mode: SchedulingMode = SchedulingMode.DISABLED
    target: OptimizationTarget = OptimizationTarget.LATENCY

    # Devices
    devices: List[str] = field(default_factory=lambda: ["cuda:0"])
    auto_detect_devices: bool = True

    # Monitoring
    monitoring_enabled: bool = True
    monitoring_level: MonitoringLevel = MonitoringLevel.BASIC
    monitoring_interval: float = 0.1  # seconds

    # Profiling
    profiling_enabled: bool = False
    profile_memory: bool = True
    profile_shapes: bool = True

    # ML model settings
    model_path: Optional[str] = None
    model_update_interval: int = 1000  # operations
    online_learning: bool = False

    # Heuristic settings
    load_balance_weight: float = 0.3
    latency_weight: float = 0.5
    memory_weight: float = 0.2

    # Memory management
    memory_pool_size: int = 1024 * 1024 * 1024  # 1GB default
    enable_memory_pool: bool = True
    memory_defrag_threshold: float = 0.3

    # Stream management
    max_streams_per_device: int = 8
    enable_stream_pool: bool = True

    # Performance tuning
    batch_operations: bool = True
    async_execution: bool = True
    prefetch_enabled: bool = True

    # Debug and validation
    shadow_mode: bool = False  # Make decisions but don't apply
    validation_enabled: bool = False
    log_decisions: bool = False
    log_file: Optional[str] = None

    # Overhead control
    max_scheduling_overhead: float = 0.001  # 0.1% max overhead
    adaptive_overhead_control: bool = True

    # Advanced settings
    custom_strategies: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchedulerConfig':
        """Create config from dictionary."""
        # Convert enum strings to enum values
        if 'mode' in data and isinstance(data['mode'], str):
            data['mode'] = SchedulingMode(data['mode'])
        if 'target' in data and isinstance(data['target'], str):
            data['target'] = OptimizationTarget(data['target'])
        if 'monitoring_level' in data and isinstance(data['monitoring_level'], str):
            data['monitoring_level'] = MonitoringLevel(data['monitoring_level'])

        return cls(**data)

    def save(self, filepath: str) -> None:
        """
        Save configuration to file.

        Args:
            filepath: Path to save configuration
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'SchedulerConfig':
        """
        Load configuration from file.

        Args:
            filepath: Path to load configuration from

        Returns:
            Loaded configuration
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate weights sum to ~1.0
        total_weight = (
            self.load_balance_weight +
            self.latency_weight +
            self.memory_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            errors.append(
                f"Heuristic weights must sum to 1.0, got {total_weight}"
            )

        # Validate device list
        if not self.devices:
            errors.append("At least one device must be specified")

        # Validate paths
        if self.model_path and not os.path.exists(self.model_path):
            errors.append(f"Model path does not exist: {self.model_path}")

        # Validate numerical ranges
        if self.max_scheduling_overhead <= 0:
            errors.append("max_scheduling_overhead must be positive")

        if self.monitoring_interval <= 0:
            errors.append("monitoring_interval must be positive")

        return errors


class RuntimeSchedulerManager:
    """
    Central manager for runtime scheduler.

    Handles:
    - Configuration management
    - Lifecycle (start/stop)
    - Component initialization
    - Global state
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Initialize runtime scheduler manager.

        Args:
            config: Optional configuration (default config used if None)
        """
        self.config = config or SchedulerConfig()

        # Validate config
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")

        # Component references (initialized on start)
        self._monitor = None
        self._profiler = None
        self._hooks = None
        self._scheduler = None

        # State
        self._started = False
        self._stats = {
            "start_time": None,
            "operations_scheduled": 0,
            "errors": 0,
        }

    def start(self) -> None:
        """Start the runtime scheduler."""
        if self._started:
            warnings.warn("Runtime scheduler already started")
            return

        # Initialize components based on config
        self._initialize_components()

        # Start monitoring if enabled
        if self.config.monitoring_enabled and self._monitor:
            self._monitor.start()

        # Enable hooks if not disabled
        if self.config.mode != SchedulingMode.DISABLED and self._hooks:
            from .integration.pytorch_hooks import HookMode
            hook_mode = HookMode.SHADOW if self.config.shadow_mode else HookMode.ENABLED
            self._hooks.set_mode(hook_mode)
            self._hooks.enable()

        self._started = True
        self._stats["start_time"] = self._get_timestamp()

    def stop(self) -> None:
        """Stop the runtime scheduler."""
        if not self._started:
            return

        # Stop monitoring
        if self._monitor:
            self._monitor.stop()

        # Disable hooks
        if self._hooks:
            self._hooks.disable()

        # Stop profiler
        if self._profiler:
            self._profiler.stop()

        self._started = False

    def _initialize_components(self) -> None:
        """Initialize scheduler components."""
        try:
            # Initialize monitor
            if self.config.monitoring_enabled:
                from .monitor import PerformanceMonitor, MetricsCollector

                collector = MetricsCollector(
                    enable_detailed=(
                        self.config.monitoring_level == MonitoringLevel.DETAILED or
                        self.config.monitoring_level == MonitoringLevel.FULL
                    )
                )
                self._monitor = PerformanceMonitor(
                    collector=collector,
                    monitoring_interval=self.config.monitoring_interval
                )

            # Initialize profiler
            if self.config.profiling_enabled:
                from .profiler import RuntimeSchedulerProfiler

                self._profiler = RuntimeSchedulerProfiler(
                    enabled=True,
                    record_shapes=self.config.profile_shapes,
                    profile_memory=self.config.profile_memory
                )

            # Initialize hooks
            from .integration.pytorch_hooks import RuntimeSchedulerHooks

            self._hooks = RuntimeSchedulerHooks()

            # Register callbacks
            if self._monitor and self._hooks:
                self._hooks.register_operation_dispatch_callback(
                    self._operation_dispatch_callback
                )

        except Exception as e:
            warnings.warn(f"Error initializing components: {e}")
            raise

    def _operation_dispatch_callback(self, op_info):
        """Callback for operation dispatch."""
        # This would integrate with the actual scheduler
        # For now, just record metrics
        if self._monitor:
            self._stats["operations_scheduled"] += 1

        return op_info

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime scheduler statistics."""
        stats = dict(self._stats)

        if self._monitor:
            stats["monitor"] = self._monitor.get_summary()

        if self._hooks:
            stats["hooks"] = self._hooks.get_stats()

        if self._profiler:
            stats["profiler"] = self._profiler.get_statistics()

        return stats

    def get_monitor(self):
        """Get performance monitor instance."""
        return self._monitor

    def get_profiler(self):
        """Get profiler instance."""
        return self._profiler

    def get_hooks(self):
        """Get hooks instance."""
        return self._hooks

    def update_config(self, **kwargs) -> None:
        """
        Update configuration dynamically.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                warnings.warn(f"Unknown config parameter: {key}")

        # Validate updated config
        errors = self.config.validate()
        if errors:
            warnings.warn(f"Configuration validation warnings: {errors}")

    @staticmethod
    def _get_timestamp() -> float:
        """Get current timestamp."""
        import time
        return time.time()

    def __enter__(self) -> 'RuntimeSchedulerManager':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# Global manager instance
_global_manager: Optional[RuntimeSchedulerManager] = None


def get_global_manager() -> RuntimeSchedulerManager:
    """Get the global runtime scheduler manager."""
    global _global_manager

    if _global_manager is None:
        _global_manager = RuntimeSchedulerManager()

    return _global_manager


def enable_runtime_scheduler(
    mode: Union[str, SchedulingMode] = 'ml',
    target: Union[str, OptimizationTarget] = 'latency',
    devices: Optional[List[str]] = None,
    monitoring: bool = True,
    **kwargs
) -> RuntimeSchedulerManager:
    """
    Enable runtime scheduler with specified configuration.

    Args:
        mode: Scheduling mode ('disabled', 'heuristic', 'ml', 'hybrid')
        target: Optimization target ('latency', 'throughput', 'memory', 'power', 'balanced')
        devices: List of devices to use (None for auto-detect)
        monitoring: Enable performance monitoring
        **kwargs: Additional configuration parameters

    Returns:
        RuntimeSchedulerManager instance

    Example:
        >>> from torch.runtime_scheduler import enable_runtime_scheduler
        >>>
        >>> scheduler = enable_runtime_scheduler(
        ...     mode='ml',
        ...     target='latency',
        ...     devices=['cuda:0', 'cuda:1'],
        ...     monitoring=True
        ... )
        >>>
        >>> # Your model execution
        >>> model = MyModel().cuda()
        >>> output = model(input_data)
        >>>
        >>> # View statistics
        >>> print(scheduler.get_stats())
    """
    # Convert string arguments to enums
    if isinstance(mode, str):
        mode = SchedulingMode(mode)
    if isinstance(target, str):
        target = OptimizationTarget(target)

    # Create configuration
    config_dict = {
        'mode': mode,
        'target': target,
        'monitoring_enabled': monitoring,
        **kwargs
    }

    if devices is not None:
        config_dict['devices'] = devices

    config = SchedulerConfig.from_dict(config_dict)

    # Create and start manager
    global _global_manager
    _global_manager = RuntimeSchedulerManager(config)
    _global_manager.start()

    return _global_manager


def disable_runtime_scheduler() -> None:
    """Disable the global runtime scheduler."""
    global _global_manager

    if _global_manager is not None:
        _global_manager.stop()
        _global_manager = None


def get_scheduler_config() -> Optional[SchedulerConfig]:
    """Get the current scheduler configuration."""
    manager = get_global_manager()
    return manager.config if manager else None


def update_scheduler_config(**kwargs) -> None:
    """
    Update the current scheduler configuration.

    Args:
        **kwargs: Configuration parameters to update
    """
    manager = get_global_manager()
    if manager:
        manager.update_config(**kwargs)


# Context manager for temporary scheduler configuration
class SchedulerContext:
    """
    Context manager for temporary scheduler configuration.

    Usage:
        with SchedulerContext(mode='ml', target='latency'):
            # Scheduler active with specified config
            model(input)
        # Original state restored
    """

    def __init__(
        self,
        mode: Optional[Union[str, SchedulingMode]] = None,
        target: Optional[Union[str, OptimizationTarget]] = None,
        **kwargs
    ):
        """
        Initialize scheduler context.

        Args:
            mode: Scheduling mode
            target: Optimization target
            **kwargs: Additional configuration
        """
        self.config_updates = {}

        if mode is not None:
            self.config_updates['mode'] = SchedulingMode(mode) if isinstance(mode, str) else mode
        if target is not None:
            self.config_updates['target'] = OptimizationTarget(target) if isinstance(target, str) else target

        self.config_updates.update(kwargs)

        self.manager: Optional[RuntimeSchedulerManager] = None
        self.original_config: Optional[SchedulerConfig] = None

    def __enter__(self) -> RuntimeSchedulerManager:
        # Get or create manager
        self.manager = get_global_manager()

        # Save original config
        self.original_config = SchedulerConfig.from_dict(
            self.manager.config.to_dict()
        )

        # Apply updates
        self.manager.update_config(**self.config_updates)

        # Start if not already started
        if not self.manager._started:
            self.manager.start()

        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore original config
        if self.manager and self.original_config:
            self.manager.config = self.original_config


# Environment variable configuration
def load_config_from_env() -> SchedulerConfig:
    """
    Load configuration from environment variables.

    Environment variables:
    - TORCH_SCHEDULER_MODE: Scheduling mode
    - TORCH_SCHEDULER_TARGET: Optimization target
    - TORCH_SCHEDULER_DEVICES: Comma-separated device list
    - TORCH_SCHEDULER_MONITORING: Enable monitoring (0/1)
    - TORCH_SCHEDULER_CONFIG: Path to config JSON file

    Returns:
        Configuration loaded from environment
    """
    config = SchedulerConfig()

    # Load from JSON file if specified
    config_file = os.environ.get('TORCH_SCHEDULER_CONFIG')
    if config_file and os.path.exists(config_file):
        config = SchedulerConfig.load(config_file)

    # Override with environment variables
    if 'TORCH_SCHEDULER_MODE' in os.environ:
        config.mode = SchedulingMode(os.environ['TORCH_SCHEDULER_MODE'])

    if 'TORCH_SCHEDULER_TARGET' in os.environ:
        config.target = OptimizationTarget(os.environ['TORCH_SCHEDULER_TARGET'])

    if 'TORCH_SCHEDULER_DEVICES' in os.environ:
        config.devices = os.environ['TORCH_SCHEDULER_DEVICES'].split(',')

    if 'TORCH_SCHEDULER_MONITORING' in os.environ:
        config.monitoring_enabled = os.environ['TORCH_SCHEDULER_MONITORING'] == '1'

    return config

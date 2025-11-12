"""
Configuration and Control API for Adaptive Flow Control

This module provides configuration management and control APIs for the
adaptive flow control system.

Features:
- Configuration validation and management
- Runtime configuration updates
- Preset configurations for common scenarios
- Configuration persistence
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Available policy types"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    FAIRNESS = "fairness"
    ENERGY = "energy"
    ADAPTIVE = "adaptive"
    ML_ADAPTIVE = "ml_adaptive"


class CongestionControlAlgorithm(Enum):
    """Available congestion control algorithms"""
    BBR = "bbr"
    VEGAS = "vegas"
    DCTCP = "dctcp"
    TIMELY = "timely"
    CUBIC = "cubic"


class TargetObjective(Enum):
    """Optimization targets"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    FAIRNESS = "fairness"
    BALANCED = "balanced"
    ENERGY = "energy"


class MonitoringLevel(Enum):
    """Monitoring detail levels"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


@dataclass
class AdaptiveFlowConfig:
    """
    Configuration for Adaptive Flow Control

    This class encapsulates all configuration options for the adaptive flow
    control system.

    Attributes:
        enabled: Enable/disable adaptive flow control
        policy: Policy to use for flow control
        target: Primary optimization target
        congestion_control: Congestion control algorithm
        monitoring_level: Level of monitoring detail
        shadow_mode: Run in shadow mode (observe but don't control)
        max_flows: Maximum number of concurrent flows to track
        metrics_history_size: Number of metric samples to keep
        update_interval: How often to update metrics (seconds)
        enable_visualization: Enable real-time visualization
        log_level: Logging level
        custom_params: Custom parameters for advanced configuration
    """

    # Core settings
    enabled: bool = True
    policy: str = "adaptive"
    target: str = "balanced"
    congestion_control: str = "bbr"

    # Monitoring settings
    monitoring_level: str = "standard"
    shadow_mode: bool = False
    max_flows: int = 1000
    metrics_history_size: int = 1000
    update_interval: float = 1.0

    # Visualization settings
    enable_visualization: bool = False
    visualization_port: int = 8080

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Advanced settings
    enable_bottleneck_detection: bool = True
    enable_fairness_enforcement: bool = True
    enable_adaptive_routing: bool = True
    enable_priority_queuing: bool = False

    # Policy-specific parameters
    latency_target_ms: float = 1.0
    throughput_target_gbps: Optional[float] = None
    fairness_threshold: float = 0.7
    energy_power_cap_w: Optional[float] = None

    # Congestion control parameters
    cc_initial_cwnd: int = 10
    cc_mss: int = 1500
    cc_min_rtt_us: float = 20.0

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters"""
        # Validate policy
        try:
            PolicyType(self.policy)
        except ValueError:
            valid_policies = [p.value for p in PolicyType]
            raise ValueError(f"Invalid policy '{self.policy}'. Must be one of: {valid_policies}")

        # Validate target
        try:
            TargetObjective(self.target)
        except ValueError:
            valid_targets = [t.value for t in TargetObjective]
            raise ValueError(f"Invalid target '{self.target}'. Must be one of: {valid_targets}")

        # Validate congestion control
        try:
            CongestionControlAlgorithm(self.congestion_control)
        except ValueError:
            valid_ccs = [c.value for c in CongestionControlAlgorithm]
            raise ValueError(f"Invalid congestion control '{self.congestion_control}'. Must be one of: {valid_ccs}")

        # Validate monitoring level
        try:
            MonitoringLevel(self.monitoring_level)
        except ValueError:
            valid_levels = [m.value for m in MonitoringLevel]
            raise ValueError(f"Invalid monitoring level '{self.monitoring_level}'. Must be one of: {valid_levels}")

        # Validate numeric parameters
        if self.max_flows <= 0:
            raise ValueError(f"max_flows must be positive, got {self.max_flows}")

        if self.metrics_history_size <= 0:
            raise ValueError(f"metrics_history_size must be positive, got {self.metrics_history_size}")

        if self.update_interval <= 0:
            raise ValueError(f"update_interval must be positive, got {self.update_interval}")

        if self.latency_target_ms <= 0:
            raise ValueError(f"latency_target_ms must be positive, got {self.latency_target_ms}")

        if self.fairness_threshold < 0 or self.fairness_threshold > 1:
            raise ValueError(f"fairness_threshold must be in [0, 1], got {self.fairness_threshold}")

        if self.cc_initial_cwnd <= 0:
            raise ValueError(f"cc_initial_cwnd must be positive, got {self.cc_initial_cwnd}")

        if self.cc_mss <= 0:
            raise ValueError(f"cc_mss must be positive, got {self.cc_mss}")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to file

        Args:
            path: Path to save configuration file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            f.write(self.to_json())

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AdaptiveFlowConfig':
        """
        Load configuration from file

        Args:
            path: Path to configuration file

        Returns:
            AdaptiveFlowConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        logger.info(f"Configuration loaded from {path}")
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> 'AdaptiveFlowConfig':
        """
        Create configuration from dictionary

        Args:
            data: Configuration dictionary

        Returns:
            AdaptiveFlowConfig instance
        """
        return cls(**data)

    def update(self, **kwargs) -> None:
        """
        Update configuration parameters

        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        # Re-validate after update
        self._validate()

        logger.info(f"Configuration updated: {kwargs}")


class ConfigPresets:
    """Pre-defined configuration presets for common scenarios"""

    @staticmethod
    def low_latency() -> AdaptiveFlowConfig:
        """
        Configuration optimized for low latency

        Best for:
        - Real-time inference
        - Interactive applications
        - Latency-sensitive workloads
        """
        return AdaptiveFlowConfig(
            enabled=True,
            policy="latency",
            target="latency",
            congestion_control="bbr",
            monitoring_level="detailed",
            latency_target_ms=0.5,
            enable_priority_queuing=True,
            enable_adaptive_routing=True,
        )

    @staticmethod
    def high_throughput() -> AdaptiveFlowConfig:
        """
        Configuration optimized for maximum throughput

        Best for:
        - Batch training
        - Large data transfers
        - Throughput-critical applications
        """
        return AdaptiveFlowConfig(
            enabled=True,
            policy="throughput",
            target="throughput",
            congestion_control="bbr",
            monitoring_level="standard",
            enable_bottleneck_detection=True,
            enable_adaptive_routing=True,
        )

    @staticmethod
    def fair_sharing() -> AdaptiveFlowConfig:
        """
        Configuration optimized for fairness

        Best for:
        - Multi-tenant environments
        - Shared resources
        - Fair resource allocation
        """
        return AdaptiveFlowConfig(
            enabled=True,
            policy="fairness",
            target="fairness",
            congestion_control="dctcp",
            monitoring_level="detailed",
            fairness_threshold=0.8,
            enable_fairness_enforcement=True,
        )

    @staticmethod
    def energy_efficient() -> AdaptiveFlowConfig:
        """
        Configuration optimized for energy efficiency

        Best for:
        - Power-constrained environments
        - Green computing
        - Cost optimization
        """
        return AdaptiveFlowConfig(
            enabled=True,
            policy="energy",
            target="energy",
            congestion_control="vegas",
            monitoring_level="standard",
            enable_adaptive_routing=False,
        )

    @staticmethod
    def balanced() -> AdaptiveFlowConfig:
        """
        Balanced configuration for general use

        Best for:
        - General workloads
        - Mixed applications
        - Default configuration
        """
        return AdaptiveFlowConfig(
            enabled=True,
            policy="adaptive",
            target="balanced",
            congestion_control="bbr",
            monitoring_level="standard",
            enable_bottleneck_detection=True,
            enable_fairness_enforcement=True,
            enable_adaptive_routing=True,
        )

    @staticmethod
    def distributed_training() -> AdaptiveFlowConfig:
        """
        Configuration optimized for distributed training

        Best for:
        - Multi-GPU training
        - Multi-node training
        - Collective communications
        """
        return AdaptiveFlowConfig(
            enabled=True,
            policy="adaptive",
            target="balanced",
            congestion_control="dctcp",
            monitoring_level="detailed",
            enable_bottleneck_detection=True,
            enable_fairness_enforcement=True,
            enable_adaptive_routing=True,
            enable_priority_queuing=True,
        )

    @staticmethod
    def debug() -> AdaptiveFlowConfig:
        """
        Configuration for debugging and development

        Best for:
        - Development
        - Debugging
        - Performance analysis
        """
        return AdaptiveFlowConfig(
            enabled=True,
            policy="adaptive",
            target="balanced",
            congestion_control="bbr",
            monitoring_level="debug",
            shadow_mode=True,  # Don't affect behavior
            enable_visualization=True,
            log_level="DEBUG",
        )

    @staticmethod
    def get_preset(name: str) -> AdaptiveFlowConfig:
        """
        Get configuration preset by name

        Args:
            name: Preset name

        Returns:
            AdaptiveFlowConfig for the preset

        Raises:
            ValueError: If preset name is invalid
        """
        presets = {
            'low_latency': ConfigPresets.low_latency,
            'high_throughput': ConfigPresets.high_throughput,
            'fair_sharing': ConfigPresets.fair_sharing,
            'energy_efficient': ConfigPresets.energy_efficient,
            'balanced': ConfigPresets.balanced,
            'distributed_training': ConfigPresets.distributed_training,
            'debug': ConfigPresets.debug,
        }

        if name not in presets:
            available = list(presets.keys())
            raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")

        return presets[name]()


class ConfigManager:
    """
    Global configuration manager

    Manages the active configuration and handles runtime updates.
    """

    _instance: Optional['ConfigManager'] = None
    _config: Optional[AdaptiveFlowConfig] = None

    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_config(self) -> AdaptiveFlowConfig:
        """Get current configuration"""
        if self._config is None:
            # Return default configuration
            self._config = ConfigPresets.balanced()
        return self._config

    def set_config(self, config: AdaptiveFlowConfig) -> None:
        """
        Set configuration

        Args:
            config: New configuration
        """
        config._validate()
        self._config = config
        logger.info("Configuration updated")

    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters

        Args:
            **kwargs: Parameters to update
        """
        config = self.get_config()
        config.update(**kwargs)

    def reset_config(self) -> None:
        """Reset to default configuration"""
        self._config = ConfigPresets.balanced()
        logger.info("Configuration reset to default")


# Convenience functions

def get_config() -> AdaptiveFlowConfig:
    """
    Get current global configuration

    Returns:
        Current AdaptiveFlowConfig

    Example:
        >>> from torch.adaptive_flow import get_config
        >>> config = get_config()
        >>> print(f"Policy: {config.policy}")
    """
    manager = ConfigManager.get_instance()
    return manager.get_config()


def set_config(config: AdaptiveFlowConfig) -> None:
    """
    Set global configuration

    Args:
        config: New configuration

    Example:
        >>> from torch.adaptive_flow import set_config, ConfigPresets
        >>> config = ConfigPresets.low_latency()
        >>> set_config(config)
    """
    manager = ConfigManager.get_instance()
    manager.set_config(config)


def update_config(**kwargs) -> None:
    """
    Update global configuration parameters

    Args:
        **kwargs: Parameters to update

    Example:
        >>> from torch.adaptive_flow import update_config
        >>> update_config(policy='latency', latency_target_ms=0.5)
    """
    manager = ConfigManager.get_instance()
    manager.update_config(**kwargs)


def reset_config() -> None:
    """
    Reset configuration to default

    Example:
        >>> from torch.adaptive_flow import reset_config
        >>> reset_config()
    """
    manager = ConfigManager.get_instance()
    manager.reset_config()


def load_config(path: Union[str, Path]) -> AdaptiveFlowConfig:
    """
    Load configuration from file

    Args:
        path: Path to configuration file

    Returns:
        Loaded AdaptiveFlowConfig

    Example:
        >>> from torch.adaptive_flow import load_config
        >>> config = load_config('my_config.json')
    """
    return AdaptiveFlowConfig.load(path)


def save_config(path: Union[str, Path], config: Optional[AdaptiveFlowConfig] = None) -> None:
    """
    Save configuration to file

    Args:
        path: Path to save configuration file
        config: Configuration to save (default: current global config)

    Example:
        >>> from torch.adaptive_flow import save_config, get_config
        >>> save_config('my_config.json')  # Save current config
    """
    if config is None:
        config = get_config()
    config.save(path)


__all__ = [
    'AdaptiveFlowConfig',
    'ConfigPresets',
    'ConfigManager',
    'PolicyType',
    'CongestionControlAlgorithm',
    'TargetObjective',
    'MonitoringLevel',
    'get_config',
    'set_config',
    'update_config',
    'reset_config',
    'load_config',
    'save_config',
]

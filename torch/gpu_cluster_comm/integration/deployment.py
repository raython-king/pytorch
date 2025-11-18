"""
Deployment Tools for GPU Cluster Communication Optimization

This module provides tools for deploying and configuring the GPU cluster
communication optimization in production environments.

Key features:
- Configuration management
- Environment detection
- Cluster setup and validation
- Performance tuning recommendations
"""

import json
import logging
import os
import socket
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for a GPU cluster."""

    # Basic cluster info
    num_nodes: int
    num_gpus_per_node: int
    master_addr: str
    master_port: int

    # Network configuration
    backend: str = "nccl"
    init_method: Optional[str] = None

    # Optimization settings
    enable_optimization: bool = True
    optimization_mode: str = "shadow"  # disabled, shadow, enabled

    # Performance tuning
    bucket_size_mb: int = 25
    enable_overlap: bool = True
    enable_compression: bool = False

    # Advanced settings
    use_nvlink: bool = True
    use_hierarchical: bool = True
    ml_algorithm_selection: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if self.num_gpus_per_node <= 0:
            raise ValueError("num_gpus_per_node must be positive")
        if self.optimization_mode not in ["disabled", "shadow", "enabled"]:
            raise ValueError("Invalid optimization_mode")

    @property
    def world_size(self) -> int:
        """Total number of GPUs in cluster."""
        return self.num_nodes * self.num_gpus_per_node

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterConfig':
        """Create from dictionary."""
        return cls(**data)

    def save(self, path: str):
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ClusterConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"Configuration loaded from {path}")
        return cls.from_dict(data)


class EnvironmentDetector:
    """Detect cluster environment and configuration."""

    @staticmethod
    def detect_slurm() -> Optional[Dict[str, Any]]:
        """Detect SLURM environment."""
        if 'SLURM_JOB_ID' not in os.environ:
            return None

        try:
            config = {
                'num_nodes': int(os.environ.get('SLURM_JOB_NUM_NODES', 1)),
                'node_id': int(os.environ.get('SLURM_NODEID', 0)),
                'num_tasks': int(os.environ.get('SLURM_NTASKS', 1)),
                'local_id': int(os.environ.get('SLURM_LOCALID', 0)),
                'job_id': os.environ['SLURM_JOB_ID'],
                'nodelist': os.environ.get('SLURM_JOB_NODELIST', ''),
            }
            logger.info(f"Detected SLURM environment: {config}")
            return config
        except Exception as e:
            logger.warning(f"Failed to parse SLURM environment: {e}")
            return None

    @staticmethod
    def detect_pbs() -> Optional[Dict[str, Any]]:
        """Detect PBS/Torque environment."""
        if 'PBS_JOBID' not in os.environ:
            return None

        try:
            config = {
                'job_id': os.environ['PBS_JOBID'],
                'nodefile': os.environ.get('PBS_NODEFILE', ''),
            }
            logger.info(f"Detected PBS environment: {config}")
            return config
        except Exception as e:
            logger.warning(f"Failed to parse PBS environment: {e}")
            return None

    @staticmethod
    def detect_kubernetes() -> Optional[Dict[str, Any]]:
        """Detect Kubernetes environment."""
        if 'KUBERNETES_SERVICE_HOST' not in os.environ:
            return None

        try:
            config = {
                'pod_name': os.environ.get('HOSTNAME', ''),
                'namespace': os.environ.get('POD_NAMESPACE', 'default'),
                'service_host': os.environ['KUBERNETES_SERVICE_HOST'],
            }
            logger.info(f"Detected Kubernetes environment: {config}")
            return config
        except Exception as e:
            logger.warning(f"Failed to parse Kubernetes environment: {e}")
            return None

    @staticmethod
    def detect_mpi() -> Optional[Dict[str, Any]]:
        """Detect MPI environment."""
        for var in ['OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK', 'PMI_RANK']:
            if var in os.environ:
                try:
                    config = {
                        'rank': int(os.environ.get(var, 0)),
                        'world_size': int(os.environ.get(
                            var.replace('RANK', 'SIZE'), 1
                        )),
                    }
                    logger.info(f"Detected MPI environment: {config}")
                    return config
                except:
                    pass
        return None

    @classmethod
    def detect_environment(cls) -> Dict[str, Any]:
        """Detect cluster environment type and configuration."""
        env_info = {
            'type': 'unknown',
            'hostname': socket.gethostname(),
            'cuda_available': torch.cuda.is_available(),
            'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        # Try to detect environment type
        if slurm_config := cls.detect_slurm():
            env_info['type'] = 'slurm'
            env_info.update(slurm_config)
        elif pbs_config := cls.detect_pbs():
            env_info['type'] = 'pbs'
            env_info.update(pbs_config)
        elif k8s_config := cls.detect_kubernetes():
            env_info['type'] = 'kubernetes'
            env_info.update(k8s_config)
        elif mpi_config := cls.detect_mpi():
            env_info['type'] = 'mpi'
            env_info.update(mpi_config)

        return env_info


class ClusterSetup:
    """Setup and validate cluster for distributed training."""

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.env_info = EnvironmentDetector.detect_environment()

    def initialize_process_group(self, rank: int, world_size: Optional[int] = None):
        """Initialize torch.distributed process group.

        Args:
            rank: Rank of this process
            world_size: Total number of processes (defaults to config.world_size)
        """
        if world_size is None:
            world_size = self.config.world_size

        if dist.is_initialized():
            logger.warning("Process group already initialized")
            return

        # Set environment variables
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = str(self.config.master_port)

        # Initialize
        logger.info(
            f"Initializing process group: "
            f"backend={self.config.backend}, "
            f"rank={rank}, "
            f"world_size={world_size}"
        )

        if self.config.init_method:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                rank=rank,
                world_size=world_size
            )
        else:
            dist.init_process_group(
                backend=self.config.backend,
                rank=rank,
                world_size=world_size
            )

        logger.info("Process group initialized successfully")

    def validate_cluster(self) -> Tuple[bool, List[str]]:
        """Validate cluster configuration and connectivity.

        Returns:
            Tuple of (success, list of issues)
        """
        issues = []

        # Check CUDA availability
        if not torch.cuda.is_available():
            issues.append("CUDA not available")

        # Check GPU count
        num_gpus = torch.cuda.device_count()
        if num_gpus < self.config.num_gpus_per_node:
            issues.append(
                f"Expected {self.config.num_gpus_per_node} GPUs, "
                f"found {num_gpus}"
            )

        # Check NCCL availability
        if self.config.backend == "nccl":
            if not hasattr(torch.cuda, 'nccl'):
                issues.append("NCCL backend not available")

        # Check distributed availability
        if not dist.is_available():
            issues.append("torch.distributed not available")

        # Check network connectivity (if process group is initialized)
        if dist.is_initialized():
            try:
                # Simple connectivity test
                test_tensor = torch.ones(1).cuda()
                dist.all_reduce(test_tensor)
                logger.info("Network connectivity test passed")
            except Exception as e:
                issues.append(f"Network connectivity test failed: {e}")

        success = len(issues) == 0
        if success:
            logger.info("Cluster validation passed")
        else:
            logger.error(f"Cluster validation failed: {issues}")

        return success, issues

    def apply_optimizations(self):
        """Apply GPU cluster communication optimizations based on configuration."""
        if not self.config.enable_optimization:
            logger.info("Optimization disabled by configuration")
            return

        # Import integration module
        try:
            from .pytorch_integration import (
                TransparentOptimization,
                IntegrationMode
            )

            # Map configuration mode to IntegrationMode
            mode_map = {
                'disabled': IntegrationMode.DISABLED,
                'shadow': IntegrationMode.SHADOW,
                'enabled': IntegrationMode.ENABLED,
            }
            mode = mode_map[self.config.optimization_mode]

            # Enable optimization
            TransparentOptimization.enable_auto_optimization(mode=mode)
            logger.info(f"Optimization enabled with mode: {self.config.optimization_mode}")

        except ImportError as e:
            logger.error(f"Failed to import optimization module: {e}")
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")

    def get_recommendations(self) -> List[str]:
        """Get performance tuning recommendations based on cluster configuration.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Bucket size recommendation
        if self.config.world_size > 16:
            if self.config.bucket_size_mb < 50:
                recommendations.append(
                    f"Consider increasing bucket_size_mb to 50-100 for "
                    f"large clusters ({self.config.world_size} GPUs)"
                )

        # NCCL backend recommendation
        if self.config.backend != "nccl" and torch.cuda.is_available():
            recommendations.append(
                "Consider using NCCL backend for better GPU communication performance"
            )

        # NVLink recommendation
        if self.config.num_gpus_per_node > 1:
            recommendations.append(
                "Ensure NVLink is enabled for optimal intra-node communication"
            )

        # Hierarchical communication
        if self.config.world_size > 8 and not self.config.use_hierarchical:
            recommendations.append(
                "Consider enabling hierarchical communication for large clusters"
            )

        # Overlap recommendation
        if not self.config.enable_overlap:
            recommendations.append(
                "Enable overlap for better computation-communication overlap"
            )

        return recommendations

    def print_setup_summary(self):
        """Print cluster setup summary."""
        print("\n" + "=" * 70)
        print("GPU Cluster Setup Summary")
        print("=" * 70)

        print("\nCluster Configuration:")
        print(f"  Nodes: {self.config.num_nodes}")
        print(f"  GPUs per node: {self.config.num_gpus_per_node}")
        print(f"  Total GPUs: {self.config.world_size}")
        print(f"  Backend: {self.config.backend}")

        print("\nOptimization Settings:")
        print(f"  Enabled: {self.config.enable_optimization}")
        print(f"  Mode: {self.config.optimization_mode}")
        print(f"  Bucket size: {self.config.bucket_size_mb} MB")
        print(f"  Enable overlap: {self.config.enable_overlap}")

        print("\nEnvironment:")
        print(f"  Type: {self.env_info['type']}")
        print(f"  Hostname: {self.env_info['hostname']}")
        print(f"  CUDA available: {self.env_info['cuda_available']}")
        print(f"  GPUs detected: {self.env_info['num_gpus']}")

        recommendations = self.get_recommendations()
        if recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print("=" * 70 + "\n")


# Convenience functions
def create_config_template(output_path: str = "cluster_config.json"):
    """Create a configuration file template.

    Args:
        output_path: Path to save the template
    """
    template = ClusterConfig(
        num_nodes=1,
        num_gpus_per_node=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        master_addr="localhost",
        master_port=29500,
    )
    template.save(output_path)
    print(f"Configuration template created: {output_path}")


def quick_setup(rank: int, world_size: int,
               master_addr: str = "localhost",
               master_port: int = 29500,
               enable_optimization: bool = True,
               mode: str = "shadow") -> ClusterSetup:
    """Quick setup for simple distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
        enable_optimization: Whether to enable optimization
        mode: Optimization mode (disabled/shadow/enabled)

    Returns:
        ClusterSetup instance
    """
    # Detect number of GPUs per node
    num_gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_nodes = (world_size + num_gpus_per_node - 1) // num_gpus_per_node

    config = ClusterConfig(
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        master_addr=master_addr,
        master_port=master_port,
        enable_optimization=enable_optimization,
        optimization_mode=mode,
    )

    setup = ClusterSetup(config)
    setup.initialize_process_group(rank, world_size)

    if enable_optimization:
        setup.apply_optimizations()

    return setup

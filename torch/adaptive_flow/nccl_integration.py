"""
NCCL Integration for Adaptive Flow Control.

Integrates adaptive flow control with NCCL collective operations,
providing dynamic algorithm selection, tuning, and optimization.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import torch
import torch.distributed as dist

from .topology_manager import NetworkTopology
from .routing_engine import RoutingEngine

logger = logging.getLogger(__name__)


class NCCLAlgorithm(Enum):
    """NCCL collective algorithms."""
    RING = "ring"
    TREE = "tree"
    COLLNET = "collnet"
    NVLS = "nvls"  # NVLink-SHARP


class CollectiveType(Enum):
    """Types of collective operations."""
    ALL_REDUCE = "all_reduce"
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    SEND = "send"
    RECV = "recv"


@dataclass
class NCCLConfig:
    """
    NCCL configuration parameters.

    Attributes:
        algorithm: Collective algorithm
        num_channels: Number of NCCL channels
        min_chunk_size: Minimum chunk size (bytes)
        max_chunk_size: Maximum chunk size (bytes)
        buffer_size: Buffer size (bytes)
        enable_pipelining: Enable pipelining
        enable_direct_nvlink: Enable direct NVLink transfers
        enable_gdrcopy: Enable GPU Direct RDMA
    """
    algorithm: NCCLAlgorithm = NCCLAlgorithm.RING
    num_channels: int = 2
    min_chunk_size: int = 128 * 1024  # 128 KB
    max_chunk_size: int = 4 * 1024 * 1024  # 4 MB
    buffer_size: int = 4 * 1024 * 1024  # 4 MB
    enable_pipelining: bool = True
    enable_direct_nvlink: bool = True
    enable_gdrcopy: bool = True


@dataclass
class CollectiveProfile:
    """
    Performance profile for a collective operation.

    Attributes:
        collective_type: Type of collective
        num_devices: Number of participating devices
        data_size_bytes: Size of data
        algorithm: Algorithm used
        duration_ms: Duration in milliseconds
        bandwidth_gbps: Achieved bandwidth (Gbps)
        bus_bandwidth_gbps: Bus bandwidth (Gbps)
        timestamp: Profile timestamp
    """
    collective_type: CollectiveType
    num_devices: int
    data_size_bytes: int
    algorithm: NCCLAlgorithm
    duration_ms: float
    bandwidth_gbps: float
    bus_bandwidth_gbps: float
    timestamp: float


class NCCLAlgorithmSelector:
    """
    Select optimal NCCL algorithm based on topology and workload.

    Uses heuristics and learned models to choose between ring, tree,
    and other algorithms.
    """

    def __init__(self, topology: NetworkTopology):
        """
        Initialize algorithm selector.

        Args:
            topology: Network topology
        """
        self.topology = topology
        self._performance_history: Dict[Tuple, List[CollectiveProfile]] = {}
        logger.info("NCCLAlgorithmSelector initialized")

    def select_algorithm(
        self,
        collective_type: CollectiveType,
        num_devices: int,
        data_size_bytes: int,
        devices: Optional[List[int]] = None
    ) -> NCCLAlgorithm:
        """
        Select optimal algorithm for collective.

        Args:
            collective_type: Type of collective operation
            num_devices: Number of participating devices
            data_size_bytes: Size of data
            devices: Device IDs (if known)

        Returns:
            Recommended NCCL algorithm
        """
        # Check performance history
        key = (collective_type, num_devices)
        if key in self._performance_history:
            best_algo = self._select_from_history(key, data_size_bytes)
            if best_algo:
                return best_algo

        # Use heuristics
        return self._heuristic_selection(
            collective_type, num_devices, data_size_bytes, devices
        )

    def _heuristic_selection(
        self,
        collective_type: CollectiveType,
        num_devices: int,
        data_size_bytes: int,
        devices: Optional[List[int]]
    ) -> NCCLAlgorithm:
        """Select algorithm using heuristics."""
        # Ring is generally good for large data and many devices
        if num_devices >= 8 and data_size_bytes > 1024 * 1024:
            return NCCLAlgorithm.RING

        # Tree is better for small messages and broadcasts
        if collective_type == CollectiveType.BROADCAST and data_size_bytes < 256 * 1024:
            return NCCLAlgorithm.TREE

        # Check for NVLink-SHARP support
        if devices and self._has_nvlink_mesh(devices):
            return NCCLAlgorithm.NVLS

        # Default to ring
        return NCCLAlgorithm.RING

    def _has_nvlink_mesh(self, devices: List[int]) -> bool:
        """Check if devices form NVLink mesh."""
        # Check if all devices are connected with NVLink
        for i, dev1 in enumerate(devices):
            for dev2 in devices[i + 1:]:
                link = self.topology.get_link(dev1, dev2)
                if not link or link.link_type.value != "nvlink":
                    return False
        return True

    def _select_from_history(
        self,
        key: Tuple,
        data_size_bytes: int
    ) -> Optional[NCCLAlgorithm]:
        """Select algorithm based on performance history."""
        profiles = self._performance_history.get(key, [])
        if not profiles:
            return None

        # Find profiles with similar data size
        similar = [
            p for p in profiles
            if 0.5 * data_size_bytes <= p.data_size_bytes <= 2.0 * data_size_bytes
        ]

        if not similar:
            return None

        # Return algorithm with best bandwidth
        best = max(similar, key=lambda p: p.bandwidth_gbps)
        return best.algorithm

    def record_performance(self, profile: CollectiveProfile) -> None:
        """Record performance profile for future decisions."""
        key = (profile.collective_type, profile.num_devices)
        if key not in self._performance_history:
            self._performance_history[key] = []

        self._performance_history[key].append(profile)

        # Keep only recent profiles
        if len(self._performance_history[key]) > 100:
            self._performance_history[key] = self._performance_history[key][-100:]


class NCCLTuner:
    """
    Dynamically tune NCCL parameters for optimal performance.

    Adjusts chunk sizes, channel counts, and other parameters based on
    observed performance.
    """

    def __init__(self, topology: NetworkTopology):
        """
        Initialize NCCL tuner.

        Args:
            topology: Network topology
        """
        self.topology = topology
        self._base_config = NCCLConfig()
        self._tuned_configs: Dict[Tuple, NCCLConfig] = {}
        logger.info("NCCLTuner initialized")

    def get_config(
        self,
        collective_type: CollectiveType,
        num_devices: int,
        data_size_bytes: int
    ) -> NCCLConfig:
        """
        Get tuned configuration for collective.

        Args:
            collective_type: Type of collective
            num_devices: Number of devices
            data_size_bytes: Data size

        Returns:
            Tuned NCCL configuration
        """
        key = (collective_type, num_devices)

        if key in self._tuned_configs:
            return self._tuned_configs[key]

        # Generate config
        config = self._generate_config(collective_type, num_devices, data_size_bytes)
        self._tuned_configs[key] = config

        return config

    def _generate_config(
        self,
        collective_type: CollectiveType,
        num_devices: int,
        data_size_bytes: int
    ) -> NCCLConfig:
        """Generate configuration based on parameters."""
        config = NCCLConfig()

        # Tune number of channels
        if num_devices >= 8:
            config.num_channels = 4
        elif num_devices >= 4:
            config.num_channels = 2
        else:
            config.num_channels = 1

        # Tune chunk size based on data size
        if data_size_bytes < 1024 * 1024:  # < 1 MB
            config.min_chunk_size = 32 * 1024  # 32 KB
            config.max_chunk_size = 512 * 1024  # 512 KB
        elif data_size_bytes < 100 * 1024 * 1024:  # < 100 MB
            config.min_chunk_size = 128 * 1024  # 128 KB
            config.max_chunk_size = 2 * 1024 * 1024  # 2 MB
        else:  # >= 100 MB
            config.min_chunk_size = 256 * 1024  # 256 KB
            config.max_chunk_size = 4 * 1024 * 1024  # 4 MB

        # Enable pipelining for large transfers
        config.enable_pipelining = data_size_bytes > 10 * 1024 * 1024

        return config

    def update_config(
        self,
        collective_type: CollectiveType,
        num_devices: int,
        profile: CollectiveProfile
    ) -> None:
        """
        Update configuration based on observed performance.

        Args:
            collective_type: Type of collective
            num_devices: Number of devices
            profile: Performance profile
        """
        key = (collective_type, num_devices)

        # Simple adaptive tuning: if performance is poor, adjust parameters
        if key in self._tuned_configs:
            config = self._tuned_configs[key]

            # If bandwidth is low, try increasing chunk size
            if profile.bandwidth_gbps < 10.0:  # Arbitrary threshold
                config.max_chunk_size = min(config.max_chunk_size * 2, 8 * 1024 * 1024)
                logger.info(f"Increased chunk size for {key}: {config.max_chunk_size}")


class NCCLProfiler:
    """
    Profile NCCL collective operations.

    Measures performance metrics and identifies bottlenecks.
    """

    def __init__(self):
        """Initialize NCCL profiler."""
        self._profiles: List[CollectiveProfile] = []
        self._active_ops: Dict[int, Tuple[float, Dict]] = {}  # op_id -> (start_time, metadata)
        self._op_id_counter = 0
        self._lock = threading.RLock()
        logger.info("NCCLProfiler initialized")

    def start_op(
        self,
        collective_type: CollectiveType,
        num_devices: int,
        data_size_bytes: int,
        algorithm: NCCLAlgorithm
    ) -> int:
        """
        Start profiling a collective operation.

        Args:
            collective_type: Type of collective
            num_devices: Number of devices
            data_size_bytes: Data size
            algorithm: Algorithm used

        Returns:
            Operation ID for tracking
        """
        with self._lock:
            op_id = self._op_id_counter
            self._op_id_counter += 1

            self._active_ops[op_id] = (
                time.time(),
                {
                    "collective_type": collective_type,
                    "num_devices": num_devices,
                    "data_size_bytes": data_size_bytes,
                    "algorithm": algorithm
                }
            )

            return op_id

    def end_op(self, op_id: int) -> Optional[CollectiveProfile]:
        """
        End profiling and record performance.

        Args:
            op_id: Operation ID

        Returns:
            Performance profile
        """
        with self._lock:
            if op_id not in self._active_ops:
                logger.warning(f"Unknown operation ID: {op_id}")
                return None

            start_time, metadata = self._active_ops[op_id]
            del self._active_ops[op_id]

            duration_s = time.time() - start_time
            duration_ms = duration_s * 1000.0

            # Calculate bandwidth
            data_size_bytes = metadata["data_size_bytes"]
            bandwidth_gbps = (data_size_bytes * 8) / (duration_s * 1e9)

            # Calculate bus bandwidth (accounting for algorithm overhead)
            # For ring all-reduce: bus_bw = 2 * (N-1) / N * bandwidth
            num_devices = metadata["num_devices"]
            if metadata["collective_type"] == CollectiveType.ALL_REDUCE:
                bus_bandwidth_gbps = 2 * (num_devices - 1) / num_devices * bandwidth_gbps
            else:
                bus_bandwidth_gbps = bandwidth_gbps

            profile = CollectiveProfile(
                collective_type=metadata["collective_type"],
                num_devices=num_devices,
                data_size_bytes=data_size_bytes,
                algorithm=metadata["algorithm"],
                duration_ms=duration_ms,
                bandwidth_gbps=bandwidth_gbps,
                bus_bandwidth_gbps=bus_bandwidth_gbps,
                timestamp=time.time()
            )

            self._profiles.append(profile)

            logger.debug(
                f"Collective profile: {profile.collective_type.value}, "
                f"{profile.num_devices} devices, "
                f"{profile.data_size_bytes / (1024**2):.2f} MB, "
                f"{profile.duration_ms:.2f} ms, "
                f"{profile.bandwidth_gbps:.2f} Gbps"
            )

            return profile

    def get_profiles(
        self,
        collective_type: Optional[CollectiveType] = None,
        limit: int = 100
    ) -> List[CollectiveProfile]:
        """
        Get recorded profiles.

        Args:
            collective_type: Filter by collective type
            limit: Maximum number of profiles to return

        Returns:
            List of profiles
        """
        with self._lock:
            profiles = self._profiles

            if collective_type:
                profiles = [p for p in profiles if p.collective_type == collective_type]

            return profiles[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        with self._lock:
            if not self._profiles:
                return {}

            total_ops = len(self._profiles)
            avg_duration = sum(p.duration_ms for p in self._profiles) / total_ops
            avg_bandwidth = sum(p.bandwidth_gbps for p in self._profiles) / total_ops

            by_type = {}
            for profile in self._profiles:
                ctype = profile.collective_type.value
                if ctype not in by_type:
                    by_type[ctype] = []
                by_type[ctype].append(profile)

            type_stats = {}
            for ctype, profiles in by_type.items():
                type_stats[ctype] = {
                    "count": len(profiles),
                    "avg_duration_ms": sum(p.duration_ms for p in profiles) / len(profiles),
                    "avg_bandwidth_gbps": sum(p.bandwidth_gbps for p in profiles) / len(profiles)
                }

            return {
                "total_operations": total_ops,
                "avg_duration_ms": avg_duration,
                "avg_bandwidth_gbps": avg_bandwidth,
                "by_type": type_stats
            }


class NCCLIntegration:
    """
    Main integration point for NCCL with adaptive flow control.

    Provides high-level API for optimized collective operations.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        routing_engine: RoutingEngine
    ):
        """
        Initialize NCCL integration.

        Args:
            topology: Network topology
            routing_engine: Routing engine
        """
        self.topology = topology
        self.routing_engine = routing_engine

        self.algorithm_selector = NCCLAlgorithmSelector(topology)
        self.tuner = NCCLTuner(topology)
        self.profiler = NCCLProfiler()

        logger.info("NCCLIntegration initialized")

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False
    ) -> Optional[dist.Work]:
        """
        Perform optimized all-reduce operation.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation
            group: Process group
            async_op: Whether to perform asynchronously

        Returns:
            Work handle if async_op=True
        """
        if not dist.is_available() or not dist.is_initialized():
            logger.warning("PyTorch distributed not initialized")
            return None

        # Get operation parameters
        world_size = dist.get_world_size(group)
        data_size_bytes = tensor.element_size() * tensor.nelement()

        # Select algorithm
        algorithm = self.algorithm_selector.select_algorithm(
            CollectiveType.ALL_REDUCE,
            world_size,
            data_size_bytes
        )

        # Get tuned configuration
        config = self.tuner.get_config(
            CollectiveType.ALL_REDUCE,
            world_size,
            data_size_bytes
        )

        # Start profiling
        op_id = self.profiler.start_op(
            CollectiveType.ALL_REDUCE,
            world_size,
            data_size_bytes,
            algorithm
        )

        # Perform all-reduce
        # Note: Actual NCCL algorithm selection would require
        # environment variable or C++ API integration
        try:
            work = dist.all_reduce(tensor, op, group, async_op)

            # End profiling if synchronous
            if not async_op:
                profile = self.profiler.end_op(op_id)
                if profile:
                    self.algorithm_selector.record_performance(profile)
                    self.tuner.update_config(
                        CollectiveType.ALL_REDUCE,
                        world_size,
                        profile
                    )

            return work

        except Exception as e:
            logger.error(f"Error in all_reduce: {e}", exc_info=True)
            self.profiler.end_op(op_id)
            raise

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False
    ) -> Optional[dist.Work]:
        """
        Perform optimized broadcast operation.

        Args:
            tensor: Tensor to broadcast
            src: Source rank
            group: Process group
            async_op: Whether to perform asynchronously

        Returns:
            Work handle if async_op=True
        """
        if not dist.is_available() or not dist.is_initialized():
            logger.warning("PyTorch distributed not initialized")
            return None

        world_size = dist.get_world_size(group)
        data_size_bytes = tensor.element_size() * tensor.nelement()

        # Select algorithm
        algorithm = self.algorithm_selector.select_algorithm(
            CollectiveType.BROADCAST,
            world_size,
            data_size_bytes
        )

        # Start profiling
        op_id = self.profiler.start_op(
            CollectiveType.BROADCAST,
            world_size,
            data_size_bytes,
            algorithm
        )

        try:
            work = dist.broadcast(tensor, src, group, async_op)

            if not async_op:
                profile = self.profiler.end_op(op_id)
                if profile:
                    self.algorithm_selector.record_performance(profile)

            return work

        except Exception as e:
            logger.error(f"Error in broadcast: {e}", exc_info=True)
            self.profiler.end_op(op_id)
            raise

    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False
    ) -> Optional[dist.Work]:
        """
        Perform optimized all-gather operation.

        Args:
            tensor_list: Output tensor list
            tensor: Input tensor
            group: Process group
            async_op: Whether to perform asynchronously

        Returns:
            Work handle if async_op=True
        """
        if not dist.is_available() or not dist.is_initialized():
            logger.warning("PyTorch distributed not initialized")
            return None

        world_size = dist.get_world_size(group)
        data_size_bytes = tensor.element_size() * tensor.nelement()

        # Select algorithm
        algorithm = self.algorithm_selector.select_algorithm(
            CollectiveType.ALL_GATHER,
            world_size,
            data_size_bytes
        )

        # Start profiling
        op_id = self.profiler.start_op(
            CollectiveType.ALL_GATHER,
            world_size,
            data_size_bytes,
            algorithm
        )

        try:
            work = dist.all_gather(tensor_list, tensor, group, async_op)

            if not async_op:
                profile = self.profiler.end_op(op_id)
                if profile:
                    self.algorithm_selector.record_performance(profile)

            return work

        except Exception as e:
            logger.error(f"Error in all_gather: {e}", exc_info=True)
            self.profiler.end_op(op_id)
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get NCCL integration statistics."""
        return {
            "profiler": self.profiler.get_statistics(),
            "topology_version": self.topology.get_topology_version()
        }

    def optimize_for_workload(
        self,
        workload_profile: Dict[str, Any]
    ) -> None:
        """
        Optimize NCCL configuration for specific workload.

        Args:
            workload_profile: Workload characteristics
        """
        logger.info(f"Optimizing NCCL for workload: {workload_profile}")

        # Extract workload characteristics
        collective_types = workload_profile.get("collective_types", [])
        data_sizes = workload_profile.get("data_sizes", [])
        num_devices = workload_profile.get("num_devices", 1)

        # Pre-generate tuned configurations
        for ctype_str in collective_types:
            try:
                ctype = CollectiveType(ctype_str)
                for data_size in data_sizes:
                    self.tuner.get_config(ctype, num_devices, data_size)
            except ValueError:
                logger.warning(f"Unknown collective type: {ctype_str}")


__all__ = [
    "NCCLAlgorithm",
    "CollectiveType",
    "NCCLConfig",
    "CollectiveProfile",
    "NCCLAlgorithmSelector",
    "NCCLTuner",
    "NCCLProfiler",
    "NCCLIntegration",
]

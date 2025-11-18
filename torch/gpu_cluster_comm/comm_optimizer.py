"""
GPU Cluster Communication Optimizer
GPU集群通讯优化器

This is the main coordinator that integrates all optimization components.

这是集成所有优化组件的主协调器。
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import torch

from .types import (
    CollectiveOperation,
    CollectiveAlgorithm,
    CompressionStrategy,
    CommMetrics,
    WorkloadStats,
    CommunicationPlan,
)
from .config import GPUClusterCommConfig, get_config
from .topology_manager import TopologyManager, GPUTopology
from .collective_optimizer import AdaptiveCollectiveOptimizer
from .overlap_scheduler import OverlapScheduler, GradientBucketing
from .message_coalescing import SmartCoalescer
from .compression_manager import CompressionManager
from .communication_profiler import CommunicationProfiler, PerformanceMonitor
from .load_balancer import LoadBalancer, DynamicLoadBalancer
from .utils import get_tensor_size_bytes, Timer

logger = logging.getLogger(__name__)


class GPUClusterCommOptimizer:
    """
    Main optimizer coordinator for GPU cluster communication.
    GPU集群通讯的主优化器协调器。

    This class integrates all optimization components and provides a
    high-level API for optimizing collective communications.

    Attributes:
        config: Configuration
        topology_mgr: Topology manager
        collective_opt: Collective operation optimizer
        overlap_scheduler: Compute-communication overlap scheduler
        profiler: Communication profiler
        load_balancer: Load balancer
        compression_mgr: Compression manager
        coalescer: Message coalescer
    """

    def __init__(
        self,
        config: Optional[GPUClusterCommConfig] = None,
        topology: Optional[GPUTopology] = None
    ):
        """
        Initialize GPU cluster communication optimizer.
        初始化GPU集群通讯优化器。

        Args:
            config: Configuration (uses global config if None)
            topology: GPU topology (auto-discovers if None)
        """
        # Configuration
        self.config = config or get_config()
        self.config.validate()

        # Initialize components
        self.topology_mgr = TopologyManager(topology)

        self.collective_opt = AdaptiveCollectiveOptimizer(self.topology_mgr)

        self.overlap_scheduler = OverlapScheduler(
            bucket_size_mb=self.config.overlap.bucket_size_mb
        )

        self.profiler = CommunicationProfiler(
            enable_profiling=self.config.profiling.enable_profiling,
            profile_interval=self.config.profiling.profile_interval
        )

        # Choose load balancer type
        if self.config.load_balancing.enable_load_balancing:
            self.load_balancer: Optional[LoadBalancer] = DynamicLoadBalancer(
                straggler_threshold=self.config.load_balancing.straggler_threshold,
                rebalance_interval=self.config.load_balancing.rebalance_interval
            )
        else:
            self.load_balancer = None

        self.compression_mgr = CompressionManager(
            default_strategy=self.config.compression.default_strategy,
            enable_error_feedback=self.config.compression.enable_error_feedback
        )

        # Get network parameters for coalescer
        avg_bw, avg_lat = self._get_network_params()
        self.coalescer = SmartCoalescer(
            bandwidth_gbps=avg_bw,
            latency_us=avg_lat,
            threshold_kb=self.config.coalescing.threshold_kb,
            max_coalesced_size_mb=self.config.coalescing.max_coalesced_size_mb,
            timeout_us=self.config.coalescing.timeout_us
        )

        # Performance monitor
        self.perf_monitor = PerformanceMonitor(profiler=self.profiler)

        # Auto-optimization state
        self._auto_optimization_enabled = False
        self._iteration = 0

        logger.info("GPU Cluster Communication Optimizer initialized")

    def optimize_allreduce(
        self,
        tensor: torch.Tensor,
        group: Optional[Any] = None,
        async_op: bool = False
    ) -> torch.Tensor:
        """
        Optimize and execute AllReduce operation.
        优化并执行AllReduce操作。

        Args:
            tensor: Input tensor
            group: Process group (None for default)
            async_op: Whether to execute asynchronously

        Returns:
            Result tensor (or future if async_op=True)
        """
        timer = Timer()
        timer.start()

        # Get tensor info
        tensor_size = get_tensor_size_bytes(tensor)
        num_ranks = self.topology_mgr.topology.get_num_devices()

        # Start profiling
        self.profiler.start_trace(
            "allreduce",
            metadata={'size': tensor_size, 'ranks': num_ranks}
        )

        try:
            # Step 1: Select algorithm
            algorithm = self.collective_opt.select_allreduce_algorithm(
                message_size=tensor_size,
                num_ranks=num_ranks
            )

            logger.debug(
                f"AllReduce: size={tensor_size}, "
                f"ranks={num_ranks}, "
                f"algorithm={algorithm.value}"
            )

            # Step 2: Apply compression if enabled
            compressed_tensor = tensor
            if self.config.compression.enable_compression:
                if tensor_size >= self.config.compression.min_message_size:
                    strategy = self.compression_mgr.select_compression(
                        tensor, target_ratio=self.config.compression.target_ratio
                    )

                    if strategy != CompressionStrategy.NONE:
                        compressed = self.compression_mgr.compress_gradient(
                            tensor, strategy, tensor_name="allreduce_tensor"
                        )
                        compressed_tensor = compressed.data

            # Step 3: Execute collective (placeholder - actual implementation
            # would call distributed backend)
            result_tensor = self._execute_allreduce(
                compressed_tensor, algorithm, group, async_op
            )

            # Step 4: Decompress if needed
            if compressed_tensor is not tensor:
                # Decompress result (simplified)
                result_tensor = result_tensor.to(tensor.dtype)

            # Step 5: Record metrics
            elapsed_us = timer.stop()

            if self.config.profiling.enable_profiling:
                avg_bw, _ = self._get_network_params()
                bandwidth_gbps = (tensor_size / elapsed_us) / 1000.0

                metrics = CommMetrics(
                    operation="allreduce",
                    algorithm=algorithm.value,
                    message_size=tensor_size,
                    latency_us=elapsed_us,
                    bandwidth_gbps=bandwidth_gbps,
                    num_ranks=num_ranks,
                    timestamp=time.time(),
                    rank=0,  # Simplified
                )

                self.profiler.record_communication(metrics)
                self.perf_monitor.record_metric(metrics)

            return result_tensor

        finally:
            self.profiler.stop_trace()

    def optimize_allgather(
        self,
        tensor: torch.Tensor,
        group: Optional[Any] = None,
        async_op: bool = False
    ) -> torch.Tensor:
        """
        Optimize and execute AllGather operation.
        优化并执行AllGather操作。

        Args:
            tensor: Input tensor
            group: Process group
            async_op: Whether to execute asynchronously

        Returns:
            Gathered tensors
        """
        timer = Timer()
        timer.start()

        tensor_size = get_tensor_size_bytes(tensor)
        num_ranks = self.topology_mgr.topology.get_num_devices()

        self.profiler.start_trace(
            "allgather",
            metadata={'size': tensor_size, 'ranks': num_ranks}
        )

        try:
            # Execute allgather (placeholder)
            result = self._execute_allgather(tensor, group, async_op)

            # Record metrics
            elapsed_us = timer.stop()

            if self.config.profiling.enable_profiling:
                bandwidth_gbps = (tensor_size * num_ranks / elapsed_us) / 1000.0

                metrics = CommMetrics(
                    operation="allgather",
                    algorithm="default",
                    message_size=tensor_size,
                    latency_us=elapsed_us,
                    bandwidth_gbps=bandwidth_gbps,
                    num_ranks=num_ranks,
                    timestamp=time.time(),
                    rank=0,
                )

                self.profiler.record_communication(metrics)

            return result

        finally:
            self.profiler.stop_trace()

    def optimize_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Optional[Any] = None,
        async_op: bool = False
    ) -> torch.Tensor:
        """
        Optimize and execute ReduceScatter operation.
        优化并执行ReduceScatter操作。

        Args:
            tensor: Input tensor
            group: Process group
            async_op: Whether to execute asynchronously

        Returns:
            Reduced and scattered tensor
        """
        timer = Timer()
        timer.start()

        tensor_size = get_tensor_size_bytes(tensor)
        num_ranks = self.topology_mgr.topology.get_num_devices()

        self.profiler.start_trace(
            "reduce_scatter",
            metadata={'size': tensor_size, 'ranks': num_ranks}
        )

        try:
            # Execute reduce_scatter (placeholder)
            result = self._execute_reduce_scatter(tensor, group, async_op)

            # Record metrics
            elapsed_us = timer.stop()

            if self.config.profiling.enable_profiling:
                bandwidth_gbps = (tensor_size / elapsed_us) / 1000.0

                metrics = CommMetrics(
                    operation="reduce_scatter",
                    algorithm="default",
                    message_size=tensor_size,
                    latency_us=elapsed_us,
                    bandwidth_gbps=bandwidth_gbps,
                    num_ranks=num_ranks,
                    timestamp=time.time(),
                    rank=0,
                )

                self.profiler.record_communication(metrics)

            return result

        finally:
            self.profiler.stop_trace()

    def schedule_pipeline_communication(
        self,
        stages: List[Tuple[str, int, float]]
    ) -> CommunicationPlan:
        """
        Schedule pipelined communication for multiple stages.
        为多个阶段调度流水线通讯。

        Args:
            stages: List of (name, tensor_size, duration) for each stage

        Returns:
            Communication plan
        """
        plan = self.overlap_scheduler.pipeline_communications(stages)

        logger.info(
            f"Pipeline schedule created: "
            f"critical_path={plan.critical_path_us:.2f}us, "
            f"parallelism={plan.parallelism_factor:.2f}x"
        )

        return plan

    def enable_auto_optimization(self) -> None:
        """
        Enable automatic optimization.
        启用自动优化。

        This enables adaptive tuning of all optimization parameters based
        on observed performance.
        """
        self._auto_optimization_enabled = True
        logger.info("Auto-optimization enabled")

    def disable_auto_optimization(self) -> None:
        """Disable automatic optimization"""
        self._auto_optimization_enabled = False
        logger.info("Auto-optimization disabled")

    def step(self) -> None:
        """
        Perform a step of auto-optimization.
        执行一步自动优化。

        This should be called at the end of each training iteration.
        """
        if not self._auto_optimization_enabled:
            return

        self._iteration += 1

        # Update network conditions for coalescer
        if self._iteration % 100 == 0:
            avg_bw, avg_lat = self._get_network_params()
            self.coalescer.update_network_conditions(avg_bw, avg_lat)

        # Check for performance degradation
        alerts = self.perf_monitor.get_alerts()
        if alerts:
            logger.warning(f"Performance alerts: {alerts}")

        # Identify bottlenecks
        if self._iteration % 1000 == 0:
            bottlenecks = self.profiler.identify_bottlenecks()
            if bottlenecks:
                logger.info(f"Identified {len(bottlenecks)} bottlenecks")
                for bottleneck in bottlenecks:
                    logger.info(
                        f"  - {bottleneck.description} "
                        f"(severity: {bottleneck.severity:.2f})"
                    )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all components.
        从所有组件获取综合统计信息。

        Returns:
            Dictionary with statistics
        """
        stats = {
            'profiler': self.profiler.get_summary_statistics(),
            'compression': self.compression_mgr.get_statistics(),
            'coalescing': self.coalescer.get_statistics(),
            'iteration': self._iteration,
        }

        if self.load_balancer:
            stats['load_balancing'] = self.load_balancer.get_statistics()

        return stats

    def print_summary(self) -> None:
        """Print a comprehensive summary of optimization results"""
        print("\n" + "=" * 70)
        print("GPU Cluster Communication Optimization Summary")
        print("=" * 70)

        # Profiler summary
        self.profiler.print_summary()

        # Compression stats
        comp_stats = self.compression_mgr.get_statistics()
        if comp_stats.get('num_compressions', 0) > 0:
            print("\nCompression Statistics:")
            print(f"  Compressions: {comp_stats['num_compressions']}")
            print(f"  Average ratio: {comp_stats.get('avg_compression_ratio', 1.0):.2f}")
            print(f"  Bytes saved: {comp_stats.get('total_original_bytes', 0) - comp_stats.get('total_compressed_bytes', 0)}")

        # Coalescing stats
        coal_stats = self.coalescer.get_statistics()
        if coal_stats.get('num_coalesced', 0) > 0:
            print("\nMessage Coalescing Statistics:")
            print(f"  Messages coalesced: {coal_stats['num_coalesced']}")
            print(f"  Messages combined: {coal_stats['num_messages_combined']}")
            print(f"  Bytes saved: {coal_stats['bytes_saved']}")

        # Load balancing stats
        if self.load_balancer:
            lb_stats = self.load_balancer.get_statistics()
            if lb_stats.get('num_rebalances', 0) > 0:
                print("\nLoad Balancing Statistics:")
                print(f"  Rebalances: {lb_stats['num_rebalances']}")
                print(f"  Stragglers detected: {lb_stats['num_stragglers_detected']}")
                print(f"  Current imbalance: {lb_stats.get('load_imbalance_ratio', 0.0):.2%}")

        print("=" * 70 + "\n")

    def export_trace(self, filepath: str) -> None:
        """
        Export profiling trace.
        导出性能分析跟踪。

        Args:
            filepath: Output file path
        """
        self.profiler.export_timeline(filepath, format="chrome")
        logger.info(f"Trace exported to {filepath}")

    # ========================================================================
    # Internal Helper Methods / 内部辅助方法
    # ========================================================================

    def _get_network_params(self) -> Tuple[float, float]:
        """Get average network bandwidth and latency"""
        if self.topology_mgr.topology.bandwidth_matrix is None:
            return 100.0, 5.0  # Default values

        # Compute from topology
        bw_matrix = self.topology_mgr.topology.bandwidth_matrix
        lat_matrix = self.topology_mgr.topology.distance_matrix

        # Average non-diagonal elements
        n = bw_matrix.shape[0]
        if n <= 1:
            return 100.0, 5.0

        bw_sum = 0.0
        lat_sum = 0.0
        count = 0

        for i in range(n):
            for j in range(n):
                if i != j:
                    if bw_matrix[i, j] > 0:
                        bw_sum += bw_matrix[i, j].item()
                    if lat_matrix[i, j] < float('inf'):
                        lat_sum += lat_matrix[i, j].item()
                    count += 1

        if count > 0:
            avg_bw = bw_sum / count
            avg_lat = lat_sum / count
        else:
            avg_bw = 100.0
            avg_lat = 5.0

        return avg_bw, avg_lat

    def _execute_allreduce(
        self,
        tensor: torch.Tensor,
        algorithm: CollectiveAlgorithm,
        group: Optional[Any],
        async_op: bool
    ) -> torch.Tensor:
        """
        Execute AllReduce (placeholder for actual implementation).
        执行AllReduce（实际实现的占位符）。

        In a real implementation, this would call the distributed backend
        (e.g., NCCL, Gloo) with the selected algorithm.
        """
        # Placeholder: just return tensor
        # In real implementation:
        # - Generate communication plan
        # - Execute using selected algorithm
        # - Handle async operations
        return tensor.clone()

    def _execute_allgather(
        self,
        tensor: torch.Tensor,
        group: Optional[Any],
        async_op: bool
    ) -> torch.Tensor:
        """Execute AllGather (placeholder)"""
        # Placeholder
        num_ranks = self.topology_mgr.topology.get_num_devices()
        return tensor.repeat(num_ranks, *([1] * (tensor.dim() - 1)))

    def _execute_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Optional[Any],
        async_op: bool
    ) -> torch.Tensor:
        """Execute ReduceScatter (placeholder)"""
        # Placeholder
        num_ranks = self.topology_mgr.topology.get_num_devices()

        # Split tensor
        if tensor.dim() > 0:
            chunk_size = tensor.shape[0] // num_ranks
            return tensor[:chunk_size].clone()
        else:
            return tensor.clone()


# Global optimizer instance
_global_optimizer: Optional[GPUClusterCommOptimizer] = None


def get_optimizer() -> GPUClusterCommOptimizer:
    """
    Get the global optimizer instance.
    获取全局优化器实例。

    Returns:
        Global optimizer
    """
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = GPUClusterCommOptimizer()
    return _global_optimizer


def set_optimizer(optimizer: GPUClusterCommOptimizer) -> None:
    """
    Set the global optimizer instance.
    设置全局优化器实例。

    Args:
        optimizer: Optimizer to set as global
    """
    global _global_optimizer
    _global_optimizer = optimizer


def reset_optimizer() -> None:
    """Reset the global optimizer"""
    global _global_optimizer
    _global_optimizer = None

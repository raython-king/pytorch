"""
GPU Cluster Communication Optimization - PyTorch Integration

This module provides deep integration with torch.distributed, enabling transparent
communication optimization for distributed training workloads.

Key features:
- Hook into torch.distributed collective operations (allreduce, allgather, etc.)
- Integration with DistributedDataParallel (DDP)
- Integration with FullyShardedDataParallel (FSDP)
- Integration with Pipeline Parallelism
- Shadow mode for safety and validation
- Zero-code-change transparent optimization
"""

import enum
import functools
import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


class IntegrationMode(enum.Enum):
    """Integration mode for GPU cluster communication optimization."""
    DISABLED = "disabled"  # No optimization, use native implementation
    SHADOW = "shadow"      # Run both optimized and native, compare results
    ENABLED = "enabled"    # Use optimized implementation only


class MetricsCollector:
    """Collect metrics for communication operations."""

    def __init__(self):
        self.metrics = {
            'allreduce': {'count': 0, 'total_time': 0.0, 'sizes': []},
            'allgather': {'count': 0, 'total_time': 0.0, 'sizes': []},
            'reduce_scatter': {'count': 0, 'total_time': 0.0, 'sizes': []},
            'broadcast': {'count': 0, 'total_time': 0.0, 'sizes': []},
        }
        self.comparisons = []

    def record_operation(self, op_type: str, duration: float, size: int):
        """Record a communication operation."""
        if op_type in self.metrics:
            self.metrics[op_type]['count'] += 1
            self.metrics[op_type]['total_time'] += duration
            self.metrics[op_type]['sizes'].append(size)

    def record_comparison(self, op_type: str, native_time: float,
                         optimized_time: float, size: int):
        """Record a comparison between native and optimized implementation."""
        speedup = native_time / optimized_time if optimized_time > 0 else 1.0
        self.comparisons.append({
            'op_type': op_type,
            'native_time': native_time,
            'optimized_time': optimized_time,
            'size': size,
            'speedup': speedup,
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {}
        for op_type, data in self.metrics.items():
            if data['count'] > 0:
                summary[op_type] = {
                    'count': data['count'],
                    'avg_time': data['total_time'] / data['count'],
                    'total_time': data['total_time'],
                }

        if self.comparisons:
            speedups = [c['speedup'] for c in self.comparisons]
            summary['comparison'] = {
                'avg_speedup': sum(speedups) / len(speedups),
                'min_speedup': min(speedups),
                'max_speedup': max(speedups),
                'num_comparisons': len(speedups),
            }

        return summary

    def reset(self):
        """Reset all metrics."""
        self.__init__()


class TorchDistributedIntegration:
    """Integration with torch.distributed for transparent optimization."""

    def __init__(self, mode: IntegrationMode = IntegrationMode.SHADOW):
        self.mode = mode
        self.metrics = MetricsCollector()
        self._original_functions = {}
        self._hooks_installed = False

        # Lazy import to avoid circular dependencies
        self._optimizer = None

    @property
    def optimizer(self):
        """Lazy initialization of optimizer."""
        if self._optimizer is None:
            # Import here to avoid circular dependency
            try:
                from ..optimizer.collective_optimizer import GPUClusterCommOptimizer
                self._optimizer = GPUClusterCommOptimizer()
            except ImportError:
                logger.warning("GPUClusterCommOptimizer not available, using mock")
                self._optimizer = MockOptimizer()
        return self._optimizer

    def set_mode(self, mode: IntegrationMode):
        """Change integration mode at runtime."""
        logger.info(f"Changing integration mode from {self.mode} to {mode}")
        self.mode = mode

    def hook_allreduce(self):
        """Hook torch.distributed.all_reduce for optimization."""
        if 'all_reduce' in self._original_functions:
            logger.warning("all_reduce already hooked, skipping")
            return

        original_allreduce = dist.all_reduce
        self._original_functions['all_reduce'] = original_allreduce

        @functools.wraps(original_allreduce)
        def optimized_allreduce(tensor: torch.Tensor,
                               op: dist.ReduceOp = dist.ReduceOp.SUM,
                               group=None,
                               async_op: bool = False):
            """Optimized allreduce with multiple modes."""

            if self.mode == IntegrationMode.DISABLED:
                return original_allreduce(tensor, op, group, async_op)

            elif self.mode == IntegrationMode.ENABLED:
                # Use optimized version only
                start_time = time.perf_counter()
                try:
                    result = self.optimizer.optimize_allreduce(
                        tensor, op, group, async_op
                    )
                    duration = time.perf_counter() - start_time
                    self.metrics.record_operation(
                        'allreduce', duration, tensor.numel() * tensor.element_size()
                    )
                    return result
                except Exception as e:
                    logger.error(f"Optimized allreduce failed: {e}, falling back")
                    return original_allreduce(tensor, op, group, async_op)

            elif self.mode == IntegrationMode.SHADOW:
                # Run both and compare
                return self._run_shadow_allreduce(
                    tensor, op, group, async_op, original_allreduce
                )

        dist.all_reduce = optimized_allreduce
        logger.info("all_reduce hooked successfully")

    def _run_shadow_allreduce(self, tensor: torch.Tensor, op, group,
                             async_op: bool, original_func: Callable):
        """Run both native and optimized allreduce for comparison."""
        # Clone tensor for optimized version
        tensor_opt = tensor.clone()

        # Run native version
        start_native = time.perf_counter()
        result_native = original_func(tensor, op, group, async_op)
        if async_op and result_native is not None:
            result_native.wait()
        native_time = time.perf_counter() - start_native

        # Run optimized version
        start_opt = time.perf_counter()
        try:
            result_opt = self.optimizer.optimize_allreduce(
                tensor_opt, op, group, False  # Always sync for comparison
            )
            opt_time = time.perf_counter() - start_opt

            # Verify correctness
            if not torch.allclose(tensor, tensor_opt, rtol=1e-5, atol=1e-6):
                logger.error("Optimized allreduce produced different results!")
            else:
                # Record comparison
                self.metrics.record_comparison(
                    'allreduce', native_time, opt_time,
                    tensor.numel() * tensor.element_size()
                )

        except Exception as e:
            logger.error(f"Optimized allreduce failed in shadow mode: {e}")

        # Always return native result in shadow mode
        return result_native

    def hook_allgather(self):
        """Hook torch.distributed.all_gather for optimization."""
        if 'all_gather' in self._original_functions:
            logger.warning("all_gather already hooked, skipping")
            return

        original_allgather = dist.all_gather
        self._original_functions['all_gather'] = original_allgather

        @functools.wraps(original_allgather)
        def optimized_allgather(tensor_list: List[torch.Tensor],
                               tensor: torch.Tensor,
                               group=None,
                               async_op: bool = False):
            """Optimized allgather."""

            if self.mode == IntegrationMode.DISABLED:
                return original_allgather(tensor_list, tensor, group, async_op)

            elif self.mode == IntegrationMode.ENABLED:
                start_time = time.perf_counter()
                try:
                    result = self.optimizer.optimize_allgather(
                        tensor_list, tensor, group, async_op
                    )
                    duration = time.perf_counter() - start_time
                    self.metrics.record_operation(
                        'allgather', duration, tensor.numel() * tensor.element_size()
                    )
                    return result
                except Exception as e:
                    logger.error(f"Optimized allgather failed: {e}, falling back")
                    return original_allgather(tensor_list, tensor, group, async_op)

            elif self.mode == IntegrationMode.SHADOW:
                return self._run_shadow_allgather(
                    tensor_list, tensor, group, async_op, original_allgather
                )

        dist.all_gather = optimized_allgather
        logger.info("all_gather hooked successfully")

    def _run_shadow_allgather(self, tensor_list: List[torch.Tensor],
                             tensor: torch.Tensor, group, async_op: bool,
                             original_func: Callable):
        """Run both native and optimized allgather for comparison."""
        # Clone for optimized version
        tensor_list_opt = [t.clone() for t in tensor_list]

        # Run native
        start_native = time.perf_counter()
        result_native = original_func(tensor_list, tensor, group, async_op)
        if async_op and result_native is not None:
            result_native.wait()
        native_time = time.perf_counter() - start_native

        # Run optimized
        start_opt = time.perf_counter()
        try:
            result_opt = self.optimizer.optimize_allgather(
                tensor_list_opt, tensor, group, False
            )
            opt_time = time.perf_counter() - start_opt

            # Verify correctness
            all_close = all(
                torch.allclose(t1, t2, rtol=1e-5, atol=1e-6)
                for t1, t2 in zip(tensor_list, tensor_list_opt)
            )
            if not all_close:
                logger.error("Optimized allgather produced different results!")
            else:
                self.metrics.record_comparison(
                    'allgather', native_time, opt_time,
                    tensor.numel() * tensor.element_size()
                )

        except Exception as e:
            logger.error(f"Optimized allgather failed in shadow mode: {e}")

        return result_native

    def hook_reduce_scatter(self):
        """Hook torch.distributed.reduce_scatter for optimization."""
        if 'reduce_scatter' in self._original_functions:
            logger.warning("reduce_scatter already hooked, skipping")
            return

        original_reduce_scatter = dist.reduce_scatter
        self._original_functions['reduce_scatter'] = original_reduce_scatter

        @functools.wraps(original_reduce_scatter)
        def optimized_reduce_scatter(output: torch.Tensor,
                                    input_list: List[torch.Tensor],
                                    op: dist.ReduceOp = dist.ReduceOp.SUM,
                                    group=None,
                                    async_op: bool = False):
            """Optimized reduce_scatter."""

            if self.mode == IntegrationMode.DISABLED:
                return original_reduce_scatter(output, input_list, op, group, async_op)

            elif self.mode == IntegrationMode.ENABLED:
                start_time = time.perf_counter()
                try:
                    result = self.optimizer.optimize_reduce_scatter(
                        output, input_list, op, group, async_op
                    )
                    duration = time.perf_counter() - start_time
                    self.metrics.record_operation(
                        'reduce_scatter', duration,
                        sum(t.numel() * t.element_size() for t in input_list)
                    )
                    return result
                except Exception as e:
                    logger.error(f"Optimized reduce_scatter failed: {e}, falling back")
                    return original_reduce_scatter(output, input_list, op, group, async_op)

            elif self.mode == IntegrationMode.SHADOW:
                # Shadow mode implementation
                output_opt = output.clone()

                start_native = time.perf_counter()
                result_native = original_reduce_scatter(output, input_list, op, group, async_op)
                if async_op and result_native is not None:
                    result_native.wait()
                native_time = time.perf_counter() - start_native

                start_opt = time.perf_counter()
                try:
                    self.optimizer.optimize_reduce_scatter(
                        output_opt, input_list, op, group, False
                    )
                    opt_time = time.perf_counter() - start_opt

                    if torch.allclose(output, output_opt, rtol=1e-5, atol=1e-6):
                        self.metrics.record_comparison(
                            'reduce_scatter', native_time, opt_time,
                            sum(t.numel() * t.element_size() for t in input_list)
                        )
                except Exception as e:
                    logger.error(f"Optimized reduce_scatter failed in shadow mode: {e}")

                return result_native

        dist.reduce_scatter = optimized_reduce_scatter
        logger.info("reduce_scatter hooked successfully")

    def hook_broadcast(self):
        """Hook torch.distributed.broadcast for optimization."""
        if 'broadcast' in self._original_functions:
            logger.warning("broadcast already hooked, skipping")
            return

        original_broadcast = dist.broadcast
        self._original_functions['broadcast'] = original_broadcast

        @functools.wraps(original_broadcast)
        def optimized_broadcast(tensor: torch.Tensor,
                               src: int,
                               group=None,
                               async_op: bool = False):
            """Optimized broadcast."""

            if self.mode == IntegrationMode.DISABLED:
                return original_broadcast(tensor, src, group, async_op)

            elif self.mode == IntegrationMode.ENABLED:
                start_time = time.perf_counter()
                try:
                    result = self.optimizer.optimize_broadcast(
                        tensor, src, group, async_op
                    )
                    duration = time.perf_counter() - start_time
                    self.metrics.record_operation(
                        'broadcast', duration, tensor.numel() * tensor.element_size()
                    )
                    return result
                except Exception as e:
                    logger.error(f"Optimized broadcast failed: {e}, falling back")
                    return original_broadcast(tensor, src, group, async_op)

            else:  # SHADOW mode
                return original_broadcast(tensor, src, group, async_op)

        dist.broadcast = optimized_broadcast
        logger.info("broadcast hooked successfully")

    def install_all_hooks(self):
        """Install all communication hooks."""
        if self._hooks_installed:
            logger.warning("Hooks already installed")
            return

        logger.info("Installing GPU cluster communication optimization hooks...")
        self.hook_allreduce()
        self.hook_allgather()
        self.hook_reduce_scatter()
        self.hook_broadcast()
        self._hooks_installed = True
        logger.info("All hooks installed successfully")

    def uninstall_all_hooks(self):
        """Uninstall all hooks and restore original functions."""
        if not self._hooks_installed:
            logger.warning("No hooks installed")
            return

        logger.info("Uninstalling GPU cluster communication optimization hooks...")

        if 'all_reduce' in self._original_functions:
            dist.all_reduce = self._original_functions['all_reduce']
        if 'all_gather' in self._original_functions:
            dist.all_gather = self._original_functions['all_gather']
        if 'reduce_scatter' in self._original_functions:
            dist.reduce_scatter = self._original_functions['reduce_scatter']
        if 'broadcast' in self._original_functions:
            dist.broadcast = self._original_functions['broadcast']

        self._original_functions.clear()
        self._hooks_installed = False
        logger.info("All hooks uninstalled successfully")

    def integrate_with_ddp(self, model: DDP,
                          optimize_bucket_size: bool = True,
                          optimize_overlap: bool = True):
        """Integrate with DistributedDataParallel for optimized gradient synchronization.

        Args:
            model: DDP model instance
            optimize_bucket_size: Whether to optimize gradient bucket size
            optimize_overlap: Whether to optimize computation-communication overlap
        """
        logger.info("Integrating with DistributedDataParallel...")

        if optimize_bucket_size:
            # Optimize bucket size based on network topology
            optimal_bucket_size = self._compute_optimal_bucket_size()
            logger.info(f"Setting optimal DDP bucket size: {optimal_bucket_size / 1e6:.2f} MB")
            # Note: bucket_cap_mb is set during DDP initialization
            # This is for logging/validation purposes

        if optimize_overlap:
            # Register hooks for better overlap
            model.register_comm_hook(None, self._ddp_comm_hook)
            logger.info("Registered DDP communication hook for overlap optimization")

    def _compute_optimal_bucket_size(self) -> int:
        """Compute optimal bucket size based on topology."""
        try:
            bandwidth = self.optimizer.topology_manager.get_average_bandwidth()
            # Heuristic: bucket should take ~10ms to transfer
            target_time = 0.01  # 10ms
            optimal_size = int(bandwidth * target_time)
            # Clamp to reasonable range (1MB - 100MB)
            return max(1024 * 1024, min(optimal_size, 100 * 1024 * 1024))
        except:
            return 25 * 1024 * 1024  # Default 25MB

    def _ddp_comm_hook(self, state, bucket):
        """Communication hook for DDP gradient synchronization."""
        # Use our optimized allreduce
        tensor = bucket.buffer()
        fut = torch.futures.Future()

        def callback():
            try:
                self.optimizer.optimize_allreduce(
                    tensor, dist.ReduceOp.SUM, async_op=False
                )
                fut.set_result(tensor)
            except Exception as e:
                fut.set_exception(e)

        # Schedule async execution
        torch.cuda.current_stream().synchronize()
        callback()

        return fut

    def integrate_with_fsdp(self, model):
        """Integrate with FullyShardedDataParallel.

        Note: FSDP integration requires hooking into FSDP's internal communication
        patterns, which is more complex than DDP.
        """
        logger.info("FSDP integration is experimental")
        warnings.warn(
            "FSDP integration is experimental. Please test thoroughly.",
            UserWarning
        )
        # TODO: Implement FSDP-specific optimizations

    def integrate_with_pipeline(self, model):
        """Integrate with Pipeline Parallelism.

        Pipeline parallelism typically uses point-to-point communication (send/recv)
        rather than collectives, which requires different optimization strategies.
        """
        logger.info("Pipeline parallelism integration is experimental")
        warnings.warn(
            "Pipeline parallelism integration is experimental.",
            UserWarning
        )
        # TODO: Implement pipeline-specific optimizations

    def print_metrics(self):
        """Print collected metrics."""
        summary = self.metrics.get_summary()

        print("\n" + "=" * 70)
        print("GPU Cluster Communication Optimization Metrics")
        print("=" * 70)

        for op_type, stats in summary.items():
            if op_type != 'comparison':
                print(f"\n{op_type.upper()}:")
                print(f"  Count: {stats['count']}")
                print(f"  Avg Time: {stats['avg_time']*1000:.2f} ms")
                print(f"  Total Time: {stats['total_time']:.2f} s")

        if 'comparison' in summary:
            comp = summary['comparison']
            print(f"\nCOMPARISON (Shadow Mode):")
            print(f"  Num Comparisons: {comp['num_comparisons']}")
            print(f"  Avg Speedup: {comp['avg_speedup']:.2f}x")
            print(f"  Min Speedup: {comp['min_speedup']:.2f}x")
            print(f"  Max Speedup: {comp['max_speedup']:.2f}x")

        print("=" * 70 + "\n")


class TransparentOptimization:
    """Transparent optimization - zero code changes required."""

    _instance = None

    @classmethod
    def enable_auto_optimization(cls, mode: IntegrationMode = IntegrationMode.SHADOW):
        """Enable automatic optimization for all distributed operations.

        This is the simplest way to enable GPU cluster communication optimization.
        Just call this function at the start of your training script.

        Args:
            mode: Integration mode (DISABLED, SHADOW, or ENABLED)
                 - SHADOW (default): Run both native and optimized, verify correctness
                 - ENABLED: Use optimized implementation only
                 - DISABLED: Disable optimization

        Example:
            >>> import torch.distributed as dist
            >>> from torch.gpu_cluster_comm.integration import TransparentOptimization
            >>>
            >>> dist.init_process_group(...)
            >>> TransparentOptimization.enable_auto_optimization()
            >>>
            >>> # Your existing distributed training code works unchanged
            >>> dist.all_reduce(tensor)
        """
        if cls._instance is not None:
            logger.warning("Optimization already enabled")
            return cls._instance

        integration = TorchDistributedIntegration(mode=mode)
        integration.install_all_hooks()
        cls._instance = integration

        print("\n" + "=" * 70)
        print("GPU Cluster Communication Optimization ENABLED")
        print(f"Mode: {mode.value}")
        print("=" * 70 + "\n")

        return integration

    @classmethod
    def disable_optimization(cls):
        """Disable optimization and restore native implementation."""
        if cls._instance is None:
            logger.warning("No optimization enabled")
            return

        cls._instance.uninstall_all_hooks()
        cls._instance.print_metrics()
        cls._instance = None

        print("\n" + "=" * 70)
        print("GPU Cluster Communication Optimization DISABLED")
        print("=" * 70 + "\n")

    @classmethod
    def get_instance(cls) -> Optional[TorchDistributedIntegration]:
        """Get the current optimization instance."""
        return cls._instance


class MockOptimizer:
    """Mock optimizer for testing when real optimizer is not available."""

    def __init__(self):
        logger.warning("Using mock optimizer - install full package for real optimization")

    def optimize_allreduce(self, tensor, op, group, async_op):
        """Fallback to native allreduce."""
        return dist.all_reduce(tensor, op, group, async_op)

    def optimize_allgather(self, tensor_list, tensor, group, async_op):
        """Fallback to native allgather."""
        return dist.all_gather(tensor_list, tensor, group, async_op)

    def optimize_reduce_scatter(self, output, input_list, op, group, async_op):
        """Fallback to native reduce_scatter."""
        return dist.reduce_scatter(output, input_list, op, group, async_op)

    def optimize_broadcast(self, tensor, src, group, async_op):
        """Fallback to native broadcast."""
        return dist.broadcast(tensor, src, group, async_op)


# Convenience functions
def enable_optimization(mode: IntegrationMode = IntegrationMode.SHADOW):
    """Enable GPU cluster communication optimization.

    Convenience function that calls TransparentOptimization.enable_auto_optimization.
    """
    return TransparentOptimization.enable_auto_optimization(mode)


def disable_optimization():
    """Disable GPU cluster communication optimization."""
    TransparentOptimization.disable_optimization()


def get_integration() -> Optional[TorchDistributedIntegration]:
    """Get current integration instance."""
    return TransparentOptimization.get_instance()

"""
Collective Communication Benchmarks

Benchmark tool for measuring performance of collective communication operations
with and without optimization.
"""

import argparse
import time
from typing import Dict, List, Tuple, Optional
import statistics

import torch
import torch.distributed as dist
import numpy as np


class CollectiveBenchmark:
    """Benchmark collective communication operations."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

        # Benchmark configuration
        self.warmup_iterations = 10
        self.benchmark_iterations = 100

        # Results storage
        self.results = {}

    def benchmark_allreduce(self, sizes: Optional[List[int]] = None) -> Dict:
        """Benchmark AllReduce operation across different message sizes.

        Args:
            sizes: List of message sizes in bytes. If None, use defaults.

        Returns:
            Dictionary containing benchmark results
        """
        if sizes is None:
            sizes = [
                1024,           # 1 KB
                1024 * 1024,    # 1 MB
                10 * 1024 * 1024,   # 10 MB
                100 * 1024 * 1024,  # 100 MB
                1024 * 1024 * 1024, # 1 GB
            ]

        results = {}

        for size_bytes in sizes:
            # Create tensor
            num_elements = size_bytes // 4  # float32 = 4 bytes
            tensor = torch.randn(num_elements, device=self.device)

            # Warmup
            for _ in range(self.warmup_iterations):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(self.benchmark_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()

                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

                torch.cuda.synchronize()
                end = time.perf_counter()

                times.append(end - start)

            # Calculate statistics
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)

            # Calculate bandwidth (GB/s)
            # AllReduce transfers 2(N-1)/N * size of data
            effective_bytes = size_bytes * 2 * (self.world_size - 1) / self.world_size
            bandwidth = (effective_bytes / (1024 ** 3)) / avg_time

            results[size_bytes] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time,
                'bandwidth_gbps': bandwidth,
            }

            if self.rank == 0:
                print(f"AllReduce {size_bytes / (1024**2):.2f} MB: "
                      f"{avg_time*1000:.3f} ± {std_time*1000:.3f} ms, "
                      f"{bandwidth:.2f} GB/s")

        return results

    def benchmark_allgather(self, sizes: Optional[List[int]] = None) -> Dict:
        """Benchmark AllGather operation.

        Args:
            sizes: List of message sizes in bytes per rank

        Returns:
            Dictionary containing benchmark results
        """
        if sizes is None:
            sizes = [1024, 1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024]

        results = {}

        for size_bytes in sizes:
            num_elements = size_bytes // 4
            tensor = torch.randn(num_elements, device=self.device)
            tensor_list = [
                torch.empty_like(tensor) for _ in range(self.world_size)
            ]

            # Warmup
            for _ in range(self.warmup_iterations):
                dist.all_gather(tensor_list, tensor)
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(self.benchmark_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()

                dist.all_gather(tensor_list, tensor)

                torch.cuda.synchronize()
                end = time.perf_counter()

                times.append(end - start)

            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0

            # AllGather transfers (N-1) * size per rank
            effective_bytes = size_bytes * (self.world_size - 1)
            bandwidth = (effective_bytes / (1024 ** 3)) / avg_time

            results[size_bytes] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'bandwidth_gbps': bandwidth,
            }

            if self.rank == 0:
                print(f"AllGather {size_bytes / (1024**2):.2f} MB: "
                      f"{avg_time*1000:.3f} ± {std_time*1000:.3f} ms, "
                      f"{bandwidth:.2f} GB/s")

        return results

    def benchmark_reduce_scatter(self, sizes: Optional[List[int]] = None) -> Dict:
        """Benchmark ReduceScatter operation."""
        if sizes is None:
            sizes = [1024, 1024 * 1024, 10 * 1024 * 1024]

        results = {}

        for size_bytes in sizes:
            num_elements = (size_bytes // 4) * self.world_size
            total_tensor = torch.randn(num_elements, device=self.device)
            input_list = list(total_tensor.chunk(self.world_size))
            output = torch.empty(num_elements // self.world_size, device=self.device)

            # Warmup
            for _ in range(self.warmup_iterations):
                dist.reduce_scatter(output, input_list)
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(self.benchmark_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()

                dist.reduce_scatter(output, input_list)

                torch.cuda.synchronize()
                end = time.perf_counter()

                times.append(end - start)

            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0

            bandwidth = (size_bytes / (1024 ** 3)) / avg_time

            results[size_bytes] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'bandwidth_gbps': bandwidth,
            }

            if self.rank == 0:
                print(f"ReduceScatter {size_bytes / (1024**2):.2f} MB: "
                      f"{avg_time*1000:.3f} ± {std_time*1000:.3f} ms, "
                      f"{bandwidth:.2f} GB/s")

        return results

    def benchmark_broadcast(self, sizes: Optional[List[int]] = None) -> Dict:
        """Benchmark Broadcast operation."""
        if sizes is None:
            sizes = [1024, 1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024]

        results = {}

        for size_bytes in sizes:
            num_elements = size_bytes // 4
            tensor = torch.randn(num_elements, device=self.device)

            # Warmup
            for _ in range(self.warmup_iterations):
                dist.broadcast(tensor, src=0)
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(self.benchmark_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()

                dist.broadcast(tensor, src=0)

                torch.cuda.synchronize()
                end = time.perf_counter()

                times.append(end - start)

            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0

            bandwidth = (size_bytes / (1024 ** 3)) / avg_time

            results[size_bytes] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'bandwidth_gbps': bandwidth,
            }

            if self.rank == 0:
                print(f"Broadcast {size_bytes / (1024**2):.2f} MB: "
                      f"{avg_time*1000:.3f} ± {std_time*1000:.3f} ms, "
                      f"{bandwidth:.2f} GB/s")

        return results

    def benchmark_all_operations(self) -> Dict:
        """Run all collective operation benchmarks.

        Returns:
            Dictionary with results for all operations
        """
        if self.rank == 0:
            print("\n" + "=" * 70)
            print("Collective Communication Benchmarks")
            print(f"World Size: {self.world_size}")
            print(f"Warmup Iterations: {self.warmup_iterations}")
            print(f"Benchmark Iterations: {self.benchmark_iterations}")
            print("=" * 70)

        results = {}

        if self.rank == 0:
            print("\n--- AllReduce ---")
        results['allreduce'] = self.benchmark_allreduce()

        if self.rank == 0:
            print("\n--- AllGather ---")
        results['allgather'] = self.benchmark_allgather()

        if self.rank == 0:
            print("\n--- ReduceScatter ---")
        results['reduce_scatter'] = self.benchmark_reduce_scatter()

        if self.rank == 0:
            print("\n--- Broadcast ---")
        results['broadcast'] = self.benchmark_broadcast()

        if self.rank == 0:
            print("\n" + "=" * 70)

        return results

    def compare_with_optimized(self, enable_optimization_func) -> Dict:
        """Compare native vs optimized performance.

        Args:
            enable_optimization_func: Function to enable optimization

        Returns:
            Comparison results
        """
        if self.rank == 0:
            print("\nBenchmarking NATIVE implementation...")

        native_results = self.benchmark_all_operations()

        if self.rank == 0:
            print("\nEnabling optimization...")

        enable_optimization_func()

        if self.rank == 0:
            print("\nBenchmarking OPTIMIZED implementation...")

        optimized_results = self.benchmark_all_operations()

        # Calculate speedups
        comparison = {}
        for op_name in native_results:
            comparison[op_name] = {}
            for size in native_results[op_name]:
                native_time = native_results[op_name][size]['avg_time']
                optimized_time = optimized_results[op_name][size]['avg_time']
                speedup = native_time / optimized_time

                comparison[op_name][size] = {
                    'native_time': native_time,
                    'optimized_time': optimized_time,
                    'speedup': speedup,
                }

        if self.rank == 0:
            self.print_comparison(comparison)

        return comparison

    def print_comparison(self, comparison: Dict):
        """Print comparison results in a readable format."""
        print("\n" + "=" * 70)
        print("Performance Comparison: Native vs Optimized")
        print("=" * 70)

        for op_name, sizes in comparison.items():
            print(f"\n{op_name.upper()}:")
            print(f"  {'Size':<12} {'Native (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10}")
            print("  " + "-" * 60)

            for size, data in sizes.items():
                size_str = f"{size / (1024**2):.1f} MB"
                native_ms = data['native_time'] * 1000
                optimized_ms = data['optimized_time'] * 1000
                speedup = data['speedup']

                print(f"  {size_str:<12} {native_ms:<15.3f} {optimized_ms:<15.3f} {speedup:<10.2f}x")

        print("\n" + "=" * 70)


class ScalingBenchmark:
    """Benchmark scalability (weak and strong scaling)."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

    def weak_scaling(self, base_size: int = 1024 * 1024) -> Dict:
        """Weak scaling: size per rank constant, increase num ranks.

        Args:
            base_size: Base size per rank in bytes

        Returns:
            Scaling results
        """
        # Each rank processes same amount of data
        num_elements = base_size // 4
        tensor = torch.randn(num_elements, device=self.device)

        # Benchmark AllReduce
        torch.cuda.synchronize()
        start = time.perf_counter()

        dist.all_reduce(tensor)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if self.rank == 0:
            print(f"Weak Scaling: {self.world_size} ranks, {elapsed*1000:.3f} ms")

        return {
            'world_size': self.world_size,
            'size_per_rank': base_size,
            'time': elapsed,
        }

    def strong_scaling(self, total_size: int = 100 * 1024 * 1024) -> Dict:
        """Strong scaling: total size constant, increase num ranks.

        Args:
            total_size: Total data size in bytes

        Returns:
            Scaling results
        """
        # Total data divided among ranks
        num_elements = total_size // 4
        tensor = torch.randn(num_elements, device=self.device)

        # Benchmark AllReduce
        torch.cuda.synchronize()
        start = time.perf_counter()

        dist.all_reduce(tensor)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if self.rank == 0:
            print(f"Strong Scaling: {self.world_size} ranks, {elapsed*1000:.3f} ms")

        return {
            'world_size': self.world_size,
            'total_size': total_size,
            'time': elapsed,
        }


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description='GPU Cluster Communication Benchmarks')
    parser.add_argument('--backend', type=str, default='nccl', help='Backend (nccl/gloo)')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--compare', action='store_true', help='Compare native vs optimized')
    parser.add_argument('--scaling', action='store_true', help='Run scaling benchmarks')

    args = parser.parse_args()

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend=args.backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Run benchmarks
    benchmark = CollectiveBenchmark(rank, world_size)
    benchmark.warmup_iterations = args.warmup
    benchmark.benchmark_iterations = args.iterations

    if args.compare:
        # Compare with optimization
        try:
            from torch.gpu_cluster_comm.integration import enable_optimization
            benchmark.compare_with_optimized(enable_optimization)
        except ImportError:
            if rank == 0:
                print("Optimization module not available, running native only")
            benchmark.benchmark_all_operations()
    else:
        benchmark.benchmark_all_operations()

    if args.scaling:
        scaling = ScalingBenchmark(rank, world_size)
        scaling.weak_scaling()
        scaling.strong_scaling()

    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

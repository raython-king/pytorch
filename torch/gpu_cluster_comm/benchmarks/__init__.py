"""
Benchmarking Tools for GPU Cluster Communication Optimization

This package provides comprehensive benchmarking tools for measuring
performance of collective communication operations.

Modules:
- benchmark_collectives: Benchmark individual collective operations
- benchmark_end_to_end: End-to-end training benchmarks
- profiling_tools: Communication profiling and analysis tools

Usage:
    # Run collective benchmarks
    torchrun --nproc_per_node=8 \
        torch/gpu_cluster_comm/benchmarks/benchmark_collectives.py \
        --compare

    # Run end-to-end benchmark
    torchrun --nproc_per_node=8 \
        torch/gpu_cluster_comm/benchmarks/benchmark_end_to_end.py \
        --model resnet50

    # Profile communication patterns
    from torch.gpu_cluster_comm.benchmarks import profiling_tools
    profiler = profiling_tools.CommunicationProfiler()
    profiler.profile_communication_pattern(model, dataloader)
"""

__all__ = [
    'benchmark_collectives',
    'profiling_tools',
]

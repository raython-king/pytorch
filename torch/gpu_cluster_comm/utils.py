"""
GPU Cluster Communication Utilities
GPU集群通讯工具函数

This module provides utility functions for the GPU cluster communication system.

本模块提供GPU集群通讯系统的工具函数。
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Set
import torch
import logging

from .types import (
    InterconnectType,
    GPUDevice,
    Link,
    CollectiveOperation,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Timing Utilities / 计时工具
# ============================================================================

class Timer:
    """
    High-resolution timer for performance measurements.
    高精度计时器用于性能测量。
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.elapsed_us: float = 0.0

    def start(self) -> None:
        """Start the timer"""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            Elapsed time in microseconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        end_time = time.perf_counter()
        self.elapsed_us = (end_time - self.start_time) * 1e6
        return self.elapsed_us

    def reset(self) -> None:
        """Reset the timer"""
        self.start_time = None
        self.elapsed_us = 0.0


class SynchronizedTimer:
    """
    CUDA-synchronized timer for accurate GPU timing.
    CUDA同步计时器用于精确的GPU计时。
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda')
        self.start_event: Optional[torch.cuda.Event] = None
        self.end_event: Optional[torch.cuda.Event] = None

    def start(self) -> None:
        """Start the timer with CUDA synchronization"""
        torch.cuda.synchronize(self.device)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record(torch.cuda.current_stream(self.device))

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            Elapsed time in microseconds
        """
        if self.start_event is None:
            raise RuntimeError("Timer was not started")

        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_event.record(torch.cuda.current_stream(self.device))
        torch.cuda.synchronize(self.device)

        # Event.elapsed_time returns milliseconds
        elapsed_ms = self.start_event.elapsed_time(self.end_event)
        return elapsed_ms * 1000.0  # Convert to microseconds


# ============================================================================
# Memory Utilities / 内存工具
# ============================================================================

def get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """
    Get the size of a tensor in bytes.
    获取张量的字节大小。

    Args:
        tensor: Input tensor

    Returns:
        Size in bytes
    """
    return tensor.numel() * tensor.element_size()


def estimate_buffer_size(num_elements: int, dtype: torch.dtype) -> int:
    """
    Estimate buffer size needed for a given number of elements.
    估计给定元素数量所需的缓冲区大小。

    Args:
        num_elements: Number of elements
        dtype: Data type

    Returns:
        Size in bytes
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    return num_elements * element_size


def align_size(size: int, alignment: int = 512) -> int:
    """
    Align size to the nearest multiple of alignment.
    将大小对齐到alignment的最近倍数。

    Args:
        size: Input size in bytes
        alignment: Alignment requirement in bytes

    Returns:
        Aligned size
    """
    return ((size + alignment - 1) // alignment) * alignment


# ============================================================================
# Bandwidth and Latency Utilities / 带宽和延迟工具
# ============================================================================

def compute_bandwidth_gbps(data_size_bytes: int, time_us: float) -> float:
    """
    Compute bandwidth in GB/s.
    计算GB/s带宽。

    Args:
        data_size_bytes: Data size in bytes
        time_us: Time in microseconds

    Returns:
        Bandwidth in GB/s
    """
    if time_us <= 0:
        return 0.0
    # GB/s = (bytes / 1e9) / (microseconds / 1e6)
    return (data_size_bytes / time_us) / 1000.0


def estimate_communication_time(
    message_size: int,
    bandwidth_gbps: float,
    latency_us: float
) -> float:
    """
    Estimate communication time using the alpha-beta model.
    使用alpha-beta模型估计通讯时间。

    Time = alpha + beta * message_size
    where alpha is latency and beta is inverse bandwidth.

    Args:
        message_size: Message size in bytes
        bandwidth_gbps: Bandwidth in GB/s
        latency_us: Latency in microseconds

    Returns:
        Estimated time in microseconds
    """
    if bandwidth_gbps <= 0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth_gbps}")

    # Convert bandwidth to bytes per microsecond
    bandwidth_bytes_per_us = bandwidth_gbps * 1000.0  # GB/s to MB/us to B/us

    transfer_time_us = message_size / bandwidth_bytes_per_us
    return latency_us + transfer_time_us


def get_effective_bandwidth(
    interconnect: InterconnectType,
    message_size: int
) -> float:
    """
    Get effective bandwidth based on interconnect type and message size.
    根据互联类型和消息大小获取有效带宽。

    Args:
        interconnect: Type of interconnect
        message_size: Message size in bytes

    Returns:
        Effective bandwidth in GB/s
    """
    # Base bandwidths (theoretical peak)
    base_bw = {
        InterconnectType.NVLINK: 600.0,  # NVLink 4.0: ~600 GB/s
        InterconnectType.PCIE: 64.0,     # PCIe 4.0 x16: ~64 GB/s
        InterconnectType.INFINIBAND: 25.0,  # HDR InfiniBand: ~25 GB/s
        InterconnectType.ETHERNET: 12.5,    # 100GbE: ~12.5 GB/s
        InterconnectType.UNKNOWN: 10.0,
    }

    peak_bw = base_bw.get(interconnect, 10.0)

    # Apply efficiency factor based on message size
    # Small messages achieve lower efficiency due to protocol overhead
    if message_size < 4096:  # < 4KB
        efficiency = 0.3
    elif message_size < 65536:  # < 64KB
        efficiency = 0.6
    elif message_size < 1048576:  # < 1MB
        efficiency = 0.8
    else:  # >= 1MB
        efficiency = 0.9

    return peak_bw * efficiency


# ============================================================================
# Topology Utilities / 拓扑工具
# ============================================================================

def compute_distance_matrix(
    devices: List[GPUDevice],
    links: List[Link]
) -> torch.Tensor:
    """
    Compute all-pairs shortest path distance matrix.
    计算全对最短路径距离矩阵。

    Uses Floyd-Warshall algorithm.

    Args:
        devices: List of GPU devices
        links: List of communication links

    Returns:
        Distance matrix (num_devices x num_devices)
        Value is the minimum latency in microseconds
    """
    n = len(devices)
    rank_to_idx = {dev.rank: i for i, dev in enumerate(devices)}

    # Initialize with infinity
    dist = torch.full((n, n), float('inf'))

    # Distance from device to itself is 0
    for i in range(n):
        dist[i, i] = 0.0

    # Set initial distances from links
    for link in links:
        i = rank_to_idx.get(link.source)
        j = rank_to_idx.get(link.target)
        if i is not None and j is not None:
            dist[i, j] = min(dist[i, j].item(), link.latency_us)
            if link.bidirectional:
                dist[j, i] = min(dist[j, i].item(), link.latency_us)

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

    return dist


def compute_bandwidth_matrix(
    devices: List[GPUDevice],
    links: List[Link]
) -> torch.Tensor:
    """
    Compute pairwise bandwidth matrix.
    计算成对带宽矩阵。

    Args:
        devices: List of GPU devices
        links: List of communication links

    Returns:
        Bandwidth matrix (num_devices x num_devices)
        Value is the bandwidth in GB/s
    """
    n = len(devices)
    rank_to_idx = {dev.rank: i for i, dev in enumerate(devices)}

    # Initialize with zeros
    bw = torch.zeros((n, n))

    # Set bandwidths from links
    for link in links:
        i = rank_to_idx.get(link.source)
        j = rank_to_idx.get(link.target)
        if i is not None and j is not None:
            # Use maximum bandwidth if multiple links exist
            bw[i, j] = max(bw[i, j].item(), link.bandwidth_gbps)
            if link.bidirectional:
                bw[j, i] = max(bw[j, i].item(), link.bandwidth_gbps)

    return bw


def find_nvlink_domains(devices: List[GPUDevice]) -> Dict[int, Set[int]]:
    """
    Find NVLink domains (groups of GPUs connected via NVLink).
    查找NVLink域（通过NVLink连接的GPU组）。

    Args:
        devices: List of GPU devices

    Returns:
        Dictionary mapping domain ID to set of ranks in that domain
    """
    domains: Dict[int, Set[int]] = {}

    for device in devices:
        if device.nvlink_domain is not None:
            domain_id = device.nvlink_domain
            if domain_id not in domains:
                domains[domain_id] = set()
            domains[domain_id].add(device.rank)

    return domains


def group_devices_by_node(devices: List[GPUDevice]) -> Dict[int, List[GPUDevice]]:
    """
    Group devices by their node ID.
    按节点ID对设备分组。

    Args:
        devices: List of GPU devices

    Returns:
        Dictionary mapping node ID to list of devices on that node
    """
    node_map: Dict[int, List[GPUDevice]] = {}

    for device in devices:
        if device.node_id not in node_map:
            node_map[device.node_id] = []
        node_map[device.node_id].append(device)

    return node_map


# ============================================================================
# Algorithm Utilities / 算法工具
# ============================================================================

def compute_ring_order(num_ranks: int) -> List[int]:
    """
    Compute simple ring order for ring-based algorithms.
    计算基于环的算法的简单环序。

    Args:
        num_ranks: Number of ranks

    Returns:
        List of ranks in ring order [0, 1, 2, ..., n-1]
    """
    return list(range(num_ranks))


def compute_optimal_ring_order(
    bandwidth_matrix: torch.Tensor,
    start_rank: int = 0
) -> List[int]:
    """
    Compute optimal ring order to maximize bandwidth.
    计算最佳环序以最大化带宽。

    Uses a greedy nearest-neighbor heuristic.

    Args:
        bandwidth_matrix: Pairwise bandwidth matrix
        start_rank: Starting rank for the ring

    Returns:
        List of ranks in optimal ring order
    """
    n = bandwidth_matrix.shape[0]
    visited = set([start_rank])
    order = [start_rank]
    current = start_rank

    while len(visited) < n:
        # Find unvisited rank with highest bandwidth from current
        best_next = -1
        best_bw = -1.0

        for rank in range(n):
            if rank not in visited:
                bw = bandwidth_matrix[current, rank].item()
                if bw > best_bw:
                    best_bw = bw
                    best_next = rank

        if best_next == -1:
            # No more reachable ranks, add remaining in order
            for rank in range(n):
                if rank not in visited:
                    order.append(rank)
                    visited.add(rank)
            break

        order.append(best_next)
        visited.add(best_next)
        current = best_next

    return order


def build_binary_tree(num_ranks: int, root: int = 0) -> Dict[int, Tuple[Optional[int], List[int]]]:
    """
    Build a binary tree topology.
    构建二叉树拓扑。

    Args:
        num_ranks: Number of ranks
        root: Root rank

    Returns:
        Dictionary mapping rank to (parent, [children])
    """
    tree: Dict[int, Tuple[Optional[int], List[int]]] = {}

    # Initialize all ranks
    for rank in range(num_ranks):
        tree[rank] = (None, [])

    # Build tree with root at specified rank
    # Map logical index to actual rank
    logical_to_rank = [(root + i) % num_ranks for i in range(num_ranks)]

    for logical_idx in range(num_ranks):
        rank = logical_to_rank[logical_idx]
        left_idx = 2 * logical_idx + 1
        right_idx = 2 * logical_idx + 2

        children = []
        if left_idx < num_ranks:
            left_rank = logical_to_rank[left_idx]
            children.append(left_rank)
            tree[left_rank] = (rank, tree[left_rank][1])

        if right_idx < num_ranks:
            right_rank = logical_to_rank[right_idx]
            children.append(right_rank)
            tree[right_rank] = (rank, tree[right_rank][1])

        tree[rank] = (tree[rank][0], children)

    return tree


def compute_chunk_size(
    total_size: int,
    num_chunks: int,
    alignment: int = 512
) -> List[int]:
    """
    Compute chunk sizes for pipelined communication.
    计算流水线通讯的块大小。

    Args:
        total_size: Total data size in bytes
        num_chunks: Number of chunks
        alignment: Alignment requirement in bytes

    Returns:
        List of chunk sizes
    """
    if num_chunks <= 0:
        raise ValueError(f"Number of chunks must be positive, got {num_chunks}")

    base_chunk_size = total_size // num_chunks
    aligned_chunk_size = align_size(base_chunk_size, alignment)

    chunk_sizes = [aligned_chunk_size] * num_chunks

    # Adjust last chunk to account for total size
    total_allocated = aligned_chunk_size * num_chunks
    if total_allocated < total_size:
        chunk_sizes[-1] += (total_size - total_allocated)
    elif total_allocated > total_size:
        # Last chunk is smaller
        chunk_sizes[-1] = total_size - aligned_chunk_size * (num_chunks - 1)
        if chunk_sizes[-1] <= 0:
            # Remove last chunk if it's negative or zero
            chunk_sizes.pop()

    return chunk_sizes


def compute_optimal_chunk_count(
    message_size: int,
    bandwidth_gbps: float,
    latency_us: float,
    num_ranks: int
) -> int:
    """
    Compute optimal number of chunks for pipelined communication.
    计算流水线通讯的最佳块数量。

    Args:
        message_size: Total message size in bytes
        bandwidth_gbps: Bandwidth in GB/s
        latency_us: Latency in microseconds
        num_ranks: Number of ranks

    Returns:
        Optimal number of chunks
    """
    # For ring allreduce, the optimal number of chunks balances
    # pipelining benefits against overhead

    # Minimum chunk size (avoid too many small chunks)
    min_chunk_size = 256 * 1024  # 256 KB

    # Maximum number of chunks
    max_chunks = message_size // min_chunk_size

    if max_chunks <= 1:
        return 1

    # Heuristic: use 2-4x number of ranks for good pipelining
    optimal_chunks = min(max_chunks, 4 * num_ranks)

    return max(1, optimal_chunks)


# ============================================================================
# Logging Utilities / 日志工具
# ============================================================================

def format_bytes(num_bytes: int) -> str:
    """
    Format bytes in human-readable form.
    将字节格式化为人类可读形式。

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_bandwidth(bandwidth_gbps: float) -> str:
    """
    Format bandwidth in human-readable form.
    将带宽格式化为人类可读形式。

    Args:
        bandwidth_gbps: Bandwidth in GB/s

    Returns:
        Formatted string (e.g., "10.5 GB/s")
    """
    if bandwidth_gbps < 1.0:
        return f"{bandwidth_gbps * 1000:.2f} MB/s"
    return f"{bandwidth_gbps:.2f} GB/s"


def format_time(time_us: float) -> str:
    """
    Format time in human-readable form.
    将时间格式化为人类可读形式。

    Args:
        time_us: Time in microseconds

    Returns:
        Formatted string (e.g., "1.5 ms")
    """
    if time_us < 1000:
        return f"{time_us:.2f} us"
    elif time_us < 1000000:
        return f"{time_us / 1000:.2f} ms"
    else:
        return f"{time_us / 1000000:.2f} s"


# ============================================================================
# Validation Utilities / 验证工具
# ============================================================================

def validate_collective_args(
    operation: CollectiveOperation,
    tensor: torch.Tensor,
    world_size: int
) -> None:
    """
    Validate arguments for collective operations.
    验证集合操作的参数。

    Args:
        operation: Type of collective operation
        tensor: Input tensor
        world_size: Number of ranks

    Raises:
        ValueError: If arguments are invalid
    """
    if world_size <= 0:
        raise ValueError(f"World size must be positive, got {world_size}")

    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA device")

    if operation == CollectiveOperation.REDUCE_SCATTER:
        # For reduce_scatter, tensor size must be divisible by world_size
        if tensor.numel() % world_size != 0:
            raise ValueError(
                f"Tensor size {tensor.numel()} must be divisible by "
                f"world_size {world_size} for reduce_scatter"
            )


def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.
    检查数字是否为2的幂。

    Args:
        n: Input number

    Returns:
        True if n is a power of two
    """
    return n > 0 and (n & (n - 1)) == 0

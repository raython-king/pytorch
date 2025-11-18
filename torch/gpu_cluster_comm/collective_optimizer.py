"""
Collective Communication Optimizer
集合通讯优化器

This module provides adaptive optimization for collective communication operations.

本模块提供集合通讯操作的自适应优化。
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
import torch

from .types import (
    CollectiveAlgorithm,
    CollectiveOperation,
    CommunicationPlan,
    CommunicationStep,
    ReductionOp,
)
from .topology_manager import GPUTopology, TopologyManager
from .utils import (
    estimate_communication_time,
    compute_chunk_size,
    compute_optimal_chunk_count,
    is_power_of_two,
)

logger = logging.getLogger(__name__)


class AlgorithmCostModel:
    """
    Cost model for collective communication algorithms.
    集合通讯算法的成本模型。

    This class estimates the cost (latency) of different collective algorithms
    based on message size, number of ranks, and topology.
    """

    @staticmethod
    def ring_allreduce_cost(
        message_size: int,
        num_ranks: int,
        bandwidth_gbps: float,
        latency_us: float,
        num_chunks: int = 1
    ) -> float:
        """
        Estimate cost of ring AllReduce.
        估计环形AllReduce的成本。

        Ring AllReduce algorithm:
        - Phase 1: Reduce-scatter (N-1 steps)
        - Phase 2: AllGather (N-1 steps)
        - Total: 2(N-1) steps

        Time = 2(N-1) * (alpha + beta * M/N)
        where N = num_ranks, M = message_size

        Args:
            message_size: Total message size in bytes
            num_ranks: Number of ranks
            bandwidth_gbps: Link bandwidth in GB/s
            latency_us: Link latency in microseconds
            num_chunks: Number of chunks for pipelining

        Returns:
            Estimated time in microseconds
        """
        if num_ranks <= 1:
            return 0.0

        chunk_size = message_size // num_ranks

        # Cost per step
        step_time = estimate_communication_time(chunk_size, bandwidth_gbps, latency_us)

        # Total steps: 2(N-1) for reduce-scatter and allgather
        total_steps = 2 * (num_ranks - 1)

        # With pipelining, we can overlap some steps
        if num_chunks > 1:
            # Pipeline factor reduces latency
            pipeline_factor = 1.0 + (num_chunks - 1) / num_chunks
            total_time = step_time * total_steps / pipeline_factor
        else:
            total_time = step_time * total_steps

        return total_time

    @staticmethod
    def tree_allreduce_cost(
        message_size: int,
        num_ranks: int,
        bandwidth_gbps: float,
        latency_us: float
    ) -> float:
        """
        Estimate cost of tree-based AllReduce.
        估计基于树的AllReduce的成本。

        Binary tree AllReduce:
        - Reduce phase: log2(N) steps (leaf to root)
        - Broadcast phase: log2(N) steps (root to leaf)
        - Total: 2 * log2(N) steps

        Time = 2 * log2(N) * (alpha + beta * M)

        Args:
            message_size: Total message size in bytes
            num_ranks: Number of ranks
            bandwidth_gbps: Link bandwidth in GB/s
            latency_us: Link latency in microseconds

        Returns:
            Estimated time in microseconds
        """
        if num_ranks <= 1:
            return 0.0

        # Number of tree levels
        tree_depth = math.ceil(math.log2(num_ranks))

        # Cost per step
        step_time = estimate_communication_time(message_size, bandwidth_gbps, latency_us)

        # Total steps: 2 * tree_depth (up and down)
        total_time = 2 * tree_depth * step_time

        return total_time

    @staticmethod
    def halving_doubling_cost(
        message_size: int,
        num_ranks: int,
        bandwidth_gbps: float,
        latency_us: float
    ) -> float:
        """
        Estimate cost of recursive halving-doubling AllReduce.
        估计递归减半-加倍AllReduce的成本。

        Recursive halving-doubling:
        - Works best when num_ranks is power of 2
        - Reduce-scatter phase: log2(N) steps
        - AllGather phase: log2(N) steps
        - Total: 2 * log2(N) steps

        Time = 2 * log2(N) * (alpha + beta * M/2)

        Args:
            message_size: Total message size in bytes
            num_ranks: Number of ranks
            bandwidth_gbps: Link bandwidth in GB/s
            latency_us: Link latency in microseconds

        Returns:
            Estimated time in microseconds
        """
        if num_ranks <= 1:
            return 0.0

        # Penalty if not power of 2
        if not is_power_of_two(num_ranks):
            # Add overhead for non-power-of-2 case
            overhead_factor = 1.2
        else:
            overhead_factor = 1.0

        num_steps = math.ceil(math.log2(num_ranks))

        # Average message size per step (halves each iteration)
        avg_chunk_size = message_size // 2

        step_time = estimate_communication_time(avg_chunk_size, bandwidth_gbps, latency_us)

        total_time = 2 * num_steps * step_time * overhead_factor

        return total_time

    @staticmethod
    def hierarchical_allreduce_cost(
        message_size: int,
        num_ranks: int,
        intra_node_bw: float,
        inter_node_bw: float,
        latency_us: float,
        num_nodes: int
    ) -> float:
        """
        Estimate cost of hierarchical AllReduce.
        估计分层AllReduce的成本。

        Hierarchical AllReduce:
        - Intra-node reduce (within each node)
        - Inter-node allreduce (between node leaders)
        - Intra-node broadcast (within each node)

        Args:
            message_size: Total message size in bytes
            num_ranks: Total number of ranks
            intra_node_bw: Intra-node bandwidth (GB/s)
            inter_node_bw: Inter-node bandwidth (GB/s)
            latency_us: Base latency in microseconds
            num_nodes: Number of nodes

        Returns:
            Estimated time in microseconds
        """
        if num_ranks <= 1:
            return 0.0

        ranks_per_node = num_ranks // num_nodes

        # Intra-node reduce (using ring or tree)
        if ranks_per_node <= 4:
            intra_reduce_time = AlgorithmCostModel.tree_allreduce_cost(
                message_size, ranks_per_node, intra_node_bw, latency_us
            )
        else:
            intra_reduce_time = AlgorithmCostModel.ring_allreduce_cost(
                message_size, ranks_per_node, intra_node_bw, latency_us
            )

        # Inter-node allreduce (between leaders)
        if num_nodes <= 4:
            inter_allreduce_time = AlgorithmCostModel.tree_allreduce_cost(
                message_size, num_nodes, inter_node_bw, latency_us * 2
            )
        else:
            inter_allreduce_time = AlgorithmCostModel.ring_allreduce_cost(
                message_size, num_nodes, inter_node_bw, latency_us * 2
            )

        # Intra-node broadcast
        intra_broadcast_time = estimate_communication_time(
            message_size, intra_node_bw, latency_us
        ) * math.ceil(math.log2(ranks_per_node))

        total_time = intra_reduce_time + inter_allreduce_time + intra_broadcast_time

        return total_time


class AdaptiveCollectiveOptimizer:
    """
    Adaptive optimizer for collective communications.
    集合通讯的自适应优化器。

    This class selects the optimal algorithm and parameters for collective
    operations based on message size, number of ranks, and network topology.

    Attributes:
        topology_mgr: Topology manager
        cost_model: Cost model for algorithms
    """

    def __init__(self, topology_mgr: Optional[TopologyManager] = None):
        """
        Initialize the optimizer.
        初始化优化器。

        Args:
            topology_mgr: Topology manager (will auto-discover if None)
        """
        if topology_mgr is None:
            topology_mgr = TopologyManager()

        self.topology_mgr = topology_mgr
        self.cost_model = AlgorithmCostModel()

        # Cache for algorithm selection decisions
        self._selection_cache: Dict[Tuple, CollectiveAlgorithm] = {}

    def select_allreduce_algorithm(
        self,
        message_size: int,
        num_ranks: int,
        topology: Optional[GPUTopology] = None
    ) -> CollectiveAlgorithm:
        """
        Select optimal AllReduce algorithm.
        选择最佳AllReduce算法。

        Args:
            message_size: Message size in bytes
            num_ranks: Number of ranks
            topology: GPU topology (uses default if None)

        Returns:
            Selected algorithm
        """
        # Check cache
        cache_key = (message_size, num_ranks, CollectiveOperation.ALLREDUCE)
        if cache_key in self._selection_cache:
            return self._selection_cache[cache_key]

        if topology is None:
            topology = self.topology_mgr.topology

        # Get average bandwidth and latency
        avg_bandwidth, avg_latency = self._get_avg_network_params(topology)

        # Small message threshold (32 KB)
        small_msg_threshold = 32 * 1024

        # Large message threshold (1 MB)
        large_msg_threshold = 1 * 1024 * 1024

        # Decision logic
        selected_algo = None

        if num_ranks <= 2:
            # For 2 ranks, just do direct send/recv
            selected_algo = CollectiveAlgorithm.RING

        elif message_size < small_msg_threshold:
            # Small messages: latency-bound
            # Use tree-based algorithm to minimize latency
            if is_power_of_two(num_ranks):
                selected_algo = CollectiveAlgorithm.HALVING_DOUBLING
            else:
                selected_algo = CollectiveAlgorithm.TREE

        elif message_size > large_msg_threshold:
            # Large messages: bandwidth-bound
            # Use ring algorithm for better bandwidth utilization
            selected_algo = CollectiveAlgorithm.RING

        else:
            # Medium messages: compare algorithms
            costs = {}

            # Ring
            costs[CollectiveAlgorithm.RING] = self.cost_model.ring_allreduce_cost(
                message_size, num_ranks, avg_bandwidth, avg_latency
            )

            # Tree
            costs[CollectiveAlgorithm.TREE] = self.cost_model.tree_allreduce_cost(
                message_size, num_ranks, avg_bandwidth, avg_latency
            )

            # Halving-doubling (if power of 2)
            if is_power_of_two(num_ranks):
                costs[CollectiveAlgorithm.HALVING_DOUBLING] = (
                    self.cost_model.halving_doubling_cost(
                        message_size, num_ranks, avg_bandwidth, avg_latency
                    )
                )

            # Hierarchical (if multi-node)
            num_nodes = topology.get_num_nodes()
            if num_nodes > 1:
                intra_bw, inter_bw = self._get_hierarchical_bandwidth(topology)
                costs[CollectiveAlgorithm.HIERARCHICAL] = (
                    self.cost_model.hierarchical_allreduce_cost(
                        message_size, num_ranks, intra_bw, inter_bw,
                        avg_latency, num_nodes
                    )
                )

            # Select algorithm with minimum cost
            selected_algo = min(costs, key=costs.get)

        # Cache decision
        self._selection_cache[cache_key] = selected_algo

        logger.debug(
            f"Selected {selected_algo.value} for AllReduce "
            f"(size={message_size}, ranks={num_ranks})"
        )

        return selected_algo

    def compute_chunks(
        self,
        message_size: int,
        bandwidth_gbps: float,
        latency_us: float,
        num_ranks: int
    ) -> Tuple[int, List[int]]:
        """
        Compute optimal chunking for pipelined communication.
        计算流水线通讯的最佳分块。

        Args:
            message_size: Total message size in bytes
            bandwidth_gbps: Bandwidth in GB/s
            latency_us: Latency in microseconds
            num_ranks: Number of ranks

        Returns:
            Tuple of (num_chunks, chunk_sizes)
        """
        # Compute optimal number of chunks
        num_chunks = compute_optimal_chunk_count(
            message_size, bandwidth_gbps, latency_us, num_ranks
        )

        # Compute chunk sizes
        chunk_sizes = compute_chunk_size(message_size, num_chunks)

        return num_chunks, chunk_sizes

    def generate_communication_plan(
        self,
        operation: CollectiveOperation,
        algorithm: CollectiveAlgorithm,
        message_size: int,
        num_ranks: int,
        topology: Optional[GPUTopology] = None
    ) -> CommunicationPlan:
        """
        Generate a detailed communication plan.
        生成详细的通讯计划。

        Args:
            operation: Type of collective operation
            algorithm: Algorithm to use
            message_size: Message size in bytes
            num_ranks: Number of ranks
            topology: GPU topology

        Returns:
            Communication plan with steps
        """
        if topology is None:
            topology = self.topology_mgr.topology

        plan = CommunicationPlan(
            operation=operation,
            algorithm=algorithm,
            num_ranks=num_ranks,
            message_size=message_size,
        )

        # Generate steps based on algorithm
        if algorithm == CollectiveAlgorithm.RING:
            if operation == CollectiveOperation.ALLREDUCE:
                plan.steps = self._generate_ring_allreduce_steps(
                    message_size, num_ranks
                )
            elif operation == CollectiveOperation.ALLGATHER:
                plan.steps = self._generate_ring_allgather_steps(
                    message_size, num_ranks
                )
            elif operation == CollectiveOperation.REDUCE_SCATTER:
                plan.steps = self._generate_ring_reduce_scatter_steps(
                    message_size, num_ranks
                )

        elif algorithm == CollectiveAlgorithm.TREE:
            if operation == CollectiveOperation.ALLREDUCE:
                plan.steps = self._generate_tree_allreduce_steps(
                    message_size, num_ranks
                )

        elif algorithm == CollectiveAlgorithm.HIERARCHICAL:
            if operation == CollectiveOperation.ALLREDUCE:
                plan.steps = self._generate_hierarchical_allreduce_steps(
                    message_size, num_ranks, topology
                )

        # Estimate total time
        avg_bw, avg_lat = self._get_avg_network_params(topology)
        plan.estimated_time_us = self._estimate_plan_time(plan, avg_bw, avg_lat)

        return plan

    def _generate_ring_allreduce_steps(
        self,
        message_size: int,
        num_ranks: int
    ) -> List[CommunicationStep]:
        """
        Generate steps for ring AllReduce.
        生成环形AllReduce的步骤。

        Ring AllReduce:
        1. Reduce-scatter: Each rank sends/receives (N-1) times
        2. AllGather: Each rank sends/receives (N-1) times
        """
        steps = []
        chunk_size = message_size // num_ranks
        step_id = 0

        # Get ring order
        ring_order = self.topology_mgr.compute_optimal_ring_order()

        # Phase 1: Reduce-scatter
        for step in range(num_ranks - 1):
            for i, rank in enumerate(ring_order):
                next_rank = ring_order[(i + 1) % num_ranks]
                chunk_id = (i - step) % num_ranks

                comm_step = CommunicationStep(
                    step_id=step_id,
                    operation="reduce_send",
                    source_ranks=[rank],
                    target_ranks=[next_rank],
                    message_size=chunk_size,
                    chunk_id=chunk_id,
                )
                steps.append(comm_step)
                step_id += 1

        # Phase 2: AllGather
        for step in range(num_ranks - 1):
            for i, rank in enumerate(ring_order):
                next_rank = ring_order[(i + 1) % num_ranks]
                chunk_id = (i - step + 1) % num_ranks

                comm_step = CommunicationStep(
                    step_id=step_id,
                    operation="send",
                    source_ranks=[rank],
                    target_ranks=[next_rank],
                    message_size=chunk_size,
                    chunk_id=chunk_id,
                )
                steps.append(comm_step)
                step_id += 1

        return steps

    def _generate_ring_allgather_steps(
        self,
        message_size: int,
        num_ranks: int
    ) -> List[CommunicationStep]:
        """Generate steps for ring AllGather"""
        steps = []
        chunk_size = message_size // num_ranks
        step_id = 0

        ring_order = self.topology_mgr.compute_optimal_ring_order()

        for step in range(num_ranks - 1):
            for i, rank in enumerate(ring_order):
                next_rank = ring_order[(i + 1) % num_ranks]
                chunk_id = (i - step) % num_ranks

                comm_step = CommunicationStep(
                    step_id=step_id,
                    operation="send",
                    source_ranks=[rank],
                    target_ranks=[next_rank],
                    message_size=chunk_size,
                    chunk_id=chunk_id,
                )
                steps.append(comm_step)
                step_id += 1

        return steps

    def _generate_ring_reduce_scatter_steps(
        self,
        message_size: int,
        num_ranks: int
    ) -> List[CommunicationStep]:
        """Generate steps for ring Reduce-Scatter"""
        steps = []
        chunk_size = message_size // num_ranks
        step_id = 0

        ring_order = self.topology_mgr.compute_optimal_ring_order()

        for step in range(num_ranks - 1):
            for i, rank in enumerate(ring_order):
                next_rank = ring_order[(i + 1) % num_ranks]
                chunk_id = (i - step) % num_ranks

                comm_step = CommunicationStep(
                    step_id=step_id,
                    operation="reduce_send",
                    source_ranks=[rank],
                    target_ranks=[next_rank],
                    message_size=chunk_size,
                    chunk_id=chunk_id,
                )
                steps.append(comm_step)
                step_id += 1

        return steps

    def _generate_tree_allreduce_steps(
        self,
        message_size: int,
        num_ranks: int
    ) -> List[CommunicationStep]:
        """Generate steps for tree-based AllReduce"""
        steps = []
        step_id = 0

        # Build tree
        tree = self.topology_mgr.build_communication_tree(0, CollectiveAlgorithm.TREE)

        # Phase 1: Reduce up the tree (leaf to root)
        # Process in reverse topological order
        for rank in range(num_ranks - 1, -1, -1):
            parent, children = tree[rank]
            if parent is not None:
                comm_step = CommunicationStep(
                    step_id=step_id,
                    operation="reduce_send",
                    source_ranks=[rank],
                    target_ranks=[parent],
                    message_size=message_size,
                )
                steps.append(comm_step)
                step_id += 1

        # Phase 2: Broadcast down the tree (root to leaf)
        for rank in range(num_ranks):
            parent, children = tree[rank]
            for child in children:
                comm_step = CommunicationStep(
                    step_id=step_id,
                    operation="send",
                    source_ranks=[rank],
                    target_ranks=[child],
                    message_size=message_size,
                )
                steps.append(comm_step)
                step_id += 1

        return steps

    def _generate_hierarchical_allreduce_steps(
        self,
        message_size: int,
        num_ranks: int,
        topology: GPUTopology
    ) -> List[CommunicationStep]:
        """Generate steps for hierarchical AllReduce"""
        steps = []
        step_id = 0

        # Get intra-node groups
        intra_groups = self.topology_mgr.get_intra_node_groups()

        # Get inter-node leaders
        leader_ranks, all_ranks = self.topology_mgr.get_inter_node_groups()

        # Phase 1: Intra-node reduce (within each node)
        for group in intra_groups:
            if len(group) > 1:
                # Reduce to first rank in group
                leader = group[0]
                for rank in group[1:]:
                    comm_step = CommunicationStep(
                        step_id=step_id,
                        operation="reduce_send",
                        source_ranks=[rank],
                        target_ranks=[leader],
                        message_size=message_size,
                    )
                    steps.append(comm_step)
                    step_id += 1

        # Phase 2: Inter-node allreduce (between leaders)
        if len(leader_ranks) > 1:
            # Use ring allreduce for leaders
            chunk_size = message_size // len(leader_ranks)
            for step in range(len(leader_ranks) - 1):
                for i, rank in enumerate(leader_ranks):
                    next_rank = leader_ranks[(i + 1) % len(leader_ranks)]
                    comm_step = CommunicationStep(
                        step_id=step_id,
                        operation="reduce_send",
                        source_ranks=[rank],
                        target_ranks=[next_rank],
                        message_size=chunk_size,
                    )
                    steps.append(comm_step)
                    step_id += 1

        # Phase 3: Intra-node broadcast (within each node)
        for group in intra_groups:
            if len(group) > 1:
                leader = group[0]
                for rank in group[1:]:
                    comm_step = CommunicationStep(
                        step_id=step_id,
                        operation="send",
                        source_ranks=[leader],
                        target_ranks=[rank],
                        message_size=message_size,
                    )
                    steps.append(comm_step)
                    step_id += 1

        return steps

    def optimize_hierarchical_allreduce(
        self,
        message_size: int,
        num_ranks: int,
        topology: GPUTopology
    ) -> Tuple[CollectiveAlgorithm, CollectiveAlgorithm]:
        """
        Optimize hierarchical AllReduce by selecting best intra/inter algorithms.
        通过选择最佳节点内/节点间算法优化分层AllReduce。

        Args:
            message_size: Message size in bytes
            num_ranks: Number of ranks
            topology: GPU topology

        Returns:
            Tuple of (intra_node_algorithm, inter_node_algorithm)
        """
        # Get topology info
        intra_groups = self.topology_mgr.get_intra_node_groups()
        avg_intra_size = sum(len(g) for g in intra_groups) / len(intra_groups)

        num_nodes = topology.get_num_nodes()

        # Select intra-node algorithm
        if avg_intra_size <= 4:
            intra_algo = CollectiveAlgorithm.TREE
        else:
            intra_algo = CollectiveAlgorithm.RING

        # Select inter-node algorithm
        if num_nodes <= 4:
            inter_algo = CollectiveAlgorithm.TREE
        else:
            inter_algo = CollectiveAlgorithm.RING

        return intra_algo, inter_algo

    def _get_avg_network_params(
        self,
        topology: GPUTopology
    ) -> Tuple[float, float]:
        """
        Get average bandwidth and latency from topology.
        从拓扑获取平均带宽和延迟。

        Returns:
            Tuple of (avg_bandwidth_gbps, avg_latency_us)
        """
        if topology.bandwidth_matrix is None or topology.distance_matrix is None:
            # Use default values
            return 100.0, 5.0

        # Compute average of non-zero, non-diagonal elements
        n = topology.bandwidth_matrix.shape[0]
        if n <= 1:
            return 100.0, 5.0

        # Bandwidth
        bw_sum = 0.0
        bw_count = 0
        for i in range(n):
            for j in range(n):
                if i != j and topology.bandwidth_matrix[i, j] > 0:
                    bw_sum += topology.bandwidth_matrix[i, j].item()
                    bw_count += 1

        avg_bw = bw_sum / bw_count if bw_count > 0 else 100.0

        # Latency
        lat_sum = 0.0
        lat_count = 0
        for i in range(n):
            for j in range(n):
                if i != j and topology.distance_matrix[i, j] < float('inf'):
                    lat_sum += topology.distance_matrix[i, j].item()
                    lat_count += 1

        avg_lat = lat_sum / lat_count if lat_count > 0 else 5.0

        return avg_bw, avg_lat

    def _get_hierarchical_bandwidth(
        self,
        topology: GPUTopology
    ) -> Tuple[float, float]:
        """
        Get intra-node and inter-node bandwidth.
        获取节点内和节点间带宽。

        Returns:
            Tuple of (intra_node_bw, inter_node_bw)
        """
        # Simplified: assume intra-node is faster
        intra_bw = 300.0  # GB/s (NVLink)
        inter_bw = 12.5   # GB/s (InfiniBand)

        # TODO: Extract from topology
        return intra_bw, inter_bw

    def _estimate_plan_time(
        self,
        plan: CommunicationPlan,
        bandwidth_gbps: float,
        latency_us: float
    ) -> float:
        """
        Estimate total time for a communication plan.
        估计通讯计划的总时间。

        Args:
            plan: Communication plan
            bandwidth_gbps: Average bandwidth
            latency_us: Average latency

        Returns:
            Estimated time in microseconds
        """
        # Simplified: sum up all step times
        total_time = 0.0

        for step in plan.steps:
            step_time = estimate_communication_time(
                step.message_size, bandwidth_gbps, latency_us
            )
            total_time += step_time

        # Account for potential parallelism
        # (in reality, many steps can run in parallel)
        parallelism_factor = 0.5
        total_time *= parallelism_factor

        return total_time

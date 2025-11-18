"""
Compute-Communication Overlap Scheduler
计算-通讯重叠调度器

This module provides scheduling for overlapping computation and communication
to hide communication latency.

本模块提供重叠计算和通讯的调度以隐藏通讯延迟。
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
import torch

from .types import (
    ScheduleNode,
    SchedulePlan,
    OperationType,
)
from .utils import estimate_communication_time

logger = logging.getLogger(__name__)


class DependencyGraph:
    """
    Dependency graph for compute and communication operations.
    计算和通讯操作的依赖图。

    This class represents a DAG (Directed Acyclic Graph) where nodes are
    operations and edges represent dependencies.

    Attributes:
        nodes: Dictionary mapping node ID to ScheduleNode
        edges: Dictionary mapping node ID to set of dependent node IDs
    """

    def __init__(self):
        self.nodes: Dict[int, ScheduleNode] = {}
        self.edges: Dict[int, Set[int]] = defaultdict(set)
        self._next_node_id: int = 0

    def add_compute_node(
        self,
        name: str,
        duration_us: float,
        dependencies: Optional[List[int]] = None
    ) -> int:
        """
        Add a computation node.
        添加计算节点。

        Args:
            name: Human-readable name
            duration_us: Estimated duration in microseconds
            dependencies: List of node IDs that must complete first

        Returns:
            Node ID
        """
        node_id = self._next_node_id
        self._next_node_id += 1

        node = ScheduleNode(
            node_id=node_id,
            op_type=OperationType.COMPUTE,
            name=name,
            duration_us=duration_us,
            dependencies=set(dependencies) if dependencies else set(),
        )

        self.nodes[node_id] = node

        # Add edges
        if dependencies:
            for dep_id in dependencies:
                self.edges[dep_id].add(node_id)

        return node_id

    def add_comm_node(
        self,
        name: str,
        duration_us: float,
        tensor_size: int,
        dependencies: Optional[List[int]] = None
    ) -> int:
        """
        Add a communication node.
        添加通讯节点。

        Args:
            name: Human-readable name
            duration_us: Estimated duration in microseconds
            tensor_size: Size of tensor being communicated
            dependencies: List of node IDs that must complete first

        Returns:
            Node ID
        """
        node_id = self._next_node_id
        self._next_node_id += 1

        node = ScheduleNode(
            node_id=node_id,
            op_type=OperationType.COMMUNICATION,
            name=name,
            duration_us=duration_us,
            dependencies=set(dependencies) if dependencies else set(),
            tensor_size=tensor_size,
        )

        self.nodes[node_id] = node

        # Add edges
        if dependencies:
            for dep_id in dependencies:
                self.edges[dep_id].add(node_id)

        return node_id

    def add_dependency(self, from_node: int, to_node: int) -> None:
        """
        Add a dependency edge.
        添加依赖边。

        Args:
            from_node: Source node ID (must complete first)
            to_node: Target node ID (depends on source)
        """
        if from_node not in self.nodes:
            raise ValueError(f"Source node {from_node} not found")
        if to_node not in self.nodes:
            raise ValueError(f"Target node {to_node} not found")

        self.nodes[to_node].add_dependency(from_node)
        self.edges[from_node].add(to_node)

    def remove_dependency(self, from_node: int, to_node: int) -> None:
        """
        Remove a dependency edge.
        移除依赖边。

        Args:
            from_node: Source node ID
            to_node: Target node ID
        """
        if to_node in self.nodes:
            self.nodes[to_node].remove_dependency(from_node)

        if from_node in self.edges:
            self.edges[from_node].discard(to_node)

    def topological_sort(self) -> List[int]:
        """
        Perform topological sort on the graph.
        对图进行拓扑排序。

        Returns:
            List of node IDs in topological order

        Raises:
            ValueError: If graph has cycles
        """
        # Kahn's algorithm
        in_degree = {node_id: len(node.dependencies)
                     for node_id, node in self.nodes.items()}

        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            # Reduce in-degree of neighbors
            for neighbor_id in self.edges[node_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(result) != len(self.nodes):
            raise ValueError("Graph has cycles")

        return result

    def compute_critical_path(self) -> Tuple[List[int], float]:
        """
        Compute the critical path (longest path) in the graph.
        计算图中的关键路径（最长路径）。

        Returns:
            Tuple of (path_node_ids, path_length_us)
        """
        # Topological sort
        topo_order = self.topological_sort()

        # Compute longest path to each node
        longest_path: Dict[int, float] = {}
        predecessor: Dict[int, Optional[int]] = {}

        for node_id in topo_order:
            node = self.nodes[node_id]

            if not node.dependencies:
                # Source node
                longest_path[node_id] = node.duration_us
                predecessor[node_id] = None
            else:
                # Find max path from predecessors
                max_path = 0.0
                max_pred = None

                for dep_id in node.dependencies:
                    path_len = longest_path[dep_id]
                    if path_len > max_path:
                        max_path = path_len
                        max_pred = dep_id

                longest_path[node_id] = max_path + node.duration_us
                predecessor[node_id] = max_pred

        # Find node with longest path
        max_node = max(longest_path, key=longest_path.get)
        path_length = longest_path[max_node]

        # Reconstruct path
        path = []
        current = max_node
        while current is not None:
            path.append(current)
            current = predecessor[current]

        path.reverse()

        return path, path_length

    def get_ready_nodes(self, completed: Set[int]) -> List[int]:
        """
        Get nodes that are ready to execute (all dependencies completed).
        获取准备执行的节点（所有依赖已完成）。

        Args:
            completed: Set of completed node IDs

        Returns:
            List of ready node IDs
        """
        ready = []
        for node_id, node in self.nodes.items():
            if node_id not in completed:
                if node.dependencies.issubset(completed):
                    ready.append(node_id)
        return ready


class GradientBucketing:
    """
    Gradient bucketing for efficient all-reduce.
    梯度分桶以实现高效的all-reduce。

    Groups gradients into buckets for coalesced communication.

    Attributes:
        bucket_size_mb: Target bucket size in MB
        buckets: List of buckets, each containing tensor sizes
    """

    def __init__(self, bucket_size_mb: float = 25.0):
        """
        Initialize gradient bucketing.
        初始化梯度分桶。

        Args:
            bucket_size_mb: Target bucket size in MB
        """
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.buckets: List[List[Tuple[str, int]]] = []

    def create_buckets(
        self,
        gradients: List[Tuple[str, int]]
    ) -> List[List[Tuple[str, int]]]:
        """
        Create buckets from gradient list.
        从梯度列表创建分桶。

        Args:
            gradients: List of (name, size_bytes) tuples

        Returns:
            List of buckets, each is a list of (name, size) tuples
        """
        self.buckets = []

        # Sort gradients by size (largest first)
        # This helps with better packing
        sorted_grads = sorted(gradients, key=lambda x: x[1], reverse=True)

        current_bucket = []
        current_size = 0

        for name, size in sorted_grads:
            if current_size + size > self.bucket_size_bytes and current_bucket:
                # Start new bucket
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append((name, size))
            current_size += size

        # Add last bucket
        if current_bucket:
            self.buckets.append(current_bucket)

        logger.info(
            f"Created {len(self.buckets)} gradient buckets "
            f"(target size: {self.bucket_size_mb} MB)"
        )

        return self.buckets

    def optimize_bucket_size(
        self,
        gradients: List[Tuple[str, int]],
        bandwidth_gbps: float,
        compute_time_us: float
    ) -> float:
        """
        Optimize bucket size to maximize overlap.
        优化分桶大小以最大化重叠。

        Args:
            gradients: List of (name, size_bytes) tuples
            bandwidth_gbps: Network bandwidth in GB/s
            compute_time_us: Compute time for one gradient

        Returns:
            Optimal bucket size in MB
        """
        total_size = sum(size for _, size in gradients)
        num_gradients = len(gradients)

        # Communication time per gradient (average)
        avg_grad_size = total_size / num_gradients if num_gradients > 0 else 0
        comm_time_per_grad = estimate_communication_time(
            avg_grad_size, bandwidth_gbps, 5.0  # Assume 5us latency
        )

        # Optimal bucket size balances:
        # - Larger buckets: better bandwidth utilization
        # - Smaller buckets: more opportunities for overlap

        # Heuristic: bucket size should allow communication time ≈ compute time
        optimal_size_bytes = int(
            (compute_time_us / comm_time_per_grad) * avg_grad_size
        )

        # Clamp to reasonable range
        min_size = 1 * 1024 * 1024  # 1 MB
        max_size = 256 * 1024 * 1024  # 256 MB

        optimal_size_bytes = max(min_size, min(optimal_size_bytes, max_size))

        optimal_size_mb = optimal_size_bytes / (1024 * 1024)

        logger.debug(
            f"Optimized bucket size: {optimal_size_mb:.2f} MB "
            f"(compute: {compute_time_us:.2f} us, "
            f"comm: {comm_time_per_grad:.2f} us)"
        )

        return optimal_size_mb


class OverlapScheduler:
    """
    Scheduler for overlapping computation and communication.
    重叠计算和通讯的调度器。

    This scheduler analyzes the dependency graph and schedules operations
    to maximize overlap between computation and communication.

    Attributes:
        dependency_graph: Dependency graph of operations
        bucketing: Gradient bucketing manager
    """

    def __init__(self, bucket_size_mb: float = 25.0):
        """
        Initialize overlap scheduler.
        初始化重叠调度器。

        Args:
            bucket_size_mb: Gradient bucket size in MB
        """
        self.dependency_graph = DependencyGraph()
        self.bucketing = GradientBucketing(bucket_size_mb)

    def analyze_overlap_opportunities(
        self,
        forward_graph: DependencyGraph,
        backward_graph: DependencyGraph
    ) -> Dict[str, float]:
        """
        Analyze overlap opportunities in forward and backward passes.
        分析前向和反向传播中的重叠机会。

        Args:
            forward_graph: Dependency graph for forward pass
            backward_graph: Dependency graph for backward pass

        Returns:
            Dictionary with overlap analysis metrics
        """
        metrics = {}

        # Analyze forward pass
        forward_compute_time = 0.0
        forward_comm_time = 0.0

        for node in forward_graph.nodes.values():
            if node.op_type == OperationType.COMPUTE:
                forward_compute_time += node.duration_us
            elif node.op_type == OperationType.COMMUNICATION:
                forward_comm_time += node.duration_us

        # Analyze backward pass
        backward_compute_time = 0.0
        backward_comm_time = 0.0

        for node in backward_graph.nodes.values():
            if node.op_type == OperationType.COMPUTE:
                backward_compute_time += node.duration_us
            elif node.op_type == OperationType.COMMUNICATION:
                backward_comm_time += node.duration_us

        # Compute critical paths
        _, forward_critical_path_time = forward_graph.compute_critical_path()
        _, backward_critical_path_time = backward_graph.compute_critical_path()

        # Potential overlap (difference between sum and critical path)
        forward_overlap = (
            (forward_compute_time + forward_comm_time) - forward_critical_path_time
        )
        backward_overlap = (
            (backward_compute_time + backward_comm_time) - backward_critical_path_time
        )

        metrics['forward_compute_time'] = forward_compute_time
        metrics['forward_comm_time'] = forward_comm_time
        metrics['forward_critical_path'] = forward_critical_path_time
        metrics['forward_overlap_potential'] = forward_overlap

        metrics['backward_compute_time'] = backward_compute_time
        metrics['backward_comm_time'] = backward_comm_time
        metrics['backward_critical_path'] = backward_critical_path_time
        metrics['backward_overlap_potential'] = backward_overlap

        # Overall overlap efficiency
        total_compute = forward_compute_time + backward_compute_time
        total_comm = forward_comm_time + backward_comm_time
        total_critical = forward_critical_path_time + backward_critical_path_time

        if total_comm > 0:
            overlap_efficiency = (
                ((total_compute + total_comm) - total_critical) / total_comm
            )
        else:
            overlap_efficiency = 0.0

        metrics['overlap_efficiency'] = overlap_efficiency

        return metrics

    def schedule_async_communications(
        self,
        comm_ops: List[Tuple[str, int, float]],
        compute_ops: List[Tuple[str, float]]
    ) -> SchedulePlan:
        """
        Schedule asynchronous communications with computations.
        调度异步通讯与计算。

        Args:
            comm_ops: List of (name, tensor_size, duration_us) for communications
            compute_ops: List of (name, duration_us) for computations

        Returns:
            Schedule plan with overlapped operations
        """
        graph = DependencyGraph()

        # Add compute nodes
        compute_nodes = []
        for i, (name, duration) in enumerate(compute_ops):
            deps = [compute_nodes[-1]] if i > 0 else None
            node_id = graph.add_compute_node(name, duration, deps)
            compute_nodes.append(node_id)

        # Add communication nodes
        # Try to overlap each comm with compute
        comm_nodes = []
        for i, (name, tensor_size, duration) in enumerate(comm_ops):
            # Communication can start after corresponding compute
            # (e.g., gradient is ready)
            if i < len(compute_nodes):
                deps = [compute_nodes[i]]
            else:
                deps = None

            node_id = graph.add_comm_node(name, duration, tensor_size, deps)
            comm_nodes.append(node_id)

        # Create schedule plan
        plan = SchedulePlan()

        for node_id in graph.topological_sort():
            plan.nodes.append(graph.nodes[node_id])

        # Compute metrics
        critical_path, critical_path_time = graph.compute_critical_path()
        plan.critical_path_us = critical_path_time

        plan.compute_time_us = sum(
            n.duration_us for n in plan.nodes
            if n.op_type == OperationType.COMPUTE
        )
        plan.comm_time_us = sum(
            n.duration_us for n in plan.nodes
            if n.op_type == OperationType.COMMUNICATION
        )

        plan.overlap_time_us = (
            (plan.compute_time_us + plan.comm_time_us) - plan.critical_path_us
        )

        # Average parallelism
        if plan.critical_path_us > 0:
            plan.parallelism_factor = (
                (plan.compute_time_us + plan.comm_time_us) / plan.critical_path_us
            )
        else:
            plan.parallelism_factor = 1.0

        return plan

    def optimize_bucket_size(
        self,
        gradients: List[Tuple[str, int]],
        bandwidth_gbps: float
    ) -> float:
        """
        Optimize gradient bucket size.
        优化梯度分桶大小。

        Args:
            gradients: List of (name, size_bytes) tuples
            bandwidth_gbps: Network bandwidth in GB/s

        Returns:
            Optimal bucket size in MB
        """
        # Estimate average compute time per gradient
        # (simplified: assume 100 us)
        avg_compute_time = 100.0

        return self.bucketing.optimize_bucket_size(
            gradients, bandwidth_gbps, avg_compute_time
        )

    def pipeline_communications(
        self,
        stages: List[Tuple[str, int, float]]
    ) -> SchedulePlan:
        """
        Create a pipelined schedule for multi-stage communications.
        为多阶段通讯创建流水线调度。

        Args:
            stages: List of (name, tensor_size, duration_us) for each stage

        Returns:
            Pipelined schedule plan
        """
        graph = DependencyGraph()

        # Create pipeline stages
        # Each stage can start when previous stage's first chunk is done
        stage_nodes = []

        for i, (name, tensor_size, duration) in enumerate(stages):
            if i == 0:
                # First stage has no dependencies
                node_id = graph.add_comm_node(name, duration, tensor_size, None)
            else:
                # Subsequent stages depend on previous stage
                # But can start before previous stage fully completes (pipelining)
                deps = [stage_nodes[i - 1]]
                node_id = graph.add_comm_node(name, duration, tensor_size, deps)

            stage_nodes.append(node_id)

        # Create schedule plan
        plan = SchedulePlan()

        for node_id in graph.topological_sort():
            plan.nodes.append(graph.nodes[node_id])

        # Compute metrics
        critical_path, critical_path_time = graph.compute_critical_path()
        plan.critical_path_us = critical_path_time

        plan.comm_time_us = sum(n.duration_us for n in plan.nodes)
        plan.compute_time_us = 0.0

        # With pipelining, overlap is the difference
        plan.overlap_time_us = plan.comm_time_us - plan.critical_path_us

        if plan.critical_path_us > 0:
            plan.parallelism_factor = plan.comm_time_us / plan.critical_path_us
        else:
            plan.parallelism_factor = 1.0

        return plan

    def create_backward_schedule(
        self,
        layer_compute_times: List[float],
        gradient_sizes: List[int],
        bandwidth_gbps: float,
        latency_us: float
    ) -> SchedulePlan:
        """
        Create optimized schedule for backward pass with gradient all-reduce.
        为反向传播创建优化的调度，包含梯度all-reduce。

        Args:
            layer_compute_times: Compute time for each layer's backward pass
            gradient_sizes: Gradient tensor size for each layer
            bandwidth_gbps: Network bandwidth
            latency_us: Network latency

        Returns:
            Optimized schedule plan
        """
        graph = DependencyGraph()

        num_layers = len(layer_compute_times)
        compute_nodes = []
        comm_nodes = []

        # Backward pass goes in reverse order
        for i in range(num_layers):
            # Compute node for this layer's backward
            compute_name = f"backward_layer_{i}"
            compute_duration = layer_compute_times[i]

            # Dependencies: previous layer's compute (if not first)
            compute_deps = [compute_nodes[-1]] if i > 0 else None

            compute_id = graph.add_compute_node(
                compute_name, compute_duration, compute_deps
            )
            compute_nodes.append(compute_id)

            # Communication node for gradient all-reduce
            comm_name = f"allreduce_layer_{i}"
            comm_duration = estimate_communication_time(
                gradient_sizes[i], bandwidth_gbps, latency_us
            )

            # Communication can start as soon as gradient is ready
            comm_deps = [compute_id]

            comm_id = graph.add_comm_node(
                comm_name, comm_duration, gradient_sizes[i], comm_deps
            )
            comm_nodes.append(comm_id)

        # Create schedule plan
        plan = SchedulePlan()

        for node_id in graph.topological_sort():
            plan.nodes.append(graph.nodes[node_id])

        # Compute metrics
        critical_path, critical_path_time = graph.compute_critical_path()
        plan.critical_path_us = critical_path_time

        plan.compute_time_us = sum(layer_compute_times)
        plan.comm_time_us = sum(
            estimate_communication_time(size, bandwidth_gbps, latency_us)
            for size in gradient_sizes
        )

        plan.overlap_time_us = (
            (plan.compute_time_us + plan.comm_time_us) - plan.critical_path_us
        )

        if plan.critical_path_us > 0:
            plan.parallelism_factor = (
                (plan.compute_time_us + plan.comm_time_us) / plan.critical_path_us
            )
        else:
            plan.parallelism_factor = 1.0

        logger.info(
            f"Backward schedule: compute={plan.compute_time_us:.2f}us, "
            f"comm={plan.comm_time_us:.2f}us, "
            f"critical_path={plan.critical_path_us:.2f}us, "
            f"overlap={plan.overlap_time_us:.2f}us "
            f"(efficiency={plan.efficiency:.2%})"
        )

        return plan

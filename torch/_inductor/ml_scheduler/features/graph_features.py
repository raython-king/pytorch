"""
Graph feature extraction from IR graphs.

Extract global features from the entire IR graph, including topology,
workload characteristics, and resource utilization.
"""

import torch
import math
from typing import List, Set, Dict, Optional
from collections import defaultdict, deque
import logging

log = logging.getLogger(__name__)


class GraphFeatureExtractor:
    """
    Extract global features from the entire IR graph.

    Extracts 32-dimensional feature vectors representing graph-level
    properties and characteristics.

    Features include:
    - Graph size (nodes, edges, fused nodes)
    - Memory statistics (peak memory, total allocations)
    - Workload statistics (total FLOPs, reduction ratio)
    - Device distribution
    - Parallelism metrics (max parallel nodes, critical path length)
    - Graph topology metrics (diameter, clustering coefficient)

    Output: 32-dimensional feature vector
    """

    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim

    def extract_features(self, nodes: List) -> torch.Tensor:
        """
        Extract comprehensive features from an IR graph.

        Args:
            nodes: List of BaseSchedulerNode instances

        Returns:
            Feature tensor [feature_dim]

        Example:
            >>> extractor = GraphFeatureExtractor()
            >>> nodes = [node1, node2, node3]
            >>> features = extractor.extract_features(nodes)
            >>> assert features.shape == (32,)
        """
        features = []

        try:
            # 1. Graph structure features (8 dims)
            structure_features = self._get_structure_features(nodes)
            features.extend(structure_features)

            # 2. Computational workload features (8 dims)
            workload_features = self._get_workload_features(nodes)
            features.extend(workload_features)

            # 3. Memory features (8 dims)
            memory_features = self._get_memory_features(nodes)
            features.extend(memory_features)

            # 4. Parallelism and scheduling features (8 dims)
            parallelism_features = self._get_parallelism_features(nodes)
            features.extend(parallelism_features)

            # Ensure we have exactly feature_dim features
            features = features[:self.feature_dim]
            while len(features) < self.feature_dim:
                features.append(0.0)

            return torch.tensor(features, dtype=torch.float32)

        except Exception as e:
            log.warning(f"Error extracting graph features: {e}, returning zeros")
            return torch.zeros(self.feature_dim, dtype=torch.float32)

    def _get_structure_features(self, nodes: List) -> list:
        """
        Extract graph topology features (8 dims).

        Features:
        - num_nodes: Total number of nodes
        - num_edges: Total number of edges
        - num_fused_nodes: Number of fused nodes
        - avg_degree: Average node degree
        - graph_density: Edge density
        - max_degree: Maximum node degree
        - num_roots: Number of nodes with no dependencies
        - num_leaves: Number of nodes with no users
        """
        features = []

        # Number of nodes
        num_nodes = len(nodes)
        features.append(math.log1p(num_nodes))

        if num_nodes == 0:
            features.extend([0.0] * 7)
            return features

        # Count edges and compute degrees
        num_edges = 0
        degrees = []
        num_fused = 0
        num_roots = 0
        num_leaves = 0

        for node in nodes:
            # Count dependencies (incoming edges)
            if hasattr(node, 'unmet_dependencies'):
                node_in_degree = len(node.unmet_dependencies)
            else:
                node_in_degree = 0

            # Count users (outgoing edges)
            if hasattr(node, 'users'):
                node_out_degree = len(node.users)
            else:
                node_out_degree = 0

            degree = node_in_degree + node_out_degree
            degrees.append(degree)
            num_edges += node_in_degree

            # Count fused nodes
            if 'Fused' in type(node).__name__:
                num_fused += 1

            # Count roots (no dependencies)
            if node_in_degree == 0:
                num_roots += 1

            # Count leaves (no users)
            if node_out_degree == 0:
                num_leaves += 1

        # Number of edges
        features.append(math.log1p(num_edges))

        # Number of fused nodes
        features.append(math.log1p(num_fused))

        # Average degree
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
        features.append(avg_degree)

        # Graph density (actual edges / possible edges)
        max_edges = num_nodes * (num_nodes - 1)  # Directed graph
        density = num_edges / max_edges if max_edges > 0 else 0.0
        features.append(density)

        # Maximum degree
        max_degree = max(degrees) if degrees else 0.0
        features.append(float(max_degree))

        # Number of roots
        features.append(math.log1p(num_roots))

        # Number of leaves
        features.append(math.log1p(num_leaves))

        return features

    def _get_workload_features(self, nodes: List) -> list:
        """
        Extract computational workload features (8 dims).

        Features:
        - total_flops: Total FLOPs across all nodes (log scale)
        - avg_flops_per_node: Average FLOPs per node (log scale)
        - reduction_ratio: Ratio of reduction operations
        - pointwise_ratio: Ratio of pointwise operations
        - matmul_ratio: Ratio of matmul operations
        - extern_kernel_ratio: Ratio of extern kernels
        - num_outputs: Total number of outputs
        - op_type_diversity: Diversity of operation types
        """
        features = []

        if not nodes:
            features.extend([0.0] * 8)
            return features

        total_flops = 0.0
        num_reduction = 0
        num_pointwise = 0
        num_matmul = 0
        num_extern = 0
        num_outputs = 0
        op_types = set()

        for node in nodes:
            # Estimate FLOPs
            if hasattr(node, 'estimate_flops'):
                try:
                    flops = node.estimate_flops()
                    if flops:
                        total_flops += flops
                except Exception:
                    pass

            # Count operation types
            node_type = type(node).__name__
            op_types.add(node_type)

            # Check if reduction
            if hasattr(node, 'is_reduction'):
                try:
                    if node.is_reduction():
                        num_reduction += 1
                except Exception:
                    pass

            # Check operation type
            if 'matmul' in node_type.lower() or 'mm' in node_type.lower():
                num_matmul += 1
            elif 'Extern' in node_type:
                num_extern += 1

            # Check for pointwise (heuristic)
            if hasattr(node, 'group'):
                try:
                    group = node.group
                    if group and isinstance(group, tuple) and len(group) >= 2:
                        device, group_info = group[0], group[1]
                        if isinstance(group_info, tuple) and len(group_info) == 1:
                            num_pointwise += 1
                except Exception:
                    pass

            # Count outputs
            if hasattr(node, 'get_outputs'):
                try:
                    num_outputs += len(node.get_outputs())
                except Exception:
                    pass

        num_nodes = len(nodes)

        # Total FLOPs (log scale)
        features.append(math.log1p(total_flops))

        # Average FLOPs per node
        avg_flops = total_flops / num_nodes if num_nodes > 0 else 0.0
        features.append(math.log1p(avg_flops))

        # Reduction ratio
        reduction_ratio = num_reduction / num_nodes if num_nodes > 0 else 0.0
        features.append(reduction_ratio)

        # Pointwise ratio
        pointwise_ratio = num_pointwise / num_nodes if num_nodes > 0 else 0.0
        features.append(pointwise_ratio)

        # Matmul ratio
        matmul_ratio = num_matmul / num_nodes if num_nodes > 0 else 0.0
        features.append(matmul_ratio)

        # Extern kernel ratio
        extern_ratio = num_extern / num_nodes if num_nodes > 0 else 0.0
        features.append(extern_ratio)

        # Number of outputs
        features.append(math.log1p(num_outputs))

        # Operation type diversity (entropy-like measure)
        if len(op_types) > 0:
            diversity = len(op_types) / num_nodes if num_nodes > 0 else 0.0
            features.append(diversity)
        else:
            features.append(0.0)

        return features

    def _get_memory_features(self, nodes: List) -> list:
        """
        Extract memory-related features (8 dims).

        Features:
        - total_read_bytes: Total bytes read (log scale)
        - total_write_bytes: Total bytes written (log scale)
        - avg_read_bytes: Average read bytes per node (log scale)
        - avg_write_bytes: Average write bytes per node (log scale)
        - peak_memory_estimate: Estimated peak memory (log scale)
        - num_unique_buffers: Number of unique buffers accessed
        - memory_reuse_ratio: Buffer reuse ratio
        - has_inplace_ops: Has in-place operations
        """
        features = []

        if not nodes:
            features.extend([0.0] * 8)
            return features

        total_read_bytes = 0.0
        total_write_bytes = 0.0
        buffer_read_counts = defaultdict(int)
        buffer_write_counts = defaultdict(int)
        has_inplace = False

        for node in nodes:
            if hasattr(node, 'read_writes'):
                try:
                    # Read dependencies
                    for dep in node.read_writes.reads:
                        try:
                            bytes_read = dep.numbytes_hint()
                            total_read_bytes += bytes_read
                            buffer_read_counts[dep.name] += 1
                        except Exception:
                            pass

                    # Write dependencies
                    for dep in node.read_writes.writes:
                        try:
                            bytes_written = dep.numbytes_hint()
                            total_write_bytes += bytes_written
                            buffer_write_counts[dep.name] += 1
                        except Exception:
                            pass
                except Exception:
                    pass

            # Check for in-place operations
            if hasattr(node, 'is_inplace') or 'inplace' in type(node).__name__.lower():
                has_inplace = True

        num_nodes = len(nodes)

        # Total read bytes
        features.append(math.log1p(total_read_bytes))

        # Total write bytes
        features.append(math.log1p(total_write_bytes))

        # Average read bytes per node
        avg_read = total_read_bytes / num_nodes if num_nodes > 0 else 0.0
        features.append(math.log1p(avg_read))

        # Average write bytes per node
        avg_write = total_write_bytes / num_nodes if num_nodes > 0 else 0.0
        features.append(math.log1p(avg_write))

        # Estimated peak memory (simple estimate: sum of all allocations)
        # In practice, this would need liveness analysis
        peak_memory_estimate = total_read_bytes + total_write_bytes
        features.append(math.log1p(peak_memory_estimate))

        # Number of unique buffers
        unique_buffers = set(buffer_read_counts.keys()) | set(buffer_write_counts.keys())
        features.append(math.log1p(len(unique_buffers)))

        # Memory reuse ratio (how often buffers are accessed)
        if unique_buffers:
            total_accesses = sum(buffer_read_counts.values()) + sum(buffer_write_counts.values())
            reuse_ratio = total_accesses / len(unique_buffers)
            features.append(math.log1p(reuse_ratio))
        else:
            features.append(0.0)

        # Has in-place operations
        features.append(1.0 if has_inplace else 0.0)

        return features

    def _get_parallelism_features(self, nodes: List) -> list:
        """
        Extract parallelism and scheduling features (8 dims).

        Features:
        - critical_path_length: Length of critical path
        - max_parallel_nodes: Maximum number of nodes that can run in parallel
        - avg_depth: Average depth of nodes
        - max_depth: Maximum depth
        - width_variance: Variance in graph width across levels
        - num_devices: Number of unique devices
        - cross_device_edges: Number of cross-device dependencies
        - scheduling_flexibility: Average scheduling slack
        """
        features = []

        if not nodes:
            features.extend([0.0] * 8)
            return features

        # Build dependency graph and compute depths
        node_to_depth = {}
        node_to_index = {id(node): i for i, node in enumerate(nodes)}

        # Compute depths using topological order
        depths = self._compute_node_depths(nodes)

        if depths:
            avg_depth = sum(depths.values()) / len(depths)
            max_depth = max(depths.values())
            critical_path_length = max_depth
        else:
            avg_depth = 0.0
            max_depth = 0.0
            critical_path_length = 0.0

        # Critical path length
        features.append(math.log1p(critical_path_length))

        # Maximum parallelism (width at each level)
        level_widths = defaultdict(int)
        for depth in depths.values():
            level_widths[depth] += 1

        max_parallel = max(level_widths.values()) if level_widths else 0
        features.append(math.log1p(max_parallel))

        # Average depth
        features.append(math.log1p(avg_depth))

        # Maximum depth
        features.append(math.log1p(max_depth))

        # Width variance
        if level_widths:
            widths = list(level_widths.values())
            mean_width = sum(widths) / len(widths)
            variance = sum((w - mean_width) ** 2 for w in widths) / len(widths)
            features.append(math.log1p(variance))
        else:
            features.append(0.0)

        # Device distribution
        devices = set()
        for node in nodes:
            if hasattr(node, 'get_device'):
                try:
                    device = node.get_device()
                    if device is not None:
                        devices.add(str(device))
                except Exception:
                    pass

        num_devices = len(devices)
        features.append(float(num_devices))

        # Cross-device edges
        cross_device_count = 0
        for node in nodes:
            if hasattr(node, 'get_device') and hasattr(node, 'unmet_dependencies'):
                try:
                    node_device = str(node.get_device())
                    for dep_node in self._get_dependency_nodes(node, nodes):
                        if hasattr(dep_node, 'get_device'):
                            dep_device = str(dep_node.get_device())
                            if node_device != dep_device:
                                cross_device_count += 1
                except Exception:
                    pass

        features.append(math.log1p(cross_device_count))

        # Scheduling flexibility (average slack)
        # Slack = max_order - min_order if available
        slack_values = []
        for node in nodes:
            if hasattr(node, 'min_order') and hasattr(node, 'max_order'):
                try:
                    min_order = node.min_order if node.min_order is not None else 0
                    max_order = node.max_order if node.max_order is not None else float('inf')
                    if max_order != float('inf'):
                        slack = max_order - min_order
                        slack_values.append(slack)
                except Exception:
                    pass

        if slack_values:
            avg_slack = sum(slack_values) / len(slack_values)
            features.append(math.log1p(avg_slack))
        else:
            features.append(0.0)

        return features

    def _compute_node_depths(self, nodes: List) -> Dict:
        """
        Compute depth of each node in the graph using BFS.

        Args:
            nodes: List of scheduler nodes

        Returns:
            Dictionary mapping node id to depth
        """
        depths = {}
        node_map = {id(node): node for node in nodes}

        # Find root nodes (no dependencies)
        roots = []
        for node in nodes:
            if hasattr(node, 'unmet_dependencies'):
                if len(node.unmet_dependencies) == 0:
                    roots.append(node)
            else:
                roots.append(node)

        # BFS to compute depths
        queue = deque([(node, 0) for node in roots])
        visited = set()

        while queue:
            node, depth = queue.popleft()
            node_id = id(node)

            if node_id in visited:
                continue

            visited.add(node_id)
            depths[node_id] = depth

            # Add users (children) to queue
            if hasattr(node, 'users'):
                for user in node.users:
                    if id(user) not in visited:
                        queue.append((user, depth + 1))

        return depths

    def _get_dependency_nodes(self, node, all_nodes: List) -> List:
        """
        Get list of nodes that this node depends on.

        Args:
            node: Scheduler node
            all_nodes: All nodes in the graph

        Returns:
            List of dependency nodes
        """
        dep_nodes = []

        if not hasattr(node, 'unmet_dependencies'):
            return dep_nodes

        # Build name to node mapping
        name_to_node = {}
        for n in all_nodes:
            if hasattr(n, 'get_name'):
                try:
                    name_to_node[n.get_name()] = n
                except Exception:
                    pass

        # Find dependency nodes by name
        for dep in node.unmet_dependencies:
            if hasattr(dep, 'name') and dep.name in name_to_node:
                dep_nodes.append(name_to_node[dep.name])

        return dep_nodes

    def extract_batch_features(self, node_lists: List[List]) -> torch.Tensor:
        """
        Extract features for a batch of graphs.

        Args:
            node_lists: List of node lists (one per graph)

        Returns:
            Feature tensor [num_graphs, feature_dim]

        Example:
            >>> extractor = GraphFeatureExtractor()
            >>> graphs = [[node1, node2], [node3, node4, node5]]
            >>> features = extractor.extract_batch_features(graphs)
            >>> assert features.shape == (2, 32)
        """
        if not node_lists:
            return torch.zeros((0, self.feature_dim), dtype=torch.float32)

        feature_list = []
        for nodes in node_lists:
            features = self.extract_features(nodes)
            feature_list.append(features)

        return torch.stack(feature_list, dim=0)

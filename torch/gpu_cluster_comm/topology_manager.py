"""
GPU Cluster Topology Manager
GPU集群拓扑管理器

This module provides topology discovery, modeling, and management for GPU clusters.

本模块提供GPU集群的拓扑发现、建模和管理。
"""

import os
import re
import subprocess
import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import torch

from .types import (
    GPUDevice,
    TopologyNode,
    Link,
    InterconnectType,
    CollectiveAlgorithm,
)
from .utils import (
    compute_distance_matrix,
    compute_bandwidth_matrix,
    find_nvlink_domains,
    group_devices_by_node,
    build_binary_tree,
    compute_optimal_ring_order,
)

logger = logging.getLogger(__name__)


class GPUTopology:
    """
    GPU cluster topology representation.
    GPU集群拓扑表示。

    This class models the hierarchical structure of a GPU cluster, including:
    - Intra-node topology (NVLink, PCIe)
    - Inter-node topology (InfiniBand, Ethernet)
    - Bandwidth and latency characteristics

    Attributes:
        devices: List of all GPU devices
        nodes: Dictionary mapping node ID to TopologyNode
        links: List of communication links
        distance_matrix: Pairwise latency matrix (microseconds)
        bandwidth_matrix: Pairwise bandwidth matrix (GB/s)
        nvlink_domains: Mapping of NVLink domain ID to set of ranks
    """

    def __init__(self):
        self.devices: List[GPUDevice] = []
        self.nodes: Dict[int, TopologyNode] = {}
        self.links: List[Link] = []
        self.distance_matrix: Optional[torch.Tensor] = None
        self.bandwidth_matrix: Optional[torch.Tensor] = None
        self.nvlink_domains: Dict[int, Set[int]] = {}

    def add_device(self, device: GPUDevice) -> None:
        """
        Add a GPU device to the topology.
        向拓扑中添加GPU设备。

        Args:
            device: GPU device to add
        """
        self.devices.append(device)

        # Add to node
        if device.node_id not in self.nodes:
            self.nodes[device.node_id] = TopologyNode(node_id=device.node_id)
        self.nodes[device.node_id].gpus.append(device)

    def add_link(self, link: Link) -> None:
        """
        Add a communication link to the topology.
        向拓扑中添加通讯链路。

        Args:
            link: Communication link to add
        """
        self.links.append(link)

    def build_matrices(self) -> None:
        """
        Build distance and bandwidth matrices.
        构建距离和带宽矩阵。
        """
        if not self.devices:
            logger.warning("No devices in topology, cannot build matrices")
            return

        self.distance_matrix = compute_distance_matrix(self.devices, self.links)
        self.bandwidth_matrix = compute_bandwidth_matrix(self.devices, self.links)
        self.nvlink_domains = find_nvlink_domains(self.devices)

        logger.info(
            f"Built topology matrices for {len(self.devices)} devices, "
            f"{len(self.links)} links"
        )

    def get_bandwidth(self, rank1: int, rank2: int) -> float:
        """
        Get bandwidth between two ranks.
        获取两个rank之间的带宽。

        Args:
            rank1: First rank
            rank2: Second rank

        Returns:
            Bandwidth in GB/s
        """
        if self.bandwidth_matrix is None:
            raise RuntimeError("Bandwidth matrix not built")

        rank_to_idx = {dev.rank: i for i, dev in enumerate(self.devices)}
        i = rank_to_idx.get(rank1)
        j = rank_to_idx.get(rank2)

        if i is None or j is None:
            raise ValueError(f"Invalid ranks: {rank1}, {rank2}")

        return self.bandwidth_matrix[i, j].item()

    def get_latency(self, rank1: int, rank2: int) -> float:
        """
        Get latency between two ranks.
        获取两个rank之间的延迟。

        Args:
            rank1: First rank
            rank2: Second rank

        Returns:
            Latency in microseconds
        """
        if self.distance_matrix is None:
            raise RuntimeError("Distance matrix not built")

        rank_to_idx = {dev.rank: i for i, dev in enumerate(self.devices)}
        i = rank_to_idx.get(rank1)
        j = rank_to_idx.get(rank2)

        if i is None or j is None:
            raise ValueError(f"Invalid ranks: {rank1}, {rank2}")

        return self.distance_matrix[i, j].item()

    def are_nvlink_connected(self, rank1: int, rank2: int) -> bool:
        """
        Check if two ranks are connected via NVLink.
        检查两个rank是否通过NVLink连接。

        Args:
            rank1: First rank
            rank2: Second rank

        Returns:
            True if connected via NVLink
        """
        for domain_ranks in self.nvlink_domains.values():
            if rank1 in domain_ranks and rank2 in domain_ranks:
                return True
        return False

    def get_intra_node_ranks(self, node_id: int) -> List[int]:
        """
        Get all ranks within a node.
        获取节点内的所有rank。

        Args:
            node_id: Node identifier

        Returns:
            List of ranks in the node
        """
        if node_id not in self.nodes:
            return []
        return [dev.rank for dev in self.nodes[node_id].gpus]

    def get_num_nodes(self) -> int:
        """Get number of nodes in the cluster"""
        return len(self.nodes)

    def get_num_devices(self) -> int:
        """Get total number of devices"""
        return len(self.devices)

    def __repr__(self) -> str:
        return (
            f"GPUTopology(devices={len(self.devices)}, "
            f"nodes={len(self.nodes)}, "
            f"links={len(self.links)})"
        )


class TopologyDiscovery:
    """
    Automatic topology discovery.
    自动拓扑发现。

    Discovers GPU topology using:
    - CUDA device properties
    - nvidia-smi
    - NCCL/RCCL topology detection
    - System information
    """

    @staticmethod
    def discover() -> GPUTopology:
        """
        Discover the GPU topology of the current system.
        发现当前系统的GPU拓扑。

        Returns:
            Discovered GPU topology
        """
        topology = GPUTopology()

        try:
            # Discover devices
            devices = TopologyDiscovery._discover_devices()
            for device in devices:
                topology.add_device(device)

            # Discover links
            links = TopologyDiscovery._discover_links(devices)
            for link in links:
                topology.add_link(link)

            # Build matrices
            topology.build_matrices()

            logger.info(f"Discovered topology: {topology}")

        except Exception as e:
            logger.error(f"Failed to discover topology: {e}")
            # Return empty topology on failure
            topology = GPUTopology()

        return topology

    @staticmethod
    def _discover_devices() -> List[GPUDevice]:
        """
        Discover GPU devices in the system.
        发现系统中的GPU设备。

        Returns:
            List of discovered GPU devices
        """
        devices = []

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, no devices discovered")
            return devices

        num_gpus = torch.cuda.device_count()
        logger.info(f"Discovering {num_gpus} GPU devices")

        for device_id in range(num_gpus):
            try:
                # Get device properties
                props = torch.cuda.get_device_properties(device_id)

                # Parse PCIe bus ID
                pcie_bus_id = TopologyDiscovery._get_pcie_bus_id(device_id)

                # Determine node ID (simplified: assume single node for now)
                # In multi-node, this would use hostname or node rank
                node_id = 0

                # Detect NVLink domain
                nvlink_domain = TopologyDiscovery._detect_nvlink_domain(
                    device_id, pcie_bus_id
                )

                device = GPUDevice(
                    rank=device_id,  # Simplified: rank = device_id
                    local_rank=device_id,
                    node_id=node_id,
                    device_id=device_id,
                    pcie_bus_id=pcie_bus_id,
                    compute_capability=(props.major, props.minor),
                    memory_gb=props.total_memory / (1024 ** 3),
                    nvlink_domain=nvlink_domain,
                )

                devices.append(device)
                logger.debug(f"Discovered device: {device}")

            except Exception as e:
                logger.warning(f"Failed to discover device {device_id}: {e}")

        return devices

    @staticmethod
    def _get_pcie_bus_id(device_id: int) -> str:
        """
        Get PCIe bus ID for a GPU device.
        获取GPU设备的PCIe总线ID。

        Args:
            device_id: CUDA device ID

        Returns:
            PCIe bus ID string
        """
        try:
            # Use nvidia-smi to get PCIe bus ID
            result = subprocess.run(
                ['nvidia-smi', '-i', str(device_id), '--query-gpu=pci.bus_id',
                 '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            bus_id = result.stdout.strip()
            return bus_id
        except Exception as e:
            logger.debug(f"Could not get PCIe bus ID for device {device_id}: {e}")
            return f"unknown_{device_id}"

    @staticmethod
    def _detect_nvlink_domain(device_id: int, pcie_bus_id: str) -> Optional[int]:
        """
        Detect NVLink domain for a GPU device.
        检测GPU设备的NVLink域。

        Args:
            device_id: CUDA device ID
            pcie_bus_id: PCIe bus ID

        Returns:
            NVLink domain ID, or None if not detected
        """
        try:
            # Use nvidia-smi to check for NVLink connections
            result = subprocess.run(
                ['nvidia-smi', 'topo', '-m'],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse topology matrix to detect NVLink groups
            # This is a simplified heuristic
            # In practice, would need more sophisticated parsing

            # For now, assume all GPUs on same node are in same NVLink domain
            # if NVLink is supported by the GPU architecture
            props = torch.cuda.get_device_properties(device_id)

            # NVLink is available on Volta (SM70) and newer
            if props.major >= 7:
                return 0  # Single NVLink domain
            else:
                return None  # No NVLink

        except Exception as e:
            logger.debug(f"Could not detect NVLink domain for device {device_id}: {e}")
            return None

    @staticmethod
    def _discover_links(devices: List[GPUDevice]) -> List[Link]:
        """
        Discover communication links between devices.
        发现设备之间的通讯链路。

        Args:
            devices: List of GPU devices

        Returns:
            List of communication links
        """
        links = []

        # Create links between all pairs of devices
        for i, dev1 in enumerate(devices):
            for dev2 in devices[i + 1:]:
                link = TopologyDiscovery._create_link(dev1, dev2)
                if link:
                    links.append(link)

        logger.info(f"Discovered {len(links)} links")
        return links

    @staticmethod
    def _create_link(dev1: GPUDevice, dev2: GPUDevice) -> Optional[Link]:
        """
        Create a link between two devices.
        在两个设备之间创建链路。

        Args:
            dev1: First device
            dev2: Second device

        Returns:
            Communication link, or None if devices cannot communicate
        """
        # Determine interconnect type
        if dev1.node_id == dev2.node_id:
            # Intra-node: NVLink or PCIe
            if (dev1.nvlink_domain is not None and
                    dev1.nvlink_domain == dev2.nvlink_domain):
                interconnect = InterconnectType.NVLINK
                bandwidth_gbps = 300.0  # NVLink bandwidth (simplified)
                latency_us = 1.0  # Low latency
            else:
                interconnect = InterconnectType.PCIE
                bandwidth_gbps = 32.0  # PCIe bandwidth (simplified)
                latency_us = 5.0  # Medium latency
        else:
            # Inter-node: InfiniBand or Ethernet
            # Simplified: assume InfiniBand
            interconnect = InterconnectType.INFINIBAND
            bandwidth_gbps = 12.5  # 100 Gbps InfiniBand
            latency_us = 10.0  # Higher latency

        link = Link(
            source=dev1.rank,
            target=dev2.rank,
            interconnect=interconnect,
            bandwidth_gbps=bandwidth_gbps,
            latency_us=latency_us,
            bidirectional=True,
        )

        return link


class TopologyManager:
    """
    Manager for GPU topology and communication tree construction.
    GPU拓扑和通讯树构建的管理器。

    This class provides high-level APIs for:
    - Topology discovery and management
    - Communication tree construction for different algorithms
    - Bandwidth and latency queries

    Attributes:
        topology: The GPU cluster topology
    """

    def __init__(self, topology: Optional[GPUTopology] = None):
        """
        Initialize topology manager.
        初始化拓扑管理器。

        Args:
            topology: Pre-built topology, or None to auto-discover
        """
        if topology is None:
            self.topology = TopologyDiscovery.discover()
        else:
            self.topology = topology

    def discover_topology(self) -> GPUTopology:
        """
        Discover or refresh the topology.
        发现或刷新拓扑。

        Returns:
            Updated topology
        """
        self.topology = TopologyDiscovery.discover()
        return self.topology

    def build_communication_tree(
        self,
        root: int,
        algorithm: CollectiveAlgorithm
    ) -> Dict[int, Tuple[Optional[int], List[int]]]:
        """
        Build a communication tree for collective operations.
        为集合操作构建通讯树。

        Args:
            root: Root rank
            algorithm: Collective algorithm

        Returns:
            Dictionary mapping rank to (parent, [children])
        """
        num_ranks = self.topology.get_num_devices()

        if algorithm == CollectiveAlgorithm.TREE:
            # Binary tree
            return build_binary_tree(num_ranks, root)

        elif algorithm == CollectiveAlgorithm.DOUBLE_BINARY_TREE:
            # For double binary tree, build two trees
            tree1 = build_binary_tree(num_ranks, root)
            # Second tree with different structure (shifted)
            tree2 = build_binary_tree(num_ranks, (root + num_ranks // 2) % num_ranks)
            # Combine (simplified: just return first tree)
            return tree1

        elif algorithm == CollectiveAlgorithm.HIERARCHICAL:
            # Two-level hierarchical tree
            return self._build_hierarchical_tree(root)

        else:
            # Default: binary tree
            return build_binary_tree(num_ranks, root)

    def _build_hierarchical_tree(
        self,
        root: int
    ) -> Dict[int, Tuple[Optional[int], List[int]]]:
        """
        Build a two-level hierarchical tree.
        构建两级分层树。

        First level: inter-node tree
        Second level: intra-node trees

        Args:
            root: Root rank

        Returns:
            Dictionary mapping rank to (parent, [children])
        """
        tree: Dict[int, Tuple[Optional[int], List[int]]] = {}

        # Group devices by node
        node_map = group_devices_by_node(self.topology.devices)

        # Find root's node
        root_node = None
        for dev in self.topology.devices:
            if dev.rank == root:
                root_node = dev.node_id
                break

        if root_node is None:
            raise ValueError(f"Root rank {root} not found")

        # Build inter-node tree (one representative per node)
        node_representatives = {}
        for node_id, devices in node_map.items():
            # Use first device as representative
            node_representatives[node_id] = devices[0].rank

        # Build tree among representatives
        num_nodes = len(node_representatives)
        node_ids = sorted(node_representatives.keys())

        # Reorder so root's node is first
        if root_node in node_ids:
            node_ids.remove(root_node)
            node_ids.insert(0, root_node)

        node_tree = build_binary_tree(num_nodes, 0)

        # Build full tree
        for rank_idx, dev in enumerate(self.topology.devices):
            node_id = dev.node_id
            node_idx = node_ids.index(node_id)

            # Get node's parent and children in inter-node tree
            node_parent_idx, node_children_indices = node_tree[node_idx]

            # Map back to actual ranks
            parent_rank = None
            if node_parent_idx is not None:
                parent_node_id = node_ids[node_parent_idx]
                parent_rank = node_representatives[parent_node_id]

            children_ranks = []
            for child_idx in node_children_indices:
                child_node_id = node_ids[child_idx]
                child_rank = node_representatives[child_node_id]
                children_ranks.append(child_rank)

            # Add intra-node connections
            # Within each node, use a simple linear chain
            node_ranks = [d.rank for d in node_map[node_id]]
            node_ranks.sort()

            rank_pos = node_ranks.index(dev.rank)

            if rank_pos > 0:
                # Not first in node, parent is previous rank
                if parent_rank is None:
                    parent_rank = node_ranks[rank_pos - 1]
            else:
                # First in node, already has inter-node parent
                pass

            if rank_pos < len(node_ranks) - 1:
                # Not last in node, child is next rank
                children_ranks.append(node_ranks[rank_pos + 1])

            tree[dev.rank] = (parent_rank, children_ranks)

        return tree

    def compute_optimal_ring_order(self) -> List[int]:
        """
        Compute optimal ring order for ring-based algorithms.
        计算基于环的算法的最佳环序。

        Returns:
            List of ranks in optimal ring order
        """
        if self.topology.bandwidth_matrix is None:
            self.topology.build_matrices()

        return compute_optimal_ring_order(self.topology.bandwidth_matrix, start_rank=0)

    def get_bandwidth_between(self, rank1: int, rank2: int) -> float:
        """
        Get bandwidth between two ranks.
        获取两个rank之间的带宽。

        Args:
            rank1: First rank
            rank2: Second rank

        Returns:
            Bandwidth in GB/s
        """
        return self.topology.get_bandwidth(rank1, rank2)

    def get_latency_between(self, rank1: int, rank2: int) -> float:
        """
        Get latency between two ranks.
        获取两个rank之间的延迟。

        Args:
            rank1: First rank
            rank2: Second rank

        Returns:
            Latency in microseconds
        """
        return self.topology.get_latency(rank1, rank2)

    def detect_nvlink_domains(self) -> Dict[int, Set[int]]:
        """
        Detect NVLink domains in the topology.
        检测拓扑中的NVLink域。

        Returns:
            Dictionary mapping domain ID to set of ranks
        """
        return self.topology.nvlink_domains

    def get_intra_node_groups(self) -> List[List[int]]:
        """
        Get groups of ranks within the same node.
        获取同一节点内的rank组。

        Returns:
            List of rank groups (one per node)
        """
        node_map = group_devices_by_node(self.topology.devices)
        groups = []

        for node_id in sorted(node_map.keys()):
            ranks = [dev.rank for dev in node_map[node_id]]
            ranks.sort()
            groups.append(ranks)

        return groups

    def get_inter_node_groups(self) -> Tuple[List[int], List[int]]:
        """
        Get inter-node communication groups.
        获取节点间通讯组。

        Returns:
            Tuple of (leader_ranks, all_ranks)
            where leader_ranks are one representative per node
        """
        node_map = group_devices_by_node(self.topology.devices)
        leader_ranks = []
        all_ranks = []

        for node_id in sorted(node_map.keys()):
            ranks = [dev.rank for dev in node_map[node_id]]
            ranks.sort()
            leader_ranks.append(ranks[0])  # First rank as leader
            all_ranks.extend(ranks)

        return leader_ranks, all_ranks

    def __repr__(self) -> str:
        return f"TopologyManager(topology={self.topology})"

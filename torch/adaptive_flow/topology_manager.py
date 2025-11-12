"""
Network Topology Manager for Adaptive Flow Control.

This module provides comprehensive network topology discovery, monitoring,
and management for GPU/device interconnects including PCIe, NVLink, and network fabrics.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import torch

logger = logging.getLogger(__name__)


class LinkType(Enum):
    """Types of interconnect links."""
    PCIE = "pcie"
    NVLINK = "nvlink"
    INFINIBAND = "infiniband"
    ETHERNET = "ethernet"
    ROCE = "roce"
    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"


class LinkStatus(Enum):
    """Status of a network link."""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    CONGESTED = "congested"
    RECOVERING = "recovering"


@dataclass
class LinkState:
    """
    Track the state of a network link between devices.

    Attributes:
        src_device: Source device ID
        dst_device: Destination device ID
        link_type: Type of interconnect
        bandwidth_gbps: Maximum bandwidth in Gbps
        latency_us: Base latency in microseconds
        available_bandwidth: Currently available bandwidth (Gbps)
        packet_loss_rate: Packet loss rate (0.0-1.0)
        congestion_level: Congestion level (0.0-1.0)
        status: Current link status
        last_updated: Timestamp of last update
        error_count: Number of errors detected
        utilization_history: Recent utilization samples
    """
    src_device: int
    dst_device: int
    link_type: LinkType
    bandwidth_gbps: float
    latency_us: float
    available_bandwidth: float = 0.0
    packet_loss_rate: float = 0.0
    congestion_level: float = 0.0
    status: LinkStatus = LinkStatus.UP
    last_updated: float = field(default_factory=time.time)
    error_count: int = 0
    utilization_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Initialize available bandwidth."""
        if self.available_bandwidth == 0.0:
            self.available_bandwidth = self.bandwidth_gbps

    def update_metrics(
        self,
        utilization: float,
        packet_loss: Optional[float] = None,
        latency: Optional[float] = None
    ) -> None:
        """
        Update link metrics.

        Args:
            utilization: Current utilization (0.0-1.0)
            packet_loss: Packet loss rate if available
            latency: Current latency if available
        """
        self.utilization_history.append(utilization)
        if len(self.utilization_history) > 100:
            self.utilization_history.pop(0)

        self.available_bandwidth = self.bandwidth_gbps * (1.0 - utilization)

        if packet_loss is not None:
            self.packet_loss_rate = packet_loss

        if latency is not None:
            self.latency_us = latency

        # Update congestion level based on utilization
        if utilization > 0.9:
            self.congestion_level = 1.0
            self.status = LinkStatus.CONGESTED
        elif utilization > 0.7:
            self.congestion_level = utilization
            self.status = LinkStatus.DEGRADED
        else:
            self.congestion_level = 0.0
            self.status = LinkStatus.UP

        self.last_updated = time.time()

    def is_healthy(self) -> bool:
        """Check if link is healthy."""
        return (
            self.status in (LinkStatus.UP, LinkStatus.DEGRADED) and
            self.packet_loss_rate < 0.01 and
            self.error_count < 10
        )

    def get_effective_bandwidth(self) -> float:
        """Get effective bandwidth considering congestion and packet loss."""
        return self.available_bandwidth * (1.0 - self.packet_loss_rate)

    def get_avg_utilization(self, window: int = 10) -> float:
        """Get average utilization over recent window."""
        if not self.utilization_history:
            return 0.0
        recent = self.utilization_history[-window:]
        return sum(recent) / len(recent)


@dataclass
class DeviceInfo:
    """Information about a device in the topology."""
    device_id: int
    device_type: str  # "gpu", "cpu", "switch", etc.
    pcie_bus_id: Optional[str] = None
    numa_node: Optional[int] = None
    compute_capability: Optional[Tuple[int, int]] = None
    memory_gb: Optional[float] = None
    nv_link_count: int = 0
    is_local: bool = True
    hostname: Optional[str] = None


class NetworkTopology:
    """
    Represents the network topology of GPU/device interconnects.

    This class maintains a graph representation of the network topology,
    including devices and links between them.
    """

    def __init__(self):
        """Initialize network topology."""
        self.devices: Dict[int, DeviceInfo] = {}
        self.links: Dict[Tuple[int, int], LinkState] = {}
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)
        self._lock = threading.RLock()
        self._topology_version = 0

        logger.info("NetworkTopology initialized")

    def add_device(self, device_info: DeviceInfo) -> None:
        """
        Add a device to the topology.

        Args:
            device_info: Device information
        """
        with self._lock:
            self.devices[device_info.device_id] = device_info
            if device_info.device_id not in self.adjacency:
                self.adjacency[device_info.device_id] = set()
            self._topology_version += 1
            logger.debug(f"Added device {device_info.device_id} to topology")

    def add_link(self, link_state: LinkState) -> None:
        """
        Add a link between devices.

        Args:
            link_state: Link state information
        """
        with self._lock:
            src, dst = link_state.src_device, link_state.dst_device
            self.links[(src, dst)] = link_state
            self.adjacency[src].add(dst)
            self._topology_version += 1
            logger.debug(f"Added link {src} -> {dst} ({link_state.link_type.value})")

    def remove_link(self, src_device: int, dst_device: int) -> None:
        """Remove a link from topology."""
        with self._lock:
            if (src_device, dst_device) in self.links:
                del self.links[(src_device, dst_device)]
                self.adjacency[src_device].discard(dst_device)
                self._topology_version += 1
                logger.warning(f"Removed link {src_device} -> {dst_device}")

    def get_link(self, src: int, dst: int) -> Optional[LinkState]:
        """Get link state between two devices."""
        with self._lock:
            return self.links.get((src, dst))

    def get_neighbors(self, device_id: int) -> Set[int]:
        """Get neighboring devices."""
        with self._lock:
            return self.adjacency.get(device_id, set()).copy()

    def get_all_devices(self) -> List[int]:
        """Get all device IDs."""
        with self._lock:
            return list(self.devices.keys())

    def get_device_info(self, device_id: int) -> Optional[DeviceInfo]:
        """Get device information."""
        with self._lock:
            return self.devices.get(device_id)

    def update_link_state(
        self,
        src: int,
        dst: int,
        utilization: float,
        packet_loss: Optional[float] = None,
        latency: Optional[float] = None
    ) -> None:
        """Update link state metrics."""
        with self._lock:
            link = self.links.get((src, dst))
            if link:
                link.update_metrics(utilization, packet_loss, latency)

    def mark_link_down(self, src: int, dst: int) -> None:
        """Mark a link as down."""
        with self._lock:
            link = self.links.get((src, dst))
            if link:
                link.status = LinkStatus.DOWN
                link.available_bandwidth = 0.0
                logger.warning(f"Link {src} -> {dst} marked as DOWN")

    def get_topology_version(self) -> int:
        """Get current topology version (increments on changes)."""
        with self._lock:
            return self._topology_version

    def get_link_cost(self, src: int, dst: int, metric: str = "latency") -> float:
        """
        Get link cost for routing algorithms.

        Args:
            src: Source device
            dst: Destination device
            metric: Cost metric ("latency", "bandwidth", "hops")

        Returns:
            Link cost (lower is better)
        """
        link = self.get_link(src, dst)
        if not link or link.status == LinkStatus.DOWN:
            return float('inf')

        if metric == "latency":
            return link.latency_us
        elif metric == "bandwidth":
            # Inverse of available bandwidth (lower cost = higher bandwidth)
            bw = link.get_effective_bandwidth()
            return 1.0 / bw if bw > 0 else float('inf')
        elif metric == "hops":
            return 1.0
        elif metric == "congestion":
            return link.congestion_level + 1.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def export_graph(self) -> Dict[str, Any]:
        """Export topology as a graph dictionary."""
        with self._lock:
            return {
                "devices": {
                    dev_id: {
                        "type": info.device_type,
                        "pcie_bus_id": info.pcie_bus_id,
                        "numa_node": info.numa_node,
                        "is_local": info.is_local,
                    }
                    for dev_id, info in self.devices.items()
                },
                "links": {
                    f"{src}->{dst}": {
                        "type": link.link_type.value,
                        "bandwidth_gbps": link.bandwidth_gbps,
                        "latency_us": link.latency_us,
                        "available_bw": link.available_bandwidth,
                        "status": link.status.value,
                    }
                    for (src, dst), link in self.links.items()
                }
            }


class PathFinder:
    """
    Find optimal paths between devices in the network topology.

    Supports multiple algorithms:
    - Dijkstra's algorithm for shortest path
    - Widest path algorithm
    - K-shortest paths for multipath routing
    """

    def __init__(self, topology: NetworkTopology):
        """
        Initialize PathFinder.

        Args:
            topology: Network topology
        """
        self.topology = topology

    def find_shortest_path(
        self,
        src: int,
        dst: int,
        metric: str = "latency"
    ) -> Optional[List[int]]:
        """
        Find shortest path using Dijkstra's algorithm.

        Args:
            src: Source device
            dst: Destination device
            metric: Cost metric to optimize

        Returns:
            Path as list of device IDs, or None if no path exists
        """
        if src == dst:
            return [src]

        # Dijkstra's algorithm
        distances = {src: 0.0}
        previous = {}
        unvisited = set(self.topology.get_all_devices())

        while unvisited:
            # Find node with minimum distance
            current = None
            min_dist = float('inf')
            for node in unvisited:
                if node in distances and distances[node] < min_dist:
                    min_dist = distances[node]
                    current = node

            if current is None or current == dst:
                break

            unvisited.remove(current)

            # Update distances to neighbors
            for neighbor in self.topology.get_neighbors(current):
                if neighbor not in unvisited:
                    continue

                cost = self.topology.get_link_cost(current, neighbor, metric)
                if cost == float('inf'):
                    continue

                new_distance = distances[current] + cost
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current

        # Reconstruct path
        if dst not in previous:
            return None

        path = []
        current = dst
        while current != src:
            path.append(current)
            current = previous[current]
        path.append(src)
        path.reverse()

        return path

    def find_widest_path(self, src: int, dst: int) -> Optional[List[int]]:
        """
        Find path with maximum minimum bandwidth (widest path).

        Args:
            src: Source device
            dst: Destination device

        Returns:
            Path as list of device IDs
        """
        if src == dst:
            return [src]

        # Modified Dijkstra for widest path
        bandwidth = {src: float('inf')}
        previous = {}
        unvisited = set(self.topology.get_all_devices())

        while unvisited:
            current = None
            max_bw = 0.0
            for node in unvisited:
                if node in bandwidth and bandwidth[node] > max_bw:
                    max_bw = bandwidth[node]
                    current = node

            if current is None or current == dst:
                break

            unvisited.remove(current)

            for neighbor in self.topology.get_neighbors(current):
                if neighbor not in unvisited:
                    continue

                link = self.topology.get_link(current, neighbor)
                if not link or link.status == LinkStatus.DOWN:
                    continue

                link_bw = link.get_effective_bandwidth()
                path_bw = min(bandwidth[current], link_bw)

                if neighbor not in bandwidth or path_bw > bandwidth[neighbor]:
                    bandwidth[neighbor] = path_bw
                    previous[neighbor] = current

        # Reconstruct path
        if dst not in previous:
            return None

        path = []
        current = dst
        while current != src:
            path.append(current)
            current = previous[current]
        path.append(src)
        path.reverse()

        return path

    def find_k_shortest_paths(
        self,
        src: int,
        dst: int,
        k: int = 3,
        metric: str = "latency"
    ) -> List[List[int]]:
        """
        Find K shortest paths for multipath routing.

        Args:
            src: Source device
            dst: Destination device
            k: Number of paths to find
            metric: Cost metric

        Returns:
            List of paths (each path is a list of device IDs)
        """
        paths = []

        # Simple approach: find shortest path, then find alternatives
        # by temporarily removing links
        first_path = self.find_shortest_path(src, dst, metric)
        if not first_path:
            return []

        paths.append(first_path)

        # Try to find alternative paths by avoiding links in existing paths
        for _ in range(k - 1):
            # This is a simplified version; a full implementation would use
            # Yen's algorithm or similar
            best_alt_path = None
            best_alt_cost = float('inf')

            # Try removing each link in the first path
            for i in range(len(first_path) - 1):
                u, v = first_path[i], first_path[i + 1]
                original_link = self.topology.get_link(u, v)

                if not original_link:
                    continue

                # Temporarily mark link as down
                original_status = original_link.status
                original_link.status = LinkStatus.DOWN

                # Find alternative path
                alt_path = self.find_shortest_path(src, dst, metric)

                # Restore link
                original_link.status = original_status

                if alt_path and alt_path not in paths:
                    # Calculate path cost
                    path_cost = self._calculate_path_cost(alt_path, metric)
                    if path_cost < best_alt_cost:
                        best_alt_cost = path_cost
                        best_alt_path = alt_path

            if best_alt_path:
                paths.append(best_alt_path)
            else:
                break

        return paths

    def _calculate_path_cost(self, path: List[int], metric: str) -> float:
        """Calculate total cost of a path."""
        total_cost = 0.0
        for i in range(len(path) - 1):
            cost = self.topology.get_link_cost(path[i], path[i + 1], metric)
            if cost == float('inf'):
                return float('inf')
            total_cost += cost
        return total_cost


class TopologyMonitor:
    """
    Monitor network topology for changes, failures, and performance degradation.

    Runs periodic checks and triggers callbacks on topology changes.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        monitor_interval: float = 1.0
    ):
        """
        Initialize topology monitor.

        Args:
            topology: Network topology to monitor
            monitor_interval: Monitoring interval in seconds
        """
        self.topology = topology
        self.monitor_interval = monitor_interval
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[callable] = []
        self._last_topology_version = 0

        logger.info(f"TopologyMonitor initialized (interval={monitor_interval}s)")

    def add_callback(self, callback: callable) -> None:
        """
        Add callback for topology changes.

        Args:
            callback: Function to call on topology changes
        """
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start monitoring."""
        if self._running:
            logger.warning("TopologyMonitor already running")
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="TopologyMonitor"
        )
        self._monitor_thread.start()
        logger.info("TopologyMonitor started")

    def stop(self) -> None:
        """Stop monitoring."""
        if not self._running:
            return

        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("TopologyMonitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_topology()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)

    def _check_topology(self) -> None:
        """Check topology for changes and issues."""
        current_version = self.topology.get_topology_version()

        # Check for topology changes
        if current_version != self._last_topology_version:
            logger.info(f"Topology changed (version {current_version})")
            self._last_topology_version = current_version
            self._notify_callbacks("topology_changed", {
                "version": current_version
            })

        # Check link health
        self._check_link_health()

    def _check_link_health(self) -> None:
        """Check health of all links."""
        for (src, dst), link in self.topology.links.items():
            if not link.is_healthy():
                logger.warning(
                    f"Link {src} -> {dst} unhealthy: "
                    f"status={link.status.value}, "
                    f"packet_loss={link.packet_loss_rate:.3f}, "
                    f"errors={link.error_count}"
                )

                # Check if link should be marked down
                if link.error_count > 20 or link.packet_loss_rate > 0.1:
                    self.topology.mark_link_down(src, dst)
                    self._notify_callbacks("link_down", {
                        "src": src,
                        "dst": dst,
                        "link": link
                    })

    def _notify_callbacks(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in callback: {e}", exc_info=True)


class TopologyDiscovery:
    """
    Auto-discover network topology using CUDA APIs and system information.

    Detects:
    - GPU devices and their capabilities
    - PCIe topology
    - NVLink connections
    - NUMA topology
    - Network interfaces
    """

    @staticmethod
    def discover_topology() -> NetworkTopology:
        """
        Discover network topology.

        Returns:
            NetworkTopology with discovered devices and links
        """
        topology = NetworkTopology()

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, creating minimal topology")
            return topology

        try:
            # Discover GPU devices
            device_count = torch.cuda.device_count()
            logger.info(f"Discovered {device_count} GPU devices")

            for device_id in range(device_count):
                device_info = TopologyDiscovery._get_device_info(device_id)
                topology.add_device(device_info)

            # Discover links between devices
            TopologyDiscovery._discover_links(topology, device_count)

        except Exception as e:
            logger.error(f"Error during topology discovery: {e}", exc_info=True)

        return topology

    @staticmethod
    def _get_device_info(device_id: int) -> DeviceInfo:
        """Get information about a GPU device."""
        props = torch.cuda.get_device_properties(device_id)

        return DeviceInfo(
            device_id=device_id,
            device_type="gpu",
            pcie_bus_id=None,  # Would extract from CUDA APIs
            numa_node=None,
            compute_capability=(props.major, props.minor),
            memory_gb=props.total_memory / (1024 ** 3),
            nv_link_count=0,  # Would detect from CUDA
            is_local=True,
            hostname=None
        )

    @staticmethod
    def _discover_links(topology: NetworkTopology, device_count: int) -> None:
        """Discover links between devices."""
        for src in range(device_count):
            for dst in range(device_count):
                if src == dst:
                    continue

                # Check if devices can access each other
                try:
                    can_access = torch.cuda.can_device_access_peer(src, dst)
                    if can_access:
                        # Determine link type and characteristics
                        link_type, bandwidth, latency = TopologyDiscovery._probe_link(src, dst)

                        link = LinkState(
                            src_device=src,
                            dst_device=dst,
                            link_type=link_type,
                            bandwidth_gbps=bandwidth,
                            latency_us=latency
                        )
                        topology.add_link(link)

                except Exception as e:
                    logger.debug(f"Cannot check peer access {src} -> {dst}: {e}")

    @staticmethod
    def _probe_link(src: int, dst: int) -> Tuple[LinkType, float, float]:
        """
        Probe link characteristics.

        Returns:
            (link_type, bandwidth_gbps, latency_us)
        """
        # This would use CUDA APIs to determine actual link type
        # For now, use heuristics

        # Assume NVLink if devices can access peer
        # In reality, would check CUDA topology APIs
        link_type = LinkType.NVLINK
        bandwidth_gbps = 50.0  # Typical NVLink bandwidth
        latency_us = 1.0  # Low latency for NVLink

        return link_type, bandwidth_gbps, latency_us


__all__ = [
    "LinkType",
    "LinkStatus",
    "LinkState",
    "DeviceInfo",
    "NetworkTopology",
    "PathFinder",
    "TopologyMonitor",
    "TopologyDiscovery",
]

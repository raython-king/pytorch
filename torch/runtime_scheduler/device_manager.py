"""
Multi-Device Manager for Runtime Scheduling

Manages multiple GPUs/devices with ML-based device selection,
load balancing, and peer-to-peer transfer optimization.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


class DeviceType(Enum):
    """Device type enumeration"""
    CUDA = "cuda"
    CPU = "cpu"
    XPU = "xpu"
    MPS = "mps"


@dataclass
class DeviceCapability:
    """Device capability information"""
    device_id: int
    device_type: DeviceType
    compute_capability: Tuple[int, int]
    total_memory: int
    max_threads_per_block: int
    max_shared_memory: int
    multi_processor_count: int
    supports_p2p: Set[int] = field(default_factory=set)

    @classmethod
    def detect(cls, device: torch.device) -> "DeviceCapability":
        """Detect device capabilities"""
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            supports_p2p = set()

            # Detect P2P support with other devices
            if torch.cuda.device_count() > 1:
                for other_id in range(torch.cuda.device_count()):
                    if other_id != device.index:
                        try:
                            if torch.cuda.can_device_access_peer(device.index, other_id):
                                supports_p2p.add(other_id)
                        except RuntimeError:
                            pass

            return cls(
                device_id=device.index,
                device_type=DeviceType.CUDA,
                compute_capability=(props.major, props.minor),
                total_memory=props.total_memory,
                max_threads_per_block=props.max_threads_per_block,
                max_shared_memory=props.max_shared_memory_per_block,
                multi_processor_count=props.multi_processor_count,
                supports_p2p=supports_p2p,
            )
        else:
            # CPU or other devices
            return cls(
                device_id=0,
                device_type=DeviceType.CPU,
                compute_capability=(0, 0),
                total_memory=0,
                max_threads_per_block=0,
                max_shared_memory=0,
                multi_processor_count=0,
            )


@dataclass
class DeviceStats:
    """Runtime device statistics"""
    utilization: float = 0.0  # 0-1
    memory_used: int = 0
    memory_free: int = 0
    active_streams: int = 0
    queued_ops: int = 0
    completed_ops: int = 0
    avg_op_time: float = 0.0
    last_update: float = field(default_factory=time.time)


class DeviceInfo:
    """Track per-device state"""

    def __init__(self, device: torch.device):
        self.device = device
        self.capability = DeviceCapability.detect(device)
        self.stats = DeviceStats()
        self.lock = threading.RLock()

        # Track active operations and streams
        self.active_ops: Set[int] = set()
        self.streams: List[torch.cuda.Stream] = []

        # History for ML features
        self.utilization_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.op_time_history = deque(maxlen=1000)

        # Performance counters
        self.total_ops = 0
        self.total_time = 0.0

    def update_stats(self):
        """Update device statistics"""
        with self.lock:
            current_time = time.time()

            if self.device.type == "cuda":
                # Get current memory usage
                self.stats.memory_used = torch.cuda.memory_allocated(self.device)
                self.stats.memory_free = (
                    self.capability.total_memory - self.stats.memory_used
                )

                # Estimate utilization based on queued operations
                # In production, use NVML for accurate utilization
                self.stats.utilization = min(
                    1.0,
                    (self.stats.queued_ops + len(self.active_ops)) / 10.0
                )

                self.stats.active_streams = len(self.streams)

            # Update average operation time
            if self.op_time_history:
                self.stats.avg_op_time = sum(self.op_time_history) / len(
                    self.op_time_history
                )

            # Record history
            self.utilization_history.append(self.stats.utilization)
            self.memory_history.append(
                self.stats.memory_used / max(1, self.capability.total_memory)
            )

            self.stats.last_update = current_time

    def record_op_start(self, op_id: int):
        """Record operation start"""
        with self.lock:
            self.active_ops.add(op_id)
            self.stats.queued_ops = max(0, self.stats.queued_ops - 1)

    def record_op_end(self, op_id: int, duration: float):
        """Record operation completion"""
        with self.lock:
            self.active_ops.discard(op_id)
            self.stats.completed_ops += 1
            self.total_ops += 1
            self.total_time += duration
            self.op_time_history.append(duration)

    def get_load(self) -> float:
        """Get current load score (0-1)"""
        with self.lock:
            # Combine utilization and memory pressure
            memory_pressure = self.stats.memory_used / max(
                1, self.capability.total_memory
            )
            return (self.stats.utilization * 0.7 + memory_pressure * 0.3)

    def can_fit_memory(self, size: int) -> bool:
        """Check if device can fit memory allocation"""
        with self.lock:
            # Keep some margin for safety
            available = self.stats.memory_free - (1024 * 1024 * 100)  # 100MB margin
            return size <= available


class DeviceSelector:
    """ML-based device selection"""

    def __init__(self, devices: List[DeviceInfo]):
        self.devices = devices
        self.device_map = {d.device: d for d in devices}

        # Simple heuristic-based selector
        # In production, use ML model from device_models.py
        self.selection_history = deque(maxlen=1000)

    def select_device(
        self,
        op_type: str,
        input_sizes: List[int],
        required_memory: int,
        affinity: Optional[torch.device] = None,
    ) -> torch.device:
        """Select best device for operation"""

        # Filter devices that can fit the operation
        candidates = [
            d for d in self.devices
            if d.can_fit_memory(required_memory)
        ]

        if not candidates:
            # Fallback to device with most free memory
            candidates = sorted(
                self.devices,
                key=lambda d: d.stats.memory_free,
                reverse=True,
            )

        if not candidates:
            # Last resort: use first device
            return self.devices[0].device

        # Prefer affinity device if specified and suitable
        if affinity and affinity in self.device_map:
            affinity_dev = self.device_map[affinity]
            if affinity_dev in candidates:
                if affinity_dev.get_load() < 0.9:  # Not overloaded
                    return affinity

        # Score each candidate
        best_device = None
        best_score = float("inf")

        for device_info in candidates:
            score = self._score_device(
                device_info, op_type, input_sizes, required_memory
            )

            if score < best_score:
                best_score = score
                best_device = device_info.device

        # Record selection for learning
        self.selection_history.append({
            "op_type": op_type,
            "device": best_device,
            "load": self.device_map[best_device].get_load(),
        })

        return best_device or self.devices[0].device

    def _score_device(
        self,
        device_info: DeviceInfo,
        op_type: str,
        input_sizes: List[int],
        required_memory: int,
    ) -> float:
        """Score device for operation (lower is better)"""

        # Load component
        load_score = device_info.get_load() * 100

        # Memory pressure component
        memory_after = device_info.stats.memory_used + required_memory
        memory_pressure = memory_after / max(1, device_info.capability.total_memory)
        memory_score = memory_pressure * 50

        # Queue length component
        queue_score = device_info.stats.queued_ops * 5

        # Historical performance component
        avg_time_score = device_info.stats.avg_op_time * 1000

        return load_score + memory_score + queue_score + avg_time_score


class LoadBalancer:
    """Balance work across devices"""

    def __init__(self, devices: List[DeviceInfo]):
        self.devices = devices
        self.lock = threading.Lock()

        # Load balancing strategies
        self.strategy = "dynamic"  # round_robin, least_loaded, dynamic
        self.round_robin_idx = 0

        # Migration tracking
        self.migration_count = 0
        self.migration_overhead = 0.0

    def balance_load(self) -> List[Tuple[torch.device, torch.device]]:
        """
        Analyze load and return migration suggestions.

        Returns:
            List of (from_device, to_device) migration pairs
        """
        with self.lock:
            migrations = []

            # Calculate average load
            loads = [(d.device, d.get_load()) for d in self.devices]
            if not loads:
                return migrations

            avg_load = sum(load for _, load in loads) / len(loads)

            # Find overloaded and underloaded devices
            overloaded = [(dev, load) for dev, load in loads if load > avg_load + 0.2]
            underloaded = [(dev, load) for dev, load in loads if load < avg_load - 0.2]

            # Suggest migrations from overloaded to underloaded
            for over_dev, over_load in overloaded:
                for under_dev, under_load in underloaded:
                    if over_load - under_load > 0.3:
                        migrations.append((over_dev, under_dev))
                        break

            return migrations

    def should_migrate(self, from_device: torch.device, to_device: torch.device) -> bool:
        """Decide if migration is beneficial"""

        from_info = next((d for d in self.devices if d.device == from_device), None)
        to_info = next((d for d in self.devices if d.device == to_device), None)

        if not from_info or not to_info:
            return False

        # Check load difference
        load_diff = from_info.get_load() - to_info.get_load()

        # Migration is beneficial if load difference is significant
        return load_diff > 0.25

    def get_next_device_round_robin(self) -> torch.device:
        """Get next device using round-robin"""
        with self.lock:
            device = self.devices[self.round_robin_idx].device
            self.round_robin_idx = (self.round_robin_idx + 1) % len(self.devices)
            return device

    def get_least_loaded_device(self) -> torch.device:
        """Get device with least load"""
        with self.lock:
            return min(self.devices, key=lambda d: d.get_load()).device


class DeviceManager:
    """Coordinate all devices"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.lock = threading.RLock()

        # Discover and initialize devices
        self.devices: List[DeviceInfo] = []
        self._discover_devices()

        # Initialize components
        self.selector = DeviceSelector(self.devices)
        self.balancer = LoadBalancer(self.devices)

        # P2P transfer cache
        self.p2p_enabled: Dict[Tuple[int, int], bool] = {}
        self._setup_p2p()

        # Monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _discover_devices(self):
        """Discover available devices"""
        # Add CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                self.devices.append(DeviceInfo(device))

        # Add CPU device
        cpu_device = torch.device("cpu")
        self.devices.append(DeviceInfo(cpu_device))

        if not self.devices:
            raise RuntimeError("No devices available")

    def _setup_p2p(self):
        """Setup peer-to-peer transfers"""
        if torch.cuda.device_count() < 2:
            return

        # Enable P2P access between compatible devices
        for i in range(torch.cuda.device_count()):
            for j in range(torch.cuda.device_count()):
                if i != j:
                    try:
                        if torch.cuda.can_device_access_peer(i, j):
                            # Enable P2P access
                            with torch.cuda.device(i):
                                # P2P is automatically enabled when available
                                self.p2p_enabled[(i, j)] = True
                    except RuntimeError:
                        self.p2p_enabled[(i, j)] = False

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Update all device stats
                for device_info in self.devices:
                    device_info.update_stats()

                # Check for load balancing opportunities
                migrations = self.balancer.balance_load()

                # Log migrations (in production, trigger actual migrations)
                if migrations:
                    pass  # TODO: Implement migration execution

                time.sleep(0.1)  # Update every 100ms

            except Exception as e:
                # Log error but keep monitoring
                print(f"Monitor error: {e}")
                time.sleep(1.0)

    def select_device(
        self,
        op_type: str,
        input_sizes: List[int],
        required_memory: int = 0,
        affinity: Optional[torch.device] = None,
    ) -> torch.device:
        """Select best device for operation"""
        return self.selector.select_device(
            op_type, input_sizes, required_memory, affinity
        )

    def get_device_info(self, device: torch.device) -> Optional[DeviceInfo]:
        """Get device info"""
        with self.lock:
            for dev_info in self.devices:
                if dev_info.device == device:
                    return dev_info
            return None

    def supports_p2p(self, src_device: int, dst_device: int) -> bool:
        """Check if P2P is supported between devices"""
        return self.p2p_enabled.get((src_device, dst_device), False)

    def record_op_start(self, device: torch.device, op_id: int):
        """Record operation start"""
        dev_info = self.get_device_info(device)
        if dev_info:
            dev_info.record_op_start(op_id)

    def record_op_end(self, device: torch.device, op_id: int, duration: float):
        """Record operation completion"""
        dev_info = self.get_device_info(device)
        if dev_info:
            dev_info.record_op_end(op_id, duration)

    def get_stats(self) -> Dict[str, any]:
        """Get overall statistics"""
        with self.lock:
            return {
                "total_devices": len(self.devices),
                "cuda_devices": sum(
                    1 for d in self.devices if d.device.type == "cuda"
                ),
                "devices": [
                    {
                        "device": str(d.device),
                        "load": d.get_load(),
                        "utilization": d.stats.utilization,
                        "memory_used": d.stats.memory_used,
                        "memory_free": d.stats.memory_free,
                        "active_ops": len(d.active_ops),
                        "completed_ops": d.stats.completed_ops,
                    }
                    for d in self.devices
                ],
                "p2p_enabled": len(self.p2p_enabled),
            }

    def shutdown(self):
        """Shutdown device manager"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)


# Global singleton instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get global device manager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager

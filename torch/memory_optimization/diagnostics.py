"""
Hardware Diagnostics and Memory Profiling

Automatically detects hardware capabilities and profiles memory usage
to inform optimization decisions.
"""

import torch
import psutil
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import numpy as np


@dataclass
class HardwareCapabilities:
    """Hardware capability information"""
    # GPU info
    num_gpus: int
    gpu_names: List[str]
    gpu_total_memory: List[float]  # GB
    gpu_compute_capability: List[Tuple[int, int]]
    gpu_memory_bandwidth: List[float]  # GB/s

    # CPU info
    num_cpu_cores: int
    cpu_total_memory: float  # GB
    cpu_memory_bandwidth: float  # GB/s

    # Network info
    has_nvlink: bool
    nvlink_bandwidth: Optional[float]  # GB/s
    has_infiniband: bool
    network_bandwidth: Optional[float]  # GB/s

    # Features
    supports_amp: bool
    supports_bf16: bool
    supports_tensor_cores: bool
    supports_unified_memory: bool


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float

    # GPU memory
    gpu_allocated: List[float]  # GB
    gpu_reserved: List[float]  # GB
    gpu_free: List[float]  # GB
    gpu_utilization: List[float]  # Percentage

    # CPU memory
    cpu_used: float  # GB
    cpu_available: float  # GB
    cpu_percent: float

    # Training metrics
    batch_size: Optional[int] = None
    throughput: Optional[float] = None  # samples/sec
    iteration_time: Optional[float] = None  # seconds


class HardwareDiagnostics:
    """
    Diagnoses hardware capabilities and constraints
    """

    def __init__(self):
        self.capabilities: Optional[HardwareCapabilities] = None
        self._cache_valid = False

    def diagnose(self) -> HardwareCapabilities:
        """Perform comprehensive hardware diagnostics"""
        if self._cache_valid and self.capabilities:
            return self.capabilities

        # Detect GPU capabilities
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        gpu_names = []
        gpu_total_memory = []
        gpu_compute_capability = []
        gpu_memory_bandwidth = []

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            gpu_names.append(props.name)
            gpu_total_memory.append(props.total_memory / 1024**3)  # Convert to GB
            gpu_compute_capability.append((props.major, props.minor))
            # Estimate bandwidth (this is approximate)
            gpu_memory_bandwidth.append(self._estimate_gpu_bandwidth(i))

        # Detect CPU capabilities
        num_cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_memory = psutil.virtual_memory()
        cpu_total_memory = cpu_memory.total / 1024**3  # GB
        cpu_memory_bandwidth = self._estimate_cpu_bandwidth()

        # Detect interconnect
        has_nvlink = self._detect_nvlink()
        nvlink_bandwidth = self._measure_nvlink_bandwidth() if has_nvlink else None
        has_infiniband = self._detect_infiniband()
        network_bandwidth = self._measure_network_bandwidth() if has_infiniband else None

        # Feature detection
        supports_amp = num_gpus > 0 and any(cc[0] >= 7 for cc in gpu_compute_capability)
        supports_bf16 = num_gpus > 0 and any(cc[0] >= 8 for cc in gpu_compute_capability)
        supports_tensor_cores = supports_amp
        supports_unified_memory = num_gpus > 0

        self.capabilities = HardwareCapabilities(
            num_gpus=num_gpus,
            gpu_names=gpu_names,
            gpu_total_memory=gpu_total_memory,
            gpu_compute_capability=gpu_compute_capability,
            gpu_memory_bandwidth=gpu_memory_bandwidth,
            num_cpu_cores=num_cpu_cores,
            cpu_total_memory=cpu_total_memory,
            cpu_memory_bandwidth=cpu_memory_bandwidth,
            has_nvlink=has_nvlink,
            nvlink_bandwidth=nvlink_bandwidth,
            has_infiniband=has_infiniband,
            network_bandwidth=network_bandwidth,
            supports_amp=supports_amp,
            supports_bf16=supports_bf16,
            supports_tensor_cores=supports_tensor_cores,
            supports_unified_memory=supports_unified_memory,
        )

        self._cache_valid = True
        return self.capabilities

    def _estimate_gpu_bandwidth(self, device_id: int) -> float:
        """Estimate GPU memory bandwidth through micro-benchmark"""
        if not torch.cuda.is_available():
            return 0.0

        device = torch.device(f'cuda:{device_id}')
        size = 100 * 1024 * 1024  # 100MB

        try:
            # Warm up
            a = torch.randn(size // 4, device=device)
            b = a.clone()
            torch.cuda.synchronize(device)

            # Measure
            start = time.perf_counter()
            for _ in range(10):
                b.copy_(a)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            bandwidth = (size * 10) / elapsed / 1024**3  # GB/s
            return bandwidth
        except:
            # Fallback to typical values
            props = torch.cuda.get_device_properties(device_id)
            if 'A100' in props.name:
                return 1555.0
            elif 'V100' in props.name:
                return 900.0
            elif 'T4' in props.name:
                return 300.0
            else:
                return 500.0  # Conservative estimate

    def _estimate_cpu_bandwidth(self) -> float:
        """Estimate CPU memory bandwidth"""
        # Typical DDR4 bandwidth is 20-25 GB/s per channel
        # Most systems have 2-4 channels
        return 80.0  # Conservative estimate

    def _detect_nvlink(self) -> bool:
        """Detect if NVLink is available"""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            return False

        try:
            # Try to detect NVLink through device properties
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                # A100, V100 typically have NVLink
                if 'A100' in props.name or 'V100' in props.name:
                    return True
            return False
        except:
            return False

    def _measure_nvlink_bandwidth(self) -> Optional[float]:
        """Measure NVLink bandwidth if available"""
        if not self._detect_nvlink():
            return None

        try:
            size = 100 * 1024 * 1024  # 100MB
            a = torch.randn(size // 4, device='cuda:0')
            b = torch.empty_like(a, device='cuda:1')

            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                b.copy_(a)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            bandwidth = (size * 10) / elapsed / 1024**3
            return bandwidth
        except:
            return 300.0  # NVLink 3.0 typical bandwidth

    def _detect_infiniband(self) -> bool:
        """Detect if InfiniBand is available"""
        try:
            import subprocess
            result = subprocess.run(['ibstat'], capture_output=True, timeout=1)
            return result.returncode == 0
        except:
            return False

    def _measure_network_bandwidth(self) -> Optional[float]:
        """Measure network bandwidth"""
        # This would require actual network transfer tests
        # Return typical InfiniBand bandwidth
        return 100.0  # 100 Gb/s = 12.5 GB/s

    def get_recommended_strategies(self) -> List[str]:
        """Get recommended optimization strategies based on hardware"""
        if not self.capabilities:
            self.diagnose()

        recommendations = []
        caps = self.capabilities

        # Mixed precision if supported
        if caps.supports_amp:
            recommendations.append("mixed_precision")

        # Gradient checkpointing for limited memory
        if caps.num_gpus > 0:
            avg_memory = sum(caps.gpu_total_memory) / len(caps.gpu_total_memory)
            if avg_memory < 16:
                recommendations.append("gradient_checkpointing")
                recommendations.append("activation_checkpointing")

        # CPU offloading if good CPU memory
        if caps.cpu_total_memory > caps.gpu_total_memory[0] * 2:
            recommendations.append("cpu_offloading")

        # Gradient compression for distributed training
        if caps.num_gpus > 1:
            recommendations.append("gradient_compression")
            if caps.has_nvlink:
                recommendations.append("tensor_parallelism")

        return recommendations


class MemoryProfiler:
    """
    Profiles memory usage during training and provides insights
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.snapshots = deque(maxlen=window_size)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def _monitor_loop(self, interval: float):
        """Monitoring loop"""
        while self._monitoring:
            snapshot = self.capture_snapshot()
            with self._lock:
                self.snapshots.append(snapshot)
            time.sleep(interval)

    def capture_snapshot(self) -> MemorySnapshot:
        """Capture current memory state"""
        timestamp = time.time()

        # GPU memory
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_allocated = []
        gpu_reserved = []
        gpu_free = []
        gpu_utilization = []

        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024**3
            free = total - reserved

            gpu_allocated.append(allocated)
            gpu_reserved.append(reserved)
            gpu_free.append(free)
            gpu_utilization.append(reserved / total * 100)

        # CPU memory
        cpu_memory = psutil.virtual_memory()
        cpu_used = cpu_memory.used / 1024**3
        cpu_available = cpu_memory.available / 1024**3
        cpu_percent = cpu_memory.percent

        return MemorySnapshot(
            timestamp=timestamp,
            gpu_allocated=gpu_allocated,
            gpu_reserved=gpu_reserved,
            gpu_free=gpu_free,
            gpu_utilization=gpu_utilization,
            cpu_used=cpu_used,
            cpu_available=cpu_available,
            cpu_percent=cpu_percent,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of memory usage"""
        if not self.snapshots:
            return {}

        with self._lock:
            snapshots = list(self.snapshots)

        if not snapshots:
            return {}

        num_gpus = len(snapshots[0].gpu_allocated)
        stats = {}

        # GPU statistics
        for i in range(num_gpus):
            allocated = [s.gpu_allocated[i] for s in snapshots]
            utilization = [s.gpu_utilization[i] for s in snapshots]

            stats[f'gpu_{i}'] = {
                'allocated_mean': np.mean(allocated),
                'allocated_std': np.std(allocated),
                'allocated_max': np.max(allocated),
                'allocated_min': np.min(allocated),
                'utilization_mean': np.mean(utilization),
                'utilization_max': np.max(utilization),
            }

        # CPU statistics
        cpu_used = [s.cpu_used for s in snapshots]
        cpu_percent = [s.cpu_percent for s in snapshots]

        stats['cpu'] = {
            'used_mean': np.mean(cpu_used),
            'used_std': np.std(cpu_used),
            'used_max': np.max(cpu_used),
            'percent_mean': np.mean(cpu_percent),
            'percent_max': np.max(cpu_percent),
        }

        return stats

    def detect_memory_pressure(self, threshold: float = 0.85) -> bool:
        """Detect if system is under memory pressure"""
        snapshot = self.capture_snapshot()

        # Check GPU memory
        if snapshot.gpu_utilization:
            max_gpu_util = max(snapshot.gpu_utilization)
            if max_gpu_util > threshold * 100:
                return True

        # Check CPU memory
        if snapshot.cpu_percent > threshold * 100:
            return True

        return False

    def predict_oom_risk(self) -> float:
        """Predict risk of out-of-memory error (0-1 scale)"""
        if len(self.snapshots) < 10:
            return 0.0

        with self._lock:
            recent_snapshots = list(self.snapshots)[-10:]

        # Analyze trend in GPU memory usage
        if not recent_snapshots[0].gpu_utilization:
            return 0.0

        gpu_utils = [max(s.gpu_utilization) for s in recent_snapshots]

        # Check if approaching limit
        current_util = gpu_utils[-1]
        if current_util > 95:
            return 0.9
        elif current_util > 90:
            return 0.7
        elif current_util > 85:
            return 0.5

        # Check if increasing rapidly
        if len(gpu_utils) >= 5:
            recent_trend = np.mean(np.diff(gpu_utils[-5:]))
            if recent_trend > 2:  # Increasing by >2% per snapshot
                return min(0.8, current_util / 100)

        return current_util / 100 * 0.3

    def get_memory_breakdown(self) -> Dict[str, float]:
        """Get breakdown of memory usage by category"""
        snapshot = self.capture_snapshot()

        breakdown = {}
        for i, allocated in enumerate(snapshot.gpu_allocated):
            breakdown[f'gpu_{i}_allocated'] = allocated

        breakdown['cpu_used'] = snapshot.cpu_used

        return breakdown

"""
Runtime Feature Extraction for ML-based Scheduling.

This module extracts features from operations and system state for ML models:
- Operation features: compute type, tensor shapes, memory footprint
- System state features: device util, memory pressure, queue depth
- Dependency features: ready dependencies, blocking operations
- Historical features: recent execution times, success rates

All feature extraction is optimized for speed (< 0.1ms per extraction).
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch

from ..state_tracker import RuntimeStateTracker, DeviceType


@dataclass
class OperationFeatures:
    """Features describing a single operation."""
    # Operation identity
    op_id: int
    op_type_hash: int  # Hash of operation type for embedding

    # Tensor features
    num_inputs: int
    num_outputs: int
    total_elements: int
    total_bytes: int
    max_dimension: int

    # Compute features
    estimated_flops: float
    is_compute_bound: bool
    is_memory_bound: bool

    # Device features
    target_device_type: int  # 0=CPU, 1=CUDA, 2=MPS, etc.
    target_device_id: int

    # Cached features for reuse
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemStateFeatures:
    """Features describing current system state."""
    # Device utilization (up to 8 GPUs + 1 CPU)
    compute_util: List[float] = field(default_factory=lambda: [0.0] * 9)
    memory_util: List[float] = field(default_factory=lambda: [0.0] * 9)

    # Memory pressure
    total_memory_used_gb: float = 0.0
    memory_fragmentation: float = 0.0

    # Queue state
    total_queue_depth: int = 0
    queue_depth_per_device: List[int] = field(default_factory=lambda: [0] * 9)

    # Stream state
    active_streams: int = 0

    # Load indicators
    overall_load: float = 0.0  # 0.0 to 1.0

    timestamp: float = field(default_factory=time.time)


@dataclass
class DependencyFeatures:
    """Features describing operation dependencies."""
    # Dependency counts
    num_dependencies: int
    num_ready_dependencies: int
    num_blocking_ops: int

    # Dependency graph features
    dependency_depth: int
    is_on_critical_path: bool

    # Data movement
    requires_device_transfer: bool
    transfer_size_bytes: int

    # Batching potential
    has_batching_potential: bool
    similar_ops_in_queue: int


@dataclass
class HistoricalFeatures:
    """Features from historical execution data."""
    # Recent performance
    avg_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    duration_std_ms: float = 0.0

    # Success metrics
    success_rate: float = 1.0
    recent_failures: int = 0

    # Execution count
    times_executed: int = 0

    # Device affinity
    preferred_device: Optional[str] = None
    device_speedup: float = 1.0  # Speedup vs CPU


class FeatureCache:
    """Thread-safe cache for computed features."""

    def __init__(self, max_size: int = 10000, ttl_ms: float = 100.0):
        """
        Initialize feature cache.

        Args:
            max_size: Maximum number of cached entries
            ttl_ms: Time-to-live for cached entries (milliseconds)
        """
        self.max_size = max_size
        self.ttl_ms = ttl_ms
        self._cache: Dict[int, Tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: int) -> Optional[Any]:
        """Get cached features if still valid."""
        with self._lock:
            if key in self._cache:
                features, timestamp = self._cache[key]
                age_ms = (time.time() - timestamp) * 1000
                if age_ms < self.ttl_ms:
                    self._hits += 1
                    return features
                else:
                    del self._cache[key]

            self._misses += 1
            return None

    def put(self, key: int, features: Any) -> None:
        """Cache features."""
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (features, time.time())

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0
            }


class RuntimeFeatureExtractor:
    """
    Extract features for ML-based runtime scheduling decisions.

    Optimized for speed with caching and incremental computation.
    Target: < 0.1ms per feature extraction.
    """

    def __init__(
        self,
        state_tracker: RuntimeStateTracker,
        enable_caching: bool = True,
        cache_size: int = 10000,
        cache_ttl_ms: float = 100.0
    ):
        """
        Initialize feature extractor.

        Args:
            state_tracker: Runtime state tracker instance
            enable_caching: Whether to enable feature caching
            cache_size: Maximum cache size
            cache_ttl_ms: Cache entry time-to-live (ms)
        """
        self.state_tracker = state_tracker
        self.enable_caching = enable_caching

        # Feature caches
        self._op_feature_cache = FeatureCache(cache_size, cache_ttl_ms)
        self._system_feature_cache = FeatureCache(1, cache_ttl_ms)  # Single entry
        self._historical_cache = FeatureCache(cache_size, cache_ttl_ms * 10)  # Longer TTL

        # Feature normalization parameters (learned online)
        self._feature_stats: Dict[str, Tuple[float, float]] = {}  # mean, std
        self._lock = threading.RLock()

        # Performance tracking
        self._total_extraction_time_ns = 0
        self._extraction_count = 0

    def extract_operation_features(
        self,
        op_id: int,
        op_type: str,
        input_shapes: List[Tuple[int, ...]],
        output_shapes: List[Tuple[int, ...]],
        device: torch.device,
        estimated_flops: Optional[float] = None
    ) -> OperationFeatures:
        """
        Extract features for an operation.

        Args:
            op_id: Unique operation identifier
            op_type: Type of operation (e.g., "matmul", "conv2d")
            input_shapes: List of input tensor shapes
            output_shapes: List of output tensor shapes
            device: Target device
            estimated_flops: Estimated FLOPs (optional)

        Returns:
            OperationFeatures object
        """
        start = time.perf_counter_ns()

        # Check cache
        if self.enable_caching:
            cached = self._op_feature_cache.get(op_id)
            if cached is not None:
                return cached

        # Compute tensor features
        total_elements = sum(math.prod(shape) for shape in input_shapes)
        total_elements += sum(math.prod(shape) for shape in output_shapes)

        # Assume float32 (4 bytes per element)
        total_bytes = total_elements * 4

        max_dim = 0
        for shape in input_shapes + output_shapes:
            if shape:
                max_dim = max(max_dim, max(shape))

        # Estimate FLOPs if not provided
        if estimated_flops is None:
            estimated_flops = self._estimate_flops(op_type, input_shapes)

        # Determine compute vs memory bound
        # Rough heuristic: ops/byte ratio
        ops_per_byte = estimated_flops / max(total_bytes, 1)
        is_compute_bound = ops_per_byte > 10  # Arbitrary threshold
        is_memory_bound = not is_compute_bound

        # Device features
        device_type_map = {"cpu": 0, "cuda": 1, "mps": 2, "xla": 3}
        target_device_type = device_type_map.get(device.type, 0)
        target_device_id = device.index if device.index is not None else 0

        features = OperationFeatures(
            op_id=op_id,
            op_type_hash=hash(op_type) % 10000,  # Hash for embedding
            num_inputs=len(input_shapes),
            num_outputs=len(output_shapes),
            total_elements=total_elements,
            total_bytes=total_bytes,
            max_dimension=max_dim,
            estimated_flops=estimated_flops,
            is_compute_bound=is_compute_bound,
            is_memory_bound=is_memory_bound,
            target_device_type=target_device_type,
            target_device_id=target_device_id
        )

        # Cache result
        if self.enable_caching:
            self._op_feature_cache.put(op_id, features)

        # Track performance
        elapsed = time.perf_counter_ns() - start
        self._total_extraction_time_ns += elapsed
        self._extraction_count += 1

        return features

    def extract_system_state_features(self) -> SystemStateFeatures:
        """
        Extract features describing current system state.

        Returns:
            SystemStateFeatures object
        """
        start = time.perf_counter_ns()

        # Check cache
        if self.enable_caching:
            cached = self._system_feature_cache.get(0)
            if cached is not None:
                return cached

        features = SystemStateFeatures()

        # Get device utilization
        for device_type, device_id in [(DeviceType.CPU, 0)] + \
                [(DeviceType.CUDA, i) for i in range(8)]:
            util = self.state_tracker.get_device_utilization(device_type, device_id)

            if util:
                idx = 0 if device_type == DeviceType.CPU else device_id + 1
                features.compute_util[idx] = util.compute_utilization
                features.memory_util[idx] = util.memory_utilization

        # Get memory state
        total_used = 0.0
        total_free = 0.0
        total_fragmented = 0.0

        for device_id in range(8):
            mem = self.state_tracker.get_memory_state(DeviceType.CUDA, device_id)
            if mem:
                total_used += mem.allocated_bytes
                total_free += mem.free_bytes
                # Fragmentation: cached but not allocated
                total_fragmented += mem.cached_bytes

        features.total_memory_used_gb = total_used / (1024 ** 3)

        # Fragmentation ratio
        total_reserved = total_used + total_fragmented
        if total_reserved > 0:
            features.memory_fragmentation = total_fragmented / total_reserved

        # Get queue metrics
        queue_length, pending_by_device = self.state_tracker.get_queue_metrics()
        features.total_queue_depth = queue_length

        # Map device names to indices
        for device_name, count in pending_by_device.items():
            if device_name.startswith("cuda:"):
                device_id = int(device_name.split(":")[1])
                if 0 <= device_id < 8:
                    features.queue_depth_per_device[device_id + 1] = count
            elif device_name == "cpu":
                features.queue_depth_per_device[0] = count

        # Get stream state
        stream_states = self.state_tracker.get_stream_states()
        features.active_streams = len(stream_states)

        # Compute overall load (weighted average of device utilization)
        utils = [u for u in features.compute_util if u > 0]
        if utils:
            features.overall_load = sum(utils) / len(utils)

        # Cache result
        if self.enable_caching:
            self._system_feature_cache.put(0, features)

        # Track performance
        elapsed = time.perf_counter_ns() - start
        self._total_extraction_time_ns += elapsed
        self._extraction_count += 1

        return features

    def extract_dependency_features(
        self,
        op_id: int,
        num_dependencies: int,
        num_ready: int,
        num_blocking: int,
        dependency_depth: int,
        is_critical_path: bool,
        requires_transfer: bool,
        transfer_bytes: int,
        similar_ops_count: int
    ) -> DependencyFeatures:
        """
        Extract features describing operation dependencies.

        Args:
            op_id: Operation identifier
            num_dependencies: Total number of dependencies
            num_ready: Number of satisfied dependencies
            num_blocking: Number of operations blocked by this one
            dependency_depth: Depth in dependency graph
            is_critical_path: Whether on critical path
            requires_transfer: Whether requires device transfer
            transfer_bytes: Size of data transfer (bytes)
            similar_ops_count: Number of similar ops in queue

        Returns:
            DependencyFeatures object
        """
        start = time.perf_counter_ns()

        # Determine batching potential
        has_batching_potential = (
            similar_ops_count > 0 and
            num_ready == num_dependencies and  # All deps ready
            not requires_transfer  # No device transfer needed
        )

        features = DependencyFeatures(
            num_dependencies=num_dependencies,
            num_ready_dependencies=num_ready,
            num_blocking_ops=num_blocking,
            dependency_depth=dependency_depth,
            is_on_critical_path=is_critical_path,
            requires_device_transfer=requires_transfer,
            transfer_size_bytes=transfer_bytes,
            has_batching_potential=has_batching_potential,
            similar_ops_in_queue=similar_ops_count
        )

        # Track performance
        elapsed = time.perf_counter_ns() - start
        self._total_extraction_time_ns += elapsed
        self._extraction_count += 1

        return features

    def extract_historical_features(
        self,
        op_type: str,
        device: str,
        window_size: int = 100
    ) -> HistoricalFeatures:
        """
        Extract features from historical execution data.

        Args:
            op_type: Type of operation
            device: Device identifier
            window_size: Number of recent ops to analyze

        Returns:
            HistoricalFeatures object
        """
        start = time.perf_counter_ns()

        # Check cache
        cache_key = hash(f"{op_type}:{device}") % 1000000
        if self.enable_caching:
            cached = self._historical_cache.get(cache_key)
            if cached is not None:
                return cached

        # Get operation statistics
        stats = self.state_tracker.get_operation_stats(
            op_type=op_type,
            window_size=window_size
        )

        # Get recent operations for more detailed stats
        recent_ops = self.state_tracker.get_recent_operations(
            count=window_size,
            op_type=op_type,
            device=device
        )

        features = HistoricalFeatures()

        if stats["count"] > 0:
            features.avg_duration_ms = stats["avg_duration_ms"]
            features.min_duration_ms = stats["min_duration_ms"]
            features.max_duration_ms = stats["max_duration_ms"]
            features.success_rate = stats["success_rate"]
            features.times_executed = stats["count"]

            # Compute standard deviation
            durations = [op.duration_ms for op in recent_ops]
            if len(durations) > 1:
                mean = features.avg_duration_ms
                variance = sum((d - mean) ** 2 for d in durations) / len(durations)
                features.duration_std_ms = math.sqrt(variance)

            # Count recent failures
            features.recent_failures = sum(1 for op in recent_ops if not op.success)

        # Determine preferred device (device with best performance)
        features.preferred_device = device
        features.device_speedup = 1.0

        # Compare with CPU performance if on GPU
        if device.startswith("cuda:"):
            cpu_stats = self.state_tracker.get_operation_stats(
                op_type=op_type,
                window_size=window_size
            )
            if cpu_stats["count"] > 0 and cpu_stats["avg_duration_ms"] > 0:
                cpu_avg = cpu_stats["avg_duration_ms"]
                if features.avg_duration_ms > 0:
                    features.device_speedup = cpu_avg / features.avg_duration_ms

        # Cache result
        if self.enable_caching:
            self._historical_cache.put(cache_key, features)

        # Track performance
        elapsed = time.perf_counter_ns() - start
        self._total_extraction_time_ns += elapsed
        self._extraction_count += 1

        return features

    def extract_all_features(
        self,
        op_id: int,
        op_type: str,
        input_shapes: List[Tuple[int, ...]],
        output_shapes: List[Tuple[int, ...]],
        device: torch.device,
        dependency_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract all feature types for an operation.

        Args:
            op_id: Operation identifier
            op_type: Type of operation
            input_shapes: Input tensor shapes
            output_shapes: Output tensor shapes
            device: Target device
            dependency_info: Dictionary with dependency information

        Returns:
            Dictionary containing all feature types
        """
        op_features = self.extract_operation_features(
            op_id, op_type, input_shapes, output_shapes, device
        )

        system_features = self.extract_system_state_features()

        dep_features = self.extract_dependency_features(
            op_id=op_id,
            num_dependencies=dependency_info.get("num_dependencies", 0),
            num_ready=dependency_info.get("num_ready", 0),
            num_blocking=dependency_info.get("num_blocking", 0),
            dependency_depth=dependency_info.get("depth", 0),
            is_critical_path=dependency_info.get("is_critical_path", False),
            requires_transfer=dependency_info.get("requires_transfer", False),
            transfer_bytes=dependency_info.get("transfer_bytes", 0),
            similar_ops_count=dependency_info.get("similar_ops_count", 0)
        )

        hist_features = self.extract_historical_features(
            op_type=op_type,
            device=str(device)
        )

        return {
            "operation": op_features,
            "system": system_features,
            "dependencies": dep_features,
            "historical": hist_features
        }

    def _estimate_flops(
        self,
        op_type: str,
        input_shapes: List[Tuple[int, ...]]
    ) -> float:
        """
        Estimate FLOPs for an operation.

        Args:
            op_type: Type of operation
            input_shapes: Input tensor shapes

        Returns:
            Estimated number of FLOPs
        """
        if not input_shapes:
            return 0.0

        # Simple heuristics for common operations
        if "matmul" in op_type.lower() or "mm" in op_type.lower():
            # Matrix multiply: 2 * M * N * K
            if len(input_shapes) >= 2:
                shape1, shape2 = input_shapes[0], input_shapes[1]
                if len(shape1) >= 2 and len(shape2) >= 2:
                    m, k = shape1[-2], shape1[-1]
                    n = shape2[-1]
                    return 2.0 * m * n * k

        elif "conv" in op_type.lower():
            # Convolution: approximate
            if input_shapes:
                return float(math.prod(input_shapes[0]) * 10)  # Rough estimate

        elif "add" in op_type.lower() or "sub" in op_type.lower():
            # Element-wise operations
            if input_shapes:
                return float(math.prod(input_shapes[0]))

        # Default: proportional to input size
        return float(sum(math.prod(shape) for shape in input_shapes))

    def normalize_features(
        self,
        features: Dict[str, Any],
        update_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Normalize features using running statistics.

        Args:
            features: Dictionary of features
            update_stats: Whether to update normalization statistics

        Returns:
            Normalized features
        """
        # This would be implemented with proper running mean/std tracking
        # For now, return as-is
        return features

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get feature extraction performance metrics.

        Returns:
            Dictionary with performance statistics
        """
        if self._extraction_count == 0:
            return {
                "total_time_ms": 0.0,
                "avg_time_us": 0.0,
                "extraction_count": 0
            }

        return {
            "total_time_ms": self._total_extraction_time_ns / 1_000_000,
            "avg_time_us": (
                self._total_extraction_time_ns / self._extraction_count
            ) / 1000,
            "extraction_count": self._extraction_count,
            "op_cache_stats": self._op_feature_cache.get_stats(),
            "hist_cache_stats": self._historical_cache.get_stats()
        }

    def reset_performance_metrics(self) -> None:
        """Reset performance tracking."""
        self._total_extraction_time_ns = 0
        self._extraction_count = 0

    def clear_caches(self) -> None:
        """Clear all feature caches."""
        self._op_feature_cache.clear()
        self._system_feature_cache.clear()
        self._historical_cache.clear()

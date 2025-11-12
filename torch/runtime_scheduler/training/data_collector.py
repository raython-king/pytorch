"""
Data collector for runtime scheduler training.

Collects execution traces during normal model execution for later training.
"""

import time
import pickle
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings


@dataclass
class ExecutionTrace:
    """A single execution trace for training."""
    timestamp: float
    operation: str
    input_shapes: List[Tuple[int, ...]]
    input_dtypes: List[str]
    device: str
    actual_latency: float
    memory_allocated: int
    memory_cached: int
    device_utilization: float
    concurrent_ops: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionTrace':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SchedulingExample:
    """A training example for scheduling model."""
    features: Dict[str, Any]
    target_device: str
    target_latency: float
    alternative_devices: Dict[str, float]  # device -> latency
    context: Dict[str, Any] = field(default_factory=dict)


class RuntimeDataCollector:
    """
    Collects execution traces for training runtime models.

    Features:
    - Collect traces during normal execution
    - Export to various formats
    - Filtering and preprocessing
    - Multi-threaded collection
    """

    def __init__(
        self,
        max_traces: int = 100000,
        sampling_rate: float = 1.0,
        auto_save: bool = True,
        save_interval: int = 1000,
        save_dir: Optional[str] = None
    ):
        """
        Initialize data collector.

        Args:
            max_traces: Maximum traces to keep in memory
            sampling_rate: Fraction of operations to trace (0.0-1.0)
            auto_save: Automatically save traces periodically
            save_interval: Number of traces between auto-saves
            save_dir: Directory for saving traces
        """
        self.max_traces = max_traces
        self.sampling_rate = sampling_rate
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.save_dir = Path(save_dir) if save_dir else Path("./traces")

        # Trace storage
        self._traces: deque = deque(maxlen=max_traces)

        # Statistics
        self._stats = {
            "total_collected": 0,
            "total_sampled": 0,
            "total_saved": 0,
        }

        # Thread safety
        self._lock = threading.RLock()

        # Collection state
        self._enabled = True
        self._save_counter = 0

        # Create save directory
        if self.auto_save:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def collect(
        self,
        operation: str,
        input_shapes: List[Tuple[int, ...]],
        input_dtypes: List[str],
        device: str,
        actual_latency: float,
        memory_allocated: int = 0,
        memory_cached: int = 0,
        device_utilization: float = 0.0,
        concurrent_ops: int = 0,
        **metadata
    ) -> None:
        """
        Collect an execution trace.

        Args:
            operation: Operation name
            input_shapes: Input tensor shapes
            input_dtypes: Input data types
            device: Device where operation executed
            actual_latency: Measured latency
            memory_allocated: Memory allocated
            memory_cached: Memory cached
            device_utilization: Device utilization
            concurrent_ops: Number of concurrent operations
            **metadata: Additional metadata
        """
        if not self._enabled:
            return

        # Sample based on sampling rate
        import random
        if random.random() > self.sampling_rate:
            with self._lock:
                self._stats["total_sampled"] += 1
            return

        trace = ExecutionTrace(
            timestamp=time.time(),
            operation=operation,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            device=device,
            actual_latency=actual_latency,
            memory_allocated=memory_allocated,
            memory_cached=memory_cached,
            device_utilization=device_utilization,
            concurrent_ops=concurrent_ops,
            metadata=metadata
        )

        with self._lock:
            self._traces.append(trace)
            self._stats["total_collected"] += 1
            self._save_counter += 1

            # Auto-save if needed
            if self.auto_save and self._save_counter >= self.save_interval:
                self._auto_save()
                self._save_counter = 0

    def get_traces(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        device_filter: Optional[str] = None,
        operation_filter: Optional[str] = None
    ) -> List[ExecutionTrace]:
        """
        Get collected traces with optional filtering.

        Args:
            start_idx: Start index
            end_idx: End index (None for all)
            device_filter: Filter by device
            operation_filter: Filter by operation name pattern

        Returns:
            List of execution traces
        """
        with self._lock:
            traces = list(self._traces)[start_idx:end_idx]

        # Apply filters
        if device_filter:
            traces = [t for t in traces if t.device == device_filter]

        if operation_filter:
            traces = [t for t in traces if operation_filter in t.operation]

        return traces

    def create_training_examples(
        self,
        include_alternatives: bool = True
    ) -> List[SchedulingExample]:
        """
        Create training examples from collected traces.

        Args:
            include_alternatives: Include alternative device latencies

        Returns:
            List of training examples
        """
        examples = []

        with self._lock:
            traces = list(self._traces)

        # Group traces by operation signature
        op_groups: Dict[str, List[ExecutionTrace]] = {}
        for trace in traces:
            signature = self._get_operation_signature(trace)
            if signature not in op_groups:
                op_groups[signature] = []
            op_groups[signature].append(trace)

        # Create examples
        for signature, group_traces in op_groups.items():
            if len(group_traces) < 2:
                continue

            # Use each trace as a potential example
            for trace in group_traces:
                features = self._extract_features(trace)

                # Get alternative device latencies if available
                alternatives = {}
                if include_alternatives:
                    for alt_trace in group_traces:
                        if alt_trace.device != trace.device:
                            if alt_trace.device not in alternatives:
                                alternatives[alt_trace.device] = []
                            alternatives[alt_trace.device].append(
                                alt_trace.actual_latency
                            )

                    # Average alternative latencies
                    alternatives = {
                        dev: sum(lats) / len(lats)
                        for dev, lats in alternatives.items()
                    }

                example = SchedulingExample(
                    features=features,
                    target_device=trace.device,
                    target_latency=trace.actual_latency,
                    alternative_devices=alternatives,
                    context={
                        "timestamp": trace.timestamp,
                        "operation": trace.operation,
                    }
                )

                examples.append(example)

        return examples

    def _get_operation_signature(self, trace: ExecutionTrace) -> str:
        """Get a signature for operation type."""
        # Signature: operation + input shapes + dtypes
        shapes_str = "_".join(str(s) for s in trace.input_shapes)
        dtypes_str = "_".join(trace.input_dtypes)
        return f"{trace.operation}_{shapes_str}_{dtypes_str}"

    def _extract_features(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """Extract features from a trace for training."""
        features = {
            "operation": trace.operation,
            "input_shapes": trace.input_shapes,
            "input_dtypes": trace.input_dtypes,
            "memory_allocated": trace.memory_allocated,
            "memory_cached": trace.memory_cached,
            "device_utilization": trace.device_utilization,
            "concurrent_ops": trace.concurrent_ops,
        }

        # Add tensor size features
        if trace.input_shapes:
            features["num_inputs"] = len(trace.input_shapes)
            features["total_elements"] = sum(
                self._compute_num_elements(shape)
                for shape in trace.input_shapes
            )
            features["max_dimension"] = max(
                max(shape) if shape else 0
                for shape in trace.input_shapes
            )

        return features

    @staticmethod
    def _compute_num_elements(shape: Tuple[int, ...]) -> int:
        """Compute number of elements in a tensor."""
        if not shape:
            return 0
        result = 1
        for dim in shape:
            result *= dim
        return result

    def _auto_save(self) -> None:
        """Automatically save traces."""
        try:
            timestamp = int(time.time())
            filename = self.save_dir / f"traces_{timestamp}.pkl"

            with self._lock:
                traces = list(self._traces)

            with open(filename, 'wb') as f:
                pickle.dump(traces, f)

            with self._lock:
                self._stats["total_saved"] += len(traces)

        except Exception as e:
            warnings.warn(f"Error auto-saving traces: {e}")

    def save(self, filename: str, format: str = 'pickle') -> None:
        """
        Save collected traces to file.

        Args:
            filename: Output filename
            format: Format ('pickle', 'json')
        """
        with self._lock:
            traces = list(self._traces)

        if format == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(traces, f)

        elif format == 'json':
            import json
            with open(filename, 'w') as f:
                json.dump([t.to_dict() for t in traces], f, indent=2)

        else:
            raise ValueError(f"Unknown format: {format}")

    def load(self, filename: str, format: str = 'pickle') -> None:
        """
        Load traces from file.

        Args:
            filename: Input filename
            format: Format ('pickle', 'json')
        """
        if format == 'pickle':
            with open(filename, 'rb') as f:
                traces = pickle.load(f)

        elif format == 'json':
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
                traces = [ExecutionTrace.from_dict(d) for d in data]

        else:
            raise ValueError(f"Unknown format: {format}")

        with self._lock:
            self._traces.extend(traces)
            self._stats["total_collected"] += len(traces)

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["traces_in_memory"] = len(self._traces)

        return stats

    def reset(self) -> None:
        """Reset collected traces."""
        with self._lock:
            self._traces.clear()
            self._stats["total_collected"] = 0
            self._stats["total_sampled"] = 0
            self._save_counter = 0

    def enable(self) -> None:
        """Enable trace collection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable trace collection."""
        self._enabled = False

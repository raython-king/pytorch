"""
Integration with PyTorch profiler for runtime scheduler analysis.

This module provides:
- Custom profiler for scheduler events
- Integration with torch.profiler
- Timeline visualization
- Scheduling decision analysis
- What-if analysis capabilities
"""

import time
import json
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import warnings

try:
    import torch
    import torch.profiler as profiler
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
    warnings.warn("PyTorch profiler not available")


class EventType(Enum):
    """Types of scheduler events."""
    SCHEDULE_DECISION = "schedule_decision"
    DEVICE_PLACEMENT = "device_placement"
    MEMORY_ALLOCATION = "memory_allocation"
    OPERATION_START = "operation_start"
    OPERATION_END = "operation_end"
    TRANSFER_START = "transfer_start"
    TRANSFER_END = "transfer_end"
    SYNC_POINT = "sync_point"


@dataclass
class SchedulerEvent:
    """A single scheduler event."""
    event_id: int
    event_type: EventType
    timestamp: float
    duration: float = 0.0
    device: str = ""
    operation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulingDecision:
    """Record of a scheduling decision."""
    decision_id: int
    timestamp: float
    operation: str
    chosen_device: str
    candidate_devices: List[str]
    decision_factors: Dict[str, float]
    predicted_latency: float
    actual_latency: Optional[float] = None
    decision_quality: Optional[float] = None


class RuntimeSchedulerProfiler:
    """
    Custom profiler for runtime scheduler.

    Captures:
    - Scheduling decisions and their outcomes
    - Device placement rationale
    - Memory allocation patterns
    - Operation timelines per device
    - Synchronization points
    """

    def __init__(
        self,
        enabled: bool = True,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False
    ):
        """
        Initialize runtime scheduler profiler.

        Args:
            enabled: Enable profiling
            record_shapes: Record tensor shapes
            profile_memory: Profile memory usage
            with_stack: Record stack traces
        """
        self.enabled = enabled
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        # Event storage
        self._events: List[SchedulerEvent] = []
        self._decisions: List[SchedulingDecision] = []

        # Active operations (event_id -> start_time)
        self._active_ops: Dict[int, float] = {}

        # Counter for event IDs
        self._next_event_id = 0
        self._next_decision_id = 0

        # PyTorch profiler context
        self._torch_profiler: Optional[Any] = None

        # Timeline data
        self._timeline: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)

        # Statistics
        self._stats = {
            "total_events": 0,
            "total_decisions": 0,
            "correct_decisions": 0,
            "suboptimal_decisions": 0
        }

    def start(self) -> None:
        """Start profiling."""
        if not self.enabled:
            return

        # Reset state
        self._events.clear()
        self._decisions.clear()
        self._active_ops.clear()
        self._timeline.clear()

        # Start PyTorch profiler if available
        if HAS_PROFILER:
            self._torch_profiler = profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=self.record_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
            )
            self._torch_profiler.__enter__()

    def stop(self) -> None:
        """Stop profiling."""
        if not self.enabled:
            return

        # Stop PyTorch profiler
        if self._torch_profiler:
            self._torch_profiler.__exit__(None, None, None)
            self._torch_profiler = None

        # Finalize statistics
        self._compute_statistics()

    def record_event(
        self,
        event_type: EventType,
        device: str = "",
        operation: str = "",
        **metadata
    ) -> int:
        """
        Record a scheduler event.

        Args:
            event_type: Type of event
            device: Device associated with event
            operation: Operation name
            **metadata: Additional event metadata

        Returns:
            Event ID
        """
        if not self.enabled:
            return -1

        event_id = self._next_event_id
        self._next_event_id += 1

        event = SchedulerEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=time.time(),
            device=device,
            operation=operation,
            metadata=metadata
        )

        self._events.append(event)
        self._stats["total_events"] += 1

        # Track active operations
        if event_type == EventType.OPERATION_START:
            self._active_ops[event_id] = event.timestamp

        elif event_type == EventType.OPERATION_END:
            # Find matching start event
            if "start_event_id" in metadata:
                start_id = metadata["start_event_id"]
                if start_id in self._active_ops:
                    start_time = self._active_ops[start_id]
                    event.duration = event.timestamp - start_time
                    del self._active_ops[start_id]

                    # Add to timeline
                    self._timeline[device].append(
                        (start_time, event.timestamp, operation)
                    )

        return event_id

    def record_decision(
        self,
        operation: str,
        chosen_device: str,
        candidate_devices: List[str],
        decision_factors: Dict[str, float],
        predicted_latency: float
    ) -> int:
        """
        Record a scheduling decision.

        Args:
            operation: Operation being scheduled
            chosen_device: Device chosen by scheduler
            candidate_devices: All candidate devices
            decision_factors: Factors that influenced decision
            predicted_latency: Predicted latency for chosen device

        Returns:
            Decision ID
        """
        if not self.enabled:
            return -1

        decision_id = self._next_decision_id
        self._next_decision_id += 1

        decision = SchedulingDecision(
            decision_id=decision_id,
            timestamp=time.time(),
            operation=operation,
            chosen_device=chosen_device,
            candidate_devices=candidate_devices,
            decision_factors=decision_factors,
            predicted_latency=predicted_latency
        )

        self._decisions.append(decision)
        self._stats["total_decisions"] += 1

        return decision_id

    def update_decision_outcome(
        self,
        decision_id: int,
        actual_latency: float
    ) -> None:
        """
        Update a decision with actual outcome.

        Args:
            decision_id: ID of decision to update
            actual_latency: Actual measured latency
        """
        if decision_id < 0 or decision_id >= len(self._decisions):
            return

        decision = self._decisions[decision_id]
        decision.actual_latency = actual_latency

        # Compute decision quality (predicted vs actual)
        if decision.predicted_latency > 0:
            error = abs(actual_latency - decision.predicted_latency)
            decision.decision_quality = 1.0 - min(error / decision.predicted_latency, 1.0)

            # Update statistics
            if decision.decision_quality > 0.8:
                self._stats["correct_decisions"] += 1
            elif decision.decision_quality < 0.5:
                self._stats["suboptimal_decisions"] += 1

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        device: Optional[str] = None
    ) -> List[SchedulerEvent]:
        """
        Get recorded events with optional filtering.

        Args:
            event_type: Filter by event type
            device: Filter by device

        Returns:
            List of matching events
        """
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if device:
            events = [e for e in events if e.device == device]

        return events

    def get_decisions(self) -> List[SchedulingDecision]:
        """Get all recorded scheduling decisions."""
        return self._decisions

    def get_timeline(self, device: Optional[str] = None) -> Dict[str, List[Tuple[float, float, str]]]:
        """
        Get operation timeline.

        Args:
            device: Specific device (None for all devices)

        Returns:
            Timeline data {device: [(start, end, operation), ...]}
        """
        if device:
            return {device: self._timeline.get(device, [])}
        return dict(self._timeline)

    def get_statistics(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        stats = dict(self._stats)

        # Add decision accuracy
        if stats["total_decisions"] > 0:
            stats["decision_accuracy"] = (
                stats["correct_decisions"] / stats["total_decisions"]
            )
        else:
            stats["decision_accuracy"] = 0.0

        # Add average decision quality
        decisions_with_quality = [
            d for d in self._decisions if d.decision_quality is not None
        ]
        if decisions_with_quality:
            stats["avg_decision_quality"] = sum(
                d.decision_quality for d in decisions_with_quality
            ) / len(decisions_with_quality)
        else:
            stats["avg_decision_quality"] = 0.0

        return stats

    def _compute_statistics(self) -> None:
        """Compute summary statistics."""
        # Already computed incrementally
        pass

    def export_chrome_trace(self, filename: str) -> None:
        """
        Export timeline to Chrome trace format.

        Args:
            filename: Output filename
        """
        trace_events = []

        # Convert timeline to Chrome trace format
        for device, operations in self._timeline.items():
            for start_time, end_time, op_name in operations:
                trace_events.append({
                    "name": op_name,
                    "cat": "operation",
                    "ph": "X",  # Complete event
                    "ts": int(start_time * 1e6),  # microseconds
                    "dur": int((end_time - start_time) * 1e6),
                    "pid": hash(device) % 1000,
                    "tid": 1,
                    "args": {"device": device}
                })

        # Add scheduling decision markers
        for decision in self._decisions:
            trace_events.append({
                "name": f"Schedule: {decision.operation}",
                "cat": "scheduling",
                "ph": "i",  # Instant event
                "ts": int(decision.timestamp * 1e6),
                "pid": 0,
                "tid": 0,
                "s": "g",  # Global scope
                "args": {
                    "device": decision.chosen_device,
                    "predicted_latency": decision.predicted_latency,
                    "actual_latency": decision.actual_latency,
                }
            })

        # Write to file
        with open(filename, 'w') as f:
            json.dump({"traceEvents": trace_events}, f, indent=2)

    def analyze_scheduling_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze scheduling effectiveness.

        Returns:
            Analysis results including:
            - Decision accuracy
            - Latency prediction errors
            - Device utilization balance
            - Bottlenecks
        """
        if not self._decisions:
            return {"error": "No decisions recorded"}

        analysis = {
            "total_decisions": len(self._decisions),
            "decisions_with_outcomes": 0,
            "prediction_errors": [],
            "device_usage": defaultdict(int),
            "operation_latencies": defaultdict(list),
        }

        # Analyze each decision
        for decision in self._decisions:
            analysis["device_usage"][decision.chosen_device] += 1

            if decision.actual_latency is not None:
                analysis["decisions_with_outcomes"] += 1

                # Prediction error
                error = abs(decision.actual_latency - decision.predicted_latency)
                relative_error = error / decision.predicted_latency if decision.predicted_latency > 0 else 0
                analysis["prediction_errors"].append(relative_error)

                # Operation latencies
                analysis["operation_latencies"][decision.operation].append(
                    decision.actual_latency
                )

        # Compute statistics
        if analysis["prediction_errors"]:
            analysis["mean_prediction_error"] = sum(analysis["prediction_errors"]) / len(analysis["prediction_errors"])
            analysis["max_prediction_error"] = max(analysis["prediction_errors"])

        # Device balance
        if analysis["device_usage"]:
            usage_values = list(analysis["device_usage"].values())
            analysis["device_balance"] = min(usage_values) / max(usage_values) if max(usage_values) > 0 else 0

        # Convert defaultdicts to regular dicts
        analysis["device_usage"] = dict(analysis["device_usage"])
        analysis["operation_latencies"] = dict(analysis["operation_latencies"])

        return analysis

    def what_if_analysis(
        self,
        alternative_placements: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Perform what-if analysis with alternative device placements.

        Args:
            alternative_placements: {decision_id: alternative_device}

        Returns:
            Comparison of original vs alternative placements
        """
        results = {
            "original": {"total_time": 0.0, "device_load": defaultdict(float)},
            "alternative": {"total_time": 0.0, "device_load": defaultdict(float)},
        }

        # Simulate both scenarios
        for decision in self._decisions:
            if decision.actual_latency is None:
                continue

            # Original placement
            results["original"]["total_time"] += decision.actual_latency
            results["original"]["device_load"][decision.chosen_device] += decision.actual_latency

            # Alternative placement
            if decision.decision_id in alternative_placements:
                alt_device = alternative_placements[decision.decision_id]
                # Estimate latency (simplified - assumes same latency)
                results["alternative"]["total_time"] += decision.actual_latency
                results["alternative"]["device_load"][alt_device] += decision.actual_latency
            else:
                results["alternative"]["total_time"] += decision.actual_latency
                results["alternative"]["device_load"][decision.chosen_device] += decision.actual_latency

        # Convert defaultdicts
        results["original"]["device_load"] = dict(results["original"]["device_load"])
        results["alternative"]["device_load"] = dict(results["alternative"]["device_load"])

        # Compute improvement
        if results["original"]["total_time"] > 0:
            improvement = (
                results["original"]["total_time"] - results["alternative"]["total_time"]
            ) / results["original"]["total_time"]
            results["improvement_percentage"] = improvement * 100

        return results


class ProfilerContext:
    """
    Context manager for easy profiling.

    Usage:
        with ProfilerContext() as prof:
            # Your code
            model(input)

        prof.export_chrome_trace("trace.json")
    """

    def __init__(
        self,
        profiler: Optional[RuntimeSchedulerProfiler] = None,
        **kwargs
    ):
        """
        Initialize profiler context.

        Args:
            profiler: Optional existing profiler
            **kwargs: Arguments for RuntimeSchedulerProfiler
        """
        self.profiler = profiler or RuntimeSchedulerProfiler(**kwargs)

    def __enter__(self) -> RuntimeSchedulerProfiler:
        self.profiler.start()
        return self.profiler

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.profiler.stop()


# Integration with torch.profiler
def create_scheduler_profiler_schedule(
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1
) -> Optional[Any]:
    """
    Create a profiler schedule for runtime scheduler.

    Args:
        wait: Number of steps to skip at the beginning
        warmup: Number of warmup steps
        active: Number of active profiling steps
        repeat: Number of times to repeat the cycle

    Returns:
        Profiler schedule or None if profiler not available
    """
    if not HAS_PROFILER:
        return None

    return profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat
    )


def profile_with_scheduler(
    model_fn,
    num_steps: int = 10,
    trace_file: str = "scheduler_trace.json",
    **profiler_kwargs
) -> RuntimeSchedulerProfiler:
    """
    Profile a model execution with runtime scheduler profiling.

    Args:
        model_fn: Function that executes one step of the model
        num_steps: Number of steps to profile
        trace_file: Output trace file
        **profiler_kwargs: Additional profiler arguments

    Returns:
        RuntimeSchedulerProfiler with results
    """
    scheduler_profiler = RuntimeSchedulerProfiler(**profiler_kwargs)

    scheduler_profiler.start()

    try:
        for step in range(num_steps):
            model_fn()

    finally:
        scheduler_profiler.stop()

        if trace_file:
            scheduler_profiler.export_chrome_trace(trace_file)

    return scheduler_profiler

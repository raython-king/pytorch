"""
Flow Scheduler for Adaptive Flow Control.

Implements multiple scheduling policies including SJF, WFQ, EDF,
and ML-based scheduling for efficient flow prioritization.
"""

import time
import heapq
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FlowInfo:
    """Information about a flow for scheduling.

    Attributes:
        flow_id: Flow identifier
        size: Flow size in bytes
        priority: Flow priority (lower = higher priority)
        deadline: Optional deadline timestamp
        arrival_time: When flow arrived
        estimated_duration: Estimated completion time
        weight: Scheduling weight for weighted fair queuing
    """
    flow_id: str
    size: int
    priority: int = 1
    deadline: Optional[float] = None
    arrival_time: float = 0.0
    estimated_duration: float = 0.0
    weight: float = 1.0


class FlowScheduler(ABC):
    """Abstract base class for flow schedulers."""

    def __init__(self):
        """Initialize scheduler."""
        self._lock = threading.RLock()
        self._scheduled_flows: Set[str] = set()

    @abstractmethod
    def schedule(self, flows: List[FlowInfo], available_slots: int) -> List[str]:
        """Schedule flows for execution.

        Args:
            flows: List of flows to schedule
            available_slots: Number of available execution slots

        Returns:
            List of flow IDs to execute
        """
        pass

    @abstractmethod
    def update(self, flow_id: str, progress: float) -> None:
        """Update scheduler with flow progress.

        Args:
            flow_id: Flow identifier
            progress: Progress fraction [0.0, 1.0]
        """
        pass

    def reset(self) -> None:
        """Reset scheduler state."""
        with self._lock:
            self._scheduled_flows.clear()


class SJF_Scheduler(FlowScheduler):
    """Shortest Job First scheduler.

    Schedules flows with smallest size/duration first to minimize
    average completion time.
    """

    def __init__(self, preemptive: bool = False):
        """Initialize SJF scheduler.

        Args:
            preemptive: Enable preemption (SRTF - Shortest Remaining Time First)
        """
        super().__init__()
        self._preemptive = preemptive
        self._remaining_time: Dict[str, float] = {}

    def schedule(self, flows: List[FlowInfo], available_slots: int) -> List[str]:
        """Schedule flows by shortest job first.

        Args:
            flows: List of flows to schedule
            available_slots: Number of available execution slots

        Returns:
            List of flow IDs to execute
        """
        with self._lock:
            if self._preemptive:
                # Sort by remaining time (SRTF)
                sorted_flows = sorted(
                    flows,
                    key=lambda f: self._remaining_time.get(f.flow_id, f.estimated_duration)
                )
            else:
                # Sort by estimated duration
                sorted_flows = sorted(flows, key=lambda f: f.estimated_duration)

            # Select top flows up to available slots
            selected = []
            for flow in sorted_flows[:available_slots]:
                selected.append(flow.flow_id)
                self._scheduled_flows.add(flow.flow_id)
                if flow.flow_id not in self._remaining_time:
                    self._remaining_time[flow.flow_id] = flow.estimated_duration

            logger.debug(f"SJF scheduled {len(selected)} flows")
            return selected

    def update(self, flow_id: str, progress: float) -> None:
        """Update remaining time for flow.

        Args:
            flow_id: Flow identifier
            progress: Progress fraction [0.0, 1.0]
        """
        with self._lock:
            if flow_id in self._remaining_time:
                original = self._remaining_time[flow_id]
                self._remaining_time[flow_id] = original * (1.0 - progress)

                if progress >= 1.0:
                    del self._remaining_time[flow_id]
                    self._scheduled_flows.discard(flow_id)


class WFQ_Scheduler(FlowScheduler):
    """Weighted Fair Queuing scheduler.

    Allocates bandwidth proportional to flow weights while ensuring
    fairness and preventing starvation.
    """

    def __init__(self):
        """Initialize WFQ scheduler."""
        super().__init__()

        # Virtual time tracking
        self._virtual_time = 0.0
        self._flow_finish_times: Dict[str, float] = {}
        self._flow_start_times: Dict[str, float] = {}

    def schedule(self, flows: List[FlowInfo], available_slots: int) -> List[str]:
        """Schedule flows using weighted fair queuing.

        Args:
            flows: List of flows to schedule
            available_slots: Number of available execution slots

        Returns:
            List of flow IDs to execute
        """
        with self._lock:
            # Calculate virtual finish times for new flows
            for flow in flows:
                if flow.flow_id not in self._flow_finish_times:
                    self._flow_start_times[flow.flow_id] = self._virtual_time
                    # Finish time = start time + (size / weight)
                    service_time = flow.size / max(flow.weight, 0.1)
                    self._flow_finish_times[flow.flow_id] = self._virtual_time + service_time

            # Sort by virtual finish time
            flow_queue = [
                (self._flow_finish_times[f.flow_id], f.flow_id)
                for f in flows if f.flow_id in self._flow_finish_times
            ]
            heapq.heapify(flow_queue)

            # Select flows with earliest finish times
            selected = []
            while flow_queue and len(selected) < available_slots:
                finish_time, flow_id = heapq.heappop(flow_queue)
                selected.append(flow_id)
                self._scheduled_flows.add(flow_id)

            logger.debug(f"WFQ scheduled {len(selected)} flows")
            return selected

    def update(self, flow_id: str, progress: float) -> None:
        """Update virtual time based on progress.

        Args:
            flow_id: Flow identifier
            progress: Progress fraction [0.0, 1.0]
        """
        with self._lock:
            if flow_id in self._flow_finish_times:
                # Update virtual time
                start_time = self._flow_start_times[flow_id]
                finish_time = self._flow_finish_times[flow_id]
                self._virtual_time = start_time + (finish_time - start_time) * progress

                if progress >= 1.0:
                    del self._flow_finish_times[flow_id]
                    del self._flow_start_times[flow_id]
                    self._scheduled_flows.discard(flow_id)


class EDF_Scheduler(FlowScheduler):
    """Earliest Deadline First scheduler.

    Schedules flows with earliest deadlines first to maximize
    deadline satisfaction rate.
    """

    def __init__(self, slack_threshold: float = 0.1):
        """Initialize EDF scheduler.

        Args:
            slack_threshold: Minimum slack time as fraction of deadline
        """
        super().__init__()
        self._slack_threshold = slack_threshold
        self._missed_deadlines = 0
        self._total_deadlines = 0

    def schedule(self, flows: List[FlowInfo], available_slots: int) -> List[str]:
        """Schedule flows by earliest deadline first.

        Args:
            flows: List of flows to schedule
            available_slots: Number of available execution slots

        Returns:
            List of flow IDs to execute
        """
        with self._lock:
            current_time = time.time()

            # Filter flows with deadlines
            deadline_flows = [f for f in flows if f.deadline is not None]
            no_deadline_flows = [f for f in flows if f.deadline is None]

            # Calculate slack time for deadline flows
            flow_slack = []
            for flow in deadline_flows:
                slack = flow.deadline - current_time - flow.estimated_duration
                urgency = slack / max(flow.estimated_duration, 1.0)
                flow_slack.append((urgency, flow.deadline, flow.flow_id))

            # Sort by urgency (slack ratio), then deadline
            flow_slack.sort()

            # Select urgent flows first
            selected = []
            for urgency, deadline, flow_id in flow_slack:
                if len(selected) >= available_slots:
                    break

                selected.append(flow_id)
                self._scheduled_flows.add(flow_id)

                # Warn if tight deadline
                if urgency < self._slack_threshold:
                    logger.warning(f"Flow {flow_id} has tight deadline: {urgency:.2%} slack")

            # Fill remaining slots with non-deadline flows (FCFS)
            for flow in no_deadline_flows:
                if len(selected) >= available_slots:
                    break
                selected.append(flow.flow_id)
                self._scheduled_flows.add(flow.flow_id)

            logger.debug(f"EDF scheduled {len(selected)} flows")
            return selected

    def update(self, flow_id: str, progress: float) -> None:
        """Update deadline tracking.

        Args:
            flow_id: Flow identifier
            progress: Progress fraction [0.0, 1.0]
        """
        with self._lock:
            if progress >= 1.0:
                self._scheduled_flows.discard(flow_id)
                self._total_deadlines += 1

    def record_missed_deadline(self, flow_id: str) -> None:
        """Record a missed deadline.

        Args:
            flow_id: Flow identifier
        """
        with self._lock:
            self._missed_deadlines += 1
            logger.warning(f"Flow {flow_id} missed deadline")

    def get_deadline_stats(self) -> Dict[str, float]:
        """Get deadline satisfaction statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if self._total_deadlines == 0:
                return {'satisfaction_rate': 1.0, 'missed': 0, 'total': 0}

            satisfied = self._total_deadlines - self._missed_deadlines
            return {
                'satisfaction_rate': satisfied / self._total_deadlines,
                'missed': self._missed_deadlines,
                'total': self._total_deadlines,
            }


class ML_Scheduler(FlowScheduler):
    """ML-based scheduler using learned policies.

    Uses machine learning to predict optimal scheduling decisions
    based on flow characteristics and system state.
    """

    def __init__(self, model=None):
        """Initialize ML scheduler.

        Args:
            model: Optional trained model (sklearn-compatible)
        """
        super().__init__()
        self._model = model
        self._feature_history: List[Tuple] = []
        self._decision_history: List[Tuple] = []

    def schedule(self, flows: List[FlowInfo], available_slots: int) -> List[str]:
        """Schedule flows using ML model.

        Args:
            flows: List of flows to schedule
            available_slots: Number of available execution slots

        Returns:
            List of flow IDs to execute
        """
        with self._lock:
            if self._model is None or len(flows) == 0:
                # Fallback to priority-based scheduling
                sorted_flows = sorted(flows, key=lambda f: (f.priority, f.arrival_time))
                selected = [f.flow_id for f in sorted_flows[:available_slots]]
            else:
                # Extract features and predict scores
                features = self._extract_features(flows)
                scores = self._model.predict(features)

                # Rank flows by predicted score
                flow_scores = list(zip(scores, [f.flow_id for f in flows]))
                flow_scores.sort(reverse=True)

                selected = [flow_id for _, flow_id in flow_scores[:available_slots]]

            for flow_id in selected:
                self._scheduled_flows.add(flow_id)

            logger.debug(f"ML scheduler scheduled {len(selected)} flows")
            return selected

    def _extract_features(self, flows: List[FlowInfo]) -> np.ndarray:
        """Extract features from flows for ML model.

        Args:
            flows: List of flows

        Returns:
            Feature matrix (n_flows x n_features)
        """
        current_time = time.time()
        features = []

        for flow in flows:
            flow_features = [
                flow.size,
                flow.priority,
                flow.weight,
                flow.estimated_duration,
                current_time - flow.arrival_time,  # Wait time
            ]

            # Deadline features
            if flow.deadline is not None:
                flow_features.extend([
                    1.0,  # Has deadline
                    flow.deadline - current_time,  # Time to deadline
                    (flow.deadline - current_time) / max(flow.estimated_duration, 1.0)  # Slack ratio
                ])
            else:
                flow_features.extend([0.0, 0.0, 0.0])

            features.append(flow_features)

        return np.array(features)

    def update(self, flow_id: str, progress: float) -> None:
        """Update with flow progress for learning.

        Args:
            flow_id: Flow identifier
            progress: Progress fraction [0.0, 1.0]
        """
        with self._lock:
            if progress >= 1.0:
                self._scheduled_flows.discard(flow_id)

    def update_model(self, model) -> None:
        """Update the ML model.

        Args:
            model: New trained model
        """
        with self._lock:
            self._model = model
            logger.info("Updated ML scheduler model")


class CompositeScheduler(FlowScheduler):
    """Composite scheduler combining multiple policies.

    Intelligently switches between schedulers based on workload
    characteristics and system conditions.
    """

    def __init__(self):
        """Initialize composite scheduler."""
        super().__init__()

        # Available schedulers
        self._schedulers = {
            'sjf': SJF_Scheduler(preemptive=True),
            'wfq': WFQ_Scheduler(),
            'edf': EDF_Scheduler(),
            'ml': ML_Scheduler(),
        }

        self._current_scheduler = 'wfq'  # Default
        self._scheduler_stats: Dict[str, Dict] = defaultdict(
            lambda: {'uses': 0, 'avg_completion_time': 0.0}
        )

    def schedule(self, flows: List[FlowInfo], available_slots: int) -> List[str]:
        """Schedule flows using selected policy.

        Args:
            flows: List of flows to schedule
            available_slots: Number of available execution slots

        Returns:
            List of flow IDs to execute
        """
        with self._lock:
            # Select scheduler based on workload
            scheduler_name = self._select_scheduler(flows)
            scheduler = self._schedulers[scheduler_name]

            # Delegate to selected scheduler
            selected = scheduler.schedule(flows, available_slots)

            # Update stats
            self._scheduler_stats[scheduler_name]['uses'] += 1

            logger.debug(f"Composite scheduler using {scheduler_name}, selected {len(selected)} flows")
            return selected

    def _select_scheduler(self, flows: List[FlowInfo]) -> str:
        """Select best scheduler for current workload.

        Args:
            flows: List of flows to schedule

        Returns:
            Scheduler name
        """
        if not flows:
            return self._current_scheduler

        # Count flows with deadlines
        deadline_flows = sum(1 for f in flows if f.deadline is not None)
        deadline_ratio = deadline_flows / len(flows)

        # Count flows with different priorities
        priorities = set(f.priority for f in flows)
        has_mixed_priorities = len(priorities) > 1

        # Decision logic
        if deadline_ratio > 0.5:
            # Many deadline flows -> use EDF
            return 'edf'
        elif has_mixed_priorities:
            # Mixed priorities -> use WFQ
            return 'wfq'
        else:
            # Homogeneous workload -> use SJF
            return 'sjf'

    def update(self, flow_id: str, progress: float) -> None:
        """Update all schedulers.

        Args:
            flow_id: Flow identifier
            progress: Progress fraction [0.0, 1.0]
        """
        with self._lock:
            for scheduler in self._schedulers.values():
                scheduler.update(flow_id, progress)

    def get_scheduler_stats(self) -> Dict[str, Dict]:
        """Get statistics for all schedulers.

        Returns:
            Dictionary of scheduler statistics
        """
        with self._lock:
            return dict(self._scheduler_stats)

    def set_active_scheduler(self, scheduler_name: str) -> None:
        """Manually set active scheduler.

        Args:
            scheduler_name: Name of scheduler to use

        Raises:
            ValueError: If scheduler name invalid
        """
        with self._lock:
            if scheduler_name not in self._schedulers:
                raise ValueError(f"Unknown scheduler: {scheduler_name}")

            self._current_scheduler = scheduler_name
            logger.info(f"Set active scheduler to {scheduler_name}")


class StarvationPrevention:
    """Prevent flow starvation in scheduling.

    Tracks waiting times and boosts priority of flows that have
    waited too long to ensure fairness.
    """

    def __init__(self, max_wait_time: float = 60.0):
        """Initialize starvation prevention.

        Args:
            max_wait_time: Maximum wait time in seconds before boost
        """
        self._lock = threading.RLock()
        self._max_wait_time = max_wait_time

        # Track flow wait times
        self._flow_arrival: Dict[str, float] = {}
        self._flow_boosts: Dict[str, int] = defaultdict(int)

    def register_flow(self, flow_id: str, arrival_time: Optional[float] = None) -> None:
        """Register a new flow.

        Args:
            flow_id: Flow identifier
            arrival_time: Arrival time (default: current time)
        """
        with self._lock:
            if arrival_time is None:
                arrival_time = time.time()
            self._flow_arrival[flow_id] = arrival_time

    def check_starvation(self, flow_id: str) -> int:
        """Check if flow is starving and return priority boost.

        Args:
            flow_id: Flow identifier

        Returns:
            Priority boost level (0 = no boost, higher = more boost)
        """
        with self._lock:
            arrival = self._flow_arrival.get(flow_id)
            if arrival is None:
                return 0

            wait_time = time.time() - arrival
            boost = int(wait_time / self._max_wait_time)

            if boost > self._flow_boosts[flow_id]:
                self._flow_boosts[flow_id] = boost
                logger.warning(f"Flow {flow_id} starving: {wait_time:.1f}s wait, boost={boost}")

            return boost

    def apply_boost(self, flows: List[FlowInfo]) -> List[FlowInfo]:
        """Apply starvation prevention boosts to flows.

        Args:
            flows: List of flows

        Returns:
            Flows with adjusted priorities
        """
        with self._lock:
            boosted_flows = []
            for flow in flows:
                boost = self.check_starvation(flow.flow_id)
                if boost > 0:
                    # Create copy with boosted priority (lower value = higher priority)
                    boosted = FlowInfo(
                        flow_id=flow.flow_id,
                        size=flow.size,
                        priority=max(0, flow.priority - boost),
                        deadline=flow.deadline,
                        arrival_time=flow.arrival_time,
                        estimated_duration=flow.estimated_duration,
                        weight=flow.weight * (1.0 + boost * 0.5),  # Also boost weight
                    )
                    boosted_flows.append(boosted)
                else:
                    boosted_flows.append(flow)

            return boosted_flows

    def remove_flow(self, flow_id: str) -> None:
        """Remove flow tracking.

        Args:
            flow_id: Flow identifier
        """
        with self._lock:
            self._flow_arrival.pop(flow_id, None)
            self._flow_boosts.pop(flow_id, None)

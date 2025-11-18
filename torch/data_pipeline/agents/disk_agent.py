"""
Disk Reader Agent

Manages disk I/O operations and implements intelligent disk read strategies.
"""

import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple
import time

from .base_agent import (
    BaseDataPipelineAgent,
    AgentRole,
    AgentAction,
    AgentDecision,
    DataRequest,
    DataItem,
    PipelineEnvironment,
)


class DiskReaderAgent(BaseDataPipelineAgent):
    """
    Agent responsible for reading data from disk.

    Features:
    - Asynchronous I/O with thread pool
    - Read-ahead buffering
    - I/O pattern detection
    - Adaptive prefetching
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], dataset):
        super().__init__(agent_id, AgentRole.DISK_READER, config)

        self.dataset = dataset
        self.disk_config = config.get("disk", {})

        # I/O thread pool
        max_workers = self.disk_config.get("max_workers", 4)
        self.io_executor = ThreadPoolExecutor(max_workers=max_workers)

        # Read buffer
        self.read_buffer: Dict[Any, DataItem] = {}
        self.buffer_lock = threading.Lock()

        # Statistics
        self.read_count = 0
        self.total_read_time = 0.0
        self.cache_hits = 0

        # Pattern detection
        self.access_pattern: List[Any] = []
        self.detected_patterns: List[List[Any]] = []

    def observe(self, environment: PipelineEnvironment) -> None:
        """Observe pipeline environment and update internal state"""
        # Update access patterns
        self.access_pattern.extend(environment.recent_access_sequence)

        # Keep pattern history bounded
        max_pattern_history = self.config.get("pattern_history_size", 1000)
        if len(self.access_pattern) > max_pattern_history:
            self.access_pattern = self.access_pattern[-max_pattern_history:]

        # Detect patterns periodically
        if len(self.access_pattern) % 100 == 0:
            self._detect_patterns()

        # Update state
        disk_hit_rate = environment.cache_hit_rates.get("disk", 0.0)
        disk_latency = environment.average_latencies.get("disk", 0.0)

        self.update_state(
            cache_hit_rate=disk_hit_rate,
            average_latency_ms=disk_latency,
            current_load=len(self.read_buffer) / 1000.0,
        )

    def decide(self, request: Optional[DataRequest] = None) -> AgentDecision:
        """Decide whether to prefetch or read on demand"""

        if request is not None:
            # Specific request - read from disk
            return AgentDecision(
                action=AgentAction.LOAD_FROM_DISK,
                confidence=1.0,
                target_data=[request.sample_id],
                reasoning="Explicit data request",
            )

        # Check if we should prefetch based on patterns
        if self.detected_patterns:
            # Predict next samples based on pattern
            next_samples = self._predict_next_samples()

            if next_samples:
                return AgentDecision(
                    action=AgentAction.PREFETCH_DATA,
                    confidence=0.8,
                    target_data=next_samples,
                    reasoning="Pattern-based prefetching",
                    expected_benefit=0.5,  # Reduce future latency
                )

        return AgentDecision(
            action=AgentAction.NO_ACTION,
            confidence=1.0,
            reasoning="No pending requests or patterns",
        )

    def execute(self, decision: AgentDecision) -> Tuple[bool, Any]:
        """Execute disk read decision"""
        try:
            if decision.action == AgentAction.LOAD_FROM_DISK:
                results = []
                for sample_id in decision.target_data:
                    data_item = self._read_from_disk(sample_id)
                    if data_item:
                        results.append(data_item)

                self.state.success_count += len(results)
                return True, results

            elif decision.action == AgentAction.PREFETCH_DATA:
                # Prefetch in background
                for sample_id in decision.target_data:
                    self.io_executor.submit(self._read_from_disk, sample_id)

                return True, None

            return True, None

        except Exception as e:
            self.state.error_count += 1
            return False, str(e)

    def learn(self, reward: float, next_environment: PipelineEnvironment) -> None:
        """Learn from outcomes"""
        self.reward_history.append(reward)

        # Keep history bounded
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

        # Adjust prefetch strategy based on reward
        avg_reward = sum(self.reward_history[-100:]) / 100 if len(self.reward_history) >= 100 else 0

        if avg_reward < 0.3:
            # Poor performance - reduce prefetching
            self.config["prefetch_factor"] = max(1, self.config.get("prefetch_factor", 2) - 1)
        elif avg_reward > 0.7:
            # Good performance - increase prefetching
            self.config["prefetch_factor"] = min(5, self.config.get("prefetch_factor", 2) + 1)

    def _read_from_disk(self, sample_id: Any) -> Optional[DataItem]:
        """Read a sample from disk"""
        start_time = time.time()

        try:
            # Check buffer first
            with self.buffer_lock:
                if sample_id in self.read_buffer:
                    item = self.read_buffer[sample_id]
                    item.access_count += 1
                    item.last_access_time = time.time()
                    self.cache_hits += 1
                    return item

            # Read from dataset
            data = self.dataset[sample_id]

            # Estimate size
            if hasattr(data, "nbytes"):
                size_bytes = data.nbytes
            else:
                # Rough estimate using pickle
                size_bytes = len(pickle.dumps(data))

            # Create data item
            data_item = DataItem(
                sample_id=sample_id,
                data=data,
                size_bytes=size_bytes,
                location="disk",
                access_count=1,
                load_time=time.time() - start_time,
            )

            # Add to buffer
            with self.buffer_lock:
                self.read_buffer[sample_id] = data_item

                # Evict old items if buffer too large
                if len(self.read_buffer) > 1000:
                    oldest = min(
                        self.read_buffer.values(),
                        key=lambda x: x.last_access_time
                    )
                    del self.read_buffer[oldest.sample_id]

            # Update statistics
            self.read_count += 1
            self.total_read_time += data_item.load_time
            self.state.total_requests += 1
            self.state.total_bytes_processed += size_bytes

            return data_item

        except Exception as e:
            print(f"Error reading sample {sample_id}: {e}")
            self.state.error_count += 1
            return None

    def _detect_patterns(self) -> None:
        """Detect sequential or repeated patterns in access sequence"""
        if len(self.access_pattern) < 10:
            return

        # Detect sequential access
        recent = self.access_pattern[-100:]

        # Check for sequential pattern (e.g., 1, 2, 3, 4, ...)
        if all(
            isinstance(recent[i], int) and recent[i] == recent[i-1] + 1
            for i in range(1, min(10, len(recent)))
        ):
            # Sequential pattern detected
            if not self.detected_patterns or self.detected_patterns[-1] != ["sequential"]:
                self.detected_patterns.append(["sequential"])

        # Detect repeated patterns
        for pattern_len in range(3, 10):
            if len(recent) >= pattern_len * 2:
                pattern = recent[-pattern_len:]
                prev_pattern = recent[-pattern_len*2:-pattern_len]

                if pattern == prev_pattern:
                    if pattern not in self.detected_patterns:
                        self.detected_patterns.append(pattern)
                        break

        # Keep only recent patterns
        if len(self.detected_patterns) > 10:
            self.detected_patterns = self.detected_patterns[-10:]

    def _predict_next_samples(self) -> List[Any]:
        """Predict next samples based on detected patterns"""
        if not self.access_pattern or not self.detected_patterns:
            return []

        predictions = []
        last_access = self.access_pattern[-1] if self.access_pattern else None

        for pattern in self.detected_patterns:
            if pattern == ["sequential"] and isinstance(last_access, int):
                # Predict next sequential samples
                prefetch_factor = self.config.get("prefetch_factor", 2)
                predictions.extend(range(last_access + 1, last_access + 1 + prefetch_factor))

            elif isinstance(pattern, list) and len(pattern) > 0:
                # Find where we are in the pattern
                try:
                    idx = pattern.index(last_access)
                    # Predict next items in pattern
                    next_idx = (idx + 1) % len(pattern)
                    predictions.append(pattern[next_idx])
                except ValueError:
                    pass

        return predictions[:10]  # Limit predictions

    def get_statistics(self) -> Dict[str, Any]:
        """Get disk agent statistics"""
        avg_read_time = (
            self.total_read_time / self.read_count
            if self.read_count > 0 else 0.0
        )

        hit_rate = (
            self.cache_hits / self.read_count
            if self.read_count > 0 else 0.0
        )

        return {
            "total_reads": self.read_count,
            "average_read_time_ms": avg_read_time * 1000,
            "buffer_hit_rate": hit_rate,
            "buffer_size": len(self.read_buffer),
            "detected_patterns": len(self.detected_patterns),
        }

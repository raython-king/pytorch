"""
Message Coalescing
消息聚合

This module provides message coalescing to combine small messages into larger
ones for better bandwidth utilization.

本模块提供消息聚合以将小消息组合成较大的消息以提高带宽利用率。
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import torch

from .types import Message, CoalescedMessage
from .utils import (
    get_tensor_size_bytes,
    estimate_communication_time,
    align_size,
)

logger = logging.getLogger(__name__)


class MessageCoalescer:
    """
    Message coalescing for small message aggregation.
    小消息聚合的消息聚合器。

    This class coalesces small messages to reduce communication overhead.
    Small messages suffer from poor bandwidth utilization due to high
    latency/bandwidth ratio.

    Attributes:
        threshold_kb: Size threshold for coalescing (KB)
        max_coalesced_size_mb: Maximum size of coalesced message (MB)
        timeout_us: Timeout for waiting for more messages (microseconds)
    """

    def __init__(
        self,
        threshold_kb: float = 64.0,
        max_coalesced_size_mb: float = 256.0,
        timeout_us: float = 100.0
    ):
        """
        Initialize message coalescer.
        初始化消息聚合器。

        Args:
            threshold_kb: Threshold for coalescing in KB
            max_coalesced_size_mb: Maximum coalesced message size in MB
            timeout_us: Timeout for waiting in microseconds
        """
        self.threshold_kb = threshold_kb
        self.threshold_bytes = int(threshold_kb * 1024)

        self.max_coalesced_size_mb = max_coalesced_size_mb
        self.max_coalesced_bytes = int(max_coalesced_size_mb * 1024 * 1024)

        self.timeout_us = timeout_us

        # Pending messages grouped by (source, target)
        self._pending: Dict[Tuple[int, int], List[Message]] = defaultdict(list)

        # Statistics
        self._stats = {
            'num_coalesced': 0,
            'num_messages_combined': 0,
            'bytes_saved': 0,
        }

    def should_coalesce(self, messages: List[Message]) -> bool:
        """
        Determine if messages should be coalesced.
        确定是否应该聚合消息。

        Args:
            messages: List of messages to potentially coalesce

        Returns:
            True if coalescing is beneficial
        """
        if len(messages) < 2:
            return False

        # Check if all messages are below threshold
        all_small = all(msg.size_bytes < self.threshold_bytes for msg in messages)
        if not all_small:
            return False

        # Check if total size is within limits
        total_size = sum(msg.size_bytes for msg in messages)
        if total_size > self.max_coalesced_bytes:
            return False

        # Check if messages have same source and target
        if len(messages) > 1:
            first_msg = messages[0]
            same_route = all(
                msg.source == first_msg.source and msg.target == first_msg.target
                for msg in messages
            )
            if not same_route:
                return False

        return True

    def coalesce_messages(self, messages: List[Message]) -> CoalescedMessage:
        """
        Coalesce multiple messages into one.
        将多个消息聚合成一个。

        Args:
            messages: List of messages to coalesce

        Returns:
            Coalesced message

        Raises:
            ValueError: If messages cannot be coalesced
        """
        if not self.should_coalesce(messages):
            raise ValueError("Messages cannot be coalesced")

        # Calculate total size
        total_size = sum(msg.size_bytes for msg in messages)

        # Create buffer for coalesced data
        # Align total size for better memory access
        aligned_size = align_size(total_size)

        # Determine dtype (use first message's dtype)
        first_dtype = messages[0].data.dtype

        # Flatten all message data
        flattened_data = []
        offsets = [0]

        current_offset = 0
        for msg in messages:
            # Flatten tensor
            flat_data = msg.data.flatten()
            flattened_data.append(flat_data)

            current_offset += flat_data.numel()
            offsets.append(current_offset)

        # Concatenate all data
        coalesced_tensor = torch.cat(flattened_data)

        # Create coalesced message
        coalesced = CoalescedMessage(
            original_messages=messages,
            coalesced_data=coalesced_tensor,
            offsets=offsets,
            total_size=total_size,
        )

        # Update statistics
        self._stats['num_coalesced'] += 1
        self._stats['num_messages_combined'] += len(messages)

        # Estimate bytes saved (overhead reduction)
        # Each message has some fixed overhead, coalescing reduces this
        overhead_per_msg = 100  # bytes (simplified)
        self._stats['bytes_saved'] += overhead_per_msg * (len(messages) - 1)

        logger.debug(
            f"Coalesced {len(messages)} messages "
            f"(total size: {total_size / 1024:.2f} KB)"
        )

        return coalesced

    def split_coalesced(self, coalesced: CoalescedMessage) -> List[Message]:
        """
        Split a coalesced message back into original messages.
        将聚合消息拆分回原始消息。

        Args:
            coalesced: Coalesced message to split

        Returns:
            List of original messages with updated data
        """
        return coalesced.split()

    def adaptive_threshold(
        self,
        bandwidth_gbps: float,
        latency_us: float
    ) -> float:
        """
        Compute adaptive coalescing threshold based on network characteristics.
        根据网络特性计算自适应聚合阈值。

        For small messages, latency dominates. The threshold should be set
        such that communication time is dominated by bandwidth, not latency.

        Args:
            bandwidth_gbps: Network bandwidth in GB/s
            latency_us: Network latency in microseconds

        Returns:
            Adaptive threshold in KB
        """
        # Find message size where latency = bandwidth time
        # latency = (size / bandwidth)
        # size = latency * bandwidth

        # Convert bandwidth to bytes per microsecond
        bandwidth_bytes_per_us = bandwidth_gbps * 1000.0  # GB/s to B/us

        # Size where latency equals transfer time
        crossover_size_bytes = latency_us * bandwidth_bytes_per_us

        # Use 2x this size as threshold (to be safely in bandwidth-bound region)
        threshold_bytes = 2 * crossover_size_bytes

        # Clamp to reasonable range
        min_threshold = 4 * 1024  # 4 KB
        max_threshold = 1024 * 1024  # 1 MB

        threshold_bytes = max(min_threshold, min(threshold_bytes, max_threshold))

        threshold_kb = threshold_bytes / 1024

        logger.debug(
            f"Adaptive threshold: {threshold_kb:.2f} KB "
            f"(bandwidth={bandwidth_gbps:.2f} GB/s, latency={latency_us:.2f} us)"
        )

        return threshold_kb

    def add_message(self, message: Message) -> Optional[CoalescedMessage]:
        """
        Add a message to the coalescer.
        向聚合器添加消息。

        If enough messages are pending or timeout is reached, returns
        a coalesced message. Otherwise, returns None.

        Args:
            message: Message to add

        Returns:
            Coalesced message if ready, otherwise None
        """
        key = (message.source, message.target)

        # Add to pending
        self._pending[key].append(message)
        pending_msgs = self._pending[key]

        # Check if should coalesce now
        total_size = sum(msg.size_bytes for msg in pending_msgs)

        # Coalesce if:
        # 1. Total size exceeds threshold (batching enough small messages)
        # 2. Timeout reached (don't wait too long)
        # 3. Total size approaching max limit

        should_flush = False

        if total_size >= self.threshold_bytes:
            should_flush = True
        elif total_size >= self.max_coalesced_bytes * 0.9:
            should_flush = True
        elif len(pending_msgs) > 1:
            # Check timeout
            oldest_timestamp = min(msg.timestamp for msg in pending_msgs)
            current_time = time.time()
            elapsed_us = (current_time - oldest_timestamp) * 1e6

            if elapsed_us >= self.timeout_us:
                should_flush = True

        if should_flush and self.should_coalesce(pending_msgs):
            # Coalesce and clear pending
            coalesced = self.coalesce_messages(pending_msgs)
            self._pending[key] = []
            return coalesced

        return None

    def flush_pending(
        self,
        source: Optional[int] = None,
        target: Optional[int] = None
    ) -> List[CoalescedMessage]:
        """
        Flush pending messages.
        清空待处理消息。

        Args:
            source: Flush only messages from this source (None for all)
            target: Flush only messages to this target (None for all)

        Returns:
            List of coalesced messages
        """
        coalesced_list = []

        # Determine which keys to flush
        if source is not None and target is not None:
            keys_to_flush = [(source, target)]
        elif source is not None:
            keys_to_flush = [k for k in self._pending.keys() if k[0] == source]
        elif target is not None:
            keys_to_flush = [k for k in self._pending.keys() if k[1] == target]
        else:
            keys_to_flush = list(self._pending.keys())

        # Flush each key
        for key in keys_to_flush:
            pending_msgs = self._pending[key]
            if pending_msgs and self.should_coalesce(pending_msgs):
                coalesced = self.coalesce_messages(pending_msgs)
                coalesced_list.append(coalesced)
            self._pending[key] = []

        return coalesced_list

    def get_statistics(self) -> Dict[str, int]:
        """
        Get coalescing statistics.
        获取聚合统计信息。

        Returns:
            Dictionary with statistics
        """
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """Reset statistics counters"""
        self._stats = {
            'num_coalesced': 0,
            'num_messages_combined': 0,
            'bytes_saved': 0,
        }


class SmartCoalescer(MessageCoalescer):
    """
    Smart message coalescer with adaptive behavior.
    具有自适应行为的智能消息聚合器。

    This extends MessageCoalescer with:
    - Automatic threshold adaptation based on network conditions
    - Priority-aware coalescing
    - Traffic pattern learning

    Attributes:
        bandwidth_gbps: Current network bandwidth estimate
        latency_us: Current network latency estimate
    """

    def __init__(
        self,
        bandwidth_gbps: float = 100.0,
        latency_us: float = 5.0,
        **kwargs
    ):
        """
        Initialize smart coalescer.
        初始化智能聚合器。

        Args:
            bandwidth_gbps: Initial bandwidth estimate in GB/s
            latency_us: Initial latency estimate in microseconds
            **kwargs: Additional arguments for MessageCoalescer
        """
        # Compute adaptive threshold
        temp_coalescer = MessageCoalescer()
        adaptive_threshold = temp_coalescer.adaptive_threshold(
            bandwidth_gbps, latency_us
        )

        # Override threshold if not provided
        if 'threshold_kb' not in kwargs:
            kwargs['threshold_kb'] = adaptive_threshold

        super().__init__(**kwargs)

        self.bandwidth_gbps = bandwidth_gbps
        self.latency_us = latency_us

        # Message pattern history
        self._message_history: List[Tuple[int, float]] = []  # (size, timestamp)
        self._max_history = 1000

    def update_network_conditions(
        self,
        bandwidth_gbps: float,
        latency_us: float
    ) -> None:
        """
        Update network condition estimates.
        更新网络状况估计。

        This will re-compute the adaptive threshold.

        Args:
            bandwidth_gbps: New bandwidth estimate
            latency_us: New latency estimate
        """
        self.bandwidth_gbps = bandwidth_gbps
        self.latency_us = latency_us

        # Update threshold
        new_threshold = self.adaptive_threshold(bandwidth_gbps, latency_us)
        self.threshold_kb = new_threshold
        self.threshold_bytes = int(new_threshold * 1024)

        logger.info(
            f"Updated coalescing threshold to {new_threshold:.2f} KB "
            f"(bandwidth={bandwidth_gbps:.2f} GB/s, latency={latency_us:.2f} us)"
        )

    def add_message(self, message: Message) -> Optional[CoalescedMessage]:
        """
        Add message with pattern learning.
        添加具有模式学习的消息。

        Args:
            message: Message to add

        Returns:
            Coalesced message if ready
        """
        # Record message in history
        self._message_history.append((message.size_bytes, message.timestamp))

        # Trim history if too long
        if len(self._message_history) > self._max_history:
            self._message_history = self._message_history[-self._max_history:]

        # Use parent's add_message
        return super().add_message(message)

    def should_coalesce_priority(
        self,
        messages: List[Message],
        max_priority_gap: int = 2
    ) -> bool:
        """
        Check if messages should be coalesced considering priority.
        考虑优先级检查是否应聚合消息。

        Args:
            messages: List of messages
            max_priority_gap: Maximum priority difference allowed

        Returns:
            True if should coalesce
        """
        if not self.should_coalesce(messages):
            return False

        # Check priority gap
        if len(messages) > 1:
            priorities = [msg.priority for msg in messages]
            priority_range = max(priorities) - min(priorities)

            if priority_range > max_priority_gap:
                logger.debug(
                    f"Skipping coalescing due to priority gap: {priority_range}"
                )
                return False

        return True

    def analyze_message_patterns(self) -> Dict[str, float]:
        """
        Analyze message size patterns.
        分析消息大小模式。

        Returns:
            Dictionary with pattern statistics
        """
        if not self._message_history:
            return {}

        sizes = [size for size, _ in self._message_history]

        stats = {
            'mean_size': sum(sizes) / len(sizes),
            'min_size': min(sizes),
            'max_size': max(sizes),
            'num_small_messages': sum(1 for s in sizes if s < self.threshold_bytes),
        }

        # Compute percentage of small messages
        stats['small_message_ratio'] = (
            stats['num_small_messages'] / len(sizes)
        )

        # Compute message rate
        if len(self._message_history) > 1:
            time_span = (
                self._message_history[-1][1] - self._message_history[0][1]
            )
            if time_span > 0:
                stats['message_rate'] = len(self._message_history) / time_span
            else:
                stats['message_rate'] = 0.0
        else:
            stats['message_rate'] = 0.0

        return stats

    def recommend_threshold(self) -> float:
        """
        Recommend optimal threshold based on observed patterns.
        根据观察到的模式推荐最佳阈值。

        Returns:
            Recommended threshold in KB
        """
        patterns = self.analyze_message_patterns()

        if not patterns:
            # Use adaptive threshold based on network
            return self.adaptive_threshold(self.bandwidth_gbps, self.latency_us)

        # If most messages are small, use a larger threshold to batch more
        small_ratio = patterns.get('small_message_ratio', 0.0)

        base_threshold = self.adaptive_threshold(
            self.bandwidth_gbps, self.latency_us
        )

        # Adjust based on small message ratio
        if small_ratio > 0.8:
            # Many small messages, increase threshold to batch more
            recommended = base_threshold * 1.5
        elif small_ratio < 0.2:
            # Few small messages, decrease threshold
            recommended = base_threshold * 0.7
        else:
            recommended = base_threshold

        # Clamp to reasonable range
        recommended = max(4.0, min(recommended, 1024.0))

        logger.info(
            f"Recommended threshold: {recommended:.2f} KB "
            f"(small_ratio={small_ratio:.2%}, base={base_threshold:.2f} KB)"
        )

        return recommended

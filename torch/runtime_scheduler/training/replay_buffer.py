"""
Replay buffer for storing and replaying execution traces.

Used for experience replay in online learning and offline training.
"""

import random
import threading
from collections import deque
from typing import List, Optional, Tuple, Any, Dict
import warnings


class ReplayBuffer:
    """
    Experience replay buffer for runtime scheduler training.

    Features:
    - Fixed-size circular buffer
    - Prioritized sampling
    - Multi-threaded access
    - Batch sampling
    """

    def __init__(
        self,
        capacity: int = 100000,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer capacity
            prioritized: Enable prioritized experience replay
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta

        # Storage
        self._buffer: deque = deque(maxlen=capacity)
        self._priorities: deque = deque(maxlen=capacity)

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "total_added": 0,
            "total_sampled": 0,
        }

    def add(
        self,
        experience: Any,
        priority: Optional[float] = None
    ) -> None:
        """
        Add an experience to the buffer.

        Args:
            experience: Experience data (any type)
            priority: Optional priority (for prioritized replay)
        """
        with self._lock:
            self._buffer.append(experience)

            if self.prioritized:
                # Use max priority for new experiences
                if priority is None:
                    priority = max(self._priorities) if self._priorities else 1.0
                self._priorities.append(priority)

            self._stats["total_added"] += 1

    def sample(
        self,
        batch_size: int,
        return_indices: bool = False
    ) -> Tuple[List[Any], Optional[List[int]]]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Batch size
            return_indices: Return sampled indices

        Returns:
            Batch of experiences and optionally indices
        """
        with self._lock:
            if len(self._buffer) == 0:
                return [], [] if return_indices else None

            batch_size = min(batch_size, len(self._buffer))

            if self.prioritized:
                indices, experiences = self._prioritized_sample(batch_size)
            else:
                indices = random.sample(range(len(self._buffer)), batch_size)
                experiences = [self._buffer[i] for i in indices]

            self._stats["total_sampled"] += len(experiences)

            if return_indices:
                return experiences, indices
            return experiences, None

    def _prioritized_sample(
        self,
        batch_size: int
    ) -> Tuple[List[int], List[Any]]:
        """Sample using prioritized experience replay."""
        # Compute sampling probabilities
        priorities = list(self._priorities)
        priorities_alpha = [p ** self.alpha for p in priorities]
        total_priority = sum(priorities_alpha)

        if total_priority == 0:
            # Fallback to uniform sampling
            indices = random.sample(range(len(self._buffer)), batch_size)
            experiences = [self._buffer[i] for i in indices]
            return indices, experiences

        probs = [p / total_priority for p in priorities_alpha]

        # Sample indices based on probabilities
        indices = random.choices(
            range(len(self._buffer)),
            weights=probs,
            k=batch_size
        )

        experiences = [self._buffer[i] for i in indices]

        return indices, experiences

    def update_priorities(
        self,
        indices: List[int],
        priorities: List[float]
    ) -> None:
        """
        Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences
            priorities: New priorities
        """
        if not self.prioritized:
            return

        with self._lock:
            for idx, priority in zip(indices, priorities):
                if 0 <= idx < len(self._priorities):
                    self._priorities[idx] = priority

    def __len__(self) -> int:
        """Get buffer size."""
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._priorities.clear()
            self._stats["total_added"] = 0
            self._stats["total_sampled"] = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["current_size"] = len(self._buffer)
            stats["capacity"] = self.capacity

        return stats

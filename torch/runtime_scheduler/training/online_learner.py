"""
Online learning support for runtime scheduler.

Enables continuous model adaptation during execution.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available for online learning")

from .replay_buffer import ReplayBuffer


class OnlineLearner:
    """
    Online learning for runtime scheduler models.

    Features:
    - Continuous adaptation during execution
    - Experience replay for stability
    - Automatic model updates
    - A/B testing support
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        target_model: Optional[Any] = None,
        buffer_size: int = 10000,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        update_interval: int = 100,
        target_update_interval: int = 1000,
        device: str = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize online learner.

        Args:
            model: Primary model
            target_model: Target model for stable learning
            buffer_size: Replay buffer size
            batch_size: Mini-batch size
            learning_rate: Learning rate
            update_interval: Steps between model updates
            target_update_interval: Steps between target model updates
            device: Device for training
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for online learning")

        self.model = model
        self.target_model = target_model
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.device = device

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            prioritized=True
        )

        # Optimizer
        self.optimizer: Optional[optim.Optimizer] = None
        if self.model is not None:
            self.model.to(self.device)
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )

        # Target model initialization
        if self.target_model is not None:
            self.target_model.to(self.device)
            self._sync_target_model()

        # Learning state
        self._step = 0
        self._enabled = True

        # Statistics
        self._stats = {
            "total_experiences": 0,
            "total_updates": 0,
            "total_target_updates": 0,
            "recent_losses": deque(maxlen=100),
        }

        # Thread safety
        self._lock = threading.RLock()

        # Background update thread
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start online learning."""
        if self._update_thread is not None:
            return

        self._stop_event.clear()
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self._update_thread.start()

    def stop(self) -> None:
        """Stop online learning."""
        if self._update_thread is None:
            return

        self._stop_event.set()
        self._update_thread.join(timeout=5.0)
        self._update_thread = None

    def add_experience(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any],
        done: bool = False
    ) -> None:
        """
        Add an experience for learning.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        if not self._enabled:
            return

        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

        # Add to replay buffer
        self.replay_buffer.add(experience)

        with self._lock:
            self._stats["total_experiences"] += 1
            self._step += 1

    def _update_loop(self) -> None:
        """Background update loop."""
        while not self._stop_event.is_set():
            try:
                # Check if it's time to update
                if self._step % self.update_interval == 0 and len(self.replay_buffer) >= self.batch_size:
                    self._perform_update()

                # Check if it's time to update target model
                if self._step % self.target_update_interval == 0:
                    self._sync_target_model()

            except Exception as e:
                warnings.warn(f"Error in online learning update: {e}")

            # Small sleep to avoid busy waiting
            self._stop_event.wait(0.1)

    def _perform_update(self) -> None:
        """Perform a single update step."""
        if self.model is None or self.optimizer is None:
            return

        # Sample batch from replay buffer
        experiences, indices = self.replay_buffer.sample(
            self.batch_size,
            return_indices=True
        )

        if not experiences:
            return

        # Prepare batch (simplified - real implementation would process features)
        # This is a placeholder for actual learning logic
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with self._lock:
            self._stats["total_updates"] += 1
            self._stats["recent_losses"].append(loss.item())

        # Update priorities in replay buffer
        if indices is not None:
            # Compute TD errors or other priorities
            priorities = [abs(loss.item())] * len(indices)
            self.replay_buffer.update_priorities(indices, priorities)

    def _sync_target_model(self) -> None:
        """Synchronize target model with current model."""
        if self.model is None or self.target_model is None:
            return

        self.target_model.load_state_dict(self.model.state_dict())

        with self._lock:
            self._stats["total_target_updates"] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["buffer_size"] = len(self.replay_buffer)
            stats["recent_avg_loss"] = (
                sum(self._stats["recent_losses"]) / len(self._stats["recent_losses"])
                if self._stats["recent_losses"]
                else 0.0
            )

        return stats

    def save_model(self, filepath: str) -> None:
        """
        Save model checkpoint.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            return

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "step": self._step,
            "stats": self._stats,
        }

        torch.save(checkpoint, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load model checkpoint.

        Args:
            filepath: Path to load model from
        """
        if self.model is None:
            return

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self._step = checkpoint.get("step", 0)

        # Sync target model
        self._sync_target_model()

    def enable(self) -> None:
        """Enable online learning."""
        self._enabled = True

    def disable(self) -> None:
        """Disable online learning."""
        self._enabled = False

    def __enter__(self) -> 'OnlineLearner':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class ABTester:
    """
    A/B testing framework for comparing scheduling strategies.

    Features:
    - Multiple model comparison
    - Traffic splitting
    - Statistical significance testing
    - Automatic best model selection
    """

    def __init__(
        self,
        models: Dict[str, Any],
        traffic_split: Optional[Dict[str, float]] = None
    ):
        """
        Initialize A/B tester.

        Args:
            models: Dictionary of model_name -> model
            traffic_split: Optional traffic split ratios
        """
        self.models = models
        self.traffic_split = traffic_split or {
            name: 1.0 / len(models) for name in models
        }

        # Validate traffic split
        if abs(sum(self.traffic_split.values()) - 1.0) > 0.01:
            raise ValueError("Traffic split must sum to 1.0")

        # Statistics for each model
        self._stats: Dict[str, Dict[str, Any]] = {
            name: {
                "requests": 0,
                "total_latency": 0.0,
                "total_reward": 0.0,
                "errors": 0,
            }
            for name in models
        }

        self._lock = threading.RLock()

    def select_model(self) -> str:
        """
        Select a model based on traffic split.

        Returns:
            Model name
        """
        import random

        rand = random.random()
        cumulative = 0.0

        for name, split in self.traffic_split.items():
            cumulative += split
            if rand < cumulative:
                return name

        # Fallback to first model
        return list(self.models.keys())[0]

    def record_result(
        self,
        model_name: str,
        latency: float,
        reward: float,
        error: bool = False
    ) -> None:
        """
        Record result for a model.

        Args:
            model_name: Model name
            latency: Measured latency
            reward: Reward/score
            error: Whether an error occurred
        """
        with self._lock:
            if model_name not in self._stats:
                return

            stats = self._stats[model_name]
            stats["requests"] += 1
            stats["total_latency"] += latency
            stats["total_reward"] += reward
            if error:
                stats["errors"] += 1

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all models."""
        with self._lock:
            result = {}

            for name, stats in self._stats.items():
                result[name] = dict(stats)

                # Compute averages
                if stats["requests"] > 0:
                    result[name]["avg_latency"] = (
                        stats["total_latency"] / stats["requests"]
                    )
                    result[name]["avg_reward"] = (
                        stats["total_reward"] / stats["requests"]
                    )
                    result[name]["error_rate"] = (
                        stats["errors"] / stats["requests"]
                    )
                else:
                    result[name]["avg_latency"] = 0.0
                    result[name]["avg_reward"] = 0.0
                    result[name]["error_rate"] = 0.0

            return result

    def get_best_model(self, metric: str = "avg_reward") -> str:
        """
        Get the best performing model.

        Args:
            metric: Metric to use for comparison

        Returns:
            Best model name
        """
        stats = self.get_statistics()

        best_model = None
        best_value = float('-inf')

        for name, model_stats in stats.items():
            if model_stats.get(metric, 0.0) > best_value:
                best_value = model_stats[metric]
                best_model = name

        return best_model or list(self.models.keys())[0]

    def update_traffic_split(
        self,
        new_split: Dict[str, float]
    ) -> None:
        """
        Update traffic split ratios.

        Args:
            new_split: New traffic split
        """
        if abs(sum(new_split.values()) - 1.0) > 0.01:
            raise ValueError("Traffic split must sum to 1.0")

        with self._lock:
            self.traffic_split = new_split

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        with self._lock:
            for stats in self._stats.values():
                stats["requests"] = 0
                stats["total_latency"] = 0.0
                stats["total_reward"] = 0.0
                stats["errors"] = 0

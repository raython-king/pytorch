"""
Training infrastructure for runtime scheduler models.

Provides offline training on collected traces.
"""

import time
import os
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available for training")

from .data_collector import RuntimeDataCollector, SchedulingExample
from .replay_buffer import ReplayBuffer


class SchedulingDataset(Dataset):
    """Dataset for scheduling examples."""

    def __init__(self, examples: List[SchedulingExample]):
        """
        Initialize dataset.

        Args:
            examples: List of scheduling examples
        """
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Convert to tensors (simplified)
        # In production, this would do proper feature encoding
        return {
            "features": example.features,
            "target_device": example.target_device,
            "target_latency": example.target_latency,
            "alternatives": example.alternative_devices,
        }


class RuntimeTrainer:
    """
    Trainer for runtime scheduler models.

    Features:
    - Offline training on collected traces
    - Model checkpointing
    - Training metrics and logging
    - Multiple training objectives
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        device: str = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train (None for default)
            device: Device for training
            learning_rate: Learning rate
            batch_size: Batch size
            checkpoint_dir: Directory for checkpoints
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for training")

        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer (initialized when model is set)
        self.optimizer: Optional[optim.Optimizer] = None

        # Training state
        self._epoch = 0
        self._step = 0

        # Metrics
        self._metrics = {
            "train_loss": [],
            "val_loss": [],
            "best_val_loss": float('inf'),
        }

        # Initialize optimizer if model provided
        if self.model is not None:
            self._initialize_optimizer()

    def _initialize_optimizer(self) -> None:
        """Initialize optimizer."""
        if self.model is None:
            return

        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

    def train(
        self,
        data_collector: RuntimeDataCollector,
        num_epochs: int = 10,
        val_split: float = 0.2,
        early_stopping_patience: int = 5,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """
        Train model on collected data.

        Args:
            data_collector: Data collector with traces
            num_epochs: Number of training epochs
            val_split: Validation split fraction
            early_stopping_patience: Patience for early stopping
            log_interval: Steps between logging

        Returns:
            Training metrics
        """
        if self.model is None:
            raise ValueError("Model not set")

        # Create training examples
        print("Creating training examples...")
        examples = data_collector.create_training_examples(
            include_alternatives=True
        )

        if not examples:
            raise ValueError("No training examples available")

        print(f"Created {len(examples)} training examples")

        # Split into train/val
        split_idx = int(len(examples) * (1 - val_split))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

        # Create datasets and loaders
        train_dataset = SchedulingDataset(train_examples)
        val_dataset = SchedulingDataset(val_examples)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Simplified
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self._epoch = epoch

            # Train
            train_loss = self._train_epoch(train_loader, log_interval)
            self._metrics["train_loss"].append(train_loss)

            # Validate
            val_loss = self._validate(val_loader)
            self._metrics["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._metrics["best_val_loss"] = best_val_loss
                self.save_checkpoint("best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        # Load best model
        self.load_checkpoint("best_model.pt")

        return self._metrics

    def _train_epoch(
        self,
        train_loader: DataLoader,
        log_interval: int
    ) -> float:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Forward pass (simplified - real implementation would process features)
            # This is a placeholder for actual training logic
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self._step += 1

            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                print(f"  Step {self._step}, Batch {batch_idx + 1}: Loss = {avg_loss:.4f}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Forward pass (simplified)
                loss = torch.tensor(0.0, device=self.device)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        if self.model is None:
            return

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self._epoch,
            "step": self._step,
            "metrics": self._metrics,
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        if self.model is None:
            return

        filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            warnings.warn(f"Checkpoint not found: {filepath}")
            return

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epoch = checkpoint["epoch"]
        self._step = checkpoint["step"]
        self._metrics = checkpoint["metrics"]

        print(f"Checkpoint loaded: {filepath}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return dict(self._metrics)


class TrainingPipeline:
    """
    End-to-end training pipeline.

    Coordinates data collection, training, and model deployment.
    """

    def __init__(
        self,
        data_collector: RuntimeDataCollector,
        trainer: RuntimeTrainer,
        min_examples: int = 1000
    ):
        """
        Initialize training pipeline.

        Args:
            data_collector: Data collector
            trainer: Model trainer
            min_examples: Minimum examples before training
        """
        self.data_collector = data_collector
        self.trainer = trainer
        self.min_examples = min_examples

        # State
        self._last_training_time = 0.0
        self._training_count = 0

    def should_train(self) -> bool:
        """Check if enough data is available for training."""
        stats = self.data_collector.get_statistics()
        return stats["traces_in_memory"] >= self.min_examples

    def run_training(self, **train_kwargs) -> Dict[str, Any]:
        """
        Run training with collected data.

        Args:
            **train_kwargs: Arguments for trainer.train()

        Returns:
            Training metrics
        """
        if not self.should_train():
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "traces_available": self.data_collector.get_statistics()["traces_in_memory"],
                "traces_required": self.min_examples
            }

        print("Starting training...")
        start_time = time.time()

        try:
            metrics = self.trainer.train(
                self.data_collector,
                **train_kwargs
            )

            duration = time.time() - start_time
            self._last_training_time = time.time()
            self._training_count += 1

            metrics["status"] = "success"
            metrics["duration"] = duration
            metrics["training_count"] = self._training_count

            print(f"Training completed in {duration:.2f} seconds")

            return metrics

        except Exception as e:
            print(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "data_collector": self.data_collector.get_statistics(),
            "trainer": self.trainer.get_metrics(),
            "last_training_time": self._last_training_time,
            "training_count": self._training_count,
        }

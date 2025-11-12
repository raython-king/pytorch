"""
ML Scheduler Trainer

Implements training infrastructure for ML-based scheduler models.
Supports three training paradigms:
1. Supervised Learning: Learn from heuristic scheduler decisions
2. Imitation Learning: Mimic expert traces
3. Reinforcement Learning: Optimize for measured performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import json

from ..models.gnn_model import FusionGNN, SchedulingGNN
from .dataset import IRGraphDataset, collate_graphs

log = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training mode
    mode: str = "supervised"  # 'supervised', 'imitation', 'rl'

    # Model settings
    model_type: str = "fusion_gnn"  # 'fusion_gnn', 'scheduling_gnn'
    node_feat_dim: int = 64
    edge_feat_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5

    # Loss weights
    fusion_loss_weight: float = 1.0
    schedule_loss_weight: float = 0.5
    performance_loss_weight: float = 2.0

    # Multi-GPU training
    use_distributed: bool = False
    local_rank: int = 0
    world_size: int = 1

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5  # Save checkpoint every N epochs
    keep_last_n: int = 3  # Keep last N checkpoints

    # Logging
    log_dir: str = "./logs"
    log_every: int = 10  # Log every N steps
    eval_every: int = 100  # Evaluate every N steps

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Gradient clipping
    grad_clip_norm: float = 1.0

    # Mixed precision training
    use_amp: bool = True

    # Reproducibility
    seed: int = 42

    # RL-specific (for reinforcement learning mode)
    rl_discount_factor: float = 0.99
    rl_entropy_coef: float = 0.01
    rl_value_loss_coef: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class MLSchedulerTrainer:
    """
    Trainer for ML scheduler models.

    Supports multiple training paradigms and handles all aspects of training:
    - Data loading and batching
    - Model optimization
    - Checkpointing and resuming
    - Logging and monitoring
    - Multi-GPU training
    - Hyperparameter tuning

    Example:
        config = TrainingConfig(
            mode='supervised',
            num_epochs=50,
            batch_size=32,
        )

        trainer = MLSchedulerTrainer(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: IRGraphDataset,
        val_dataset: Optional[IRGraphDataset] = None,
        test_dataset: Optional[IRGraphDataset] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            test_dataset: Test dataset (optional)
        """
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Device setup
        self.device = self._setup_device()

        # Model
        self.model = self._create_model()
        self.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_lr_scheduler()

        # Loss functions
        self.loss_fn = self._create_loss_fn()

        # Data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None
        self.test_loader = self._create_dataloader(test_dataset, shuffle=False) if test_dataset else None

        # Logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        log.info(f"TensorBoard logging to: {config.log_dir}")

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        log.info(f"MLSchedulerTrainer initialized: mode={config.mode}, device={self.device}")

    def _setup_device(self) -> torch.device:
        """Setup device for training."""
        if self.config.use_distributed:
            torch.cuda.set_device(self.config.local_rank)
            return torch.device(f'cuda:{self.config.local_rank}')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def _create_model(self) -> nn.Module:
        """Create model based on config."""
        if self.config.model_type == "fusion_gnn":
            model = FusionGNN(
                node_feat_dim=self.config.node_feat_dim,
                edge_feat_dim=self.config.edge_feat_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )
        elif self.config.model_type == "scheduling_gnn":
            model = SchedulingGNN(
                node_feat_dim=self.config.node_feat_dim,
                edge_feat_dim=self.config.edge_feat_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

        log.info(f"Created model: {self.config.model_type}")
        log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return optimizer

    def _create_lr_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
            )
        elif self.config.lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif self.config.lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5,
            )
        else:
            scheduler = None

        return scheduler

    def _create_loss_fn(self) -> Callable:
        """Create loss function based on training mode."""
        if self.config.mode == "supervised":
            return self._supervised_loss
        elif self.config.mode == "imitation":
            return self._imitation_loss
        elif self.config.mode == "rl":
            return self._rl_loss
        else:
            raise ValueError(f"Unknown training mode: {self.config.mode}")

    def _create_dataloader(
        self,
        dataset: Optional[IRGraphDataset],
        shuffle: bool
    ) -> Optional[DataLoader]:
        """Create data loader."""
        if dataset is None:
            return None

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_graphs,
            num_workers=4,
            pin_memory=True,
        )

    def train(self):
        """Main training loop."""
        log.info("Starting training...")
        log.info(f"Training for {self.config.num_epochs} epochs")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self._train_epoch()

            # Log epoch metrics
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            log.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"train_loss: {train_metrics['loss']:.4f}"
            )

            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate()
                self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)

                log.info(f"Validation - loss: {val_metrics['loss']:.4f}")

                # Check for improvement
                if val_metrics['loss'] < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0

                    # Save best model
                    self.save_checkpoint('best_model.pt')
                    log.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    log.info(f"No improvement for {self.patience_counter} epochs")

                # Early stopping
                if self.patience_counter >= self.config.patience:
                    log.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Learning rate scheduling
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_metrics['loss'] if self.val_loader else train_metrics['loss'])
                else:
                    self.lr_scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        # Final evaluation on test set
        if self.test_loader is not None:
            log.info("Running final evaluation on test set...")
            test_metrics = self._evaluate(self.test_loader)
            log.info(f"Test metrics: {test_metrics}")

        log.info("Training completed!")
        self.writer.close()

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._to_device(batch)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    batch['x'],
                    batch['edge_index'],
                    batch.get('edge_attr', None),
                    batch.get('batch', None),
                )

                # Compute loss
                loss = self.loss_fn(outputs, batch)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip_norm
                    )

                self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if (batch_idx + 1) % self.config.log_every == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                log.debug(f"Batch {batch_idx + 1}/{len(self.train_loader)}, loss: {loss.item():.4f}")

        return {
            'loss': total_loss / num_batches,
        }

    def _validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        return self._evaluate(self.val_loader)

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = self._to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    batch['x'],
                    batch['edge_index'],
                    batch.get('edge_attr', None),
                    batch.get('batch', None),
                )

                loss = self.loss_fn(outputs, batch)

            total_loss += loss.item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
        }

    def _supervised_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Supervised learning loss.

        Learn from heuristic scheduler decisions (ground truth labels).
        """
        loss = 0.0

        # Fusion loss (if present)
        if 'fusion_matrix' in outputs and 'y_fusion' in batch:
            fusion_pred = outputs['fusion_matrix']

            # Handle batched graphs
            if isinstance(batch['y_fusion'], list):
                # Compute loss for each graph in batch
                fusion_losses = []
                node_offset = 0

                for graph_fusion_labels in batch['y_fusion']:
                    num_nodes = graph_fusion_labels.size(0)
                    graph_fusion_pred = fusion_pred[node_offset:node_offset + num_nodes, node_offset:node_offset + num_nodes]

                    # Binary cross-entropy
                    graph_loss = nn.functional.binary_cross_entropy(
                        graph_fusion_pred,
                        graph_fusion_labels.to(self.device)
                    )
                    fusion_losses.append(graph_loss)

                    node_offset += num_nodes

                fusion_loss = torch.stack(fusion_losses).mean()
            else:
                fusion_loss = nn.functional.binary_cross_entropy(
                    fusion_pred,
                    batch['y_fusion']
                )

            loss += self.config.fusion_loss_weight * fusion_loss

        # Schedule order loss (if present)
        if 'priority_scores' in outputs and 'y_schedule' in batch:
            # Ranking loss or MSE loss
            if isinstance(batch['y_schedule'], list):
                schedule_losses = []
                node_offset = 0

                for graph_schedule_labels in batch['y_schedule']:
                    num_nodes = len(graph_schedule_labels)
                    graph_schedule_pred = outputs['priority_scores'][node_offset:node_offset + num_nodes]

                    # MSE loss
                    graph_loss = nn.functional.mse_loss(
                        graph_schedule_pred,
                        graph_schedule_labels.float().to(self.device)
                    )
                    schedule_losses.append(graph_loss)

                    node_offset += num_nodes

                schedule_loss = torch.stack(schedule_losses).mean()
            else:
                schedule_loss = nn.functional.mse_loss(
                    outputs['priority_scores'],
                    batch['y_schedule'].float()
                )

            loss += self.config.schedule_loss_weight * schedule_loss

        # Performance metric loss (if present)
        if 'y_performance' in batch:
            # This would require a performance predictor head
            # Placeholder for now
            pass

        return loss

    def _imitation_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Imitation learning loss.

        Learn to mimic expert traces using behavior cloning.
        Similar to supervised loss but with additional techniques like GAIL or DAgger.
        """
        # For now, use supervised loss as base
        # Can be extended with adversarial imitation learning
        return self._supervised_loss(outputs, batch)

    def _rl_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Reinforcement learning loss (PPO-style).

        Optimize for measured performance (compilation time, runtime, etc.).
        """
        # This requires an environment and reward signal
        # Placeholder for RL training pipeline

        # Policy loss (actor)
        policy_loss = 0.0

        # Value loss (critic)
        value_loss = 0.0

        # Entropy bonus
        entropy = 0.0

        total_loss = (
            policy_loss
            + self.config.rl_value_loss_coef * value_loss
            - self.config.rl_entropy_coef * entropy
        )

        return total_loss

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                batch[key] = [v.to(self.device) for v in value]

        return batch

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        log.info(f"Checkpoint saved: {checkpoint_path}")

        # Keep only last N checkpoints (except best model)
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: p.stat().st_mtime
        )

        if len(checkpoints) > self.config.keep_last_n:
            for checkpoint in checkpoints[:-self.config.keep_last_n]:
                checkpoint.unlink()
                log.debug(f"Removed old checkpoint: {checkpoint}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        log.info(f"Checkpoint loaded: {checkpoint_path}")
        log.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def export_model(self, output_path: str):
        """Export trained model for inference."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(self.model.state_dict(), output_path)

        # Save config
        config_path = output_path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        log.info(f"Model exported to: {output_path}")
        log.info(f"Config saved to: {config_path}")

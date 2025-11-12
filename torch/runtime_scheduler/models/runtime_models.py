"""
ML Models for Runtime Scheduling Decisions.

This module implements ML models for runtime decisions:
- RuntimeGNN: Graph Neural Network for operation scheduling
- PriorityPredictor: Predicts operation priority
- BatchingPredictor: Identifies batching opportunities
- LatencyPredictor: Predicts operation latency

All models are optimized for fast inference (< 1ms) and support online learning.
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Deque
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PredictionResult:
    """Result from a model prediction."""
    value: float
    confidence: float
    inference_time_us: float


class FastGNNLayer(nn.Module):
    """
    Fast Graph Neural Network layer optimized for runtime scheduling.

    Uses simplified message passing for speed.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Initialize GNN layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Linear transformations
        self.W_msg = nn.Linear(in_features, out_features, bias=False)
        self.W_self = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_features: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_weights: Optional edge weights [num_edges]

        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Message passing
        messages = self.W_msg(node_features)

        # Aggregate messages from neighbors
        num_nodes = node_features.size(0)
        aggregated = torch.zeros(num_nodes, self.out_features, device=node_features.device)

        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]

            # Weighted aggregation if weights provided
            if edge_weights is not None:
                messages_weighted = messages[src] * edge_weights.unsqueeze(-1)
                aggregated.index_add_(0, dst, messages_weighted)
            else:
                aggregated.index_add_(0, dst, messages[src])

        # Self-transformation
        self_features = self.W_self(node_features)

        # Combine and apply activation
        out = aggregated + self_features + self.bias
        return F.relu(out)


class RuntimeGNN(nn.Module):
    """
    Graph Neural Network for operation scheduling.

    Models the dependency graph and predicts optimal scheduling decisions.
    Optimized for fast inference.
    """

    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize RuntimeGNN.

        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        in_dim = node_feature_dim
        for _ in range(num_layers):
            self.gnn_layers.append(FastGNNLayer(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_features: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weights: Optional edge weights [num_edges]

        Returns:
            Node scores [num_nodes, 1]
        """
        x = node_features

        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_weights)
            x = self.dropout(x)

        # Output projection
        scores = self.output_proj(x)
        return scores

    @torch.no_grad()
    def predict(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fast inference without gradients.

        Args:
            node_features: Node features
            edge_index: Edge connectivity
            edge_weights: Optional edge weights

        Returns:
            Node scores
        """
        self.eval()
        return self(node_features, edge_index, edge_weights)


class PriorityPredictor(nn.Module):
    """
    Predicts operation priority for scheduling.

    Fast feedforward network with ensemble predictions.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_ensembles: int = 3
    ):
        """
        Initialize PriorityPredictor.

        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_ensembles: Number of ensemble models
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_ensembles = num_ensembles

        # Ensemble of small networks
        self.ensembles = nn.ModuleList()
        for _ in range(num_ensembles):
            ensemble = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.ensembles.append(ensemble)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with ensemble predictions.

        Args:
            features: Input features [batch_size, feature_dim]

        Returns:
            Tuple of (mean_priority, std_priority)
        """
        predictions = []
        for ensemble in self.ensembles:
            pred = ensemble(features)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # [num_ensembles, batch_size, 1]

        # Compute mean and std for confidence
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor
    ) -> PredictionResult:
        """
        Fast inference with confidence estimation.

        Args:
            features: Input features [feature_dim]

        Returns:
            PredictionResult with priority, confidence, and timing
        """
        self.eval()
        start = time.perf_counter()

        features = features.unsqueeze(0)  # Add batch dim
        mean_pred, std_pred = self(features)

        priority = mean_pred.item()
        # Confidence inversely proportional to std
        confidence = 1.0 / (1.0 + std_pred.item())

        elapsed_us = (time.perf_counter() - start) * 1_000_000

        return PredictionResult(
            value=priority,
            confidence=confidence,
            inference_time_us=elapsed_us
        )


class BatchingPredictor(nn.Module):
    """
    Identifies batching opportunities for similar operations.

    Predicts whether operations can be efficiently batched together.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 64
    ):
        """
        Initialize BatchingPredictor.

        Args:
            feature_dim: Input feature dimension (for pair of ops)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Small network for speed
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),  # Concatenated features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features_a: Features for first operation [batch_size, feature_dim]
            features_b: Features for second operation [batch_size, feature_dim]

        Returns:
            Batching score [batch_size, 1]
        """
        # Concatenate features
        combined = torch.cat([features_a, features_b], dim=-1)
        return self.network(combined)

    @torch.no_grad()
    def predict(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> PredictionResult:
        """
        Fast inference for batching decision.

        Args:
            features_a: Features for first operation
            features_b: Features for second operation

        Returns:
            PredictionResult with batching score
        """
        self.eval()
        start = time.perf_counter()

        features_a = features_a.unsqueeze(0)
        features_b = features_b.unsqueeze(0)

        score = self(features_a, features_b).item()

        # Confidence based on how close to 0 or 1
        confidence = abs(2 * score - 1)  # High confidence when close to 0 or 1

        elapsed_us = (time.perf_counter() - start) * 1_000_000

        return PredictionResult(
            value=score,
            confidence=confidence,
            inference_time_us=elapsed_us
        )


class LatencyPredictor(nn.Module):
    """
    Predicts operation latency for scheduling decisions.

    Uses historical data and current system state.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128
    ):
        """
        Initialize LatencyPredictor.

        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Network for latency prediction
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive output
        )

        # Uncertainty estimation head
        self.uncertainty = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Args:
            features: Input features [batch_size, feature_dim]

        Returns:
            Tuple of (predicted_latency_ms, uncertainty)
        """
        latency = self.network(features)
        uncertainty = self.uncertainty(features)
        return latency, uncertainty

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor
    ) -> PredictionResult:
        """
        Fast inference with uncertainty.

        Args:
            features: Input features [feature_dim]

        Returns:
            PredictionResult with latency and confidence
        """
        self.eval()
        start = time.perf_counter()

        features = features.unsqueeze(0)
        latency, uncertainty = self(features)

        latency_ms = latency.item()
        # Confidence inversely related to uncertainty
        confidence = 1.0 / (1.0 + uncertainty.item())

        elapsed_us = (time.perf_counter() - start) * 1_000_000

        return PredictionResult(
            value=latency_ms,
            confidence=confidence,
            inference_time_us=elapsed_us
        )


@dataclass
class TrainingExample:
    """Training example for online learning."""
    features: torch.Tensor
    target: float
    weight: float = 1.0
    timestamp: float = 0.0


class OnlineLearner:
    """
    Online learning manager for runtime models.

    Supports incremental updates with experience replay.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        buffer_size: int = 10000,
        batch_size: int = 32,
        update_interval: int = 100
    ):
        """
        Initialize online learner.

        Args:
            model: PyTorch model to train
            learning_rate: Learning rate
            buffer_size: Size of replay buffer
            batch_size: Batch size for updates
            update_interval: Update model every N examples
        """
        self.model = model
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_interval = update_interval

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer: Deque[TrainingExample] = deque(maxlen=buffer_size)

        # Training state
        self.examples_seen = 0
        self.updates_performed = 0
        self._lock = threading.Lock()

    def add_example(
        self,
        features: torch.Tensor,
        target: float,
        weight: float = 1.0
    ) -> None:
        """
        Add a training example to the replay buffer.

        Args:
            features: Input features
            target: Target value
            weight: Example weight (for importance sampling)
        """
        with self._lock:
            example = TrainingExample(
                features=features.detach().cpu(),
                target=target,
                weight=weight,
                timestamp=time.time()
            )
            self.replay_buffer.append(example)
            self.examples_seen += 1

    def should_update(self) -> bool:
        """Check if model should be updated."""
        return (
            self.examples_seen > 0 and
            self.examples_seen % self.update_interval == 0 and
            len(self.replay_buffer) >= self.batch_size
        )

    def update(self) -> Dict[str, float]:
        """
        Perform a model update using replay buffer.

        Returns:
            Dictionary with training metrics
        """
        with self._lock:
            if len(self.replay_buffer) < self.batch_size:
                return {"loss": 0.0, "examples": 0}

            # Sample from replay buffer
            indices = torch.randperm(len(self.replay_buffer))[:self.batch_size]
            batch = [self.replay_buffer[i] for i in indices]

            # Prepare batch
            features = torch.stack([ex.features for ex in batch])
            targets = torch.tensor([ex.target for ex in batch], dtype=torch.float32)
            weights = torch.tensor([ex.weight for ex in batch], dtype=torch.float32)

            # Move to model device
            device = next(self.model.parameters()).device
            features = features.to(device)
            targets = targets.to(device)
            weights = weights.to(device)

            # Forward pass
            self.model.train()
            predictions = self.model(features)

            # Handle different output formats
            if isinstance(predictions, tuple):
                predictions = predictions[0]  # Use mean prediction

            predictions = predictions.squeeze()

            # Compute weighted loss
            loss = F.mse_loss(predictions, targets, reduction='none')
            loss = (loss * weights).mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            self.updates_performed += 1

            return {
                "loss": loss.item(),
                "examples": len(batch),
                "updates": self.updates_performed
            }

    def get_stats(self) -> Dict[str, int]:
        """Get learning statistics."""
        return {
            "examples_seen": self.examples_seen,
            "buffer_size": len(self.replay_buffer),
            "updates_performed": self.updates_performed
        }


class EnsembleScheduler:
    """
    Ensemble of multiple models for robust predictions.

    Combines predictions from multiple models with confidence weighting.
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble.

        Args:
            models: List of PyTorch models
            weights: Optional weights for each model
        """
        self.models = models
        self.num_models = len(models)

        if weights is None:
            weights = [1.0 / self.num_models] * self.num_models
        self.weights = weights

        self._lock = threading.Lock()

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> PredictionResult:
        """
        Ensemble prediction with confidence.

        Args:
            features: Input features

        Returns:
            PredictionResult with ensemble prediction
        """
        start = time.perf_counter()

        predictions = []
        confidences = []

        with self._lock:
            for model in self.models:
                model.eval()
                if hasattr(model, 'predict'):
                    result = model.predict(features)
                    predictions.append(result.value)
                    confidences.append(result.confidence)
                else:
                    # Fallback for models without predict method
                    pred = model(features.unsqueeze(0))
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    predictions.append(pred.item())
                    confidences.append(1.0)

        # Weighted average
        weighted_pred = sum(
            p * w * c
            for p, w, c in zip(predictions, self.weights, confidences)
        ) / sum(w * c for w, c in zip(self.weights, confidences))

        # Overall confidence (mean of individual confidences)
        overall_confidence = sum(confidences) / len(confidences)

        elapsed_us = (time.perf_counter() - start) * 1_000_000

        return PredictionResult(
            value=weighted_pred,
            confidence=overall_confidence,
            inference_time_us=elapsed_us
        )

    def update_weights(self, new_weights: List[float]) -> None:
        """Update ensemble weights."""
        with self._lock:
            self.weights = new_weights


# Factory functions for creating models

def create_priority_predictor(
    feature_dim: int = 64,
    hidden_dim: int = 128,
    num_ensembles: int = 3
) -> PriorityPredictor:
    """Create and initialize a PriorityPredictor model."""
    model = PriorityPredictor(feature_dim, hidden_dim, num_ensembles)
    # Initialize weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


def create_latency_predictor(
    feature_dim: int = 64,
    hidden_dim: int = 128
) -> LatencyPredictor:
    """Create and initialize a LatencyPredictor model."""
    model = LatencyPredictor(feature_dim, hidden_dim)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


def create_batching_predictor(
    feature_dim: int = 64,
    hidden_dim: int = 64
) -> BatchingPredictor:
    """Create and initialize a BatchingPredictor model."""
    model = BatchingPredictor(feature_dim, hidden_dim)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


def create_runtime_gnn(
    node_feature_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 2
) -> RuntimeGNN:
    """Create and initialize a RuntimeGNN model."""
    model = RuntimeGNN(node_feature_dim, hidden_dim, num_layers)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model

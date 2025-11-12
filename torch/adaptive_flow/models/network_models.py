"""
Network Prediction Models for Adaptive Flow Control.

ML models for predicting network behavior:
- Path latency prediction
- Congestion point prediction
- Optimal route prediction
- Transfer time prediction
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class NetworkSample:
    """
    Training sample for network models.

    Attributes:
        timestamp: Sample timestamp
        topology_features: Network topology features
        link_states: State of all links
        flow_features: Flow characteristics
        target: Target value (latency, congestion, etc.)
        metadata: Additional metadata
    """
    timestamp: float
    topology_features: torch.Tensor
    link_states: torch.Tensor
    flow_features: torch.Tensor
    target: float
    metadata: Dict[str, Any] = None


class PathLatencyPredictor(nn.Module):
    """
    Predict end-to-end latency for a path through the network.

    Uses graph neural network to capture network topology and state.
    """

    def __init__(
        self,
        num_devices: int,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        """
        Initialize path latency predictor.

        Args:
            num_devices: Number of devices in topology
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
        """
        super().__init__()
        self.num_devices = num_devices
        self.hidden_dim = hidden_dim

        # Link state encoder
        self.link_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # bandwidth, latency, utilization, congestion, packet_loss, status
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Device encoder
        self.device_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # device type, memory, compute capability
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Graph neural network layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Path encoder (LSTM over path)
        self.path_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        logger.info(
            f"PathLatencyPredictor initialized: "
            f"devices={num_devices}, hidden={hidden_dim}, layers={num_layers}"
        )

    def forward(
        self,
        link_states: torch.Tensor,
        device_features: torch.Tensor,
        adjacency: torch.Tensor,
        path: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict path latency.

        Args:
            link_states: Link state features [num_links, 6]
            device_features: Device features [num_devices, 4]
            adjacency: Adjacency matrix [num_devices, num_devices]
            path: Path as device indices [path_length]

        Returns:
            Predicted latency (microseconds)
        """
        batch_size = link_states.size(0) if link_states.dim() > 2 else 1

        # Encode link states
        link_embeddings = self.link_encoder(link_states)  # [num_links, hidden]

        # Encode devices
        device_embeddings = self.device_encoder(device_features)  # [num_devices, hidden]

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            device_embeddings = gnn_layer(device_embeddings, link_embeddings, adjacency)

        # Extract path embeddings
        path_embeddings = device_embeddings[path]  # [path_length, hidden]

        # Encode path with LSTM
        if path_embeddings.dim() == 2:
            path_embeddings = path_embeddings.unsqueeze(0)  # Add batch dimension

        lstm_out, _ = self.path_lstm(path_embeddings)  # [1, path_length, hidden]

        # Use final hidden state for prediction
        path_encoding = lstm_out[:, -1, :]  # [1, hidden]

        # Predict latency
        latency = self.predictor(path_encoding)  # [1, 1]

        return latency.squeeze()


class GNNLayer(nn.Module):
    """Graph Neural Network layer for message passing."""

    def __init__(self, hidden_dim: int):
        """
        Initialize GNN layer.

        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_features: Node features [num_nodes, hidden]
            edge_features: Edge features [num_edges, hidden]
            adjacency: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Updated node features
        """
        num_nodes = node_features.size(0)

        # Aggregate messages from neighbors
        messages = torch.zeros_like(node_features)

        edge_idx = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adjacency[i, j] > 0 and edge_idx < edge_features.size(0):
                    # Concatenate source node and edge features
                    msg_input = torch.cat([node_features[j], edge_features[edge_idx]], dim=-1)
                    msg = self.message_net(msg_input)
                    messages[i] += msg
                    edge_idx += 1

        # Update node features
        update_input = torch.cat([node_features, messages], dim=-1)
        updated = self.update_net(update_input)

        return updated + node_features  # Residual connection


class CongestionPointPredictor(nn.Module):
    """
    Predict where congestion is likely to occur in the network.

    Outputs probability distribution over links.
    """

    def __init__(
        self,
        num_links: int,
        hidden_dim: int = 128,
        sequence_length: int = 10
    ):
        """
        Initialize congestion predictor.

        Args:
            num_links: Number of links in topology
            hidden_dim: Hidden dimension
            sequence_length: Length of time series to consider
        """
        super().__init__()
        self.num_links = num_links
        self.sequence_length = sequence_length

        # Time series encoder (LSTM)
        self.encoder = nn.LSTM(
            input_size=num_links * 4,  # utilization, bandwidth, latency, packet_loss per link
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_links),
            nn.Softmax(dim=-1)  # Probability distribution over links
        )

        logger.info(
            f"CongestionPointPredictor initialized: "
            f"links={num_links}, hidden={hidden_dim}, seq_len={sequence_length}"
        )

    def forward(self, link_history: torch.Tensor) -> torch.Tensor:
        """
        Predict congestion probability for each link.

        Args:
            link_history: Historical link states [batch, seq_len, num_links * 4]

        Returns:
            Congestion probabilities [batch, num_links]
        """
        # Encode time series
        lstm_out, _ = self.encoder(link_history)  # [batch, seq_len, hidden]

        # Apply attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use final timestep for prediction
        final_state = attended[:, -1, :]  # [batch, hidden]

        # Predict congestion points
        congestion_probs = self.predictor(final_state)  # [batch, num_links]

        return congestion_probs


class OptimalRoutePredictor(nn.Module):
    """
    Predict optimal route between two devices using reinforcement learning.

    Uses policy gradient to learn routing decisions.
    """

    def __init__(
        self,
        num_devices: int,
        hidden_dim: int = 128
    ):
        """
        Initialize optimal route predictor.

        Args:
            num_devices: Number of devices
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.num_devices = num_devices

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(num_devices * num_devices * 4 + 2, hidden_dim),  # topology + src/dst
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy network (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_devices),
            nn.Softmax(dim=-1)
        )

        # Value network (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        logger.info(
            f"OptimalRoutePredictor initialized: devices={num_devices}, hidden={hidden_dim}"
        )

    def forward(
        self,
        network_state: torch.Tensor,
        src: int,
        dst: int,
        current_device: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next hop and value estimate.

        Args:
            network_state: Flattened network state [num_devices * num_devices * 4]
            src: Source device
            dst: Destination device
            current_device: Current device in route

        Returns:
            Tuple of (action_probs, value_estimate)
        """
        # Add source and destination to state
        src_dst = torch.tensor([src, dst], dtype=torch.float32)
        state = torch.cat([network_state, src_dst])

        # Encode state
        encoded = self.state_encoder(state.unsqueeze(0))  # [1, hidden]

        # Get action probabilities
        action_probs = self.policy(encoded)  # [1, num_devices]

        # Get value estimate
        value = self.value(encoded)  # [1, 1]

        return action_probs.squeeze(), value.squeeze()

    def select_action(
        self,
        network_state: torch.Tensor,
        src: int,
        dst: int,
        current_device: int,
        valid_neighbors: List[int]
    ) -> int:
        """
        Select next hop using policy.

        Args:
            network_state: Network state
            src: Source device
            dst: Destination device
            current_device: Current device
            valid_neighbors: List of valid next hops

        Returns:
            Selected next hop device ID
        """
        action_probs, _ = self.forward(network_state, src, dst, current_device)

        # Mask invalid actions
        mask = torch.zeros(self.num_devices)
        for neighbor in valid_neighbors:
            mask[neighbor] = 1.0

        masked_probs = action_probs * mask
        masked_probs = masked_probs / masked_probs.sum()  # Renormalize

        # Sample action
        action = torch.multinomial(masked_probs, 1).item()

        return action


class TransferTimePredictor(nn.Module):
    """
    Predict end-to-end transfer time for a given flow.

    Considers network state, flow size, and routing.
    """

    def __init__(
        self,
        num_devices: int,
        hidden_dim: int = 128
    ):
        """
        Initialize transfer time predictor.

        Args:
            num_devices: Number of devices
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # Flow encoder
        self.flow_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # src, dst, size, priority
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Network state encoder
        self.network_encoder = nn.Sequential(
            nn.Linear(num_devices * num_devices * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Combined predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        logger.info(
            f"TransferTimePredictor initialized: devices={num_devices}, hidden={hidden_dim}"
        )

    def forward(
        self,
        flow_features: torch.Tensor,
        network_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict transfer time.

        Args:
            flow_features: Flow features [batch, 4]
            network_state: Network state [batch, num_devices * num_devices * 4]

        Returns:
            Predicted transfer time in seconds [batch]
        """
        # Encode inputs
        flow_encoded = self.flow_encoder(flow_features)  # [batch, hidden]
        network_encoded = self.network_encoder(network_state)  # [batch, hidden]

        # Combine
        combined = torch.cat([flow_encoded, network_encoded], dim=-1)  # [batch, hidden * 2]

        # Predict
        transfer_time = self.predictor(combined)  # [batch, 1]

        return transfer_time.squeeze()


class NetworkModelTrainer:
    """
    Trainer for network prediction models.

    Handles data collection, training, and model management.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        device: str = "cpu"
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            learning_rate: Learning rate
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.training_history: deque = deque(maxlen=1000)
        self.best_loss = float('inf')

        logger.info(f"NetworkModelTrainer initialized (lr={learning_rate}, device={device})")

    def train_step(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> float:
        """
        Perform one training step.

        Args:
            inputs: Dictionary of model inputs
            targets: Target values

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        predictions = self.model(**inputs)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        loss_value = loss.item()
        self.training_history.append({
            "timestamp": time.time(),
            "loss": loss_value
        })

        return loss_value

    def evaluate(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> float:
        """
        Evaluate model.

        Args:
            inputs: Dictionary of model inputs
            targets: Target values

        Returns:
            Loss value
        """
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(**inputs)
            loss = self.loss_fn(predictions, targets)

        return loss.item()

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "training_history": list(self.training_history)
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        logger.info(f"Loaded checkpoint from {path}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.training_history:
            return {}

        recent_losses = [h["loss"] for h in list(self.training_history)[-100:]]

        return {
            "total_steps": len(self.training_history),
            "best_loss": self.best_loss,
            "recent_avg_loss": sum(recent_losses) / len(recent_losses) if recent_losses else 0.0,
            "recent_min_loss": min(recent_losses) if recent_losses else 0.0,
            "recent_max_loss": max(recent_losses) if recent_losses else 0.0
        }


class TraceCollector:
    """
    Collect network traces for training ML models.

    Records network state, flows, and performance metrics.
    """

    def __init__(self, max_samples: int = 10000):
        """
        Initialize trace collector.

        Args:
            max_samples: Maximum number of samples to keep
        """
        self.max_samples = max_samples
        self.samples: deque = deque(maxlen=max_samples)
        logger.info(f"TraceCollector initialized (max_samples={max_samples})")

    def record_sample(self, sample: NetworkSample) -> None:
        """Record a network sample."""
        self.samples.append(sample)

    def get_samples(self, n: Optional[int] = None) -> List[NetworkSample]:
        """
        Get recorded samples.

        Args:
            n: Number of recent samples to return (None for all)

        Returns:
            List of samples
        """
        if n is None:
            return list(self.samples)
        return list(self.samples)[-n:]

    def clear(self) -> None:
        """Clear all samples."""
        self.samples.clear()
        logger.info("Cleared all samples")

    def export_dataset(self, path: str) -> None:
        """Export samples as dataset."""
        dataset = {
            "samples": [
                {
                    "timestamp": s.timestamp,
                    "topology_features": s.topology_features.tolist(),
                    "link_states": s.link_states.tolist(),
                    "flow_features": s.flow_features.tolist(),
                    "target": s.target,
                    "metadata": s.metadata
                }
                for s in self.samples
            ]
        }

        torch.save(dataset, path)
        logger.info(f"Exported {len(self.samples)} samples to {path}")


__all__ = [
    "NetworkSample",
    "PathLatencyPredictor",
    "GNNLayer",
    "CongestionPointPredictor",
    "OptimalRoutePredictor",
    "TransferTimePredictor",
    "NetworkModelTrainer",
    "TraceCollector",
]

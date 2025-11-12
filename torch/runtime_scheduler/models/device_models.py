"""
ML Models for Device and Memory Decisions

ML models for device placement, memory eviction, prefetching,
and transfer scheduling with fast inference and online learning.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DevicePlacementFeatures:
    """Features for device placement prediction"""
    # Operation features
    op_type_id: int
    input_count: int
    total_input_size: int
    estimated_flops: float
    is_compute_intensive: bool

    # Device features (per candidate)
    device_id: int
    device_utilization: float
    device_memory_used: float
    device_memory_free: float
    device_queue_length: int
    device_avg_op_time: float

    # Historical features
    op_count_on_device: int
    avg_runtime_on_device: float

    # Data locality
    inputs_on_device: int
    transfer_cost: float

    def to_tensor(self) -> torch.Tensor:
        """Convert features to tensor"""
        return torch.tensor([
            float(self.op_type_id),
            float(self.input_count),
            float(self.total_input_size),
            self.estimated_flops,
            float(self.is_compute_intensive),
            float(self.device_id),
            self.device_utilization,
            self.device_memory_used,
            self.device_memory_free,
            float(self.device_queue_length),
            self.device_avg_op_time,
            float(self.op_count_on_device),
            self.avg_runtime_on_device,
            float(self.inputs_on_device),
            self.transfer_cost,
        ], dtype=torch.float32)


class DevicePlacementNetwork(nn.Module):
    """Neural network for device placement"""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 64):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Predicted runtime [batch_size, 1]
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DevicePlacementModel:
    """Predict best device for operation"""

    def __init__(self, num_devices: int = 4):
        self.num_devices = num_devices
        self.lock = threading.Lock()

        # Model
        self.model = DevicePlacementNetwork()
        self.model.eval()

        # Optimizer for online learning
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Training data buffer
        self.training_buffer: deque = deque(maxlen=10000)
        self.batch_size = 32

        # Feature statistics for normalization
        self.feature_means = torch.zeros(15)
        self.feature_stds = torch.ones(15)

        # Op type encoding
        self.op_type_to_id: Dict[str, int] = {}
        self.next_op_id = 0

        # Performance tracking
        self.predictions = 0
        self.correct_predictions = 0

    def predict(
        self,
        features_list: List[DevicePlacementFeatures],
    ) -> int:
        """
        Predict best device.

        Args:
            features_list: Features for each candidate device

        Returns:
            Device ID with lowest predicted runtime
        """
        with self.lock:
            self.predictions += 1

            if not features_list:
                return 0

            # Convert features to tensor
            feature_tensors = [f.to_tensor() for f in features_list]
            x = torch.stack(feature_tensors)

            # Normalize
            x = (x - self.feature_means) / (self.feature_stds + 1e-8)

            # Predict
            with torch.no_grad():
                predictions = self.model(x)

            # Select device with minimum predicted runtime
            best_idx = predictions.argmin().item()
            return features_list[best_idx].device_id

    def record_result(
        self,
        features: DevicePlacementFeatures,
        actual_runtime: float,
    ):
        """Record actual result for online learning"""
        with self.lock:
            # Add to training buffer
            feature_tensor = features.to_tensor()
            self.training_buffer.append((feature_tensor, actual_runtime))

            # Update feature statistics
            self._update_statistics()

            # Train periodically
            if len(self.training_buffer) >= self.batch_size:
                self._train_step()

    def _update_statistics(self):
        """Update feature statistics for normalization"""
        if len(self.training_buffer) < 10:
            return

        # Compute mean and std from recent samples
        recent_features = torch.stack([f for f, _ in list(self.training_buffer)[-1000:]])
        self.feature_means = recent_features.mean(dim=0)
        self.feature_stds = recent_features.std(dim=0)

    def _train_step(self):
        """Perform one training step"""
        # Sample batch
        import random
        batch_indices = random.sample(
            range(len(self.training_buffer)),
            min(self.batch_size, len(self.training_buffer))
        )

        batch_data = [self.training_buffer[i] for i in batch_indices]
        features = torch.stack([f for f, _ in batch_data])
        targets = torch.tensor([r for _, r in batch_data], dtype=torch.float32).unsqueeze(1)

        # Normalize features
        features = (features - self.feature_means) / (self.feature_stds + 1e-8)

        # Forward pass
        self.model.train()
        predictions = self.model(features)

        # Compute loss (MSE)
        loss = F.mse_loss(predictions, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.model.eval()

    def encode_op_type(self, op_type: str) -> int:
        """Encode operation type to ID"""
        with self.lock:
            if op_type not in self.op_type_to_id:
                self.op_type_to_id[op_type] = self.next_op_id
                self.next_op_id += 1
            return self.op_type_to_id[op_type]


@dataclass
class MemoryEvictionFeatures:
    """Features for memory eviction prediction"""
    # Block features
    block_size: int
    allocated_time: float
    last_access_time: float
    access_count: int
    ref_count: int

    # Device features
    memory_pressure: float
    fragmentation: float

    # Access pattern
    avg_access_interval: float
    access_trend: float  # Increasing or decreasing

    def to_tensor(self) -> torch.Tensor:
        """Convert features to tensor"""
        current_time = time.time()
        age = current_time - self.allocated_time
        recency = current_time - self.last_access_time

        return torch.tensor([
            float(self.block_size),
            age,
            recency,
            float(self.access_count),
            float(self.ref_count),
            self.memory_pressure,
            self.fragmentation,
            self.avg_access_interval,
            self.access_trend,
        ], dtype=torch.float32)


class MemoryEvictionNetwork(nn.Module):
    """Neural network for memory eviction"""

    def __init__(self, input_dim: int = 9, hidden_dim: int = 32):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Eviction score [batch_size, 1] (higher = evict)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Score between 0 and 1
        return x


class MemoryEvictionModel:
    """Predict what to evict"""

    def __init__(self):
        self.lock = threading.Lock()

        # Model
        self.model = MemoryEvictionNetwork()
        self.model.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Training buffer
        self.training_buffer: deque = deque(maxlen=10000)

        # Feature statistics
        self.feature_means = torch.zeros(9)
        self.feature_stds = torch.ones(9)

    def predict_eviction_scores(
        self,
        features_list: List[MemoryEvictionFeatures],
    ) -> List[float]:
        """
        Predict eviction scores for blocks.

        Args:
            features_list: Features for each block

        Returns:
            Eviction scores (higher = evict first)
        """
        with self.lock:
            if not features_list:
                return []

            # Convert features to tensor
            feature_tensors = [f.to_tensor() for f in features_list]
            x = torch.stack(feature_tensors)

            # Normalize
            x = (x - self.feature_means) / (self.feature_stds + 1e-8)

            # Predict
            with torch.no_grad():
                scores = self.model(x)

            return scores.squeeze().tolist()

    def record_eviction_result(
        self,
        features: MemoryEvictionFeatures,
        was_good_choice: bool,
    ):
        """Record eviction result for learning"""
        with self.lock:
            feature_tensor = features.to_tensor()
            label = 1.0 if was_good_choice else 0.0
            self.training_buffer.append((feature_tensor, label))

            # Update statistics periodically
            if len(self.training_buffer) % 100 == 0:
                self._update_statistics()

    def _update_statistics(self):
        """Update feature statistics"""
        if len(self.training_buffer) < 10:
            return

        recent_features = torch.stack([f for f, _ in list(self.training_buffer)[-1000:]])
        self.feature_means = recent_features.mean(dim=0)
        self.feature_stds = recent_features.std(dim=0)


@dataclass
class PrefetchFeatures:
    """Features for prefetch prediction"""
    # Tensor features
    tensor_id: int
    tensor_size: int
    current_location: int  # Device ID

    # Access pattern
    access_count: int
    last_access_time: float
    avg_access_interval: float
    access_regularity: float  # 0-1, higher = more regular

    # Context
    recent_op_types: List[int]  # Last N operation types
    target_device_load: float

    def to_tensor(self) -> torch.Tensor:
        """Convert features to tensor"""
        current_time = time.time()
        recency = current_time - self.last_access_time

        # Pad or truncate recent op types
        op_types = self.recent_op_types[-5:] + [0] * 5
        op_types = op_types[:5]

        return torch.tensor([
            float(self.tensor_id % 10000),  # Hash to reasonable range
            float(self.tensor_size),
            float(self.current_location),
            float(self.access_count),
            recency,
            self.avg_access_interval,
            self.access_regularity,
            self.target_device_load,
        ] + [float(op) for op in op_types], dtype=torch.float32)


class PrefetchNetwork(nn.Module):
    """Neural network for prefetch prediction"""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 32):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Prefetch probability [batch_size, 1]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class PrefetchModel:
    """Predict what to prefetch"""

    def __init__(self):
        self.lock = threading.Lock()

        # Model
        self.model = PrefetchNetwork()
        self.model.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Training buffer
        self.training_buffer: deque = deque(maxlen=10000)

        # Feature statistics
        self.feature_means = torch.zeros(13)
        self.feature_stds = torch.ones(13)

        # Tracking
        self.prefetch_hits = 0
        self.prefetch_misses = 0

    def predict_prefetch_priority(
        self,
        features_list: List[PrefetchFeatures],
    ) -> List[Tuple[int, float]]:
        """
        Predict prefetch priorities.

        Args:
            features_list: Features for each candidate tensor

        Returns:
            List of (tensor_id, priority) sorted by priority
        """
        with self.lock:
            if not features_list:
                return []

            # Convert features to tensor
            feature_tensors = [f.to_tensor() for f in features_list]
            x = torch.stack(feature_tensors)

            # Normalize
            x = (x - self.feature_means) / (self.feature_stds + 1e-8)

            # Predict
            with torch.no_grad():
                priorities = self.model(x)

            # Create (tensor_id, priority) pairs and sort
            results = [
                (features_list[i].tensor_id, priorities[i].item())
                for i in range(len(features_list))
            ]
            results.sort(key=lambda x: x[1], reverse=True)

            return results

    def record_prefetch_result(
        self,
        features: PrefetchFeatures,
        was_used: bool,
    ):
        """Record prefetch result"""
        with self.lock:
            if was_used:
                self.prefetch_hits += 1
            else:
                self.prefetch_misses += 1

            feature_tensor = features.to_tensor()
            label = 1.0 if was_used else 0.0
            self.training_buffer.append((feature_tensor, label))

    def get_hit_rate(self) -> float:
        """Get prefetch hit rate"""
        with self.lock:
            total = self.prefetch_hits + self.prefetch_misses
            if total == 0:
                return 0.0
            return self.prefetch_hits / total


@dataclass
class TransferSchedulingFeatures:
    """Features for transfer scheduling"""
    # Transfer features
    transfer_size: int
    transfer_type: int  # Encoded type
    priority: int

    # Source/destination features
    src_device_load: float
    dst_device_load: float
    bandwidth_estimate: float

    # Queue features
    queue_length: int
    queue_total_size: int

    # Dependencies
    has_dependencies: bool
    dependency_count: int

    def to_tensor(self) -> torch.Tensor:
        """Convert features to tensor"""
        return torch.tensor([
            float(self.transfer_size),
            float(self.transfer_type),
            float(self.priority),
            self.src_device_load,
            self.dst_device_load,
            self.bandwidth_estimate,
            float(self.queue_length),
            float(self.queue_total_size),
            float(self.has_dependencies),
            float(self.dependency_count),
        ], dtype=torch.float32)


class TransferSchedulingNetwork(nn.Module):
    """Neural network for transfer scheduling"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Predicted transfer time [batch_size, 1]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TransferSchedulingModel:
    """Optimize transfer order"""

    def __init__(self):
        self.lock = threading.Lock()

        # Model
        self.model = TransferSchedulingNetwork()
        self.model.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Training buffer
        self.training_buffer: deque = deque(maxlen=10000)

        # Feature statistics
        self.feature_means = torch.zeros(10)
        self.feature_stds = torch.ones(10)

    def predict_transfer_times(
        self,
        features_list: List[TransferSchedulingFeatures],
    ) -> List[float]:
        """
        Predict transfer times.

        Args:
            features_list: Features for each transfer

        Returns:
            Predicted transfer times
        """
        with self.lock:
            if not features_list:
                return []

            # Convert features to tensor
            feature_tensors = [f.to_tensor() for f in features_list]
            x = torch.stack(feature_tensors)

            # Normalize
            x = (x - self.feature_means) / (self.feature_stds + 1e-8)

            # Predict
            with torch.no_grad():
                times = self.model(x)

            return times.squeeze().tolist()

    def record_transfer_result(
        self,
        features: TransferSchedulingFeatures,
        actual_time: float,
    ):
        """Record transfer result"""
        with self.lock:
            feature_tensor = features.to_tensor()
            self.training_buffer.append((feature_tensor, actual_time))


class ModelManager:
    """Manage all ML models"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True

        # Initialize models
        self.device_placement = DevicePlacementModel()
        self.memory_eviction = MemoryEvictionModel()
        self.prefetch = PrefetchModel()
        self.transfer_scheduling = TransferSchedulingModel()

    def get_device_placement_model(self) -> DevicePlacementModel:
        """Get device placement model"""
        return self.device_placement

    def get_memory_eviction_model(self) -> MemoryEvictionModel:
        """Get memory eviction model"""
        return self.memory_eviction

    def get_prefetch_model(self) -> PrefetchModel:
        """Get prefetch model"""
        return self.prefetch

    def get_transfer_scheduling_model(self) -> TransferSchedulingModel:
        """Get transfer scheduling model"""
        return self.transfer_scheduling


# Global singleton
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

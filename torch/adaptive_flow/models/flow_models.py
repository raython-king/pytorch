"""
Machine Learning Models for Flow Prediction and Optimization.

Implements predictive models for bandwidth, congestion, flow size,
and latency prediction to enable proactive traffic management.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for deep learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using simplified models")


@dataclass
class FlowFeatures:
    """Feature vector for a flow.

    Attributes:
        flow_size: Size in bytes
        priority: Priority level
        source_load: Source node load [0.0, 1.0]
        dest_load: Destination node load [0.0, 1.0]
        link_utilization: Link utilization [0.0, 1.0]
        time_of_day: Time of day [0.0, 1.0]
        historical_bandwidth: Recent average bandwidth
        queue_length: Current queue length
        num_active_flows: Number of active flows on link
    """
    flow_size: float = 0.0
    priority: int = 1
    source_load: float = 0.0
    dest_load: float = 0.0
    link_utilization: float = 0.0
    time_of_day: float = 0.0
    historical_bandwidth: float = 0.0
    queue_length: int = 0
    num_active_flows: int = 0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            np.log1p(self.flow_size),  # Log scale for size
            float(self.priority),
            self.source_load,
            self.dest_load,
            self.link_utilization,
            self.time_of_day,
            np.log1p(self.historical_bandwidth),
            float(self.queue_length),
            float(self.num_active_flows),
        ], dtype=np.float32)


class BandwidthPredictor:
    """Predict available bandwidth using LSTM.

    Uses historical bandwidth measurements to predict future
    available bandwidth with uncertainty quantification.
    """

    def __init__(self, sequence_length: int = 20, hidden_size: int = 64):
        """Initialize bandwidth predictor.

        Args:
            sequence_length: Number of historical samples to use
            hidden_size: LSTM hidden size
        """
        self._lock = threading.RLock()
        self._sequence_length = sequence_length
        self._hidden_size = hidden_size

        # Historical data
        self._link_history: Dict[str, deque] = {}

        # Model
        if TORCH_AVAILABLE:
            self._model = BandwidthLSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=2
            )
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        else:
            self._model = None

        # Fallback: exponential moving average
        self._ema_alpha = 0.3
        self._ema_predictions: Dict[str, float] = {}

        # Statistics
        self._predictions_made = 0
        self._mae = deque(maxlen=1000)  # Mean absolute error

    def update(self, link_id: str, bandwidth: float, timestamp: Optional[float] = None) -> None:
        """Update with new bandwidth measurement.

        Args:
            link_id: Link identifier
            bandwidth: Measured bandwidth in bytes/sec
            timestamp: Optional timestamp
        """
        with self._lock:
            if link_id not in self._link_history:
                self._link_history[link_id] = deque(maxlen=self._sequence_length * 2)

            if timestamp is None:
                timestamp = time.time()

            self._link_history[link_id].append((timestamp, bandwidth))

            # Update EMA
            if link_id in self._ema_predictions:
                self._ema_predictions[link_id] = (
                    self._ema_alpha * bandwidth +
                    (1 - self._ema_alpha) * self._ema_predictions[link_id]
                )
            else:
                self._ema_predictions[link_id] = bandwidth

    def predict(self, link_id: str, horizon: int = 1) -> Tuple[float, float]:
        """Predict bandwidth for future timesteps.

        Args:
            link_id: Link identifier
            horizon: Number of timesteps ahead to predict

        Returns:
            Tuple of (predicted_bandwidth, confidence_interval)
        """
        with self._lock:
            self._predictions_made += 1

            if self._model is not None and TORCH_AVAILABLE:
                return self._predict_lstm(link_id, horizon)
            else:
                return self._predict_ema(link_id)

    def _predict_lstm(self, link_id: str, horizon: int) -> Tuple[float, float]:
        """Predict using LSTM model.

        Args:
            link_id: Link identifier
            horizon: Prediction horizon

        Returns:
            Tuple of (prediction, confidence)
        """
        history = self._link_history.get(link_id)
        if not history or len(history) < self._sequence_length:
            # Not enough data, use EMA
            return self._predict_ema(link_id)

        # Prepare sequence
        recent = list(history)[-self._sequence_length:]
        sequence = np.array([bw for _, bw in recent], dtype=np.float32)
        sequence = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)

        # Predict
        self._model.eval()
        with torch.no_grad():
            prediction = self._model(sequence)
            prediction = prediction.item()

        # Estimate confidence based on recent variance
        variance = np.var([bw for _, bw in recent])
        confidence = np.sqrt(variance)

        return max(0.0, prediction), confidence

    def _predict_ema(self, link_id: str) -> Tuple[float, float]:
        """Predict using exponential moving average.

        Args:
            link_id: Link identifier

        Returns:
            Tuple of (prediction, confidence)
        """
        prediction = self._ema_predictions.get(link_id, 0.0)

        # Estimate confidence from recent history
        history = self._link_history.get(link_id)
        if history and len(history) >= 5:
            recent_values = [bw for _, bw in list(history)[-10:]]
            confidence = np.std(recent_values)
        else:
            confidence = prediction * 0.2  # 20% uncertainty

        return prediction, confidence

    def train_online(self, link_id: str, actual_bandwidth: float) -> None:
        """Perform online learning with new observation.

        Args:
            link_id: Link identifier
            actual_bandwidth: Actual observed bandwidth
        """
        if not TORCH_AVAILABLE or self._model is None:
            return

        with self._lock:
            history = self._link_history.get(link_id)
            if not history or len(history) < self._sequence_length + 1:
                return

            # Prepare training sample
            recent = list(history)[-self._sequence_length - 1:]
            sequence = np.array([bw for _, bw in recent[:-1]], dtype=np.float32)
            target = recent[-1][1]

            sequence = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)
            target = torch.FloatTensor([target])

            # Train step
            self._model.train()
            self._optimizer.zero_grad()
            prediction = self._model(sequence)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            self._optimizer.step()

            # Track error
            error = abs(prediction.item() - target.item())
            self._mae.append(error)

    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            mae = np.mean(self._mae) if self._mae else 0.0
            return {
                'predictions_made': self._predictions_made,
                'mean_absolute_error': mae,
                'num_links': len(self._link_history),
            }


class CongestionPredictor:
    """Predict congestion using Graph Neural Network.

    Models network topology and flow patterns to predict
    congestion before it occurs.
    """

    def __init__(self, num_nodes: int = 100):
        """Initialize congestion predictor.

        Args:
            num_nodes: Number of nodes in network
        """
        self._lock = threading.RLock()
        self._num_nodes = num_nodes

        # Network state
        self._node_features: Dict[str, np.ndarray] = {}
        self._link_features: Dict[str, np.ndarray] = {}

        # Model (simplified without actual GNN for now)
        self._threshold_model = {
            'utilization_threshold': 0.8,
            'queue_threshold': 100,
            'rtt_multiplier': 2.0,
        }

        # Statistics
        self._predictions = deque(maxlen=1000)
        self._true_positives = 0
        self._false_positives = 0
        self._true_negatives = 0
        self._false_negatives = 0

    def update_network_state(self, node_id: str, features: np.ndarray) -> None:
        """Update node features.

        Args:
            node_id: Node identifier
            features: Feature vector for node
        """
        with self._lock:
            self._node_features[node_id] = features

    def update_link_state(self, link_id: str, features: np.ndarray) -> None:
        """Update link features.

        Args:
            link_id: Link identifier
            features: Feature vector for link
        """
        with self._lock:
            self._link_features[link_id] = features

    def predict_congestion(self, link_id: str) -> Tuple[float, float]:
        """Predict congestion probability for link.

        Args:
            link_id: Link identifier

        Returns:
            Tuple of (congestion_probability, confidence)
        """
        with self._lock:
            features = self._link_features.get(link_id)
            if features is None:
                return 0.0, 0.0

            # Simplified prediction based on utilization and queue
            # features = [utilization, queue_length, rtt, loss_rate, ...]
            utilization = features[0] if len(features) > 0 else 0.0
            queue_length = features[1] if len(features) > 1 else 0.0

            # Probability based on weighted combination
            prob = 0.0

            if utilization > self._threshold_model['utilization_threshold']:
                prob += 0.5 * (utilization - self._threshold_model['utilization_threshold']) / (
                    1.0 - self._threshold_model['utilization_threshold']
                )

            if queue_length > self._threshold_model['queue_threshold']:
                prob += 0.5 * min(1.0, queue_length / (self._threshold_model['queue_threshold'] * 2))

            prob = min(1.0, prob)
            confidence = 0.8  # Fixed confidence for simple model

            return prob, confidence

    def update_prediction(self, link_id: str, predicted: bool, actual: bool) -> None:
        """Update with actual congestion outcome.

        Args:
            link_id: Link identifier
            predicted: Predicted congestion
            actual: Actual congestion
        """
        with self._lock:
            self._predictions.append((predicted, actual))

            if predicted and actual:
                self._true_positives += 1
            elif predicted and not actual:
                self._false_positives += 1
            elif not predicted and actual:
                self._false_negatives += 1
            else:
                self._true_negatives += 1

    def get_accuracy(self) -> Dict[str, float]:
        """Get prediction accuracy metrics.

        Returns:
            Dictionary with accuracy metrics
        """
        with self._lock:
            total = (self._true_positives + self._false_positives +
                    self._true_negatives + self._false_negatives)

            if total == 0:
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}

            accuracy = (self._true_positives + self._true_negatives) / total

            if self._true_positives + self._false_positives > 0:
                precision = self._true_positives / (self._true_positives + self._false_positives)
            else:
                precision = 0.0

            if self._true_positives + self._false_negatives > 0:
                recall = self._true_positives / (self._true_positives + self._false_negatives)
            else:
                recall = 0.0

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'total_predictions': total,
            }


class FlowSizeEstimator:
    """Estimate flow transfer size using ensemble methods.

    Combines multiple estimators to predict flow completion time
    and size for improved scheduling decisions.
    """

    def __init__(self):
        """Initialize flow size estimator."""
        self._lock = threading.RLock()

        # Historical data
        self._flow_patterns: Dict[str, List[float]] = {}  # pattern_id -> sizes

        # Statistics
        self._estimates_made = 0
        self._errors = deque(maxlen=1000)

    def add_pattern(self, pattern_id: str, size: float) -> None:
        """Add observed flow size for pattern.

        Args:
            pattern_id: Pattern identifier (e.g., "allreduce_32nodes")
            size: Observed size in bytes
        """
        with self._lock:
            if pattern_id not in self._flow_patterns:
                self._flow_patterns[pattern_id] = []

            self._flow_patterns[pattern_id].append(size)

            # Keep last 100 samples per pattern
            if len(self._flow_patterns[pattern_id]) > 100:
                self._flow_patterns[pattern_id] = self._flow_patterns[pattern_id][-100:]

    def estimate(self, pattern_id: str, features: Optional[FlowFeatures] = None) -> Tuple[float, float]:
        """Estimate flow size.

        Args:
            pattern_id: Pattern identifier
            features: Optional flow features

        Returns:
            Tuple of (estimated_size, confidence_interval)
        """
        with self._lock:
            self._estimates_made += 1

            # Use historical pattern if available
            if pattern_id in self._flow_patterns:
                sizes = self._flow_patterns[pattern_id]
                median = np.median(sizes)
                std = np.std(sizes)
                return median, std

            # Fallback: use features if available
            if features is not None:
                # Simple heuristic based on flow_size feature
                return features.flow_size, features.flow_size * 0.3

            # No information
            return 0.0, 0.0

    def update_estimate(self, pattern_id: str, estimated: float, actual: float) -> None:
        """Update with actual size.

        Args:
            pattern_id: Pattern identifier
            estimated: Estimated size
            actual: Actual size
        """
        with self._lock:
            error = abs(estimated - actual) / max(actual, 1.0)
            self._errors.append(error)

            # Add to pattern history
            self.add_pattern(pattern_id, actual)

    def get_statistics(self) -> Dict[str, Any]:
        """Get estimator statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            mape = np.mean(self._errors) if self._errors else 0.0  # Mean absolute percentage error
            return {
                'estimates_made': self._estimates_made,
                'mean_absolute_percentage_error': mape,
                'num_patterns': len(self._flow_patterns),
            }


class LatencyPredictor:
    """Predict flow completion time using regression.

    Predicts end-to-end latency and completion time for flows
    based on flow characteristics and network state.
    """

    def __init__(self):
        """Initialize latency predictor."""
        self._lock = threading.RLock()

        # Simple regression coefficients (can be updated online)
        self._coefficients = {
            'base_latency': 0.001,  # 1ms base
            'size_factor': 1e-9,    # 1ns per byte
            'load_factor': 0.01,    # 10ms per unit load
            'queue_factor': 0.0001, # 0.1ms per queue item
        }

        # Historical data for model updates
        self._training_data: List[Tuple[FlowFeatures, float]] = []

        # Statistics
        self._predictions_made = 0
        self._errors = deque(maxlen=1000)

    def predict(self, features: FlowFeatures) -> Tuple[float, float]:
        """Predict completion latency.

        Args:
            features: Flow features

        Returns:
            Tuple of (predicted_latency_seconds, confidence_interval)
        """
        with self._lock:
            self._predictions_made += 1

            # Linear combination of features
            latency = self._coefficients['base_latency']
            latency += features.flow_size * self._coefficients['size_factor']
            latency += (features.source_load + features.dest_load) * self._coefficients['load_factor']
            latency += features.queue_length * self._coefficients['queue_factor']

            # Adjust for link utilization
            if features.link_utilization > 0.8:
                # Exponential increase when congested
                congestion_factor = 1.0 + (features.link_utilization - 0.8) * 5
                latency *= congestion_factor

            # Confidence based on number of training samples
            confidence = latency * 0.2  # 20% confidence interval

            return max(0.0, latency), confidence

    def update(self, features: FlowFeatures, actual_latency: float) -> None:
        """Update with actual latency.

        Args:
            features: Flow features
            actual_latency: Actual observed latency
        """
        with self._lock:
            # Store training data
            self._training_data.append((features, actual_latency))

            # Keep last 1000 samples
            if len(self._training_data) > 1000:
                self._training_data = self._training_data[-1000:]

            # Compute prediction error
            predicted, _ = self.predict(features)
            error = abs(predicted - actual_latency)
            self._errors.append(error)

            # Periodically update model
            if len(self._training_data) % 100 == 0:
                self._update_model()

    def _update_model(self) -> None:
        """Update regression model with training data."""
        if len(self._training_data) < 10:
            return

        # Simple gradient descent update
        learning_rate = 0.01

        for features, actual_latency in self._training_data[-100:]:
            predicted, _ = self.predict(features)
            error = predicted - actual_latency

            # Update coefficients
            self._coefficients['size_factor'] -= learning_rate * error * features.flow_size
            self._coefficients['load_factor'] -= (
                learning_rate * error * (features.source_load + features.dest_load)
            )
            self._coefficients['queue_factor'] -= learning_rate * error * features.queue_length

        logger.debug("Updated latency predictor model")

    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            mae = np.mean(self._errors) if self._errors else 0.0
            return {
                'predictions_made': self._predictions_made,
                'mean_absolute_error': mae,
                'training_samples': len(self._training_data),
            }


# PyTorch Models (only if torch is available)
if TORCH_AVAILABLE:
    class BandwidthLSTM(nn.Module):
        """LSTM model for bandwidth prediction."""

        def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
            """Initialize LSTM model.

            Args:
                input_size: Input feature size
                hidden_size: LSTM hidden size
                num_layers: Number of LSTM layers
            """
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0
            )

            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            """Forward pass.

            Args:
                x: Input tensor (batch, seq_len, input_size)

            Returns:
                Prediction tensor (batch, 1)
            """
            # LSTM
            out, _ = self.lstm(x)

            # Take last timestep
            out = out[:, -1, :]

            # Fully connected
            out = self.fc(out)

            return out


# Factory function
def create_predictor(predictor_type: str, **kwargs):
    """Create predictor instance.

    Args:
        predictor_type: Type of predictor ("bandwidth", "congestion", "size", "latency")
        **kwargs: Additional arguments for predictor

    Returns:
        Predictor instance

    Raises:
        ValueError: If predictor type unknown
    """
    if predictor_type == "bandwidth":
        return BandwidthPredictor(**kwargs)
    elif predictor_type == "congestion":
        return CongestionPredictor(**kwargs)
    elif predictor_type == "size":
        return FlowSizeEstimator(**kwargs)
    elif predictor_type == "latency":
        return LatencyPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")

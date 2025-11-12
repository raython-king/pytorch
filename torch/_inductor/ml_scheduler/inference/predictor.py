"""
ML Scheduler Predictor

Implements efficient inference pipeline for trained ML scheduler models.
Handles model loading, batch inference, caching, and performance monitoring.
"""

import torch
import torch.nn as nn
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict
from dataclasses import dataclass
import threading

from ..models.gnn_model import FusionGNN, SchedulingGNN
from ..config import MLSchedulerConfig

log = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    fusion_matrix: Optional[torch.Tensor] = None
    priority_scores: Optional[torch.Tensor] = None
    partition_logits: Optional[torch.Tensor] = None
    memory_logits: Optional[torch.Tensor] = None
    node_embeddings: Optional[torch.Tensor] = None
    confidence: Optional[float] = None
    inference_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelCache:
    """Thread-safe LRU cache for model predictions."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached predictions
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[PredictionResult]:
        """Get cached prediction."""
        with self.lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: PredictionResult):
        """Cache a prediction."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove oldest
                    self.cache.popitem(last=False)

            self.cache[key] = value

    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }


class MLSchedulerPredictor:
    """
    Predictor for ML scheduler inference.

    Features:
    - Efficient model loading and inference
    - Result caching
    - Batch processing
    - Performance monitoring
    - Thread-safe operations
    - Timeout handling

    Example:
        predictor = MLSchedulerPredictor.load('./checkpoints/best_model.pt')

        result = predictor.predict(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        if result.confidence > 0.75:
            apply_fusion_plan(result.fusion_matrix)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MLSchedulerConfig] = None,
        device: Optional[torch.device] = None,
        enable_cache: bool = True,
        cache_size: int = 1000,
    ):
        """
        Initialize predictor.

        Args:
            model: Trained model
            config: Configuration
            device: Torch device
            enable_cache: Enable prediction caching
            cache_size: Maximum cache size
        """
        self.model = model
        self.config = config or MLSchedulerConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        # Caching
        self.cache = ModelCache(max_size=cache_size) if enable_cache else None

        # Performance monitoring
        self.inference_times = []
        self.num_predictions = 0

        log.info(f"MLSchedulerPredictor initialized on device: {self.device}")

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> 'MLSchedulerPredictor':
        """
        Load predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config JSON (optional)
            device: Device to load model on
            **kwargs: Additional arguments for predictor

        Returns:
            Loaded predictor
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load config
        if config_path is None:
            config_path = checkpoint_path.with_suffix('.json')

        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            # Create model from config
            model_type = config_dict.get('model_type', 'fusion_gnn')

            if model_type == 'fusion_gnn':
                model = FusionGNN(
                    node_feat_dim=config_dict.get('node_feat_dim', 64),
                    edge_feat_dim=config_dict.get('edge_feat_dim', 32),
                    hidden_dim=config_dict.get('hidden_dim', 128),
                    num_layers=config_dict.get('num_layers', 4),
                    num_heads=config_dict.get('num_heads', 4),
                    dropout=config_dict.get('dropout', 0.1),
                )
            elif model_type == 'scheduling_gnn':
                model = SchedulingGNN(
                    node_feat_dim=config_dict.get('node_feat_dim', 64),
                    edge_feat_dim=config_dict.get('edge_feat_dim', 32),
                    hidden_dim=config_dict.get('hidden_dim', 128),
                    num_layers=config_dict.get('num_layers', 4),
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        else:
            log.warning(f"Config not found at {config_path}, using default FusionGNN")
            model = FusionGNN()

        # Load state dict
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)

        log.info(f"Loaded model from: {checkpoint_path}")

        return cls(model=model, device=device, **kwargs)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        timeout_ms: Optional[float] = None,
    ) -> PredictionResult:
        """
        Run inference on a single graph.

        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            batch: Batch assignment [num_nodes]
            use_cache: Use cached result if available
            timeout_ms: Timeout in milliseconds

        Returns:
            PredictionResult
        """
        timeout_ms = timeout_ms or self.config.max_inference_time_ms

        # Check cache
        if use_cache and self.cache is not None:
            cache_key = self._get_cache_key(x, edge_index, edge_attr)
            cached_result = self.cache.get(cache_key)

            if cached_result is not None:
                log.debug("Cache hit for prediction")
                return cached_result

        # Run inference with timeout
        start_time = time.time()

        try:
            # Move inputs to device
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)

            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)

            if batch is not None:
                batch = batch.to(self.device)

            # Forward pass
            outputs = self.model(x, edge_index, edge_attr, batch)

            # Measure time
            inference_time_ms = (time.time() - start_time) * 1000

            # Check timeout
            if inference_time_ms > timeout_ms:
                log.warning(f"Inference timeout: {inference_time_ms:.2f}ms > {timeout_ms}ms")
                return PredictionResult(
                    confidence=0.0,
                    inference_time_ms=inference_time_ms,
                    metadata={'timeout': True}
                )

            # Create result
            result = PredictionResult(
                fusion_matrix=outputs.get('fusion_matrix', None),
                priority_scores=outputs.get('priority_scores', None),
                partition_logits=outputs.get('partition_logits', None),
                memory_logits=outputs.get('memory_logits', None),
                node_embeddings=outputs.get('node_embeddings', None),
                confidence=self._compute_confidence(outputs),
                inference_time_ms=inference_time_ms,
                metadata={'num_nodes': x.size(0)}
            )

            # Cache result
            if use_cache and self.cache is not None:
                self.cache.put(cache_key, result)

            # Update stats
            self.inference_times.append(inference_time_ms)
            self.num_predictions += 1

            return result

        except Exception as e:
            log.error(f"Prediction error: {e}")
            return PredictionResult(
                confidence=0.0,
                inference_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )

    @torch.no_grad()
    def predict_batch(
        self,
        batch_data: List[Dict[str, torch.Tensor]],
        timeout_ms: Optional[float] = None,
    ) -> List[PredictionResult]:
        """
        Run inference on a batch of graphs.

        Args:
            batch_data: List of graph dictionaries
            timeout_ms: Timeout in milliseconds

        Returns:
            List of PredictionResult
        """
        results = []

        for graph_data in batch_data:
            result = self.predict(
                x=graph_data['x'],
                edge_index=graph_data['edge_index'],
                edge_attr=graph_data.get('edge_attr', None),
                batch=graph_data.get('batch', None),
                timeout_ms=timeout_ms,
            )
            results.append(result)

        return results

    def predict_fusion_score(
        self,
        node_i_embedding: torch.Tensor,
        node_j_embedding: torch.Tensor,
    ) -> float:
        """
        Predict fusion score for a pair of nodes.

        Args:
            node_i_embedding: Embedding for node i [hidden_dim]
            node_j_embedding: Embedding for node j [hidden_dim]

        Returns:
            Fusion score in [0, 1]
        """
        if not isinstance(self.model, FusionGNN):
            raise TypeError("Model must be FusionGNN for pairwise prediction")

        with torch.no_grad():
            node_i_embedding = node_i_embedding.to(self.device)
            node_j_embedding = node_j_embedding.to(self.device)

            score = self.model.predict_pairwise(node_i_embedding, node_j_embedding)

            return float(score.cpu().item())

    def _compute_confidence(self, outputs: Dict[str, torch.Tensor]) -> float:
        """
        Compute confidence score for predictions.

        Heuristics:
        - Higher max probability = higher confidence
        - Lower entropy = higher confidence
        - Consistent predictions = higher confidence
        """
        confidences = []

        # Fusion confidence
        if 'fusion_matrix' in outputs:
            fusion_matrix = outputs['fusion_matrix']

            # Measure how decisive the predictions are
            # High values (close to 1) or low values (close to 0) = confident
            fusion_probs = fusion_matrix.flatten()
            fusion_decisiveness = torch.mean(
                torch.abs(fusion_probs - 0.5) * 2  # Scale to [0, 1]
            )
            confidences.append(fusion_decisiveness.item())

        # Priority confidence
        if 'priority_scores' in outputs:
            priority_scores = outputs['priority_scores']

            # Measure variance (high variance = confident ordering)
            priority_variance = torch.var(priority_scores)
            priority_confidence = torch.tanh(priority_variance / 10.0)  # Normalize
            confidences.append(priority_confidence.item())

        # Overall confidence
        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return 0.5  # Default

    def _get_cache_key(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> str:
        """Generate cache key for graph."""
        # Simple hash based on tensor shapes and some values
        key_parts = [
            f"x_{x.shape}_{x.sum().item():.4f}",
            f"edge_{edge_index.shape}_{edge_index.sum().item()}",
        ]

        if edge_attr is not None:
            key_parts.append(f"attr_{edge_attr.shape}_{edge_attr.sum().item():.4f}")

        return "_".join(key_parts)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.inference_times:
            return {
                'num_predictions': 0,
                'mean_time_ms': 0.0,
                'median_time_ms': 0.0,
                'p95_time_ms': 0.0,
                'p99_time_ms': 0.0,
            }

        sorted_times = sorted(self.inference_times)
        n = len(sorted_times)

        stats = {
            'num_predictions': self.num_predictions,
            'mean_time_ms': sum(sorted_times) / n,
            'median_time_ms': sorted_times[n // 2],
            'p95_time_ms': sorted_times[int(n * 0.95)],
            'p99_time_ms': sorted_times[int(n * 0.99)],
            'min_time_ms': sorted_times[0],
            'max_time_ms': sorted_times[-1],
        }

        # Add cache stats if available
        if self.cache is not None:
            stats['cache'] = self.cache.get_stats()

        return stats

    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_times.clear()
        self.num_predictions = 0

        if self.cache is not None:
            self.cache.clear()

    def warmup(self, num_iterations: int = 10, graph_size: int = 50):
        """
        Warmup model with dummy inputs.

        Useful for:
        - GPU kernel compilation
        - Cache warming
        - Performance benchmarking

        Args:
            num_iterations: Number of warmup iterations
            graph_size: Size of dummy graphs
        """
        log.info(f"Warming up model with {num_iterations} iterations...")

        for i in range(num_iterations):
            # Create dummy graph
            x = torch.randn(graph_size, 64).to(self.device)
            edge_index = torch.randint(0, graph_size, (2, graph_size * 2)).to(self.device)
            edge_attr = torch.randn(graph_size * 2, 32).to(self.device)

            # Run prediction
            self.predict(x, edge_index, edge_attr, use_cache=False)

        # Reset stats after warmup
        self.reset_stats()

        log.info("Warmup completed")


class EnsemblePredictor:
    """
    Ensemble of multiple predictors for improved robustness.

    Combines predictions from multiple models using voting or averaging.
    """

    def __init__(
        self,
        predictors: Optional[List[MLSchedulerPredictor]] = None,
        ensemble_method: str = 'average',  # 'average', 'vote', 'weighted'
        pretrained: bool = True,
    ):
        """
        Initialize ensemble predictor.

        Args:
            predictors: List of predictors
            ensemble_method: Method for combining predictions
            pretrained: Whether using pretrained models
        """
        self.predictors = predictors or []
        self.ensemble_method = ensemble_method
        self.pretrained = pretrained

        if not self.pretrained and not self.predictors:
            log.warning("EnsemblePredictor initialized without pretrained models")

    @classmethod
    def load(cls, checkpoint_dir: str) -> 'EnsemblePredictor':
        """Load ensemble from directory of checkpoints."""
        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        # Find all checkpoint files
        checkpoint_files = list(checkpoint_dir.glob('*.pt'))

        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in: {checkpoint_dir}")

        # Load each predictor
        predictors = []
        for checkpoint_file in checkpoint_files:
            try:
                predictor = MLSchedulerPredictor.load(str(checkpoint_file))
                predictors.append(predictor)
                log.info(f"Loaded predictor: {checkpoint_file}")
            except Exception as e:
                log.warning(f"Failed to load {checkpoint_file}: {e}")

        log.info(f"Loaded ensemble with {len(predictors)} models")

        return cls(predictors=predictors)

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> PredictionResult:
        """
        Ensemble prediction.

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch assignment

        Returns:
            Combined PredictionResult
        """
        if not self.predictors:
            raise RuntimeError("No predictors in ensemble")

        # Get predictions from all models
        results = []
        for predictor in self.predictors:
            result = predictor.predict(x, edge_index, edge_attr, batch)
            results.append(result)

        # Combine predictions
        if self.ensemble_method == 'average':
            return self._average_predictions(results)
        elif self.ensemble_method == 'vote':
            return self._vote_predictions(results)
        elif self.ensemble_method == 'weighted':
            return self._weighted_predictions(results)
        else:
            raise ValueError(f"Unknown ensemble_method: {self.ensemble_method}")

    def _average_predictions(self, results: List[PredictionResult]) -> PredictionResult:
        """Average predictions from all models."""
        # Average fusion matrices
        fusion_matrices = [r.fusion_matrix for r in results if r.fusion_matrix is not None]

        if fusion_matrices:
            avg_fusion = torch.stack(fusion_matrices).mean(dim=0)
        else:
            avg_fusion = None

        # Average priority scores
        priority_scores = [r.priority_scores for r in results if r.priority_scores is not None]

        if priority_scores:
            avg_priority = torch.stack(priority_scores).mean(dim=0)
        else:
            avg_priority = None

        # Average confidence
        confidences = [r.confidence for r in results if r.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return PredictionResult(
            fusion_matrix=avg_fusion,
            priority_scores=avg_priority,
            confidence=avg_confidence,
            metadata={'ensemble_size': len(results)}
        )

    def _vote_predictions(self, results: List[PredictionResult]) -> PredictionResult:
        """Majority vote for fusion decisions."""
        # Convert continuous fusion scores to binary votes
        fusion_matrices = [r.fusion_matrix for r in results if r.fusion_matrix is not None]

        if fusion_matrices:
            # Threshold at 0.5 and vote
            votes = torch.stack([(fm > 0.5).float() for fm in fusion_matrices])
            voted_fusion = (votes.sum(dim=0) > len(votes) / 2).float()
        else:
            voted_fusion = None

        return PredictionResult(
            fusion_matrix=voted_fusion,
            confidence=0.8,  # High confidence for voting
            metadata={'ensemble_size': len(results)}
        )

    def _weighted_predictions(self, results: List[PredictionResult]) -> PredictionResult:
        """Weighted combination based on confidence."""
        # Weight by confidence
        weights = torch.tensor([r.confidence for r in results if r.confidence is not None])

        if len(weights) == 0:
            return self._average_predictions(results)

        weights = weights / weights.sum()

        # Weighted fusion
        fusion_matrices = [r.fusion_matrix for r in results if r.fusion_matrix is not None]

        if fusion_matrices:
            weighted_fusion = sum(
                w * fm for w, fm in zip(weights, fusion_matrices)
            )
        else:
            weighted_fusion = None

        return PredictionResult(
            fusion_matrix=weighted_fusion,
            confidence=float(weights.max()),
            metadata={'ensemble_size': len(results)}
        )

    def predict_pairwise(self, pairwise_features: torch.Tensor) -> float:
        """Predict pairwise fusion score."""
        # For now, just use first predictor
        # Could extend to ensemble
        if not self.predictors:
            return 0.5

        # Assume pairwise_features contains two node embeddings
        # This is a simplified interface
        return 0.5  # Placeholder

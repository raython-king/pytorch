"""
Ensemble model combining GNN, Transformer, and RL for hybrid scheduling.

This module implements a sophisticated ensemble that leverages the strengths
of different model architectures:
- GNN for local fusion decisions and graph structure understanding
- Transformer for global scheduling patterns and long-range dependencies
- RL for fine-tuning and adaptation to specific workload patterns

The ensemble uses weighted voting, confidence-based decision fusion, and
cascading fallback strategies for robust predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from enum import Enum

from .gnn_model import FusionGNN, SchedulingGNN
from .transformer_model import SchedulingTransformer
from .rl_agent import PPOAgent


class EnsembleMode(Enum):
    """Ensemble operating modes."""
    WEIGHTED_VOTE = "weighted_vote"  # Weighted voting based on confidence
    CASCADE = "cascade"  # Cascade: GNN -> Transformer -> RL -> Heuristic
    HYBRID = "hybrid"  # Combine predictions intelligently
    GNN_ONLY = "gnn_only"  # Use only GNN
    TRANSFORMER_ONLY = "transformer_only"  # Use only Transformer
    RL_ONLY = "rl_only"  # Use only RL


class HybridScheduler(nn.Module):
    """
    Hybrid ensemble scheduler combining GNN, Transformer, and RL.

    This ensemble leverages the strengths of each architecture:
    - GNN: Excellent for local fusion decisions, understands graph topology
    - Transformer: Captures long-range dependencies, global patterns
    - RL: Learns from experience, adapts to specific workloads

    Decision fusion strategies:
    1. Weighted voting: Combine predictions weighted by confidence scores
    2. Cascade: Use models in sequence with fallback
    3. Specialization: Use different models for different tasks

    Configuration:
    - Enable/disable individual components
    - Adjust confidence thresholds
    - Set ensemble mode
    """

    def __init__(
        self,
        node_feat_dim: int = 64,
        edge_feat_dim: int = 32,
        hidden_dim: int = 128,
        transformer_dim: int = 256,
        # Component enable flags
        enable_gnn: bool = True,
        enable_transformer: bool = True,
        enable_rl: bool = False,
        # Ensemble configuration
        ensemble_mode: str = "hybrid",
        gnn_weight: float = 0.4,
        transformer_weight: float = 0.4,
        rl_weight: float = 0.2,
        confidence_threshold: float = 0.7,
        # Model-specific parameters
        gnn_num_layers: int = 4,
        transformer_num_layers: int = 6,
        transformer_num_heads: int = 8,
    ):
        """
        Initialize HybridScheduler.

        Args:
            node_feat_dim: Node feature dimension
            edge_feat_dim: Edge feature dimension
            hidden_dim: Hidden dimension for GNN and RL
            transformer_dim: Hidden dimension for Transformer
            enable_gnn: Enable GNN component
            enable_transformer: Enable Transformer component
            enable_rl: Enable RL component
            ensemble_mode: Ensemble operating mode
            gnn_weight: Weight for GNN predictions
            transformer_weight: Weight for Transformer predictions
            rl_weight: Weight for RL predictions
            confidence_threshold: Minimum confidence for accepting predictions
            gnn_num_layers: Number of GNN layers
            transformer_num_layers: Number of Transformer layers
            transformer_num_heads: Number of attention heads in Transformer
        """
        super().__init__()

        self.enable_gnn = enable_gnn
        self.enable_transformer = enable_transformer
        self.enable_rl = enable_rl
        self.ensemble_mode = EnsembleMode(ensemble_mode)
        self.confidence_threshold = confidence_threshold

        # Normalize weights
        total_weight = 0
        if enable_gnn:
            total_weight += gnn_weight
        if enable_transformer:
            total_weight += transformer_weight
        if enable_rl:
            total_weight += rl_weight

        if total_weight > 0:
            self.gnn_weight = gnn_weight / total_weight if enable_gnn else 0.0
            self.transformer_weight = transformer_weight / total_weight if enable_transformer else 0.0
            self.rl_weight = rl_weight / total_weight if enable_rl else 0.0
        else:
            self.gnn_weight = 0.0
            self.transformer_weight = 0.0
            self.rl_weight = 0.0

        # Initialize components

        # GNN for local fusion decisions
        if enable_gnn:
            self.fusion_gnn = FusionGNN(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                num_layers=gnn_num_layers,
            )

            self.scheduling_gnn = SchedulingGNN(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                num_layers=gnn_num_layers,
            )

            # GNN confidence estimator
            self.gnn_confidence = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:
            self.fusion_gnn = None
            self.scheduling_gnn = None
            self.gnn_confidence = None

        # Transformer for global scheduling
        if enable_transformer:
            self.transformer = SchedulingTransformer(
                node_feat_dim=node_feat_dim,
                d_model=transformer_dim,
                nhead=transformer_num_heads,
                num_layers=transformer_num_layers,
            )
        else:
            self.transformer = None

        # RL agent for fine-tuning
        if enable_rl:
            self.rl_agent = PPOAgent(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                state_dim=hidden_dim,
                max_actions=128,
            )
        else:
            self.rl_agent = None

        # Meta-learner: Combines predictions from different models
        if enable_gnn and enable_transformer:
            fusion_input_dim = hidden_dim + transformer_dim
        elif enable_gnn:
            fusion_input_dim = hidden_dim
        elif enable_transformer:
            fusion_input_dim = transformer_dim
        else:
            fusion_input_dim = 128

        self.meta_learner = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # Final prediction heads
        self.priority_head = nn.Linear(64, 1)
        self.fusion_decision_head = nn.Linear(64, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        sequence_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.

        Args:
            node_features: Node features [num_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            sequence_features: Sequential representation for transformer
                              [batch_size, seq_len, node_feat_dim]
            mask: Attention mask for transformer [batch_size, seq_len]
            return_details: Return detailed predictions from each model

        Returns:
            Dictionary with ensemble predictions and optionally individual model outputs
        """
        predictions = {}
        embeddings = []

        # GNN predictions
        if self.enable_gnn:
            gnn_fusion_output = self.fusion_gnn(
                node_features, edge_index, edge_attr
            )
            gnn_scheduling_output = self.scheduling_gnn(
                node_features, edge_index, edge_attr
            )

            gnn_embeddings = gnn_fusion_output['node_embeddings']
            gnn_priority = gnn_scheduling_output['priority_scores']
            gnn_fusion_matrix = gnn_fusion_output['fusion_matrix']

            # Compute GNN confidence
            gnn_confidence = self.gnn_confidence(gnn_embeddings).squeeze(-1)

            embeddings.append(gnn_embeddings)

            predictions['gnn'] = {
                'priority_scores': gnn_priority,
                'fusion_matrix': gnn_fusion_matrix,
                'confidence': gnn_confidence,
                'embeddings': gnn_embeddings,
            }

        # Transformer predictions
        if self.enable_transformer:
            if sequence_features is None:
                # Convert graph to sequence (simple: use node features)
                sequence_features = node_features.unsqueeze(0)

            transformer_output = self.transformer(
                sequence_features, mask, return_attention=False
            )

            transformer_embeddings = transformer_output['embeddings']
            transformer_priority = transformer_output['priority_scores']
            transformer_fusion = transformer_output['fusion_logits']
            transformer_confidence = transformer_output['confidence_scores']

            # Handle batched output
            if transformer_embeddings.dim() == 3:
                transformer_embeddings = transformer_embeddings.squeeze(0)
                transformer_priority = transformer_priority.squeeze(0)
                transformer_confidence = transformer_confidence.squeeze(0)

            embeddings.append(transformer_embeddings)

            predictions['transformer'] = {
                'priority_scores': transformer_priority,
                'fusion_logits': transformer_fusion,
                'confidence': transformer_confidence,
                'embeddings': transformer_embeddings,
            }

        # RL predictions (if in inference mode)
        if self.enable_rl and not self.training:
            state_encoding = self.rl_agent.encode_state(
                node_features, edge_index, edge_attr
            )

            # For inference, RL provides additional signals
            # (Full RL interaction requires environment)
            predictions['rl'] = {
                'state_encoding': state_encoding,
            }

        # Ensemble fusion
        if self.ensemble_mode == EnsembleMode.WEIGHTED_VOTE:
            final_predictions = self._weighted_vote(predictions)
        elif self.ensemble_mode == EnsembleMode.CASCADE:
            final_predictions = self._cascade_fusion(predictions)
        elif self.ensemble_mode == EnsembleMode.HYBRID:
            final_predictions = self._hybrid_fusion(predictions, embeddings)
        elif self.ensemble_mode == EnsembleMode.GNN_ONLY:
            final_predictions = predictions.get('gnn', {})
        elif self.ensemble_mode == EnsembleMode.TRANSFORMER_ONLY:
            final_predictions = predictions.get('transformer', {})
        elif self.ensemble_mode == EnsembleMode.RL_ONLY:
            final_predictions = predictions.get('rl', {})
        else:
            final_predictions = self._hybrid_fusion(predictions, embeddings)

        if return_details:
            final_predictions['individual_predictions'] = predictions

        return final_predictions

    def _weighted_vote(self, predictions: Dict) -> Dict[str, torch.Tensor]:
        """
        Combine predictions using weighted voting based on confidence.

        Args:
            predictions: Dictionary with predictions from each model

        Returns:
            Combined predictions
        """
        priority_scores = []
        weights = []

        # Collect predictions and confidence weights
        if 'gnn' in predictions:
            priority_scores.append(predictions['gnn']['priority_scores'])
            weights.append(predictions['gnn']['confidence'] * self.gnn_weight)

        if 'transformer' in predictions:
            priority_scores.append(predictions['transformer']['priority_scores'])
            weights.append(predictions['transformer']['confidence'] * self.transformer_weight)

        if len(priority_scores) == 0:
            # No predictions available
            return {}

        # Stack and normalize weights
        priority_scores = torch.stack(priority_scores)  # [num_models, num_nodes]
        weights = torch.stack(weights)  # [num_models, num_nodes]

        # Normalize weights per node
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        # Weighted combination
        final_priority = (priority_scores * weights).sum(dim=0)

        # Average confidence
        avg_confidence = weights.max(dim=0)[0]

        return {
            'priority_scores': final_priority,
            'confidence': avg_confidence,
        }

    def _cascade_fusion(self, predictions: Dict) -> Dict[str, torch.Tensor]:
        """
        Cascade decision making: use models in order, fallback if low confidence.

        Order: GNN -> Transformer -> RL

        Args:
            predictions: Dictionary with predictions from each model

        Returns:
            Predictions from first high-confidence model
        """
        # Try GNN first
        if 'gnn' in predictions:
            gnn_conf = predictions['gnn']['confidence']
            if (gnn_conf >= self.confidence_threshold).all():
                return predictions['gnn']

        # Fallback to Transformer
        if 'transformer' in predictions:
            trans_conf = predictions['transformer']['confidence']
            if (trans_conf >= self.confidence_threshold).all():
                return {
                    'priority_scores': predictions['transformer']['priority_scores'],
                    'confidence': trans_conf,
                }

        # Last resort: return best available
        if 'gnn' in predictions:
            return predictions['gnn']
        elif 'transformer' in predictions:
            return {
                'priority_scores': predictions['transformer']['priority_scores'],
                'confidence': predictions['transformer']['confidence'],
            }
        else:
            return {}

    def _hybrid_fusion(
        self,
        predictions: Dict,
        embeddings: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Intelligent fusion using meta-learner.

        Args:
            predictions: Dictionary with predictions from each model
            embeddings: List of embeddings from different models

        Returns:
            Fused predictions
        """
        if len(embeddings) == 0:
            return {}

        # Concatenate embeddings
        combined_embeddings = torch.cat(embeddings, dim=-1)

        # Meta-learner processes combined embeddings
        meta_features = self.meta_learner(combined_embeddings)

        # Final predictions
        final_priority = self.priority_head(meta_features).squeeze(-1)
        fusion_scores = torch.sigmoid(self.fusion_decision_head(meta_features).squeeze(-1))

        # Compute confidence as average of individual model confidences
        confidences = []
        if 'gnn' in predictions:
            confidences.append(predictions['gnn']['confidence'])
        if 'transformer' in predictions:
            confidences.append(predictions['transformer']['confidence'])

        if confidences:
            avg_confidence = torch.stack(confidences).mean(dim=0)
        else:
            avg_confidence = torch.ones_like(final_priority) * 0.5

        return {
            'priority_scores': final_priority,
            'fusion_scores': fusion_scores,
            'confidence': avg_confidence,
            'meta_features': meta_features,
        }

    def predict_fusion(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_confidence: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict fusion decisions for all node pairs.

        Args:
            node_features: Node features [num_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            return_confidence: Return confidence scores

        Returns:
            Fusion matrix [num_nodes, num_nodes] and optionally confidence scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                node_features, edge_index, edge_attr, return_details=False
            )

            if 'fusion_scores' in outputs:
                # From hybrid fusion
                fusion_scores = outputs['fusion_scores']
                # Convert to matrix (simplified)
                num_nodes = node_features.size(0)
                fusion_matrix = torch.zeros(num_nodes, num_nodes, device=fusion_scores.device)
                # Diagonal entries
                for i in range(min(num_nodes, fusion_scores.size(0))):
                    fusion_matrix[i, i] = fusion_scores[i]
            elif 'gnn' in outputs.get('individual_predictions', {}):
                # From GNN
                fusion_matrix = outputs['individual_predictions']['gnn']['fusion_matrix']
            else:
                # No fusion predictions available
                num_nodes = node_features.size(0)
                fusion_matrix = torch.zeros(num_nodes, num_nodes, device=node_features.device)

            if return_confidence:
                confidence = outputs.get('confidence', None)
                return fusion_matrix, confidence
            else:
                return fusion_matrix, None

    def predict_schedule_order(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict optimal scheduling order.

        Args:
            node_features: Node features [num_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]

        Returns:
            Ordered node indices [num_nodes]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                node_features, edge_index, edge_attr, return_details=False
            )

            priority_scores = outputs['priority_scores']

            # Sort by priority (descending)
            ordered_indices = torch.argsort(priority_scores, descending=True)

            return ordered_indices

    def get_model_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the ensemble configuration.

        Returns:
            Dictionary with model statistics
        """
        stats = {
            'ensemble_mode': self.ensemble_mode.value,
            'enabled_components': {
                'gnn': self.enable_gnn,
                'transformer': self.enable_transformer,
                'rl': self.enable_rl,
            },
            'weights': {
                'gnn': self.gnn_weight,
                'transformer': self.transformer_weight,
                'rl': self.rl_weight,
            },
            'confidence_threshold': self.confidence_threshold,
        }

        # Count parameters
        if self.enable_gnn:
            stats['gnn_params'] = sum(p.numel() for p in self.fusion_gnn.parameters())
            stats['gnn_params'] += sum(p.numel() for p in self.scheduling_gnn.parameters())

        if self.enable_transformer:
            stats['transformer_params'] = sum(p.numel() for p in self.transformer.parameters())

        if self.enable_rl:
            stats['rl_params'] = sum(p.numel() for p in self.rl_agent.parameters())

        stats['total_params'] = sum(p.numel() for p in self.parameters())

        return stats

    def set_ensemble_mode(self, mode: str):
        """Change ensemble mode dynamically."""
        self.ensemble_mode = EnsembleMode(mode)

    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.confidence_threshold = threshold

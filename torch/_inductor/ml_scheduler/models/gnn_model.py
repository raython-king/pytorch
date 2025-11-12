"""
Graph Neural Network models for fusion prediction and scheduling.

Enhanced with:
- Confidence scoring for predictions
- Attention visualization for interpretability
- Graph pooling for graph-level predictions
- Inference optimizations (batch processing, caching)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class FusionGNN(nn.Module):
    """
    GNN for predicting fusion decisions between pairs of nodes.
    
    Architecture:
    - Node encoder: MLP(node_features) -> node_embedding
    - Edge encoder: MLP(edge_features) -> edge_embedding
    - Message passing: 4 layers of GATv2Conv
    - Pairwise decoder: MLP(concat(node_i, node_j)) -> fusion_score
    
    Input:
        - x: Node features [num_nodes, node_feat_dim]
        - edge_index: Edge connectivity [2, num_edges]
        - edge_attr: Edge features [num_edges, edge_feat_dim]
        
    Output:
        - fusion_matrix: Pairwise fusion scores [num_nodes, num_nodes]
    """
    
    def __init__(
        self,
        node_feat_dim: int = 64,
        edge_feat_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install with: pip install torch_geometric"
            )
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_feat_dim),
            nn.ReLU(),
        )
        
        # Message passing layers (Graph Attention Networks)
        self.conv_layers = nn.ModuleList([
            GATv2Conv(
                hidden_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                edge_dim=edge_feat_dim,
                dropout=dropout,
                concat=True,
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Pairwise fusion decoder
        self.fusion_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Confidence score in [0, 1]
        )

        # Graph pooling layers
        self.graph_pool_mean = global_mean_pool
        self.graph_pool_max = global_max_pool

        # Graph-level prediction head (for graph-level properties)
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat mean and max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),  # Predict graph-level properties
        )

        # Attention tracking for visualization
        self.attention_weights = []
        self.track_attention = False

        # Inference cache
        self._cache_enabled = False
        self._embedding_cache = {}
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Forward pass with enhanced features.

        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            batch: Batch assignment [num_nodes] (for batched graphs)
            return_attention: Return attention weights for visualization

        Returns:
            dict with:
            - 'node_embeddings': Final node embeddings [num_nodes, hidden_dim]
            - 'fusion_matrix': Fusion scores [num_nodes, num_nodes]
            - 'confidence_scores': Confidence per node [num_nodes]
            - 'graph_features': Graph-level features (if batch provided)
            - 'attention_weights': Attention weights (if return_attention=True)
        """
        # Check cache for embeddings
        cache_key = self._get_cache_key(x, edge_index)
        if self._cache_enabled and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Clear attention tracking
        if return_attention or self.track_attention:
            self.attention_weights = []

        # Encode features
        x = self.node_encoder(x)

        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # Message passing with attention tracking
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            x_residual = x

            # Get attention weights if needed (GAT provides them)
            if return_attention or self.track_attention:
                x, (edge_index_att, attention) = conv(
                    x, edge_index, edge_attr, return_attention_weights=True
                )
                self.attention_weights.append({
                    'layer': i,
                    'edge_index': edge_index_att,
                    'attention': attention,
                })
            else:
                x = conv(x, edge_index, edge_attr)

            x = norm(x)
            x = F.relu(x)

            # Residual connection
            if i > 0:
                x = x + x_residual

        # Compute confidence scores
        confidence_scores = self.confidence_head(x).squeeze(-1)

        # Compute pairwise fusion scores (optimized for batch processing)
        num_nodes = x.size(0)
        fusion_matrix = self._compute_fusion_matrix_batched(x, num_nodes)

        result = {
            'node_embeddings': x,
            'fusion_matrix': fusion_matrix,
            'confidence_scores': confidence_scores,
        }

        # Graph-level features (if batch provided)
        if batch is not None:
            graph_features = self._compute_graph_features(x, batch)
            result['graph_features'] = graph_features

        # Attention weights
        if return_attention or self.track_attention:
            result['attention_weights'] = self.attention_weights

        # Cache result if enabled
        if self._cache_enabled:
            self._embedding_cache[cache_key] = result

        return result

    def _compute_fusion_matrix_batched(
        self,
        x: torch.Tensor,
        num_nodes: int,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Compute fusion matrix with batched processing for efficiency.

        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            num_nodes: Number of nodes
            batch_size: Batch size for processing pairs

        Returns:
            Fusion matrix [num_nodes, num_nodes]
        """
        fusion_matrix = torch.zeros(num_nodes, num_nodes, device=x.device)

        # Process pairs in batches
        pairs = []
        indices = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                pairs.append(torch.cat([x[i], x[j]], dim=0))
                indices.append((i, j))

                # Process batch
                if len(pairs) >= batch_size or (i == num_nodes - 1 and j == num_nodes - 1):
                    if pairs:
                        pair_batch = torch.stack(pairs)
                        fusion_scores = torch.sigmoid(self.fusion_decoder(pair_batch)).squeeze(-1)

                        # Fill matrix
                        for k, (ii, jj) in enumerate(indices):
                            fusion_matrix[ii, jj] = fusion_scores[k]
                            fusion_matrix[jj, ii] = fusion_scores[k]

                        pairs = []
                        indices = []

        return fusion_matrix

    def _compute_graph_features(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute graph-level features using pooling.

        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph features [num_graphs, 16]
        """
        # Multiple pooling strategies
        mean_pool = self.graph_pool_mean(x, batch)
        max_pool = self.graph_pool_max(x, batch)

        # Concatenate pooled features
        graph_repr = torch.cat([mean_pool, max_pool], dim=-1)

        # Graph-level predictions
        graph_features = self.graph_head(graph_repr)

        return graph_features

    def _get_cache_key(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> str:
        """
        Generate cache key for embeddings.

        Args:
            x: Node features
            edge_index: Edge connectivity

        Returns:
            Cache key (string)
        """
        # Simple hash based on shapes and device
        key = f"{x.shape}_{edge_index.shape}_{x.device}"
        return key

    def enable_cache(self):
        """Enable embedding caching for inference."""
        self._cache_enabled = True

    def disable_cache(self):
        """Disable embedding caching."""
        self._cache_enabled = False
        self._embedding_cache.clear()

    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()

    def enable_attention_tracking(self):
        """Enable attention weight tracking for visualization."""
        self.track_attention = True

    def disable_attention_tracking(self):
        """Disable attention weight tracking."""
        self.track_attention = False

    def get_attention_weights(self) -> List[Dict]:
        """Get tracked attention weights."""
        return self.attention_weights

    def visualize_attention(
        self,
        layer_idx: int = -1,
        max_nodes: int = 20,
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights for visualization.

        Args:
            layer_idx: Which layer to visualize (-1 for last layer)
            max_nodes: Maximum nodes to include in visualization

        Returns:
            Attention matrix [num_nodes, num_nodes] or None
        """
        if not self.attention_weights:
            return None

        attention_data = self.attention_weights[layer_idx]
        edge_index = attention_data['edge_index']
        attention = attention_data['attention']

        # Determine number of nodes
        num_nodes = min(edge_index.max().item() + 1, max_nodes)

        # Create attention matrix
        attn_matrix = torch.zeros(num_nodes, num_nodes, device=attention.device)

        # Fill matrix (average over attention heads)
        for k in range(edge_index.size(1)):
            src, dst = edge_index[:, k]
            if src < num_nodes and dst < num_nodes:
                # Average over heads if multi-head attention
                if attention.dim() == 2:
                    attn_matrix[src, dst] = attention[k].mean()
                else:
                    attn_matrix[src, dst] = attention[k]

        return attn_matrix
    
    def predict_pairwise(
        self,
        node_i: torch.Tensor,
        node_j: torch.Tensor,
        return_confidence: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict fusion score for a single pair of nodes.

        Args:
            node_i: Node embedding [hidden_dim]
            node_j: Node embedding [hidden_dim]
            return_confidence: Return confidence scores

        Returns:
            Fusion score [1] and optionally confidence scores
        """
        pair_embedding = torch.cat([node_i, node_j], dim=0)
        fusion_score = torch.sigmoid(self.fusion_decoder(pair_embedding))

        if return_confidence:
            # Average confidence of both nodes
            conf_i = self.confidence_head(node_i.unsqueeze(0))
            conf_j = self.confidence_head(node_j.unsqueeze(0))
            confidence = (conf_i + conf_j) / 2
            return fusion_score, confidence
        else:
            return fusion_score, None

    def quantize(self, dtype=torch.qint8):
        """
        Quantize model for faster inference.

        Args:
            dtype: Quantization dtype (torch.qint8 or torch.quint8)

        Returns:
            Quantized model
        """
        # Prepare model for quantization
        self.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            self, {nn.Linear}, dtype=dtype
        )
        return quantized_model

    def export_onnx(
        self,
        filepath: str,
        num_nodes: int = 32,
        num_edges: int = 64,
    ):
        """
        Export model to ONNX format.

        Args:
            filepath: Path to save ONNX model
            num_nodes: Example number of nodes
            num_edges: Example number of edges
        """
        self.eval()

        # Create dummy inputs
        x = torch.randn(num_nodes, self.node_feat_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, self.edge_feat_dim)

        # Export
        torch.onnx.export(
            self,
            (x, edge_index, edge_attr),
            filepath,
            input_names=['node_features', 'edge_index', 'edge_attr'],
            output_names=['node_embeddings', 'fusion_matrix', 'confidence_scores'],
            dynamic_axes={
                'node_features': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'edge_attr': {0: 'num_edges'},
                'node_embeddings': {0: 'num_nodes'},
                'fusion_matrix': {0: 'num_nodes', 1: 'num_nodes'},
                'confidence_scores': {0: 'num_nodes'},
            },
            opset_version=14,
        )

    def batch_inference(
        self,
        batch_graphs: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Perform batched inference on multiple graphs efficiently.

        Args:
            batch_graphs: List of graph dictionaries with keys:
                         'x', 'edge_index', 'edge_attr'

        Returns:
            List of prediction dictionaries
        """
        self.eval()

        # Enable cache for efficiency
        self.enable_cache()

        results = []
        with torch.no_grad():
            for graph in batch_graphs:
                output = self.forward(
                    graph['x'],
                    graph['edge_index'],
                    graph.get('edge_attr', None),
                )
                results.append(output)

        # Disable cache after batch
        self.disable_cache()

        return results


class SchedulingGNN(nn.Module):
    """
    Enhanced GNN for graph-level scheduling decisions.

    Predicts:
    - Priority scores for each node (execution order)
    - Partition assignments (for multi-GPU)
    - Memory planning decisions

    Enhanced with:
    - Confidence scoring
    - Graph pooling
    - Batch inference support
    """

    def __init__(
        self,
        node_feat_dim: int = 64,
        edge_feat_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install with: pip install torch_geometric"
            )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Similar encoder as FusionGNN
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing
        self.conv_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_feat_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Multiple prediction heads
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Priority score
        )

        self.partition_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),  # 16 possible partitions
        )

        self.memory_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # Memory planning: [allocate, reuse, deallocate]
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Graph pooling
        self.graph_pool_mean = global_mean_pool
        self.graph_pool_max = global_max_pool

        # Graph-level features
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple prediction heads.

        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Dictionary with predictions
        """
        # Encode
        x = self.node_encoder(x)

        # Message passing with residual connections
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)

            # Residual connection (after first layer)
            if i > 0:
                x = x + x_residual

        # Predictions
        priority_scores = self.priority_head(x).squeeze(-1)  # [num_nodes]
        partition_logits = self.partition_head(x)  # [num_nodes, 16]
        memory_logits = self.memory_head(x)  # [num_nodes, 3]
        confidence_scores = self.confidence_head(x).squeeze(-1)  # [num_nodes]

        result = {
            'priority_scores': priority_scores,
            'partition_logits': partition_logits,
            'memory_logits': memory_logits,
            'confidence_scores': confidence_scores,
            'node_embeddings': x,
        }

        # Graph-level features
        if batch is not None:
            mean_pool = self.graph_pool_mean(x, batch)
            max_pool = self.graph_pool_max(x, batch)
            graph_repr = torch.cat([mean_pool, max_pool], dim=-1)
            graph_features = self.graph_head(graph_repr)
            result['graph_features'] = graph_features

        return result

    def predict_schedule_order(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict scheduling order.

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features

        Returns:
            Ordered node indices [num_nodes]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, edge_index, edge_attr)
            priority_scores = outputs['priority_scores']
            return torch.argsort(priority_scores, descending=True)

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence filtering.

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            confidence_threshold: Minimum confidence

        Returns:
            Predictions with high-confidence mask
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, edge_index, edge_attr)

            high_confidence_mask = outputs['confidence_scores'] >= confidence_threshold

            return {
                'priority_scores': outputs['priority_scores'],
                'partition_logits': outputs['partition_logits'],
                'memory_logits': outputs['memory_logits'],
                'confidence_scores': outputs['confidence_scores'],
                'high_confidence_mask': high_confidence_mask,
            }

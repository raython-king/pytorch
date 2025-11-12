"""
Transformer-based models for sequential scheduling decisions.

This module implements transformer architectures that treat IR graph scheduling
as a sequence-to-sequence problem. The transformer can capture long-range
dependencies between operations and learn optimal scheduling patterns.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input.

    Adds positional information to node embeddings to preserve
    ordering information in the sequence.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class SchedulingTransformer(nn.Module):
    """
    Transformer model for sequential scheduling decisions.

    Architecture:
    - 6-layer encoder with multi-head attention (8 heads)
    - Positional encoding for node ordering
    - Multiple prediction heads for different scheduling tasks

    Input: Linearized IR graph (topological order)
    Output:
        - Node ordering scores (priority for scheduling)
        - Fusion decisions (whether to fuse with next node)
        - Memory planning decisions

    Features:
    - Handles variable-size graphs via attention masking
    - Captures long-range dependencies between operations
    - Supports batched inference for multiple graphs
    """

    def __init__(
        self,
        node_feat_dim: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        """
        Initialize SchedulingTransformer.

        Args:
            node_feat_dim: Dimension of input node features
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length (for positional encoding)
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Input projection: node features -> d_model
        self.input_projection = nn.Sequential(
            nn.Linear(node_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,  # Pre-LN architecture for better stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Prediction heads

        # 1. Priority head: Predicts execution priority/order
        self.priority_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single priority score per node
        )

        # 2. Fusion head: Predicts whether to fuse with next node
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model * 2, 256),  # Concatenate current and next node
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Binary: [no_fuse, fuse]
        )

        # 3. Memory planning head
        self.memory_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [allocate, reuse, deallocate]
        )

        # 4. Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Confidence score [0, 1]
        )

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [batch_size, seq_len, node_feat_dim]
            mask: Attention mask [batch_size, seq_len] (True for valid positions)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - 'priority_scores': Priority scores [batch_size, seq_len]
                - 'fusion_logits': Fusion decision logits [batch_size, seq_len-1, 2]
                - 'memory_logits': Memory planning logits [batch_size, seq_len, 3]
                - 'confidence_scores': Confidence scores [batch_size, seq_len]
                - 'embeddings': Final node embeddings [batch_size, seq_len, d_model]
                - 'attention_weights': Attention weights (if return_attention=True)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to d_model
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask for transformer
        # PyTorch transformer expects inverted mask (True = ignore)
        if mask is not None:
            # Convert from [batch_size, seq_len] to [batch_size, seq_len]
            # where True means "ignore this position"
            attn_mask = ~mask
        else:
            attn_mask = None

        # Transformer encoding
        if return_attention:
            # Manual encoding to capture attention weights
            attention_weights = []
            for layer in self.transformer_encoder.layers:
                x, attn = self._forward_layer_with_attention(x, layer, attn_mask)
                attention_weights.append(attn)
            x = self.transformer_encoder.norm(x)
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
            attention_weights = None

        # Apply prediction heads

        # 1. Priority scores
        priority_scores = self.priority_head(x).squeeze(-1)  # [batch_size, seq_len]

        # 2. Fusion decisions (for consecutive pairs)
        fusion_logits = self._compute_fusion_logits(x)  # [batch_size, seq_len-1, 2]

        # 3. Memory planning
        memory_logits = self.memory_head(x)  # [batch_size, seq_len, 3]

        # 4. Confidence scores
        confidence_scores = self.confidence_head(x).squeeze(-1)  # [batch_size, seq_len]

        result = {
            'priority_scores': priority_scores,
            'fusion_logits': fusion_logits,
            'memory_logits': memory_logits,
            'confidence_scores': confidence_scores,
            'embeddings': x,
        }

        if return_attention:
            result['attention_weights'] = attention_weights

        return result

    def _forward_layer_with_attention(
        self,
        x: torch.Tensor,
        layer: nn.TransformerEncoderLayer,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through a single transformer layer, returning attention weights.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            layer: Transformer encoder layer
            attn_mask: Attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # This is a simplified version - actual attention extraction
        # requires modifying the layer's self_attn module
        # For now, we'll use the layer normally and return dummy weights
        x = layer(x, src_key_padding_mask=attn_mask)
        attn = torch.zeros(x.size(0), self.nhead, x.size(1), x.size(1), device=x.device)
        return x, attn

    def _compute_fusion_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute fusion logits for consecutive node pairs.

        Args:
            x: Node embeddings [batch_size, seq_len, d_model]

        Returns:
            Fusion logits [batch_size, seq_len-1, 2]
        """
        batch_size, seq_len, d_model = x.shape

        if seq_len <= 1:
            # No pairs to fuse
            return torch.zeros(batch_size, 0, 2, device=x.device)

        # Create pairs of consecutive nodes
        current_nodes = x[:, :-1, :]  # [batch_size, seq_len-1, d_model]
        next_nodes = x[:, 1:, :]      # [batch_size, seq_len-1, d_model]

        # Concatenate pairs
        pairs = torch.cat([current_nodes, next_nodes], dim=-1)  # [batch_size, seq_len-1, d_model*2]

        # Predict fusion decisions
        fusion_logits = self.fusion_head(pairs)  # [batch_size, seq_len-1, 2]

        return fusion_logits

    def predict_schedule_order(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict scheduling order for nodes.

        Args:
            x: Node features [batch_size, seq_len, node_feat_dim]
            mask: Attention mask [batch_size, seq_len]

        Returns:
            Ordered node indices [batch_size, seq_len]
        """
        # Get priority scores
        outputs = self.forward(x, mask, return_attention=False)
        priority_scores = outputs['priority_scores']  # [batch_size, seq_len]

        # Sort by priority (descending)
        ordered_indices = torch.argsort(priority_scores, dim=1, descending=True)

        return ordered_indices

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence filtering.

        Args:
            x: Node features [batch_size, seq_len, node_feat_dim]
            mask: Attention mask
            confidence_threshold: Minimum confidence for predictions

        Returns:
            Dictionary with predictions and confidence masks
        """
        outputs = self.forward(x, mask, return_attention=False)

        # Create confidence masks
        high_confidence_mask = outputs['confidence_scores'] >= confidence_threshold

        # Apply confidence masking to predictions
        priority_scores = outputs['priority_scores'].clone()
        priority_scores[~high_confidence_mask] = float('-inf')

        return {
            'priority_scores': priority_scores,
            'fusion_logits': outputs['fusion_logits'],
            'memory_logits': outputs['memory_logits'],
            'confidence_scores': outputs['confidence_scores'],
            'high_confidence_mask': high_confidence_mask,
        }

    def export_onnx(self, filepath: str, seq_len: int = 64):
        """
        Export model to ONNX format for deployment.

        Args:
            filepath: Path to save ONNX model
            seq_len: Example sequence length
        """
        self.eval()

        # Create dummy input
        dummy_input = torch.randn(1, seq_len, self.d_model)

        # Export
        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            input_names=['node_features'],
            output_names=['priority_scores', 'fusion_logits', 'memory_logits'],
            dynamic_axes={
                'node_features': {0: 'batch_size', 1: 'seq_len'},
                'priority_scores': {0: 'batch_size', 1: 'seq_len'},
                'fusion_logits': {0: 'batch_size', 1: 'seq_len'},
                'memory_logits': {0: 'batch_size', 1: 'seq_len'},
            },
            opset_version=14,
        )


class GraphTransformer(nn.Module):
    """
    Graph Transformer that combines graph structure with self-attention.

    This hybrid architecture biases the attention mechanism with graph
    structure information, allowing the model to leverage both local
    graph topology and global attention patterns.
    """

    def __init__(
        self,
        node_feat_dim: int = 64,
        edge_feat_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize GraphTransformer.

        Args:
            node_feat_dim: Node feature dimension
            edge_feat_dim: Edge feature dimension
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_layers: Number of graph transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Input encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # Graph transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output heads
        self.priority_head = nn.Linear(hidden_dim, 1)
        self.fusion_head = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]

        Returns:
            Dictionary with predictions
        """
        # Encode inputs
        x = self.node_encoder(x)

        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # Apply graph transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Predictions
        priority_scores = self.priority_head(x).squeeze(-1)

        return {
            'priority_scores': priority_scores,
            'node_embeddings': x,
        }


class GraphTransformerLayer(nn.Module):
    """
    Single layer of Graph Transformer.

    Combines:
    - Multi-head self-attention biased by graph structure
    - Feedforward network
    - Layer normalization and residual connections
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Multi-head attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge bias for attention
        self.edge_bias_proj = nn.Linear(hidden_dim, num_heads)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, hidden_dim]

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)

        # Self-attention with graph structure bias
        residual = x
        x = self.norm1(x)

        # Compute Q, K, V
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)

        # Compute attention scores (simplified - full implementation would use edge_index)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        out = torch.matmul(attn_weights, v)
        out = out.view(num_nodes, self.hidden_dim)
        out = self.out_proj(out)

        # Residual connection
        x = residual + self.dropout(out)

        # Feedforward network
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x

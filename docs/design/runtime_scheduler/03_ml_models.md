# ML Models for Runtime Scheduling

## Overview

This document describes the machine learning models used for making scheduling decisions in the Runtime Scheduler.

---

## Model Architecture Selection

### Model Comparison

| Model Type | Strengths | Weaknesses | Use Case |
|-----------|-----------|------------|----------|
| **GNN** | Graph structure, dependencies | Slower inference | Fusion, data dependencies |
| **Transformer** | Sequence modeling, attention | Memory intensive | Operation ordering |
| **MLP** | Fast inference, simple | Limited expressiveness | Quick decisions, fallback |
| **RL** | Adapts online, explores | Training instability | Online learning |
| **Ensemble** | Best accuracy, robust | Higher overhead | Production deployment |

### Recommended Architecture: Hybrid Ensemble

```
┌──────────────────────────────────────────────────────────┐
│                    Ensemble Model                         │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Fast Path   │  │  GNN Path    │  │ Transformer  │  │
│  │    (MLP)     │  │              │  │    Path      │  │
│  │              │  │              │  │              │  │
│  │ Features ──> │  │ Features ──> │  │ Features ──> │  │
│  │  Decision    │  │  Decision    │  │  Decision    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │           │
│         └─────────────────┴──────────────────┘           │
│                           │                              │
│                           ▼                              │
│                  ┌─────────────────┐                     │
│                  │ Voting/Ensemble │                     │
│                  │    Combiner     │                     │
│                  └─────────┬───────┘                     │
│                            │                             │
│                            ▼                             │
│                   Final Decision                         │
└──────────────────────────────────────────────────────────┘

Decision Flow:
1. Check confidence threshold
2. If high confidence from fast path → use it (< 100 μs)
3. If medium confidence → ensemble vote (< 500 μs)
4. If low confidence → fallback to heuristics
```

---

## 1. Fast Path MLP Model

### Purpose
Ultra-low latency decisions for common cases (> 80% of operations).

### Architecture

```python
class FastPathMLP(nn.Module):
    """
    Fast MLP for low-latency scheduling decisions.
    
    Target: < 100 μs inference time
    """
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 32
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        op_features: torch.Tensor,  # [batch, op_dim]
        state_features: torch.Tensor  # [batch, state_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate features
        x = torch.cat([op_features, state_features], dim=-1)
        
        # Forward pass
        logits = self.model(x)  # [batch, output_dim]
        confidence = self.confidence_head(logits)  # [batch, 1]
        
        return logits, confidence

# Feature Extraction for MLP
class OperationFeatureExtractor:
    """Extract features from operation metadata."""
    
    def extract(self, op: OperationMetadata) -> torch.Tensor:
        features = [
            # Compute features (8 dims)
            math.log(op.estimated_flops + 1),  # Log FLOPs
            math.log(op.estimated_memory_bytes + 1),  # Log memory
            float(op.estimated_flops) / max(op.estimated_memory_bytes, 1),  # Compute intensity
            len(op.input_tensor_names),  # Number of inputs
            len(op.output_tensor_names),  # Number of outputs
            hash(op.op_name) % 1000 / 1000.0,  # Op type embedding
            op.sequence_number / 1000.0,  # Normalized sequence
            0.0,  # Reserved
            
            # Shape features (16 dims)
            *self._encode_shape(op.input_shapes),  # Quantized shape encoding
            
            # Data type features (4 dims)
            *self._encode_dtypes(op.input_dtypes),
            
            # Device features (4 dims)
            *self._encode_devices(op.input_devices),
        ]
        return torch.tensor(features, dtype=torch.float32)
    
    def _encode_shape(self, shapes: List[List[int]]) -> List[float]:
        """Encode shapes into fixed-size representation."""
        # Use quantized size buckets
        encoding = []
        for shape in shapes[:4]:  # Max 4 inputs
            numel = np.prod(shape) if len(shape) > 0 else 0
            # Log-scale bucketing
            bucket = min(15, int(math.log2(numel + 1)))
            encoding.append(bucket / 15.0)
        
        # Pad to 4 inputs
        encoding.extend([0.0] * (4 - len(encoding)))
        return encoding
```

### Training Data

```python
# Training sample structure
class SchedulingTrainingSample:
    """Single training sample for scheduling model."""
    
    # Features
    op_features: torch.Tensor  # [op_dim]
    state_features: torch.Tensor  # [state_dim]
    graph_features: torch.Tensor  # [graph_dim]
    
    # Labels
    optimal_device: int  # Device ID
    optimal_stream: int  # Stream ID
    optimal_priority: float  # Priority [0, 1]
    
    # Reward signal
    actual_time: float  # Actual execution time (ms)
    baseline_time: float  # Baseline time without scheduler
    reward: float  # (baseline_time - actual_time) / baseline_time
```

---

## 2. Graph Neural Network (GNN) Model

### Purpose
Model operation dependencies and data flow for fusion and reordering decisions.

### Architecture

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class SchedulingGNN(nn.Module):
    """
    GNN for operation scheduling with dependency awareness.
    
    Target: < 500 μs inference time
    """
    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4
    ):
        super().__init__()
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                edge_dim=hidden_dim,
                concat=True
            )
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,  # [num_nodes, node_dim]
        edge_index: torch.Tensor,  # [2, num_edges]
        edge_attr: torch.Tensor,  # [num_edges, edge_dim]
        batch: Optional[torch.Tensor] = None  # [num_nodes]
    ) -> Dict[str, torch.Tensor]:
        # Encode nodes and edges
        x = self.node_encoder(x)  # [num_nodes, hidden_dim]
        edge_attr = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]
        
        # Apply GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Node-level predictions
        fusion_scores = self.fusion_head(x).squeeze(-1)  # [num_nodes]
        priorities = self.priority_head(x).squeeze(-1)  # [num_nodes]
        
        # Graph-level aggregation (for global decisions)
        if batch is not None:
            graph_repr = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        else:
            graph_repr = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        
        return {
            'node_embeddings': x,
            'fusion_scores': fusion_scores,
            'priorities': priorities,
            'graph_repr': graph_repr
        }

# Graph construction from operations
class OperationGraphBuilder:
    """Build computation graph from operations."""
    
    def build_graph(
        self,
        ops: List[OperationMetadata]
    ) -> Data:
        """
        Build PyG Data object from operations.
        
        Nodes: Operations
        Edges: Data dependencies (producer -> consumer)
        """
        from torch_geometric.data import Data
        
        num_nodes = len(ops)
        
        # Extract node features
        node_features = []
        for op in ops:
            features = self._extract_node_features(op)
            node_features.append(features)
        
        x = torch.stack(node_features)  # [num_nodes, node_dim]
        
        # Build edges (dependencies)
        edge_index = []
        edge_features = []
        
        # Map tensor names to producing operations
        tensor_to_op = {}
        for i, op in enumerate(ops):
            for output in op.output_tensor_names:
                tensor_to_op[output] = i
        
        # Create edges
        for dst_idx, op in enumerate(ops):
            for input_tensor in op.input_tensor_names:
                if input_tensor in tensor_to_op:
                    src_idx = tensor_to_op[input_tensor]
                    edge_index.append([src_idx, dst_idx])
                    
                    # Edge features
                    edge_feat = self._extract_edge_features(
                        ops[src_idx], op, input_tensor
                    )
                    edge_features.append(edge_feat)
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.stack(edge_features)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 32))
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _extract_node_features(self, op: OperationMetadata) -> torch.Tensor:
        """Extract per-operation features."""
        features = [
            # Compute characteristics
            math.log(op.estimated_flops + 1),
            math.log(op.estimated_memory_bytes + 1),
            float(op.estimated_flops) / max(op.estimated_memory_bytes, 1),
            
            # Graph structure
            len(op.input_tensor_names),
            len(op.output_tensor_names),
            
            # ... (64 total features)
        ]
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_edge_features(
        self,
        src_op: OperationMetadata,
        dst_op: OperationMetadata,
        tensor_name: str
    ) -> torch.Tensor:
        """Extract features for data dependency edge."""
        features = [
            # Tensor size (from src op outputs)
            0.0,  # Log tensor size
            
            # Data type compatibility
            1.0 if src_op.input_dtypes == dst_op.input_dtypes else 0.0,
            
            # Device locality
            1.0 if src_op.current_device == dst_op.current_device else 0.0,
            
            # ... (32 total features)
        ]
        return torch.tensor(features, dtype=torch.float32)
```

---

## 3. Transformer Model

### Purpose
Model operation sequences and predict optimal orderings.

### Architecture

```python
class SchedulingTransformer(nn.Module):
    """
    Transformer for sequence-based scheduling decisions.
    
    Models temporal dependencies and execution patterns.
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.op_embedding = nn.Linear(128, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output heads
        self.schedule_head = nn.Linear(d_model, 1)  # Priority score
        self.device_head = nn.Linear(d_model, 8)  # Device logits
        self.stream_head = nn.Linear(d_model, 32)  # Stream logits
    
    def forward(
        self,
        op_features: torch.Tensor,  # [batch, seq_len, 128]
        mask: Optional[torch.Tensor] = None  # [batch, seq_len]
    ) -> Dict[str, torch.Tensor]:
        # Embed and add positional encoding
        x = self.op_embedding(op_features)  # [batch, seq_len, d_model]
        x = self.pos_encoding(x)
        
        # Create attention mask if needed
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, self.nhead, mask.size(1), -1)
        else:
            attn_mask = None
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=mask)
        
        # Predictions
        priorities = self.schedule_head(encoded).squeeze(-1)  # [batch, seq_len]
        device_logits = self.device_head(encoded)  # [batch, seq_len, 8]
        stream_logits = self.stream_head(encoded)  # [batch, seq_len, 32]
        
        return {
            'priorities': priorities,
            'device_logits': device_logits,
            'stream_logits': stream_logits,
            'embeddings': encoded
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]
```

---

## 4. Reinforcement Learning Model

### Purpose
Online learning and adaptation to workload changes.

### Architecture

```python
class SchedulingRLAgent(nn.Module):
    """
    RL agent for adaptive scheduling.
    
    Uses PPO (Proximal Policy Optimization) for stable training.
    """
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 64,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = self.actor(state)  # [batch, action_dim]
        value = self.critic(state)  # [batch, 1]
        return action_probs, value
    
    def select_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Returns:
            action: Selected action index
            log_prob: Log probability of action
            value: State value estimate
        """
        action_probs, value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()

# RL Training Loop
class RLTrainer:
    """Trainer for RL scheduling agent."""
    
    def __init__(self, agent: SchedulingRLAgent):
        self.agent = agent
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.clip_epsilon = 0.2  # PPO clip
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
    
    def compute_reward(
        self,
        decision: SchedulingDecision,
        actual_time: float,
        baseline_time: float
    ) -> float:
        """
        Compute reward for scheduling decision.
        
        Reward components:
        1. Speedup over baseline
        2. Memory efficiency
        3. Load balance
        """
        # Speedup reward
        speedup = (baseline_time - actual_time) / baseline_time
        speedup_reward = max(0, speedup) * 10.0
        
        # Memory reward (penalize high memory usage)
        memory_usage = decision.memory.allocation_size / (1024**3)  # GB
        memory_reward = -0.1 * memory_usage
        
        # Load balance reward
        # ... (compute based on device utilization variance)
        
        total_reward = speedup_reward + memory_reward
        return total_reward
    
    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """
        PPO training step.
        
        batch: List of experience tuples {state, action, reward, next_state, done}
        """
        # ... (PPO implementation)
        pass
```

---

## 5. Ensemble Model

### Purpose
Combine multiple models for robust predictions.

### Architecture

```python
class EnsembleScheduler(nn.Module):
    """
    Ensemble of scheduling models for robust decisions.
    """
    def __init__(
        self,
        mlp_model: FastPathMLP,
        gnn_model: SchedulingGNN,
        transformer_model: SchedulingTransformer,
        confidence_threshold: float = 0.75
    ):
        super().__init__()
        
        self.mlp = mlp_model
        self.gnn = gnn_model
        self.transformer = transformer_model
        self.confidence_threshold = confidence_threshold
        
        # Ensemble weights (learned or fixed)
        self.ensemble_weights = nn.Parameter(
            torch.tensor([0.5, 0.3, 0.2])  # MLP, GNN, Transformer
        )
    
    def forward(
        self,
        op_features: torch.Tensor,
        state_features: torch.Tensor,
        graph_data: Optional[Data] = None,
        use_fast_path: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Ensemble prediction with adaptive model selection.
        """
        # Fast path: Try MLP first
        if use_fast_path:
            mlp_pred, mlp_conf = self.mlp(op_features, state_features)
            
            # If high confidence, return immediately
            if mlp_conf.mean() > self.confidence_threshold:
                return {
                    'prediction': mlp_pred,
                    'confidence': mlp_conf,
                    'model_used': 'mlp'
                }
        
        # Medium confidence: Use ensemble
        predictions = []
        confidences = []
        
        # MLP prediction
        mlp_pred, mlp_conf = self.mlp(op_features, state_features)
        predictions.append(mlp_pred)
        confidences.append(mlp_conf)
        
        # GNN prediction
        if graph_data is not None:
            gnn_out = self.gnn(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr
            )
            predictions.append(gnn_out['priorities'])
            confidences.append(torch.ones_like(mlp_conf) * 0.8)  # Fixed confidence
        
        # Ensemble combination
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = sum(
            w * p for w, p in zip(weights, predictions)
        )
        ensemble_conf = sum(
            w * c for w, c in zip(weights, confidences)
        )
        
        return {
            'prediction': ensemble_pred,
            'confidence': ensemble_conf,
            'model_used': 'ensemble'
        }
```

---

## Model Training

### Training Data Collection

```python
class TrainingDataCollector:
    """Collect training data from execution traces."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.samples = []
    
    def collect_sample(
        self,
        op: OperationMetadata,
        decision: SchedulingDecision,
        actual_time: float,
        baseline_time: float
    ):
        """Collect single training sample."""
        sample = {
            'op_features': self._extract_features(op),
            'decision': self._encode_decision(decision),
            'actual_time': actual_time,
            'baseline_time': baseline_time,
            'reward': (baseline_time - actual_time) / baseline_time
        }
        self.samples.append(sample)
    
    def save(self, filename: str):
        """Save collected samples to disk."""
        torch.save(self.samples, os.path.join(self.output_dir, filename))
```

### Training Script

```python
def train_scheduling_model(
    model: nn.Module,
    train_data: List[Dict],
    val_data: List[Dict],
    epochs: int = 100
):
    """Train scheduling model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in DataLoader(train_data, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch['features'])
            
            # Loss: MSE for regression + reward signal
            loss = F.mse_loss(pred['decision'], batch['optimal_decision'])
            loss += -batch['reward'].mean()  # Reward maximization
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in DataLoader(val_data, batch_size=32):
                pred = model(batch['features'])
                loss = F.mse_loss(pred['decision'], batch['optimal_decision'])
                val_loss += loss.item()
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        scheduler.step()
```

---

## Next Steps

See:
- [04_integration_guide.md](./04_integration_guide.md) for implementation
- [05_performance_analysis.md](./05_performance_analysis.md) for benchmarks


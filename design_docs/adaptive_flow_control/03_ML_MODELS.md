# Adaptive Flow Control - ML Models and Intelligence

## Table of Contents

1. [Overview](#overview)
2. [Model Architectures](#model-architectures)
3. [Training Infrastructure](#training-infrastructure)
4. [Online Learning](#online-learning)
5. [Feature Engineering](#feature-engineering)
6. [Performance Considerations](#performance-considerations)

---

## Overview

### ML Models in the System

```
┌────────────────────────────────────────────────────────────────┐
│                    ML Intelligence Layer                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ Bandwidth        │  │  Congestion      │                   │
│  │ Predictor        │  │  Predictor       │                   │
│  │                  │  │                  │                   │
│  │ Input: System    │  │ Input: Link      │                   │
│  │ state, history   │  │ utilization,     │                   │
│  │                  │  │ queue depth      │                   │
│  │ Output: Future   │  │ Output: Cong.    │                   │
│  │ bandwidth        │  │ probability      │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Routing         │  │  Latency         │                   │
│  │  Optimizer       │  │  Predictor       │                   │
│  │                  │  │                  │                   │
│  │ Input: Topology, │  │ Input: Transfer  │                   │
│  │ flows, load      │  │ size, path       │                   │
│  │                  │  │                  │                   │
│  │ Output: Best     │  │ Output: Expected │                   │
│  │ path selection   │  │ latency          │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Flow Size       │  │  Priority        │                   │
│  │  Estimator       │  │  Predictor       │                   │
│  │                  │  │                  │                   │
│  │ Input: Op type,  │  │ Input: Flow      │                   │
│  │ tensor shapes    │  │ features, deps   │                   │
│  │                  │  │                  │                   │
│  │ Output: Transfer │  │ Output: Optimal  │                   │
│  │ size estimate    │  │ priority score   │                   │
│  └──────────────────┘  └──────────────────┘                   │
└────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Low Latency:** Inference must be < 0.2ms (< 0.2% overhead budget)
2. **Small Models:** Keep model size < 10MB for fast loading
3. **Online Learning:** Adapt to changing conditions without retraining
4. **Robustness:** Graceful degradation on prediction errors
5. **Interpretability:** Provide confidence scores and explanations

---

## Model Architectures

### 1. Bandwidth Predictor

**Purpose:** Predict available bandwidth over next time window

**Architecture:** LSTM + Attention for time-series prediction

```python
class BandwidthPredictor(nn.Module):
    """
    Predict future bandwidth availability.
    
    Architecture:
    - LSTM for temporal patterns
    - Attention mechanism for important time steps
    - MLP for final prediction
    """
    
    def __init__(
        self,
        feature_dim=32,
        hidden_dim=64,
        num_layers=2,
        prediction_horizon=10  # Predict next 10 time steps
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, prediction_horizon)
        )
        
        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, prediction_horizon),
            nn.Sigmoid()
        )
    
    def forward(self, features, history):
        """
        Predict future bandwidth.
        
        Args:
            features: Current system features [batch, feature_dim]
            history: Historical bandwidth [batch, seq_len, feature_dim]
            
        Returns:
            predictions: [batch, prediction_horizon]
            confidence: [batch, prediction_horizon]
        """
        # LSTM processes history
        lstm_out, (h_n, c_n) = self.lstm(history)
        
        # Attention over LSTM outputs
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Use last hidden state
        final_repr = attn_out[:, -1, :]
        
        # Predict bandwidth
        predictions = self.fc(final_repr)
        
        # Estimate confidence
        confidence = self.confidence(final_repr)
        
        return predictions, confidence


# Feature extraction for bandwidth prediction
def extract_bandwidth_features(system_state, history):
    """
    Extract features for bandwidth prediction.
    
    Args:
        system_state: Current system state
        history: Historical measurements (last 100 samples)
        
    Returns:
        features: Tensor of shape [feature_dim]
    """
    features = []
    
    # Current state
    features.append(system_state.current_bandwidth_gbps)
    features.append(system_state.utilization)
    features.append(system_state.num_active_flows)
    features.append(system_state.queue_depth)
    
    # Statistical features from history
    if len(history) > 0:
        bandwidth_history = [h.bandwidth for h in history]
        features.append(np.mean(bandwidth_history))
        features.append(np.std(bandwidth_history))
        features.append(np.min(bandwidth_history))
        features.append(np.max(bandwidth_history))
        features.append(np.median(bandwidth_history))
        
        # Trend detection
        if len(bandwidth_history) >= 10:
            recent_trend = np.polyfit(
                range(len(bandwidth_history[-10:])),
                bandwidth_history[-10:],
                deg=1
            )[0]
            features.append(recent_trend)
    
    # Time-of-day features (cyclic encoding)
    hour = time.localtime().tm_hour
    features.append(np.sin(2 * np.pi * hour / 24))
    features.append(np.cos(2 * np.pi * hour / 24))
    
    # Pad to feature_dim
    while len(features) < 32:
        features.append(0.0)
    
    return torch.tensor(features[:32], dtype=torch.float32)
```

### 2. Congestion Predictor

**Purpose:** Predict congestion probability before it occurs

**Architecture:** Gradient Boosting (XGBoost) for tabular data

```python
class CongestionPredictor:
    """
    Predict congestion events.
    
    Uses XGBoost for fast inference and good performance
    on tabular features.
    """
    
    def __init__(self):
        self.model = xgb.XGBClassifier(
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',  # Fast histogram-based method
            predictor='cpu_predictor'  # Use CPU for inference
        )
        
        self.feature_names = [
            'utilization',
            'queue_depth',
            'num_flows',
            'bandwidth_variance',
            'latency_p99',
            'transfer_rate',
            'utilization_trend',
            'queue_depth_trend',
            'time_since_last_congestion',
            'flow_arrival_rate'
        ]
        
        self.scaler = StandardScaler()
    
    def extract_features(self, system_state, history):
        """Extract features for congestion prediction."""
        features = {}
        
        # Current state
        features['utilization'] = system_state.utilization
        features['queue_depth'] = system_state.queue_depth
        features['num_flows'] = system_state.num_active_flows
        features['bandwidth_variance'] = system_state.bandwidth_variance
        features['latency_p99'] = system_state.latency_p99_ms
        features['transfer_rate'] = system_state.current_transfer_rate
        
        # Trends from history
        if len(history) >= 10:
            recent = history[-10:]
            
            util_trend = np.polyfit(
                range(len(recent)),
                [h.utilization for h in recent],
                deg=1
            )[0]
            features['utilization_trend'] = util_trend
            
            queue_trend = np.polyfit(
                range(len(recent)),
                [h.queue_depth for h in recent],
                deg=1
            )[0]
            features['queue_depth_trend'] = queue_trend
        else:
            features['utilization_trend'] = 0.0
            features['queue_depth_trend'] = 0.0
        
        # Time since last congestion
        last_congestion = None
        for h in reversed(history):
            if h.congestion_detected:
                last_congestion = h.timestamp
                break
        
        if last_congestion:
            features['time_since_last_congestion'] = time.time() - last_congestion
        else:
            features['time_since_last_congestion'] = float('inf')
        
        # Flow arrival rate
        if len(history) >= 10:
            flow_arrivals = [h.flow_arrivals for h in history[-10:]]
            features['flow_arrival_rate'] = np.mean(flow_arrivals)
        else:
            features['flow_arrival_rate'] = 0.0
        
        # Convert to array in correct order
        feature_array = np.array([
            features[name] for name in self.feature_names
        ]).reshape(1, -1)
        
        return feature_array
    
    def predict(self, system_state, history):
        """
        Predict congestion probability.
        
        Returns:
            (probability, confidence)
        """
        features = self.extract_features(system_state, history)
        features_scaled = self.scaler.transform(features)
        
        # Get probability
        prob = self.model.predict_proba(features_scaled)[0, 1]
        
        # Estimate confidence based on leaf node statistics
        # (higher confidence when prediction is based on more training samples)
        leaf_ids = self.model.apply(features_scaled)
        # Simplified confidence: higher for more extreme predictions
        confidence = abs(prob - 0.5) * 2
        
        return prob, confidence
    
    def train(self, X, y, X_val=None, y_val=None):
        """Train the model."""
        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train XGBoost
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        self.model.fit(
            X_scaled,
            y,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
```

### 3. Routing Optimizer

**Purpose:** Select optimal path for transfers

**Architecture:** Graph Neural Network (GNN) + MLP

```python
class RoutingOptimizer(nn.Module):
    """
    Learn optimal routing decisions using GNN.
    
    Uses Graph Neural Network to process network topology
    and predict path quality.
    """
    
    def __init__(
        self,
        node_feature_dim=16,
        edge_feature_dim=8,
        hidden_dim=64,
        num_gnn_layers=3
    ):
        super().__init__()
        
        # Node embedding
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # Edge embedding
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)
        
        # GNN layers (Graph Attention Network)
        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            for _ in range(num_gnn_layers)
        ])
        
        # Path scoring MLP
        self.path_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat src and dst
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Path quality score
            nn.Sigmoid()
        )
    
    def forward(self, graph, path):
        """
        Score a path through the network.
        
        Args:
            graph: NetworkX graph with node and edge features
            path: List of node IDs in path
            
        Returns:
            path_score: Quality score for this path [0, 1]
        """
        # Convert graph to PyG format
        edge_index, edge_features, node_features = self._graph_to_pyg(graph)
        
        # Embed nodes and edges
        x = self.node_embedding(node_features)
        edge_attr = self.edge_embedding(edge_features)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Get embeddings for source and destination
        src_node = path[0]
        dst_node = path[-1]
        src_embedding = x[src_node]
        dst_embedding = x[dst_node]
        
        # Concatenate and score
        path_repr = torch.cat([src_embedding, dst_embedding], dim=-1)
        score = self.path_scorer(path_repr)
        
        return score
    
    def _graph_to_pyg(self, graph):
        """Convert NetworkX graph to PyTorch Geometric format."""
        # Node features
        node_features = torch.stack([
            self._get_node_features(graph, node)
            for node in graph.nodes()
        ])
        
        # Edge index and features
        edges = list(graph.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_features = torch.stack([
            self._get_edge_features(graph, u, v)
            for u, v in edges
        ])
        
        return edge_index, edge_features, node_features
    
    def _get_node_features(self, graph, node):
        """Extract features for a node (device)."""
        node_data = graph.nodes[node]
        features = [
            node_data.get('utilization', 0.0),
            node_data.get('memory_util', 0.0),
            node_data.get('num_active_transfers', 0),
            node_data.get('queue_depth', 0),
            # One-hot encoding of device type
            1.0 if node_data.get('type') == 'gpu' else 0.0,
            1.0 if node_data.get('type') == 'cpu' else 0.0,
        ]
        # Pad to feature dim
        while len(features) < 16:
            features.append(0.0)
        return torch.tensor(features[:16], dtype=torch.float32)
    
    def _get_edge_features(self, graph, u, v):
        """Extract features for an edge (link)."""
        edge_data = graph[u][v]
        features = [
            edge_data.get('bandwidth_gbps', 0.0) / 100.0,  # Normalize
            edge_data.get('utilization', 0.0),
            edge_data.get('latency_ms', 0.0) / 10.0,  # Normalize
            edge_data.get('loss_rate', 0.0) * 1000,  # Scale up
            # Link type one-hot
            1.0 if edge_data.get('type') == 'nvlink' else 0.0,
            1.0 if edge_data.get('type') == 'pcie' else 0.0,
        ]
        # Pad
        while len(features) < 8:
            features.append(0.0)
        return torch.tensor(features[:8], dtype=torch.float32)
```

### 4. Latency Predictor

**Purpose:** Predict transfer latency for scheduling decisions

**Architecture:** Simple MLP with quantile regression

```python
class LatencyPredictor(nn.Module):
    """
    Predict transfer latency with uncertainty quantification.
    
    Uses quantile regression to predict P50, P90, P99 latencies.
    """
    
    def __init__(self, feature_dim=64, hidden_dim=128):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # Quantile prediction heads
        self.p50_head = nn.Linear(hidden_dim, 1)
        self.p90_head = nn.Linear(hidden_dim, 1)
        self.p99_head = nn.Linear(hidden_dim, 1)
        
        # Mean and variance for calibration
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive variance
        )
    
    def forward(self, features):
        """
        Predict latency quantiles.
        
        Args:
            features: Transfer features [batch, feature_dim]
            
        Returns:
            predictions: Dict with 'p50', 'p90', 'p99', 'mean', 'std'
        """
        x = self.shared(features)
        
        predictions = {
            'p50': self.p50_head(x).squeeze(-1),
            'p90': self.p90_head(x).squeeze(-1),
            'p99': self.p99_head(x).squeeze(-1),
            'mean': self.mean_head(x).squeeze(-1),
            'std': torch.sqrt(self.var_head(x)).squeeze(-1)
        }
        
        return predictions
    
    def loss(self, predictions, targets, quantiles=[0.5, 0.9, 0.99]):
        """
        Quantile loss for training.
        
        Args:
            predictions: Model predictions
            targets: Actual latencies
            quantiles: Quantile levels
            
        Returns:
            total_loss: Combined loss
        """
        losses = []
        
        # Quantile losses
        for q, pred_key in zip(quantiles, ['p50', 'p90', 'p99']):
            pred = predictions[pred_key]
            errors = targets - pred
            losses.append(
                torch.max(q * errors, (q - 1) * errors).mean()
            )
        
        # Gaussian NLL for mean/variance
        mean_pred = predictions['mean']
        std_pred = predictions['std']
        nll = 0.5 * (
            torch.log(2 * np.pi * std_pred ** 2) +
            ((targets - mean_pred) ** 2) / (std_pred ** 2)
        ).mean()
        losses.append(nll)
        
        return sum(losses)


# Feature extraction for latency prediction
def extract_latency_features(transfer, system_state):
    """Extract features for latency prediction."""
    features = []
    
    # Transfer characteristics
    features.append(np.log1p(transfer.size_gb))  # Log size
    features.append(transfer.priority / 10.0)
    features.append(len(transfer.dependencies))
    
    # Path characteristics
    if transfer.path:
        features.append(len(transfer.path))  # Hops
        features.append(min(link.capacity for link in transfer.path))  # Bottleneck
        features.append(sum(link.latency_ms for link in transfer.path))  # Base latency
        features.append(max(link.utilization for link in transfer.path))  # Max util
    else:
        features.extend([0, 0, 0, 0])
    
    # System state
    features.append(system_state.utilization)
    features.append(system_state.num_active_flows)
    features.append(system_state.queue_depth)
    features.append(system_state.bandwidth_variance)
    
    # Historical statistics
    if transfer.op_type in system_state.historical_stats:
        hist = system_state.historical_stats[transfer.op_type]
        features.append(hist.mean_latency_ms)
        features.append(hist.std_latency_ms)
        features.append(hist.p99_latency_ms)
    else:
        features.extend([0, 0, 0])
    
    # Pad to feature_dim
    while len(features) < 64:
        features.append(0.0)
    
    return torch.tensor(features[:64], dtype=torch.float32)
```

Continuing in next message with training infrastructure...

### 5. Flow Size Estimator

**Purpose:** Estimate transfer size for unknown operations

**Architecture:** Ensemble of decision trees + neural network

```python
class FlowSizeEstimator:
    """Estimate flow size from operation metadata."""
    
    def __init__(self):
        # Random forest for robustness
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        
        # Neural network for complex patterns
        self.nn_model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive output
        )
        
        # Ensemble weight
        self.ensemble_weight = 0.7  # 70% RF, 30% NN
    
    def predict(self, operation):
        """Predict flow size in GB."""
        features = self._extract_features(operation)
        
        # Random forest prediction
        rf_pred = self.rf_model.predict(features.numpy().reshape(1, -1))[0]
        
        # Neural network prediction
        with torch.no_grad():
            nn_pred = self.nn_model(features).item()
        
        # Ensemble
        final_pred = (
            self.ensemble_weight * rf_pred +
            (1 - self.ensemble_weight) * nn_pred
        )
        
        return max(0.001, final_pred)  # At least 1MB
    
    def _extract_features(self, operation):
        """Extract features from operation."""
        features = []
        
        # Operation type (hashed)
        features.append(hash(operation.op_type) % 1000 / 1000.0)
        
        # Tensor shapes
        if operation.tensor_shapes:
            for shape in operation.tensor_shapes[:3]:  # Max 3 tensors
                features.append(np.log1p(np.prod(shape)))
            # Pad if fewer than 3
            while len(features) < 1 + 3:
                features.append(0.0)
        else:
            features.extend([0.0] * 3)
        
        # Data type size
        dtype_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.int32: 4,
            torch.int64: 8
        }.get(operation.dtype, 4)
        features.append(dtype_size)
        
        # Device types
        features.append(1.0 if operation.src_device.type == 'cuda' else 0.0)
        features.append(1.0 if operation.dst_device.type == 'cuda' else 0.0)
        
        # Pad to 32
        while len(features) < 32:
            features.append(0.0)
        
        return torch.tensor(features[:32], dtype=torch.float32)
```

---

## Training Infrastructure

### Offline Training Pipeline

```python
class OfflineTrainer:
    """Offline training for all ML models."""
    
    def __init__(self, config):
        self.config = config
        self.models = {
            'bandwidth': BandwidthPredictor(),
            'congestion': CongestionPredictor(),
            'routing': RoutingOptimizer(),
            'latency': LatencyPredictor(),
            'flow_size': FlowSizeEstimator()
        }
        
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=0.001)
            for name, model in self.models.items()
            if isinstance(model, nn.Module)
        }
    
    def train_all(self, dataset_path, num_epochs=100):
        """Train all models on historical data."""
        print("Loading training data...")
        data = self.load_data(dataset_path)
        
        # Train each model
        for model_name in self.models.keys():
            print(f"\nTraining {model_name} model...")
            self.train_model(
                model_name,
                data[model_name],
                num_epochs=num_epochs
            )
        
        # Save models
        self.save_all_models()
    
    def train_model(self, model_name, data, num_epochs):
        """Train a specific model."""
        model = self.models[model_name]
        
        if model_name in ['congestion', 'flow_size']:
            # Sklearn models
            self._train_sklearn_model(model, data)
        else:
            # PyTorch models
            self._train_pytorch_model(
                model,
                self.optimizers[model_name],
                data,
                num_epochs
            )
    
    def _train_pytorch_model(self, model, optimizer, data, num_epochs):
        """Train PyTorch model."""
        train_loader = DataLoader(
            data['train'],
            batch_size=256,
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            data['val'],
            batch_size=256,
            shuffle=False
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(model, BandwidthPredictor):
                    features, history, targets = batch
                    predictions, _ = model(features, history)
                    loss = F.mse_loss(predictions, targets)
                
                elif isinstance(model, LatencyPredictor):
                    features, targets = batch
                    predictions = model(features)
                    loss = model.loss(predictions, targets)
                
                elif isinstance(model, RoutingOptimizer):
                    graphs, paths, targets = batch
                    predictions = torch.stack([
                        model(g, p) for g, p in zip(graphs, paths)
                    ])
                    loss = F.binary_cross_entropy(predictions, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Similar to training but no backprop
                    # ... compute validation loss ...
                    pass
            
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(model, f"best_{model.__class__.__name__}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
    
    def _train_sklearn_model(self, model, data):
        """Train sklearn model."""
        X_train, y_train = data['train']
        X_val, y_val = data['val']
        
        model.train(X_train, y_train, X_val, y_val)
    
    def load_data(self, dataset_path):
        """Load training data from disk."""
        # Load preprocessed datasets
        data = {}
        
        for model_name in self.models.keys():
            data_file = os.path.join(dataset_path, f"{model_name}_data.pkl")
            with open(data_file, 'rb') as f:
                data[model_name] = pickle.load(f)
        
        return data
    
    def save_all_models(self):
        """Save all trained models."""
        save_dir = self.config.get('model_save_dir', './models')
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            save_path = os.path.join(save_dir, f"{name}_model.pt")
            
            if isinstance(model, nn.Module):
                torch.save(model.state_dict(), save_path)
            else:
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
            
            print(f"Saved {name} model to {save_path}")
```

### Data Collection

```python
class TrainingDataCollector:
    """Collect training data from live system."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.data_buffers = {
            'bandwidth': [],
            'congestion': [],
            'routing': [],
            'latency': [],
            'flow_size': []
        }
        
        self.buffer_size = 10000
    
    def record_transfer(self, transfer, system_state, outcome):
        """Record a completed transfer for training."""
        
        # Bandwidth predictor data
        self.data_buffers['bandwidth'].append({
            'timestamp': transfer.start_time,
            'features': extract_bandwidth_features(system_state, []),
            'actual_bandwidth': outcome.actual_bandwidth_gbps,
            'system_state': system_state
        })
        
        # Congestion predictor data
        self.data_buffers['congestion'].append({
            'timestamp': transfer.start_time,
            'features': extract_congestion_features(system_state),
            'congestion_occurred': outcome.congestion_detected
        })
        
        # Routing data
        if transfer.path:
            self.data_buffers['routing'].append({
                'timestamp': transfer.start_time,
                'graph': system_state.topology,
                'path': transfer.path,
                'performance': outcome.throughput / outcome.expected_throughput
            })
        
        # Latency predictor data
        self.data_buffers['latency'].append({
            'timestamp': transfer.start_time,
            'features': extract_latency_features(transfer, system_state),
            'actual_latency': outcome.actual_latency_ms,
            'p50_latency': outcome.p50_latency_ms,
            'p90_latency': outcome.p90_latency_ms,
            'p99_latency': outcome.p99_latency_ms
        })
        
        # Flow size data
        if transfer.actual_size_gb:
            self.data_buffers['flow_size'].append({
                'timestamp': transfer.start_time,
                'operation': transfer.operation,
                'actual_size': transfer.actual_size_gb
            })
        
        # Flush buffers if full
        for model_name, buffer in self.data_buffers.items():
            if len(buffer) >= self.buffer_size:
                self._flush_buffer(model_name)
    
    def _flush_buffer(self, model_name):
        """Write buffer to disk and clear."""
        buffer = self.data_buffers[model_name]
        
        if not buffer:
            return
        
        timestamp = int(time.time())
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(buffer, f)
        
        print(f"Flushed {len(buffer)} samples to {filepath}")
        buffer.clear()
    
    def flush_all(self):
        """Flush all buffers."""
        for model_name in self.data_buffers.keys():
            self._flush_buffer(model_name)
```

---

## Online Learning

### Incremental Model Updates

```python
class OnlineLearner:
    """Online learning for continuous model improvement."""
    
    def __init__(self, model, learning_rate=0.0001, update_interval=100):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        
        self.update_interval = update_interval
        self.sample_buffer = deque(maxlen=1000)
        self.samples_since_update = 0
    
    def add_sample(self, features, target):
        """Add a new training sample."""
        self.sample_buffer.append((features, target))
        self.samples_since_update += 1
        
        # Update if enough samples accumulated
        if self.samples_since_update >= self.update_interval:
            self.update()
    
    def update(self):
        """Perform online model update."""
        if len(self.sample_buffer) < 10:
            return  # Need minimum samples
        
        # Convert buffer to batch
        features = torch.stack([s[0] for s in self.sample_buffer])
        targets = torch.tensor([s[1] for s in self.sample_buffer])
        
        # Single gradient step
        self.optimizer.zero_grad()
        
        if isinstance(self.model, BandwidthPredictor):
            # Need to handle sequence data differently
            pass
        elif isinstance(self.model, LatencyPredictor):
            predictions = self.model(features)
            loss = self.model.loss(predictions, targets)
        else:
            predictions = self.model(features).squeeze()
            loss = F.mse_loss(predictions, targets)
        
        loss.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        self.samples_since_update = 0
        
        print(f"Online update: loss = {loss.item():.4f}")


class AdaptiveLearningRate:
    """Adjust learning rate based on performance."""
    
    def __init__(self, initial_lr=0.001, min_lr=0.00001, max_lr=0.01):
        self.lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        self.performance_history = deque(maxlen=100)
    
    def update(self, performance_metric):
        """Adjust learning rate based on performance."""
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) < 10:
            return self.lr
        
        # Compute trend
        recent = list(self.performance_history)[-10:]
        trend = np.polyfit(range(len(recent)), recent, deg=1)[0]
        
        if trend > 0:
            # Performance improving, keep or increase LR
            self.lr = min(self.lr * 1.05, self.max_lr)
        else:
            # Performance degrading, decrease LR
            self.lr = max(self.lr * 0.95, self.min_lr)
        
        return self.lr
```

---

## Feature Engineering

### Feature Importance Analysis

```python
class FeatureImportanceAnalyzer:
    """Analyze and rank feature importance."""
    
    def __init__(self):
        self.feature_importances = {}
    
    def analyze_tree_model(self, model, feature_names):
        """Analyze feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            self.feature_importances = {
                name: imp
                for name, imp in zip(feature_names, importances)
            }
            
            # Sort by importance
            sorted_features = sorted(
                self.feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return sorted_features
        
        return []
    
    def analyze_neural_network(self, model, dataset):
        """Analyze feature importance using gradient-based methods."""
        importances = defaultdict(float)
        
        model.eval()
        for features, targets in dataset:
            features.requires_grad_(True)
            
            # Forward pass
            outputs = model(features)
            loss = F.mse_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradient magnitudes
            if features.grad is not None:
                grad_magnitudes = torch.abs(features.grad).mean(dim=0)
                for i, mag in enumerate(grad_magnitudes):
                    importances[f'feature_{i}'] += mag.item()
        
        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v/total for k, v in importances.items()}
        
        return sorted(importances.items(), key=lambda x: x[1], reverse=True)
```

---

## Performance Considerations

### Inference Optimization

```python
class ModelInferenceOptimizer:
    """Optimize models for fast inference."""
    
    @staticmethod
    def quantize_model(model):
        """Quantize model to INT8 for faster inference."""
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8
        )
        return quantized_model
    
    @staticmethod
    def export_to_onnx(model, example_input, output_path):
        """Export model to ONNX for optimized inference."""
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    @staticmethod
    def compile_with_torchscript(model, example_input):
        """Compile model with TorchScript for faster inference."""
        traced_model = torch.jit.trace(model, example_input)
        traced_model = torch.jit.optimize_for_inference(traced_model)
        return traced_model


# Usage example
model = LatencyPredictor()
# ... train model ...

# Optimize for inference
optimizer = ModelInferenceOptimizer()

# Method 1: Quantization
quantized_model = optimizer.quantize_model(model)

# Method 2: TorchScript
example_input = torch.randn(1, 64)
compiled_model = optimizer.compile_with_torchscript(model, example_input)

# Method 3: ONNX
optimizer.export_to_onnx(model, example_input, "latency_model.onnx")
```

### Batched Inference

```python
class BatchedInferenceEngine:
    """Batch multiple inference requests for efficiency."""
    
    def __init__(self, model, max_batch_size=32, max_wait_ms=1.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        self.pending_requests = []
        self.lock = threading.Lock()
        
        # Start background thread
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
    
    def predict_async(self, features, callback):
        """Submit async prediction request."""
        with self.lock:
            self.pending_requests.append((features, callback, time.time()))
    
    def _inference_loop(self):
        """Background inference loop."""
        while self.running:
            time.sleep(0.001)  # 1ms
            
            with self.lock:
                if not self.pending_requests:
                    continue
                
                # Check if should process batch
                oldest_time = self.pending_requests[0][2]
                should_process = (
                    len(self.pending_requests) >= self.max_batch_size or
                    (time.time() - oldest_time) * 1000 >= self.max_wait_ms
                )
                
                if not should_process:
                    continue
                
                # Get batch
                batch_size = min(len(self.pending_requests), self.max_batch_size)
                batch = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]
            
            # Process batch
            features = torch.stack([req[0] for req in batch])
            
            with torch.no_grad():
                predictions = self.model(features)
            
            # Call callbacks
            for (_, callback, _), pred in zip(batch, predictions):
                callback(pred)
```

---

## Model Versioning and A/B Testing

```python
class ModelVersionManager:
    """Manage multiple model versions for A/B testing."""
    
    def __init__(self):
        self.models = {}  # version -> model
        self.active_versions = {}  # model_name -> version
        self.version_stats = defaultdict(lambda: {
            'predictions': 0,
            'total_error': 0.0,
            'samples': []
        })
    
    def register_model(self, model_name, version, model):
        """Register a new model version."""
        key = (model_name, version)
        self.models[key] = model
    
    def set_active_version(self, model_name, version):
        """Set active version for a model."""
        self.active_versions[model_name] = version
    
    def predict(self, model_name, features, ab_test_ratio=0.1):
        """
        Predict with active model, occasionally test new version.
        
        Args:
            model_name: Name of model
            features: Input features
            ab_test_ratio: Probability of using test version
            
        Returns:
            prediction, version_used
        """
        active_version = self.active_versions.get(model_name)
        
        # Randomly select version for A/B testing
        if random.random() < ab_test_ratio:
            # Use a random version for testing
            available_versions = [
                v for (n, v) in self.models.keys() if n == model_name
            ]
            if len(available_versions) > 1:
                test_version = random.choice(
                    [v for v in available_versions if v != active_version]
                )
                version = test_version
            else:
                version = active_version
        else:
            version = active_version
        
        # Get model and predict
        model = self.models[(model_name, version)]
        with torch.no_grad():
            prediction = model(features)
        
        self.version_stats[version]['predictions'] += 1
        
        return prediction, version
    
    def record_outcome(self, version, actual_value, predicted_value):
        """Record actual outcome for version."""
        stats = self.version_stats[version]
        error = abs(actual_value - predicted_value)
        stats['total_error'] += error
        stats['samples'].append((actual_value, predicted_value))
    
    def get_version_performance(self, version):
        """Get performance metrics for a version."""
        stats = self.version_stats[version]
        
        if stats['predictions'] == 0:
            return {}
        
        mae = stats['total_error'] / stats['predictions']
        
        if stats['samples']:
            actuals = [s[0] for s in stats['samples']]
            preds = [s[1] for s in stats['samples']]
            rmse = np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2))
        else:
            rmse = 0.0
        
        return {
            'predictions': stats['predictions'],
            'mae': mae,
            'rmse': rmse
        }
```

---

## Summary

This document covers:

1. **Model Architectures:** Six specialized ML models for different aspects of flow control
2. **Training Infrastructure:** Offline training pipeline and data collection
3. **Online Learning:** Incremental updates and adaptive learning rates
4. **Feature Engineering:** Feature extraction and importance analysis
5. **Performance Optimization:** Quantization, batching, and compilation
6. **Model Management:** Versioning and A/B testing

**Key Performance Metrics:**
- Inference latency: < 0.2ms per prediction
- Model size: < 10MB total
- Training data: Collect from live system
- Update frequency: Every 100-1000 samples
- A/B test ratio: 10% traffic for new models

Next: See `04_COMPONENT_INTERFACES.md` for detailed API specifications.

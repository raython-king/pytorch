# ML-Based Scheduling System for PyTorch IR Graphs

## Executive Summary

This document outlines a comprehensive machine learning-based scheduling system for PyTorch Inductor IR graphs. The system leverages Graph Neural Networks (GNNs), Transformer models, and Reinforcement Learning to optimize fusion decisions, kernel scheduling, and memory planning. The design emphasizes modularity, incremental deployment, and backward compatibility with existing heuristic-based approaches.

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PyTorch Inductor Pipeline                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    IR Graph Construction                         │
│  (torch/_inductor/graph.py, ir.py)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ML Scheduler Orchestrator                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Feature Extraction Layer                                │   │
│  │  • Node Features    • Edge Features    • Graph Features  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ML Model Ensemble                                       │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │   │
│  │  │  GNN   │  │Transf. │  │  RL    │  │ Hybrid │        │   │
│  │  │ Model  │  │ Model  │  │ Agent  │  │ Model  │        │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Decision Fusion & Heuristic Fallback                    │   │
│  │  • Ensemble Predictions  • Confidence Scoring            │   │
│  │  • Heuristic Override    • Safety Checks                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Traditional Scheduler (Modified)                    │
│  • Fusion Decisions    • Memory Planning    • Code Generation   │
│  (torch/_inductor/scheduler.py)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Code Generation                              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Hierarchy

```
torch/_inductor/ml_scheduler/
├── __init__.py
├── orchestrator.py          # Main ML scheduler orchestrator
├── models/
│   ├── __init__.py
│   ├── gnn_model.py         # Graph Neural Network models
│   ├── transformer_model.py # Transformer-based models
│   ├── rl_agent.py          # Reinforcement Learning agent
│   ├── hybrid_model.py      # Hybrid GNN+Transformer
│   └── ensemble.py          # Ensemble predictor
├── features/
│   ├── __init__.py
│   ├── node_features.py     # Node feature extraction
│   ├── edge_features.py     # Edge feature extraction
│   ├── graph_features.py    # Global graph features
│   └── feature_cache.py     # Feature caching system
├── training/
│   ├── __init__.py
│   ├── data_collector.py    # Trace collection from scheduler
│   ├── dataset.py           # PyTorch Geometric dataset
│   ├── trainer.py           # Training orchestrator
│   ├── supervised.py        # Supervised learning trainer
│   ├── rl_trainer.py        # RL training loop
│   └── curriculum.py        # Curriculum learning strategy
├── inference/
│   ├── __init__.py
│   ├── predictor.py         # Model inference
│   ├── confidence.py        # Confidence scoring
│   └── fallback.py          # Heuristic fallback logic
├── utils/
│   ├── __init__.py
│   ├── graph_utils.py       # Graph manipulation utilities
│   ├── metrics.py           # Performance metrics
│   └── visualization.py     # Visualization tools
└── config.py                # Configuration management
```

---

## 2. Model Architecture Options

### 2.1 Graph Neural Network (GNN) Architecture

#### 2.1.1 Node-Level Fusion Predictor

**Purpose**: Predict whether two nodes should be fused based on local and global graph structure.

**Architecture**:
```python
class FusionGNN(nn.Module):
    """
    GNN for predicting fusion decisions between pairs of nodes.
    
    Architecture:
    - Node encoder: MLP(node_features) -> node_embedding
    - Edge encoder: MLP(edge_features) -> edge_embedding
    - Message passing: 3-5 layers of GraphConv/GATv2
    - Pairwise decoder: MLP(concat(node_i, node_j, context)) -> fusion_score
    """
    
    def __init__(self, node_feat_dim=64, edge_feat_dim=32, 
                 hidden_dim=128, num_layers=4):
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.conv_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_feat_dim)
            for _ in range(num_layers)
        ])
        
        # Pairwise fusion predictor
        self.fusion_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, hidden_dim),  # 32 = context features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Binary fusion decision
            nn.Sigmoid()
        )
```

**Key Features**:
- Graph Attention Networks (GAT) for adaptive message passing
- Edge features capture data dependencies and memory access patterns
- Pairwise decoder predicts fusion compatibility
- Context vector includes device type, memory budget, current fusion stage

#### 2.1.2 Graph-Level Scheduling GNN

**Purpose**: Predict optimal execution order for the entire graph.

**Architecture**:
```python
class SchedulingGNN(nn.Module):
    """
    GNN for graph-level scheduling decisions.
    
    Output: 
    - Priority scores for each node
    - Partition assignments
    - Memory planning decisions
    """
    
    def __init__(self, hidden_dim=128):
        # Similar message passing as FusionGNN
        self.message_passing = MessagePassingStack(...)
        
        # Multiple prediction heads
        self.priority_head = nn.Linear(hidden_dim, 1)  # Execution priority
        self.partition_head = nn.Linear(hidden_dim, 16)  # Partition ID
        self.memory_head = nn.Linear(hidden_dim, 3)  # Memory planning
```

### 2.2 Transformer-Based Architecture

#### 2.2.1 Sequence-Based Scheduler

**Purpose**: Treat scheduling as a sequence-to-sequence problem.

**Architecture**:
```python
class SchedulerTransformer(nn.Module):
    """
    Transformer model for sequential scheduling decisions.
    
    Input: Linearized IR graph (topological order)
    Output: Sequence of scheduling decisions
    """
    
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        # Token embedding for IR nodes
        self.node_embedding = nn.Embedding(
            num_op_types=200,  # Different op types
            embedding_dim=d_model
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Decision heads
        self.fusion_head = nn.Linear(d_model, 2)  # Fuse with next or not
        self.order_head = nn.Linear(d_model, 1)   # Relative priority
```

**Key Features**:
- Self-attention captures long-range dependencies
- Learns to model dataflow and control dependencies
- Can handle variable-length graphs
- Efficient for inference on GPU

#### 2.2.2 Graph Transformer

**Purpose**: Combine GNN message passing with Transformer attention.

**Architecture**:
```python
class GraphTransformer(nn.Module):
    """
    Hybrid architecture combining graph structure with self-attention.
    """
    
    def __init__(self, hidden_dim=128, num_layers=4):
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=8,
                use_graph_structure=True  # Bias attention with edges
            )
            for _ in range(num_layers)
        ])
```

### 2.3 Reinforcement Learning Agent

#### 2.3.1 Policy Network

**Purpose**: Learn scheduling policy through interaction with compilation environment.

**Architecture**:
```python
class SchedulerRLAgent(nn.Module):
    """
    RL agent for sequential scheduling decisions.
    
    State: Current partial schedule + graph representation
    Action: Which node to schedule next, fusion decisions
    Reward: Runtime performance, compilation time, memory usage
    """
    
    def __init__(self):
        # State encoder (GNN or Transformer)
        self.state_encoder = FusionGNN(...)
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, max_actions)  # Action logits
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # State value
        )
    
    def select_action(self, state, mask=None):
        """Sample action from policy with masking for invalid actions."""
        logits = self.policy_net(state)
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        return Categorical(logits=logits).sample()
```

**RL Algorithm**: Proximal Policy Optimization (PPO)
- More stable than vanilla policy gradients
- Efficient sample reuse
- Good for discrete action spaces

#### 2.3.2 Reward Function Design

```python
def compute_reward(schedule_result, baseline_result):
    """
    Multi-objective reward function.
    
    Combines:
    - Runtime speedup (primary)
    - Compilation time overhead (penalty)
    - Memory efficiency
    - Numerical stability
    """
    # Normalized speedup
    speedup = baseline_result.runtime / schedule_result.runtime
    runtime_reward = (speedup - 1.0) * 10.0  # Scale to reasonable range
    
    # Compilation time penalty (should be small)
    compile_penalty = -0.1 * (schedule_result.compile_time / baseline_result.compile_time - 1.0)
    
    # Memory penalty for exceeding budget
    memory_penalty = 0.0
    if schedule_result.peak_memory > memory_budget:
        memory_penalty = -5.0 * (schedule_result.peak_memory / memory_budget - 1.0)
    
    # Combine rewards
    total_reward = runtime_reward + compile_penalty + memory_penalty
    
    return total_reward
```

### 2.4 Hybrid Architecture

**Purpose**: Combine strengths of GNN and Transformer for robust predictions.

```python
class HybridScheduler(nn.Module):
    """
    Hybrid model combining GNN for local structure and Transformer for global patterns.
    """
    
    def __init__(self):
        # GNN for local graph structure
        self.gnn_encoder = FusionGNN(...)
        
        # Transformer for global patterns
        self.transformer_encoder = SchedulerTransformer(...)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(gnn_dim + transformer_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, graph_data):
        # GNN path: Process graph structure
        gnn_features = self.gnn_encoder(
            graph_data.x, 
            graph_data.edge_index,
            graph_data.edge_attr
        )
        
        # Transformer path: Process as sequence
        sequence_features = self.transformer_encoder(
            graph_data.sequence_representation
        )
        
        # Combine both representations
        combined = torch.cat([gnn_features, sequence_features], dim=-1)
        return self.fusion_layer(combined)
```

---

## 3. Feature Engineering

### 3.1 Node Features

**Dimensions**: 64-dimensional vector per node

```python
class NodeFeatureExtractor:
    """Extract features from BaseSchedulerNode."""
    
    def extract_features(self, node: BaseSchedulerNode) -> torch.Tensor:
        """
        Extract comprehensive node features.
        
        Feature Categories:
        1. Operation Type (one-hot or embedding)
        2. Computational Cost
        3. Memory Access Pattern
        4. Data Dependencies
        5. Device Properties
        """
        features = []
        
        # 1. Operation type (20 dims - embedding)
        op_type = self._get_op_type_embedding(node)
        features.append(op_type)
        
        # 2. Computational features (10 dims)
        features.extend([
            math.log1p(node.estimate_flops() or 0),  # Log-scale FLOPS
            self._get_arithmetic_intensity(node),     # FLOPS/byte ratio
            self._get_parallelism_degree(node),       # Inherent parallelism
            self._is_reduction(node),                 # Boolean
            self._is_pointwise(node),                 # Boolean
            self._is_broadcast(node),                 # Boolean
            len(node.get_outputs()),                  # Number of outputs
            self._get_output_reuse(node),             # How many consumers
            self._get_compute_type(node),             # FP32/FP16/INT8/etc
            self._get_reduction_hint(node),           # Reduction type
        ])
        
        # 3. Memory access features (15 dims)
        features.extend([
            math.log1p(self._get_memory_read_bytes(node)),
            math.log1p(self._get_memory_write_bytes(node)),
            self._get_memory_reuse_factor(node),
            self._is_contiguous_access(node),
            self._get_stride_pattern(node),           # Regular/irregular
            self._get_cache_locality(node),           # Estimated cache hit rate
            *self._get_tensor_shape_features(node, max_dims=5),  # Shape statistics
            self._get_memory_overlap_score(node),     # Overlaps with other ops
            self._is_inplace_operation(node),
            self._get_temporary_buffer_size(node),
            self._get_num_unique_reads(node),         # Unique input tensors
        ])
        
        # 4. Dependency features (10 dims)
        features.extend([
            len(node.unmet_dependencies),
            len(node.read_writes.reads),
            len(node.read_writes.writes),
            self._get_dependency_depth(node),         # Critical path depth
            self._get_dependency_width(node),         # Max parallelism
            self._has_cyclic_dependency(node),
            self._is_on_critical_path(node),
            self._get_fusion_group_id(node),
            len(node.users),                          # Number of users
            self._get_earliest_start_time(node),      # Scheduling constraint
        ])
        
        # 5. Device-specific features (9 dims)
        device = node.get_device()
        features.extend([
            self._get_device_type_id(device),         # CPU/CUDA/XPU
            self._get_device_compute_capability(device),
            self._get_device_memory_bandwidth(device),
            self._get_device_peak_flops(device),
            self._is_device_available(device),
            self._get_current_device_utilization(device),
            self._get_available_memory(device),
            self._supports_tensor_cores(device),
            self._get_warp_size(device),
        ])
        
        return torch.tensor(features, dtype=torch.float32)
```

### 3.2 Edge Features

**Dimensions**: 32-dimensional vector per edge

```python
class EdgeFeatureExtractor:
    """Extract features from data dependencies between nodes."""
    
    def extract_features(
        self, 
        src_node: BaseSchedulerNode,
        dst_node: BaseSchedulerNode,
        dependency: Dep
    ) -> torch.Tensor:
        """
        Extract edge features representing data flow.
        
        Feature Categories:
        1. Dependency Type
        2. Data Transfer Cost
        3. Synchronization Cost
        4. Fusion Compatibility
        """
        features = []
        
        # 1. Dependency type (8 dims)
        features.extend([
            isinstance(dependency, MemoryDep),
            isinstance(dependency, WeakDep),
            isinstance(dependency, StarDep),
            self._is_read_dependency(dependency),
            self._is_write_dependency(dependency),
            self._is_read_write_dependency(dependency),
            self._is_control_dependency(dependency),
            self._is_anti_dependency(dependency),      # Write-after-read
        ])
        
        # 2. Data transfer features (12 dims)
        features.extend([
            math.log1p(dependency.numbytes_hint()),    # Data size
            self._get_transfer_latency(src_node, dst_node),
            self._get_bandwidth_utilization(src_node, dst_node),
            self._requires_device_sync(src_node, dst_node),
            self._is_same_device(src_node, dst_node),
            self._is_contiguous_transfer(dependency),
            self._get_tensor_overlap_ratio(dependency),  # Reused data
            self._get_memory_bank_conflict(dependency),
            self._is_broadcast_transfer(dependency),
            self._get_stride_mismatch(dependency),     # Layout mismatch
            self._is_cross_partition(src_node, dst_node),
            self._get_data_locality_score(dependency),
        ])
        
        # 3. Fusion compatibility (8 dims)
        features.extend([
            self._has_compatible_iteration_space(src_node, dst_node),
            self._has_compatible_memory_layout(src_node, dst_node),
            self._has_compatible_device(src_node, dst_node),
            self._fusion_would_increase_register_pressure(src_node, dst_node),
            self._fusion_would_save_memory_traffic(src_node, dst_node),
            self._get_loop_order_compatibility(src_node, dst_node),
            self._get_reduction_compatibility(src_node, dst_node),
            self._would_create_cyclic_dependency(src_node, dst_node),
        ])
        
        # 4. Scheduling constraints (4 dims)
        features.extend([
            self._get_dependency_slack(src_node, dst_node),  # Scheduling flexibility
            self._is_producer_consumer_pair(src_node, dst_node),
            self._get_priority_difference(src_node, dst_node),
            self._should_schedule_together(src_node, dst_node),
        ])
        
        return torch.tensor(features, dtype=torch.float32)
```

### 3.3 Global Graph Features

**Dimensions**: 32-dimensional vector per graph

```python
class GraphFeatureExtractor:
    """Extract features from entire IR graph."""
    
    def extract_features(self, nodes: list[BaseSchedulerNode]) -> torch.Tensor:
        """
        Extract graph-level features.
        
        Feature Categories:
        1. Graph Statistics
        2. Resource Constraints
        3. Workload Characteristics
        4. Hardware Context
        """
        features = []
        
        # 1. Graph structure (10 dims)
        features.extend([
            len(nodes),                                # Number of nodes
            self._get_num_edges(nodes),
            self._get_graph_diameter(nodes),           # Longest path
            self._get_average_degree(nodes),
            self._get_clustering_coefficient(nodes),
            self._get_num_connected_components(nodes),
            self._get_graph_density(nodes),
            self._get_num_fusion_opportunities(nodes),
            self._get_parallelism_factor(nodes),       # Available parallelism
            self._get_critical_path_length(nodes),
        ])
        
        # 2. Computational characteristics (8 dims)
        features.extend([
            math.log1p(sum(n.estimate_flops() or 0 for n in nodes)),
            math.log1p(self._estimate_total_memory(nodes)),
            self._get_compute_intensity_distribution(nodes),  # Variance
            self._get_op_type_diversity(nodes),        # Entropy
            self._get_reduction_ratio(nodes),          # % reduction ops
            self._get_pointwise_ratio(nodes),          # % pointwise ops
            self._get_matmul_ratio(nodes),             # % matmul ops
            self._get_memory_boundedness(nodes),       # Compute vs memory bound
        ])
        
        # 3. Resource constraints (8 dims)
        features.extend([
            self._get_device_memory_budget(),
            self._get_device_compute_budget(),
            self._get_current_memory_pressure(),
            self._get_register_pressure(),
            self._get_num_available_devices(),
            self._get_batch_size(),                    # If known
            self._is_inference_mode(),                 # vs training
            self._get_precision_mode(),                # FP32/FP16/mixed
        ])
        
        # 4. Context features (6 dims)
        features.extend([
            self._get_graph_id(),                      # For tracking
            self._get_compilation_stage(),             # Pre/post fusion
            self._is_dynamic_shapes(),
            self._has_control_flow(),
            self._has_custom_ops(),
            self._get_cudagraph_compatibility(),
        ])
        
        return torch.tensor(features, dtype=torch.float32)
```

### 3.4 Feature Normalization and Preprocessing

```python
class FeatureNormalizer:
    """Normalize features for stable training."""
    
    def __init__(self):
        self.node_mean = None
        self.node_std = None
        self.edge_mean = None
        self.edge_std = None
        self.graph_mean = None
        self.graph_std = None
    
    def fit(self, dataset):
        """Compute normalization statistics from training data."""
        # Collect all features
        all_node_features = []
        all_edge_features = []
        all_graph_features = []
        
        for graph_data in dataset:
            all_node_features.append(graph_data.x)
            all_edge_features.append(graph_data.edge_attr)
            all_graph_features.append(graph_data.global_features)
        
        # Compute statistics
        self.node_mean = torch.cat(all_node_features).mean(dim=0)
        self.node_std = torch.cat(all_node_features).std(dim=0) + 1e-6
        
        self.edge_mean = torch.cat(all_edge_features).mean(dim=0)
        self.edge_std = torch.cat(all_edge_features).std(dim=0) + 1e-6
        
        self.graph_mean = torch.stack(all_graph_features).mean(dim=0)
        self.graph_std = torch.stack(all_graph_features).std(dim=0) + 1e-6
    
    def normalize(self, graph_data):
        """Apply normalization."""
        graph_data.x = (graph_data.x - self.node_mean) / self.node_std
        graph_data.edge_attr = (graph_data.edge_attr - self.edge_mean) / self.edge_std
        graph_data.global_features = (graph_data.global_features - self.graph_mean) / self.graph_std
        return graph_data
```

---

## 4. Training Strategy

### 4.1 Supervised Learning from Existing Heuristics

**Goal**: Bootstrap ML models using existing scheduler decisions.

```python
class SupervisedTrainer:
    """
    Train models using expert demonstrations from existing scheduler.
    """
    
    def collect_training_data(self, num_graphs=10000):
        """
        Run existing scheduler and record decisions.
        
        For each compilation:
        1. Record IR graph structure
        2. Record fusion decisions (labels)
        3. Record scheduling order
        4. Record performance metrics
        """
        dataset = []
        
        for model in self.benchmark_models:
            # Compile with existing scheduler
            with scheduler_instrumentation_enabled():
                compiled = torch.compile(model)
                compiled(sample_input)
                
                # Extract decisions made by scheduler
                trace = get_scheduler_trace()
                
                # Convert to training example
                example = {
                    'graph': self.extract_graph_structure(trace),
                    'fusion_decisions': trace.fusion_decisions,
                    'schedule_order': trace.schedule_order,
                    'runtime': trace.runtime,
                }
                dataset.append(example)
        
        return dataset
    
    def train(self, model, dataset, epochs=100):
        """Train model using supervised learning."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            for batch in dataset:
                # Forward pass
                predictions = model(batch['graph'])
                
                # Compute loss
                fusion_loss = F.binary_cross_entropy(
                    predictions['fusion_scores'],
                    batch['fusion_decisions']
                )
                
                order_loss = self.ranking_loss(
                    predictions['order_scores'],
                    batch['schedule_order']
                )
                
                total_loss = fusion_loss + 0.5 * order_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
```

### 4.2 Reinforcement Learning with Performance Rewards

**Goal**: Optimize directly for runtime performance.

```python
class RLTrainer:
    """
    Train RL agent through interaction with compilation environment.
    """
    
    def __init__(self):
        self.agent = SchedulerRLAgent()
        self.ppo_optimizer = PPO(
            self.agent,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01
        )
        
    def train(self, num_iterations=1000):
        """
        RL training loop.
        
        For each iteration:
        1. Generate episode by scheduling a graph
        2. Compile and benchmark the result
        3. Compute reward based on performance
        4. Update policy using PPO
        """
        for iteration in range(num_iterations):
            # Collect trajectories
            trajectories = []
            for _ in range(self.episodes_per_iteration):
                trajectory = self.collect_episode()
                trajectories.append(trajectory)
            
            # Update policy
            self.ppo_optimizer.step(trajectories)
            
            # Logging
            avg_reward = np.mean([t.total_reward for t in trajectories])
            print(f"Iteration {iteration}: Avg Reward = {avg_reward:.3f}")
    
    def collect_episode(self):
        """
        Generate one scheduling episode.
        
        Returns trajectory of (state, action, reward, next_state).
        """
        # Get a random IR graph
        graph = self.sample_graph()
        
        # Initialize state
        state = SchedulingState(
            graph=graph,
            scheduled_nodes=[],
            available_nodes=graph.get_roots()
        )
        
        trajectory = []
        
        # Rollout policy
        while not state.is_terminal():
            # Get available actions (which nodes can be scheduled)
            action_mask = state.get_action_mask()
            
            # Agent selects action
            action = self.agent.select_action(state, mask=action_mask)
            
            # Take action (schedule node, make fusion decision)
            next_state, intermediate_reward = state.step(action)
            
            # Store transition
            trajectory.append((state, action, intermediate_reward, next_state))
            
            state = next_state
        
        # Compile and benchmark final schedule
        final_schedule = state.get_schedule()
        runtime = self.benchmark_schedule(final_schedule)
        baseline_runtime = self.get_baseline_runtime(graph)
        
        # Compute final reward
        final_reward = self.compute_reward(runtime, baseline_runtime)
        
        # Propagate final reward to all steps
        for i in range(len(trajectory)):
            state, action, intermediate_reward, next_state = trajectory[i]
            total_reward = intermediate_reward + final_reward / len(trajectory)
            trajectory[i] = (state, action, total_reward, next_state)
        
        return trajectory
```

### 4.3 Imitation Learning from Expert Traces

**Goal**: Learn from high-quality manual optimizations.

```python
class ImitationLearner:
    """
    Learn from expert-optimized schedules.
    
    Combines:
    - Behavioral cloning (BC)
    - Dataset Aggregation (DAgger)
    """
    
    def train_behavioral_cloning(self, expert_traces):
        """
        Pure supervised learning from expert demonstrations.
        """
        for epoch in range(self.bc_epochs):
            for trace in expert_traces:
                # Replay expert trajectory
                for state, expert_action in trace:
                    # Predict action
                    predicted_action = self.agent(state)
                    
                    # Match expert action
                    loss = F.cross_entropy(predicted_action, expert_action)
                    
                    # Update
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
    
    def train_dagger(self, expert_policy, num_iterations=50):
        """
        Iteratively collect data and aggregate with expert corrections.
        """
        dataset = self.collect_expert_demonstrations()
        
        for iteration in range(num_iterations):
            # Train on current dataset
            self.train_on_dataset(dataset)
            
            # Collect new trajectories using learned policy
            new_trajectories = self.collect_trajectories_with_policy()
            
            # Get expert corrections for states visited by policy
            for trajectory in new_trajectories:
                for state, agent_action in trajectory:
                    # Query expert for correct action
                    expert_action = expert_policy(state)
                    
                    # Add to dataset
                    dataset.append((state, expert_action))
            
            # Mix old and new data
            dataset = self.aggregate_dataset(dataset, new_trajectories)
```

### 4.4 Curriculum Learning Strategy

**Goal**: Gradually increase task difficulty for stable learning.

```python
class CurriculumLearning:
    """
    Curriculum for training scheduler models.
    
    Stages:
    1. Small graphs (10-50 nodes)
    2. Medium graphs (50-200 nodes)
    3. Large graphs (200-1000 nodes)
    4. Complex patterns (control flow, dynamic shapes)
    """
    
    def __init__(self):
        self.current_stage = 0
        self.stages = [
            {'max_nodes': 50, 'complexity': 'simple'},
            {'max_nodes': 200, 'complexity': 'medium'},
            {'max_nodes': 1000, 'complexity': 'high'},
            {'max_nodes': float('inf'), 'complexity': 'all'},
        ]
    
    def should_advance_stage(self, metrics):
        """
        Advance to next stage if performance threshold met.
        """
        if self.current_stage >= len(self.stages) - 1:
            return False
        
        # Check if performance is good enough
        current_accuracy = metrics['accuracy']
        threshold = 0.90  # 90% accuracy before advancing
        
        if current_accuracy >= threshold:
            self.current_stage += 1
            return True
        return False
    
    def get_training_graphs(self):
        """Sample graphs appropriate for current curriculum stage."""
        stage = self.stages[self.current_stage]
        
        return self.sample_graphs(
            max_nodes=stage['max_nodes'],
            complexity=stage['complexity']
        )
```

### 4.5 Online Learning and Adaptation

**Goal**: Continuously improve from production compilations.

```python
class OnlineLearner:
    """
    Continuously learn from production workloads.
    
    Features:
    - Collect performance feedback
    - Periodic model updates
    - A/B testing for new models
    """
    
    def __init__(self):
        self.feedback_buffer = []
        self.update_interval = 1000  # Update every N compilations
        self.compilation_count = 0
    
    def record_feedback(self, graph, predictions, actual_performance):
        """
        Record feedback from a compilation.
        """
        feedback = {
            'graph': graph,
            'predictions': predictions,
            'actual_runtime': actual_performance.runtime,
            'actual_memory': actual_performance.peak_memory,
            'timestamp': time.time(),
        }
        self.feedback_buffer.append(feedback)
        
        self.compilation_count += 1
        
        # Periodic update
        if self.compilation_count % self.update_interval == 0:
            self.update_model()
    
    def update_model(self):
        """
        Fine-tune model on recent feedback.
        """
        if len(self.feedback_buffer) < self.min_samples:
            return
        
        # Sample recent experiences
        batch = random.sample(self.feedback_buffer, self.batch_size)
        
        # Fine-tune model
        for _ in range(self.fine_tune_steps):
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear old feedback
        self.feedback_buffer = self.feedback_buffer[-self.buffer_size:]
```

---

## 5. Integration Points with Existing Scheduler

### 5.1 Hooking into scheduler.py

**Primary Integration Point**: `Scheduler.fuse_nodes()`

```python
# File: torch/_inductor/scheduler.py

class Scheduler:
    def __init__(self, nodes: list[ir.Operation]) -> None:
        # ... existing code ...
        
        # Initialize ML scheduler if enabled
        if config.ml_scheduler_enabled:
            from torch._inductor.ml_scheduler import MLSchedulerOrchestrator
            self.ml_orchestrator = MLSchedulerOrchestrator()
        else:
            self.ml_orchestrator = None
    
    def fuse_nodes(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Modified fusion logic with ML scheduler integration.
        """
        # Option 1: Full ML-based fusion (experimental)
        if config.ml_scheduler_mode == "full":
            return self._ml_fuse_nodes(nodes)
        
        # Option 2: Hybrid mode (ML + heuristics)
        elif config.ml_scheduler_mode == "hybrid":
            return self._hybrid_fuse_nodes(nodes)
        
        # Option 3: Traditional heuristics (default)
        else:
            return self._heuristic_fuse_nodes(nodes)
    
    def _ml_fuse_nodes(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Pure ML-based fusion.
        """
        if self.ml_orchestrator is None:
            # Fallback to heuristics if ML not available
            return self._heuristic_fuse_nodes(nodes)
        
        try:
            # Let ML scheduler make all fusion decisions
            fusion_plan = self.ml_orchestrator.predict_fusion_plan(
                nodes=nodes,
                device=self.current_device
            )
            
            # Apply fusion plan
            fused_nodes = self._apply_fusion_plan(nodes, fusion_plan)
            
            # Log decision for feedback
            if config.ml_scheduler_collect_feedback:
                self.ml_orchestrator.record_prediction(
                    nodes=nodes,
                    fusion_plan=fusion_plan,
                    result_nodes=fused_nodes
                )
            
            return fused_nodes
            
        except Exception as e:
            # Safety fallback
            log.warning(f"ML scheduler failed: {e}, falling back to heuristics")
            return self._heuristic_fuse_nodes(nodes)
    
    def _hybrid_fuse_nodes(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Hybrid approach: ML makes recommendations, heuristics validate.
        """
        # Get ML recommendations with confidence scores
        recommendations = self.ml_orchestrator.predict_fusion_candidates(
            nodes=nodes,
            return_confidence=True
        )
        
        # Filter by confidence threshold
        high_confidence_fusions = [
            (node1, node2) 
            for node1, node2, confidence in recommendations
            if confidence > config.ml_scheduler_confidence_threshold
        ]
        
        # Apply high-confidence fusions directly
        nodes = self._apply_fusions(nodes, high_confidence_fusions)
        
        # Use heuristics for remaining decisions
        nodes = self._heuristic_fuse_nodes(nodes)
        
        return nodes
    
    def _heuristic_fuse_nodes(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Original heuristic-based fusion (unchanged).
        """
        # Existing implementation
        return self.fuse_nodes_once(nodes, is_reorder_round=False)
```

### 5.2 Overriding can_fuse Decisions

```python
# File: torch/_inductor/ml_scheduler/fusion_override.py

class FusionOverride:
    """
    Override can_fuse decisions with ML predictions.
    """
    
    @staticmethod
    def can_fuse_ml(
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        ml_orchestrator: MLSchedulerOrchestrator
    ) -> tuple[bool, float]:
        """
        ML-based fusion decision with confidence score.
        
        Returns:
            (should_fuse, confidence)
        """
        # Extract features for this pair
        features = ml_orchestrator.extract_pairwise_features(node1, node2)
        
        # Get ML prediction
        fusion_score = ml_orchestrator.predict_fusion_score(features)
        
        # Convert to binary decision
        should_fuse = fusion_score > 0.5
        confidence = abs(fusion_score - 0.5) * 2  # Map [0, 1] to confidence
        
        return should_fuse, confidence
    
    @staticmethod
    def can_fuse_with_fallback(
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        ml_orchestrator: Optional[MLSchedulerOrchestrator]
    ) -> bool:
        """
        Try ML first, fallback to heuristics if low confidence.
        """
        if ml_orchestrator is not None and config.ml_scheduler_enabled:
            should_fuse, confidence = FusionOverride.can_fuse_ml(
                node1, node2, ml_orchestrator
            )
            
            # Use ML decision if confident
            if confidence > config.ml_scheduler_confidence_threshold:
                return should_fuse
        
        # Fallback to original heuristic
        return FusedSchedulerNode.can_fuse(node1, node2)
```

### 5.3 Integration with Code Cache

```python
# File: torch/_inductor/ml_scheduler/cache_integration.py

class MLSchedulerCache:
    """
    Cache ML predictions and compiled results.
    
    Benefits:
    - Avoid recomputing features for similar graphs
    - Reuse compiled kernels
    - Track performance feedback
    """
    
    def __init__(self):
        self.graph_cache = {}  # Graph signature -> predictions
        self.kernel_cache = {}  # Schedule -> compiled kernel
        self.performance_cache = {}  # Schedule -> runtime metrics
    
    def get_cached_prediction(self, graph_signature: str):
        """Retrieve cached prediction if available."""
        return self.graph_cache.get(graph_signature)
    
    def cache_prediction(self, graph_signature: str, predictions):
        """Cache predictions for future use."""
        self.graph_cache[graph_signature] = predictions
    
    def get_graph_signature(self, nodes: list[BaseSchedulerNode]) -> str:
        """
        Compute unique signature for a graph.
        
        Factors:
        - Node types and counts
        - Dependency structure
        - Tensor shapes (if static)
        """
        node_types = tuple(sorted([n.__class__.__name__ for n in nodes]))
        num_nodes = len(nodes)
        num_edges = sum(len(n.unmet_dependencies) for n in nodes)
        
        # Hash key
        key = f"{num_nodes}_{num_edges}_{hash(node_types)}"
        
        return key
```

### 5.4 Backward Compatibility and Safety

```python
# File: torch/_inductor/ml_scheduler/safety.py

class SafetyChecker:
    """
    Ensure ML predictions don't break correctness.
    """
    
    @staticmethod
    def validate_fusion_plan(
        fusion_plan: FusionPlan,
        nodes: list[BaseSchedulerNode]
    ) -> tuple[bool, str]:
        """
        Validate fusion plan for correctness.
        
        Checks:
        1. No cyclic dependencies created
        2. Device compatibility maintained
        3. Memory budget not exceeded
        4. Reduction semantics preserved
        """
        errors = []
        
        # 1. Check for cycles
        if SafetyChecker._creates_cycle(fusion_plan, nodes):
            errors.append("Fusion plan creates cyclic dependency")
        
        # 2. Check device compatibility
        if not SafetyChecker._check_device_compatibility(fusion_plan, nodes):
            errors.append("Fusion across incompatible devices")
        
        # 3. Check memory budget
        estimated_memory = SafetyChecker._estimate_memory_usage(fusion_plan, nodes)
        if estimated_memory > SafetyChecker._get_memory_budget():
            errors.append(f"Memory budget exceeded: {estimated_memory}")
        
        # 4. Check reduction semantics
        if not SafetyChecker._check_reduction_semantics(fusion_plan, nodes):
            errors.append("Reduction semantics violated")
        
        is_valid = len(errors) == 0
        error_msg = "; ".join(errors) if errors else ""
        
        return is_valid, error_msg
    
    @staticmethod
    def fallback_on_error(
        ml_result,
        heuristic_result,
        error: Exception
    ):
        """
        Gracefully fallback to heuristics on ML failure.
        """
        log.warning(
            f"ML scheduler encountered error: {error}. "
            f"Falling back to heuristic scheduler."
        )
        
        # Record failure for monitoring
        if config.ml_scheduler_collect_feedback:
            record_ml_failure(error)
        
        return heuristic_result
```

---

## 6. Evaluation Metrics

### 6.1 Performance Metrics

```python
class PerformanceEvaluator:
    """
    Comprehensive evaluation of ML scheduler performance.
    """
    
    def evaluate(self, test_set, baseline_scheduler, ml_scheduler):
        """
        Evaluate ML scheduler against baseline.
        
        Returns comprehensive metrics dict.
        """
        metrics = {
            'runtime': [],
            'compilation_time': [],
            'memory_usage': [],
            'numerical_accuracy': [],
        }
        
        for graph in test_set:
            # Baseline compilation
            baseline_result = self.compile_with_scheduler(
                graph, baseline_scheduler
            )
            
            # ML compilation
            ml_result = self.compile_with_scheduler(
                graph, ml_scheduler
            )
            
            # Collect metrics
            metrics['runtime'].append(
                ml_result.runtime / baseline_result.runtime  # Speedup
            )
            metrics['compilation_time'].append(
                ml_result.compile_time / baseline_result.compile_time
            )
            metrics['memory_usage'].append(
                ml_result.peak_memory / baseline_result.peak_memory
            )
            metrics['numerical_accuracy'].append(
                self.check_numerical_accuracy(
                    baseline_result.output,
                    ml_result.output
                )
            )
        
        # Aggregate statistics
        return {
            'mean_speedup': np.mean(metrics['runtime']),
            'geomean_speedup': gmean(metrics['runtime']),
            'p50_speedup': np.percentile(metrics['runtime'], 50),
            'p90_speedup': np.percentile(metrics['runtime'], 90),
            'p99_speedup': np.percentile(metrics['runtime'], 99),
            'compilation_overhead': np.mean(metrics['compilation_time']),
            'memory_efficiency': np.mean(metrics['memory_usage']),
            'accuracy_issues': sum(a < 0.99 for a in metrics['numerical_accuracy']),
        }
```

### 6.2 Compilation Time Overhead

```python
class CompilationProfiler:
    """
    Profile compilation time breakdown.
    """
    
    def profile_compilation(self, graph):
        """
        Measure time spent in each compilation phase.
        """
        with Timer() as total_timer:
            with Timer() as feature_timer:
                features = self.extract_features(graph)
            
            with Timer() as inference_timer:
                predictions = self.model(features)
            
            with Timer() as fusion_timer:
                fused_graph = self.apply_fusion(graph, predictions)
            
            with Timer() as codegen_timer:
                compiled_kernel = self.generate_code(fused_graph)
        
        return {
            'total_time': total_timer.elapsed,
            'feature_extraction': feature_timer.elapsed,
            'ml_inference': inference_timer.elapsed,
            'fusion_application': fusion_timer.elapsed,
            'code_generation': codegen_timer.elapsed,
            
            # Percentages
            'ml_overhead_pct': (
                (feature_timer.elapsed + inference_timer.elapsed) / 
                total_timer.elapsed * 100
            ),
        }
```

### 6.3 Runtime Performance Improvement

```python
class RuntimeBenchmark:
    """
    Benchmark runtime performance of generated code.
    """
    
    def benchmark_kernel(self, kernel, inputs, num_warmup=10, num_iterations=100):
        """
        Benchmark kernel execution time.
        """
        # Warmup
        for _ in range(num_warmup):
            kernel(*inputs)
        
        # Synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            kernel(*inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'p50_time': np.percentile(times, 50),
            'p95_time': np.percentile(times, 95),
            'p99_time': np.percentile(times, 99),
        }
```

### 6.4 Memory Efficiency

```python
class MemoryProfiler:
    """
    Profile memory usage of compiled kernels.
    """
    
    def profile_memory(self, kernel, inputs):
        """
        Measure peak memory usage and memory efficiency.
        """
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Run kernel
        kernel(*inputs)
        torch.cuda.synchronize()
        
        # Collect memory stats
        peak_memory = torch.cuda.max_memory_allocated()
        
        # Estimate theoretical minimum
        input_memory = sum(t.numel() * t.element_size() for t in inputs)
        output_memory = self._estimate_output_memory(kernel, inputs)
        theoretical_min = input_memory + output_memory
        
        return {
            'peak_memory_bytes': peak_memory,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'theoretical_min_mb': theoretical_min / 1024 / 1024,
            'memory_efficiency': theoretical_min / peak_memory if peak_memory > 0 else 0,
        }
```

### 6.5 Generalization to Unseen Graphs

```python
class GeneralizationEvaluator:
    """
    Evaluate model generalization across different graph types.
    """
    
    def evaluate_generalization(self, model):
        """
        Test generalization across multiple dimensions:
        1. Graph size (small, medium, large)
        2. Graph structure (linear, tree, DAG, complex)
        3. Op types (pointwise, reduction, matmul, mixed)
        4. Hardware (different GPUs, CPU)
        """
        test_suites = {
            'size_small': self.load_graphs(max_nodes=50),
            'size_medium': self.load_graphs(max_nodes=200),
            'size_large': self.load_graphs(max_nodes=1000),
            'structure_linear': self.load_graphs(structure='linear'),
            'structure_tree': self.load_graphs(structure='tree'),
            'structure_dag': self.load_graphs(structure='dag'),
            'ops_pointwise': self.load_graphs(op_filter='pointwise'),
            'ops_reduction': self.load_graphs(op_filter='reduction'),
            'ops_matmul': self.load_graphs(op_filter='matmul'),
        }
        
        results = {}
        for suite_name, graphs in test_suites.items():
            suite_results = []
            for graph in graphs:
                result = self.evaluate_single_graph(model, graph)
                suite_results.append(result)
            
            results[suite_name] = {
                'mean_speedup': np.mean([r['speedup'] for r in suite_results]),
                'success_rate': np.mean([r['success'] for r in suite_results]),
                'num_graphs': len(suite_results),
            }
        
        return results
```

### 6.6 Comprehensive Metric Dashboard

```python
class MetricsDashboard:
    """
    Unified dashboard for tracking all metrics.
    """
    
    def generate_report(self, evaluation_results):
        """
        Generate comprehensive evaluation report.
        """
        report = {
            'summary': {
                'geomean_speedup': evaluation_results['geomean_speedup'],
                'compilation_overhead': evaluation_results['compilation_overhead'],
                'memory_efficiency': evaluation_results['memory_efficiency'],
                'success_rate': evaluation_results['success_rate'],
            },
            
            'performance_breakdown': {
                'p50_speedup': evaluation_results['p50_speedup'],
                'p90_speedup': evaluation_results['p90_speedup'],
                'p99_speedup': evaluation_results['p99_speedup'],
                'regression_rate': evaluation_results['regression_rate'],
            },
            
            'compilation_breakdown': {
                'feature_extraction_ms': evaluation_results['feature_time'],
                'inference_ms': evaluation_results['inference_time'],
                'total_overhead_ms': evaluation_results['total_overhead'],
            },
            
            'generalization': evaluation_results['generalization_scores'],
            
            'reliability': {
                'fallback_rate': evaluation_results['fallback_rate'],
                'numerical_errors': evaluation_results['numerical_errors'],
                'correctness_rate': evaluation_results['correctness_rate'],
            },
        }
        
        return report
```

---

## 7. Deployment and Rollout Strategy

### 7.1 Phased Rollout Plan

```
Phase 1: Data Collection (Month 1-2)
├── Instrument existing scheduler
├── Collect 10K+ compilation traces
├── Build training dataset
└── Establish baseline metrics

Phase 2: Model Development (Month 3-4)
├── Train initial models
├── Offline evaluation
├── Model selection and tuning
└── Safety validation

Phase 3: Shadow Mode (Month 5)
├── Run ML scheduler in parallel
├── Compare predictions with heuristics
├── Collect discrepancy data
└── No production impact

Phase 4: Canary Deployment (Month 6)
├── Enable for 1% of compilations
├── Monitor for regressions
├── Gradual increase to 10%
└── Gather performance data

Phase 5: Full Deployment (Month 7-8)
├── Roll out to 100% with fallback
├── Continuous monitoring
├── Online learning enabled
└── Iterative improvements

Phase 6: Optimization (Month 9+)
├── Remove heuristic fallback (optional)
├── Model compression for faster inference
├── Advanced features
└── Multi-device support
```

### 7.2 Configuration Management

```python
# File: torch/_inductor/config.py

# ML Scheduler Configuration
ml_scheduler_enabled = False  # Master switch
ml_scheduler_mode = "hybrid"  # Options: "off", "shadow", "hybrid", "full"
ml_scheduler_model_path = None  # Path to trained model
ml_scheduler_confidence_threshold = 0.75  # Min confidence for using ML prediction
ml_scheduler_collect_feedback = True  # Collect performance feedback
ml_scheduler_online_learning = False  # Enable online learning
ml_scheduler_fallback_on_error = True  # Fallback to heuristics on error
ml_scheduler_cache_predictions = True  # Cache predictions for similar graphs
ml_scheduler_log_level = "INFO"  # Logging verbosity
```

---

## 8. Future Enhancements

### 8.1 Multi-Device Scheduling

Extend to handle scheduling across multiple devices (multi-GPU, CPU+GPU).

### 8.2 Dynamic Shape Support

Enhance feature extraction to handle dynamic shapes and symbolic computation.

### 8.3 Autotuning Integration

Integrate with Triton autotuner for end-to-end optimization.

### 8.4 Transfer Learning

Fine-tune models for specific workload patterns (transformers, CNNs, etc.).

### 8.5 Explainability

Add interpretability tools to understand why certain fusion decisions are made.

---

## 9. Monitoring and Observability

```python
class MLSchedulerMonitor:
    """
    Monitor ML scheduler in production.
    """
    
    def __init__(self):
        self.metrics = {
            'total_compilations': 0,
            'ml_predictions': 0,
            'heuristic_fallbacks': 0,
            'errors': 0,
            'speedup_distribution': [],
            'compilation_time_overhead': [],
        }
    
    def record_compilation(self, result):
        """Record compilation result."""
        self.metrics['total_compilations'] += 1
        
        if result.used_ml:
            self.metrics['ml_predictions'] += 1
            self.metrics['speedup_distribution'].append(result.speedup)
            self.metrics['compilation_time_overhead'].append(result.overhead)
        else:
            self.metrics['heuristic_fallbacks'] += 1
        
        if result.error:
            self.metrics['errors'] += 1
    
    def get_summary(self):
        """Get monitoring summary."""
        return {
            'ml_usage_rate': self.metrics['ml_predictions'] / max(self.metrics['total_compilations'], 1),
            'fallback_rate': self.metrics['heuristic_fallbacks'] / max(self.metrics['total_compilations'], 1),
            'error_rate': self.metrics['errors'] / max(self.metrics['total_compilations'], 1),
            'mean_speedup': np.mean(self.metrics['speedup_distribution']) if self.metrics['speedup_distribution'] else 0,
            'mean_overhead_ms': np.mean(self.metrics['compilation_time_overhead']) if self.metrics['compilation_time_overhead'] else 0,
        }
```

---

## 10. Conclusion

This design provides a comprehensive, modular framework for integrating machine learning into PyTorch's IR graph scheduling system. The key principles are:

1. **Modularity**: Each component can be developed and tested independently
2. **Safety**: Extensive fallback mechanisms ensure correctness
3. **Incremental Deployment**: Gradual rollout minimizes risk
4. **Observability**: Comprehensive monitoring and metrics
5. **Flexibility**: Support for multiple model architectures and training strategies

The system is designed to work alongside existing heuristics initially, with the option to fully replace them as confidence grows. This approach balances innovation with the reliability requirements of a production compiler.


# GPU集群通讯优化系统 - ML集成详细设计

本文档详细描述GPU集群自适应通讯优化系统中机器学习模型的架构、特征工程、训练策略和推理优化。

---

## 目录

1. [ML系统架构](#1-ml系统架构)
2. [特征工程](#2-特征工程)
3. [模型架构详解](#3-模型架构详解)
4. [训练策略](#4-训练策略)
5. [在线学习与持续优化](#5-在线学习与持续优化)
6. [推理优化](#6-推理优化)
7. [模型评估与验证](#7-模型评估与验证)

---

## 1. ML系统架构

### 1.1 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         ML Pipeline                               │
│                                                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│  │  Feature    │   │   Model     │   │  Inference  │            │
│  │ Engineering │──>│  Training   │──>│   Engine    │            │
│  └─────────────┘   └─────────────┘   └─────────────┘            │
│         ↑                  ↑                  │                   │
│         │                  │                  ↓                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│  │   Data      │   │ Experience  │   │  Decision   │            │
│  │ Collector   │   │   Replay    │   │   Output    │            │
│  └─────────────┘   └─────────────┘   └─────────────┘            │
│         ↑                  ↑                  │                   │
└─────────┼──────────────────┼──────────────────┼───────────────────┘
          │                  │                  │
          │                  │                  ↓
┌─────────┴──────────────────┴──────────────────────────────────────┐
│                   Communication System                            │
│  • Profiler  • Topology  • Scheduler  • Executor                 │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 ML模型组件

系统使用多个专门化的ML模型，每个模型负责特定的决策任务：

| 模型 | 输入 | 输出 | 任务 | 模型类型 |
|------|------|------|------|----------|
| **AlgorithmSelector** | 消息大小、拓扑、负载 | 算法选择（分类） | 选择最优集合通讯算法 | GNN + MLP |
| **RoutingOptimizer** | 拓扑图、负载、目标 | 路径（序列） | 选择最优通讯路径 | GNN + RL |
| **CompressionAdvisor** | 梯度统计、层信息、带宽 | 压缩类型（分类） | 选择压缩策略 | Transformer + MLP |
| **OverlapScheduler** | 计算图、通讯计划 | 调度方案 | 优化overlap | Seq2Seq + RL |
| **TimePredictor** | 所有上下文特征 | 通讯时间（回归） | 预测执行时间 | Ensemble |
| **CongestionPredictor** | 时间序列负载 | 未来拥塞（分类） | 预测网络拥塞 | LSTM/Transformer |

### 1.3 模型选择决策流程

```
                通讯请求
                    |
                    ↓
        ┌───────────────────────┐
        │ 特征提取与编码         │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │   TimePredictor       │ (快速估计各方案的时间)
        └───────────┬───────────┘
                    │
        ┌───────────┴────────────┐
        │  初步筛选候选方案       │ (保留top-3)
        └───────────┬────────────┘
                    │
        ┌───────────┴─────────────┐
        │   AlgorithmSelector     │ (精确选择算法)
        └───────────┬─────────────┘
                    │
        ┌───────────┴─────────────┐
        │   RoutingOptimizer      │ (选择路由)
        └───────────┬─────────────┘
                    │
        ┌───────────┴─────────────┐
        │  CompressionAdvisor     │ (选择压缩)
        └───────────┬─────────────┘
                    │
        ┌───────────┴─────────────┐
        │   OverlapScheduler      │ (优化overlap)
        └───────────┬─────────────┘
                    │
                    ↓
                执行方案
```

---

## 2. 特征工程

### 2.1 特征分类

#### 2.1.1 消息特征

```python
@dataclass
class MessageFeatures:
    """消息相关特征"""

    # 基本特征
    size_bytes: int          # 消息大小（字节）
    size_mb: float          # 消息大小（MB）
    size_log: float         # log(size_bytes)
    dtype: str              # 数据类型（'float32', 'float16', ...）
    operation: str          # 操作类型（'all_reduce', 'all_gather', ...）

    # 统计特征
    size_percentile: float  # 在历史分布中的百分位
    is_outlier: bool        # 是否异常大小

    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.size_log,
            _encode_dtype(self.dtype),
            _encode_operation(self.operation),
            self.size_percentile,
            float(self.is_outlier)
        ])
```

#### 2.1.2 拓扑特征

```python
@dataclass
class TopologyFeatures:
    """拓扑相关特征"""

    # 全局拓扑特征
    num_gpus: int
    num_nodes: int
    gpus_per_node: int

    # 层级特征
    has_nvlink: bool
    nvlink_bandwidth_gbps: float
    pcie_bandwidth_gbps: float
    ib_bandwidth_gbps: float

    # 图结构特征
    graph_diameter: int          # 图的直径（最长最短路径）
    graph_avg_degree: float      # 平均度数
    graph_clustering: float      # 聚类系数

    # 异构性特征
    bandwidth_variance: float    # 带宽方差
    latency_variance: float      # 延迟方差

    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.num_gpus,
            self.num_nodes,
            float(self.has_nvlink),
            self.nvlink_bandwidth_gbps / 1000.0,  # 归一化
            self.pcie_bandwidth_gbps / 100.0,
            self.ib_bandwidth_gbps / 100.0,
            self.graph_diameter / self.num_gpus,
            self.graph_avg_degree / self.num_gpus,
            self.graph_clustering,
            self.bandwidth_variance,
            self.latency_variance
        ])
```

#### 2.1.3 系统状态特征

```python
@dataclass
class SystemStateFeatures:
    """系统状态特征"""

    # 负载特征
    avg_bandwidth_utilization: float     # 平均带宽利用率
    max_bandwidth_utilization: float     # 最大带宽利用率
    num_congested_links: int             # 拥塞链路数

    # 拥塞模式
    congestion_distribution: np.ndarray  # 拥塞分布（各层级）
    hotspot_locations: List[Tuple[int, int]]  # 热点位置

    # 历史性能
    recent_avg_latency_ms: float
    recent_p99_latency_ms: float
    recent_throughput_gbps: float

    # 时间特征
    time_since_last_comm_ms: float
    comm_frequency: float  # 通讯频率（次/秒）

    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.avg_bandwidth_utilization,
            self.max_bandwidth_utilization,
            self.num_congested_links / 100.0,  # 归一化
            *self.congestion_distribution,
            self.recent_avg_latency_ms / 100.0,
            self.recent_p99_latency_ms / 100.0,
            self.recent_throughput_gbps / 100.0,
            self.time_since_last_comm_ms / 1000.0,
            self.comm_frequency
        ])
```

#### 2.1.4 图嵌入特征

对于拓扑图，使用图神经网络提取嵌入：

```python
class TopologyEmbedding:
    """拓扑图嵌入"""

    def __init__(self, gnn_model: torch.nn.Module):
        self.gnn_model = gnn_model

    def embed_topology(
        self,
        topology: ClusterTopology
    ) -> torch.Tensor:
        """
        将拓扑图转换为嵌入向量

        Args:
            topology: 集群拓扑

        Returns:
            [embed_dim] 拓扑嵌入向量
        """

        # 1. 构建节点特征矩阵
        node_features = []
        for gpu in topology.gpus:
            features = [
                gpu.compute_capability[0] + gpu.compute_capability[1] * 0.1,
                gpu.memory_gb / 80.0,  # 归一化到0-1
                gpu.node_id / topology.num_nodes,
                gpu.socket_id / 4.0
            ]
            node_features.append(features)

        node_features = torch.tensor(node_features, dtype=torch.float32)

        # 2. 构建边索引和边特征
        edge_index = []
        edge_features = []

        for i, j in topology.link_graph.edges():
            edge_index.append([i, j])
            edge_index.append([j, i])  # 无向图，添加反向边

            # 边特征：带宽、延迟、链路类型
            bandwidth = topology.bandwidth_matrix[i, j]
            latency = topology.latency_matrix[i, j]
            link_type = topology.get_hierarchy_level(i, j)

            edge_feat = [
                bandwidth / 1000.0,  # 归一化
                latency / 100.0,
                _encode_link_type(link_type)
            ]

            edge_features.append(edge_feat)
            edge_features.append(edge_feat)  # 反向边

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_features = torch.tensor(edge_features, dtype=torch.float32)

        # 3. GNN前向传播
        with torch.no_grad():
            node_embeddings = self.gnn_model(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features
            )

        # 4. 全局池化
        graph_embedding = torch.mean(node_embeddings, dim=0)

        return graph_embedding
```

### 2.2 特征预处理

```python
class FeaturePreprocessor:
    """特征预处理器"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def fit(self, features: List[Dict[str, Any]]):
        """
        拟合归一化器和编码器

        Args:
            features: 训练数据特征列表
        """

        # 数值特征：StandardScaler
        numerical_features = ['size_log', 'bandwidth_utilization', ...]
        for feat_name in numerical_features:
            values = [f[feat_name] for f in features]
            scaler = StandardScaler()
            scaler.fit(np.array(values).reshape(-1, 1))
            self.scalers[feat_name] = scaler

        # 类别特征：LabelEncoder
        categorical_features = ['dtype', 'operation', 'link_type', ...]
        for feat_name in categorical_features:
            values = [f[feat_name] for f in features]
            encoder = LabelEncoder()
            encoder.fit(values)
            self.encoders[feat_name] = encoder

    def transform(self, features: Dict[str, Any]) -> np.ndarray:
        """
        转换特征

        Returns:
            归一化和编码后的特征向量
        """

        transformed = []

        # 数值特征
        for feat_name in self.scalers:
            value = features[feat_name]
            scaled = self.scalers[feat_name].transform([[value]])[0][0]
            transformed.append(scaled)

        # 类别特征（one-hot编码）
        for feat_name in self.encoders:
            value = features[feat_name]
            encoded = self.encoders[feat_name].transform([value])[0]

            # one-hot
            num_classes = len(self.encoders[feat_name].classes_)
            one_hot = np.zeros(num_classes)
            one_hot[encoded] = 1.0

            transformed.extend(one_hot)

        return np.array(transformed)
```

---

## 3. 模型架构详解

### 3.1 AlgorithmSelector (GNN + MLP)

#### 架构

```python
class AlgorithmSelectorModel(torch.nn.Module):
    """算法选择模型

    输入：拓扑图 + 消息特征 + 系统状态
    输出：算法选择概率分布
    """

    def __init__(
        self,
        node_feature_dim: int = 8,
        edge_feature_dim: int = 4,
        message_feature_dim: int = 16,
        state_feature_dim: int = 20,
        gnn_hidden_dim: int = 64,
        gnn_layers: int = 3,
        mlp_hidden_dim: int = 128,
        num_algorithms: int = 5
    ):
        super().__init__()

        # GNN部分：处理拓扑图
        self.gnn_layers = torch.nn.ModuleList([
            GATConv(
                in_channels=node_feature_dim if i == 0 else gnn_hidden_dim,
                out_channels=gnn_hidden_dim,
                heads=4,
                concat=False,
                edge_dim=edge_feature_dim
            )
            for i in range(gnn_layers)
        ])

        # MLP部分：处理消息和状态特征
        self.message_encoder = torch.nn.Sequential(
            torch.nn.Linear(message_feature_dim, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        )

        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(state_feature_dim, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        )

        # 融合层
        fusion_dim = gnn_hidden_dim + 2 * mlp_hidden_dim

        self.fusion_layers = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, mlp_hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(mlp_hidden_dim * 2, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(mlp_hidden_dim, num_algorithms)
        )

    def forward(
        self,
        # 拓扑图输入
        node_features: torch.Tensor,      # [num_nodes, node_feature_dim]
        edge_index: torch.Tensor,         # [2, num_edges]
        edge_attr: torch.Tensor,          # [num_edges, edge_feature_dim]

        # 消息特征
        message_features: torch.Tensor,   # [message_feature_dim]

        # 状态特征
        state_features: torch.Tensor      # [state_feature_dim]
    ) -> torch.Tensor:
        """
        前向传播

        Returns:
            [num_algorithms] 每个算法的logits
        """

        # 1. GNN处理拓扑图
        x = node_features

        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # 全局池化
        topology_embedding = torch.mean(x, dim=0)  # [gnn_hidden_dim]

        # 2. MLP处理消息和状态特征
        message_embedding = self.message_encoder(message_features)
        state_embedding = self.state_encoder(state_features)

        # 3. 融合所有特征
        fused = torch.cat([
            topology_embedding,
            message_embedding,
            state_embedding
        ], dim=0)

        # 4. 输出层
        logits = self.fusion_layers(fused)

        return logits
```

#### 训练

```python
class AlgorithmSelectorTrainer:
    """AlgorithmSelector训练器"""

    def __init__(
        self,
        model: AlgorithmSelectorModel,
        learning_rate: float = 0.001
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> float:
        """
        单步训练

        Args:
            batch: {
                'node_features': [batch, num_nodes, node_feat_dim],
                'edge_index': [batch, 2, num_edges],
                'edge_attr': [batch, num_edges, edge_feat_dim],
                'message_features': [batch, msg_feat_dim],
                'state_features': [batch, state_feat_dim],
                'labels': [batch],  # 真实选择的算法ID
                'performance': [batch, num_algorithms]  # 各算法的实际性能
            }
        """

        self.model.train()

        # 前向传播（处理batch）
        batch_size = batch['labels'].size(0)
        all_logits = []

        for i in range(batch_size):
            logits = self.model(
                node_features=batch['node_features'][i],
                edge_index=batch['edge_index'][i],
                edge_attr=batch['edge_attr'][i],
                message_features=batch['message_features'][i],
                state_features=batch['state_features'][i]
            )
            all_logits.append(logits)

        logits = torch.stack(all_logits)  # [batch, num_algorithms]

        # 损失函数：组合分类损失和回归损失
        labels = batch['labels']
        performance = batch['performance']

        # 分类损失：预测正确的算法
        classification_loss = self.criterion(logits, labels)

        # 回归损失：预测的算法性能应该接近实际性能
        predicted_probs = F.softmax(logits, dim=-1)
        predicted_performance = (predicted_probs * performance).sum(dim=-1)
        actual_performance = performance.gather(1, labels.unsqueeze(1)).squeeze()

        regression_loss = F.mse_loss(predicted_performance, actual_performance)

        # 总损失
        loss = classification_loss + 0.5 * regression_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
```

---

### 3.2 RoutingOptimizer (GNN + RL)

#### 架构

路由优化是一个序列决策问题，使用强化学习：

**State:** 当前拓扑、已选择的路径、目标节点
**Action:** 选择下一个节点
**Reward:** 到达目标时，负的路径代价

```python
class RoutingOptimizerModel(torch.nn.Module):
    """路由优化模型（基于RL）"""

    def __init__(
        self,
        node_feature_dim: int = 8,
        edge_feature_dim: int = 4,
        hidden_dim: int = 64
    ):
        super().__init__()

        # GNN Encoder：编码拓扑
        self.gnn = GATConv(
            in_channels=node_feature_dim,
            out_channels=hidden_dim,
            heads=4,
            concat=False,
            edge_dim=edge_feature_dim
        )

        # Policy Network：给定当前节点，选择下一个节点
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),  # 当前节点 + 目标节点
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # Value Network：估计state value
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        current_node_id: int,
        target_node_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            (action_probs, value)
            action_probs: [num_nodes] 选择各节点作为下一跳的概率
            value: scalar, state value
        """

        # 1. GNN编码所有节点
        node_embeddings = self.gnn(node_features, edge_index, edge_attr)
        node_embeddings = F.relu(node_embeddings)

        # 2. 提取当前节点和目标节点的嵌入
        current_embedding = node_embeddings[current_node_id]
        target_embedding = node_embeddings[target_node_id]

        state_embedding = torch.cat([current_embedding, target_embedding], dim=0)

        # 3. Policy network：计算选择各节点的logits
        policy_features = self.policy_net(state_embedding)

        # 计算与所有节点的相似度作为logits
        logits = torch.matmul(node_embeddings, policy_features)

        # Mask掉不可达的节点（非邻居节点）
        neighbors = edge_index[1][edge_index[0] == current_node_id]
        mask = torch.ones(node_embeddings.size(0), dtype=torch.bool)
        mask[neighbors] = False
        mask[current_node_id] = False  # 不能选择自己
        mask[target_node_id] = False   # 可以选择目标（终止）

        logits[mask] = -float('inf')

        action_probs = F.softmax(logits, dim=0)

        # 4. Value network
        value = self.value_net(state_embedding)

        return action_probs, value

    def select_next_node(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        current_node: int,
        target_node: int
    ) -> int:
        """
        选择下一个节点（inference）
        """

        action_probs, _ = self.forward(
            node_features, edge_index, edge_attr, current_node, target_node
        )

        # 贪心选择
        next_node = torch.argmax(action_probs).item()

        return next_node

    def find_path(
        self,
        topology: ClusterTopology,
        src_node: int,
        dst_node: int,
        max_hops: int = 10
    ) -> List[int]:
        """
        使用RL policy找到路径
        """

        path = [src_node]
        current = src_node

        # 准备拓扑图数据
        node_features, edge_index, edge_attr = self._prepare_topology(topology)

        for _ in range(max_hops):
            if current == dst_node:
                break

            next_node = self.select_next_node(
                node_features, edge_index, edge_attr, current, dst_node
            )

            path.append(next_node)
            current = next_node

        return path
```

#### RL训练（PPO）

```python
class RoutingOptimizerTrainer:
    """路由优化器训练（PPO）"""

    def __init__(
        self,
        model: RoutingOptimizerModel,
        topology: ClusterTopology,
        learning_rate: float = 0.0003
    ):
        self.model = model
        self.topology = topology
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def collect_trajectories(
        self,
        num_episodes: int = 100
    ) -> List[Dict]:
        """
        收集轨迹数据
        """

        trajectories = []

        for _ in range(num_episodes):
            # 随机选择源和目标
            src = random.randint(0, self.topology.num_gpus - 1)
            dst = random.randint(0, self.topology.num_gpus - 1)

            if src == dst:
                continue

            # 执行一个episode
            trajectory = self.run_episode(src, dst)
            trajectories.append(trajectory)

        return trajectories

    def run_episode(
        self,
        src: int,
        dst: int
    ) -> Dict:
        """
        运行一个episode
        """

        states = []
        actions = []
        rewards = []
        log_probs = []

        current = src
        path_cost = 0.0

        # 准备拓扑
        node_features, edge_index, edge_attr = self._prepare_topology(self.topology)

        for step in range(20):  # 最多20跳
            if current == dst:
                break

            # 选择动作
            action_probs, value = self.model(
                node_features, edge_index, edge_attr, current, dst
            )

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_node = action.item()

            # 计算reward（负的链路代价）
            link_cost = self.topology.get_link_cost(current, next_node)
            reward = -link_cost

            # 记录
            states.append((current, dst))
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            # 更新状态
            current = next_node
            path_cost += link_cost

        # 如果成功到达目标，额外奖励
        if current == dst:
            rewards[-1] += 10.0
        else:
            # 失败，惩罚
            rewards[-1] -= 20.0

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs
        }

    def update_policy(
        self,
        trajectories: List[Dict],
        num_epochs: int = 10
    ):
        """
        PPO更新
        """

        # 计算advantages
        all_advantages = []
        for traj in trajectories:
            rewards = traj['rewards']
            # 计算discounted returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + 0.99 * R
                returns.insert(0, R)

            returns = torch.tensor(returns)

            # 估计values（重新前向传播）
            values = []
            for state in traj['states']:
                current, dst = state
                node_features, edge_index, edge_attr = self._prepare_topology(self.topology)
                _, value = self.model(node_features, edge_index, edge_attr, current, dst)
                values.append(value)

            values = torch.stack(values).squeeze()

            advantages = returns - values.detach()
            all_advantages.append(advantages)

        # PPO更新
        for epoch in range(num_epochs):
            for traj_idx, traj in enumerate(trajectories):
                states = traj['states']
                actions = torch.stack(traj['actions'])
                old_log_probs = torch.stack(traj['log_probs'])
                advantages = all_advantages[traj_idx]

                # 重新计算log_probs和values
                new_log_probs = []
                new_values = []

                for state in states:
                    current, dst = state
                    node_features, edge_index, edge_attr = self._prepare_topology(self.topology)
                    action_probs, value = self.model(
                        node_features, edge_index, edge_attr, current, dst
                    )

                    dist = torch.distributions.Categorical(action_probs)
                    log_prob = dist.log_prob(actions[len(new_log_probs)])

                    new_log_probs.append(log_prob)
                    new_values.append(value)

                new_log_probs = torch.stack(new_log_probs)
                new_values = torch.stack(new_values).squeeze()

                # PPO clip
                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, advantages + new_values.detach())

                loss = policy_loss + 0.5 * value_loss

                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

---

### 3.3 CompressionAdvisor (Transformer + MLP)

```python
class CompressionAdvisorModel(torch.nn.Module):
    """压缩建议模型

    输入：梯度历史序列 + 层信息 + 系统状态
    输出：压缩类型概率分布
    """

    def __init__(
        self,
        gradient_feature_dim: int = 16,
        layer_feature_dim: int = 8,
        state_feature_dim: int = 20,
        sequence_length: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        num_compression_types: int = 6
    ):
        super().__init__()

        # Transformer处理梯度历史序列
        self.gradient_embedding = torch.nn.Linear(gradient_feature_dim, d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )

        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # MLP处理层信息和状态
        self.layer_encoder = torch.nn.Sequential(
            torch.nn.Linear(layer_feature_dim, d_model),
            torch.nn.ReLU()
        )

        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(state_feature_dim, d_model),
            torch.nn.ReLU()
        )

        # 融合和输出
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(d_model * 3, d_model * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(d_model * 2, d_model),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(d_model, num_compression_types)
        )

    def forward(
        self,
        gradient_sequence: torch.Tensor,  # [seq_len, gradient_feature_dim]
        layer_features: torch.Tensor,     # [layer_feature_dim]
        state_features: torch.Tensor      # [state_feature_dim]
    ) -> torch.Tensor:
        """
        前向传播

        Returns:
            [num_compression_types] logits
        """

        # 1. Transformer处理梯度序列
        gradient_embedded = self.gradient_embedding(gradient_sequence)
        gradient_encoded = self.transformer(gradient_embedded)

        # 取最后一个时间步
        gradient_repr = gradient_encoded[-1]

        # 2. 编码层信息和状态
        layer_repr = self.layer_encoder(layer_features)
        state_repr = self.state_encoder(state_features)

        # 3. 融合
        fused = torch.cat([gradient_repr, layer_repr, state_repr], dim=0)

        # 4. 输出
        logits = self.output_layer(fused)

        return logits
```

---

### 3.4 TimePredictor (Ensemble)

时间预测使用集成模型：

```python
class TimePredictorEnsemble(torch.nn.Module):
    """时间预测集成模型"""

    def __init__(self):
        super().__init__()

        # 模型1：MLP
        self.mlp_model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

        # 模型2：GBT (使用sklearn，离线训练)
        self.gbt_model = None  # GradientBoostingRegressor

        # 模型3：物理模型（基于性能公式）
        # T = alpha * S / B + beta * L
        self.physics_params = torch.nn.Parameter(
            torch.tensor([1.0, 1.0])  # alpha, beta
        )

    def forward(
        self,
        features: torch.Tensor,
        message_size: float,
        bandwidth: float,
        latency: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            (predicted_time, uncertainty)
        """

        # 预测1：MLP
        pred1 = self.mlp_model(features)

        # 预测2：GBT (离线)
        if self.gbt_model is not None:
            pred2 = self.gbt_model.predict(features.detach().cpu().numpy())
            pred2 = torch.tensor(pred2, device=features.device)
        else:
            pred2 = pred1  # fallback

        # 预测3：物理模型
        alpha, beta = self.physics_params
        pred3 = alpha * message_size / bandwidth + beta * latency

        # 集成（加权平均）
        weights = torch.tensor([0.4, 0.3, 0.3], device=features.device)
        predictions = torch.stack([pred1.squeeze(), pred2, pred3])

        final_pred = (predictions * weights).sum()

        # 不确定性（标准差）
        uncertainty = predictions.std()

        return final_pred, uncertainty
```

---

## 4. 训练策略

### 4.1 离线训练

#### 数据收集

```python
class OfflineDataCollector:
    """离线数据收集器"""

    def __init__(self, topology: ClusterTopology):
        self.topology = topology
        self.dataset = []

    def collect_synthetic_data(
        self,
        num_samples: int = 10000
    ):
        """
        生成合成数据

        策略：
        1. 随机生成消息大小、操作类型
        2. 尝试所有算法
        3. 使用性能模型或实际执行测量时间
        4. 记录特征和标签
        """

        for _ in range(num_samples):
            # 随机参数
            message_size = random.choice([
                1024,                 # 1KB
                1024 * 1024,         # 1MB
                10 * 1024 * 1024,    # 10MB
                100 * 1024 * 1024,   # 100MB
                1024 * 1024 * 1024   # 1GB
            ])

            operation = random.choice(['all_reduce', 'all_gather', 'reduce_scatter'])
            world_size = self.topology.num_gpus

            # 提取特征
            message_features = extract_message_features(message_size, operation)
            topology_features = extract_topology_features(self.topology)
            state_features = extract_system_state_features()

            features = {
                'message': message_features,
                'topology': topology_features,
                'state': state_features
            }

            # 尝试所有算法并测量性能
            algorithms = ['ring', 'tree', 'double_binary_tree', 'hierarchical']
            performance = {}

            for algo in algorithms:
                # 使用性能模型估计时间
                estimated_time = estimate_algorithm_time(
                    algo, message_size, world_size, self.topology
                )

                performance[algo] = estimated_time

            # 最优算法
            best_algo = min(performance, key=performance.get)
            best_algo_id = algorithms.index(best_algo)

            # 记录样本
            self.dataset.append({
                'features': features,
                'label': best_algo_id,
                'performance': [performance[algo] for algo in algorithms]
            })

    def collect_real_data(
        self,
        workload: Callable
    ):
        """
        收集真实数据

        Args:
            workload: 真实工作负载函数
        """

        # 运行工作负载，记录所有通讯
        profiler = CommunicationProfiler()

        with profiler.enable():
            workload()

        # 提取样本
        for comm_event in profiler.get_all_events():
            features = extract_features_from_event(comm_event, self.topology)

            # 如果有多种算法的性能数据，记录
            if hasattr(comm_event, 'algorithm_comparison'):
                self.dataset.append({
                    'features': features,
                    'label': comm_event.algorithm_used,
                    'performance': comm_event.algorithm_comparison
                })
```

#### 训练流程

```python
class OfflineTrainer:
    """离线训练流程"""

    def __init__(self, models: Dict[str, torch.nn.Module]):
        self.models = models

    def train_all_models(
        self,
        dataset: List[Dict],
        num_epochs: int = 100,
        batch_size: int = 32
    ):
        """训练所有模型"""

        # 分割数据集
        train_data, val_data = train_test_split(dataset, test_size=0.2)

        # 训练AlgorithmSelector
        algo_trainer = AlgorithmSelectorTrainer(self.models['algorithm_selector'])
        algo_trainer.train(train_data, val_data, num_epochs, batch_size)

        # 训练CompressionAdvisor
        comp_trainer = CompressionAdvisorTrainer(self.models['compression_advisor'])
        comp_trainer.train(train_data, val_data, num_epochs, batch_size)

        # 训练TimePredictor
        time_trainer = TimePredictorTrainer(self.models['time_predictor'])
        time_trainer.train(train_data, val_data, num_epochs, batch_size)

        # RL模型（RoutingOptimizer）需要交互式训练
        routing_trainer = RoutingOptimizerTrainer(self.models['routing_optimizer'])
        routing_trainer.train_offline(num_episodes=10000)
```

---

### 4.2 迁移学习

```python
class TransferLearning:
    """迁移学习：在新拓扑上快速适应"""

    def __init__(
        self,
        pretrained_model: torch.nn.Module,
        new_topology: ClusterTopology
    ):
        self.pretrained_model = pretrained_model
        self.new_topology = new_topology

    def adapt_to_new_topology(
        self,
        num_samples: int = 1000,
        num_epochs: int = 20
    ):
        """
        适应新拓扑

        策略：
        1. 冻结大部分预训练层
        2. 仅微调最后几层
        3. 使用少量新拓扑的数据
        """

        # 1. 冻结底层
        for name, param in self.pretrained_model.named_parameters():
            if 'gnn' in name or 'embedding' in name:
                param.requires_grad = False

        # 2. 收集少量新拓扑数据
        collector = OfflineDataCollector(self.new_topology)
        collector.collect_synthetic_data(num_samples)

        # 3. 微调
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.pretrained_model.parameters()),
            lr=0.0001  # 小学习率
        )

        for epoch in range(num_epochs):
            for batch in get_batches(collector.dataset, batch_size=32):
                loss = self.pretrained_model.compute_loss(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

---

## 5. 在线学习与持续优化

### 5.1 经验回放

```python
class ExperienceReplayBuffer:
    """经验回放缓冲区"""

    def __init__(
        self,
        capacity: int = 100000,
        priority_alpha: float = 0.6
    ):
        self.capacity = capacity
        self.priority_alpha = priority_alpha

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(
        self,
        experience: Dict[str, Any],
        priority: Optional[float] = None
    ):
        """
        添加经验

        Args:
            experience: {
                'features': ...,
                'action': ...,
                'reward': ...,
                'next_state': ...
            }
            priority: 优先级（越高越重要，越可能被采样）
        """

        self.buffer.append(experience)

        if priority is None:
            # 默认优先级：基于reward的绝对值
            priority = abs(experience.get('reward', 0.0)) + 1e-5

        self.priorities.append(priority)

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4  # 重要性采样权重
    ) -> List[Dict[str, Any]]:
        """
        优先级采样
        """

        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()

        # 采样
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probs
        )

        samples = [self.buffer[i] for i in indices]

        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        for sample, weight in zip(samples, weights):
            sample['importance_weight'] = weight

        return samples

    def update_priority(
        self,
        index: int,
        priority: float
    ):
        """更新优先级（用于TD error）"""
        self.priorities[index] = priority
```

---

### 5.2 在线更新策略

```python
class OnlineLearningEngine:
    """在线学习引擎"""

    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        replay_buffer: ExperienceReplayBuffer,
        update_interval: int = 100,
        batch_size: int = 32
    ):
        self.models = models
        self.replay_buffer = replay_buffer
        self.update_interval = update_interval
        self.batch_size = batch_size

        self.step_count = 0

        # 优化器
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=0.0001)
            for name, model in models.items()
        }

    def observe(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any]
    ):
        """
        观察一个经验

        Args:
            state: 决策时的系统状态
            action: 采取的动作（算法选择、压缩配置等）
            reward: 奖励（负的通讯时间）
            next_state: 下一个状态
        """

        # 添加到replay buffer
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        }

        # 计算优先级（基于TD error）
        priority = self.compute_priority(experience)

        self.replay_buffer.add(experience, priority)

        self.step_count += 1

        # 定期更新模型
        if self.step_count % self.update_interval == 0:
            self.update_models()

    def compute_priority(
        self,
        experience: Dict[str, Any]
    ) -> float:
        """
        计算经验优先级（基于TD error）

        高TD error意味着模型预测不准，应该优先学习
        """

        # 使用TimePredictor估计时间
        state = experience['state']
        reward = experience['reward']  # 实际时间（负值）

        # 预测时间
        predicted_time, _ = self.models['time_predictor'].predict(state)

        # TD error
        td_error = abs(predicted_time - abs(reward))

        return td_error

    def update_models(self):
        """更新模型"""

        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        # 采样batch
        batch = self.replay_buffer.sample(self.batch_size)

        # 更新AlgorithmSelector
        self.update_algorithm_selector(batch)

        # 更新CompressionAdvisor
        self.update_compression_advisor(batch)

        # 更新TimePredictor
        self.update_time_predictor(batch)

    def update_algorithm_selector(
        self,
        batch: List[Dict[str, Any]]
    ):
        """更新算法选择模型"""

        model = self.models['algorithm_selector']
        optimizer = self.optimizers['algorithm_selector']

        # 准备数据
        # ... (类似离线训练)

        # 计算损失（考虑重要性采样权重）
        loss = 0.0
        for exp in batch:
            weight = exp.get('importance_weight', 1.0)
            # 计算该样本的损失
            sample_loss = model.compute_loss(exp)
            loss += weight * sample_loss

        loss /= len(batch)

        # 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 6. 推理优化

### 6.1 模型量化

```python
class ModelQuantizer:
    """模型量化：减少推理开销"""

    @staticmethod
    def quantize_model(
        model: torch.nn.Module,
        calibration_data: List[torch.Tensor]
    ) -> torch.nn.Module:
        """
        动态量化模型（INT8）
        """

        # PyTorch动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},  # 量化的层类型
            dtype=torch.qint8
        )

        return quantized_model
```

---

### 6.2 模型蒸馏

```python
class ModelDistillation:
    """模型蒸馏：大模型 -> 小模型"""

    def __init__(
        self,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        temperature: float = 3.0
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature

    def distill(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_epochs: int = 50
    ):
        """
        蒸馏训练
        """

        optimizer = torch.optim.Adam(self.student.parameters(), lr=0.001)

        self.teacher.eval()

        for epoch in range(num_epochs):
            for batch in data_loader:
                inputs, targets = batch

                # Teacher预测（soft labels）
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                    teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

                # Student预测
                student_logits = self.student(inputs)
                student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

                # 蒸馏损失（KL divergence）
                distill_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction='batchmean'
                ) * (self.temperature ** 2)

                # 硬标签损失
                hard_loss = F.cross_entropy(student_logits, targets)

                # 总损失
                loss = 0.7 * distill_loss + 0.3 * hard_loss

                # 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

---

### 6.3 推理加速

```python
class FastInference:
    """快速推理引擎"""

    def __init__(
        self,
        model: torch.nn.Module,
        use_torchscript: bool = True,
        use_cuda_graph: bool = True
    ):
        self.model = model

        # TorchScript编译
        if use_torchscript:
            self.model = torch.jit.script(model)

        # CUDA Graph（消除kernel launch overhead）
        if use_cuda_graph and torch.cuda.is_available():
            self.use_cuda_graph = True
            self.static_input = None
            self.static_output = None
            self.graph = None
        else:
            self.use_cuda_graph = False

    def warmup_cuda_graph(
        self,
        sample_input: Dict[str, torch.Tensor]
    ):
        """预热CUDA Graph"""

        if not self.use_cuda_graph:
            return

        # 创建static tensors
        self.static_input = {
            k: v.clone() for k, v in sample_input.items()
        }

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for _ in range(3):
                self.model(**self.static_input)

        torch.cuda.current_stream().wait_stream(s)

        # Capture graph
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            self.static_output = self.model(**self.static_input)

    def infer(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """快速推理"""

        if self.use_cuda_graph and self.graph is not None:
            # 使用CUDA Graph
            # 复制输入到static tensors
            for k, v in inputs.items():
                self.static_input[k].copy_(v)

            # Replay graph
            self.graph.replay()

            return self.static_output.clone()

        else:
            # 普通推理
            with torch.no_grad():
                return self.model(**inputs)
```

---

## 7. 模型评估与验证

### 7.1 离线评估

```python
class OfflineEvaluator:
    """离线评估器"""

    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        test_dataset: List[Dict]
    ):
        self.models = models
        self.test_dataset = test_dataset

    def evaluate_algorithm_selector(self) -> Dict[str, float]:
        """评估算法选择模型"""

        model = self.models['algorithm_selector']
        model.eval()

        correct = 0
        total = 0

        regret_sum = 0.0  # 累积遗憾（选择的算法 vs 最优算法的时间差）

        for sample in self.test_dataset:
            features = sample['features']
            true_label = sample['label']
            performance = sample['performance']

            # 预测
            with torch.no_grad():
                logits = model(**features)
                predicted_label = torch.argmax(logits).item()

            # 准确率
            if predicted_label == true_label:
                correct += 1
            total += 1

            # Regret
            predicted_time = performance[predicted_label]
            optimal_time = min(performance)
            regret = predicted_time - optimal_time
            regret_sum += regret

        accuracy = correct / total
        avg_regret = regret_sum / total
        regret_ratio = avg_regret / np.mean([min(s['performance']) for s in self.test_dataset])

        return {
            'accuracy': accuracy,
            'avg_regret_ms': avg_regret,
            'regret_ratio': regret_ratio
        }

    def evaluate_time_predictor(self) -> Dict[str, float]:
        """评估时间预测模型"""

        model = self.models['time_predictor']
        model.eval()

        errors = []

        for sample in self.test_dataset:
            features = sample['features']
            true_time = sample['actual_time']

            # 预测
            with torch.no_grad():
                predicted_time, uncertainty = model.predict(**features)

            # 误差
            error = abs(predicted_time - true_time)
            errors.append(error)

        mape = np.mean(np.array(errors) / np.array([s['actual_time'] for s in self.test_dataset])) * 100

        return {
            'mae': np.mean(errors),
            'mape': mape,
            'p50_error': np.percentile(errors, 50),
            'p95_error': np.percentile(errors, 95)
        }
```

---

### 7.2 在线A/B测试

```python
class ABTestFramework:
    """A/B测试框架"""

    def __init__(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        split_ratio: float = 0.5
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio

        self.stats_a = {'count': 0, 'total_time': 0.0}
        self.stats_b = {'count': 0, 'total_time': 0.0}

    def select_model(self) -> Tuple[torch.nn.Module, str]:
        """随机选择模型"""

        if random.random() < self.split_ratio:
            return self.model_a, 'A'
        else:
            return self.model_b, 'B'

    def record_result(
        self,
        model_id: str,
        communication_time: float
    ):
        """记录结果"""

        if model_id == 'A':
            self.stats_a['count'] += 1
            self.stats_a['total_time'] += communication_time
        else:
            self.stats_b['count'] += 1
            self.stats_b['total_time'] += communication_time

    def get_results(self) -> Dict[str, Any]:
        """获取结果"""

        avg_time_a = self.stats_a['total_time'] / max(1, self.stats_a['count'])
        avg_time_b = self.stats_b['total_time'] / max(1, self.stats_b['count'])

        improvement = (avg_time_a - avg_time_b) / avg_time_a * 100

        # 统计显著性检验（t-test）
        # ... (需要收集所有样本的时间)

        return {
            'model_a_avg_time': avg_time_a,
            'model_b_avg_time': avg_time_b,
            'improvement_%': improvement,
            'model_a_count': self.stats_a['count'],
            'model_b_count': self.stats_b['count']
        }
```

---

## 总结

本文档详细描述了GPU集群通讯优化系统的ML集成方案，包括：

1. **ML系统架构**：多模型协作的决策系统
2. **特征工程**：消息、拓扑、状态、图嵌入等多维特征
3. **模型架构**：GNN、Transformer、RL等多种模型
4. **训练策略**：离线训练、迁移学习、在线学习
5. **推理优化**：量化、蒸馏、CUDA Graph等加速技术
6. **评估验证**：离线评估和在线A/B测试

下一步将在04_INTERFACES_AND_API.md中详细描述系统的接口和API设计。

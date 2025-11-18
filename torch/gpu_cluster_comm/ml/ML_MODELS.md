# ML模型详解

本文档详细介绍GPU集群通讯优化中使用的所有机器学习模型。

## 目录

1. [架构概览](#架构概览)
2. [特征工程](#特征工程)
3. [性能预测模型](#性能预测模型)
4. [算法选择模型](#算法选择模型)
5. [强化学习模型](#强化学习模型)
6. [模型集成](#模型集成)

---

## 架构概览

整体ML架构包含以下组件：

```
特征提取器 → ML预测模型 → 策略选择器 → 执行 → 在线学习
     ↑                                        ↓
     └──────────── 反馈循环 ─────────────────┘
```

### 核心模型

| 模型 | 类型 | 输入 | 输出 | 用途 |
|------|------|------|------|------|
| CommunicationTimePredictor | 回归 | 120维特征 | 预测时间(ms) | 时间预测 |
| BandwidthPredictor | 时序 | 历史带宽+系统特征 | 未来10步带宽 | 带宽预测 |
| CongestionPredictor | GNN | 拓扑图 | 每条链路拥塞概率 | 拥塞预测 |
| AlgorithmSelectorModel | 分类 | 120维特征 | 6个算法得分 | 算法选择 |
| ParameterOptimizer | 多任务 | 特征+算法 | 参数配置 | 参数优化 |
| CommunicationRLAgent | RL | 状态147维 | 动作7维 | 端到端优化 |

---

## 特征工程

### 特征维度分配

**总特征维度：147维**

1. **消息特征 (32维)**
   - 基本特征: size_log, dtype, contiguous, element_count (8维)
   - 数据统计: mean, std, min, max, median, q25, q75, sparsity, skewness, kurtosis (10维)
   - 内存布局: stride patterns, memory efficiency, cache friendliness (8维)
   - 形状特征: num_dims, dim_0 ~ dim_4 (6维)

2. **拓扑特征 (48维)**
   - 基本拓扑: num_ranks, topology_type, pcie_ratio, nvlink_ratio (8维)
   - 带宽统计: min, max, avg, std, median, p25, p75, p90, cv, entropy (10维)
   - 延迟统计: min, max, avg, std, median, p50, p90, p99 (8维)
   - 图特征: diameter, avg_path_length, clustering, centrality等 (12维)
   - 连接度特征: degree stats, connectivity (10维)

3. **工作负载特征 (40维)**
   - 频率特征: avg_freq, std_interval等 (6维)
   - 大小分布: mean, std, min, max, percentiles等 (10维)
   - 时序模式: autocorrelation, trend, seasonality等 (12维)
   - Burstiness: burst_coef, peak_ratio等 (6维)
   - 局部性: temporal_locality, spatial_locality (6维)

4. **系统特征 (24维)**
   - GPU特征: num_gpus, utilization, memory (8维)
   - 内存特征: usage, page_fault, swap (6维)
   - 网络特征: congestion, packet_loss, utilization (6维)
   - CPU特征: utilization, context_switch, load (4维)

5. **额外上下文 (3维)**
   - pending_ops_count
   - current_congestion
   - available_bandwidth

### 特征提取流程

```python
from torch.gpu_cluster_comm.ml import CommunicationFeatureExtractor

extractor = CommunicationFeatureExtractor()

# 提取各类特征
msg_features = extractor.extract_message_features(message)        # 32维
topo_features = extractor.extract_topology_features(topology)     # 48维
workload_features = extractor.extract_workload_features(history)  # 40维
sys_features = extractor.extract_system_features()                # 24维

# 组合
features = torch.cat([msg_features, topo_features, workload_features, sys_features])
```

---

## 性能预测模型

### 1. CommunicationTimePredictor

**架构: MLP with Residual Connections**

```
Input(120) → Linear(256) → LayerNorm → ReLU → Dropout
           → ResidualBlock × 4
           → Linear(128) → Linear(64) → Linear(1) → Softplus
```

**关键特性:**
- 4层残差块，防止梯度消失
- LayerNorm稳定训练
- Softplus输出确保预测时间为正
- 支持MC Dropout进行不确定性估计

**训练:**
- Loss: MSE
- Optimizer: Adam (lr=1e-3)
- Batch size: 64
- Epochs: 100

**性能指标:**
- MAE < 5ms
- MAPE < 15%
- R² > 0.85

### 2. BandwidthPredictor

**架构: LSTM + Attention**

```
Input: history[seq_len, 32], system_features[24]

LSTM(32 → 128, 3 layers) → Attention(8 heads)
                           ↓
System Features → Linear(24 → 128)
                           ↓
Concat → Fusion Layer → Linear(128 → 64 → 10) → Softplus
```

**关键特性:**
- 3层LSTM捕获时序依赖
- 多头注意力机制关注重要时间步
- 融合系统状态信息
- 预测未来10个时间步

**训练:**
- Loss: MSE
- Optimizer: Adam (lr=1e-3)
- Sequence length: 50
- Prediction horizon: 10

**性能指标:**
- RMSE < 10 GB/s
- 单步预测MAPE < 20%

### 3. CongestionPredictor

**架构: Graph Attention Network (GAT)**

```
Node Features[num_nodes, 32] → Linear(64)
Edge Features[num_edges, 16] → Linear(64)
                ↓
GAT Layer(64 → 64, 4 heads) × 3
                ↓
Edge Prediction: Concat(src_node, dst_node, edge_feat) → MLP → Sigmoid
```

**关键特性:**
- 3层GAT学习图结构
- 4个注意力头捕获不同关系
- 边级别预测拥塞概率
- 支持动态拓扑

**训练:**
- Loss: BCE
- Optimizer: Adam (lr=1e-3)
- Node sampling: 邻域采样

**性能指标:**
- 拥塞检测准确率 > 90%
- AUC > 0.95

---

## 算法选择模型

### 1. AlgorithmSelectorModel

**架构: Transformer Encoder**

```
Input(120) → Linear(128) → LayerNorm → ReLU
           → TransformerEncoderLayer × 4
           → Mean Pooling
           → Linear(64) → Linear(6) (算法分类)
           → Softmax
```

**支持的算法:**
1. Ring AllReduce
2. Tree AllReduce
3. Double Binary Tree
4. Halving-Doubling
5. Rabenseifner
6. Hierarchical

**训练:**
- Loss: CrossEntropy
- Optimizer: Adam (lr=1e-3)
- Class balancing: 加权采样

**性能指标:**
- Top-1 准确率 > 85%
- Top-3 准确率 > 95%

### 2. ParameterOptimizer

**架构: Multi-Head Prediction**

```
Input Features(120) → Linear(256)
Algorithm Embedding(6 → 256)
           ↓
Fusion(512 → 256) → 分支预测各参数
           ├→ chunk_size (连续)
           ├→ num_chunks (连续)
           ├→ pipeline_depth (离散, 1-8)
           ├→ compression_enable (二分类)
           ├→ compression_ratio (连续)
           ├→ enable_overlap (二分类)
           └→ priority (离散, 0-10)
```

**参数优化范围:**
- chunk_size: 1KB ~ 128MB
- num_chunks: 1 ~ 100
- pipeline_depth: 1 ~ 8
- compression_ratio: 0 ~ 1
- priority: 0 ~ 10

**训练:**
- Loss: MSE (连续) + CrossEntropy (离散)
- Multi-task learning

**性能指标:**
- 参数配置优于启发式基线 20%

---

## 强化学习模型

### CommunicationRLAgent

**算法: Proximal Policy Optimization (PPO)**

#### Actor Network (策略网络)

```
State(147) → Shared Encoder(256)
            ├→ Algorithm Head (Categorical, 6类)
            ├→ Chunk Size Head (Normal分布)
            ├→ Compression Ratio Head (Normal分布)
            ├→ Pipeline Depth Head (Categorical, 8类)
            ├→ Compression Enable Head (Categorical, 2类)
            ├→ Overlap Enable Head (Categorical, 2类)
            └→ Priority Head (Categorical, 11类)
```

#### Critic Network (价值网络)

```
State(147) → Linear(256) → LayerNorm → ReLU
           → Linear(256) → ReLU
           → Linear(128) → Linear(1) (状态价值)
```

#### 状态空间 (147维)

- topology_state: 48维
- workload_state: 40维
- system_state: 24维
- message_features: 32维
- pending_ops_count: 1维
- current_congestion: 1维
- available_bandwidth: 1维

#### 动作空间 (7维)

1. algorithm_choice: Discrete(6)
2. chunk_size_ratio: Continuous[0, 1]
3. pipeline_depth: Discrete(8)
4. enable_compression: Discrete(2)
5. compression_ratio: Continuous[0, 1]
6. enable_overlap: Discrete(2)
7. priority: Discrete(11)

#### 奖励函数

```python
reward = -actual_time / 1000.0              # 时间奖励
       + (-prediction_error) * 0.2          # 准确性奖励
       + bandwidth_utilization * 0.5        # 带宽利用率奖励
       + (-congestion_penalty) * 0.3        # 拥塞惩罚
```

#### PPO超参数

- gamma (折扣因子): 0.99
- gae_lambda (GAE参数): 0.95
- clip_epsilon (clip范围): 0.2
- value_loss_coef: 0.5
- entropy_coef: 0.01
- learning_rate: 3e-4
- num_epochs: 10
- batch_size: 64

#### 训练流程

1. 收集经验 (RolloutBuffer)
2. 计算GAE优势估计
3. PPO更新 (clip objective)
4. 更新Actor和Critic网络
5. 重复直到收敛

**性能指标:**
- 平均episode reward > 100
- 策略优于监督学习基线 15%

---

## 模型集成

### EnsemblePredictor

组合多个模型提高鲁棒性：

```python
from torch.gpu_cluster_comm.ml import EnsemblePredictor

models = [
    CommunicationTimePredictor(),
    CommunicationTimePredictor(),
    CommunicationTimePredictor(),
]

ensemble = EnsemblePredictor(models)
prediction, variance = ensemble.predict_with_variance(features)
```

**集成策略:**
- 加权平均 (权重可学习)
- Bagging (不同数据子集训练)
- Boosting (序列训练)

**优势:**
- 降低预测方差
- 提供不确定性估计
- 更好的泛化性能

---

## 在线学习

### OnlinePredictor

支持增量学习，持续优化模型：

```python
from torch.gpu_cluster_comm.ml import OnlinePredictor

online_predictor = OnlinePredictor(
    model=time_predictor,
    learning_rate=1e-4,
    window_size=1000
)

# 预测
prediction = online_predictor.predict(features)

# 在线更新
online_predictor.update(features, actual_time, update_frequency=10)

# 获取性能指标
metrics = online_predictor.get_metrics()
```

**特性:**
- 滑动窗口缓冲区
- 周期性增量更新
- 自适应学习率
- 性能监控

---

## 模型压缩和加速

### 量化

降低模型精度以加速推理：

```python
from torch.gpu_cluster_comm.ml import OptimizedInference

optimized = OptimizedInference(
    model=time_predictor,
    use_quantization=True
)
```

**量化方案:**
- 动态量化 (INT8)
- 静态量化 (校准后)
- 混合精度 (FP16/INT8)

**加速效果:**
- 推理速度提升 2-3x
- 模型大小减少 75%
- 准确率下降 < 2%

### JIT编译

使用TorchScript加速：

```python
# 自动JIT编译
inference_engine = FastInference(use_jit=True)
```

**优势:**
- 消除Python开销
- 算子融合优化
- 推理延迟降低 30-50%

---

## 模型性能基准

### 推理延迟

| 模型 | 单样本 | 批量(64) | 加速比 |
|------|--------|----------|--------|
| TimePredictor | 0.45ms | 0.15ms | 3.0x |
| BandwidthPredictor | 0.78ms | 0.25ms | 3.1x |
| AlgorithmSelector | 0.32ms | 0.10ms | 3.2x |
| ParameterOptimizer | 0.38ms | 0.12ms | 3.2x |
| RLAgent | 0.95ms | 0.30ms | 3.2x |

### 预测准确率

| 模型 | 指标 | 值 |
|------|------|-----|
| TimePredictor | MAPE | 12.3% |
| TimePredictor | R² | 0.89 |
| BandwidthPredictor | RMSE | 8.5 GB/s |
| AlgorithmSelector | Accuracy | 87.2% |
| RLAgent | Avg Reward | 125.3 |

### 缓存命中率

使用LRU缓存可显著提升性能：

- Cache size: 1000
- Hit rate: 65-75%
- 缓存命中时延迟 < 0.01ms

---

## 模型版本和更新

### 版本管理

```
models/
├── time_predictor_v1.0.pt
├── time_predictor_v1.1.pt
├── time_predictor_v2.0.pt (当前)
└── ...
```

### 模型更新策略

1. **A/B测试**: 新旧模型并行部署
2. **灰度发布**: 逐步增加新模型流量
3. **回滚机制**: 性能下降时快速回退
4. **在线学习**: 持续微调适应新数据

### 性能监控

- 预测误差趋势
- 推理延迟分布
- 模型置信度
- 缓存命中率

---

## 最佳实践

### 模型选择

- **简单场景**: 使用监督学习模型（快速、稳定）
- **复杂场景**: 使用RL agent（探索最优策略）
- **高可靠性**: 使用集成模型（降低方差）

### 训练数据

- 最少10K样本
- 覆盖不同消息大小、拓扑、工作负载
- 定期更新数据集
- 数据增强（噪声、变换）

### 推理优化

- 启用JIT编译
- 批量推理
- 特征缓存
- 模型量化

### 在线学习

- 滑动窗口1000样本
- 每100样本更新一次
- 监控性能指标
- 定期保存checkpoint

---

## 参考文献

1. PPO算法: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
2. GAT网络: Veličković et al., "Graph Attention Networks", 2018
3. Transformer: Vaswani et al., "Attention Is All You Need", 2017
4. 模型量化: Jacob et al., "Quantization and Training of Neural Networks", 2018

---

## 联系和支持

如有问题或建议，请联系开发团队或提交issue。

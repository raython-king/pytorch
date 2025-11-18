# API文档

GPU集群通讯ML优化模块的完整API参考。

## 目录

1. [特征工程](#特征工程)
2. [性能预测](#性能预测)
3. [算法选择](#算法选择)
4. [强化学习](#强化学习)
5. [训练](#训练)
6. [推理](#推理)
7. [自适应策略](#自适应策略)

---

## 特征工程

### CommunicationFeatureExtractor

从通讯操作中提取特征。

```python
from torch.gpu_cluster_comm.ml import CommunicationFeatureExtractor

extractor = CommunicationFeatureExtractor()
```

#### extract_message_features

```python
def extract_message_features(
    message: torch.Tensor,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor
```

提取消息特征（32维）。

**参数:**
- `message`: 消息张量
- `dtype`: 数据类型（可选）

**返回:** 32维特征向量

**示例:**
```python
message = torch.randn(1000, 1000)
features = extractor.extract_message_features(message)
print(features.shape)  # torch.Size([32])
```

#### extract_topology_features

```python
def extract_topology_features(topology: Any) -> torch.Tensor
```

提取拓扑特征（48维）。

**参数:**
- `topology`: 拓扑对象

**返回:** 48维特征向量

#### extract_workload_features

```python
def extract_workload_features(
    workload_history: List[Dict[str, Any]]
) -> torch.Tensor
```

提取工作负载特征（40维）。

**参数:**
- `workload_history`: 历史工作负载记录列表

**返回:** 40维特征向量

#### extract_system_features

```python
def extract_system_features() -> torch.Tensor
```

提取系统特征（24维）。

**返回:** 24维特征向量

---

## 性能预测

### CommunicationTimePredictor

预测通讯操作耗时。

```python
from torch.gpu_cluster_comm.ml import CommunicationTimePredictor

predictor = CommunicationTimePredictor(
    input_dim=120,
    hidden_dim=256,
    num_layers=4,
    dropout=0.1
)
```

#### forward

```python
def forward(features: torch.Tensor) -> torch.Tensor
```

前向传播，预测时间。

**参数:**
- `features`: [batch_size, 120] 特征张量

**返回:** [batch_size, 1] 预测时间（毫秒）

**示例:**
```python
features = torch.randn(1, 120)
predicted_time = predictor(features)
print(f"Predicted time: {predicted_time.item():.2f} ms")
```

#### predict_with_uncertainty

```python
def predict_with_uncertainty(
    features: torch.Tensor,
    num_samples: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]
```

预测时间并估计不确定性。

**返回:** (mean, std)

### BandwidthPredictor

预测未来带宽。

```python
from torch.gpu_cluster_comm.ml import BandwidthPredictor

predictor = BandwidthPredictor(
    history_dim=32,
    system_feature_dim=24,
    hidden_dim=128,
    num_layers=3,
    prediction_horizon=10
)
```

#### forward

```python
def forward(
    history: torch.Tensor,
    system_features: torch.Tensor
) -> torch.Tensor
```

**参数:**
- `history`: [batch_size, seq_len, 32] 历史带宽序列
- `system_features`: [batch_size, 24] 系统特征

**返回:** [batch_size, 10] 未来10步带宽预测

### CongestionPredictor

预测网络拥塞。

```python
from torch.gpu_cluster_comm.ml import CongestionPredictor

predictor = CongestionPredictor(
    node_feature_dim=32,
    edge_feature_dim=16,
    hidden_dim=64,
    num_gnn_layers=3
)
```

#### forward

```python
def forward(
    node_features: torch.Tensor,
    edge_features: torch.Tensor,
    edge_index: torch.Tensor
) -> torch.Tensor
```

**参数:**
- `node_features`: [num_nodes, 32] 节点特征
- `edge_features`: [num_edges, 16] 边特征
- `edge_index`: [2, num_edges] 边连接关系

**返回:** [num_edges, 1] 拥塞概率

---

## 算法选择

### AlgorithmSelectorModel

选择最优通讯算法。

```python
from torch.gpu_cluster_comm.ml import AlgorithmSelectorModel

selector = AlgorithmSelectorModel(
    input_dim=120,
    d_model=128,
    nhead=8,
    num_layers=4,
    num_algorithms=6
)
```

#### forward

```python
def forward(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

**返回:** (scores, confidence)
- `scores`: [batch_size, 6] 每种算法的得分
- `confidence`: [batch_size, 1] 预测置信度

#### predict_best_algorithm

```python
def predict_best_algorithm(
    features: torch.Tensor,
    return_probabilities: bool = False
) -> Tuple[int, Optional[torch.Tensor]]
```

预测最佳算法。

**返回:** (algorithm_index, probabilities)

**示例:**
```python
features = torch.randn(1, 120)
best_algo, probs = selector.predict_best_algorithm(features, return_probabilities=True)
print(f"Best algorithm: {CommunicationAlgorithm(best_algo).name}")
print(f"Probabilities: {probs}")
```

### ParameterOptimizer

优化算法参数。

```python
from torch.gpu_cluster_comm.ml import ParameterOptimizer

optimizer = ParameterOptimizer(
    input_dim=120,
    hidden_dim=256,
    num_algorithms=6
)
```

#### predict_optimal_params

```python
def predict_optimal_params(
    features: torch.Tensor,
    algorithm: CommunicationAlgorithm,
    message_size: int
) -> AlgorithmConfig
```

预测最优参数配置。

**返回:** AlgorithmConfig对象

**示例:**
```python
config = optimizer.predict_optimal_params(
    features,
    CommunicationAlgorithm.RING,
    message_size=1024*1024
)
print(f"Chunk size: {config.chunk_size}")
print(f"Pipeline depth: {config.pipeline_depth}")
```

### AlgorithmConfig

算法配置数据类。

**属性:**
- `algorithm`: CommunicationAlgorithm
- `chunk_size`: int
- `num_chunks`: int
- `pipeline_depth`: int
- `enable_compression`: bool
- `compression_ratio`: float
- `enable_overlap`: bool
- `priority`: int

**方法:**
```python
def to_dict() -> Dict[str, Any]
```

---

## 强化学习

### CommunicationRLAgent

基于PPO的RL智能体。

```python
from torch.gpu_cluster_comm.ml import CommunicationRLAgent

agent = CommunicationRLAgent(
    state_dim=147,
    hidden_dim=256,
    learning_rate=3e-4,
    gamma=0.99,
    clip_epsilon=0.2,
    device='cuda'
)
```

#### select_action

```python
def select_action(
    state: State,
    deterministic: bool = False
) -> Tuple[Action, torch.Tensor]
```

选择动作。

**参数:**
- `state`: 当前状态
- `deterministic`: 是否确定性选择

**返回:** (action, log_prob)

**示例:**
```python
state = State(
    topology_state=torch.randn(48),
    workload_state=torch.randn(40),
    system_state=torch.randn(24),
    message_features=torch.randn(32),
    pending_ops_count=0,
    current_congestion=0.0,
    available_bandwidth=100.0
)

action, log_prob = agent.select_action(state)
print(f"Algorithm: {CommunicationAlgorithm(action.algorithm_choice).name}")
print(f"Chunk size ratio: {action.chunk_size_ratio:.2f}")
```

#### store_transition

```python
def store_transition(
    state: State,
    action: Action,
    reward: float,
    next_state: State,
    done: bool
)
```

存储经验到回放缓冲区。

#### update_policy

```python
def update_policy(
    num_epochs: int = 10,
    batch_size: int = 64
) -> Dict[str, float]
```

更新策略（PPO）。

**返回:** 训练指标字典

#### save / load

```python
def save(path: str)
def load(path: str)
```

保存/加载模型。

### State

RL状态数据类。

**属性:**
- `topology_state`: torch.Tensor (48维)
- `workload_state`: torch.Tensor (40维)
- `system_state`: torch.Tensor (24维)
- `message_features`: torch.Tensor (32维)
- `pending_ops_count`: int
- `current_congestion`: float
- `available_bandwidth`: float

**方法:**
```python
def to_tensor() -> torch.Tensor
```

### Action

RL动作数据类。

**属性:**
- `algorithm_choice`: int (0-5)
- `chunk_size_ratio`: float (0-1)
- `pipeline_depth`: int (1-8)
- `enable_compression`: bool
- `compression_ratio`: float (0-1)
- `enable_overlap`: bool
- `priority`: int (0-10)

---

## 训练

### MLTrainer

监督学习模型训练器。

```python
from torch.gpu_cluster_comm.ml import MLTrainer, TrainingConfig

config = TrainingConfig(
    batch_size=64,
    num_epochs=100,
    learning_rate=1e-3,
    device='cuda'
)

trainer = MLTrainer(config)
```

#### train_time_predictor

```python
def train_time_predictor(
    train_dataset: CommunicationDataset,
    val_dataset: Optional[CommunicationDataset] = None
) -> Dict[str, Any]
```

训练时间预测模型。

**返回:** 训练信息字典

#### train_bandwidth_predictor

```python
def train_bandwidth_predictor(
    train_dataset: CommunicationDataset,
    val_dataset: Optional[CommunicationDataset] = None
) -> Dict[str, Any]
```

#### train_algorithm_selector

```python
def train_algorithm_selector(
    train_dataset: CommunicationDataset,
    val_dataset: Optional[CommunicationDataset] = None
) -> Dict[str, Any]
```

#### train_parameter_optimizer

```python
def train_parameter_optimizer(
    train_dataset: CommunicationDataset,
    val_dataset: Optional[CommunicationDataset] = None
) -> Dict[str, Any]
```

#### evaluate_models

```python
def evaluate_models(
    test_dataset: CommunicationDataset
) -> Dict[str, Any]
```

评估所有模型。

### RLTrainer

强化学习训练器。

```python
from torch.gpu_cluster_comm.ml import RLTrainer

trainer = RLTrainer(config)
```

#### train_rl_agent

```python
def train_rl_agent(
    simulator,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 1000
) -> Dict[str, Any]
```

训练RL智能体。

### TrainingConfig

训练配置数据类。

```python
@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_dir: str = './checkpoints'
    device: str = 'cuda'
    log_interval: int = 10
    save_interval: int = 5
```

### CommunicationDataset

通讯数据集类。

```python
from torch.gpu_cluster_comm.ml import CommunicationDataset

dataset = CommunicationDataset(data_path='./data.pt')
```

#### add_sample

```python
def add_sample(
    features: torch.Tensor,
    target: torch.Tensor,
    metadata: Optional[Dict] = None
)
```

#### save_to_file / load_from_file

```python
def save_to_file(path: str)
def load_from_file(path: str)
```

### DataCollector

数据收集器。

```python
from torch.gpu_cluster_comm.ml import DataCollector

collector = DataCollector()
```

#### collect_communication_data

```python
def collect_communication_data(
    message: torch.Tensor,
    topology: Any,
    actual_time: float,
    algorithm_used: int,
    metadata: Optional[Dict] = None
)
```

#### create_dataset

```python
def create_dataset(
    task: str = 'time_prediction'
) -> CommunicationDataset
```

**参数:**
- `task`: 'time_prediction' 或 'algorithm_selection'

---

## 推理

### FastInference

快速推理引擎。

```python
from torch.gpu_cluster_comm.ml import FastInference

engine = FastInference(
    model_dir='./checkpoints',
    device='cuda',
    use_jit=True,
    cache_size=1000
)
```

#### predict_time

```python
def predict_time(
    features: torch.Tensor,
    use_cache: bool = True
) -> torch.Tensor
```

预测通讯时间。

#### predict_bandwidth

```python
def predict_bandwidth(
    history: torch.Tensor,
    system_features: torch.Tensor
) -> torch.Tensor
```

#### select_algorithm

```python
def select_algorithm(
    features: torch.Tensor
) -> Tuple[int, torch.Tensor]
```

**返回:** (algorithm_idx, confidence)

#### optimize_parameters

```python
def optimize_parameters(
    features: torch.Tensor,
    algorithm: int,
    message_size: int
) -> AlgorithmConfig
```

#### get_statistics

```python
def get_statistics() -> Dict[str, Any]
```

获取性能统计。

**返回:**
```python
{
    'avg_inference_time_ms': float,
    'p50_inference_time_ms': float,
    'p95_inference_time_ms': float,
    'p99_inference_time_ms': float,
    'cache_hits': int,
    'cache_misses': int,
    'cache_hit_rate': float,
    'total_inferences': int,
}
```

### 便捷函数

```python
from torch.gpu_cluster_comm.ml import (
    create_fast_inference,
    benchmark_inference
)

# 创建推理引擎
engine = create_fast_inference(model_dir='./models', device='cuda')

# 基准测试
results = benchmark_inference(engine, num_samples=1000)
```

---

## 自适应策略

### AdaptiveStrategy

自适应策略引擎。

```python
from torch.gpu_cluster_comm.ml import AdaptiveStrategy

strategy = AdaptiveStrategy(
    model_dir='./checkpoints',
    device='cuda',
    use_rl=False,
    enable_online_learning=True
)
```

#### decide_communication_strategy

```python
def decide_communication_strategy(
    context: CommunicationContext,
    explore: bool = False
) -> CommunicationStrategy
```

决定通讯策略。

**参数:**
- `context`: 通讯上下文
- `explore`: 是否探索新策略

**返回:** CommunicationStrategy

**示例:**
```python
context = CommunicationContext(
    message=torch.randn(1000, 1000),
    topology=my_topology,
    workload_history=[],
    system_state={},
    message_size=1000*1000*4,
    rank=0,
    world_size=8,
    operation_type='allreduce'
)

strategy = adaptive_strategy.decide_communication_strategy(context)
print(f"Algorithm: {strategy.algorithm.name}")
print(f"Predicted time: {strategy.predicted_time:.2f} ms")
print(f"Confidence: {strategy.confidence:.2f}")
```

#### learn_from_execution

```python
def learn_from_execution(
    strategy: CommunicationStrategy,
    result: ExecutionResult
)
```

从执行结果中学习。

**示例:**
```python
result = ExecutionResult(
    actual_time=12.5,
    predicted_time=10.3,
    bandwidth_utilization=0.85,
    success=True
)

adaptive_strategy.learn_from_execution(strategy, result)
```

#### get_performance_report

```python
def get_performance_report() -> Dict[str, Any]
```

获取性能报告。

### CommunicationContext

通讯上下文数据类。

```python
@dataclass
class CommunicationContext:
    message: torch.Tensor
    topology: Any
    workload_history: List[Dict]
    system_state: Dict
    message_size: int
    rank: int
    world_size: int
    operation_type: str
```

### CommunicationStrategy

通讯策略数据类。

```python
@dataclass
class CommunicationStrategy:
    algorithm: CommunicationAlgorithm
    config: AlgorithmConfig
    predicted_time: float
    confidence: float
    metadata: Dict[str, Any]
```

**方法:**
```python
def to_dict() -> Dict[str, Any]
```

### ExecutionResult

执行结果数据类。

```python
@dataclass
class ExecutionResult:
    actual_time: float
    predicted_time: float
    bandwidth_utilization: float
    success: bool
    error_message: Optional[str] = None
```

### PolicyOptimizer

策略优化器。

```python
from torch.gpu_cluster_comm.ml import PolicyOptimizer

optimizer = PolicyOptimizer(adaptive_strategy)
```

#### optimize_for_workload

```python
def optimize_for_workload(
    workload_profile: Dict[str, Any]
) -> Dict[str, Any]
```

为特定工作负载优化策略。

---

## 工具函数

### 数据集工具

```python
from torch.gpu_cluster_comm.ml.datasets import (
    create_synthetic_dataset,
    split_dataset,
    DatasetStatistics
)

# 创建合成数据集
dataset = create_synthetic_dataset(num_samples=10000)

# 划分数据集
train, val, test = split_dataset(dataset)

# 统计信息
DatasetStatistics.print_statistics(dataset)
```

---

## 完整示例

### 端到端流程

```python
import torch
from torch.gpu_cluster_comm.ml import (
    AdaptiveStrategy,
    CommunicationContext,
    ExecutionResult
)

# 1. 创建自适应策略引擎
strategy_engine = AdaptiveStrategy(
    model_dir='./models',
    device='cuda',
    use_rl=False
)

# 2. 准备通讯上下文
context = CommunicationContext(
    message=torch.randn(1024, 1024),
    topology=topology,
    workload_history=[],
    system_state={},
    message_size=1024*1024*4,
    rank=0,
    world_size=8,
    operation_type='allreduce'
)

# 3. 决定策略
strategy = strategy_engine.decide_communication_strategy(context)

# 4. 执行通讯（伪代码）
actual_time = execute_communication(strategy)

# 5. 反馈学习
result = ExecutionResult(
    actual_time=actual_time,
    predicted_time=strategy.predicted_time,
    bandwidth_utilization=0.9,
    success=True
)
strategy_engine.learn_from_execution(strategy, result)

# 6. 获取报告
report = strategy_engine.get_performance_report()
print(report)
```

---

## 枚举类型

### CommunicationAlgorithm

```python
class CommunicationAlgorithm(Enum):
    RING = 0
    TREE = 1
    DOUBLE_BINARY_TREE = 2
    HALVING_DOUBLING = 3
    RABENSEIFNER = 4
    HIERARCHICAL = 5
```

---

## 异常处理

所有函数可能抛出的异常：

- `RuntimeError`: 模型未加载或配置错误
- `ValueError`: 参数值非法
- `FileNotFoundError`: 模型文件不存在
- `AssertionError`: 输入维度不匹配

**建议:**
```python
try:
    prediction = engine.predict_time(features)
except RuntimeError as e:
    print(f"Inference failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## 性能提示

1. **批量推理**: 使用`predict_batch`代替循环调用
2. **缓存**: 启用`use_cache=True`
3. **JIT编译**: 启用`use_jit=True`
4. **量化**: 使用`OptimizedInference`进行量化
5. **GPU**: 确保数据和模型在同一设备上

---

## 版本信息

当前版本: **0.1.0**

---

## 参考链接

- [ML模型详解](ML_MODELS.md)
- [训练指南](TRAINING_GUIDE.md)
- [GitHub仓库](#)

---

## 许可证

MIT License

---

本API文档覆盖了所有公开接口。如有疑问，请参考源代码或提交issue。

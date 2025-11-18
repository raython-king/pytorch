# 模型训练指南

本指南详细说明如何训练和优化GPU集群通讯的ML模型。

## 目录

1. [快速开始](#快速开始)
2. [数据收集](#数据收集)
3. [数据准备](#数据准备)
4. [监督学习模型训练](#监督学习模型训练)
5. [强化学习训练](#强化学习训练)
6. [模型评估](#模型评估)
7. [超参数调优](#超参数调优)
8. [部署和推理](#部署和推理)

---

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install torch torchvision numpy scipy

# 检查GPU可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### 30秒训练一个模型

```python
from torch.gpu_cluster_comm.ml import (
    MLTrainer,
    TrainingConfig,
    create_synthetic_dataset
)

# 1. 创建合成数据集
dataset = create_synthetic_dataset(num_samples=10000)

# 2. 配置训练
config = TrainingConfig(
    batch_size=64,
    num_epochs=50,
    learning_rate=1e-3,
    device='cuda'
)

# 3. 创建训练器
trainer = MLTrainer(config)

# 4. 训练时间预测模型
train_dataset, val_dataset = dataset[:8000], dataset[8000:]
trainer.train_time_predictor(train_dataset, val_dataset)

print("训练完成!")
```

---

## 数据收集

### 从实际通讯中收集数据

```python
from torch.gpu_cluster_comm.ml import DataCollector

# 创建数据收集器
collector = DataCollector()

# 在每次通讯后收集数据
def on_communication_complete(message, topology, actual_time, algorithm):
    collector.collect_communication_data(
        message=message,
        topology=topology,
        actual_time=actual_time,
        algorithm_used=algorithm,
        metadata={'rank': rank, 'operation': 'allreduce'}
    )

# 保存收集的数据
collector.save_collected_data('./collected_data.pt')
```

### 数据收集最佳实践

1. **覆盖范围**:
   - 不同消息大小: 1KB ~ 1GB
   - 不同拓扑: 单机、多机、不同连接方式
   - 不同工作负载: 稀疏、密集、突发

2. **数据量**:
   - 最少10,000个样本
   - 推荐50,000+样本用于生产级模型

3. **数据质量**:
   - 准确的时间测量
   - 完整的上下文信息
   - 去除异常值

4. **数据平衡**:
   - 各算法样本数量均衡
   - 各消息大小区间均衡

---

## 数据准备

### 创建数据集

```python
from torch.gpu_cluster_comm.ml import DataCollector

# 从收集的数据创建数据集
collector = DataCollector()
collector.load_collected_data('./collected_data.pt')

# 创建时间预测数据集
time_dataset = collector.create_dataset(task='time_prediction')

# 创建算法选择数据集
algo_dataset = collector.create_dataset(task='algorithm_selection')
```

### 数据集划分

```python
from torch.gpu_cluster_comm.ml.datasets import split_dataset

# 划分数据集
train_dataset, val_dataset, test_dataset = split_dataset(
    dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)

print(f"Train: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")
```

### 数据统计分析

```python
from torch.gpu_cluster_comm.ml.datasets import DatasetStatistics

# 打印统计信息
DatasetStatistics.print_statistics(train_dataset)

# 保存统计信息
DatasetStatistics.save_statistics(train_dataset, './dataset_stats.json')
```

### 数据增强

```python
def augment_dataset(dataset, augmentation_factor=2):
    """数据增强"""
    augmented = CommunicationDataset()

    for features, target in dataset:
        # 原始样本
        augmented.add_sample(features, target)

        # 添加噪声
        for _ in range(augmentation_factor - 1):
            noise = torch.randn_like(features) * 0.05
            aug_features = features + noise
            augmented.add_sample(aug_features, target)

    return augmented
```

---

## 监督学习模型训练

### 训练配置

```python
from torch.gpu_cluster_comm.ml import TrainingConfig

config = TrainingConfig(
    batch_size=64,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-5,
    val_split=0.2,
    early_stopping_patience=10,
    checkpoint_dir='./checkpoints',
    device='cuda',
    log_interval=10,
    save_interval=5
)
```

### 训练时间预测模型

```python
from torch.gpu_cluster_comm.ml import MLTrainer

trainer = MLTrainer(config)

# 训练
training_info = trainer.train_time_predictor(
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

print(f"Best validation loss: {training_info['best_val_loss']:.6f}")

# 可视化训练历史
import matplotlib.pyplot as plt

plt.plot(trainer.training_history['time_predictor_train_loss'], label='Train')
plt.plot(trainer.training_history['time_predictor_val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_curve.png')
```

### 训练带宽预测模型

```python
# 准备时序数据集
# 确保数据集包含历史序列

training_info = trainer.train_bandwidth_predictor(
    train_dataset=bandwidth_train_dataset,
    val_dataset=bandwidth_val_dataset
)
```

### 训练算法选择模型

```python
training_info = trainer.train_algorithm_selector(
    train_dataset=algo_train_dataset,
    val_dataset=algo_val_dataset
)

print(f"Best validation accuracy: {training_info['best_val_acc']:.4f}")
```

### 训练参数优化模型

```python
training_info = trainer.train_parameter_optimizer(
    train_dataset=param_train_dataset,
    val_dataset=param_val_dataset
)
```

### 批量训练所有模型

```python
def train_all_models(trainer, datasets):
    """训练所有监督学习模型"""

    results = {}

    # 时间预测器
    print("Training Time Predictor...")
    results['time'] = trainer.train_time_predictor(
        datasets['time_train'],
        datasets['time_val']
    )

    # 带宽预测器
    print("Training Bandwidth Predictor...")
    results['bandwidth'] = trainer.train_bandwidth_predictor(
        datasets['bandwidth_train'],
        datasets['bandwidth_val']
    )

    # 算法选择器
    print("Training Algorithm Selector...")
    results['algorithm'] = trainer.train_algorithm_selector(
        datasets['algo_train'],
        datasets['algo_val']
    )

    # 参数优化器
    print("Training Parameter Optimizer...")
    results['parameter'] = trainer.train_parameter_optimizer(
        datasets['param_train'],
        datasets['param_val']
    )

    return results

# 执行训练
results = train_all_models(trainer, prepared_datasets)
```

---

## 强化学习训练

### 创建模拟环境

```python
class CommunicationSimulator:
    """通讯模拟器环境"""

    def reset(self):
        """重置环境"""
        # 随机初始化状态
        state = State(
            topology_state=torch.randn(48),
            workload_state=torch.randn(40),
            system_state=torch.randn(24),
            message_features=torch.randn(32),
            pending_ops_count=0,
            current_congestion=0.0,
            available_bandwidth=100.0
        )
        return state

    def step(self, action):
        """执行动作"""
        # 模拟通讯执行
        actual_time = self._simulate_communication(action)

        # 计算奖励
        reward = -actual_time / 1000.0

        # 下一个状态
        next_state = self._get_next_state()

        # 是否结束
        done = self._is_done()

        return next_state, reward, done, {}
```

### 训练RL Agent

```python
from torch.gpu_cluster_comm.ml import RLTrainer, TrainingConfig

# 创建配置
config = TrainingConfig(
    checkpoint_dir='./rl_checkpoints',
    device='cuda'
)

# 创建训练器
rl_trainer = RLTrainer(config)

# 创建模拟器
simulator = CommunicationSimulator()

# 训练
training_info = rl_trainer.train_rl_agent(
    simulator=simulator,
    num_episodes=1000,
    max_steps_per_episode=1000
)

print(f"Average reward: {training_info['average_reward']:.2f}")
print(f"Final reward: {training_info['final_reward']:.2f}")
```

### RL训练监控

```python
# 可视化训练进度
import matplotlib.pyplot as plt
import numpy as np

episode_rewards = training_info['episode_rewards']

# 滑动平均
window_size = 100
moving_avg = np.convolve(
    episode_rewards,
    np.ones(window_size) / window_size,
    mode='valid'
)

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
plt.plot(moving_avg, label=f'Moving Average ({window_size})')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.title('RL Training Progress')
plt.savefig('rl_training.png')
```

### RL超参数调优

```python
# 网格搜索
hyperparams = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'gamma': [0.95, 0.99],
    'clip_epsilon': [0.1, 0.2, 0.3],
}

best_reward = float('-inf')
best_params = None

for lr in hyperparams['learning_rate']:
    for gamma in hyperparams['gamma']:
        for clip_eps in hyperparams['clip_epsilon']:
            # 创建新agent
            agent = CommunicationRLAgent(
                learning_rate=lr,
                gamma=gamma,
                clip_epsilon=clip_eps
            )

            # 训练
            trainer = RLTrainer(config)
            trainer.agent = agent
            result = trainer.train_rl_agent(simulator, num_episodes=500)

            if result['average_reward'] > best_reward:
                best_reward = result['average_reward']
                best_params = (lr, gamma, clip_eps)

print(f"Best params: LR={best_params[0]}, Gamma={best_params[1]}, Clip={best_params[2]}")
print(f"Best reward: {best_reward:.2f}")
```

---

## 模型评估

### 评估监督学习模型

```python
# 评估所有模型
evaluation_results = trainer.evaluate_models(test_dataset)

for model_name, metrics in evaluation_results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
```

### 详细评估时间预测器

```python
from torch.gpu_cluster_comm.ml import PerformanceMetrics

metrics = PerformanceMetrics()

for features, actual_time in test_dataset:
    predicted_time = trainer.time_predictor(features.unsqueeze(0)).item()
    metrics.add_prediction(predicted_time, actual_time.item(), time.time())

# 计算指标
results = metrics.compute_metrics()

print("Time Predictor Evaluation:")
print(f"  MAE: {results['mae']:.2f} ms")
print(f"  RMSE: {results['rmse']:.2f} ms")
print(f"  MAPE: {results['mape']:.2f}%")
print(f"  R²: {results['r2']:.4f}")
print(f"  Median AE: {results['median_ae']:.2f} ms")
print(f"  95th percentile error: {results['p95_error']:.2f} ms")
```

### 评估算法选择器

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

predictions = []
actuals = []

for features, actual_algo in algo_test_dataset:
    pred_algo, _ = trainer.algorithm_selector.predict_best_algorithm(
        features.unsqueeze(0)
    )
    predictions.append(pred_algo)
    actuals.append(actual_algo.item())

# 混淆矩阵
cm = confusion_matrix(actuals, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Algorithm Selection Confusion Matrix')
plt.savefig('confusion_matrix.png')

# 分类报告
print(classification_report(actuals, predictions, target_names=[
    'Ring', 'Tree', 'DoubleBinaryTree',
    'HalvingDoubling', 'Rabenseifner', 'Hierarchical'
]))
```

### 交叉验证

```python
from torch.utils.data import Subset
import numpy as np

def k_fold_cross_validation(dataset, k=5):
    """K折交叉验证"""
    fold_size = len(dataset) // k
    results = []

    for fold in range(k):
        # 划分数据
        val_indices = range(fold * fold_size, (fold + 1) * fold_size)
        train_indices = list(set(range(len(dataset))) - set(val_indices))

        train_fold = Subset(dataset, train_indices)
        val_fold = Subset(dataset, val_indices)

        # 训练
        config = TrainingConfig(num_epochs=20)
        trainer = MLTrainer(config)
        info = trainer.train_time_predictor(train_fold, val_fold)

        results.append(info['best_val_loss'])
        print(f"Fold {fold + 1}: Val Loss = {info['best_val_loss']:.6f}")

    print(f"\nCross-Validation Results:")
    print(f"  Mean: {np.mean(results):.6f}")
    print(f"  Std: {np.std(results):.6f}")

k_fold_cross_validation(full_dataset, k=5)
```

---

## 超参数调优

### 使用Optuna进行自动调优

```python
import optuna

def objective(trial):
    """Optuna目标函数"""
    # 建议超参数
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.3)

    # 配置
    config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=lr,
        num_epochs=50
    )

    # 创建模型
    model = CommunicationTimePredictor(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    # 训练
    trainer = MLTrainer(config)
    trainer.time_predictor = model
    info = trainer.train_time_predictor(train_dataset, val_dataset)

    return info['best_val_loss']

# 运行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value:.6f}")
```

### 学习率调度

```python
# 余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.num_epochs,
    eta_min=1e-6
)

# 指数衰减
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95
)

# ReduceLROnPlateau (推荐)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)
```

---

## 部署和推理

### 保存训练好的模型

```python
# 保存单个模型
trainer.save_model(trainer.time_predictor, 'time_predictor_best.pt')

# 保存所有模型
def save_all_models(trainer, save_dir='./models'):
    Path(save_dir).mkdir(exist_ok=True)

    trainer.save_model(trainer.time_predictor, f'{save_dir}/time_predictor_best.pt')
    trainer.save_model(trainer.bandwidth_predictor, f'{save_dir}/bandwidth_predictor_best.pt')
    trainer.save_model(trainer.algorithm_selector, f'{save_dir}/algorithm_selector_best.pt')
    trainer.save_model(trainer.parameter_optimizer, f'{save_dir}/parameter_optimizer_best.pt')

    print(f"All models saved to {save_dir}")

save_all_models(trainer)
```

### 加载和推理

```python
from torch.gpu_cluster_comm.ml import create_fast_inference

# 创建推理引擎
inference_engine = create_fast_inference(
    model_dir='./models',
    device='cuda',
    use_jit=True
)

# 推理
features = torch.randn(1, 120).cuda()
predicted_time = inference_engine.predict_time(features)

print(f"Predicted time: {predicted_time.item():.2f} ms")
```

### 性能基准测试

```python
from torch.gpu_cluster_comm.ml import benchmark_inference

# 运行基准测试
results = benchmark_inference(
    inference_engine,
    num_samples=1000,
    feature_dim=120
)

print("Benchmark Results:")
print(f"  Single inference avg: {results['single_inference_avg_ms']:.4f} ms")
print(f"  Batch inference per sample: {results['batch_inference_per_sample_ms']:.4f} ms")
print(f"  Speedup factor: {results['speedup_factor']:.2f}x")
print(f"  Cache hit rate: {results['cache_hit_rate']:.2%}")
```

---

## 最佳实践

### 数据收集

- 收集至少10K真实样本
- 覆盖不同场景和配置
- 定期更新数据集

### 训练

- 使用early stopping防止过拟合
- 监控训练曲线
- 保存最佳模型
- 使用交叉验证

### 超参数

- 学习率: 1e-4 ~ 1e-3
- Batch size: 64 (平衡速度和性能)
- 正则化: weight_decay=1e-5
- Dropout: 0.1 ~ 0.2

### 评估

- 多个指标综合评估
- 在真实场景测试
- A/B测试对比基线

### 部署

- JIT编译加速
- 批量推理
- 缓存机制
- 性能监控

---

## 故障排除

### 训练不收敛

- 降低学习率
- 检查数据质量
- 增加模型容量
- 使用梯度裁剪

### 过拟合

- 增加dropout
- 使用正则化
- 减少模型复杂度
- 增加训练数据

### 推理延迟高

- 启用JIT编译
- 使用批量推理
- 模型量化
- 减少模型大小

### 内存不足

- 减少batch size
- 使用梯度累积
- 混合精度训练
- 模型并行

---

## 进阶主题

### 分布式训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 初始化分布式
dist.init_process_group(backend='nccl')

# 包装模型
model = DistributedDataParallel(model)

# 使用DistributedSampler
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
```

### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for features, targets in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(features)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 迁移学习

```python
# 加载预训练模型
pretrained_model = torch.load('pretrained_model.pt')

# 冻结部分层
for param in pretrained_model.shared.parameters():
    param.requires_grad = False

# 只训练最后几层
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, pretrained_model.parameters()),
    lr=1e-4
)
```

---

## 总结

本指南涵盖了从数据收集到模型部署的完整训练流程。关键要点：

1. 高质量数据是成功的基础
2. 合理的超参数至关重要
3. 充分评估和测试
4. 持续优化和更新

祝训练顺利！

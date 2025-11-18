# GPU集群通讯优化系统 - 接口与API设计

本文档详细描述GPU集群自适应通讯优化系统（AGCCS）的完整API设计、配置选项和使用示例。

---

## 目录

1. [用户API](#1-用户api)
2. [配置系统](#2-配置系统)
3. [监控与诊断API](#3-监控与诊断api)
4. [扩展接口](#4-扩展接口)
5. [与torch.distributed集成](#5-与torchdistributed集成)
6. [完整示例](#6-完整示例)

---

## 1. 用户API

### 1.1 快速启用API

#### 基础用法

```python
import torch
import torch.distributed as dist
from torch.adaptive_comm import enable_adaptive_comm, AdaptiveCommConfig

# 方式1：一行启用，使用默认配置
enable_adaptive_comm()

# 方式2：使用预设配置
from torch.adaptive_comm import ConfigPresets

enable_adaptive_comm(ConfigPresets.high_throughput())

# 方式3：自定义配置
config = AdaptiveCommConfig(
    algorithm_selection='ml',       # 'ml', 'heuristic', 'fixed'
    compression_enabled=True,
    overlap_enabled=True,
    monitoring_enabled=True
)

enable_adaptive_comm(config)

# 之后的所有distributed操作都会自动优化
model = YourModel().cuda()
ddp_model = torch.nn.parallel.DistributedDataParallel(model)

# 训练循环
for batch in dataloader:
    output = ddp_model(batch)
    loss.backward()
    optimizer.step()
    # AllReduce会自动使用AGCCS优化
```

### 1.2 显式API

对于需要精细控制的用户：

```python
from torch.adaptive_comm import AdaptiveCollective

class MyDistributedTrainer:
    def __init__(self):
        # 创建AdaptiveCollective实例
        self.comm = AdaptiveCollective(
            world_size=dist.get_world_size(),
            rank=dist.get_rank()
        )

    def all_reduce_gradients(self, gradients: List[torch.Tensor]):
        """使用自适应通讯进行AllReduce"""

        # 方式1：自动优化
        for grad in gradients:
            self.comm.all_reduce(grad)

        # 方式2：指定算法
        for grad in gradients:
            self.comm.all_reduce(
                grad,
                algorithm='ring',         # 'ring', 'tree', 'auto'
                compression='fp16',       # 'none', 'fp16', 'bf16', 'int8', 'auto'
                overlap_compute=True,
                async_op=False
            )

        # 方式3：批量操作
        self.comm.all_reduce_coalesced(
            gradients,
            bucket_size_mb=25.0
        )

    def all_gather_features(self, local_features: torch.Tensor):
        """AllGather操作"""

        all_features = self.comm.all_gather(
            local_features,
            algorithm='auto'
        )

        return all_features

    def reduce_scatter(self, tensors: List[torch.Tensor]):
        """ReduceScatter操作"""

        output = self.comm.reduce_scatter(
            tensors,
            algorithm='auto'
        )

        return output
```

### 1.3 上下文管理器API

```python
from torch.adaptive_comm import adaptive_comm_context, QoSLevel

# 方式1：临时启用/禁用
with adaptive_comm_context(enabled=True):
    # 该块内的通讯使用AGCCS
    dist.all_reduce(tensor)

with adaptive_comm_context(enabled=False):
    # 该块内的通讯使用原始NCCL
    dist.all_reduce(tensor)

# 方式2：指定QoS级别
with adaptive_comm_context(qos=QoSLevel.HIGH_PRIORITY):
    # 高优先级通讯（低延迟）
    dist.all_reduce(critical_tensor)

with adaptive_comm_context(qos=QoSLevel.BEST_EFFORT):
    # 尽力而为（可以被抢占）
    dist.all_reduce(non_critical_tensor)

# 方式3：指定压缩策略
with adaptive_comm_context(compression='int8'):
    # 该块内所有通讯使用INT8压缩
    for grad in gradients:
        dist.all_reduce(grad)
```

### 1.4 装饰器API

```python
from torch.adaptive_comm import optimize_comm, CommOptimizer

class DistributedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # ...

    @optimize_comm(
        algorithm='auto',
        compression='auto',
        overlap=True
    )
    def forward(self, x):
        # 前向传播中的所有通讯自动优化
        # ...
        return output

    @optimize_comm(compression='fp16')
    def backward_hook(self, grad):
        # 反向传播中的梯度通讯使用FP16
        # ...
        return grad
```

---

## 2. 配置系统

### 2.1 配置类定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class AdaptiveCommConfig:
    """自适应通讯配置"""

    # ========== 全局开关 ==========
    enabled: bool = True
    monitoring_enabled: bool = True
    ml_enabled: bool = True

    # ========== 算法选择 ==========
    algorithm_selection: str = 'ml'  # 'ml', 'heuristic', 'fixed'
    default_algorithm: str = 'ring'  # 当algorithm_selection='fixed'时使用

    # 算法参数
    ring_chunk_size_mb: float = 10.0
    tree_arity: int = 2  # 二叉树 or 多叉树
    hierarchical_enabled: bool = True

    # ========== 压缩配置 ==========
    compression_enabled: bool = True
    compression_selection: str = 'ml'  # 'ml', 'heuristic', 'fixed'
    default_compression: str = 'none'  # 'none', 'fp16', 'bf16', 'int8', 'topk'

    # 压缩参数
    compression_error_budget: float = 0.01  # 允许的最大累积误差
    topk_ratio: float = 0.01  # Top-K稀疏化保留比例
    error_feedback_enabled: bool = True

    # ========== Overlap配置 ==========
    overlap_enabled: bool = True
    bucket_size_mb: float = 25.0
    max_buckets: int = 100
    pipeline_depth: int = 4
    num_streams: int = 2
    dynamic_bucketing: bool = True

    # ========== 拓扑配置 ==========
    topology_discovery: str = 'auto'  # 'auto', 'manual', 'file'
    topology_file: Optional[str] = None
    routing_objective: str = 'balanced'  # 'latency', 'bandwidth', 'balanced'
    load_balancing_enabled: bool = True

    # ========== 负载均衡 ==========
    straggler_detection_enabled: bool = True
    straggler_threshold_percentile: float = 90.0
    work_stealing_enabled: bool = False
    rebalance_interval: int = 100  # 迭代数

    # ========== 消息聚合 ==========
    message_coalescing_enabled: bool = True
    coalescing_buffer_size_mb: float = 10.0
    coalescing_timeout_ms: float = 10.0

    # ========== ML配置 ==========
    ml_model_path: Optional[str] = None
    ml_online_learning: bool = True
    ml_update_interval: int = 100
    ml_inference_batch_size: int = 1

    # ========== 监控配置 ==========
    profiling_enabled: bool = True
    profiling_sample_rate: float = 0.1  # 采样率
    metrics_export_enabled: bool = False
    metrics_export_path: Optional[str] = None

    # ========== 性能调优 ==========
    fast_path_threshold_kb: int = 100  # < 100KB的消息走fast path
    gpu_direct_enabled: bool = True
    zero_copy_enabled: bool = True

    # ========== 调试配置 ==========
    debug_mode: bool = False
    verbose: bool = False
    trace_enabled: bool = False
    trace_output_path: Optional[str] = None

    # ========== 兼容性配置 ==========
    fallback_on_error: bool = True
    compatibility_mode: bool = False  # 兼容模式，禁用部分优化


# 预设配置
class ConfigPresets:
    """配置预设"""

    @staticmethod
    def low_latency() -> AdaptiveCommConfig:
        """低延迟配置（适合推理、小batch训练）"""
        return AdaptiveCommConfig(
            algorithm_selection='ml',
            default_algorithm='tree',
            compression_enabled=False,
            overlap_enabled=True,
            bucket_size_mb=10.0,
            routing_objective='latency',
            message_coalescing_enabled=False
        )

    @staticmethod
    def high_throughput() -> AdaptiveCommConfig:
        """高吞吐配置（适合大batch训练）"""
        return AdaptiveCommConfig(
            algorithm_selection='ml',
            default_algorithm='ring',
            compression_enabled=True,
            compression_selection='ml',
            overlap_enabled=True,
            bucket_size_mb=50.0,
            routing_objective='bandwidth',
            message_coalescing_enabled=True
        )

    @staticmethod
    def fair_sharing() -> AdaptiveCommConfig:
        """公平共享配置（适合多租户环境）"""
        return AdaptiveCommConfig(
            algorithm_selection='heuristic',
            load_balancing_enabled=True,
            straggler_detection_enabled=True,
            work_stealing_enabled=True,
            routing_objective='balanced'
        )

    @staticmethod
    def distributed_training() -> AdaptiveCommConfig:
        """分布式训练配置（DDP/FSDP）"""
        return AdaptiveCommConfig(
            algorithm_selection='ml',
            compression_enabled=True,
            compression_selection='ml',
            overlap_enabled=True,
            bucket_size_mb=25.0,
            dynamic_bucketing=True,
            hierarchical_enabled=True,
            ml_online_learning=True
        )

    @staticmethod
    def bandwidth_constrained() -> AdaptiveCommConfig:
        """带宽受限配置（适合跨数据中心）"""
        return AdaptiveCommConfig(
            compression_enabled=True,
            default_compression='int8',
            compression_selection='ml',
            message_coalescing_enabled=True,
            hierarchical_enabled=True
        )
```

### 2.2 配置加载与保存

```python
class ConfigManager:
    """配置管理器"""

    @staticmethod
    def load_from_file(path: str) -> AdaptiveCommConfig:
        """从YAML/JSON文件加载配置"""

        import yaml

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return AdaptiveCommConfig(**config_dict)

    @staticmethod
    def save_to_file(config: AdaptiveCommConfig, path: str):
        """保存配置到文件"""

        import yaml

        config_dict = asdict(config)

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @staticmethod
    def load_from_env() -> AdaptiveCommConfig:
        """从环境变量加载配置

        环境变量格式：AGCCS_<CONFIG_NAME>=<VALUE>
        例如：AGCCS_ALGORITHM_SELECTION=ml
        """

        import os

        config = AdaptiveCommConfig()

        for field_name in config.__dataclass_fields__:
            env_name = f"AGCCS_{field_name.upper()}"
            if env_name in os.environ:
                value = os.environ[env_name]

                # 类型转换
                field_type = config.__dataclass_fields__[field_name].type
                if field_type == bool:
                    value = value.lower() in ('true', '1', 'yes')
                elif field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)

                setattr(config, field_name, value)

        return config


# 使用示例
# 方式1：从文件加载
config = ConfigManager.load_from_file('agccs_config.yaml')
enable_adaptive_comm(config)

# 方式2：从环境变量
import os
os.environ['AGCCS_ALGORITHM_SELECTION'] = 'ml'
os.environ['AGCCS_COMPRESSION_ENABLED'] = 'true'

config = ConfigManager.load_from_env()
enable_adaptive_comm(config)

# 方式3：保存配置
config = ConfigPresets.high_throughput()
ConfigManager.save_to_file(config, 'my_config.yaml')
```

---

## 3. 监控与诊断API

### 3.1 性能统计API

```python
from torch.adaptive_comm import get_comm_stats, CommStats

# 获取全局统计
stats = get_comm_stats()

print(f"Total communications: {stats.total_count}")
print(f"Total bytes transferred: {stats.total_bytes / 1e9:.2f} GB")
print(f"Average bandwidth: {stats.avg_bandwidth_gbps:.2f} GB/s")
print(f"Average latency: {stats.avg_latency_ms:.2f} ms")
print(f"P99 latency: {stats.p99_latency_ms:.2f} ms")

# 按操作类型分组统计
for op_type in ['all_reduce', 'all_gather', 'reduce_scatter']:
    op_stats = stats.get_by_operation(op_type)
    print(f"{op_type}: {op_stats.total_count} ops, "
          f"{op_stats.avg_latency_ms:.2f} ms avg latency")

# 按消息大小范围统计
for size_range in ['<1MB', '1MB-10MB', '10MB-100MB', '>100MB']:
    range_stats = stats.get_by_size_range(size_range)
    print(f"{size_range}: {range_stats.total_count} ops")

# 算法使用统计
algo_usage = stats.get_algorithm_usage()
for algo, count in algo_usage.items():
    print(f"{algo}: {count} times ({count/stats.total_count*100:.1f}%)")

# 压缩统计
compression_stats = stats.get_compression_stats()
print(f"Compression ratio: {compression_stats.avg_compression_ratio:.2f}x")
print(f"Bytes saved: {compression_stats.bytes_saved / 1e9:.2f} GB")
```

### 3.2 实时监控API

```python
from torch.adaptive_comm import CommunicationMonitor

class MyTrainer:
    def __init__(self):
        # 创建监控器
        self.monitor = CommunicationMonitor(
            sample_rate=0.1,      # 采样10%的通讯
            window_size_sec=10.0  # 10秒滑动窗口
        )

    def train_step(self, batch):
        # 开启监控
        with self.monitor.start_recording():
            # ... 训练代码 ...
            pass

        # 获取本步的统计
        step_stats = self.monitor.get_last_step_stats()

        # 检查异常
        if step_stats.avg_latency_ms > 100.0:
            print(f"Warning: High latency detected: {step_stats.avg_latency_ms:.2f} ms")

        # 检查拥塞
        congestion_report = self.monitor.get_congestion_report()
        if congestion_report.num_congested_links > 0:
            print(f"Warning: {congestion_report.num_congested_links} links congested")
            print(f"Hotspots: {congestion_report.hotspot_links}")
```

### 3.3 Profiling API

```python
from torch.adaptive_comm import CommunicationProfiler

# 方式1：上下文管理器
with CommunicationProfiler(output_path='comm_profile.json') as profiler:
    # 训练代码
    for epoch in range(num_epochs):
        for batch in dataloader:
            # ...
            pass

# Profile自动保存到comm_profile.json

# 方式2：手动控制
profiler = CommunicationProfiler()

profiler.start()
# ... 训练代码 ...
profiler.stop()

# 导出trace
profiler.export_chrome_trace('comm_trace.json')
profiler.export_tensorboard('runs/comm_profile')

# 分析结果
report = profiler.generate_report()

print(report.summary)
print(f"Top 10 slowest communications:")
for comm in report.top_slow_comms(10):
    print(f"  {comm.operation} {comm.size_mb:.2f}MB: {comm.latency_ms:.2f}ms")

print(f"Communication pattern: {report.pattern_type}")
if report.pattern_type == 'periodic':
    print(f"  Period: {report.period_ms:.2f}ms")
```

### 3.4 诊断API

```python
from torch.adaptive_comm import diagnose_communication

# 自动诊断通讯问题
diagnosis = diagnose_communication(
    duration_sec=60.0,  # 诊断1分钟
    verbose=True
)

print("Diagnosis Report:")
print("=" * 60)

# 性能问题
if diagnosis.has_performance_issues:
    print("Performance Issues:")
    for issue in diagnosis.performance_issues:
        print(f"  - {issue.severity}: {issue.description}")
        print(f"    Suggestion: {issue.suggestion}")

# 拥塞问题
if diagnosis.has_congestion:
    print("\nCongestion Detected:")
    print(f"  Congested links: {diagnosis.congested_links}")
    print(f"  Suggested actions: {diagnosis.congestion_suggestions}")

# Stragglers
if diagnosis.has_stragglers:
    print("\nStragglers Detected:")
    print(f"  Slow ranks: {diagnosis.straggler_ranks}")
    print(f"  Speed ratio: {diagnosis.straggler_ratio:.2f}x slower")

# 配置建议
print("\nConfiguration Suggestions:")
for suggestion in diagnosis.config_suggestions:
    print(f"  - {suggestion.config_key}: {suggestion.recommended_value}")
    print(f"    Reason: {suggestion.reason}")
```

---

## 4. 扩展接口

### 4.1 自定义算法注册

```python
from torch.adaptive_comm import register_algorithm, CollectiveAlgorithm

class MyCustomRingAllReduce(CollectiveAlgorithm):
    """自定义Ring AllReduce实现"""

    def execute(
        self,
        tensor: torch.Tensor,
        world_size: int,
        rank: int,
        **kwargs
    ) -> torch.Tensor:
        """执行AllReduce"""

        # 自定义实现
        # ...

        return tensor

    def estimate_time(
        self,
        tensor_size: int,
        world_size: int,
        topology: 'ClusterTopology'
    ) -> float:
        """估计执行时间（ms）"""

        # 性能模型
        bandwidth = topology.avg_bandwidth_gbps
        latency = topology.avg_latency_ms

        time_ms = 2 * (world_size - 1) / world_size * tensor_size / bandwidth / 1e9 * 1000
        time_ms += 2 * (world_size - 1) * latency

        return time_ms

# 注册自定义算法
register_algorithm('my_ring', MyCustomRingAllReduce())

# 使用自定义算法
from torch.adaptive_comm import AdaptiveCollective

comm = AdaptiveCollective()
comm.all_reduce(tensor, algorithm='my_ring')
```

### 4.2 自定义压缩器注册

```python
from torch.adaptive_comm import register_compressor, Compressor

class MyCustomCompressor(Compressor):
    """自定义压缩器"""

    def compress(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, CompressionMetadata]:
        """压缩张量"""

        # 自定义压缩逻辑
        compressed = my_compress_function(tensor)

        metadata = CompressionMetadata(
            compression_type='my_custom',
            original_shape=tensor.shape,
            original_dtype=tensor.dtype,
            compression_ratio=tensor.numel() / compressed.numel()
        )

        return compressed, metadata

    def decompress(
        self,
        compressed: torch.Tensor,
        metadata: CompressionMetadata
    ) -> torch.Tensor:
        """解压缩张量"""

        # 自定义解压缩逻辑
        decompressed = my_decompress_function(compressed, metadata)

        return decompressed

    def get_compression_ratio(self) -> float:
        """获取压缩比"""
        return 4.0  # 例如，4x压缩

# 注册
register_compressor('my_custom', MyCustomCompressor())

# 使用
comm = AdaptiveCollective()
comm.all_reduce(tensor, compression='my_custom')
```

### 4.3 自定义ML模型集成

```python
from torch.adaptive_comm import register_ml_model, MLModel

class MyAlgorithmSelector(MLModel):
    """自定义算法选择模型"""

    def __init__(self):
        # 加载自定义模型
        self.model = torch.jit.load('my_model.pt')

    def predict(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """预测最优算法"""

        # 特征提取
        feature_vector = self.extract_features(features)

        # 推理
        with torch.no_grad():
            logits = self.model(feature_vector)
            algorithm_id = torch.argmax(logits).item()

        algorithms = ['ring', 'tree', 'double_binary_tree', 'hierarchical']

        return {
            'algorithm': algorithms[algorithm_id],
            'confidence': torch.softmax(logits, dim=0)[algorithm_id].item()
        }

    def extract_features(self, features: Dict[str, Any]) -> torch.Tensor:
        """提取特征向量"""
        # ...
        return feature_vector

# 注册
register_ml_model('algorithm_selector', MyAlgorithmSelector())
```

---

## 5. 与torch.distributed集成

### 5.1 透明拦截机制

AGCCS通过monkey-patching的方式拦截`torch.distributed`的API：

```python
# 内部实现（用户无需关心）

import torch.distributed as dist

_original_all_reduce = dist.all_reduce
_original_all_gather = dist.all_gather
_original_reduce_scatter = dist.reduce_scatter

def _agccs_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    """拦截all_reduce"""

    if not _agccs_enabled:
        # 未启用，使用原始实现
        return _original_all_reduce(tensor, op, group, async_op)

    # 使用AGCCS优化
    return _agccs_instance.all_reduce(tensor, op, group, async_op)

# Monkey-patch
dist.all_reduce = _agccs_all_reduce
dist.all_gather = _agccs_all_gather
dist.reduce_scatter = _agccs_reduce_scatter
```

用户代码无需修改：

```python
import torch
import torch.distributed as dist

# 初始化
dist.init_process_group('nccl')

# 启用AGCCS
from torch.adaptive_comm import enable_adaptive_comm
enable_adaptive_comm()

# 正常使用torch.distributed API
tensor = torch.randn(1000, 1000).cuda()
dist.all_reduce(tensor)  # 自动使用AGCCS优化

# 禁用AGCCS（回退到原始NCCL）
from torch.adaptive_comm import disable_adaptive_comm
disable_adaptive_comm()

dist.all_reduce(tensor)  # 使用原始NCCL
```

### 5.2 ProcessGroup集成

```python
from torch.adaptive_comm import AdaptiveProcessGroup

# 创建自适应ProcessGroup
world_size = dist.get_world_size()
rank = dist.get_rank()

adaptive_pg = AdaptiveProcessGroup(
    world_size=world_size,
    rank=rank,
    backend='nccl',
    config=AdaptiveCommConfig()
)

# 使用自定义ProcessGroup
dist.all_reduce(tensor, group=adaptive_pg)

# 与DDP集成
model = MyModel().cuda()
ddp_model = torch.nn.parallel.DistributedDataParallel(
    model,
    process_group=adaptive_pg
)
```

---

## 6. 完整示例

### 6.1 基础分布式训练

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.adaptive_comm import enable_adaptive_comm, ConfigPresets

def main():
    # 初始化进程组
    dist.init_process_group(backend='nccl')

    # 启用自适应通讯（使用分布式训练预设）
    enable_adaptive_comm(ConfigPresets.distributed_training())

    # 创建模型
    model = MyModel().cuda()
    ddp_model = DDP(model)

    # 训练
    optimizer = torch.optim.Adam(ddp_model.parameters())

    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.cuda(), labels.cuda()

            # 前向
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)

            # 反向（AllReduce自动使用AGCCS）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 清理
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

### 6.2 高级配置与监控

```python
import torch
import torch.distributed as dist
from torch.adaptive_comm import (
    enable_adaptive_comm,
    AdaptiveCommConfig,
    CommunicationMonitor,
    get_comm_stats
)

def main():
    # 初始化
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()

    # 自定义配置
    config = AdaptiveCommConfig(
        algorithm_selection='ml',
        compression_enabled=True,
        compression_selection='ml',
        overlap_enabled=True,
        bucket_size_mb=25.0,
        dynamic_bucketing=True,
        hierarchical_enabled=True,
        ml_online_learning=True,
        monitoring_enabled=True,
        profiling_enabled=True
    )

    enable_adaptive_comm(config)

    # 创建监控器
    monitor = CommunicationMonitor(sample_rate=0.1)

    # 训练
    model = MyModel().cuda()
    ddp_model = DDP(model)
    optimizer = torch.optim.Adam(ddp_model.parameters())

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        for step, batch in enumerate(dataloader):
            with monitor.start_recording():
                # 训练步骤
                inputs, labels = batch
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 每100步打印统计
            if step % 100 == 0 and rank == 0:
                step_stats = monitor.get_last_step_stats()
                print(f"Epoch {epoch}, Step {step}:")
                print(f"  Comm time: {step_stats.total_comm_time_ms:.2f} ms")
                print(f"  Avg bandwidth: {step_stats.avg_bandwidth_gbps:.2f} GB/s")

                # 检查异常
                if step_stats.avg_latency_ms > 50.0:
                    print(f"  Warning: High latency: {step_stats.avg_latency_ms:.2f} ms")

        # Epoch结束，打印总结
        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            stats = get_comm_stats()

            print(f"\nEpoch {epoch} Summary:")
            print(f"  Total time: {epoch_time:.2f} s")
            print(f"  Total comms: {stats.total_count}")
            print(f"  Total data: {stats.total_bytes / 1e9:.2f} GB")
            print(f"  Avg bandwidth: {stats.avg_bandwidth_gbps:.2f} GB/s")
            print(f"  P99 latency: {stats.p99_latency_ms:.2f} ms")

            # 算法使用统计
            algo_usage = stats.get_algorithm_usage()
            print(f"  Algorithm usage:")
            for algo, count in algo_usage.items():
                print(f"    {algo}: {count} ({count/stats.total_count*100:.1f}%)")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

### 6.3 FSDP集成示例

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.adaptive_comm import enable_adaptive_comm, AdaptiveCommConfig

def main():
    # 初始化
    dist.init_process_group(backend='nccl')

    # 为FSDP定制配置
    config = AdaptiveCommConfig(
        algorithm_selection='ml',
        # FSDP有大量小消息，启用消息聚合
        message_coalescing_enabled=True,
        coalescing_buffer_size_mb=20.0,
        # FSDP的all-gather和reduce-scatter优化
        compression_enabled=True,
        compression_selection='ml',
        # Overlap
        overlap_enabled=True,
        bucket_size_mb=50.0  # FSDP建议更大的bucket
    )

    enable_adaptive_comm(config)

    # 创建FSDP模型
    model = MyLargeModel()

    fsdp_model = FSDP(
        model,
        # FSDP配置...
    )

    # 训练
    optimizer = torch.optim.Adam(fsdp_model.parameters())

    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = fsdp_model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

### 6.4 异构集群示例

```python
from torch.adaptive_comm import (
    enable_adaptive_comm,
    AdaptiveCommConfig,
    TopologyDiscovery
)

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()

    # 发现拓扑
    topology = TopologyDiscovery.discover()

    # 检查是否异构
    is_heterogeneous = topology.check_heterogeneity()

    if is_heterogeneous:
        print(f"Rank {rank}: Detected heterogeneous cluster")

        # 为异构集群定制配置
        config = AdaptiveCommConfig(
            algorithm_selection='ml',  # ML模型能更好处理异构
            hierarchical_enabled=True,  # 层级化算法适合异构
            load_balancing_enabled=True,  # 负载均衡
            straggler_detection_enabled=True,
            work_stealing_enabled=True,
            routing_objective='balanced'  # 平衡路由
        )
    else:
        # 同构集群，使用标准配置
        config = ConfigPresets.distributed_training()

    enable_adaptive_comm(config)

    # 训练...

if __name__ == '__main__':
    main()
```

### 6.5 多节点大规模训练

```python
from torch.adaptive_comm import (
    enable_adaptive_comm,
    AdaptiveCommConfig,
    diagnose_communication
)

def main():
    # 初始化（假设8节点，每节点8GPU）
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size} initializing...")

    # 大规模训练配置
    config = AdaptiveCommConfig(
        algorithm_selection='ml',

        # 使用层级化算法（节点内+节点间）
        hierarchical_enabled=True,

        # 压缩（节点间通讯）
        compression_enabled=True,
        compression_selection='ml',

        # 大bucket（减少通讯次数）
        bucket_size_mb=100.0,

        # Overlap
        overlap_enabled=True,
        pipeline_depth=8,

        # 负载均衡（大规模重要）
        load_balancing_enabled=True,
        straggler_detection_enabled=True,

        # 监控
        monitoring_enabled=True,
        profiling_enabled=(rank == 0)  # 仅rank 0 profile
    )

    enable_adaptive_comm(config)

    # 模型
    model = MyModel().cuda()
    ddp_model = DDP(model)

    # 训练前先诊断
    if rank == 0:
        print("Running communication diagnosis...")
        diagnosis = diagnose_communication(duration_sec=30.0)

        if diagnosis.has_performance_issues:
            print("Performance issues detected:")
            for issue in diagnosis.performance_issues:
                print(f"  - {issue.description}")

    # 训练
    optimizer = torch.optim.Adam(ddp_model.parameters())

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每1000步检查一次
            if step % 1000 == 0 and rank == 0:
                stats = get_comm_stats()
                print(f"Step {step}: {stats.avg_bandwidth_gbps:.2f} GB/s, "
                      f"{stats.p99_latency_ms:.2f} ms p99")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

---

## 7. 调试与Troubleshooting

### 7.1 调试模式

```python
from torch.adaptive_comm import enable_adaptive_comm, AdaptiveCommConfig

# 启用调试模式
config = AdaptiveCommConfig(
    debug_mode=True,      # 启用额外检查
    verbose=True,         # 打印详细日志
    trace_enabled=True,   # 启用trace
    trace_output_path='agccs_trace.json'
)

enable_adaptive_comm(config)

# 训练代码...
# 会打印详细的决策日志，例如：
# [AGCCS] AllReduce: size=100MB, algorithm=ring (ML confidence=0.95)
# [AGCCS] Compression: selected=fp16 (ML confidence=0.88)
# [AGCCS] Estimated time: 15.2ms, Actual time: 14.8ms (error: -2.6%)
```

### 7.2 性能对比

```python
from torch.adaptive_comm import benchmark_algorithms

# 对比不同算法的性能
results = benchmark_algorithms(
    tensor_sizes=[1024**2, 10*1024**2, 100*1024**2],  # 1MB, 10MB, 100MB
    algorithms=['ring', 'tree', 'double_binary_tree', 'hierarchical'],
    num_iterations=100
)

# 打印结果
for size, algo_results in results.items():
    print(f"\nMessage size: {size / 1024**2:.0f} MB")
    for algo, stats in algo_results.items():
        print(f"  {algo}: {stats.avg_time_ms:.2f} ms ± {stats.std_time_ms:.2f} ms")
```

### 7.3 Fallback测试

```python
from torch.adaptive_comm import enable_adaptive_comm, AdaptiveCommConfig

# 启用fallback
config = AdaptiveCommConfig(
    fallback_on_error=True,
    verbose=True
)

enable_adaptive_comm(config)

# 如果AGCCS遇到错误，会自动回退到NCCL并打印警告
tensor = torch.randn(1000, 1000).cuda()
dist.all_reduce(tensor)
# 如果出错，会看到：
# [AGCCS Warning] Error in AGCCS, falling back to NCCL: <error message>
```

---

## 总结

本文档提供了GPU集群通讯优化系统的完整API设计，包括：

1. **用户API**：简单易用的启用接口、显式API、上下文管理器、装饰器
2. **配置系统**：丰富的配置选项、预设配置、配置加载/保存
3. **监控API**：实时统计、profiling、诊断工具
4. **扩展接口**：自定义算法、压缩器、ML模型
5. **torch.distributed集成**：透明拦截、ProcessGroup集成
6. **完整示例**：涵盖DDP、FSDP、异构集群、大规模训练等场景

系统设计兼顾易用性（一行启用）和灵活性（细粒度控制），适合从入门到高级的各种用户。

下一步将在05_DEPLOYMENT_STRATEGY.md中详细描述部署策略和生产环境考虑。

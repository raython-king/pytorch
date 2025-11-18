"""
快速推理引擎 - 优化的ML模型推理

本模块提供：
1. 低延迟推理（< 1ms）
2. 批量推理优化
3. 模型缓存和预加载
4. JIT编译加速
5. 特征缓存
"""

import torch
import torch.nn as nn
from torch.jit import ScriptModule, script_method
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import time
import numpy as np
from pathlib import Path

from .performance_predictor import (
    CommunicationTimePredictor,
    BandwidthPredictor,
    CongestionPredictor
)
from .algorithm_selector import (
    AlgorithmSelectorModel,
    ParameterOptimizer,
    AlgorithmConfig,
    CommunicationAlgorithm
)
from .rl_optimizer import CommunicationRLAgent, State, Action


class LRUCache:
    """LRU缓存实现"""

    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self.cache:
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        """存储缓存值"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                # Remove oldest item
                self.cache.popitem(last=False)

    def clear(self):
        """清空缓存"""
        self.cache.clear()


class BatchQueue:
    """批量推理队列"""

    def __init__(self, max_batch_size: int = 32, timeout_ms: float = 1.0):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.queue = []
        self.last_flush_time = time.time()

    def add(self, features: torch.Tensor, callback: callable):
        """添加推理请求"""
        self.queue.append((features, callback))

    def should_flush(self) -> bool:
        """判断是否应该执行批量推理"""
        if len(self.queue) >= self.max_batch_size:
            return True

        elapsed_ms = (time.time() - self.last_flush_time) * 1000
        if len(self.queue) > 0 and elapsed_ms >= self.timeout_ms:
            return True

        return False

    def flush(self) -> Tuple[torch.Tensor, List[callable]]:
        """获取批量数据并清空队列"""
        if not self.queue:
            return None, []

        features_list = [f for f, _ in self.queue]
        callbacks = [c for _, c in self.queue]

        batch_features = torch.stack(features_list)

        self.queue.clear()
        self.last_flush_time = time.time()

        return batch_features, callbacks


class FastInference:
    """
    快速推理引擎

    提供低延迟、高吞吐的模型推理
    """

    def __init__(
        self,
        model_dir: str = './checkpoints',
        device: str = 'cuda',
        use_jit: bool = True,
        cache_size: int = 1000,
        batch_size: int = 32
    ):
        self.device = device
        self.use_jit = use_jit
        self.model_dir = Path(model_dir)

        # 缓存
        self.feature_cache = LRUCache(capacity=cache_size)
        self.prediction_cache = LRUCache(capacity=cache_size)

        # 批量推理
        self.batch_queue = BatchQueue(max_batch_size=batch_size)

        # 加载模型
        self.models = {}
        self.load_all_models()

        # 性能统计
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0

    def load_all_models(self):
        """加载所有模型"""
        # 时间预测器
        try:
            time_predictor = CommunicationTimePredictor().to(self.device)
            self.load_model(time_predictor, 'time_predictor_best.pt')
            if self.use_jit:
                time_predictor = self.jit_compile(time_predictor)
            time_predictor.eval()
            self.models['time_predictor'] = time_predictor
        except Exception as e:
            print(f"Failed to load time predictor: {e}")

        # 带宽预测器
        try:
            bandwidth_predictor = BandwidthPredictor().to(self.device)
            self.load_model(bandwidth_predictor, 'bandwidth_predictor_best.pt')
            if self.use_jit:
                bandwidth_predictor = self.jit_compile(bandwidth_predictor)
            bandwidth_predictor.eval()
            self.models['bandwidth_predictor'] = bandwidth_predictor
        except Exception as e:
            print(f"Failed to load bandwidth predictor: {e}")

        # 算法选择器
        try:
            algorithm_selector = AlgorithmSelectorModel().to(self.device)
            self.load_model(algorithm_selector, 'algorithm_selector_best.pt')
            if self.use_jit:
                algorithm_selector = self.jit_compile(algorithm_selector)
            algorithm_selector.eval()
            self.models['algorithm_selector'] = algorithm_selector
        except Exception as e:
            print(f"Failed to load algorithm selector: {e}")

        # 参数优化器
        try:
            parameter_optimizer = ParameterOptimizer().to(self.device)
            self.load_model(parameter_optimizer, 'parameter_optimizer_best.pt')
            parameter_optimizer.eval()
            self.models['parameter_optimizer'] = parameter_optimizer
        except Exception as e:
            print(f"Failed to load parameter optimizer: {e}")

        # RL Agent
        try:
            rl_agent = CommunicationRLAgent(device=self.device)
            rl_agent.load(str(self.model_dir / 'rl_agent_final.pt'))
            self.models['rl_agent'] = rl_agent
        except Exception as e:
            print(f"Failed to load RL agent: {e}")

    def load_model(self, model: nn.Module, filename: str):
        """加载模型权重"""
        path = self.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

    def jit_compile(self, model: nn.Module) -> nn.Module:
        """JIT编译模型"""
        try:
            # 创建示例输入
            example_input = torch.randn(1, 120).to(self.device)
            traced_model = torch.jit.trace(model, example_input)
            return traced_model
        except Exception as e:
            print(f"JIT compilation failed: {e}, using eager mode")
            return model

    @torch.jit.export
    def predict_time(
        self,
        features: torch.Tensor,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        预测通讯时间

        Args:
            features: 输入特征 [batch_size, feature_dim]
            use_cache: 是否使用缓存

        Returns:
            predicted_time: 预测时间 [batch_size, 1]
        """
        if 'time_predictor' not in self.models:
            raise RuntimeError("Time predictor not loaded")

        start_time = time.time()

        # 检查缓存
        if use_cache:
            cache_key = self._compute_cache_key(features)
            cached_result = self.prediction_cache.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                return cached_result

            self.cache_misses += 1

        # 推理
        features = features.to(self.device)
        with torch.no_grad():
            prediction = self.models['time_predictor'](features)

        # 缓存结果
        if use_cache:
            self.prediction_cache.put(cache_key, prediction)

        # 记录推理时间
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)

        return prediction

    def predict_bandwidth(
        self,
        history: torch.Tensor,
        system_features: torch.Tensor
    ) -> torch.Tensor:
        """预测带宽"""
        if 'bandwidth_predictor' not in self.models:
            raise RuntimeError("Bandwidth predictor not loaded")

        history = history.to(self.device)
        system_features = system_features.to(self.device)

        with torch.no_grad():
            prediction = self.models['bandwidth_predictor'](history, system_features)

        return prediction

    def select_algorithm(
        self,
        features: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """
        选择最优算法

        Returns:
            algorithm_idx: 算法索引
            confidence: 置信度
        """
        if 'algorithm_selector' not in self.models:
            raise RuntimeError("Algorithm selector not loaded")

        features = features.to(self.device)

        with torch.no_grad():
            scores, confidence = self.models['algorithm_selector'](features)
            probabilities = torch.softmax(scores, dim=-1)
            algorithm_idx = torch.argmax(probabilities, dim=-1).item()

        return algorithm_idx, confidence

    def optimize_parameters(
        self,
        features: torch.Tensor,
        algorithm: int,
        message_size: int
    ) -> AlgorithmConfig:
        """
        优化算法参数

        Returns:
            config: 算法配置
        """
        if 'parameter_optimizer' not in self.models:
            raise RuntimeError("Parameter optimizer not loaded")

        features = features.to(self.device)
        algorithm_enum = CommunicationAlgorithm(algorithm)

        config = self.models['parameter_optimizer'].predict_optimal_params(
            features,
            algorithm_enum,
            message_size
        )

        return config

    def predict_batch(
        self,
        features_batch: torch.Tensor,
        model_name: str = 'time_predictor'
    ) -> torch.Tensor:
        """
        批量推理

        Args:
            features_batch: [batch_size, feature_dim]
            model_name: 模型名称

        Returns:
            predictions: [batch_size, output_dim]
        """
        if model_name not in self.models:
            raise RuntimeError(f"Model {model_name} not loaded")

        features_batch = features_batch.to(self.device)

        with torch.no_grad():
            predictions = self.models[model_name](features_batch)

        return predictions

    def _compute_cache_key(self, features: torch.Tensor) -> str:
        """计算特征的缓存键"""
        # 使用特征的哈希值作为键
        features_np = features.cpu().numpy()
        # 简化：只使用部分特征计算哈希
        key_features = features_np.flatten()[:10]
        cache_key = hash(tuple(key_features.tolist()))
        return str(cache_key)

    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            p50_time = np.percentile(self.inference_times, 50)
            p95_time = np.percentile(self.inference_times, 95)
            p99_time = np.percentile(self.inference_times, 99)
        else:
            avg_time = p50_time = p95_time = p99_time = 0.0

        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(total_requests, 1)

        return {
            'avg_inference_time_ms': float(avg_time),
            'p50_inference_time_ms': float(p50_time),
            'p95_inference_time_ms': float(p95_time),
            'p99_inference_time_ms': float(p99_time),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': float(cache_hit_rate),
            'total_inferences': len(self.inference_times),
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.inference_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self):
        """清空所有缓存"""
        self.feature_cache.clear()
        self.prediction_cache.clear()


class OptimizedInference:
    """
    进一步优化的推理引擎

    使用量化、剪枝等技术进一步加速
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_quantization: bool = False
    ):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        if use_quantization:
            self.quantize_model()

    def quantize_model(self):
        """量化模型（降低精度以加速）"""
        try:
            # Dynamic quantization
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=torch.qint8
            )
            print("Model quantized successfully")
        except Exception as e:
            print(f"Quantization failed: {e}")

    def prune_model(self, amount: float = 0.3):
        """剪枝模型（移除部分权重）"""
        try:
            import torch.nn.utils.prune as prune

            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=amount)

            print(f"Model pruned with amount {amount}")
        except Exception as e:
            print(f"Pruning failed: {e}")

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """优化的前向传播"""
        x = x.to(self.device)
        return self.model(x)


class AsyncInference:
    """
    异步推理引擎

    支持异步批量推理
    """

    def __init__(self, inference_engine: FastInference):
        self.engine = inference_engine
        self.pending_requests = []

    async def predict_async(
        self,
        features: torch.Tensor,
        model_name: str = 'time_predictor'
    ) -> torch.Tensor:
        """
        异步预测

        Args:
            features: 输入特征
            model_name: 模型名称

        Returns:
            prediction: 预测结果
        """
        # 简化实现：直接调用同步版本
        # 实际应该使用asyncio和线程池
        return self.engine.predict_batch(
            features.unsqueeze(0),
            model_name
        )[0]


# 便捷函数

def create_fast_inference(
    model_dir: str = './checkpoints',
    device: str = 'cuda',
    use_jit: bool = True
) -> FastInference:
    """创建快速推理引擎"""
    engine = FastInference(
        model_dir=model_dir,
        device=device,
        use_jit=use_jit
    )
    return engine


def benchmark_inference(
    inference_engine: FastInference,
    num_samples: int = 1000,
    feature_dim: int = 120
) -> Dict[str, float]:
    """
    基准测试推理性能

    Returns:
        benchmark_results: 性能指标
    """
    print(f"Running inference benchmark with {num_samples} samples...")

    # 生成随机特征
    features = torch.randn(num_samples, feature_dim)

    # 预热
    for _ in range(10):
        inference_engine.predict_time(features[:1], use_cache=False)

    # 重置统计
    inference_engine.reset_statistics()

    # 单样本推理测试
    start_time = time.time()
    for i in range(num_samples):
        inference_engine.predict_time(features[i:i+1], use_cache=False)
    single_time = (time.time() - start_time) * 1000  # ms

    # 批量推理测试
    start_time = time.time()
    inference_engine.predict_batch(features, 'time_predictor')
    batch_time = (time.time() - start_time) * 1000  # ms

    # 缓存测试
    inference_engine.reset_statistics()
    for _ in range(100):
        inference_engine.predict_time(features[0:1], use_cache=True)

    stats = inference_engine.get_statistics()

    results = {
        'single_inference_total_ms': single_time,
        'single_inference_avg_ms': single_time / num_samples,
        'batch_inference_total_ms': batch_time,
        'batch_inference_per_sample_ms': batch_time / num_samples,
        'speedup_factor': single_time / max(batch_time, 1e-6),
        'cache_hit_rate': stats['cache_hit_rate'],
    }

    print(f"Benchmark Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    return results

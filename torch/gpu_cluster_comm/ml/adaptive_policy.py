"""
自适应策略引擎 - ML驱动的通讯策略决策

本模块整合所有ML模型，提供端到端的智能通讯策略决策：
1. 特征提取
2. 性能预测
3. 算法选择
4. 参数优化
5. 在线学习
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque
import time
import numpy as np

from .features import CommunicationFeatureExtractor
from .performance_predictor import (
    CommunicationTimePredictor,
    BandwidthPredictor,
    CongestionPredictor
)
from .algorithm_selector import (
    AlgorithmSelectorModel,
    ParameterOptimizer,
    AlgorithmConfig,
    CommunicationAlgorithm,
    AdaptiveSelector
)
from .rl_optimizer import (
    CommunicationRLAgent,
    State,
    Action,
    compute_reward
)
from .inference import FastInference


@dataclass
class CommunicationContext:
    """通讯上下文"""
    message: torch.Tensor
    topology: Any
    workload_history: List[Dict]
    system_state: Dict
    message_size: int
    rank: int
    world_size: int
    operation_type: str  # 'allreduce', 'broadcast', 'allgather', etc.


@dataclass
class CommunicationStrategy:
    """通讯策略"""
    algorithm: CommunicationAlgorithm
    config: AlgorithmConfig
    predicted_time: float
    confidence: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm.name,
            'config': self.config.to_dict(),
            'predicted_time': self.predicted_time,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class ExecutionResult:
    """执行结果"""
    actual_time: float
    predicted_time: float
    bandwidth_utilization: float
    success: bool
    error_message: Optional[str] = None


class PerformanceTracker:
    """性能跟踪器"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

        # 统计指标
        self.total_predictions = 0
        self.correct_predictions = 0
        self.total_error = 0.0

    def record_execution(
        self,
        strategy: CommunicationStrategy,
        result: ExecutionResult
    ):
        """记录执行结果"""
        self.history.append({
            'strategy': strategy,
            'result': result,
            'timestamp': time.time()
        })

        self.total_predictions += 1

        # 计算预测误差
        error = abs(result.actual_time - result.predicted_time)
        self.total_error += error

        # 判断预测是否准确（误差 < 20%）
        if error / max(result.actual_time, 1e-6) < 0.2:
            self.correct_predictions += 1

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.total_predictions == 0:
            return {}

        accuracy = self.correct_predictions / self.total_predictions
        avg_error = self.total_error / self.total_predictions

        # 最近N次的统计
        recent_history = list(self.history)[-100:]
        if recent_history:
            recent_errors = [
                abs(h['result'].actual_time - h['strategy'].predicted_time)
                for h in recent_history
            ]
            recent_avg_error = np.mean(recent_errors)
            recent_std_error = np.std(recent_errors)
        else:
            recent_avg_error = 0.0
            recent_std_error = 0.0

        return {
            'total_predictions': self.total_predictions,
            'accuracy': float(accuracy),
            'avg_error_ms': float(avg_error),
            'recent_avg_error_ms': float(recent_avg_error),
            'recent_std_error_ms': float(recent_std_error),
        }

    def get_recent_executions(self, n: int = 10) -> List[Dict]:
        """获取最近N次执行"""
        return list(self.history)[-n:]


class AdaptiveStrategy:
    """
    自适应策略引擎

    整合所有ML模型，提供智能通讯策略决策
    """

    def __init__(
        self,
        model_dir: str = './checkpoints',
        device: str = 'cuda',
        use_rl: bool = False,
        enable_online_learning: bool = True
    ):
        self.device = device
        self.use_rl = use_rl
        self.enable_online_learning = enable_online_learning

        # 特征提取器
        self.feature_extractor = CommunicationFeatureExtractor()

        # 快速推理引擎
        self.inference_engine = FastInference(
            model_dir=model_dir,
            device=device,
            use_jit=True
        )

        # RL智能体（如果启用）
        self.rl_agent = None
        if use_rl and 'rl_agent' in self.inference_engine.models:
            self.rl_agent = self.inference_engine.models['rl_agent']

        # 性能跟踪器
        self.performance_tracker = PerformanceTracker()

        # 策略历史
        self.policy_history = deque(maxlen=1000)

        # 在线学习缓冲区
        self.online_learning_buffer = []

    def decide_communication_strategy(
        self,
        context: CommunicationContext,
        explore: bool = False
    ) -> CommunicationStrategy:
        """
        决定通讯策略

        Args:
            context: 通讯上下文
            explore: 是否探索新策略

        Returns:
            strategy: 推荐的通讯策略
        """
        # 1. 提取特征
        features = self._extract_features(context)

        # 2. 如果使用RL，让RL agent做决策
        if self.use_rl and self.rl_agent is not None:
            strategy = self._decide_with_rl(context, features, explore)
        else:
            # 使用监督学习模型
            strategy = self._decide_with_ml(context, features)

        # 3. 记录策略
        self.policy_history.append({
            'context': context,
            'strategy': strategy,
            'timestamp': time.time()
        })

        return strategy

    def _extract_features(self, context: CommunicationContext) -> torch.Tensor:
        """提取完整特征"""
        # 消息特征
        msg_features = self.feature_extractor.extract_message_features(
            context.message
        )

        # 拓扑特征
        topo_features = self.feature_extractor.extract_topology_features(
            context.topology
        )

        # 工作负载特征
        workload_features = self.feature_extractor.extract_workload_features(
            context.workload_history
        )

        # 系统特征
        system_features = self.feature_extractor.extract_system_features()

        # 组合所有特征
        features = torch.cat([
            msg_features,
            topo_features,
            workload_features,
            system_features
        ])

        return features

    def _decide_with_ml(
        self,
        context: CommunicationContext,
        features: torch.Tensor
    ) -> CommunicationStrategy:
        """使用监督学习模型做决策"""
        features_batch = features.unsqueeze(0)

        # 预测时间
        predicted_time = self.inference_engine.predict_time(
            features_batch,
            use_cache=True
        ).item()

        # 选择算法
        algorithm_idx, confidence = self.inference_engine.select_algorithm(
            features_batch
        )
        algorithm = CommunicationAlgorithm(algorithm_idx)

        # 优化参数
        config = self.inference_engine.optimize_parameters(
            features_batch,
            algorithm_idx,
            context.message_size
        )

        strategy = CommunicationStrategy(
            algorithm=algorithm,
            config=config,
            predicted_time=predicted_time,
            confidence=confidence.item(),
            metadata={
                'decision_method': 'ml',
                'features_shape': features.shape,
            }
        )

        return strategy

    def _decide_with_rl(
        self,
        context: CommunicationContext,
        features: torch.Tensor,
        explore: bool
    ) -> CommunicationStrategy:
        """使用RL智能体做决策"""
        # 构建RL状态
        state = State(
            topology_state=features[32:80],  # 假设维度
            workload_state=features[80:120],
            system_state=features[120:144],
            message_features=features[:32],
            pending_ops_count=len(context.workload_history),
            current_congestion=0.0,  # 从系统状态获取
            available_bandwidth=100.0  # 从系统状态获取
        )

        # RL agent选择动作
        action, _ = self.rl_agent.select_action(state, deterministic=not explore)

        # 转换为AlgorithmConfig
        algorithm = CommunicationAlgorithm(action.algorithm_choice)
        chunk_size = int(action.chunk_size_ratio * context.message_size)

        config = AlgorithmConfig(
            algorithm=algorithm,
            chunk_size=chunk_size,
            num_chunks=max(1, context.message_size // chunk_size),
            pipeline_depth=action.pipeline_depth,
            enable_compression=action.enable_compression,
            compression_ratio=action.compression_ratio,
            enable_overlap=action.enable_overlap,
            priority=action.priority
        )

        # 预测时间（用于参考）
        predicted_time = self.inference_engine.predict_time(
            features.unsqueeze(0),
            use_cache=True
        ).item()

        strategy = CommunicationStrategy(
            algorithm=algorithm,
            config=config,
            predicted_time=predicted_time,
            confidence=0.95,  # RL通常更自信
            metadata={
                'decision_method': 'rl',
                'action': asdict(action),
            }
        )

        return strategy

    def learn_from_execution(
        self,
        strategy: CommunicationStrategy,
        result: ExecutionResult
    ):
        """
        从执行结果中学习

        Args:
            strategy: 使用的策略
            result: 执行结果
        """
        # 记录到性能跟踪器
        self.performance_tracker.record_execution(strategy, result)

        # 如果启用在线学习
        if self.enable_online_learning:
            self._update_online_learning(strategy, result)

        # 如果使用RL，更新RL agent
        if self.use_rl and self.rl_agent is not None:
            self._update_rl_agent(strategy, result)

    def _update_online_learning(
        self,
        strategy: CommunicationStrategy,
        result: ExecutionResult
    ):
        """在线学习更新"""
        # 添加到缓冲区
        self.online_learning_buffer.append({
            'strategy': strategy,
            'result': result,
            'timestamp': time.time()
        })

        # 当缓冲区足够大时，进行增量更新
        if len(self.online_learning_buffer) >= 100:
            self._incremental_model_update()
            self.online_learning_buffer.clear()

    def _incremental_model_update(self):
        """增量模型更新"""
        # 简化实现：实际应该更新模型权重
        # 这里只做统计
        print(f"Incremental update with {len(self.online_learning_buffer)} samples")

    def _update_rl_agent(
        self,
        strategy: CommunicationStrategy,
        result: ExecutionResult
    ):
        """更新RL智能体"""
        # 计算奖励
        reward = compute_reward(
            action=strategy.metadata.get('action'),
            actual_time=result.actual_time,
            predicted_time=result.predicted_time,
            bandwidth_utilization=result.bandwidth_utilization,
            congestion_penalty=0.0
        )

        # 存储到RL agent的经验回放（简化）
        # 实际应该构建完整的transition并调用store_transition
        print(f"RL reward: {reward:.4f}")

    def evaluate_strategy(
        self,
        context: CommunicationContext,
        strategy: CommunicationStrategy
    ) -> float:
        """
        评估策略的预期性能

        Returns:
            estimated_performance: 预期性能得分（越低越好）
        """
        # 使用时间预测作为性能估计
        return strategy.predicted_time

    def compare_strategies(
        self,
        context: CommunicationContext,
        strategies: List[CommunicationStrategy]
    ) -> CommunicationStrategy:
        """
        比较多个策略，选择最优

        Returns:
            best_strategy: 最优策略
        """
        if not strategies:
            raise ValueError("No strategies to compare")

        # 评估每个策略
        scores = []
        for strategy in strategies:
            score = self.evaluate_strategy(context, strategy)
            scores.append(score)

        # 选择最低得分（最快）
        best_idx = np.argmin(scores)
        return strategies[best_idx]

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        # 性能跟踪统计
        perf_stats = self.performance_tracker.get_statistics()

        # 推理引擎统计
        inference_stats = self.inference_engine.get_statistics()

        # 策略分布统计
        algorithm_counts = {}
        for record in self.policy_history:
            algo_name = record['strategy'].algorithm.name
            algorithm_counts[algo_name] = algorithm_counts.get(algo_name, 0) + 1

        report = {
            'performance_statistics': perf_stats,
            'inference_statistics': inference_stats,
            'algorithm_distribution': algorithm_counts,
            'total_decisions': len(self.policy_history),
            'use_rl': self.use_rl,
            'online_learning_enabled': self.enable_online_learning,
        }

        return report

    def reset_statistics(self):
        """重置所有统计信息"""
        self.performance_tracker = PerformanceTracker()
        self.inference_engine.reset_statistics()
        self.policy_history.clear()

    def save_state(self, path: str):
        """保存策略引擎状态"""
        state = {
            'policy_history': list(self.policy_history),
            'performance_stats': self.performance_tracker.get_statistics(),
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """加载策略引擎状态"""
        state = torch.load(path)
        # 恢复部分状态
        print(f"Loaded state from {path}")


class PolicyOptimizer:
    """
    策略优化器

    使用历史数据优化策略选择
    """

    def __init__(self, adaptive_strategy: AdaptiveStrategy):
        self.adaptive_strategy = adaptive_strategy

    def optimize_for_workload(
        self,
        workload_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        为特定工作负载优化策略

        Args:
            workload_profile: 工作负载特征

        Returns:
            optimization_results: 优化结果
        """
        # 分析历史执行
        history = self.adaptive_strategy.performance_tracker.get_recent_executions(100)

        # 找出表现最好的策略
        best_strategies = self._find_best_strategies(history)

        # 生成优化建议
        recommendations = self._generate_recommendations(best_strategies)

        return {
            'best_strategies': best_strategies,
            'recommendations': recommendations,
        }

    def _find_best_strategies(
        self,
        history: List[Dict]
    ) -> List[Dict[str, Any]]:
        """找出表现最好的策略"""
        if not history:
            return []

        # 按实际时间排序
        sorted_history = sorted(
            history,
            key=lambda x: x['result'].actual_time
        )

        # 返回前10个最快的
        best = []
        for h in sorted_history[:10]:
            best.append({
                'algorithm': h['strategy'].algorithm.name,
                'config': h['strategy'].config.to_dict(),
                'actual_time': h['result'].actual_time,
                'predicted_time': h['strategy'].predicted_time,
            })

        return best

    def _generate_recommendations(
        self,
        best_strategies: List[Dict]
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []

        if not best_strategies:
            return recommendations

        # 统计最常见的算法
        algo_counts = {}
        for s in best_strategies:
            algo = s['algorithm']
            algo_counts[algo] = algo_counts.get(algo, 0) + 1

        most_common_algo = max(algo_counts, key=algo_counts.get)
        recommendations.append(
            f"Consider using {most_common_algo} algorithm more frequently"
        )

        # 分析参数配置
        avg_chunk_size = np.mean([
            s['config']['chunk_size'] for s in best_strategies
        ])
        recommendations.append(
            f"Optimal chunk size appears to be around {avg_chunk_size:.0f} bytes"
        )

        return recommendations


# 便捷函数

def create_adaptive_strategy(
    model_dir: str = './checkpoints',
    device: str = 'cuda',
    use_rl: bool = False
) -> AdaptiveStrategy:
    """创建自适应策略引擎"""
    strategy = AdaptiveStrategy(
        model_dir=model_dir,
        device=device,
        use_rl=use_rl,
        enable_online_learning=True
    )
    return strategy


def create_policy_optimizer(
    adaptive_strategy: AdaptiveStrategy
) -> PolicyOptimizer:
    """创建策略优化器"""
    optimizer = PolicyOptimizer(adaptive_strategy)
    return optimizer

"""
GPU集群通讯ML优化模块

提供ML驱动的通讯性能预测和自适应策略选择。
"""

from .features import (
    CommunicationFeatureExtractor,
    MessageFeatures,
    TopologyFeatures,
    WorkloadFeatures,
    SystemFeatures,
)

from .performance_predictor import (
    CommunicationTimePredictor,
    BandwidthPredictor,
    CongestionPredictor,
    EnsemblePredictor,
    OnlinePredictor,
    PerformanceMetrics,
    create_time_predictor,
    create_bandwidth_predictor,
    create_congestion_predictor,
)

from .algorithm_selector import (
    AlgorithmSelectorModel,
    ParameterOptimizer,
    CostModel,
    AdaptiveSelector,
    CommunicationAlgorithm,
    AlgorithmConfig,
    create_algorithm_selector,
    create_parameter_optimizer,
    create_adaptive_selector,
)

from .rl_optimizer import (
    CommunicationRLAgent,
    State,
    Action,
    ActorNetwork,
    CriticNetwork,
    RolloutBuffer,
    create_rl_agent,
    compute_reward,
)

from .trainer import (
    MLTrainer,
    RLTrainer,
    TrainingConfig,
    CommunicationDataset,
    DataCollector,
)

from .inference import (
    FastInference,
    OptimizedInference,
    LRUCache,
    create_fast_inference,
    benchmark_inference,
)

from .adaptive_policy import (
    AdaptiveStrategy,
    CommunicationContext,
    CommunicationStrategy,
    ExecutionResult,
    PerformanceTracker,
    PolicyOptimizer,
    create_adaptive_strategy,
    create_policy_optimizer,
)


__all__ = [
    # 特征工程
    'CommunicationFeatureExtractor',
    'MessageFeatures',
    'TopologyFeatures',
    'WorkloadFeatures',
    'SystemFeatures',

    # 性能预测
    'CommunicationTimePredictor',
    'BandwidthPredictor',
    'CongestionPredictor',
    'EnsemblePredictor',
    'OnlinePredictor',
    'PerformanceMetrics',
    'create_time_predictor',
    'create_bandwidth_predictor',
    'create_congestion_predictor',

    # 算法选择
    'AlgorithmSelectorModel',
    'ParameterOptimizer',
    'CostModel',
    'AdaptiveSelector',
    'CommunicationAlgorithm',
    'AlgorithmConfig',
    'create_algorithm_selector',
    'create_parameter_optimizer',
    'create_adaptive_selector',

    # 强化学习
    'CommunicationRLAgent',
    'State',
    'Action',
    'ActorNetwork',
    'CriticNetwork',
    'RolloutBuffer',
    'create_rl_agent',
    'compute_reward',

    # 训练
    'MLTrainer',
    'RLTrainer',
    'TrainingConfig',
    'CommunicationDataset',
    'DataCollector',

    # 推理
    'FastInference',
    'OptimizedInference',
    'LRUCache',
    'create_fast_inference',
    'benchmark_inference',

    # 自适应策略
    'AdaptiveStrategy',
    'CommunicationContext',
    'CommunicationStrategy',
    'ExecutionResult',
    'PerformanceTracker',
    'PolicyOptimizer',
    'create_adaptive_strategy',
    'create_policy_optimizer',
]


__version__ = '0.1.0'

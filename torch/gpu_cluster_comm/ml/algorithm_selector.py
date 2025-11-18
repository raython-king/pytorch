"""
算法选择器 - ML驱动的通讯算法选择

本模块使用深度学习选择最优的通讯算法和参数配置。
支持的算法包括：Ring, Tree, DoubleBinaryTree, HalvingDoubling,
Rabenseifner, Hierarchical等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from dataclasses import dataclass


class CommunicationAlgorithm(Enum):
    """通讯算法枚举"""
    RING = 0
    TREE = 1
    DOUBLE_BINARY_TREE = 2
    HALVING_DOUBLING = 3
    RABENSEIFNER = 4
    HIERARCHICAL = 5


@dataclass
class AlgorithmConfig:
    """算法配置"""
    algorithm: CommunicationAlgorithm
    chunk_size: int
    num_chunks: int
    pipeline_depth: int
    enable_compression: bool
    compression_ratio: float
    enable_overlap: bool
    priority: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm.name,
            'chunk_size': self.chunk_size,
            'num_chunks': self.num_chunks,
            'pipeline_depth': self.pipeline_depth,
            'enable_compression': self.enable_compression,
            'compression_ratio': self.compression_ratio,
            'enable_overlap': self.enable_overlap,
            'priority': self.priority,
        }


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class AlgorithmSelectorModel(nn.Module):
    """
    算法选择模型

    使用Transformer架构对通讯上下文特征进行编码，
    然后预测每种算法的得分。
    """

    def __init__(
        self,
        input_dim: int = 120,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        num_algorithms: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_algorithms = num_algorithms

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

        # Algorithm classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_algorithms)
        )

        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch_size, input_dim] 或 [batch_size, seq_len, input_dim]

        Returns:
            scores: [batch_size, num_algorithms] 每种算法的得分
            confidence: [batch_size, 1] 预测置信度
        """
        # Project input
        if features.dim() == 2:
            features = features.unsqueeze(1)  # Add sequence dimension

        x = self.input_proj(features)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Pool sequence (mean pooling)
        x_pooled = x.mean(dim=1)

        # Predict algorithm scores
        scores = self.classifier(x_pooled)

        # Predict confidence
        confidence = self.confidence_predictor(x_pooled)

        return scores, confidence

    def predict_best_algorithm(
        self,
        features: torch.Tensor,
        return_probabilities: bool = False
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """
        预测最佳算法

        Args:
            features: 输入特征
            return_probabilities: 是否返回概率分布

        Returns:
            best_algorithm: 最佳算法的索引
            probabilities: (可选) 算法概率分布
        """
        self.eval()
        with torch.no_grad():
            scores, confidence = self.forward(features)
            probabilities = F.softmax(scores, dim=-1)
            best_algorithm = torch.argmax(probabilities, dim=-1).item()

        if return_probabilities:
            return best_algorithm, probabilities
        else:
            return best_algorithm, None

    def predict_top_k_algorithms(
        self,
        features: torch.Tensor,
        k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        预测top-k算法

        Returns:
            List of (algorithm_index, probability) tuples
        """
        self.eval()
        with torch.no_grad():
            scores, _ = self.forward(features)
            probabilities = F.softmax(scores, dim=-1)

            # Get top-k
            topk_probs, topk_indices = torch.topk(probabilities[0], k)

            results = [
                (idx.item(), prob.item())
                for idx, prob in zip(topk_indices, topk_probs)
            ]

        return results


class ParameterOptimizer(nn.Module):
    """
    参数优化器

    为选定的算法优化参数配置（chunk size, pipeline depth等）
    """

    def __init__(
        self,
        input_dim: int = 120,
        hidden_dim: int = 256,
        num_algorithms: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_algorithms = num_algorithms

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Algorithm embedding
        self.algorithm_embedding = nn.Embedding(num_algorithms, hidden_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Parameter prediction heads
        self.chunk_size_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive
        )

        self.num_chunks_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

        self.pipeline_depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

        self.compression_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # [enable, ratio]
        )

        self.overlap_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.priority_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

    def forward(
        self,
        features: torch.Tensor,
        algorithm: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, input_dim] 上下文特征
            algorithm: [batch_size] 选择的算法索引

        Returns:
            parameters: 预测的参数字典
        """
        # Encode features
        feat_encoded = self.feature_encoder(features)

        # Embed algorithm
        algo_embedded = self.algorithm_embedding(algorithm)

        # Fuse
        fused = torch.cat([feat_encoded, algo_embedded], dim=-1)
        fused = self.fusion(fused)

        # Predict parameters
        chunk_size = self.chunk_size_predictor(fused)
        num_chunks = self.num_chunks_predictor(fused)
        pipeline_depth = self.pipeline_depth_predictor(fused)

        compression_output = self.compression_predictor(fused)
        compression_enable = torch.sigmoid(compression_output[:, 0:1])
        compression_ratio = torch.sigmoid(compression_output[:, 1:2])

        enable_overlap = self.overlap_predictor(fused)
        priority = self.priority_predictor(fused)

        return {
            'chunk_size': chunk_size,
            'num_chunks': num_chunks,
            'pipeline_depth': pipeline_depth,
            'compression_enable': compression_enable,
            'compression_ratio': compression_ratio,
            'enable_overlap': enable_overlap,
            'priority': priority,
        }

    def predict_optimal_params(
        self,
        features: torch.Tensor,
        algorithm: CommunicationAlgorithm,
        message_size: int
    ) -> AlgorithmConfig:
        """
        预测最优参数配置

        Args:
            features: 上下文特征
            algorithm: 选择的算法
            message_size: 消息大小（字节）

        Returns:
            AlgorithmConfig对象
        """
        self.eval()
        with torch.no_grad():
            algo_tensor = torch.tensor([algorithm.value], dtype=torch.long)
            params = self.forward(features, algo_tensor)

            # Convert to concrete values
            chunk_size = self._compute_chunk_size(
                params['chunk_size'].item(),
                message_size
            )
            num_chunks = max(1, int(params['num_chunks'].item()))
            pipeline_depth = max(1, min(8, int(params['pipeline_depth'].item())))

            enable_compression = params['compression_enable'].item() > 0.5
            compression_ratio = params['compression_ratio'].item()

            enable_overlap = params['enable_overlap'].item() > 0.5
            priority = max(0, min(10, int(params['priority'].item())))

            config = AlgorithmConfig(
                algorithm=algorithm,
                chunk_size=chunk_size,
                num_chunks=num_chunks,
                pipeline_depth=pipeline_depth,
                enable_compression=enable_compression,
                compression_ratio=compression_ratio,
                enable_overlap=enable_overlap,
                priority=priority
            )

        return config

    def _compute_chunk_size(self, predicted_ratio: float, message_size: int) -> int:
        """
        计算chunk大小

        Args:
            predicted_ratio: 预测的chunk size比例 (0-1)
            message_size: 总消息大小

        Returns:
            chunk_size: 实际chunk大小（字节）
        """
        # Map ratio to reasonable chunk size
        # Use logarithmic scale for better distribution
        min_chunk = 1024  # 1KB
        max_chunk = min(message_size, 128 * 1024 * 1024)  # 128MB

        log_min = np.log10(min_chunk)
        log_max = np.log10(max_chunk)

        log_chunk = log_min + predicted_ratio * (log_max - log_min)
        chunk_size = int(10 ** log_chunk)

        # Align to cache line (64 bytes)
        chunk_size = (chunk_size // 64) * 64

        return max(min_chunk, min(chunk_size, max_chunk))


class CostModel(nn.Module):
    """
    成本模型 - 预测算法+参数配置的执行成本

    用于辅助算法选择和参数优化
    """

    def __init__(
        self,
        input_dim: int = 120,
        param_dim: int = 7,  # chunk_size, num_chunks, etc.
        hidden_dim: int = 256,
        num_algorithms: int = 6
    ):
        super().__init__()

        self.algorithm_embedding = nn.Embedding(num_algorithms, hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + param_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.cost_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Positive cost
        )

    def forward(
        self,
        features: torch.Tensor,
        algorithm: torch.Tensor,
        parameters: torch.Tensor
    ) -> torch.Tensor:
        """
        预测执行成本（时间）

        Args:
            features: [batch_size, input_dim] 上下文特征
            algorithm: [batch_size] 算法索引
            parameters: [batch_size, param_dim] 参数向量

        Returns:
            cost: [batch_size, 1] 预测成本（毫秒）
        """
        # Embed algorithm
        algo_embedded = self.algorithm_embedding(algorithm)

        # Concatenate all inputs
        combined = torch.cat([features, parameters, algo_embedded], dim=-1)

        # Encode
        encoded = self.encoder(combined)

        # Predict cost
        cost = self.cost_predictor(encoded)

        return cost


class AdaptiveSelector:
    """
    自适应选择器 - 组合算法选择和参数优化

    提供端到端的算法和参数选择流程
    """

    def __init__(
        self,
        algorithm_selector: AlgorithmSelectorModel,
        parameter_optimizer: ParameterOptimizer,
        cost_model: Optional[CostModel] = None,
        device: str = 'cuda'
    ):
        self.algorithm_selector = algorithm_selector.to(device)
        self.parameter_optimizer = parameter_optimizer.to(device)
        self.cost_model = cost_model.to(device) if cost_model else None
        self.device = device

        self.selection_history = []

    def select_strategy(
        self,
        features: torch.Tensor,
        message_size: int,
        explore: bool = False,
        epsilon: float = 0.1
    ) -> AlgorithmConfig:
        """
        选择通讯策略（算法+参数）

        Args:
            features: 上下文特征
            message_size: 消息大小
            explore: 是否探索（epsilon-greedy）
            epsilon: 探索概率

        Returns:
            AlgorithmConfig: 选择的策略配置
        """
        features = features.to(self.device)

        # Epsilon-greedy exploration
        if explore and np.random.rand() < epsilon:
            # Random selection
            algorithm_idx = np.random.randint(0, 6)
        else:
            # Greedy selection
            algorithm_idx, _ = self.algorithm_selector.predict_best_algorithm(features)

        algorithm = CommunicationAlgorithm(algorithm_idx)

        # Optimize parameters
        config = self.parameter_optimizer.predict_optimal_params(
            features,
            algorithm,
            message_size
        )

        # Record selection
        self.selection_history.append({
            'algorithm': algorithm.name,
            'config': config.to_dict(),
            'message_size': message_size
        })

        return config

    def select_top_k_strategies(
        self,
        features: torch.Tensor,
        message_size: int,
        k: int = 3
    ) -> List[Tuple[AlgorithmConfig, float]]:
        """
        选择top-k策略

        Returns:
            List of (config, probability) tuples
        """
        features = features.to(self.device)

        # Get top-k algorithms
        top_algorithms = self.algorithm_selector.predict_top_k_algorithms(features, k)

        strategies = []
        for algo_idx, prob in top_algorithms:
            algorithm = CommunicationAlgorithm(algo_idx)
            config = self.parameter_optimizer.predict_optimal_params(
                features,
                algorithm,
                message_size
            )
            strategies.append((config, prob))

        return strategies

    def evaluate_strategy(
        self,
        features: torch.Tensor,
        config: AlgorithmConfig
    ) -> float:
        """
        使用成本模型评估策略

        Returns:
            estimated_cost: 估计的执行时间（毫秒）
        """
        if self.cost_model is None:
            return 0.0

        features = features.to(self.device)

        # Convert config to parameter tensor
        params = self._config_to_tensor(config)

        # Predict cost
        algo_tensor = torch.tensor([config.algorithm.value], dtype=torch.long, device=self.device)
        cost = self.cost_model(features, algo_tensor, params)

        return cost.item()

    def _config_to_tensor(self, config: AlgorithmConfig) -> torch.Tensor:
        """转换AlgorithmConfig为参数张量"""
        params = [
            float(config.chunk_size) / 1e8,  # Normalize
            float(config.num_chunks) / 100,
            float(config.pipeline_depth) / 10,
            1.0 if config.enable_compression else 0.0,
            config.compression_ratio,
            1.0 if config.enable_overlap else 0.0,
            float(config.priority) / 10,
        ]
        return torch.tensor([params], dtype=torch.float32, device=self.device)

    def get_statistics(self) -> Dict[str, Any]:
        """获取选择统计信息"""
        if not self.selection_history:
            return {}

        algo_counts = {}
        for record in self.selection_history:
            algo = record['algorithm']
            algo_counts[algo] = algo_counts.get(algo, 0) + 1

        return {
            'total_selections': len(self.selection_history),
            'algorithm_distribution': algo_counts,
            'recent_selections': self.selection_history[-10:]
        }


# 便捷函数

def create_algorithm_selector(device: str = 'cuda') -> AlgorithmSelectorModel:
    """创建算法选择器"""
    model = AlgorithmSelectorModel()
    model = model.to(device)
    return model


def create_parameter_optimizer(device: str = 'cuda') -> ParameterOptimizer:
    """创建参数优化器"""
    model = ParameterOptimizer()
    model = model.to(device)
    return model


def create_cost_model(device: str = 'cuda') -> CostModel:
    """创建成本模型"""
    model = CostModel()
    model = model.to(device)
    return model


def create_adaptive_selector(device: str = 'cuda') -> AdaptiveSelector:
    """创建自适应选择器"""
    algorithm_selector = create_algorithm_selector(device)
    parameter_optimizer = create_parameter_optimizer(device)
    cost_model = create_cost_model(device)

    selector = AdaptiveSelector(
        algorithm_selector,
        parameter_optimizer,
        cost_model,
        device
    )

    return selector


def algorithm_name_to_enum(name: str) -> CommunicationAlgorithm:
    """算法名称转枚举"""
    name_map = {
        'ring': CommunicationAlgorithm.RING,
        'tree': CommunicationAlgorithm.TREE,
        'double_binary_tree': CommunicationAlgorithm.DOUBLE_BINARY_TREE,
        'halving_doubling': CommunicationAlgorithm.HALVING_DOUBLING,
        'rabenseifner': CommunicationAlgorithm.RABENSEIFNER,
        'hierarchical': CommunicationAlgorithm.HIERARCHICAL,
    }
    return name_map.get(name.lower(), CommunicationAlgorithm.RING)

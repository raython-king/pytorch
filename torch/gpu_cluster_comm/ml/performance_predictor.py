"""
性能预测模型 - ML驱动的通讯性能预测

本模块包含多个神经网络模型，用于预测：
1. 通讯操作耗时
2. 可用带宽
3. 网络拥塞

支持快速推理和在线学习。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
from collections import deque
import numpy as np


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, seq_len, self.d_model)

        return self.out_proj(out)


class CommunicationTimePredictor(nn.Module):
    """
    通讯时间预测器

    预测给定通讯操作的完成时间（毫秒）
    """

    def __init__(
        self,
        input_dim: int = 120,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Predictor head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # 输出激活（确保正值）
        self.output_activation = nn.Softplus()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, input_dim] 特征向量

        Returns:
            predicted_time: [batch_size, 1] 预测时间（毫秒）
        """
        x = self.input_proj(features)

        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Predict
        time = self.predictor(x)
        time = self.output_activation(time)

        return time

    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测时间并估计不确定性（使用dropout）

        Args:
            features: 输入特征
            num_samples: MC dropout采样次数

        Returns:
            mean: 平均预测时间
            std: 标准差（不确定性）
        """
        self.train()  # Enable dropout

        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(features)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        self.eval()

        return mean, std


class BandwidthPredictor(nn.Module):
    """
    带宽预测器 - 基于LSTM的时序模型

    预测未来N个时间步的可用带宽
    """

    def __init__(
        self,
        history_dim: int = 32,
        system_feature_dim: int = 24,
        hidden_dim: int = 128,
        num_layers: int = 3,
        prediction_horizon: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()

        self.history_dim = history_dim
        self.system_feature_dim = system_feature_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=history_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # System feature encoder
        self.system_encoder = nn.Sequential(
            nn.Linear(system_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention mechanism
        self.attention = MultiHeadAttention(hidden_dim, num_heads=8, dropout=dropout)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, prediction_horizon)
        )

        # 输出激活（确保正值）
        self.output_activation = nn.Softplus()

    def forward(
        self,
        history: torch.Tensor,
        system_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            history: [batch_size, seq_len, history_dim] 历史带宽序列
            system_features: [batch_size, system_feature_dim] 系统特征

        Returns:
            predictions: [batch_size, prediction_horizon] 未来带宽预测
        """
        batch_size = history.size(0)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(history)

        # Apply attention
        attn_out = self.attention(lstm_out)

        # Take last hidden state
        last_hidden = attn_out[:, -1, :]

        # Encode system features
        sys_encoded = self.system_encoder(system_features)

        # Fuse temporal and system features
        fused = torch.cat([last_hidden, sys_encoded], dim=-1)
        fused = self.fusion(fused)

        # Predict future bandwidth
        predictions = self.output_proj(fused)
        predictions = self.output_activation(predictions)

        return predictions

    def predict_single_step(
        self,
        history: torch.Tensor,
        system_features: torch.Tensor
    ) -> torch.Tensor:
        """预测下一个时间步的带宽"""
        full_predictions = self.forward(history, system_features)
        return full_predictions[:, 0:1]

    def predict_with_confidence(
        self,
        history: torch.Tensor,
        system_features: torch.Tensor,
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测带宽并返回置信区间

        Returns:
            mean: 平均预测
            lower: 置信下界
            upper: 置信上界
        """
        mean = self.forward(history, system_features)

        # 简化：使用固定的置信区间宽度
        # 实际应该基于训练数据的残差估计
        std = mean * 0.1  # 10% 标准差

        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99%
        margin = z_score * std

        lower = torch.clamp(mean - margin, min=0)
        upper = mean + margin

        return mean, lower, upper


class CongestionPredictor(nn.Module):
    """
    拥塞预测器 - 基于图神经网络

    预测网络拓扑中每条链路的拥塞概率
    """

    def __init__(
        self,
        node_feature_dim: int = 32,
        edge_feature_dim: int = 16,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # GAT layers (Graph Attention Networks)
        self.gat_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim * num_heads

            self.gat_layers.append(
                GATLayer(in_dim, hidden_dim, num_heads, dropout)
            )

        # Edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * num_heads * 2 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_features: [num_edges, edge_feature_dim]
            edge_index: [2, num_edges] 边的连接关系

        Returns:
            congestion_probs: [num_edges, 1] 拥塞概率
        """
        # Encode features
        node_h = self.node_encoder(node_features)
        edge_h = self.edge_encoder(edge_features)

        # Apply GAT layers
        for gat in self.gat_layers:
            node_h = gat(node_h, edge_index)

        # Predict edge congestion
        # For each edge, concatenate source node, target node, and edge features
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        src_features = node_h[src_nodes]
        dst_features = node_h[dst_nodes]

        # Concatenate [src, dst, edge]
        edge_input = torch.cat([src_features, dst_features, edge_h], dim=-1)

        # Predict congestion
        congestion_probs = self.edge_predictor(edge_input)

        return congestion_probs

    def predict_critical_links(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        threshold: float = 0.7
    ) -> List[int]:
        """
        预测关键拥塞链路

        Returns:
            critical_edges: 拥塞概率超过阈值的边索引列表
        """
        probs = self.forward(node_features, edge_features, edge_index)
        critical_edges = torch.where(probs.squeeze() > threshold)[0].tolist()
        return critical_edges


class GATLayer(nn.Module):
    """Graph Attention Layer"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # Linear transformations for each head
        self.linear_proj = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # Attention parameters
        self.attn_src = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.Tensor(1, num_heads, out_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, in_dim]
            edge_index: [2, num_edges]

        Returns:
            output: [num_nodes, out_dim * num_heads]
        """
        num_nodes = node_features.size(0)

        # Linear transformation
        h = self.linear_proj(node_features)  # [num_nodes, out_dim * num_heads]
        h = h.view(num_nodes, self.num_heads, self.out_dim)

        # Compute attention scores
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # [num_edges, num_heads]
        attn_src_scores = (h[src_nodes] * self.attn_src).sum(dim=-1)
        attn_dst_scores = (h[dst_nodes] * self.attn_dst).sum(dim=-1)

        attn_scores = attn_src_scores + attn_dst_scores
        attn_scores = self.leaky_relu(attn_scores)

        # Softmax over neighbors
        attn_weights = self._softmax_per_node(attn_scores, dst_nodes, num_nodes)
        attn_weights = self.dropout(attn_weights)

        # Aggregate
        output = torch.zeros(num_nodes, self.num_heads, self.out_dim, device=h.device)

        for i in range(self.num_heads):
            # Weighted sum of neighbors
            edge_msg = h[src_nodes, i, :] * attn_weights[:, i:i+1]

            # Scatter add
            output[:, i, :].scatter_add_(
                0,
                dst_nodes.unsqueeze(-1).expand(-1, self.out_dim),
                edge_msg
            )

        # Concatenate heads
        output = output.view(num_nodes, -1)

        return output

    def _softmax_per_node(
        self,
        scores: torch.Tensor,
        node_indices: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """对每个节点的邻居进行softmax"""
        # Compute max per node for numerical stability
        max_scores = torch.full(
            (num_nodes, scores.size(1)),
            float('-inf'),
            device=scores.device
        )
        max_scores.scatter_reduce_(
            0,
            node_indices.unsqueeze(-1).expand_as(scores),
            scores,
            reduce='amax',
            include_self=False
        )

        # Subtract max
        scores_normalized = scores - max_scores[node_indices]

        # Exp
        exp_scores = torch.exp(scores_normalized)

        # Sum per node
        exp_sum = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        exp_sum.scatter_add_(
            0,
            node_indices.unsqueeze(-1).expand_as(exp_scores),
            exp_scores
        )

        # Normalize
        attn_weights = exp_scores / (exp_sum[node_indices] + 1e-10)

        return attn_weights


class EnsemblePredictor(nn.Module):
    """
    集成预测器 - 组合多个模型的预测

    使用加权平均来提高预测准确性和鲁棒性
    """

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()

        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        self.weights = nn.Parameter(
            torch.tensor(weights, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播 - 加权集成

        Returns:
            ensemble_output: 加权平均的预测结果
        """
        # Normalize weights to sum to 1
        normalized_weights = F.softmax(self.weights, dim=0)

        predictions = []
        for model in self.models:
            pred = model(*args, **kwargs)
            predictions.append(pred)

        # Weighted average
        predictions = torch.stack(predictions)
        ensemble_output = (predictions * normalized_weights.view(-1, 1, 1)).sum(dim=0)

        return ensemble_output

    def predict_with_variance(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测并返回方差（不确定性度量）

        Returns:
            mean: 集成平均预测
            variance: 预测方差
        """
        predictions = []
        for model in self.models:
            pred = model(*args, **kwargs)
            predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        return mean, variance


class PerformanceMetrics:
    """性能指标跟踪器"""

    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.timestamps = []

    def add_prediction(
        self,
        predicted: float,
        actual: float,
        timestamp: float
    ):
        """添加预测记录"""
        self.predictions.append(predicted)
        self.actuals.append(actual)
        self.timestamps.append(timestamp)

    def compute_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        if not self.predictions:
            return {}

        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)

        # MAE
        mae = np.mean(np.abs(predictions - actuals))

        # RMSE
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        # MAPE
        mask = actuals != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        else:
            mape = float('inf')

        # R2 score
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        # Median absolute error
        median_ae = np.median(np.abs(predictions - actuals))

        # 95th percentile error
        p95_error = np.percentile(np.abs(predictions - actuals), 95)

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'median_ae': float(median_ae),
            'p95_error': float(p95_error),
            'num_samples': len(self.predictions)
        }

    def reset(self):
        """重置指标"""
        self.predictions.clear()
        self.actuals.clear()
        self.timestamps.clear()


class OnlinePredictor:
    """
    在线预测器 - 支持增量学习

    维护一个滑动窗口的预测历史，用于在线更新模型
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        window_size: int = 1000
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.window_size = window_size
        self.feature_buffer = deque(maxlen=window_size)
        self.target_buffer = deque(maxlen=window_size)

        self.metrics = PerformanceMetrics()

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """预测"""
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features)
        return prediction

    def update(
        self,
        features: torch.Tensor,
        actual: torch.Tensor,
        update_frequency: int = 10
    ):
        """
        增量更新模型

        Args:
            features: 输入特征
            actual: 实际观测值
            update_frequency: 每N个样本更新一次模型
        """
        # Add to buffer
        self.feature_buffer.append(features)
        self.target_buffer.append(actual)

        # Update metrics
        with torch.no_grad():
            pred = self.predict(features)
            self.metrics.add_prediction(
                pred.item(),
                actual.item(),
                time.time()
            )

        # Periodic model update
        if len(self.feature_buffer) >= update_frequency:
            self._incremental_training()

    def _incremental_training(self, num_epochs: int = 1):
        """增量训练"""
        if len(self.feature_buffer) == 0:
            return

        self.model.train()

        # Prepare batch
        features = torch.stack(list(self.feature_buffer))
        targets = torch.stack(list(self.target_buffer))

        for _ in range(num_epochs):
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()

    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return self.metrics.compute_metrics()

    def reset_metrics(self):
        """重置指标"""
        self.metrics.reset()


# 便捷函数

def create_time_predictor(device: str = 'cuda') -> CommunicationTimePredictor:
    """创建时间预测器"""
    model = CommunicationTimePredictor()
    model = model.to(device)
    return model


def create_bandwidth_predictor(device: str = 'cuda') -> BandwidthPredictor:
    """创建带宽预测器"""
    model = BandwidthPredictor()
    model = model.to(device)
    return model


def create_congestion_predictor(device: str = 'cuda') -> CongestionPredictor:
    """创建拥塞预测器"""
    model = CongestionPredictor()
    model = model.to(device)
    return model


def create_ensemble_time_predictor(
    num_models: int = 3,
    device: str = 'cuda'
) -> EnsemblePredictor:
    """创建集成时间预测器"""
    models = [CommunicationTimePredictor() for _ in range(num_models)]
    ensemble = EnsemblePredictor(models)
    ensemble = ensemble.to(device)
    return ensemble

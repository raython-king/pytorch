"""
特征工程模块 - 提取通讯操作特征用于ML模型

本模块负责从通讯操作、拓扑结构、工作负载历史和系统状态中
提取高质量特征，为ML预测模型提供输入。
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import time


@dataclass
class MessageFeatures:
    """消息特征"""
    size_log: float
    data_type: int
    is_contiguous: bool
    mean: float
    std: float
    sparsity: float
    element_count: int
    bytes_per_element: int
    # 额外统计特征
    min_val: float
    max_val: float
    median: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    # 内存布局特征
    stride_pattern: List[int]
    memory_efficiency: float
    cache_friendliness: float


@dataclass
class TopologyFeatures:
    """拓扑特征"""
    num_ranks: int
    topology_type: str
    pcie_ratio: float
    nvlink_ratio: float
    bandwidth_min: float
    bandwidth_max: float
    bandwidth_avg: float
    bandwidth_std: float
    latency_min: float
    latency_max: float
    latency_avg: float
    latency_std: float
    connectivity_degree: float
    hierarchy_depth: int
    # 图特征
    diameter: int
    avg_path_length: float
    clustering_coefficient: float
    degree_centrality: List[float]
    betweenness_centrality: List[float]
    # 带宽分布
    bandwidth_percentiles: List[float]
    link_utilization: List[float]


@dataclass
class WorkloadFeatures:
    """工作负载特征"""
    comm_frequency: float
    avg_message_size: float
    message_size_std: float
    burstiness: float
    pattern_type: str
    temporal_locality: float
    spatial_locality: float
    # 历史统计
    recent_sizes: List[float]
    recent_intervals: List[float]
    size_distribution: Dict[str, float]
    # 模式识别
    is_periodic: bool
    period_length: Optional[int]
    seasonality: float
    trend: float


@dataclass
class SystemFeatures:
    """系统特征"""
    gpu_utilization: List[float]
    memory_usage: List[float]
    network_congestion: float
    load_balance_score: float
    # CPU特征
    cpu_utilization: float
    context_switch_rate: float
    # 内存特征
    page_fault_rate: float
    swap_usage: float
    # 网络特征
    packet_loss_rate: float
    retransmission_rate: float
    queue_length: List[int]


class CommunicationFeatureExtractor:
    """通讯特征提取器"""

    MESSAGE_FEATURE_DIM = 32
    TOPOLOGY_FEATURE_DIM = 48
    WORKLOAD_FEATURE_DIM = 40
    SYSTEM_FEATURE_DIM = 24

    def __init__(self):
        self.history_window = 1000
        self.message_history = deque(maxlen=self.history_window)
        self.timing_history = deque(maxlen=self.history_window)
        self.feature_cache = {}
        self.cache_ttl = 0.1  # 100ms缓存过期时间

    def extract_message_features(
        self,
        message: torch.Tensor,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        提取消息特征

        Args:
            message: 消息张量
            dtype: 数据类型

        Returns:
            32维特征向量
        """
        if dtype is None:
            dtype = message.dtype

        features = []

        # 1. 基本特征 (8维)
        size_bytes = message.numel() * message.element_size()
        features.append(math.log10(size_bytes + 1))  # log size
        features.append(self._encode_dtype(dtype))  # dtype
        features.append(1.0 if message.is_contiguous() else 0.0)  # contiguous
        features.append(math.log10(message.numel() + 1))  # element count
        features.append(float(message.element_size()))  # bytes per element
        features.append(float(message.dim()))  # num dimensions
        features.append(math.log10(message.shape[0] + 1) if message.numel() > 0 else 0.0)
        features.append(1.0 if message.is_cuda else 0.0)  # device type

        # 2. 数据统计特征 (10维)
        if message.numel() > 0 and message.dtype.is_floating_point:
            try:
                flat = message.flatten().float()
                # 采样以加速计算
                if flat.numel() > 10000:
                    indices = torch.randperm(flat.numel())[:10000]
                    flat = flat[indices]

                features.append(flat.mean().item())
                features.append(flat.std().item() if flat.numel() > 1 else 0.0)
                features.append(float((flat == 0).sum()) / flat.numel())  # sparsity
                features.append(flat.min().item())
                features.append(flat.max().item())
                features.append(flat.median().item())

                # 分位数
                q = torch.quantile(flat, torch.tensor([0.25, 0.75]))
                features.append(q[0].item())
                features.append(q[1].item())

                # 偏度和峰度（简化计算）
                mean = flat.mean()
                std = flat.std()
                if std > 1e-6:
                    z = (flat - mean) / std
                    skew = (z ** 3).mean().item()
                    kurt = (z ** 4).mean().item() - 3.0
                else:
                    skew = 0.0
                    kurt = 0.0
                features.append(skew)
                features.append(kurt)
            except:
                features.extend([0.0] * 10)
        else:
            features.extend([0.0] * 10)

        # 3. 内存布局特征 (8维)
        if message.numel() > 0:
            stride_info = self._analyze_stride_pattern(message)
            features.extend(stride_info)
        else:
            features.extend([0.0] * 8)

        # 4. 形状特征 (6维)
        shape_features = self._encode_shape(message.shape)
        features.extend(shape_features)

        return torch.tensor(features, dtype=torch.float32)

    def extract_topology_features(self, topology: Any) -> torch.Tensor:
        """
        提取拓扑特征

        Args:
            topology: 拓扑对象

        Returns:
            48维特征向量
        """
        features = []

        # 1. 基本拓扑特征 (8维)
        num_ranks = getattr(topology, 'num_ranks', 1)
        features.append(float(num_ranks))
        features.append(math.log2(num_ranks + 1))

        # 拓扑类型编码
        topology_type = getattr(topology, 'type', 'unknown')
        type_encoding = self._encode_topology_type(topology_type)
        features.extend(type_encoding)  # 4维

        # 连接类型比例
        pcie_ratio = getattr(topology, 'pcie_ratio', 0.0)
        nvlink_ratio = getattr(topology, 'nvlink_ratio', 0.0)
        features.append(pcie_ratio)
        features.append(nvlink_ratio)

        # 2. 带宽统计特征 (10维)
        bandwidth_stats = self._compute_bandwidth_stats(topology)
        features.extend(bandwidth_stats)

        # 3. 延迟统计特征 (8维)
        latency_stats = self._compute_latency_stats(topology)
        features.extend(latency_stats)

        # 4. 图结构特征 (12维)
        graph_features = self._compute_graph_features(topology)
        features.extend(graph_features)

        # 5. 连接度特征 (10维)
        connectivity_features = self._compute_connectivity_features(topology)
        features.extend(connectivity_features)

        return torch.tensor(features, dtype=torch.float32)

    def extract_workload_features(
        self,
        workload_history: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        提取工作负载特征

        Args:
            workload_history: 历史工作负载记录

        Returns:
            40维特征向量
        """
        features = []

        if not workload_history:
            return torch.zeros(self.WORKLOAD_FEATURE_DIM, dtype=torch.float32)

        # 1. 通讯频率特征 (6维)
        freq_features = self._compute_frequency_features(workload_history)
        features.extend(freq_features)

        # 2. 消息大小分布特征 (10维)
        size_features = self._compute_size_distribution(workload_history)
        features.extend(size_features)

        # 3. 时序模式特征 (12维)
        temporal_features = self._compute_temporal_patterns(workload_history)
        features.extend(temporal_features)

        # 4. Burstiness特征 (6维)
        burst_features = self._compute_burstiness(workload_history)
        features.extend(burst_features)

        # 5. 局部性特征 (6维)
        locality_features = self._compute_locality(workload_history)
        features.extend(locality_features)

        return torch.tensor(features, dtype=torch.float32)

    def extract_system_features(self) -> torch.Tensor:
        """
        提取系统特征

        Returns:
            24维特征向量
        """
        features = []

        # 1. GPU特征 (8维)
        gpu_features = self._collect_gpu_features()
        features.extend(gpu_features)

        # 2. 内存特征 (6维)
        memory_features = self._collect_memory_features()
        features.extend(memory_features)

        # 3. 网络特征 (6维)
        network_features = self._collect_network_features()
        features.extend(network_features)

        # 4. CPU特征 (4维)
        cpu_features = self._collect_cpu_features()
        features.extend(cpu_features)

        return torch.tensor(features, dtype=torch.float32)

    # ==================== 私有辅助方法 ====================

    def _encode_dtype(self, dtype: torch.dtype) -> float:
        """编码数据类型"""
        dtype_map = {
            torch.float32: 1.0,
            torch.float64: 2.0,
            torch.float16: 3.0,
            torch.bfloat16: 4.0,
            torch.int32: 5.0,
            torch.int64: 6.0,
            torch.int16: 7.0,
            torch.int8: 8.0,
            torch.uint8: 9.0,
            torch.bool: 10.0,
        }
        return dtype_map.get(dtype, 0.0)

    def _analyze_stride_pattern(self, tensor: torch.Tensor) -> List[float]:
        """分析stride模式"""
        features = []

        if tensor.numel() == 0:
            return [0.0] * 8

        strides = tensor.stride()

        # Stride统计
        if strides:
            features.append(float(max(strides)))
            features.append(float(min(strides)))
            features.append(float(sum(strides)) / len(strides))

            # 是否为连续stride
            expected_stride = 1
            is_standard = True
            for i in range(len(strides) - 1, -1, -1):
                if strides[i] != expected_stride:
                    is_standard = False
                    break
                expected_stride *= tensor.shape[i]
            features.append(1.0 if is_standard else 0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # 内存效率
        if tensor.is_contiguous():
            efficiency = 1.0
        else:
            actual_bytes = tensor.numel() * tensor.element_size()
            storage_bytes = tensor.storage().size() * tensor.element_size()
            efficiency = actual_bytes / max(storage_bytes, 1)
        features.append(efficiency)

        # Cache友好性（简化估计）
        cache_line_size = 64  # bytes
        elements_per_line = cache_line_size // tensor.element_size()
        if strides and strides[-1] == 1:
            cache_friendliness = min(1.0, elements_per_line / max(tensor.shape[-1], 1))
        else:
            cache_friendliness = 0.5
        features.append(cache_friendliness)

        # 填充特征
        features.extend([0.0] * (8 - len(features)))

        return features[:8]

    def _encode_shape(self, shape: torch.Size) -> List[float]:
        """编码张量形状"""
        features = []

        # 维度数量
        features.append(float(len(shape)))

        # 前5个维度（不足补0）
        for i in range(5):
            if i < len(shape):
                features.append(math.log10(shape[i] + 1))
            else:
                features.append(0.0)

        return features

    def _encode_topology_type(self, topology_type: str) -> List[float]:
        """编码拓扑类型 - one-hot编码"""
        types = ['ring', 'tree', 'mesh', 'torus', 'fat_tree', 'unknown']
        encoding = [0.0] * len(types)

        if topology_type.lower() in types:
            idx = types.index(topology_type.lower())
            encoding[idx] = 1.0
        else:
            encoding[-1] = 1.0  # unknown

        return encoding[:4]  # 返回4维

    def _compute_bandwidth_stats(self, topology: Any) -> List[float]:
        """计算带宽统计特征"""
        features = []

        # 尝试从拓扑对象获取带宽信息
        try:
            bandwidths = getattr(topology, 'bandwidths', [])
            if not bandwidths:
                bandwidths = [100.0]  # 默认值 GB/s

            bandwidths = np.array(bandwidths)
            features.append(float(bandwidths.min()))
            features.append(float(bandwidths.max()))
            features.append(float(bandwidths.mean()))
            features.append(float(bandwidths.std()))
            features.append(float(np.median(bandwidths)))

            # 百分位数
            if len(bandwidths) > 1:
                p25, p75, p90 = np.percentile(bandwidths, [25, 75, 90])
                features.extend([float(p25), float(p75), float(p90)])
            else:
                features.extend([float(bandwidths[0])] * 3)

            # 带宽变异系数
            cv = bandwidths.std() / max(bandwidths.mean(), 1e-6)
            features.append(float(cv))

            # 归一化熵（带宽分布的均匀性）
            hist, _ = np.histogram(bandwidths, bins=10)
            hist = hist / max(hist.sum(), 1)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            features.append(float(entropy))

        except:
            features = [100.0, 100.0, 100.0, 0.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0]

        return features

    def _compute_latency_stats(self, topology: Any) -> List[float]:
        """计算延迟统计特征"""
        features = []

        try:
            latencies = getattr(topology, 'latencies', [])
            if not latencies:
                latencies = [0.01]  # 默认值 10us

            latencies = np.array(latencies) * 1e6  # 转换为微秒
            features.append(float(latencies.min()))
            features.append(float(latencies.max()))
            features.append(float(latencies.mean()))
            features.append(float(latencies.std()))
            features.append(float(np.median(latencies)))

            # 百分位数
            if len(latencies) > 1:
                p50, p90, p99 = np.percentile(latencies, [50, 90, 99])
                features.extend([float(p50), float(p90), float(p99)])
            else:
                features.extend([float(latencies[0])] * 3)

        except:
            features = [10.0, 10.0, 10.0, 0.0, 10.0, 10.0, 10.0, 10.0]

        return features

    def _compute_graph_features(self, topology: Any) -> List[float]:
        """计算图结构特征"""
        features = []

        try:
            num_ranks = getattr(topology, 'num_ranks', 1)

            # 直径（最长最短路径）
            diameter = getattr(topology, 'diameter', math.ceil(math.log2(num_ranks)))
            features.append(float(diameter))

            # 平均路径长度
            avg_path_len = getattr(topology, 'avg_path_length', diameter / 2)
            features.append(float(avg_path_len))

            # 聚类系数
            clustering = getattr(topology, 'clustering_coefficient', 0.5)
            features.append(float(clustering))

            # 度中心性统计
            degree_centrality = getattr(topology, 'degree_centrality', [])
            if degree_centrality:
                dc = np.array(degree_centrality)
                features.extend([float(dc.mean()), float(dc.std()), float(dc.max())])
            else:
                features.extend([0.5, 0.1, 1.0])

            # 介数中心性统计
            betweenness = getattr(topology, 'betweenness_centrality', [])
            if betweenness:
                bc = np.array(betweenness)
                features.extend([float(bc.mean()), float(bc.std()), float(bc.max())])
            else:
                features.extend([0.5, 0.1, 1.0])

            # 层级深度
            hierarchy_depth = getattr(topology, 'hierarchy_depth', 1)
            features.append(float(hierarchy_depth))

            # 连接密度
            max_edges = num_ranks * (num_ranks - 1) / 2
            actual_edges = getattr(topology, 'num_edges', num_ranks)
            density = actual_edges / max(max_edges, 1)
            features.append(float(density))

            # 平均度
            avg_degree = 2 * actual_edges / max(num_ranks, 1)
            features.append(float(avg_degree))

        except:
            features = [2.0, 1.5, 0.5, 0.5, 0.1, 1.0, 0.5, 0.1, 1.0, 1.0, 0.5, 2.0]

        return features

    def _compute_connectivity_features(self, topology: Any) -> List[float]:
        """计算连接度特征"""
        features = []

        try:
            num_ranks = getattr(topology, 'num_ranks', 1)

            # 连接度统计
            degrees = getattr(topology, 'node_degrees', [2] * num_ranks)
            degrees = np.array(degrees)

            features.append(float(degrees.mean()))
            features.append(float(degrees.std()))
            features.append(float(degrees.min()))
            features.append(float(degrees.max()))
            features.append(float(np.median(degrees)))

            # 度分布特征
            unique, counts = np.unique(degrees, return_counts=True)
            degree_entropy = -np.sum((counts / counts.sum()) * np.log(counts / counts.sum() + 1e-10))
            features.append(float(degree_entropy))

            # 连接均衡性
            ideal_degree = 2 * getattr(topology, 'num_edges', num_ranks) / max(num_ranks, 1)
            balance_score = 1.0 - degrees.std() / max(ideal_degree, 1)
            features.append(float(max(0.0, balance_score)))

            # 孤立节点数
            isolated = np.sum(degrees == 0)
            features.append(float(isolated))

            # 枢纽节点数（度 > 2*平均度）
            hubs = np.sum(degrees > 2 * degrees.mean())
            features.append(float(hubs))

            # 连通性（简化：假设连通）
            features.append(1.0)

        except:
            features = [2.0, 0.5, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0]

        return features

    def _compute_frequency_features(self, history: List[Dict]) -> List[float]:
        """计算通讯频率特征"""
        features = []

        if len(history) < 2:
            return [0.0] * 6

        # 提取时间戳
        timestamps = [h.get('timestamp', 0.0) for h in history]
        intervals = np.diff(timestamps)

        # 频率统计
        if len(intervals) > 0:
            features.append(1.0 / max(intervals.mean(), 1e-6))  # 平均频率
            features.append(float(intervals.std()))  # 间隔标准差
            features.append(float(intervals.min()))
            features.append(float(intervals.max()))

            # 频率稳定性
            cv = intervals.std() / max(intervals.mean(), 1e-6)
            features.append(float(cv))

            # 最近vs整体频率比
            if len(intervals) > 10:
                recent_freq = 1.0 / max(intervals[-10:].mean(), 1e-6)
                overall_freq = 1.0 / max(intervals.mean(), 1e-6)
                features.append(recent_freq / max(overall_freq, 1e-6))
            else:
                features.append(1.0)
        else:
            features = [0.0] * 6

        return features

    def _compute_size_distribution(self, history: List[Dict]) -> List[float]:
        """计算消息大小分布特征"""
        features = []

        sizes = np.array([h.get('size', 0) for h in history])

        if len(sizes) > 0:
            features.append(float(sizes.mean()))
            features.append(float(sizes.std()))
            features.append(float(sizes.min()))
            features.append(float(sizes.max()))
            features.append(float(np.median(sizes)))

            # 百分位数
            if len(sizes) > 1:
                p25, p75, p90 = np.percentile(sizes, [25, 75, 90])
                features.extend([float(p25), float(p75), float(p90)])
            else:
                features.extend([float(sizes[0])] * 3)

            # 变异系数
            cv = sizes.std() / max(sizes.mean(), 1)
            features.append(float(cv))

            # 大小稳定性
            if len(sizes) > 10:
                recent_mean = sizes[-10:].mean()
                overall_mean = sizes.mean()
                stability = recent_mean / max(overall_mean, 1)
                features.append(float(stability))
            else:
                features.append(1.0)
        else:
            features = [0.0] * 10

        return features

    def _compute_temporal_patterns(self, history: List[Dict]) -> List[float]:
        """计算时序模式特征"""
        features = []

        if len(history) < 10:
            return [0.0] * 12

        # 周期性检测
        sizes = np.array([h.get('size', 0) for h in history[-100:]])

        # 自相关
        if len(sizes) > 20:
            acf_1 = np.corrcoef(sizes[:-1], sizes[1:])[0, 1]
            acf_5 = np.corrcoef(sizes[:-5], sizes[5:])[0, 1] if len(sizes) > 5 else 0.0
            acf_10 = np.corrcoef(sizes[:-10], sizes[10:])[0, 1] if len(sizes) > 10 else 0.0
            features.extend([float(acf_1), float(acf_5), float(acf_10)])
        else:
            features.extend([0.0, 0.0, 0.0])

        # 趋势
        if len(sizes) > 10:
            x = np.arange(len(sizes))
            trend = np.polyfit(x, sizes, 1)[0]
            features.append(float(trend))
        else:
            features.append(0.0)

        # 季节性强度
        if len(sizes) > 20:
            detrended = sizes - np.mean(sizes)
            seasonality = np.std(detrended) / max(np.std(sizes), 1)
            features.append(float(seasonality))
        else:
            features.append(0.0)

        # 周期长度估计
        if len(sizes) > 30:
            fft = np.fft.fft(sizes - sizes.mean())
            power = np.abs(fft[:len(fft)//2])
            if power.max() > 0:
                dominant_freq_idx = np.argmax(power[1:]) + 1
                period = len(sizes) / dominant_freq_idx
                features.append(float(period))
                features.append(1.0)  # has_period
            else:
                features.append(0.0)
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])

        # 填充剩余特征
        features.extend([0.0] * (12 - len(features)))

        return features[:12]

    def _compute_burstiness(self, history: List[Dict]) -> List[float]:
        """计算Burstiness特征"""
        features = []

        if len(history) < 10:
            return [0.0] * 6

        timestamps = np.array([h.get('timestamp', 0.0) for h in history])
        sizes = np.array([h.get('size', 0) for h in history])

        intervals = np.diff(timestamps)

        # Burstiness系数
        if len(intervals) > 0:
            burst_coef = (intervals.std() - intervals.mean()) / (intervals.std() + intervals.mean())
            features.append(float(burst_coef))
        else:
            features.append(0.0)

        # 峰值检测
        if len(sizes) > 5:
            mean_size = sizes.mean()
            std_size = sizes.std()
            peaks = np.sum(sizes > mean_size + 2 * std_size)
            features.append(float(peaks) / len(sizes))
        else:
            features.append(0.0)

        # 突发持续时间
        if len(sizes) > 10:
            threshold = sizes.mean() + sizes.std()
            in_burst = sizes > threshold
            burst_lengths = []
            current_length = 0
            for is_burst in in_burst:
                if is_burst:
                    current_length += 1
                elif current_length > 0:
                    burst_lengths.append(current_length)
                    current_length = 0
            if burst_lengths:
                features.append(float(np.mean(burst_lengths)))
                features.append(float(np.max(burst_lengths)))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])

        # 突发间隔
        features.extend([0.0] * (6 - len(features)))

        return features[:6]

    def _compute_locality(self, history: List[Dict]) -> List[float]:
        """计算局部性特征"""
        features = []

        # 时间局部性
        if len(history) > 10:
            timestamps = np.array([h.get('timestamp', 0.0) for h in history])
            intervals = np.diff(timestamps)
            # 时间局部性：短间隔的比例
            short_intervals = np.sum(intervals < np.median(intervals)) / len(intervals)
            features.append(float(short_intervals))
        else:
            features.append(0.5)

        # 空间局部性（消息大小重复模式）
        if len(history) > 10:
            sizes = [h.get('size', 0) for h in history]
            unique_sizes = len(set(sizes))
            spatial_locality = 1.0 - unique_sizes / len(sizes)
            features.append(float(spatial_locality))
        else:
            features.append(0.5)

        # 填充剩余特征
        features.extend([0.0] * (6 - len(features)))

        return features[:6]

    def _collect_gpu_features(self) -> List[float]:
        """收集GPU特征"""
        features = []

        try:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                features.append(float(num_gpus))

                # 每个GPU的利用率和内存
                for i in range(min(num_gpus, 4)):  # 最多4个GPU
                    # 这里简化处理，实际应该调用nvidia-smi或NVML
                    features.append(0.5)  # utilization placeholder
                if num_gpus < 4:
                    features.extend([0.0] * (4 - num_gpus))

                # 平均利用率
                avg_util = 0.5
                features.append(avg_util)

                # 内存使用
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                features.append(float(allocated / total_memory))

                features.append(0.0)  # reserved for future
            else:
                features = [0.0] * 8
        except:
            features = [0.0] * 8

        return features

    def _collect_memory_features(self) -> List[float]:
        """收集内存特征"""
        features = []

        try:
            if torch.cuda.is_available():
                # GPU内存
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                total = torch.cuda.get_device_properties(0).total_memory

                features.append(float(allocated / total))
                features.append(float(reserved / total))
                features.append(float((total - allocated) / total))  # free ratio
            else:
                features.extend([0.0, 0.0, 1.0])

            # CPU内存（简化）
            features.extend([0.5, 0.0, 0.0])  # usage, page_fault, swap
        except:
            features = [0.0] * 6

        return features

    def _collect_network_features(self) -> List[float]:
        """收集网络特征"""
        # 简化实现，实际应该从系统监控获取
        features = [
            0.0,  # congestion
            0.0,  # packet_loss
            0.0,  # retransmission
            0.5,  # utilization
            0.0,  # queue_length
            1.0,  # link_status
        ]
        return features

    def _collect_cpu_features(self) -> List[float]:
        """收集CPU特征"""
        # 简化实现
        features = [
            0.5,  # utilization
            0.0,  # context_switch_rate
            0.0,  # load_average
            1.0,  # cpu_available
        ]
        return features

# GPU集群通讯优化系统 - 核心组件详细设计

本文档详细描述GPU集群自适应通讯优化系统（AGCCS）的7个核心组件的设计，包括接口定义、状态管理、交互协议和实现细节。

---

## 目录

1. [CommunicationProfiler - 通讯性能分析器](#1-communicationprofiler---通讯性能分析器)
2. [TopologyAwareScheduler - 拓扑感知调度器](#2-topologyawarescheduler---拓扑感知调度器)
3. [AdaptiveCollectiveOptimizer - 自适应集合通讯优化器](#3-adaptivecollectiveoptimizer---自适应集合通讯优化器)
4. [OverlapOrchestrator - 计算通讯重叠编排器](#4-overlaporchestrator---计算通讯重叠编排器)
5. [CompressionManager - 通讯压缩管理器](#5-compressionmanager---通讯压缩管理器)
6. [LoadBalancer - 负载均衡器](#6-loadbalancer---负载均衡器)
7. [MLPredictor - ML预测器](#7-mlpredictor---ml预测器)

---

## 1. CommunicationProfiler - 通讯性能分析器

### 1.1 职责概述

CommunicationProfiler负责全面分析GPU集群中的通讯模式和性能特征，为上层决策提供数据基础。

**核心职责：**
- 追踪消息大小分布
- 测量实时延迟和带宽
- 识别热点通讯路径
- 分析通讯模式（周期性、突发性等）
- 预测未来通讯需求

### 1.2 接口定义

```python
class CommunicationProfiler:
    """通讯性能分析器"""

    def __init__(self, config: ProfilerConfig):
        """
        初始化Profiler

        Args:
            config: 配置对象
                - sampling_rate: 采样率（0-1），1表示记录所有通讯
                - window_size: 滑动窗口大小（秒）
                - histogram_bins: 直方图bins数量
        """
        pass

    def record_communication(
        self,
        comm_id: str,
        operation: str,
        tensor_size: int,
        dtype: torch.dtype,
        src_rank: int,
        dst_rank: Optional[Union[int, List[int]]],
        start_time: float,
        end_time: Optional[float] = None
    ) -> None:
        """
        记录一次通讯事件

        Args:
            comm_id: 通讯唯一标识符
            operation: 操作类型（'all_reduce', 'all_gather', 'send', 'recv'等）
            tensor_size: 张量大小（字节）
            dtype: 数据类型
            src_rank: 源rank
            dst_rank: 目标rank（点对点）或None（集合通讯）
            start_time: 开始时间戳
            end_time: 结束时间戳（可选，用于异步操作）
        """
        pass

    def get_message_size_distribution(
        self,
        operation: Optional[str] = None,
        time_window: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        获取消息大小分布

        Args:
            operation: 过滤特定操作（None表示所有操作）
            time_window: 时间窗口（秒，None表示所有历史）

        Returns:
            {
                'histogram': [(size_range, count), ...],
                'mean': float,
                'median': float,
                'p95': float,
                'p99': float,
                'min': int,
                'max': int
            }
        """
        pass

    def get_bandwidth_stats(
        self,
        src_rank: Optional[int] = None,
        dst_rank: Optional[int] = None,
        time_window: Optional[float] = None
    ) -> Dict[str, float]:
        """
        获取带宽统计信息

        Args:
            src_rank: 源rank过滤
            dst_rank: 目标rank过滤
            time_window: 时间窗口（秒）

        Returns:
            {
                'current_bw_gbps': float,      # 当前带宽
                'avg_bw_gbps': float,          # 平均带宽
                'peak_bw_gbps': float,         # 峰值带宽
                'utilization': float,          # 利用率（0-1）
                'variance': float              # 方差
            }
        """
        pass

    def get_latency_stats(
        self,
        operation: Optional[str] = None,
        time_window: Optional[float] = None
    ) -> Dict[str, float]:
        """
        获取延迟统计信息

        Args:
            operation: 操作类型过滤
            time_window: 时间窗口（秒）

        Returns:
            {
                'mean_latency_ms': float,
                'median_latency_ms': float,
                'p95_latency_ms': float,
                'p99_latency_ms': float,
                'min_latency_ms': float,
                'max_latency_ms': float
            }
        """
        pass

    def identify_hotspots(
        self,
        threshold: float = 0.8,
        metric: str = 'utilization'
    ) -> List[Tuple[int, int, float]]:
        """
        识别热点通讯路径

        Args:
            threshold: 阈值（0-1）
            metric: 指标类型（'utilization', 'bandwidth', 'frequency'）

        Returns:
            [(src_rank, dst_rank, metric_value), ...]
            按metric_value降序排列
        """
        pass

    def detect_communication_pattern(
        self,
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        检测通讯模式

        Args:
            min_confidence: 最小置信度

        Returns:
            {
                'pattern_type': str,  # 'periodic', 'bursty', 'steady', 'irregular'
                'confidence': float,
                'period_ms': Optional[float],  # 如果是periodic
                'burst_interval_ms': Optional[float],  # 如果是bursty
                'characteristics': Dict[str, Any]
            }
        """
        pass

    def predict_next_communication(
        self,
        lookback_window: float = 10.0
    ) -> Dict[str, Any]:
        """
        预测下一次通讯的特征

        Args:
            lookback_window: 回溯窗口（秒）

        Returns:
            {
                'predicted_operation': str,
                'predicted_size': int,
                'predicted_time': float,  # 距现在的时间（秒）
                'confidence': float
            }
        """
        pass

    def export_trace(
        self,
        format: str = 'chrome',
        output_path: str = 'comm_trace.json'
    ) -> None:
        """
        导出通讯trace

        Args:
            format: 输出格式（'chrome', 'tensorboard', 'csv'）
            output_path: 输出文件路径
        """
        pass
```

### 1.3 状态管理

```python
@dataclass
class CommunicationEvent:
    """单次通讯事件"""
    comm_id: str
    operation: str
    tensor_size: int
    dtype: torch.dtype
    src_rank: int
    dst_rank: Optional[Union[int, List[int]]]
    start_time: float
    end_time: Optional[float]
    bandwidth_gbps: Optional[float]
    latency_ms: Optional[float]

    # 元数据
    algorithm_used: Optional[str] = None
    compression_used: Optional[str] = None
    overlap_ratio: Optional[float] = None


class ProfilerState:
    """Profiler内部状态"""

    def __init__(self, max_history: int = 10000):
        # 事件存储（环形缓冲区）
        self.events: deque[CommunicationEvent] = deque(maxlen=max_history)

        # 实时统计
        self.link_stats: Dict[Tuple[int, int], LinkStatistics] = {}

        # 操作统计
        self.op_stats: Dict[str, OperationStatistics] = defaultdict(OperationStatistics)

        # 时间序列数据（用于模式检测）
        self.time_series: List[Tuple[float, str, int]] = []  # (timestamp, op, size)

        # 热点缓存
        self.hotspot_cache: Optional[List[Tuple[int, int, float]]] = None
        self.hotspot_cache_time: float = 0.0

@dataclass
class LinkStatistics:
    """链路统计信息"""
    total_bytes: int = 0
    total_transfers: int = 0
    total_time: float = 0.0

    # 滑动窗口统计
    window_bytes: deque[Tuple[float, int]] = field(default_factory=deque)
    window_latencies: deque[float] = field(default_factory=deque)

    # 实时指标
    current_bw_gbps: float = 0.0
    avg_bw_gbps: float = 0.0
    peak_bw_gbps: float = 0.0
    utilization: float = 0.0

    def update(self, event: CommunicationEvent):
        """更新统计信息"""
        self.total_bytes += event.tensor_size
        self.total_transfers += 1
        if event.end_time:
            self.total_time += (event.end_time - event.start_time)

        # 更新滑动窗口
        self.window_bytes.append((event.start_time, event.tensor_size))
        if event.latency_ms:
            self.window_latencies.append(event.latency_ms)

        # 计算实时带宽
        if event.bandwidth_gbps:
            self.current_bw_gbps = event.bandwidth_gbps
            self.peak_bw_gbps = max(self.peak_bw_gbps, event.bandwidth_gbps)
```

### 1.4 实现细节

#### 采样策略

为了降低开销，Profiler使用智能采样：

```python
class AdaptiveSampler:
    """自适应采样器"""

    def __init__(self, base_rate: float = 0.1):
        self.base_rate = base_rate
        self.importance_score: Dict[str, float] = {}

    def should_sample(self, operation: str, size: int) -> bool:
        """决定是否采样此事件"""

        # 规则1：重要事件总是采样
        if size > 100 * 1024 * 1024:  # > 100MB
            return True

        # 规则2：稀有操作总是采样
        if self.importance_score.get(operation, 1.0) > 0.8:
            return True

        # 规则3：基于基础采样率的概率采样
        return random.random() < self.base_rate

    def update_importance(self, operation: str, frequency: float):
        """更新操作重要性"""
        # 稀有操作重要性高
        self.importance_score[operation] = 1.0 / (1.0 + frequency)
```

#### 模式检测算法

```python
def detect_periodicity(time_series: List[float]) -> Tuple[Optional[float], float]:
    """
    检测周期性模式

    Returns:
        (period, confidence)
    """
    # 使用自相关函数检测周期
    if len(time_series) < 10:
        return None, 0.0

    # 计算自相关
    autocorr = np.correlate(time_series, time_series, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # 查找峰值
    peaks, properties = scipy.signal.find_peaks(
        autocorr,
        height=np.mean(autocorr) + np.std(autocorr)
    )

    if len(peaks) > 0:
        # 第一个峰值对应周期
        period = peaks[0] * (time_series[-1] - time_series[0]) / len(time_series)
        confidence = properties['peak_heights'][0] / np.max(autocorr)
        return period, confidence

    return None, 0.0
```

---

## 2. TopologyAwareScheduler - 拓扑感知调度器

### 2.1 职责概述

TopologyAwareScheduler深度理解GPU集群的层级拓扑结构，并基于拓扑信息进行智能调度和路由决策。

**核心职责：**
- 自动发现和建模GPU拓扑（NVLink、PCIe、InfiniBand）
- 生成拓扑感知的通讯树
- 优化rank到GPU的分配
- 选择最优通讯路径（避免拥塞、最小化延迟）

### 2.2 接口定义

```python
class TopologyAwareScheduler:
    """拓扑感知调度器"""

    def __init__(self, config: TopologyConfig):
        """
        初始化调度器

        Args:
            config: 配置对象
                - auto_discover: 是否自动发现拓扑
                - topology_file: 手动指定拓扑文件
                - optimization_objective: 'latency', 'bandwidth', 'balanced'
        """
        pass

    def discover_topology(self) -> ClusterTopology:
        """
        自动发现集群拓扑

        Returns:
            ClusterTopology对象，包含完整的拓扑图
        """
        pass

    def build_communication_tree(
        self,
        operation: str,
        root_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        objective: str = 'latency'
    ) -> CommunicationTree:
        """
        构建通讯树

        Args:
            operation: 操作类型（'reduce', 'broadcast', 'allreduce'）
            root_rank: 根节点rank（对于reduce/broadcast）
            world_size: 参与的进程数
            objective: 优化目标（'latency', 'bandwidth', 'balanced'）

        Returns:
            CommunicationTree对象，表示最优通讯拓扑
        """
        pass

    def optimize_rank_placement(
        self,
        workload: WorkloadDescription
    ) -> Dict[int, int]:
        """
        优化rank到GPU的分配

        Args:
            workload: 工作负载描述（通讯模式、计算需求）

        Returns:
            {rank_id -> gpu_id} 的映射
        """
        pass

    def select_communication_path(
        self,
        src_rank: int,
        dst_rank: int,
        message_size: int,
        objective: str = 'latency'
    ) -> List[int]:
        """
        选择最优通讯路径

        Args:
            src_rank: 源rank
            dst_rank: 目标rank
            message_size: 消息大小
            objective: 优化目标

        Returns:
            路径上的rank序列 [src, intermediate1, ..., dst]
        """
        pass

    def get_link_bandwidth(
        self,
        src_gpu: int,
        dst_gpu: int
    ) -> float:
        """
        获取链路带宽

        Args:
            src_gpu: 源GPU ID
            dst_gpu: 目标GPU ID

        Returns:
            带宽（GB/s）
        """
        pass

    def get_hierarchy_level(
        self,
        src_gpu: int,
        dst_gpu: int
    ) -> str:
        """
        获取两个GPU之间的层级关系

        Returns:
            'nvlink', 'pcie_switch', 'pcie_cpu', 'network'
        """
        pass

    def update_link_cost(
        self,
        src_gpu: int,
        dst_gpu: int,
        cost: float
    ) -> None:
        """
        更新链路代价（用于动态调整）

        Args:
            src_gpu: 源GPU
            dst_gpu: 目标GPU
            cost: 新的代价值（越小越好）
        """
        pass
```

### 2.3 拓扑数据结构

```python
@dataclass
class ClusterTopology:
    """集群拓扑描述"""

    # GPU节点
    gpus: List[GPUNode]

    # 链路图（NetworkX图）
    link_graph: nx.Graph

    # 层级信息
    hierarchy: TopologyHierarchy

    # 带宽矩阵
    bandwidth_matrix: np.ndarray  # [num_gpus, num_gpus]

    # 延迟矩阵
    latency_matrix: np.ndarray  # [num_gpus, num_gpus]

@dataclass
class GPUNode:
    """GPU节点信息"""
    gpu_id: int
    device_name: str
    compute_capability: Tuple[int, int]
    memory_gb: float
    pci_bus_id: str

    # 位置信息
    node_id: int  # 物理节点（服务器）ID
    socket_id: int  # CPU socket ID

@dataclass
class TopologyHierarchy:
    """拓扑层级结构"""

    # 层级1：NVLink域（单机内高速互联）
    nvlink_domains: List[Set[int]]  # 每个set包含通过NVLink连接的GPU IDs

    # 层级2：PCIe域（同一PCIe switch下）
    pcie_domains: List[Set[int]]

    # 层级3：节点（同一物理服务器）
    nodes: List[Set[int]]

    # 层级4：机架
    racks: List[Set[int]]

@dataclass
class CommunicationTree:
    """通讯树"""
    operation: str
    root_rank: Optional[int]

    # 树结构
    tree: nx.DiGraph  # 有向图，边表示通讯方向

    # 每个节点的子节点列表
    children: Dict[int, List[int]]

    # 树的深度
    depth: int

    # 预估性能
    estimated_latency_ms: float
    estimated_bandwidth_gbps: float
```

### 2.4 拓扑发现实现

```python
class TopologyDiscovery:
    """拓扑自动发现"""

    @staticmethod
    def discover_nvlink_topology() -> Dict[int, Set[int]]:
        """
        发现NVLink拓扑

        Returns:
            {gpu_id: {connected_gpu_ids}}
        """
        nvlink_graph = {}

        for gpu_id in range(torch.cuda.device_count()):
            # 使用nvidia-smi查询NVLink连接
            # nvlink_graph[gpu_id] = {...}
            pass

        return nvlink_graph

    @staticmethod
    def discover_pcie_topology() -> Dict[int, str]:
        """
        发现PCIe拓扑

        Returns:
            {gpu_id: pci_bus_id}
        """
        pcie_info = {}

        for gpu_id in range(torch.cuda.device_count()):
            # 获取PCI bus ID
            # pcie_info[gpu_id] = "0000:3b:00.0"
            pass

        return pcie_info

    @staticmethod
    def discover_network_topology() -> Dict[int, str]:
        """
        发现网络拓扑（IB/RoCE）

        Returns:
            {node_id: network_interface}
        """
        # 查询NCCL_IB_HCA等环境变量
        # 或使用ibv_devinfo等工具
        pass

    @staticmethod
    def build_complete_topology() -> ClusterTopology:
        """构建完整拓扑"""

        # 1. 发现各层级拓扑
        nvlink_topo = TopologyDiscovery.discover_nvlink_topology()
        pcie_topo = TopologyDiscovery.discover_pcie_topology()
        network_topo = TopologyDiscovery.discover_network_topology()

        # 2. 构建图
        G = nx.Graph()

        # 添加GPU节点
        for gpu_id in range(torch.cuda.device_count()):
            G.add_node(gpu_id, type='gpu')

        # 添加NVLink边
        for src, dsts in nvlink_topo.items():
            for dst in dsts:
                G.add_edge(src, dst, type='nvlink', bandwidth=300.0)  # 300 GB/s

        # 添加PCIe边
        # ...

        # 3. 测量带宽和延迟
        bandwidth_matrix = TopologyDiscovery.measure_bandwidth_matrix()
        latency_matrix = TopologyDiscovery.measure_latency_matrix()

        return ClusterTopology(
            gpus=[...],
            link_graph=G,
            hierarchy=TopologyHierarchy(...),
            bandwidth_matrix=bandwidth_matrix,
            latency_matrix=latency_matrix
        )

    @staticmethod
    def measure_bandwidth_matrix() -> np.ndarray:
        """
        测量所有GPU对之间的带宽

        使用micro-benchmark测量实际带宽
        """
        num_gpus = torch.cuda.device_count()
        bw_matrix = np.zeros((num_gpus, num_gpus))

        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src == dst:
                    continue
                # 执行带宽测试
                bw_matrix[src, dst] = measure_p2p_bandwidth(src, dst)

        return bw_matrix
```

### 2.5 通讯树生成算法

```python
class CommunicationTreeBuilder:
    """通讯树构建器"""

    @staticmethod
    def build_reduce_tree(
        topology: ClusterTopology,
        root_rank: int,
        world_size: int,
        objective: str = 'latency'
    ) -> CommunicationTree:
        """
        构建Reduce树（叶子 -> 根）

        策略：层级化Reduce
        1. 同一NVLink域内先reduce
        2. 跨PCIe域reduce
        3. 跨节点reduce
        """

        # 1. 按层级分组ranks
        hierarchy = topology.hierarchy
        rank_to_gpu = list(range(world_size))  # 简化，假设rank_id == gpu_id

        # 2. 构建层级树
        tree = nx.DiGraph()
        tree.add_nodes_from(range(world_size))

        # Level 1: NVLink域内reduce
        nvlink_roots = []
        for domain in hierarchy.nvlink_domains:
            domain_ranks = [r for r in range(world_size) if rank_to_gpu[r] in domain]
            if not domain_ranks:
                continue

            # 在域内构建二叉树
            local_root = CommunicationTreeBuilder._build_binary_tree(
                tree, domain_ranks, objective
            )
            nvlink_roots.append(local_root)

        # Level 2: 跨PCIe域
        pcie_roots = []
        for pcie_domain in hierarchy.pcie_domains:
            domain_roots = [r for r in nvlink_roots if rank_to_gpu[r] in pcie_domain]
            if not domain_roots:
                continue

            local_root = CommunicationTreeBuilder._build_binary_tree(
                tree, domain_roots, objective
            )
            pcie_roots.append(local_root)

        # Level 3: 跨节点，最终reduce到root_rank
        current_roots = pcie_roots
        while len(current_roots) > 1:
            next_roots = []
            for i in range(0, len(current_roots), 2):
                if i + 1 < len(current_roots):
                    # 合并两个root
                    parent = current_roots[i]
                    child = current_roots[i + 1]
                    tree.add_edge(child, parent)
                    next_roots.append(parent)
                else:
                    next_roots.append(current_roots[i])
            current_roots = next_roots

        # 确保最终root是指定的root_rank
        final_root = current_roots[0]
        if final_root != root_rank:
            tree.add_edge(final_root, root_rank)

        return CommunicationTree(
            operation='reduce',
            root_rank=root_rank,
            tree=tree,
            children=_extract_children(tree),
            depth=_compute_tree_depth(tree, root_rank),
            estimated_latency_ms=_estimate_latency(tree, topology),
            estimated_bandwidth_gbps=_estimate_bandwidth(tree, topology)
        )

    @staticmethod
    def _build_binary_tree(
        tree: nx.DiGraph,
        nodes: List[int],
        objective: str
    ) -> int:
        """
        在给定节点集合内构建二叉树

        Returns:
            树的根节点
        """
        if len(nodes) == 1:
            return nodes[0]

        # 选择根节点（例如，选择中间节点以平衡树深度）
        root = nodes[len(nodes) // 2]

        # 递归构建左右子树
        left_nodes = nodes[:len(nodes)//2]
        right_nodes = nodes[len(nodes)//2+1:]

        if left_nodes:
            left_root = CommunicationTreeBuilder._build_binary_tree(tree, left_nodes, objective)
            tree.add_edge(left_root, root)

        if right_nodes:
            right_root = CommunicationTreeBuilder._build_binary_tree(tree, right_nodes, objective)
            tree.add_edge(right_root, root)

        return root
```

---

## 3. AdaptiveCollectiveOptimizer - 自适应集合通讯优化器

### 3.1 职责概述

AdaptiveCollectiveOptimizer是系统的核心优化引擎，负责为每次集合通讯选择最优的算法、参数和执行策略。

**核心职责：**
- 算法选择（Ring, Tree, Double-Binary-Tree, Halving-Doubling等）
- 参数调优（chunk size, pipeline depth等）
- 消息分块和分段
- 性能预测和评估

### 3.2 接口定义

```python
class AdaptiveCollectiveOptimizer:
    """自适应集合通讯优化器"""

    def __init__(
        self,
        topology: ClusterTopology,
        profiler: CommunicationProfiler,
        ml_predictor: Optional['MLPredictor'] = None
    ):
        """
        初始化优化器

        Args:
            topology: 集群拓扑
            profiler: 通讯profiler
            ml_predictor: ML预测器（可选）
        """
        pass

    def optimize_collective(
        self,
        operation: str,
        tensor_size: int,
        dtype: torch.dtype,
        world_size: int,
        current_state: SystemState
    ) -> CollectiveExecutionPlan:
        """
        优化集合通讯操作

        Args:
            operation: 操作类型（'all_reduce', 'all_gather', 'reduce_scatter'）
            tensor_size: 张量大小（字节）
            dtype: 数据类型
            world_size: 参与进程数
            current_state: 当前系统状态（负载、拥塞等）

        Returns:
            CollectiveExecutionPlan对象，包含完整的执行计划
        """
        pass

    def select_algorithm(
        self,
        operation: str,
        tensor_size: int,
        world_size: int,
        topology: ClusterTopology
    ) -> Tuple[str, float]:
        """
        选择最优算法

        Args:
            operation: 操作类型
            tensor_size: 消息大小
            world_size: 进程数
            topology: 拓扑结构

        Returns:
            (algorithm_name, confidence_score)
            algorithm_name: 'ring', 'tree', 'double_binary_tree', 'halving_doubling', 'recursive_halving_doubling'
        """
        pass

    def tune_parameters(
        self,
        algorithm: str,
        tensor_size: int,
        world_size: int
    ) -> Dict[str, Any]:
        """
        调优算法参数

        Args:
            algorithm: 算法名称
            tensor_size: 消息大小
            world_size: 进程数

        Returns:
            参数字典，例如：
            {
                'chunk_size': 1048576,  # 1MB chunks
                'num_chunks': 128,
                'pipeline_depth': 4,
                'num_streams': 2
            }
        """
        pass

    def estimate_performance(
        self,
        plan: CollectiveExecutionPlan,
        topology: ClusterTopology
    ) -> PerformanceEstimate:
        """
        估计执行计划的性能

        Args:
            plan: 执行计划
            topology: 拓扑

        Returns:
            性能估计对象
        """
        pass
```

### 3.3 数据结构

```python
@dataclass
class CollectiveExecutionPlan:
    """集合通讯执行计划"""

    # 基本信息
    operation: str
    algorithm: str
    tensor_size: int
    world_size: int

    # 算法参数
    chunk_size: int
    num_chunks: int
    pipeline_depth: int
    num_streams: int

    # 通讯树/环/路径
    communication_graph: nx.DiGraph

    # 调度信息
    schedule: List[CommunicationStep]

    # 压缩配置
    compression_config: Optional[CompressionConfig]

    # 性能预估
    estimated_time_ms: float
    estimated_bandwidth_gbps: float

    # 元数据
    confidence: float
    fallback_plan: Optional['CollectiveExecutionPlan']

@dataclass
class CommunicationStep:
    """单个通讯步骤"""
    step_id: int
    operation: str  # 'send', 'recv', 'reduce'
    src_rank: Optional[int]
    dst_rank: Optional[int]
    chunk_id: int
    offset: int
    size: int
    stream_id: int
    dependencies: List[int]  # 依赖的step_ids

@dataclass
class PerformanceEstimate:
    """性能估计"""
    total_time_ms: float
    computation_time_ms: float
    communication_time_ms: float
    overlap_time_ms: float

    # 详细breakdown
    breakdown: Dict[str, float]

    # 置信区间
    confidence_interval: Tuple[float, float]  # (lower, upper)
```

### 3.4 算法实现

#### Ring-AllReduce

```python
class RingAllReduce:
    """Ring-AllReduce算法"""

    @staticmethod
    def generate_plan(
        tensor_size: int,
        world_size: int,
        chunk_size: int,
        topology: ClusterTopology
    ) -> CollectiveExecutionPlan:
        """
        生成Ring-AllReduce执行计划

        算法流程：
        1. ReduceScatter阶段：N-1步，每步每个GPU发送和接收一个chunk
        2. AllGather阶段：N-1步，每步每个GPU发送和接收一个chunk
        """

        num_chunks = (tensor_size + chunk_size - 1) // chunk_size
        schedule = []

        # 构建Ring拓扑
        ring = list(range(world_size))

        # Phase 1: ReduceScatter
        for step in range(world_size - 1):
            for rank in range(world_size):
                send_rank = rank
                recv_rank = (rank + 1) % world_size

                # 确定chunk
                chunk_id = (rank - step) % world_size

                schedule.append(CommunicationStep(
                    step_id=len(schedule),
                    operation='send_recv_reduce',
                    src_rank=send_rank,
                    dst_rank=recv_rank,
                    chunk_id=chunk_id,
                    offset=chunk_id * chunk_size,
                    size=min(chunk_size, tensor_size - chunk_id * chunk_size),
                    stream_id=0,
                    dependencies=[] if step == 0 else [len(schedule) - world_size]
                ))

        # Phase 2: AllGather
        for step in range(world_size - 1):
            for rank in range(world_size):
                send_rank = rank
                recv_rank = (rank + 1) % world_size

                chunk_id = (rank - step + 1) % world_size

                schedule.append(CommunicationStep(
                    step_id=len(schedule),
                    operation='send_recv',
                    src_rank=send_rank,
                    dst_rank=recv_rank,
                    chunk_id=chunk_id,
                    offset=chunk_id * chunk_size,
                    size=min(chunk_size, tensor_size - chunk_id * chunk_size),
                    stream_id=0,
                    dependencies=[len(schedule) - world_size]
                ))

        # 估计性能
        # Ring算法：数据量 = 2 * (N-1)/N * S
        # 时间 = 2 * (N-1)/N * S / B + 2 * (N-1) * L
        avg_bandwidth = np.mean(topology.bandwidth_matrix[topology.bandwidth_matrix > 0])
        avg_latency = np.mean(topology.latency_matrix[topology.latency_matrix > 0])

        data_volume = 2 * (world_size - 1) / world_size * tensor_size
        estimated_time = (data_volume / avg_bandwidth / 1e9 * 1000 +  # 转换为ms
                         2 * (world_size - 1) * avg_latency)

        return CollectiveExecutionPlan(
            operation='all_reduce',
            algorithm='ring',
            tensor_size=tensor_size,
            world_size=world_size,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            pipeline_depth=1,
            num_streams=1,
            communication_graph=_build_ring_graph(world_size),
            schedule=schedule,
            compression_config=None,
            estimated_time_ms=estimated_time,
            estimated_bandwidth_gbps=data_volume / estimated_time * 1000 / 1e9,
            confidence=0.9,
            fallback_plan=None
        )
```

#### Tree-AllReduce

```python
class TreeAllReduce:
    """Tree-AllReduce算法"""

    @staticmethod
    def generate_plan(
        tensor_size: int,
        world_size: int,
        chunk_size: int,
        topology: ClusterTopology
    ) -> CollectiveExecutionPlan:
        """
        生成Tree-AllReduce执行计划

        算法流程：
        1. Reduce到root（树的向上传播）
        2. Broadcast从root（树的向下传播）
        """

        schedule = []

        # 构建通讯树（基于拓扑优化）
        tree_builder = CommunicationTreeBuilder()
        reduce_tree = tree_builder.build_reduce_tree(
            topology, root_rank=0, world_size=world_size, objective='latency'
        )

        # Phase 1: Reduce阶段（叶子 -> 根）
        # 按树的层级调度，从叶子到根
        levels = _compute_tree_levels(reduce_tree.tree, root=0)

        for level in reversed(range(len(levels))):
            for node in levels[level]:
                if node == 0:  # 根节点
                    continue

                # 找到父节点
                parent = list(reduce_tree.tree.successors(node))[0]

                schedule.append(CommunicationStep(
                    step_id=len(schedule),
                    operation='send_reduce',
                    src_rank=node,
                    dst_rank=parent,
                    chunk_id=0,
                    offset=0,
                    size=tensor_size,
                    stream_id=0,
                    dependencies=_get_child_dependencies(schedule, node, reduce_tree)
                ))

        # Phase 2: Broadcast阶段（根 -> 叶子）
        for level in range(1, len(levels)):
            for node in levels[level]:
                parent = list(reduce_tree.tree.predecessors(node))[0]

                schedule.append(CommunicationStep(
                    step_id=len(schedule),
                    operation='recv',
                    src_rank=parent,
                    dst_rank=node,
                    chunk_id=0,
                    offset=0,
                    size=tensor_size,
                    stream_id=0,
                    dependencies=[_find_parent_broadcast_step(schedule, parent)]
                ))

        # 估计性能
        tree_depth = reduce_tree.depth
        # Tree算法：时间 = 2 * log(N) * S / B + 2 * log(N) * L
        avg_bandwidth = np.mean(topology.bandwidth_matrix[topology.bandwidth_matrix > 0])
        avg_latency = np.mean(topology.latency_matrix[topology.latency_matrix > 0])

        estimated_time = (2 * tree_depth * tensor_size / avg_bandwidth / 1e9 * 1000 +
                         2 * tree_depth * avg_latency)

        return CollectiveExecutionPlan(
            operation='all_reduce',
            algorithm='tree',
            tensor_size=tensor_size,
            world_size=world_size,
            chunk_size=tensor_size,  # Tree通常不分块
            num_chunks=1,
            pipeline_depth=1,
            num_streams=1,
            communication_graph=reduce_tree.tree,
            schedule=schedule,
            compression_config=None,
            estimated_time_ms=estimated_time,
            estimated_bandwidth_gbps=2 * tensor_size / estimated_time * 1000 / 1e9,
            confidence=0.85,
            fallback_plan=None
        )
```

#### Double-Binary-Tree

```python
class DoubleBinaryTreeAllReduce:
    """Double-Binary-Tree AllReduce算法

    使用两棵独立的二叉树并行传输，充分利用双向带宽
    """

    @staticmethod
    def generate_plan(
        tensor_size: int,
        world_size: int,
        chunk_size: int,
        topology: ClusterTopology
    ) -> CollectiveExecutionPlan:
        """
        生成Double-Binary-Tree执行计划

        策略：
        1. 将数据分成两半
        2. 在Tree1上reduce前一半，在Tree2上reduce后一半
        3. 交换两棵树的根节点的数据
        4. 在Tree1上broadcast后一半，在Tree2上broadcast前一半
        """

        # 构建两棵不相交的树
        tree1 = _build_binary_tree_subset(topology, range(0, world_size, 2))
        tree2 = _build_binary_tree_subset(topology, range(1, world_size, 2))

        schedule = []

        # Phase 1: Parallel Reduce on two trees
        # Tree1处理前一半数据，Tree2处理后一半数据
        half_size = tensor_size // 2

        # ... 具体调度逻辑 ...

        return CollectiveExecutionPlan(...)
```

### 3.5 算法选择逻辑

```python
class AlgorithmSelector:
    """算法选择器"""

    @staticmethod
    def select_best_algorithm(
        operation: str,
        tensor_size: int,
        world_size: int,
        topology: ClusterTopology,
        ml_predictor: Optional['MLPredictor'] = None
    ) -> str:
        """
        选择最优算法

        决策树：
        1. 如果有ML模型且置信度 > 0.8，使用ML预测
        2. 否则，使用基于规则的启发式
        """

        # 尝试ML预测
        if ml_predictor is not None:
            prediction = ml_predictor.predict_best_algorithm(
                operation, tensor_size, world_size, topology
            )
            if prediction['confidence'] > 0.8:
                return prediction['algorithm']

        # 基于规则的启发式
        if operation == 'all_reduce':
            # 小消息：Tree算法（减少延迟）
            if tensor_size < 1 * 1024 * 1024:  # < 1MB
                return 'tree'

            # 大消息 + 小集群：Ring算法（高带宽利用率）
            elif world_size <= 8:
                return 'ring'

            # 大消息 + 大集群：Hierarchical算法
            elif world_size > 16:
                # 检查是否有清晰的层级结构
                if _has_hierarchical_topology(topology):
                    return 'hierarchical_ring'
                else:
                    return 'double_binary_tree'

            # 默认：Ring
            else:
                return 'ring'

        elif operation == 'all_gather':
            # AllGather通常Ring效果好
            return 'ring'

        elif operation == 'reduce_scatter':
            return 'ring'

        else:
            raise ValueError(f"Unknown operation: {operation}")
```

---

## 4. OverlapOrchestrator - 计算通讯重叠编排器

### 4.1 职责概述

OverlapOrchestrator负责最大化计算与通讯的重叠，通过异步执行和精细的依赖管理隐藏通讯延迟。

**核心职责：**
- 分析计算图的依赖关系
- 将梯度分组为buckets
- 调度异步通讯操作
- 管理CUDA streams和events
- 优化bucket大小和数量

### 4.2 接口定义

```python
class OverlapOrchestrator:
    """计算通讯重叠编排器"""

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        config: OverlapConfig = OverlapConfig()
    ):
        """
        初始化编排器

        Args:
            model: PyTorch模型（用于分析依赖）
            config: 配置对象
                - bucket_size_mb: 默认bucket大小（MB）
                - max_buckets: 最大bucket数量
                - pipeline_depth: Pipeline深度
                - dynamic_bucketing: 是否启用动态bucketing
        """
        pass

    def analyze_dependencies(
        self,
        model: torch.nn.Module
    ) -> DependencyGraph:
        """
        分析模型的计算依赖图

        Args:
            model: PyTorch模型

        Returns:
            DependencyGraph对象，表示层级间的依赖关系
        """
        pass

    def create_buckets(
        self,
        parameters: List[torch.nn.Parameter],
        bucket_size_mb: Optional[float] = None
    ) -> List[Bucket]:
        """
        创建梯度buckets

        Args:
            parameters: 模型参数列表
            bucket_size_mb: Bucket大小（MB），None表示使用默认值

        Returns:
            Bucket列表，按反向传播顺序排列
        """
        pass

    def schedule_overlap(
        self,
        computation_dag: DependencyGraph,
        communication_ops: List[CollectiveOperation]
    ) -> OverlapSchedule:
        """
        生成overlap调度计划

        Args:
            computation_dag: 计算依赖图
            communication_ops: 通讯操作列表

        Returns:
            OverlapSchedule对象，包含详细的执行时间线
        """
        pass

    def optimize_bucket_size(
        self,
        model: torch.nn.Module,
        profile_data: Optional[Dict] = None
    ) -> float:
        """
        优化bucket大小

        Args:
            model: 模型
            profile_data: 历史profiling数据

        Returns:
            最优bucket大小（MB）
        """
        pass

    def execute_with_overlap(
        self,
        forward_fn: Callable,
        backward_fn: Callable,
        communication_fn: Callable
    ) -> Any:
        """
        执行计算和通讯的overlap

        Args:
            forward_fn: 前向计算函数
            backward_fn: 反向计算函数
            communication_fn: 通讯函数

        Returns:
            计算结果
        """
        pass
```

### 4.3 数据结构

```python
@dataclass
class DependencyGraph:
    """计算依赖图"""

    # 层级列表（反向传播顺序）
    layers: List[LayerNode]

    # 依赖关系（NetworkX DAG）
    dag: nx.DiGraph

    # 层级到参数的映射
    layer_to_params: Dict[str, List[torch.nn.Parameter]]

    # 预估的计算时间
    layer_compute_time: Dict[str, float]

@dataclass
class LayerNode:
    """层级节点"""
    name: str
    module: torch.nn.Module
    parameters: List[torch.nn.Parameter]
    gradient_size: int  # 字节
    compute_time_ms: float  # 预估计算时间

@dataclass
class Bucket:
    """梯度Bucket"""
    bucket_id: int
    parameters: List[torch.nn.Parameter]
    total_size: int  # 字节
    layers: List[str]  # 包含的层名称

    # 执行状态
    gradients_ready: bool = False
    communication_started: bool = False
    communication_done: bool = False

    # 性能统计
    gradient_ready_time: Optional[float] = None
    comm_start_time: Optional[float] = None
    comm_end_time: Optional[float] = None

@dataclass
class OverlapSchedule:
    """Overlap调度计划"""

    # 时间线事件
    timeline: List[TimelineEvent]

    # Bucket调度
    bucket_schedule: Dict[int, BucketSchedule]

    # Stream分配
    stream_assignment: Dict[str, int]  # operation_id -> stream_id

    # 性能预估
    total_time_ms: float
    compute_time_ms: float
    communication_time_ms: float
    overlap_ratio: float  # overlap的通讯时间比例

@dataclass
class TimelineEvent:
    """时间线事件"""
    timestamp_ms: float
    event_type: str  # 'compute_start', 'compute_end', 'comm_start', 'comm_end'
    operation_id: str
    stream_id: int
    dependencies: List[str]  # 依赖的operation_ids
```

### 4.4 Bucket创建算法

```python
class BucketCreator:
    """Bucket创建器"""

    @staticmethod
    def create_fixed_size_buckets(
        parameters: List[torch.nn.Parameter],
        bucket_size_mb: float
    ) -> List[Bucket]:
        """
        创建固定大小的buckets

        策略：贪心算法，按反向传播顺序填充bucket
        """

        buckets = []
        current_bucket_params = []
        current_bucket_size = 0
        bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)

        # 反向遍历参数（反向传播顺序）
        for param in reversed(parameters):
            if not param.requires_grad:
                continue

            param_size = param.numel() * param.element_size()

            # 如果当前bucket加上这个参数超过限制，创建新bucket
            if current_bucket_size + param_size > bucket_size_bytes and current_bucket_params:
                buckets.append(Bucket(
                    bucket_id=len(buckets),
                    parameters=current_bucket_params[:],
                    total_size=current_bucket_size,
                    layers=[p.name for p in current_bucket_params]
                ))
                current_bucket_params = []
                current_bucket_size = 0

            current_bucket_params.append(param)
            current_bucket_size += param_size

        # 最后一个bucket
        if current_bucket_params:
            buckets.append(Bucket(
                bucket_id=len(buckets),
                parameters=current_bucket_params,
                total_size=current_bucket_size,
                layers=[p.name for p in current_bucket_params]
            ))

        # Buckets已经是反向传播顺序，最先ready的bucket在前面
        return buckets

    @staticmethod
    def create_layer_aligned_buckets(
        parameters: List[torch.nn.Parameter],
        model: torch.nn.Module,
        target_bucket_size_mb: float
    ) -> List[Bucket]:
        """
        创建对齐到层边界的buckets

        策略：尽量不拆分单个层的参数
        """

        # 1. 按层分组参数
        layer_groups = _group_params_by_layer(parameters, model)

        # 2. 贪心合并层到bucket
        buckets = []
        current_bucket = []
        current_size = 0
        target_size = int(target_bucket_size_mb * 1024 * 1024)

        for layer_name, layer_params in reversed(layer_groups):
            layer_size = sum(p.numel() * p.element_size() for p in layer_params)

            # 如果这一层本身就超过target_size，单独成bucket
            if layer_size > target_size * 1.5:
                if current_bucket:
                    buckets.append(_create_bucket(current_bucket, current_size))
                    current_bucket = []
                    current_size = 0

                buckets.append(_create_bucket([(layer_name, layer_params)], layer_size))

            # 否则尝试合并到当前bucket
            elif current_size + layer_size > target_size and current_bucket:
                buckets.append(_create_bucket(current_bucket, current_size))
                current_bucket = [(layer_name, layer_params)]
                current_size = layer_size

            else:
                current_bucket.append((layer_name, layer_params))
                current_size += layer_size

        if current_bucket:
            buckets.append(_create_bucket(current_bucket, current_size))

        return buckets
```

### 4.5 Overlap调度算法

```python
class OverlapScheduler:
    """Overlap调度器"""

    @staticmethod
    def schedule_pipeline_overlap(
        buckets: List[Bucket],
        compute_times: Dict[int, float],  # bucket_id -> compute_time_ms
        comm_times: Dict[int, float],     # bucket_id -> comm_time_ms
        pipeline_depth: int = 4
    ) -> OverlapSchedule:
        """
        生成Pipeline Overlap调度

        策略：
        1. 计算bucket i的梯度时，可以同时通讯bucket i+1, i+2, ...
        2. 保持pipeline深度不超过限制
        """

        timeline = []
        current_time = 0.0
        compute_stream = 0
        comm_stream = 1

        # 跟踪正在执行的通讯操作
        active_comms = []

        for i, bucket in enumerate(buckets):
            # 计算操作
            compute_start = current_time
            compute_end = compute_start + compute_times[bucket.bucket_id]

            timeline.append(TimelineEvent(
                timestamp_ms=compute_start,
                event_type='compute_start',
                operation_id=f'compute_bucket_{i}',
                stream_id=compute_stream,
                dependencies=[]
            ))

            timeline.append(TimelineEvent(
                timestamp_ms=compute_end,
                event_type='compute_end',
                operation_id=f'compute_bucket_{i}',
                stream_id=compute_stream,
                dependencies=[]
            ))

            # 通讯操作（异步）
            # 通讯可以在梯度ready后立即开始
            comm_start = compute_end
            comm_end = comm_start + comm_times[bucket.bucket_id]

            timeline.append(TimelineEvent(
                timestamp_ms=comm_start,
                event_type='comm_start',
                operation_id=f'comm_bucket_{i}',
                stream_id=comm_stream,
                dependencies=[f'compute_bucket_{i}']
            ))

            active_comms.append((comm_end, f'comm_bucket_{i}'))

            # 检查pipeline深度
            # 移除已完成的通讯
            active_comms = [(t, op) for t, op in active_comms if t > current_time]

            # 如果超过pipeline深度，需要等待最早的通讯完成
            if len(active_comms) > pipeline_depth:
                wait_until = active_comms[0][0]
                current_time = max(current_time, wait_until)
                timeline.append(TimelineEvent(
                    timestamp_ms=wait_until,
                    event_type='comm_end',
                    operation_id=active_comms[0][1],
                    stream_id=comm_stream,
                    dependencies=[]
                ))
                active_comms.pop(0)

            # 更新当前时间为下一个计算的开始
            current_time = compute_end

        # 等待所有通讯完成
        for comm_end, op_id in active_comms:
            timeline.append(TimelineEvent(
                timestamp_ms=comm_end,
                event_type='comm_end',
                operation_id=op_id,
                stream_id=comm_stream,
                dependencies=[]
            ))

        # 计算overlap统计
        total_compute = sum(compute_times.values())
        total_comm = sum(comm_times.values())
        final_time = max(e.timestamp_ms for e in timeline)

        overlap_ratio = (total_compute + total_comm - final_time) / total_comm

        return OverlapSchedule(
            timeline=sorted(timeline, key=lambda e: e.timestamp_ms),
            bucket_schedule={},
            stream_assignment={},
            total_time_ms=final_time,
            compute_time_ms=total_compute,
            communication_time_ms=total_comm,
            overlap_ratio=overlap_ratio
        )
```

---

## 5. CompressionManager - 通讯压缩管理器

### 5.1 职责概述

CompressionManager负责管理梯度压缩策略，在精度和通讯速度间做出智能权衡。

**核心职责：**
- 支持多种压缩方案（FP16/BF16/INT8/INT4/Top-K/Threshold）
- 自适应选择压缩级别
- 误差累积监控和补偿
- 压缩/解压缩kernel调用

### 5.2 接口定义

```python
class CompressionManager:
    """通讯压缩管理器"""

    def __init__(self, config: CompressionConfig):
        """
        初始化压缩管理器

        Args:
            config: 配置对象
                - default_compression: 默认压缩方案
                - error_budget: 允许的最大累积误差
                - adaptive_mode: 是否启用自适应压缩
        """
        pass

    def compress(
        self,
        tensor: torch.Tensor,
        compression_type: str = 'auto'
    ) -> Tuple[torch.Tensor, CompressionMetadata]:
        """
        压缩张量

        Args:
            tensor: 待压缩的张量
            compression_type: 压缩类型（'auto', 'fp16', 'bf16', 'int8', 'topk', 'threshold'）

        Returns:
            (compressed_tensor, metadata)
        """
        pass

    def decompress(
        self,
        compressed_tensor: torch.Tensor,
        metadata: CompressionMetadata
    ) -> torch.Tensor:
        """
        解压缩张量

        Args:
            compressed_tensor: 压缩后的张量
            metadata: 压缩元数据

        Returns:
            解压缩后的张量
        """
        pass

    def select_compression(
        self,
        tensor: torch.Tensor,
        context: CompressionContext
    ) -> str:
        """
        自适应选择压缩方案

        Args:
            tensor: 张量
            context: 上下文信息（消息大小、带宽、重要性等）

        Returns:
            压缩类型
        """
        pass

    def get_compression_ratio(
        self,
        compression_type: str,
        tensor: Optional[torch.Tensor] = None
    ) -> float:
        """
        获取压缩比

        Args:
            compression_type: 压缩类型
            tensor: 张量（对于adaptive compression需要）

        Returns:
            压缩比（例如，2.0表示压缩到原来的1/2）
        """
        pass

    def track_error(
        self,
        original: torch.Tensor,
        decompressed: torch.Tensor
    ) -> Dict[str, float]:
        """
        追踪压缩误差

        Args:
            original: 原始张量
            decompressed: 解压缩后的张量

        Returns:
            误差指标字典
        """
        pass
```

### 5.3 压缩方案实现

```python
# FP16压缩
class FP16Compressor:
    """FP16压缩器"""

    @staticmethod
    def compress(tensor: torch.Tensor) -> Tuple[torch.Tensor, CompressionMetadata]:
        """FP32 -> FP16"""
        compressed = tensor.half()
        metadata = CompressionMetadata(
            compression_type='fp16',
            original_dtype=tensor.dtype,
            original_shape=tensor.shape,
            compression_ratio=2.0
        )
        return compressed, metadata

    @staticmethod
    def decompress(compressed: torch.Tensor, metadata: CompressionMetadata) -> torch.Tensor:
        """FP16 -> FP32"""
        return compressed.to(metadata.original_dtype)


# INT8量化
class INT8Quantizer:
    """INT8量化器"""

    @staticmethod
    def compress(tensor: torch.Tensor) -> Tuple[torch.Tensor, CompressionMetadata]:
        """
        对称量化：x_q = round(x / scale)
        scale = max(|x|) / 127
        """
        # 计算scale
        scale = tensor.abs().max() / 127.0

        # 量化
        quantized = torch.round(tensor / scale).to(torch.int8)

        metadata = CompressionMetadata(
            compression_type='int8',
            original_dtype=tensor.dtype,
            original_shape=tensor.shape,
            compression_ratio=4.0,
            extra_data={'scale': scale.item()}
        )

        return quantized, metadata

    @staticmethod
    def decompress(quantized: torch.Tensor, metadata: CompressionMetadata) -> torch.Tensor:
        """反量化"""
        scale = metadata.extra_data['scale']
        return quantized.to(metadata.original_dtype) * scale


# Top-K稀疏化
class TopKSparsifier:
    """Top-K稀疏化器"""

    @staticmethod
    def compress(
        tensor: torch.Tensor,
        k_ratio: float = 0.01  # 保留1%的元素
    ) -> Tuple[torch.Tensor, CompressionMetadata]:
        """
        仅保留top-k最大的元素

        Returns:
            (values, indices, metadata)
        """
        numel = tensor.numel()
        k = max(1, int(numel * k_ratio))

        # 展平并取top-k
        flat_tensor = tensor.flatten()
        values, indices = torch.topk(flat_tensor.abs(), k)

        # 保留符号
        signs = torch.sign(flat_tensor[indices])
        values = values * signs

        metadata = CompressionMetadata(
            compression_type='topk',
            original_dtype=tensor.dtype,
            original_shape=tensor.shape,
            compression_ratio=1.0 / k_ratio,
            extra_data={'k': k, 'numel': numel}
        )

        # 返回稀疏表示：(values, indices)
        # 可以进一步用COO或CSR格式
        return (values, indices), metadata

    @staticmethod
    def decompress(compressed: Tuple, metadata: CompressionMetadata) -> torch.Tensor:
        """重建稀疏张量"""
        values, indices = compressed
        numel = metadata.extra_data['numel']

        # 创建零张量并填充
        flat_output = torch.zeros(numel, dtype=metadata.original_dtype, device=values.device)
        flat_output[indices] = values

        # 恢复原始形状
        return flat_output.reshape(metadata.original_shape)
```

### 5.4 自适应压缩选择

```python
class AdaptiveCompressionSelector:
    """自适应压缩选择器"""

    def __init__(self, ml_model: Optional[torch.nn.Module] = None):
        self.ml_model = ml_model
        self.error_accumulator = {}  # layer_name -> accumulated_error

    def select_compression(
        self,
        tensor: torch.Tensor,
        context: CompressionContext
    ) -> str:
        """
        基于多因素选择压缩方案

        考虑因素：
        1. 消息大小
        2. 可用带宽
        3. 层重要性
        4. 累积误差
        """

        # 因素1：消息大小
        size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024

        # 因素2：带宽压力
        bandwidth_pressure = context.bandwidth_utilization

        # 因素3：层重要性（后面的层更重要）
        layer_importance = context.layer_importance

        # 因素4：累积误差
        accumulated_error = self.error_accumulator.get(context.layer_name, 0.0)

        # 决策逻辑
        if self.ml_model is not None:
            # 使用ML模型预测
            features = torch.tensor([
                size_mb,
                bandwidth_pressure,
                layer_importance,
                accumulated_error
            ])
            prediction = self.ml_model(features)
            return _decode_compression_type(prediction)

        else:
            # 基于规则的启发式
            # 规则1：累积误差过大，暂停压缩
            if accumulated_error > context.error_budget:
                return 'none'

            # 规则2：重要层（如最后几层），使用轻压缩
            if layer_importance > 0.9:
                return 'fp16'

            # 规则3：小消息（< 1MB），不压缩
            if size_mb < 1.0:
                return 'none'

            # 规则4：中等消息 + 低带宽压力，FP16
            elif size_mb < 10.0 and bandwidth_pressure < 0.7:
                return 'fp16'

            # 规则5：大消息 + 高带宽压力，激进压缩
            elif size_mb > 100.0 and bandwidth_pressure > 0.8:
                return 'topk'  # 或 'int8'

            # 默认：BF16
            else:
                return 'bf16'

    def update_error(self, layer_name: str, error: float):
        """更新累积误差"""
        if layer_name not in self.error_accumulator:
            self.error_accumulator[layer_name] = 0.0

        # 指数移动平均
        alpha = 0.9
        self.error_accumulator[layer_name] = (
            alpha * self.error_accumulator[layer_name] + (1 - alpha) * error
        )
```

---

## 6. LoadBalancer - 负载均衡器

### 6.1 职责概述

LoadBalancer负责检测和缓解stragglers（落后的进程），确保所有进程均衡地完成工作。

**核心职责：**
- 预测每个rank的执行时间
- 检测straggler
- 动态调整负载分配
- 工作窃取和checkpoint-restart

### 6.2 接口定义

```python
class LoadBalancer:
    """负载均衡器"""

    def __init__(self, config: LoadBalancerConfig):
        """
        初始化负载均衡器

        Args:
            config: 配置对象
                - straggler_threshold: Straggler检测阈值（P90）
                - work_stealing_enabled: 是否启用工作窃取
                - checkpoint_interval: Checkpoint间隔（迭代数）
        """
        pass

    def predict_execution_time(
        self,
        rank: int,
        workload: WorkloadDescription
    ) -> float:
        """
        预测rank的执行时间

        Args:
            rank: Rank ID
            workload: 工作负载描述

        Returns:
            预测的执行时间（秒）
        """
        pass

    def detect_stragglers(
        self,
        execution_times: Dict[int, float],
        threshold_percentile: float = 90.0
    ) -> List[int]:
        """
        检测straggler ranks

        Args:
            execution_times: {rank_id: execution_time}
            threshold_percentile: 阈值百分位

        Returns:
            Straggler rank列表
        """
        pass

    def rebalance_workload(
        self,
        current_assignment: Dict[int, WorkloadShard],
        execution_times: Dict[int, float]
    ) -> Dict[int, WorkloadShard]:
        """
        重新平衡工作负载

        Args:
            current_assignment: 当前的工作分配
            execution_times: 历史执行时间

        Returns:
            新的工作分配
        """
        pass

    def enable_work_stealing(
        self,
        master_rank: int = 0
    ) -> None:
        """
        启用工作窃取机制

        Args:
            master_rank: 协调rank
        """
        pass

    def checkpoint_progress(
        self,
        rank: int,
        progress: float
    ) -> None:
        """
        记录进度checkpoint

        Args:
            rank: Rank ID
            progress: 进度（0-1）
        """
        pass
```

### 6.3 Straggler检测

```python
class StragglerDetector:
    """Straggler检测器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.execution_history: Dict[int, deque] = {}  # rank -> [exec_times]

    def update(self, rank: int, execution_time: float):
        """更新执行时间历史"""
        if rank not in self.execution_history:
            self.execution_history[rank] = deque(maxlen=self.window_size)

        self.execution_history[rank].append(execution_time)

    def detect_stragglers(
        self,
        current_times: Dict[int, float],
        method: str = 'percentile'
    ) -> List[int]:
        """
        检测stragglers

        方法：
        1. 'percentile': 超过P90的是straggler
        2. 'std_dev': 超过mean + 2*std的是straggler
        3. 'outlier': 使用IQR方法检测outliers
        """

        if method == 'percentile':
            times = list(current_times.values())
            p90 = np.percentile(times, 90)
            return [rank for rank, t in current_times.items() if t > p90]

        elif method == 'std_dev':
            times = list(current_times.values())
            mean = np.mean(times)
            std = np.std(times)
            threshold = mean + 2 * std
            return [rank for rank, t in current_times.items() if t > threshold]

        elif method == 'outlier':
            times = list(current_times.values())
            q1, q3 = np.percentile(times, [25, 75])
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            return [rank for rank, t in current_times.items() if t > upper_bound]

    def predict_straggler(
        self,
        rank: int,
        lookback: int = 10
    ) -> Tuple[bool, float]:
        """
        预测rank是否会成为straggler

        Returns:
            (is_straggler, confidence)
        """
        if rank not in self.execution_history:
            return False, 0.0

        recent_times = list(self.execution_history[rank])[-lookback:]

        # 计算趋势
        if len(recent_times) < lookback:
            return False, 0.0

        # 线性回归预测趋势
        x = np.arange(len(recent_times))
        y = np.array(recent_times)
        slope, intercept = np.polyfit(x, y, 1)

        # 预测下一个执行时间
        next_time = slope * len(recent_times) + intercept

        # 与平均值比较
        all_times = [t for history in self.execution_history.values() for t in history]
        avg_time = np.mean(all_times)

        if next_time > avg_time * 1.2:  # 超过平均20%
            confidence = min(1.0, (next_time - avg_time) / avg_time)
            return True, confidence
        else:
            return False, 0.0
```

### 6.4 负载重平衡

```python
class WorkloadRebalancer:
    """工作负载重平衡器"""

    @staticmethod
    def rebalance(
        current_assignment: Dict[int, int],  # rank -> num_samples
        execution_times: Dict[int, float]
    ) -> Dict[int, int]:
        """
        重新平衡工作负载

        目标：使所有rank的执行时间接近相等
        """

        # 1. 估计每个rank的吞吐量（samples/sec）
        throughput = {
            rank: current_assignment[rank] / execution_times[rank]
            for rank in current_assignment
        }

        # 2. 计算总工作量和目标时间
        total_samples = sum(current_assignment.values())
        total_throughput = sum(throughput.values())
        target_time = total_samples / total_throughput

        # 3. 重新分配
        new_assignment = {
            rank: int(throughput[rank] * target_time)
            for rank in current_assignment
        }

        # 4. 调整余数
        remainder = total_samples - sum(new_assignment.values())
        if remainder > 0:
            # 分配给吞吐量最高的ranks
            sorted_ranks = sorted(throughput.keys(), key=lambda r: throughput[r], reverse=True)
            for i in range(remainder):
                new_assignment[sorted_ranks[i % len(sorted_ranks)]] += 1

        return new_assignment
```

---

## 7. MLPredictor - ML预测器

### 7.1 职责概述

MLPredictor是系统的智能核心，使用机器学习模型进行各种预测和优化决策。

**核心职责：**
- 预测通讯时间
- 选择最优算法
- 预测拥塞
- 在线学习和模型更新

### 7.2 接口定义

```python
class MLPredictor:
    """ML预测器"""

    def __init__(self, config: MLConfig):
        """
        初始化ML预测器

        Args:
            config: 配置对象
                - model_type: 模型类型（'gnn', 'transformer', 'ensemble'）
                - online_learning: 是否启用在线学习
                - update_interval: 模型更新间隔（秒）
        """
        pass

    def predict_communication_time(
        self,
        operation: str,
        tensor_size: int,
        world_size: int,
        topology: ClusterTopology,
        current_state: SystemState
    ) -> Tuple[float, float]:
        """
        预测通讯时间

        Args:
            operation: 操作类型
            tensor_size: 消息大小
            world_size: 进程数
            topology: 拓扑
            current_state: 当前系统状态

        Returns:
            (predicted_time_ms, confidence)
        """
        pass

    def predict_best_algorithm(
        self,
        operation: str,
        tensor_size: int,
        world_size: int,
        topology: ClusterTopology
    ) -> Dict[str, Any]:
        """
        预测最优算法

        Returns:
            {
                'algorithm': str,
                'confidence': float,
                'top_k_algorithms': [(algorithm, score), ...]
            }
        """
        pass

    def predict_congestion(
        self,
        link: Tuple[int, int],
        time_horizon_sec: float = 1.0
    ) -> Tuple[float, float]:
        """
        预测链路拥塞

        Args:
            link: (src_gpu, dst_gpu)
            time_horizon_sec: 预测时间范围

        Returns:
            (predicted_utilization, confidence)
        """
        pass

    def update_models(
        self,
        experiences: List[Experience]
    ) -> None:
        """
        更新ML模型（在线学习）

        Args:
            experiences: 经验样本列表
        """
        pass

    def save_models(self, path: str) -> None:
        """保存模型"""
        pass

    def load_models(self, path: str) -> None:
        """加载模型"""
        pass
```

### 7.3 模型架构

```python
# GNN模型：用于拓扑感知的算法选择
class TopologyGNN(torch.nn.Module):
    """图神经网络，用于拓扑建模"""

    def __init__(
        self,
        node_features: int = 8,
        edge_features: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()

        # 图卷积层
        self.convs = torch.nn.ModuleList([
            GCNConv(node_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # 输出层：预测最优算法
        self.output = torch.nn.Linear(hidden_dim, 5)  # 5种算法

    def forward(
        self,
        x: torch.Tensor,          # [num_nodes, node_features]
        edge_index: torch.Tensor, # [2, num_edges]
        edge_attr: torch.Tensor   # [num_edges, edge_features]
    ) -> torch.Tensor:
        """
        前向传播

        Returns:
            [num_algorithms] 每个算法的得分
        """

        # 图卷积
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # 全局池化（图级别预测）
        x = torch.mean(x, dim=0)  # [hidden_dim]

        # 输出
        logits = self.output(x)  # [num_algorithms]

        return logits


# Transformer模型：用于时间序列预测
class CommunicationPatternTransformer(torch.nn.Module):
    """Transformer，用于通讯模式预测"""

    def __init__(
        self,
        feature_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 3,
        sequence_length: int = 100
    ):
        super().__init__()

        self.embedding = torch.nn.Linear(feature_dim, feature_dim * num_heads)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=feature_dim * num_heads,
            nhead=num_heads,
            dim_feedforward=feature_dim * num_heads * 4
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output = torch.nn.Linear(feature_dim * num_heads, feature_dim)

    def forward(
        self,
        x: torch.Tensor  # [sequence_length, feature_dim]
    ) -> torch.Tensor:
        """
        预测下一个通讯的特征

        Returns:
            [feature_dim] 预测的特征
        """

        # 嵌入
        x = self.embedding(x)  # [seq_len, d_model]

        # Transformer
        x = self.transformer(x)  # [seq_len, d_model]

        # 取最后一个时间步
        x = x[-1]  # [d_model]

        # 输出
        output = self.output(x)  # [feature_dim]

        return output


# RL模型：用于策略优化
class PolicyNetwork(torch.nn.Module):
    """策略网络（Actor-Critic）"""

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 10,  # 算法、路由、压缩等组合
        hidden_dim: int = 128
    ):
        super().__init__()

        # Actor network
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            (action_probs, value)
        """
        action_probs = self.actor(state)
        value = self.critic(state)

        return action_probs, value
```

### 7.4 在线学习

```python
class OnlineLearningEngine:
    """在线学习引擎"""

    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        self.models = models
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=learning_rate)
            for name, model in models.items()
        }

        self.experience_buffer = deque(maxlen=10000)
        self.batch_size = batch_size

    def add_experience(self, experience: Experience):
        """添加经验样本"""
        self.experience_buffer.append(experience)

    def update(self):
        """执行一次模型更新"""

        if len(self.experience_buffer) < self.batch_size:
            return

        # 采样batch
        batch = random.sample(self.experience_buffer, self.batch_size)

        # 更新算法选择模型
        self._update_algorithm_selector(batch)

        # 更新通讯时间预测模型
        self._update_time_predictor(batch)

    def _update_algorithm_selector(self, batch: List[Experience]):
        """更新算法选择模型"""

        model = self.models['algorithm_selector']
        optimizer = self.optimizers['algorithm_selector']

        # 准备训练数据
        states = []
        actions = []
        rewards = []

        for exp in batch:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)

        # 前向传播
        logits = model(states)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        # 策略梯度损失
        loss = -(selected_log_probs * rewards).mean()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
```

---

## 组件间协作示例

以下是一个完整的all_reduce操作的组件协作流程：

```python
def execute_optimized_all_reduce(
    tensor: torch.Tensor,
    profiler: CommunicationProfiler,
    topology_scheduler: TopologyAwareScheduler,
    collective_optimizer: AdaptiveCollectiveOptimizer,
    overlap_orchestrator: OverlapOrchestrator,
    compression_manager: CompressionManager,
    ml_predictor: MLPredictor
) -> torch.Tensor:
    """优化的all_reduce执行流程"""

    # 1. Profiler记录开始
    comm_id = generate_comm_id()
    start_time = time.time()

    # 2. 收集当前状态
    current_state = SystemState(
        topology=topology_scheduler.topology,
        bandwidth_stats=profiler.get_bandwidth_stats(),
        congestion_状态=...
    )

    # 3. ML Predictor预测最优策略
    algorithm_prediction = ml_predictor.predict_best_algorithm(
        operation='all_reduce',
        tensor_size=tensor.numel() * tensor.element_size(),
        world_size=dist.get_world_size(),
        topology=current_state.topology
    )

    # 4. Collective Optimizer生成执行计划
    exec_plan = collective_optimizer.optimize_collective(
        operation='all_reduce',
        tensor_size=tensor.numel() * tensor.element_size(),
        dtype=tensor.dtype,
        world_size=dist.get_world_size(),
        current_state=current_state
    )

    # 5. Compression Manager决定压缩策略
    compression_type = compression_manager.select_compression(
        tensor=tensor,
        context=CompressionContext(...)
    )

    # 6. 执行压缩
    if compression_type != 'none':
        compressed_tensor, compression_metadata = compression_manager.compress(
            tensor, compression_type
        )
    else:
        compressed_tensor = tensor
        compression_metadata = None

    # 7. 执行通讯（根据执行计划）
    result_tensor = _execute_collective_plan(exec_plan, compressed_tensor)

    # 8. 解压缩
    if compression_metadata is not None:
        result_tensor = compression_manager.decompress(
            result_tensor, compression_metadata
        )

    # 9. 记录性能
    end_time = time.time()
    actual_time = (end_time - start_time) * 1000  # ms

    profiler.record_communication(
        comm_id=comm_id,
        operation='all_reduce',
        tensor_size=tensor.numel() * tensor.element_size(),
        dtype=tensor.dtype,
        src_rank=dist.get_rank(),
        dst_rank=None,
        start_time=start_time,
        end_time=end_time
    )

    # 10. 更新ML模型（在线学习）
    experience = Experience(
        state=current_state,
        action=exec_plan.algorithm,
        reward=-actual_time,  # 负的时间作为reward
        next_state=...
    )
    ml_predictor.update_models([experience])

    return result_tensor
```

---

## 总结

本文档详细描述了GPU集群通讯优化系统的7个核心组件。每个组件都有清晰的职责划分、标准化的接口定义和详细的实现算法。组件间通过定义良好的协议协作，共同实现智能化的通讯优化。

下一步将在03_OPTIMIZATION_STRATEGIES.md中详细阐述各种优化策略的算法和性能模型。

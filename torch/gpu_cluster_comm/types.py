"""
GPU Cluster Communication Types
GPU集群通讯类型定义

This module defines all the core data types, enums, and data structures
used throughout the GPU cluster communication optimization system.

本模块定义GPU集群通讯优化系统中使用的所有核心数据类型、枚举和数据结构。
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import torch


# ============================================================================
# Communication Algorithms / 通讯算法
# ============================================================================

class CollectiveAlgorithm(Enum):
    """Collective communication algorithms / 集合通讯算法"""

    RING = "ring"  # Ring AllReduce
    TREE = "tree"  # Binary Tree
    DOUBLE_BINARY_TREE = "double_binary_tree"  # Double Binary Tree
    HALVING_DOUBLING = "halving_doubling"  # Recursive Halving/Doubling
    RABENSEIFNER = "rabenseifner"  # Rabenseifner's Algorithm
    HIERARCHICAL = "hierarchical"  # Two-level hierarchical
    NCCL_AUTO = "nccl_auto"  # NCCL auto-selection


class CollectiveOperation(Enum):
    """Collective operation types / 集合操作类型"""

    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    SCATTER = "scatter"
    GATHER = "gather"
    ALL_TO_ALL = "all_to_all"
    BARRIER = "barrier"


class ReductionOp(Enum):
    """Reduction operations / 归约操作"""

    SUM = "sum"
    PRODUCT = "prod"
    MIN = "min"
    MAX = "max"
    BAND = "band"  # Bitwise AND
    BOR = "bor"   # Bitwise OR
    BXOR = "bxor"  # Bitwise XOR


# ============================================================================
# Topology Types / 拓扑类型
# ============================================================================

class InterconnectType(Enum):
    """GPU interconnect types / GPU互联类型"""

    NVLINK = "nvlink"  # NVLink (high bandwidth)
    PCIE = "pcie"      # PCIe
    INFINIBAND = "infiniband"  # InfiniBand
    ETHERNET = "ethernet"      # Ethernet
    UNKNOWN = "unknown"


class NodeType(Enum):
    """Node types in topology / 拓扑中的节点类型"""

    GPU = "gpu"
    CPU = "cpu"
    SWITCH = "switch"
    NIC = "nic"  # Network Interface Card


@dataclass
class Link:
    """
    A communication link between two devices.
    两个设备之间的通讯链路。

    Attributes:
        source: Source device rank
        target: Target device rank
        interconnect: Type of interconnect
        bandwidth_gbps: Bandwidth in GB/s
        latency_us: Latency in microseconds
        bidirectional: Whether link is bidirectional
    """
    source: int
    target: int
    interconnect: InterconnectType
    bandwidth_gbps: float
    latency_us: float
    bidirectional: bool = True

    def __post_init__(self):
        """Validate link parameters"""
        if self.bandwidth_gbps <= 0:
            raise ValueError(f"Bandwidth must be positive, got {self.bandwidth_gbps}")
        if self.latency_us < 0:
            raise ValueError(f"Latency cannot be negative, got {self.latency_us}")


@dataclass
class GPUDevice:
    """
    GPU device information.
    GPU设备信息。

    Attributes:
        rank: Global rank of this GPU
        local_rank: Rank within node
        node_id: ID of the node this GPU belongs to
        device_id: CUDA device ID
        pcie_bus_id: PCIe bus ID
        compute_capability: CUDA compute capability (e.g., (8, 0) for SM80)
        memory_gb: Total memory in GB
        nvlink_domain: NVLink domain ID (GPUs in same domain can use NVLink)
    """
    rank: int
    local_rank: int
    node_id: int
    device_id: int
    pcie_bus_id: str
    compute_capability: Tuple[int, int]
    memory_gb: float
    nvlink_domain: Optional[int] = None

    def __hash__(self):
        return hash(self.rank)

    def __eq__(self, other):
        if isinstance(other, GPUDevice):
            return self.rank == other.rank
        return False


@dataclass
class TopologyNode:
    """
    Node in the cluster topology.
    集群拓扑中的节点。

    Attributes:
        node_id: Unique node identifier
        gpus: List of GPUs in this node
        cpu_cores: Number of CPU cores
        memory_gb: Total system memory in GB
        network_interfaces: Network interfaces available
    """
    node_id: int
    gpus: List[GPUDevice] = field(default_factory=list)
    cpu_cores: int = 0
    memory_gb: float = 0.0
    network_interfaces: List[str] = field(default_factory=list)


# ============================================================================
# Communication Planning / 通讯规划
# ============================================================================

@dataclass
class CommunicationStep:
    """
    A single step in a communication plan.
    通讯计划中的单个步骤。

    Attributes:
        step_id: Sequential step identifier
        operation: Type of operation
        source_ranks: Source rank(s)
        target_ranks: Target rank(s)
        message_size: Size of message in bytes
        chunk_id: For chunked communications
        dependencies: Steps that must complete before this one
    """
    step_id: int
    operation: str  # "send", "recv", "reduce", etc.
    source_ranks: List[int]
    target_ranks: List[int]
    message_size: int
    chunk_id: Optional[int] = None
    dependencies: List[int] = field(default_factory=list)


@dataclass
class CommunicationPlan:
    """
    Complete communication plan for a collective operation.
    集合操作的完整通讯计划。

    Attributes:
        operation: Collective operation type
        algorithm: Algorithm used
        num_ranks: Total number of ranks
        message_size: Total message size in bytes
        steps: Ordered list of communication steps
        estimated_time_us: Estimated completion time in microseconds
        num_chunks: Number of chunks for pipelined execution
    """
    operation: CollectiveOperation
    algorithm: CollectiveAlgorithm
    num_ranks: int
    message_size: int
    steps: List[CommunicationStep] = field(default_factory=list)
    estimated_time_us: float = 0.0
    num_chunks: int = 1


# ============================================================================
# Compression Types / 压缩类型
# ============================================================================

class CompressionStrategy(Enum):
    """Compression strategies for communication / 通讯压缩策略"""

    NONE = "none"
    FP16 = "fp16"  # Half precision
    BF16 = "bf16"  # Brain float 16
    INT8 = "int8"  # 8-bit integer quantization
    FP8 = "fp8"    # 8-bit floating point (if supported)
    SPARSIFICATION = "sparse"  # Sparse gradients
    QUANTIZATION = "quantize"  # General quantization
    TOP_K = "topk"  # Top-k sparsification
    RANDOM_K = "randomk"  # Random-k sparsification


@dataclass
class CompressedTensor:
    """
    A compressed tensor representation.
    压缩后的张量表示。

    Attributes:
        data: Compressed data
        strategy: Compression strategy used
        original_shape: Original tensor shape
        original_dtype: Original data type
        metadata: Additional metadata for decompression
        compression_ratio: Achieved compression ratio
    """
    data: torch.Tensor
    strategy: CompressionStrategy
    original_shape: torch.Size
    original_dtype: torch.dtype
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_ratio: float = 1.0

    @property
    def compressed_size(self) -> int:
        """Get compressed data size in bytes"""
        return self.data.numel() * self.data.element_size()

    @property
    def original_size(self) -> int:
        """Get original data size in bytes"""
        numel = 1
        for dim in self.original_shape:
            numel *= dim
        return numel * torch.tensor([], dtype=self.original_dtype).element_size()


# ============================================================================
# Profiling and Metrics / 性能分析和指标
# ============================================================================

@dataclass
class CommMetrics:
    """
    Communication metrics for a single operation.
    单个操作的通讯指标。

    Attributes:
        operation: Operation type
        algorithm: Algorithm used
        message_size: Message size in bytes
        latency_us: Actual latency in microseconds
        bandwidth_gbps: Achieved bandwidth in GB/s
        num_ranks: Number of participating ranks
        timestamp: Unix timestamp
        rank: Rank that recorded this metric
        metadata: Additional metadata
    """
    operation: str
    algorithm: str
    message_size: int
    latency_us: float
    bandwidth_gbps: float
    num_ranks: int
    timestamp: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def busbw_gbps(self) -> float:
        """
        Calculate bus bandwidth (accounting for bidirectional traffic).
        计算总线带宽（考虑双向流量）。

        For AllReduce, the algorithm bandwidth is roughly 2*(N-1)/N times
        the bus bandwidth, where N is the number of GPUs.
        """
        if self.num_ranks <= 1:
            return self.bandwidth_gbps
        factor = 2.0 * (self.num_ranks - 1) / self.num_ranks
        return self.bandwidth_gbps / factor


@dataclass
class CommunicationPattern:
    """
    Detected communication pattern.
    检测到的通讯模式。

    Attributes:
        pattern_type: Type of pattern detected
        frequency: How often this pattern occurs
        avg_message_size: Average message size
        ranks_involved: Ranks involved in this pattern
        bottleneck_links: Links that are bottlenecks
    """
    pattern_type: str
    frequency: int
    avg_message_size: float
    ranks_involved: Set[int] = field(default_factory=set)
    bottleneck_links: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class Bottleneck:
    """
    Identified performance bottleneck.
    识别的性能瓶颈。

    Attributes:
        bottleneck_type: Type of bottleneck
        severity: Severity score (0-1, higher is worse)
        location: Where the bottleneck occurs
        description: Human-readable description
        suggested_fix: Suggested remediation
    """
    bottleneck_type: str
    severity: float  # 0.0 to 1.0
    location: str
    description: str
    suggested_fix: str

    def __post_init__(self):
        """Validate severity"""
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"Severity must be in [0, 1], got {self.severity}")


# ============================================================================
# Scheduling Types / 调度类型
# ============================================================================

class OperationType(Enum):
    """Operation types for scheduling / 调度的操作类型"""

    COMPUTE = "compute"
    COMMUNICATION = "communication"
    MEMORY = "memory"


@dataclass
class ScheduleNode:
    """
    A node in the dependency graph for scheduling.
    调度依赖图中的节点。

    Attributes:
        node_id: Unique node identifier
        op_type: Type of operation
        name: Human-readable name
        duration_us: Estimated duration in microseconds
        dependencies: Nodes that must complete before this one
        tensor_size: Size of tensor involved (if applicable)
    """
    node_id: int
    op_type: OperationType
    name: str
    duration_us: float
    dependencies: Set[int] = field(default_factory=set)
    tensor_size: Optional[int] = None

    def add_dependency(self, dep_id: int) -> None:
        """Add a dependency"""
        self.dependencies.add(dep_id)

    def remove_dependency(self, dep_id: int) -> None:
        """Remove a dependency"""
        self.dependencies.discard(dep_id)


@dataclass
class SchedulePlan:
    """
    Complete schedule plan for overlapping compute and communication.
    重叠计算和通讯的完整调度计划。

    Attributes:
        nodes: All nodes in the schedule
        critical_path_us: Critical path length in microseconds
        parallelism_factor: Average parallelism factor
        compute_time_us: Total compute time
        comm_time_us: Total communication time
        overlap_time_us: Amount of overlapped time
    """
    nodes: List[ScheduleNode] = field(default_factory=list)
    critical_path_us: float = 0.0
    parallelism_factor: float = 1.0
    compute_time_us: float = 0.0
    comm_time_us: float = 0.0
    overlap_time_us: float = 0.0

    @property
    def efficiency(self) -> float:
        """
        Calculate scheduling efficiency.
        计算调度效率。

        Returns:
            Ratio of overlapped time to total communication time
        """
        if self.comm_time_us == 0:
            return 1.0
        return self.overlap_time_us / self.comm_time_us


# ============================================================================
# Message Coalescing Types / 消息聚合类型
# ============================================================================

@dataclass
class Message:
    """
    A single message to be sent.
    待发送的单个消息。

    Attributes:
        msg_id: Unique message identifier
        source: Source rank
        target: Target rank
        data: Message data (tensor)
        size_bytes: Size in bytes
        priority: Message priority (higher = more important)
        timestamp: Creation timestamp
    """
    msg_id: int
    source: int
    target: int
    data: torch.Tensor
    size_bytes: int
    priority: int = 0
    timestamp: float = 0.0


@dataclass
class CoalescedMessage:
    """
    Multiple messages coalesced into one.
    聚合后的多个消息。

    Attributes:
        original_messages: Original individual messages
        coalesced_data: Combined data buffer
        offsets: Offset of each message in the buffer
        total_size: Total size in bytes
    """
    original_messages: List[Message]
    coalesced_data: torch.Tensor
    offsets: List[int]
    total_size: int

    def split(self) -> List[Message]:
        """
        Split coalesced message back into original messages.
        将聚合消息拆分回原始消息。
        """
        messages = []
        for i, msg in enumerate(self.original_messages):
            start = self.offsets[i]
            end = self.offsets[i + 1] if i + 1 < len(self.offsets) else self.total_size
            # Create new message with sliced data
            messages.append(Message(
                msg_id=msg.msg_id,
                source=msg.source,
                target=msg.target,
                data=self.coalesced_data[start:end].clone(),
                size_bytes=end - start,
                priority=msg.priority,
                timestamp=msg.timestamp
            ))
        return messages


# ============================================================================
# Load Balancing Types / 负载均衡类型
# ============================================================================

@dataclass
class WorkloadStats:
    """
    Workload statistics for a rank.
    某个rank的工作负载统计。

    Attributes:
        rank: Rank identifier
        compute_time_us: Time spent on computation
        comm_time_us: Time spent on communication
        idle_time_us: Idle time
        memory_used_gb: Memory currently used
        throughput: Items processed per second
    """
    rank: int
    compute_time_us: float
    comm_time_us: float
    idle_time_us: float
    memory_used_gb: float
    throughput: float

    @property
    def total_time_us(self) -> float:
        """Total time"""
        return self.compute_time_us + self.comm_time_us + self.idle_time_us

    @property
    def efficiency(self) -> float:
        """Compute efficiency (ratio of compute to total time)"""
        total = self.total_time_us
        if total == 0:
            return 0.0
        return self.compute_time_us / total

    @property
    def is_straggler(self, threshold: float = 0.8) -> bool:
        """
        Check if this rank is a straggler.
        检查该rank是否为落后者。

        Args:
            threshold: Efficiency threshold below which rank is a straggler
        """
        return self.efficiency < threshold


# ============================================================================
# Configuration Types / 配置类型
# ============================================================================

@dataclass
class OptimizationConfig:
    """
    Configuration for communication optimization.
    通讯优化配置。

    Attributes:
        enable_compression: Enable gradient compression
        compression_strategy: Default compression strategy
        enable_overlap: Enable compute-communication overlap
        bucket_size_mb: Gradient bucketing size in MB
        enable_coalescing: Enable message coalescing
        coalescing_threshold_kb: Threshold for coalescing in KB
        enable_profiling: Enable performance profiling
        profile_interval: Profiling interval in iterations
    """
    enable_compression: bool = False
    compression_strategy: CompressionStrategy = CompressionStrategy.NONE
    enable_overlap: bool = True
    bucket_size_mb: float = 25.0
    enable_coalescing: bool = True
    coalescing_threshold_kb: float = 64.0
    enable_profiling: bool = False
    profile_interval: int = 10

    def validate(self) -> None:
        """Validate configuration"""
        if self.bucket_size_mb <= 0:
            raise ValueError(f"Bucket size must be positive, got {self.bucket_size_mb}")
        if self.coalescing_threshold_kb < 0:
            raise ValueError(f"Coalescing threshold cannot be negative")
        if self.profile_interval <= 0:
            raise ValueError(f"Profile interval must be positive")

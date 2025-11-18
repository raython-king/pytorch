"""
Data Pipeline Orchestrator

Coordinates multiple agents to manage the multi-level data pipeline:
Disk → Memory → Redis → GPU
"""

import time
import threading
from typing import Dict, Any, List, Optional, Iterator
from collections import defaultdict

import torch

from .config import DataPipelineConfig
from .agents.base_agent import (
    DataRequest,
    DataItem,
    PipelineEnvironment,
    AgentDecision,
    AgentAction,
)
from .agents.disk_agent import DiskReaderAgent
from .agents.memory_agent import MemoryCacheAgent
from .agents.redis_agent import RedisCacheAgent
from .agents.gpu_agent import GPUTransferAgent


class DataPipelineOrchestrator:
    """
    Orchestrator for multi-agent data pipeline system.

    Coordinates agents across different layers to optimize data loading:
    - Disk I/O agent
    - Memory cache agent
    - Redis cache agent (optional)
    - GPU transfer agent

    The orchestrator implements intelligent data flow management with:
    - Dynamic prefetching
    - Multi-level caching
    - Load balancing
    - Performance monitoring
    """

    def __init__(
        self,
        dataset,
        config: Optional[DataPipelineConfig] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            dataset: PyTorch dataset to load from
            config: Pipeline configuration
        """
        self.dataset = dataset
        self.config = config or DataPipelineConfig()
        self.config.validate()

        # Initialize agents
        self.agents: Dict[str, Any] = {}
        self._initialize_agents()

        # Environment state
        self.environment = self._create_initial_environment()
        self.env_lock = threading.RLock()

        # Request queue
        self.pending_requests: List[DataRequest] = []
        self.request_lock = threading.Lock()

        # Statistics
        self.stats = defaultdict(lambda: defaultdict(float))
        self.total_requests = 0
        self.start_time = time.time()

        # Access pattern tracking
        self.access_history: List[Any] = []
        self.access_frequency: Dict[Any, int] = defaultdict(int)

        # Performance tracking
        self.latencies: List[float] = []
        self.throughput_samples: List[int] = []

        # Thread for async operations
        self.running = False
        self.background_thread: Optional[threading.Thread] = None

    def _initialize_agents(self) -> None:
        """Initialize all pipeline agents"""
        config_dict = self.config.to_dict()

        # Disk reader agent
        self.agents["disk"] = DiskReaderAgent(
            agent_id="disk_reader_0",
            config=config_dict,
            dataset=self.dataset
        )

        # Memory cache agent
        self.agents["memory"] = MemoryCacheAgent(
            agent_id="memory_manager_0",
            config=config_dict
        )

        # Redis cache agent (if enabled)
        if self.config.redis.enabled:
            self.agents["redis"] = RedisCacheAgent(
                agent_id="redis_manager_0",
                config=config_dict
            )

        # GPU transfer agent (if enabled)
        if self.config.gpu.enabled and torch.cuda.is_available():
            self.agents["gpu"] = GPUTransferAgent(
                agent_id="gpu_transfer_0",
                config=config_dict
            )

    def _create_initial_environment(self) -> PipelineEnvironment:
        """Create initial environment state"""
        return PipelineEnvironment(
            timestamp=time.time(),
            pending_requests=[],
            cached_items={
                "disk": [],
                "memory": [],
                "redis": [],
                "gpu": [],
            },
            memory_available_bytes=0,
            memory_pressure=0.0,
            gpu_memory_available_bytes=0,
            disk_io_bandwidth_mbps=100.0,
            network_bandwidth_mbps=1000.0,
            cache_hit_rates={
                "disk": 0.0,
                "memory": 0.0,
                "redis": 0.0,
                "gpu": 0.0,
            },
            average_latencies={
                "disk": 0.0,
                "memory": 0.0,
                "redis": 0.0,
                "gpu": 0.0,
            },
            throughput_mbps=0.0,
            recent_access_sequence=[],
            access_frequency={},
        )

    def get_item(self, sample_id: Any) -> Any:
        """
        Get a data item through the pipeline.

        This is the main entry point for data loading.
        The orchestrator will:
        1. Check GPU prefetch queue
        2. Check Redis cache
        3. Check memory cache
        4. Load from disk
        5. Populate caches and GPU for future access

        Args:
            sample_id: Sample identifier

        Returns:
            Data item (on GPU if enabled)
        """
        start_time = time.time()
        request = DataRequest(sample_id=sample_id, timestamp=start_time)

        # Update access tracking
        self.access_history.append(sample_id)
        self.access_frequency[sample_id] += 1

        # Keep history bounded
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-1000:]

        try:
            # Try GPU prefetch queue first
            if "gpu" in self.agents:
                gpu_agent = self.agents["gpu"]
                gpu_item = gpu_agent.get_from_prefetch_queue(sample_id)

                if gpu_item:
                    self._record_hit("gpu", time.time() - start_time)
                    self.total_requests += 1
                    return gpu_item.data

            # Try Redis cache
            if "redis" in self.agents:
                redis_agent = self.agents["redis"]
                redis_item = redis_agent.get_from_cache(sample_id)

                if redis_item:
                    self._record_hit("redis", time.time() - start_time)

                    # Transfer to GPU if enabled
                    if "gpu" in self.agents:
                        gpu_item = self.agents["gpu"].transfer_to_gpu(redis_item)
                        self.total_requests += 1
                        return gpu_item.data if gpu_item else redis_item.data

                    self.total_requests += 1
                    return redis_item.data

            # Try memory cache
            memory_agent = self.agents["memory"]
            memory_item = memory_agent.get_from_cache(sample_id)

            if memory_item:
                self._record_hit("memory", time.time() - start_time)

                # Cache to Redis if enabled
                if "redis" in self.agents:
                    self.agents["redis"].set_to_cache(memory_item)

                # Transfer to GPU if enabled
                if "gpu" in self.agents:
                    gpu_item = self.agents["gpu"].transfer_to_gpu(memory_item)
                    self.total_requests += 1
                    return gpu_item.data if gpu_item else memory_item.data

                self.total_requests += 1
                return memory_item.data

            # Load from disk
            disk_agent = self.agents["disk"]
            disk_item = disk_agent._read_from_disk(sample_id)

            if disk_item:
                self._record_hit("disk", time.time() - start_time)

                # Cache to memory
                memory_agent.add_to_cache(disk_item)

                # Cache to Redis if enabled
                if "redis" in self.agents:
                    self.agents["redis"].set_to_cache(disk_item)

                # Transfer to GPU if enabled
                if "gpu" in self.agents:
                    gpu_item = self.agents["gpu"].transfer_to_gpu(disk_item)
                    self.total_requests += 1
                    return gpu_item.data if gpu_item else disk_item.data

                self.total_requests += 1
                return disk_item.data

            # Fallback to direct dataset access
            data = self.dataset[sample_id]
            self.total_requests += 1

            # Transfer to GPU if enabled and data is tensor
            if "gpu" in self.agents and isinstance(data, torch.Tensor):
                data = self.agents["gpu"]._to_gpu_single(data)

            return data

        except Exception as e:
            print(f"Error loading sample {sample_id}: {e}")
            # Fallback to dataset
            data = self.dataset[sample_id]
            self.total_requests += 1
            return data

        finally:
            # Record latency
            latency = time.time() - start_time
            self.latencies.append(latency)

            if len(self.latencies) > 1000:
                self.latencies = self.latencies[-1000:]

            # Update environment periodically
            if self.total_requests % 10 == 0:
                self._update_environment()

            # Trigger prefetching periodically
            if self.total_requests % self.config.prefetch.adaptation_interval == 0:
                self._trigger_prefetch()

    def _record_hit(self, layer: str, latency: float) -> None:
        """Record cache hit for a layer"""
        self.stats[layer]["hits"] += 1
        self.stats[layer]["total_latency"] += latency

    def _update_environment(self) -> None:
        """Update environment state for agents"""
        with self.env_lock:
            # Calculate hit rates
            for layer in ["disk", "memory", "redis", "gpu"]:
                hits = self.stats[layer]["hits"]
                total = self.total_requests

                if total > 0:
                    self.environment.cache_hit_rates[layer] = hits / total

                total_latency = self.stats[layer]["total_latency"]
                if hits > 0:
                    self.environment.average_latencies[layer] = (total_latency / hits) * 1000  # ms

            # Update access patterns
            self.environment.recent_access_sequence = self.access_history[-100:]
            self.environment.access_frequency = dict(self.access_frequency)

            # Update throughput
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                samples_per_sec = self.total_requests / elapsed_time
                # Rough estimate of throughput (assuming avg 1MB per sample)
                self.environment.throughput_mbps = samples_per_sec

            # Update memory pressure
            if "memory" in self.agents:
                memory_stats = self.agents["memory"].get_statistics()
                self.environment.memory_pressure = memory_stats.get("utilization", 0.0)

            # Notify agents of environment update
            for agent in self.agents.values():
                agent.observe(self.environment)

    def _trigger_prefetch(self) -> None:
        """Trigger prefetching based on access patterns"""
        if not self.config.prefetch.use_ml_predictor:
            return

        # Get predictions from disk agent
        disk_agent = self.agents["disk"]

        decision = disk_agent.decide()

        if decision.action == AgentAction.PREFETCH_DATA and decision.target_data:
            # Prefetch to memory
            for sample_id in decision.target_data:
                try:
                    disk_item = disk_agent._read_from_disk(sample_id)

                    if disk_item:
                        # Add to memory cache
                        self.agents["memory"].add_to_cache(disk_item)

                        # Optionally add to GPU prefetch queue
                        if "gpu" in self.agents:
                            self.agents["gpu"].add_to_prefetch_queue(disk_item)

                except Exception as e:
                    print(f"Error prefetching {sample_id}: {e}")

    def __iter__(self) -> Iterator[Any]:
        """Iterate through dataset"""
        for idx in range(len(self.dataset)):
            yield self.get_item(idx)

    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.dataset)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "total_requests": self.total_requests,
            "uptime_seconds": time.time() - self.start_time,
        }

        # Per-layer statistics
        for layer_name, agent in self.agents.items():
            if hasattr(agent, "get_statistics"):
                stats[layer_name] = agent.get_statistics()

        # Overall hit rates
        stats["overall_cache_hit_rates"] = self.environment.cache_hit_rates.copy()

        # Latency statistics
        if self.latencies:
            import statistics
            stats["latency_ms"] = {
                "mean": statistics.mean(self.latencies) * 1000,
                "median": statistics.median(self.latencies) * 1000,
                "min": min(self.latencies) * 1000,
                "max": max(self.latencies) * 1000,
            }

        # Throughput
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            stats["throughput_samples_per_sec"] = self.total_requests / elapsed

        return stats

    def reset_statistics(self) -> None:
        """Reset all statistics"""
        self.stats.clear()
        self.total_requests = 0
        self.start_time = time.time()
        self.latencies.clear()
        self.throughput_samples.clear()
        self.access_history.clear()
        self.access_frequency.clear()

        for agent in self.agents.values():
            agent.reset()

    def clear_caches(self) -> None:
        """Clear all caches"""
        if "memory" in self.agents:
            self.agents["memory"].clear()

        if "redis" in self.agents:
            self.agents["redis"].clear()

        if "gpu" in self.agents:
            self.agents["gpu"].clear_prefetch_queue()

    def shutdown(self) -> None:
        """Shutdown orchestrator and cleanup resources"""
        self.running = False

        if self.background_thread:
            self.background_thread.join(timeout=5.0)

        # Synchronize GPU streams
        if "gpu" in self.agents:
            self.agents["gpu"].synchronize()

        # Clear caches
        self.clear_caches()

        print("Data pipeline orchestrator shut down")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except Exception:
            pass


class DataPipelineDataLoader:
    """
    Wrapper around DataPipelineOrchestrator that provides
    DataLoader-like interface.
    """

    def __init__(
        self,
        dataset,
        config: Optional[DataPipelineConfig] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs
    ):
        """
        Initialize DataLoader with pipeline.

        Args:
            dataset: PyTorch dataset
            config: Pipeline configuration
            batch_size: Batch size
            shuffle: Whether to shuffle data
            **kwargs: Additional arguments (for compatibility)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Create orchestrator
        self.orchestrator = DataPipelineOrchestrator(dataset, config)

        # Create indices
        self.indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[Any]:
        """Iterate through batches"""
        # Shuffle if needed
        indices = self.indices.copy()

        if self.shuffle:
            import random
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            # Load batch through pipeline
            batch = [self.orchestrator.get_item(idx) for idx in batch_indices]

            # Stack if possible
            if batch and isinstance(batch[0], torch.Tensor):
                try:
                    batch = torch.stack(batch)
                except Exception:
                    pass  # Keep as list if stacking fails

            yield batch

    def __len__(self) -> int:
        """Get number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.orchestrator.get_statistics()

    def shutdown(self) -> None:
        """Shutdown dataloader"""
        self.orchestrator.shutdown()

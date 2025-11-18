"""
GPU Transfer Agent

Manages data transfer to GPU with intelligent prefetching and stream management.
"""

import threading
import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Deque

import torch

from .base_agent import (
    BaseDataPipelineAgent,
    AgentRole,
    AgentAction,
    AgentDecision,
    DataRequest,
    DataItem,
    PipelineEnvironment,
)


class GPUTransferAgent(BaseDataPipelineAgent):
    """
    Agent responsible for managing GPU data transfers.

    Features:
    - CUDA stream management
    - Non-blocking transfers
    - GPU memory prefetching
    - Pin memory optimization
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentRole.GPU_TRANSFER, config)

        self.gpu_config = config.get("gpu", {})
        self.device_str = self.gpu_config.get("device", "cuda:0")

        # Check if CUDA available
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            print(f"Warning: CUDA not available. GPU agent {agent_id} disabled.")
            self.device = torch.device("cpu")
            self.enabled = False
        else:
            self.device = torch.device(self.device_str)
            self.enabled = True

        # CUDA streams
        self.streams: List[torch.cuda.Stream] = []
        if self.enabled and self.gpu_config.get("use_streams", True):
            num_streams = self.gpu_config.get("num_streams", 2)
            for _ in range(num_streams):
                self.streams.append(torch.cuda.Stream(device=self.device))

        self.current_stream_idx = 0
        self.stream_lock = threading.Lock()

        # GPU prefetch queue
        self.prefetch_queue: Deque[DataItem] = deque(
            maxlen=self.gpu_config.get("prefetch_queue_size", 2)
        )
        self.prefetch_lock = threading.Lock()

        # Statistics
        self.transfers = 0
        self.total_transfer_time = 0.0
        self.total_bytes_transferred = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

    def observe(self, environment: PipelineEnvironment) -> None:
        """Observe environment and update state"""
        if not self.enabled:
            return

        gpu_latency = environment.average_latencies.get("gpu", 0.0)

        # Get GPU memory info
        if self.cuda_available:
            try:
                gpu_memory_available = torch.cuda.mem_get_info(self.device)[0]
                gpu_memory_total = torch.cuda.mem_get_info(self.device)[1]
                gpu_utilization = 1.0 - (gpu_memory_available / gpu_memory_total)
            except Exception:
                gpu_utilization = 0.5
        else:
            gpu_utilization = 0.0

        self.update_state(
            average_latency_ms=gpu_latency,
            current_load=gpu_utilization,
        )

    def decide(self, request: Optional[DataRequest] = None) -> AgentDecision:
        """Decide on GPU transfer actions"""
        if not self.enabled:
            return AgentDecision(
                action=AgentAction.NO_ACTION,
                confidence=1.0,
                reasoning="GPU not available",
            )

        if request is not None:
            # Check if data already in prefetch queue
            with self.prefetch_lock:
                for item in self.prefetch_queue:
                    if item.sample_id == request.sample_id:
                        self.prefetch_hits += 1
                        return AgentDecision(
                            action=AgentAction.NO_ACTION,
                            confidence=1.0,
                            target_data=[request.sample_id],
                            reasoning="Data already prefetched to GPU",
                        )

            # Need to transfer
            self.prefetch_misses += 1
            return AgentDecision(
                action=AgentAction.TRANSFER_TO_GPU,
                confidence=1.0,
                target_data=[request.sample_id],
                reasoning="Transfer data to GPU",
            )

        # Check if we should prefetch
        if self.gpu_config.get("prefetch_to_gpu", True):
            queue_size = len(self.prefetch_queue)
            max_size = self.gpu_config.get("prefetch_queue_size", 2)

            if queue_size < max_size:
                return AgentDecision(
                    action=AgentAction.PREFETCH_DATA,
                    confidence=0.7,
                    reasoning="GPU prefetch queue has space",
                    expected_benefit=0.4,
                )

        return AgentDecision(
            action=AgentAction.NO_ACTION,
            confidence=1.0,
        )

    def execute(self, decision: AgentDecision) -> Tuple[bool, Any]:
        """Execute GPU transfer decision"""
        if not self.enabled:
            return False, "GPU not available"

        try:
            if decision.action == AgentAction.TRANSFER_TO_GPU:
                # This is handled by transfer_to_gpu method
                return True, None

            elif decision.action == AgentAction.PREFETCH_DATA:
                # Prefetch handled externally
                return True, None

            return True, None

        except Exception as e:
            self.state.error_count += 1
            return False, str(e)

    def learn(self, reward: float, next_environment: PipelineEnvironment) -> None:
        """Learn from outcomes and adapt"""
        self.reward_history.append(reward)

        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

        # Adapt prefetch queue size based on hit rate
        total_requests = self.prefetch_hits + self.prefetch_misses
        if total_requests > 100:
            hit_rate = self.prefetch_hits / total_requests

            current_size = self.gpu_config.get("prefetch_queue_size", 2)

            if hit_rate < 0.3:
                # Low hit rate - increase queue
                self.gpu_config["prefetch_queue_size"] = min(8, current_size + 1)
            elif hit_rate > 0.9 and current_size > 1:
                # Very high hit rate - can reduce queue to save memory
                self.gpu_config["prefetch_queue_size"] = max(1, current_size - 1)

    def transfer_to_gpu(
        self,
        data_item: DataItem,
        non_blocking: bool = None
    ) -> Optional[DataItem]:
        """Transfer data to GPU"""
        if not self.enabled:
            return data_item  # Return as-is if GPU not available

        start_time = time.time()

        try:
            data = data_item.data

            # Convert to tensor if needed
            if not isinstance(data, torch.Tensor):
                if isinstance(data, (tuple, list)):
                    # Handle tuple/list of tensors
                    data = [
                        self._to_gpu_single(x, non_blocking)
                        for x in data
                    ]
                else:
                    data = self._to_gpu_single(data, non_blocking)
            else:
                data = self._to_gpu_single(data, non_blocking)

            # Create new data item
            gpu_item = DataItem(
                sample_id=data_item.sample_id,
                data=data,
                size_bytes=data_item.size_bytes,
                location="gpu",
                access_count=data_item.access_count,
                last_access_time=time.time(),
                load_time=time.time() - start_time,
            )

            # Update statistics
            self.transfers += 1
            self.total_transfer_time += gpu_item.load_time
            self.total_bytes_transferred += data_item.size_bytes
            self.state.total_requests += 1
            self.state.total_bytes_processed += data_item.size_bytes

            return gpu_item

        except Exception as e:
            self.state.error_count += 1
            print(f"Error transferring to GPU: {e}")
            return data_item  # Return original on error

    def _to_gpu_single(self, data: Any, non_blocking: bool = None) -> Any:
        """Transfer single item to GPU"""
        if not isinstance(data, torch.Tensor):
            return data  # Can't transfer non-tensors

        if data.device == self.device:
            return data  # Already on target device

        # Use configured non_blocking if not specified
        if non_blocking is None:
            non_blocking = self.gpu_config.get("non_blocking", True)

        # Use CUDA stream if available
        if self.streams:
            with self.stream_lock:
                stream = self.streams[self.current_stream_idx]
                self.current_stream_idx = (self.current_stream_idx + 1) % len(self.streams)

            with torch.cuda.stream(stream):
                return data.to(self.device, non_blocking=non_blocking)
        else:
            return data.to(self.device, non_blocking=non_blocking)

    def add_to_prefetch_queue(self, data_item: DataItem) -> bool:
        """Add item to GPU prefetch queue"""
        if not self.enabled:
            return False

        # Transfer to GPU
        gpu_item = self.transfer_to_gpu(data_item)

        if gpu_item and gpu_item.location == "gpu":
            with self.prefetch_lock:
                self.prefetch_queue.append(gpu_item)
            return True

        return False

    def get_from_prefetch_queue(self, sample_id: Any) -> Optional[DataItem]:
        """Get item from prefetch queue"""
        with self.prefetch_lock:
            for item in self.prefetch_queue:
                if item.sample_id == sample_id:
                    # Don't remove - might be reused
                    item.access_count += 1
                    item.last_access_time = time.time()
                    return item

        return None

    def synchronize(self) -> None:
        """Synchronize all CUDA streams"""
        if self.enabled and self.streams:
            for stream in self.streams:
                stream.synchronize()

    def get_statistics(self) -> Dict[str, Any]:
        """Get GPU transfer statistics"""
        avg_transfer_time = (
            self.total_transfer_time / self.transfers
            if self.transfers > 0 else 0.0
        )

        total_mb = self.total_bytes_transferred / (1024 ** 2)
        total_time_s = self.total_transfer_time

        bandwidth = total_mb / total_time_s if total_time_s > 0 else 0.0

        total_prefetch = self.prefetch_hits + self.prefetch_misses
        prefetch_hit_rate = (
            self.prefetch_hits / total_prefetch
            if total_prefetch > 0 else 0.0
        )

        stats = {
            "enabled": self.enabled,
            "device": str(self.device),
            "total_transfers": self.transfers,
            "average_transfer_time_ms": avg_transfer_time * 1000,
            "total_mb_transferred": total_mb,
            "bandwidth_mbps": bandwidth,
            "prefetch_hit_rate": prefetch_hit_rate,
            "prefetch_queue_size": len(self.prefetch_queue),
        }

        # Add GPU memory stats if available
        if self.cuda_available:
            try:
                mem_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                mem_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
                mem_free, mem_total = torch.cuda.mem_get_info(self.device)

                stats.update({
                    "gpu_memory_allocated_mb": mem_allocated,
                    "gpu_memory_reserved_mb": mem_reserved,
                    "gpu_memory_free_mb": mem_free / (1024 ** 2),
                    "gpu_memory_total_mb": mem_total / (1024 ** 2),
                })
            except Exception:
                pass

        return stats

    def clear_prefetch_queue(self) -> None:
        """Clear the prefetch queue"""
        with self.prefetch_lock:
            # Clear queue and free GPU memory
            for item in self.prefetch_queue:
                if isinstance(item.data, torch.Tensor):
                    del item.data

            self.prefetch_queue.clear()

        # Force garbage collection
        if self.cuda_available:
            torch.cuda.empty_cache()

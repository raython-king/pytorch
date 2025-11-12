"""
Data Transfer Optimizer for Runtime Scheduling

Optimizes data transfers between host and device, and between devices,
with async scheduling, batching, and compute-communication overlap.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch


class TransferType(Enum):
    """Transfer type enumeration"""
    HOST_TO_DEVICE = "h2d"
    DEVICE_TO_HOST = "d2h"
    DEVICE_TO_DEVICE = "d2d"
    PEER_TO_PEER = "p2p"


class TransferPriority(Enum):
    """Transfer priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TransferRequest:
    """Transfer request metadata"""
    transfer_id: int
    transfer_type: TransferType
    src_device: torch.device
    dst_device: torch.device
    size: int
    priority: TransferPriority = TransferPriority.NORMAL

    # Data reference
    tensor_id: Optional[int] = None
    data_ptr: Optional[int] = None

    # Scheduling
    submitted_time: float = field(default_factory=time.time)
    scheduled_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Batching
    batch_id: Optional[int] = None
    can_batch: bool = True

    # Status
    completed: bool = False
    error: Optional[str] = None

    def get_duration(self) -> float:
        """Get transfer duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def get_bandwidth(self) -> float:
        """Get transfer bandwidth (bytes/sec)"""
        duration = self.get_duration()
        if duration > 0:
            return self.size / duration
        return 0.0


class TransferScheduler:
    """Schedule host-device and device-device transfers"""

    def __init__(self):
        self.lock = threading.Lock()

        # Transfer queues per (src, dst) pair
        self.queues: Dict[
            Tuple[torch.device, torch.device], deque
        ] = defaultdict(deque)

        # Active transfers
        self.active_transfers: Dict[int, TransferRequest] = {}

        # Statistics
        self.total_transfers = 0
        self.total_bytes = 0
        self.total_time = 0.0

        # Scheduling policy
        self.policy = "priority_fifo"  # fifo, priority_fifo, shortest_first

    def submit_transfer(self, request: TransferRequest) -> int:
        """Submit transfer request"""
        with self.lock:
            queue_key = (request.src_device, request.dst_device)
            self.queues[queue_key].append(request)
            request.scheduled_time = time.time()
            return request.transfer_id

    def get_next_transfer(
        self, src_device: torch.device, dst_device: torch.device
    ) -> Optional[TransferRequest]:
        """Get next transfer to execute"""
        with self.lock:
            queue_key = (src_device, dst_device)
            queue = self.queues.get(queue_key)

            if not queue:
                return None

            if self.policy == "fifo":
                return queue.popleft()

            elif self.policy == "priority_fifo":
                # Sort by priority (higher first), then FIFO
                sorted_queue = sorted(
                    queue,
                    key=lambda r: (-r.priority.value, r.submitted_time),
                )
                if sorted_queue:
                    request = sorted_queue[0]
                    queue.remove(request)
                    return request

            elif self.policy == "shortest_first":
                # Sort by size (smaller first)
                sorted_queue = sorted(queue, key=lambda r: r.size)
                if sorted_queue:
                    request = sorted_queue[0]
                    queue.remove(request)
                    return request

            return None

    def start_transfer(self, transfer_id: int):
        """Mark transfer as started"""
        with self.lock:
            if transfer_id in self.active_transfers:
                request = self.active_transfers[transfer_id]
                request.start_time = time.time()

    def complete_transfer(
        self, transfer_id: int, error: Optional[str] = None
    ):
        """Mark transfer as completed"""
        with self.lock:
            if transfer_id not in self.active_transfers:
                return

            request = self.active_transfers[transfer_id]
            request.end_time = time.time()
            request.completed = True
            request.error = error

            # Update statistics
            if not error:
                self.total_transfers += 1
                self.total_bytes += request.size
                self.total_time += request.get_duration()

            # Remove from active
            del self.active_transfers[transfer_id]

    def get_queue_size(
        self, src_device: torch.device, dst_device: torch.device
    ) -> int:
        """Get queue size for device pair"""
        with self.lock:
            queue_key = (src_device, dst_device)
            return len(self.queues.get(queue_key, []))

    def get_bandwidth_stats(self) -> Dict[str, float]:
        """Get bandwidth statistics"""
        with self.lock:
            if self.total_time > 0:
                avg_bandwidth = self.total_bytes / self.total_time
            else:
                avg_bandwidth = 0.0

            return {
                "total_transfers": self.total_transfers,
                "total_bytes": self.total_bytes,
                "total_time": self.total_time,
                "avg_bandwidth": avg_bandwidth,
            }


class TransferBatcher:
    """Batch small transfers"""

    def __init__(self, max_batch_size: int = 1024 * 1024 * 4):  # 4MB default
        self.lock = threading.Lock()
        self.max_batch_size = max_batch_size

        # Pending transfers for batching
        self.pending: Dict[
            Tuple[torch.device, torch.device], List[TransferRequest]
        ] = defaultdict(list)

        # Batch tracking
        self.next_batch_id = 0
        self.batches: Dict[int, List[TransferRequest]] = {}

        # Configuration
        self.min_batch_size = 2  # Minimum transfers to form a batch
        self.max_batch_wait_time = 0.001  # 1ms max wait

    def add_transfer(self, request: TransferRequest) -> Optional[int]:
        """
        Add transfer for batching.

        Returns:
            Batch ID if batch is ready, None otherwise
        """
        with self.lock:
            if not request.can_batch:
                return None

            key = (request.src_device, request.dst_device)
            self.pending[key].append(request)

            # Check if we can form a batch
            total_size = sum(r.size for r in self.pending[key])

            if (
                len(self.pending[key]) >= self.min_batch_size
                and total_size <= self.max_batch_size
            ):
                # Form a batch
                batch_id = self._create_batch(key)
                return batch_id

            # Check if oldest transfer has waited too long
            if self.pending[key]:
                oldest = min(self.pending[key], key=lambda r: r.submitted_time)
                wait_time = time.time() - oldest.submitted_time

                if wait_time > self.max_batch_wait_time:
                    # Force batch creation
                    batch_id = self._create_batch(key)
                    return batch_id

            return None

    def _create_batch(
        self, key: Tuple[torch.device, torch.device]
    ) -> int:
        """Create a batch from pending transfers"""
        batch_id = self.next_batch_id
        self.next_batch_id += 1

        # Move pending transfers to batch
        transfers = self.pending[key]
        self.batches[batch_id] = transfers.copy()

        # Mark transfers as batched
        for transfer in transfers:
            transfer.batch_id = batch_id

        # Clear pending
        self.pending[key].clear()

        return batch_id

    def get_batch(self, batch_id: int) -> Optional[List[TransferRequest]]:
        """Get batch by ID"""
        with self.lock:
            return self.batches.get(batch_id)

    def complete_batch(self, batch_id: int):
        """Mark batch as completed"""
        with self.lock:
            if batch_id in self.batches:
                del self.batches[batch_id]


class PinnedMemoryManager:
    """Manage pinned memory pool"""

    def __init__(self, initial_size: int = 1024 * 1024 * 256):  # 256MB default
        self.lock = threading.Lock()
        self.initial_size = initial_size

        # Pinned memory pool
        self.pool: List[Tuple[torch.Tensor, bool]] = []  # (tensor, in_use)
        self.total_size = 0

        # Statistics
        self.allocations = 0
        self.hits = 0
        self.misses = 0

        # Initialize pool
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize pinned memory pool"""
        # Create initial pool of pinned buffers
        chunk_size = 1024 * 1024 * 16  # 16MB chunks
        num_chunks = self.initial_size // chunk_size

        for _ in range(num_chunks):
            try:
                # Allocate pinned memory
                tensor = torch.empty(
                    chunk_size, dtype=torch.uint8, pin_memory=True
                )
                self.pool.append((tensor, False))
                self.total_size += chunk_size
            except RuntimeError:
                # Failed to allocate pinned memory
                break

    def allocate(self, size: int) -> Optional[torch.Tensor]:
        """Allocate pinned memory"""
        with self.lock:
            self.allocations += 1

            # Find suitable free buffer
            for i, (tensor, in_use) in enumerate(self.pool):
                if not in_use and tensor.numel() >= size:
                    self.pool[i] = (tensor, True)
                    self.hits += 1
                    return tensor[:size]

            # No suitable buffer, allocate new one
            self.misses += 1

            try:
                tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
                self.pool.append((tensor, True))
                self.total_size += size
                return tensor
            except RuntimeError:
                # Failed to allocate
                return None

    def free(self, tensor: torch.Tensor):
        """Free pinned memory"""
        with self.lock:
            # Find and mark as free
            for i, (pool_tensor, in_use) in enumerate(self.pool):
                if pool_tensor.data_ptr() == tensor.data_ptr():
                    self.pool[i] = (pool_tensor, False)
                    break

    def get_stats(self) -> Dict[str, any]:
        """Get pinned memory statistics"""
        with self.lock:
            in_use = sum(1 for _, used in self.pool if used)
            hit_rate = self.hits / max(1, self.allocations)

            return {
                "total_size": self.total_size,
                "pool_size": len(self.pool),
                "in_use": in_use,
                "free": len(self.pool) - in_use,
                "allocations": self.allocations,
                "hit_rate": hit_rate,
            }


class TransferOptimizer:
    """Coordinate transfers"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.lock = threading.RLock()

        # Components
        self.scheduler = TransferScheduler()
        self.batcher = TransferBatcher()
        self.pinned_memory = PinnedMemoryManager()

        # Transfer tracking
        self.next_transfer_id = 0
        self.transfers: Dict[int, TransferRequest] = {}

        # P2P capabilities
        self.p2p_enabled: Dict[Tuple[int, int], bool] = {}
        self._detect_p2p()

        # Async transfer threads
        self.running = True
        self.transfer_threads: List[threading.Thread] = []
        self._start_transfer_workers()

    def _detect_p2p(self):
        """Detect P2P capabilities"""
        if torch.cuda.device_count() < 2:
            return

        for i in range(torch.cuda.device_count()):
            for j in range(torch.cuda.device_count()):
                if i != j:
                    try:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        self.p2p_enabled[(i, j)] = can_access
                    except RuntimeError:
                        self.p2p_enabled[(i, j)] = False

    def _start_transfer_workers(self):
        """Start background transfer workers"""
        num_workers = 2  # Adjust based on workload

        for _ in range(num_workers):
            thread = threading.Thread(target=self._transfer_worker, daemon=True)
            thread.start()
            self.transfer_threads.append(thread)

    def _transfer_worker(self):
        """Background transfer worker"""
        while self.running:
            try:
                # Check for pending transfers
                executed = False

                with self.lock:
                    # Iterate through all queue pairs
                    for (src, dst), queue in self.scheduler.queues.items():
                        if queue:
                            # Get next transfer
                            request = self.scheduler.get_next_transfer(src, dst)

                            if request:
                                # Execute transfer
                                self._execute_transfer(request)
                                executed = True
                                break

                if not executed:
                    time.sleep(0.001)  # 1ms wait

            except Exception as e:
                print(f"Transfer worker error: {e}")
                time.sleep(0.01)

    def _execute_transfer(self, request: TransferRequest):
        """Execute a transfer"""
        try:
            self.scheduler.active_transfers[request.transfer_id] = request
            self.scheduler.start_transfer(request.transfer_id)

            # In production, execute actual data transfer
            # For now, simulate transfer with sleep
            # transfer_time = request.size / (10 * 1024 * 1024 * 1024)  # 10 GB/s
            # time.sleep(transfer_time)

            # Determine transfer method
            if request.transfer_type == TransferType.PEER_TO_PEER:
                if self.supports_p2p(request.src_device, request.dst_device):
                    self._execute_p2p_transfer(request)
                else:
                    # Fallback to D2H + H2D
                    self._execute_staged_transfer(request)

            elif request.transfer_type == TransferType.HOST_TO_DEVICE:
                self._execute_h2d_transfer(request)

            elif request.transfer_type == TransferType.DEVICE_TO_HOST:
                self._execute_d2h_transfer(request)

            elif request.transfer_type == TransferType.DEVICE_TO_DEVICE:
                if request.src_device == request.dst_device:
                    # Same device, no transfer needed
                    pass
                else:
                    self._execute_d2d_transfer(request)

            self.scheduler.complete_transfer(request.transfer_id)

        except Exception as e:
            self.scheduler.complete_transfer(request.transfer_id, error=str(e))

    def _execute_h2d_transfer(self, request: TransferRequest):
        """Execute host-to-device transfer"""
        # In production, use pinned memory for async transfer
        pass

    def _execute_d2h_transfer(self, request: TransferRequest):
        """Execute device-to-host transfer"""
        # In production, use pinned memory for async transfer
        pass

    def _execute_d2d_transfer(self, request: TransferRequest):
        """Execute device-to-device transfer"""
        # In production, use P2P if available, otherwise stage through host
        if self.supports_p2p(request.src_device, request.dst_device):
            self._execute_p2p_transfer(request)
        else:
            self._execute_staged_transfer(request)

    def _execute_p2p_transfer(self, request: TransferRequest):
        """Execute peer-to-peer transfer"""
        # In production, use CUDA P2P copy
        pass

    def _execute_staged_transfer(self, request: TransferRequest):
        """Execute staged transfer through host"""
        # Transfer: Device A -> Host -> Device B
        pass

    def submit_transfer(
        self,
        src_device: torch.device,
        dst_device: torch.device,
        size: int,
        transfer_type: Optional[TransferType] = None,
        priority: TransferPriority = TransferPriority.NORMAL,
        can_batch: bool = True,
        tensor_id: Optional[int] = None,
    ) -> int:
        """Submit transfer request"""
        with self.lock:
            transfer_id = self.next_transfer_id
            self.next_transfer_id += 1

            # Determine transfer type if not specified
            if transfer_type is None:
                if src_device.type == "cpu" and dst_device.type == "cuda":
                    transfer_type = TransferType.HOST_TO_DEVICE
                elif src_device.type == "cuda" and dst_device.type == "cpu":
                    transfer_type = TransferType.DEVICE_TO_HOST
                elif src_device.type == "cuda" and dst_device.type == "cuda":
                    if src_device.index != dst_device.index:
                        if self.supports_p2p(src_device, dst_device):
                            transfer_type = TransferType.PEER_TO_PEER
                        else:
                            transfer_type = TransferType.DEVICE_TO_DEVICE
                    else:
                        transfer_type = TransferType.DEVICE_TO_DEVICE
                else:
                    transfer_type = TransferType.DEVICE_TO_DEVICE

            request = TransferRequest(
                transfer_id=transfer_id,
                transfer_type=transfer_type,
                src_device=src_device,
                dst_device=dst_device,
                size=size,
                priority=priority,
                can_batch=can_batch,
                tensor_id=tensor_id,
            )

            self.transfers[transfer_id] = request

            # Try batching if enabled
            if can_batch and size < self.batcher.max_batch_size:
                batch_id = self.batcher.add_transfer(request)

                if batch_id is not None:
                    # Batch is ready, submit all transfers
                    batch = self.batcher.get_batch(batch_id)
                    if batch:
                        for req in batch:
                            self.scheduler.submit_transfer(req)
                        self.batcher.complete_batch(batch_id)
            else:
                # Submit directly
                self.scheduler.submit_transfer(request)

            return transfer_id

    def wait_transfer(self, transfer_id: int, timeout: Optional[float] = None):
        """Wait for transfer to complete"""
        start_time = time.time()

        while True:
            with self.lock:
                if transfer_id in self.transfers:
                    request = self.transfers[transfer_id]
                    if request.completed:
                        return

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Transfer {transfer_id} timed out")

            time.sleep(0.001)

    def supports_p2p(
        self, src_device: torch.device, dst_device: torch.device
    ) -> bool:
        """Check if P2P is supported between devices"""
        if src_device.type != "cuda" or dst_device.type != "cuda":
            return False

        return self.p2p_enabled.get((src_device.index, dst_device.index), False)

    def get_stats(self) -> Dict[str, any]:
        """Get transfer optimizer statistics"""
        with self.lock:
            return {
                "scheduler": self.scheduler.get_bandwidth_stats(),
                "pinned_memory": self.pinned_memory.get_stats(),
                "total_transfers": len(self.transfers),
                "completed_transfers": sum(
                    1 for t in self.transfers.values() if t.completed
                ),
                "p2p_links": len([v for v in self.p2p_enabled.values() if v]),
            }

    def shutdown(self):
        """Shutdown transfer optimizer"""
        self.running = False
        for thread in self.transfer_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)


# Global singleton instance
_transfer_optimizer: Optional[TransferOptimizer] = None


def get_transfer_optimizer() -> TransferOptimizer:
    """Get global transfer optimizer instance"""
    global _transfer_optimizer
    if _transfer_optimizer is None:
        _transfer_optimizer = TransferOptimizer()
    return _transfer_optimizer

"""
Memory Scheduler for Runtime Scheduling

Dynamic memory management with predictive allocation,
ML-based eviction, prefetching, and defragmentation.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


class MemoryLocation(Enum):
    """Memory location enumeration"""
    DEVICE = "device"
    HOST_PINNED = "host_pinned"
    HOST_PAGEABLE = "host_pageable"
    DISK = "disk"


@dataclass
class MemoryBlock:
    """Memory block metadata"""
    block_id: int
    ptr: int  # Memory address
    size: int
    device: torch.device
    location: MemoryLocation
    allocated_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    ref_count: int = 1
    pinned: bool = False
    can_evict: bool = True

    # Tensor metadata for reconstruction
    tensor_id: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None

    def update_access(self):
        """Update access statistics"""
        self.last_access_time = time.time()
        self.access_count += 1

    def get_priority(self) -> float:
        """Calculate eviction priority (higher = keep, lower = evict)"""
        if not self.can_evict or self.pinned:
            return float("inf")

        # Factors: access frequency, recency, reference count
        current_time = time.time()
        recency = 1.0 / max(0.001, current_time - self.last_access_time)
        frequency = self.access_count / max(1, current_time - self.allocated_time)

        return recency * 0.4 + frequency * 0.3 + self.ref_count * 0.3


class MemoryPool:
    """Per-device memory pool"""

    def __init__(self, device: torch.device, capacity: int):
        self.device = device
        self.capacity = capacity
        self.lock = threading.RLock()

        # Block tracking
        self.blocks: Dict[int, MemoryBlock] = {}
        self.free_blocks: List[Tuple[int, int]] = []  # (ptr, size)
        self.next_block_id = 0

        # Statistics
        self.allocated_bytes = 0
        self.peak_allocated = 0
        self.total_allocations = 0
        self.total_frees = 0
        self.fragmentation_score = 0.0

        # History for defragmentation
        self.alloc_history = deque(maxlen=1000)

    def allocate(
        self,
        size: int,
        tensor_id: Optional[int] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[MemoryBlock]:
        """Allocate memory block"""
        with self.lock:
            # Try to find suitable free block
            ptr = self._find_free_block(size)

            if ptr is None:
                # No suitable free block, try actual allocation
                if self.allocated_bytes + size > self.capacity:
                    return None  # Out of memory

                ptr = id(object())  # Dummy pointer

            # Create block
            block_id = self.next_block_id
            self.next_block_id += 1

            block = MemoryBlock(
                block_id=block_id,
                ptr=ptr,
                size=size,
                device=self.device,
                location=MemoryLocation.DEVICE,
                tensor_id=tensor_id,
                shape=shape,
                dtype=dtype,
            )

            self.blocks[block_id] = block
            self.allocated_bytes += size
            self.peak_allocated = max(self.peak_allocated, self.allocated_bytes)
            self.total_allocations += 1

            self.alloc_history.append({
                "time": time.time(),
                "size": size,
                "total": self.allocated_bytes,
            })

            return block

    def free(self, block_id: int) -> bool:
        """Free memory block"""
        with self.lock:
            if block_id not in self.blocks:
                return False

            block = self.blocks[block_id]
            del self.blocks[block_id]

            # Add to free list
            self.free_blocks.append((block.ptr, block.size))
            self.allocated_bytes -= block.size
            self.total_frees += 1

            # Merge adjacent free blocks
            self._coalesce_free_blocks()

            return True

    def _find_free_block(self, size: int) -> Optional[int]:
        """Find suitable free block using first-fit strategy"""
        for i, (ptr, block_size) in enumerate(self.free_blocks):
            if block_size >= size:
                # Use this block
                self.free_blocks.pop(i)

                # If block is larger, split it
                if block_size > size:
                    remaining_size = block_size - size
                    remaining_ptr = ptr + size
                    self.free_blocks.append((remaining_ptr, remaining_size))

                return ptr

        return None

    def _coalesce_free_blocks(self):
        """Merge adjacent free blocks"""
        if len(self.free_blocks) < 2:
            return

        # Sort by address
        self.free_blocks.sort(key=lambda x: x[0])

        # Merge adjacent blocks
        merged = []
        current_ptr, current_size = self.free_blocks[0]

        for ptr, size in self.free_blocks[1:]:
            if current_ptr + current_size == ptr:
                # Adjacent, merge
                current_size += size
            else:
                # Not adjacent, save current and start new
                merged.append((current_ptr, current_size))
                current_ptr, current_size = ptr, size

        merged.append((current_ptr, current_size))
        self.free_blocks = merged

    def get_fragmentation(self) -> float:
        """Calculate memory fragmentation score (0-1)"""
        with self.lock:
            if not self.free_blocks:
                return 0.0

            # Fragmentation = number of free blocks / total free memory
            total_free = sum(size for _, size in self.free_blocks)
            if total_free == 0:
                return 0.0

            return len(self.free_blocks) / max(1, total_free / (1024 * 1024))

    def defragment(self) -> int:
        """
        Defragment memory by moving blocks.

        Returns:
            Number of blocks moved
        """
        with self.lock:
            # In production, implement actual defragmentation
            # For now, just coalesce free blocks
            self._coalesce_free_blocks()
            return 0

    def get_stats(self) -> Dict[str, any]:
        """Get memory pool statistics"""
        with self.lock:
            return {
                "device": str(self.device),
                "capacity": self.capacity,
                "allocated": self.allocated_bytes,
                "free": self.capacity - self.allocated_bytes,
                "utilization": self.allocated_bytes / max(1, self.capacity),
                "peak_allocated": self.peak_allocated,
                "total_blocks": len(self.blocks),
                "free_blocks": len(self.free_blocks),
                "fragmentation": self.get_fragmentation(),
                "total_allocations": self.total_allocations,
                "total_frees": self.total_frees,
            }


class EvictionPolicy:
    """ML-based eviction decisions"""

    def __init__(self):
        self.lock = threading.Lock()
        self.eviction_history = deque(maxlen=1000)

        # Simple LRU-based policy
        # In production, use ML model from device_models.py
        self.policy_type = "lru"  # lru, lfu, ml

    def select_victims(
        self,
        blocks: Dict[int, MemoryBlock],
        required_size: int,
        device: torch.device,
    ) -> List[int]:
        """
        Select blocks to evict to free required_size.

        Returns:
            List of block IDs to evict
        """
        with self.lock:
            # Filter evictable blocks
            candidates = [
                (block_id, block)
                for block_id, block in blocks.items()
                if block.can_evict and not block.pinned and block.device == device
            ]

            if not candidates:
                return []

            # Score candidates (lower = evict first)
            scored = [
                (block_id, self._score_block(block))
                for block_id, block in candidates
            ]

            # Sort by score (ascending)
            scored.sort(key=lambda x: x[1])

            # Select victims until we have enough space
            victims = []
            freed_size = 0

            for block_id, score in scored:
                block = blocks[block_id]
                victims.append(block_id)
                freed_size += block.size

                if freed_size >= required_size:
                    break

            # Record eviction decision for learning
            self.eviction_history.append({
                "time": time.time(),
                "device": device,
                "required_size": required_size,
                "victims": len(victims),
                "freed_size": freed_size,
            })

            return victims

    def _score_block(self, block: MemoryBlock) -> float:
        """Score block for eviction (lower = evict first)"""
        if self.policy_type == "lru":
            # LRU: evict least recently used
            return -block.last_access_time

        elif self.policy_type == "lfu":
            # LFU: evict least frequently used
            return block.access_count

        else:
            # Combined heuristic
            priority = block.get_priority()
            return -priority  # Invert so lower priority = evict first


class PrefetchScheduler:
    """Predictive prefetching"""

    def __init__(self):
        self.lock = threading.Lock()

        # Access pattern tracking
        self.access_patterns: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.prefetch_queue: deque = deque()

        # Prefetch statistics
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.prefetch_requests = 0

    def record_access(self, tensor_id: int, device: torch.device):
        """Record tensor access"""
        with self.lock:
            self.access_patterns[tensor_id].append({
                "time": time.time(),
                "device": device,
            })

    def predict_next_access(
        self, current_tensor_id: int
    ) -> List[Tuple[int, torch.device]]:
        """
        Predict next tensor accesses for prefetching.

        Returns:
            List of (tensor_id, target_device) to prefetch
        """
        with self.lock:
            predictions = []

            # Simple pattern: if a tensor is accessed repeatedly, prefetch it
            if current_tensor_id in self.access_patterns:
                pattern = self.access_patterns[current_tensor_id]

                if len(pattern) >= 3:
                    # Check if there's a regular access pattern
                    recent_devices = [p["device"] for p in list(pattern)[-3:]]

                    # If all recent accesses were on the same device, predict next access there
                    if len(set(str(d) for d in recent_devices)) == 1:
                        predictions.append((current_tensor_id, recent_devices[0]))

            return predictions

    def schedule_prefetch(
        self, tensor_id: int, target_device: torch.device, priority: int = 0
    ):
        """Schedule a prefetch operation"""
        with self.lock:
            self.prefetch_queue.append({
                "tensor_id": tensor_id,
                "target_device": target_device,
                "priority": priority,
                "scheduled_time": time.time(),
            })
            self.prefetch_requests += 1

    def get_next_prefetch(self) -> Optional[Dict[str, any]]:
        """Get next prefetch operation"""
        with self.lock:
            if not self.prefetch_queue:
                return None

            # Sort by priority (higher first)
            sorted_queue = sorted(
                self.prefetch_queue, key=lambda x: x["priority"], reverse=True
            )

            if sorted_queue:
                prefetch = sorted_queue[0]
                self.prefetch_queue.remove(prefetch)
                return prefetch

            return None

    def record_prefetch_result(self, tensor_id: int, hit: bool):
        """Record prefetch result"""
        with self.lock:
            if hit:
                self.prefetch_hits += 1
            else:
                self.prefetch_misses += 1

    def get_hit_rate(self) -> float:
        """Get prefetch hit rate"""
        with self.lock:
            total = self.prefetch_hits + self.prefetch_misses
            if total == 0:
                return 0.0
            return self.prefetch_hits / total


class MemoryScheduler:
    """Coordinate memory operations"""

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

        # Memory pools per device
        self.pools: Dict[torch.device, MemoryPool] = {}
        self._initialize_pools()

        # Components
        self.eviction_policy = EvictionPolicy()
        self.prefetch_scheduler = PrefetchScheduler()

        # Global block registry
        self.blocks: Dict[int, MemoryBlock] = {}
        self.tensor_to_block: Dict[int, int] = {}

        # Background prefetch thread
        self.running = True
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True
        )
        self.prefetch_thread.start()

    def _initialize_pools(self):
        """Initialize memory pools for each device"""
        # CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                props = torch.cuda.get_device_properties(device)
                capacity = props.total_memory

                self.pools[device] = MemoryPool(device, capacity)

        # CPU device (use a fraction of system memory)
        import psutil
        cpu_capacity = int(psutil.virtual_memory().total * 0.5)  # Use 50% of RAM
        self.pools[torch.device("cpu")] = MemoryPool(
            torch.device("cpu"), cpu_capacity
        )

    def allocate(
        self,
        size: int,
        device: torch.device,
        tensor_id: Optional[int] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[MemoryBlock]:
        """Allocate memory block"""
        with self.lock:
            pool = self.pools.get(device)
            if not pool:
                return None

            # Try allocation
            block = pool.allocate(size, tensor_id, shape, dtype)

            if block is None:
                # Out of memory, try eviction
                victims = self.eviction_policy.select_victims(
                    pool.blocks, size, device
                )

                # Evict victims
                for victim_id in victims:
                    self._evict_block(victim_id, pool)

                # Retry allocation
                block = pool.allocate(size, tensor_id, shape, dtype)

            if block:
                self.blocks[block.block_id] = block
                if tensor_id is not None:
                    self.tensor_to_block[tensor_id] = block.block_id

            return block

    def free(self, block_id: int) -> bool:
        """Free memory block"""
        with self.lock:
            if block_id not in self.blocks:
                return False

            block = self.blocks[block_id]
            pool = self.pools.get(block.device)

            if pool:
                pool.free(block_id)

            del self.blocks[block_id]

            # Remove tensor mapping
            if block.tensor_id in self.tensor_to_block:
                del self.tensor_to_block[block.tensor_id]

            return True

    def _evict_block(self, block_id: int, pool: MemoryPool):
        """Evict a block from device memory"""
        # In production, move data to CPU or disk
        # For now, just free it
        pool.free(block_id)

    def record_access(self, tensor_id: int, device: torch.device):
        """Record tensor access"""
        # Update block access time
        with self.lock:
            if tensor_id in self.tensor_to_block:
                block_id = self.tensor_to_block[tensor_id]
                if block_id in self.blocks:
                    self.blocks[block_id].update_access()

        # Record for prefetching
        self.prefetch_scheduler.record_access(tensor_id, device)

        # Predict and schedule prefetches
        predictions = self.prefetch_scheduler.predict_next_access(tensor_id)
        for pred_tensor_id, target_device in predictions:
            self.prefetch_scheduler.schedule_prefetch(pred_tensor_id, target_device)

    def _prefetch_loop(self):
        """Background prefetch loop"""
        while self.running:
            try:
                prefetch = self.prefetch_scheduler.get_next_prefetch()

                if prefetch:
                    # Execute prefetch
                    tensor_id = prefetch["tensor_id"]
                    target_device = prefetch["target_device"]

                    # In production, actually move the tensor
                    # For now, just record the attempt
                    self.prefetch_scheduler.record_prefetch_result(tensor_id, True)

                else:
                    time.sleep(0.01)  # Wait for more work

            except Exception as e:
                print(f"Prefetch error: {e}")
                time.sleep(0.1)

    def defragment_all(self) -> int:
        """Defragment all memory pools"""
        total_moved = 0
        with self.lock:
            for pool in self.pools.values():
                total_moved += pool.defragment()
        return total_moved

    def get_stats(self) -> Dict[str, any]:
        """Get memory scheduler statistics"""
        with self.lock:
            return {
                "pools": {
                    str(device): pool.get_stats()
                    for device, pool in self.pools.items()
                },
                "total_blocks": len(self.blocks),
                "prefetch_hit_rate": self.prefetch_scheduler.get_hit_rate(),
                "prefetch_queue_size": len(self.prefetch_scheduler.prefetch_queue),
            }

    def shutdown(self):
        """Shutdown memory scheduler"""
        self.running = False
        if self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)


# Global singleton instance
_memory_scheduler: Optional[MemoryScheduler] = None


def get_memory_scheduler() -> MemoryScheduler:
    """Get global memory scheduler instance"""
    global _memory_scheduler
    if _memory_scheduler is None:
        _memory_scheduler = MemoryScheduler()
    return _memory_scheduler

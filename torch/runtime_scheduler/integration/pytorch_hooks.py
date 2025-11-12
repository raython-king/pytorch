"""
Integration hooks for runtime scheduler with PyTorch.

This module provides hooks into PyTorch's runtime system:
- Operation dispatch hook (intercept before kernel launch)
- Memory allocation hook (intercept allocator calls)
- Device placement hook (override default placement)
- Stream creation hook (manage stream pool)

Design principles:
- Minimal changes to PyTorch core
- Use existing hook mechanisms
- Fallback to default behavior
- Shadow mode for validation
"""

import functools
import threading
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    from torch.utils.hooks import RemovableHandle
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available for hooks")


class HookMode(Enum):
    """Modes for runtime scheduler hooks."""
    DISABLED = "disabled"  # Hooks disabled, use default PyTorch behavior
    SHADOW = "shadow"  # Make decisions but don't apply (for validation)
    ENABLED = "enabled"  # Fully enabled, apply decisions


@dataclass
class OperationInfo:
    """Information about an operation to be executed."""
    op_name: str
    input_shapes: List[Tuple[int, ...]]
    input_dtypes: List[str]
    output_shape: Optional[Tuple[int, ...]] = None
    current_device: str = "cpu"
    suggested_device: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RuntimeSchedulerHooks:
    """
    Main hook manager for runtime scheduler integration.

    Provides centralized management of all hooks and their state.
    """

    def __init__(self, mode: HookMode = HookMode.DISABLED):
        """
        Initialize runtime scheduler hooks.

        Args:
            mode: Hook operating mode
        """
        self.mode = mode
        self._enabled = False

        # Hook handles
        self._handles: List[Any] = []

        # Callbacks
        self._operation_dispatch_callbacks: List[Callable] = []
        self._memory_allocation_callbacks: List[Callable] = []
        self._device_placement_callbacks: List[Callable] = []
        self._stream_creation_callbacks: List[Callable] = []

        # Statistics
        self._stats = {
            "operations_intercepted": 0,
            "operations_rescheduled": 0,
            "memory_allocations": 0,
            "device_placements": 0,
            "streams_created": 0,
        }

        # Thread safety
        self._lock = threading.RLock()

        # Shadow mode data (for validation)
        self._shadow_decisions: List[Dict[str, Any]] = []

    def enable(self) -> None:
        """Enable runtime scheduler hooks."""
        if self._enabled:
            return

        with self._lock:
            self._enabled = True
            self._install_hooks()

    def disable(self) -> None:
        """Disable runtime scheduler hooks."""
        if not self._enabled:
            return

        with self._lock:
            self._enabled = False
            self._remove_hooks()

    def set_mode(self, mode: HookMode) -> None:
        """Change hook operating mode."""
        with self._lock:
            old_mode = self.mode
            self.mode = mode

            # Reinstall hooks if enabled
            if self._enabled and old_mode != mode:
                self._remove_hooks()
                self._install_hooks()

    def _install_hooks(self) -> None:
        """Install all hooks."""
        if not HAS_TORCH:
            return

        # Install operation dispatch hooks
        self._install_operation_hooks()

        # Install memory hooks
        self._install_memory_hooks()

        # Install device placement hooks
        self._install_device_hooks()

        # Install stream hooks
        self._install_stream_hooks()

    def _remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for handle in self._handles:
            if hasattr(handle, 'remove'):
                handle.remove()

        self._handles.clear()

    def _install_operation_hooks(self) -> None:
        """Install operation dispatch hooks."""
        # Hook into PyTorch's dispatcher
        # Note: This is a simplified version. Real implementation would use
        # PyTorch's C++ dispatcher hooks or Python dispatcher extensions

        if not HAS_TORCH:
            return

        # Example: Hook tensor operations
        original_tensor_ops = {}

        def create_hooked_op(op_name: str, original_op: Callable) -> Callable:
            @functools.wraps(original_op)
            def hooked_op(*args, **kwargs):
                # Extract operation info
                op_info = self._extract_operation_info(op_name, args, kwargs)

                # Call dispatch callbacks
                for callback in self._operation_dispatch_callbacks:
                    try:
                        result = callback(op_info)
                        if result is not None:
                            op_info = result
                    except Exception as e:
                        warnings.warn(f"Error in operation dispatch callback: {e}")

                self._stats["operations_intercepted"] += 1

                # Apply scheduling decision
                if self.mode == HookMode.ENABLED and op_info.suggested_device:
                    # Move tensors to suggested device
                    if op_info.suggested_device != op_info.current_device:
                        args, kwargs = self._move_to_device(
                            args, kwargs, op_info.suggested_device
                        )
                        self._stats["operations_rescheduled"] += 1

                elif self.mode == HookMode.SHADOW:
                    # Record decision but don't apply
                    self._shadow_decisions.append({
                        "type": "operation",
                        "op_name": op_name,
                        "original_device": op_info.current_device,
                        "suggested_device": op_info.suggested_device,
                    })

                # Execute original operation
                return original_op(*args, **kwargs)

            return hooked_op

        # Note: In production, this would hook at the C++ dispatcher level
        # This is a simplified Python-level demonstration

    def _install_memory_hooks(self) -> None:
        """Install memory allocation hooks."""
        if not HAS_TORCH or not torch.cuda.is_available():
            return

        # Hook into CUDA memory allocator
        # Note: PyTorch provides memory profiling hooks via torch.cuda.memory

        def memory_allocation_hook(device: int, size: int, stream: int):
            for callback in self._memory_allocation_callbacks:
                try:
                    callback(f"cuda:{device}", size, stream)
                except Exception as e:
                    warnings.warn(f"Error in memory allocation callback: {e}")

            self._stats["memory_allocations"] += 1

        # Register hook (if PyTorch supports it)
        # handle = torch.cuda.memory._register_allocation_hook(memory_allocation_hook)
        # self._handles.append(handle)

    def _install_device_hooks(self) -> None:
        """Install device placement hooks."""
        # Hook into tensor creation to intercept device placement
        pass

    def _install_stream_hooks(self) -> None:
        """Install stream creation hooks."""
        # Hook into CUDA stream creation
        pass

    def _extract_operation_info(
        self,
        op_name: str,
        args: Tuple,
        kwargs: Dict[str, Any]
    ) -> OperationInfo:
        """Extract information about an operation."""
        input_shapes = []
        input_dtypes = []
        current_device = "cpu"

        # Extract tensor info from arguments
        for arg in args:
            if HAS_TORCH and isinstance(arg, torch.Tensor):
                input_shapes.append(tuple(arg.shape))
                input_dtypes.append(str(arg.dtype))
                current_device = str(arg.device)

        return OperationInfo(
            op_name=op_name,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            current_device=current_device
        )

    def _move_to_device(
        self,
        args: Tuple,
        kwargs: Dict[str, Any],
        device: str
    ) -> Tuple[Tuple, Dict[str, Any]]:
        """Move tensor arguments to specified device."""
        if not HAS_TORCH:
            return args, kwargs

        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(arg.to(device))
            else:
                new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                new_kwargs[key] = value.to(device)
            else:
                new_kwargs[key] = value

        return tuple(new_args), new_kwargs

    # Callback registration methods

    def register_operation_dispatch_callback(
        self,
        callback: Callable[[OperationInfo], Optional[OperationInfo]]
    ) -> None:
        """
        Register a callback for operation dispatch.

        The callback receives OperationInfo and can return modified OperationInfo
        with suggested device placement.

        Args:
            callback: Callback function
        """
        with self._lock:
            self._operation_dispatch_callbacks.append(callback)

    def register_memory_allocation_callback(
        self,
        callback: Callable[[str, int, int], None]
    ) -> None:
        """
        Register a callback for memory allocation.

        Args:
            callback: Callback(device, size, stream)
        """
        with self._lock:
            self._memory_allocation_callbacks.append(callback)

    def register_device_placement_callback(
        self,
        callback: Callable[[torch.Tensor, str], str]
    ) -> None:
        """
        Register a callback for device placement.

        Args:
            callback: Callback(tensor, default_device) -> suggested_device
        """
        with self._lock:
            self._device_placement_callbacks.append(callback)

    def register_stream_creation_callback(
        self,
        callback: Callable[[str], None]
    ) -> None:
        """
        Register a callback for stream creation.

        Args:
            callback: Callback(device)
        """
        with self._lock:
            self._stream_creation_callbacks.append(callback)

    def get_stats(self) -> Dict[str, int]:
        """Get hook statistics."""
        with self._lock:
            return dict(self._stats)

    def get_shadow_decisions(self) -> List[Dict[str, Any]]:
        """Get decisions made in shadow mode."""
        with self._lock:
            return list(self._shadow_decisions)

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0
            self._shadow_decisions.clear()


# Global hook manager instance
_global_hooks: Optional[RuntimeSchedulerHooks] = None
_global_hooks_lock = threading.Lock()


def get_global_hooks() -> RuntimeSchedulerHooks:
    """Get the global hooks instance."""
    global _global_hooks

    if _global_hooks is None:
        with _global_hooks_lock:
            if _global_hooks is None:
                _global_hooks = RuntimeSchedulerHooks()

    return _global_hooks


def enable_runtime_scheduler_hooks(mode: HookMode = HookMode.ENABLED) -> None:
    """
    Enable runtime scheduler hooks globally.

    Args:
        mode: Hook operating mode
    """
    hooks = get_global_hooks()
    hooks.set_mode(mode)
    hooks.enable()


def disable_runtime_scheduler_hooks() -> None:
    """Disable runtime scheduler hooks globally."""
    hooks = get_global_hooks()
    hooks.disable()


# Module-level hook for easier integration
class ModuleHook:
    """
    Hook for nn.Module forward calls.

    This provides an easier way to intercept model forward passes.
    """

    def __init__(self, scheduler_callback: Optional[Callable] = None):
        """
        Initialize module hook.

        Args:
            scheduler_callback: Optional callback for scheduling decisions
        """
        self.scheduler_callback = scheduler_callback
        self._hooks: List[RemovableHandle] = []

    def register(self, module: nn.Module) -> None:
        """
        Register hook on a module.

        Args:
            module: Module to hook
        """
        if not HAS_TORCH:
            return

        def pre_forward_hook(mod, inputs):
            # Called before forward pass
            if self.scheduler_callback:
                try:
                    return self.scheduler_callback(mod, inputs)
                except Exception as e:
                    warnings.warn(f"Error in scheduler callback: {e}")
            return inputs

        handle = module.register_forward_pre_hook(pre_forward_hook)
        self._hooks.append(handle)

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


# Context manager for temporary hook activation
class HookContext:
    """
    Context manager for temporary hook activation.

    Usage:
        with HookContext(mode=HookMode.ENABLED):
            # Hooks are active here
            model(input)
        # Hooks are disabled here
    """

    def __init__(
        self,
        mode: HookMode = HookMode.ENABLED,
        auto_disable: bool = True
    ):
        """
        Initialize hook context.

        Args:
            mode: Hook operating mode
            auto_disable: Disable hooks on exit
        """
        self.mode = mode
        self.auto_disable = auto_disable
        self.hooks = get_global_hooks()
        self.previous_mode = self.hooks.mode
        self.was_enabled = self.hooks._enabled

    def __enter__(self) -> RuntimeSchedulerHooks:
        self.hooks.set_mode(self.mode)
        self.hooks.enable()
        return self.hooks

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.auto_disable:
            if not self.was_enabled:
                self.hooks.disable()
            else:
                self.hooks.set_mode(self.previous_mode)


# Stream pool management
class StreamPool:
    """
    Managed pool of CUDA streams for optimized scheduling.

    Features:
    - Reuse streams to avoid creation overhead
    - Per-device stream pools
    - Automatic cleanup
    """

    def __init__(self, max_streams_per_device: int = 8):
        """
        Initialize stream pool.

        Args:
            max_streams_per_device: Maximum streams per device
        """
        self.max_streams_per_device = max_streams_per_device

        # Device -> list of available streams
        self._pools: Dict[str, List[Any]] = {}

        # Device -> list of in-use streams
        self._in_use: Dict[str, Set[Any]] = {}

        self._lock = threading.Lock()

    def get_stream(self, device: str) -> Optional[Any]:
        """
        Get a stream from the pool.

        Args:
            device: Device name (e.g., "cuda:0")

        Returns:
            CUDA stream or None if not available
        """
        if not HAS_TORCH or not device.startswith("cuda"):
            return None

        with self._lock:
            # Initialize pools for device if needed
            if device not in self._pools:
                self._pools[device] = []
                self._in_use[device] = set()

            # Try to get from pool
            if self._pools[device]:
                stream = self._pools[device].pop()
                self._in_use[device].add(stream)
                return stream

            # Create new stream if under limit
            if len(self._in_use[device]) < self.max_streams_per_device:
                device_id = int(device.split(":")[1])
                stream = torch.cuda.Stream(device=device_id)
                self._in_use[device].add(stream)
                return stream

            return None

    def return_stream(self, device: str, stream: Any) -> None:
        """
        Return a stream to the pool.

        Args:
            device: Device name
            stream: Stream to return
        """
        with self._lock:
            if device in self._in_use and stream in self._in_use[device]:
                self._in_use[device].remove(stream)
                self._pools[device].append(stream)

    def synchronize_all(self) -> None:
        """Synchronize all streams in the pool."""
        if not HAS_TORCH:
            return

        with self._lock:
            for device_streams in self._in_use.values():
                for stream in device_streams:
                    stream.synchronize()

    def cleanup(self) -> None:
        """Clean up all streams."""
        self.synchronize_all()

        with self._lock:
            self._pools.clear()
            self._in_use.clear()


# Global stream pool
_global_stream_pool: Optional[StreamPool] = None


def get_stream_pool() -> StreamPool:
    """Get the global stream pool."""
    global _global_stream_pool

    if _global_stream_pool is None:
        _global_stream_pool = StreamPool()

    return _global_stream_pool

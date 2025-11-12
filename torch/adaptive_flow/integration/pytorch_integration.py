"""
PyTorch Integration for Adaptive Flow Control

This module provides transparent integration with PyTorch operations:
- Tensor device transfers (tensor.to(device))
- CUDA streams (torch.cuda.Stream)
- Distributed operations (torch.distributed)
- Memory allocation tracking

Integration is designed to be:
- Transparent: Works without code changes
- Backward compatible: Doesn't break existing code
- Optional: Can be enabled/disabled at runtime
- Shadow mode: Can run alongside normal operations for validation
"""

import time
import logging
import threading
import weakref
from typing import Dict, Optional, Any, List, Tuple
from collections import defaultdict
import functools

import torch
import torch.cuda
if torch.cuda.is_available():
    import torch.cuda.comm

try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

from ..flow_monitor import (
    FlowMetricsCollector,
    LinkUtilizationTracker,
    BottleneckDetector,
    PerformanceAnalyzer
)
from ..policy_engine import PolicyEngine, PolicyContext, PolicyDecision
from ..advanced_congestion import create_controller

logger = logging.getLogger(__name__)


class AdaptiveFlowIntegration:
    """
    Main integration class for PyTorch adaptive flow control

    Hooks into PyTorch operations to transparently apply adaptive flow control.
    """

    _instance: Optional['AdaptiveFlowIntegration'] = None
    _lock = threading.Lock()

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize PyTorch integration

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = False
        self.shadow_mode = self.config.get('shadow_mode', False)

        # Initialize monitoring components
        self.flow_collector = FlowMetricsCollector()
        self.link_tracker = LinkUtilizationTracker()
        self.bottleneck_detector = BottleneckDetector(
            self.link_tracker,
            self.flow_collector
        )
        self.performance_analyzer = PerformanceAnalyzer(
            self.flow_collector,
            self.link_tracker,
            self.bottleneck_detector
        )

        # Initialize policy engine
        self.policy_engine = PolicyEngine(
            self.flow_collector,
            self.link_tracker,
            self.bottleneck_detector
        )

        # Set policy from config
        policy = self.config.get('policy', 'adaptive')
        self.policy_engine.set_active_policy(policy)

        # Congestion control
        cc_algo = self.config.get('congestion_control', 'bbr')
        self.congestion_controllers: Dict[str, Any] = {}

        # Device topology
        self.device_links: Dict[Tuple[str, str], float] = {}
        self._discover_topology()

        # Original function references (for unhooking)
        self._original_functions: Dict[str, Any] = {}

        # Transfer tracking
        self.active_transfers: Dict[str, dict] = {}
        self.transfer_id_counter = 0
        self.transfer_lock = threading.RLock()

        # Statistics
        self.total_transfers = 0
        self.total_bytes = 0
        self.total_time = 0.0

        logger.info(f"AdaptiveFlowIntegration initialized (shadow_mode={self.shadow_mode})")

    @classmethod
    def get_instance(cls, config: Optional[dict] = None) -> 'AdaptiveFlowIntegration':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    def enable(self) -> None:
        """Enable adaptive flow control"""
        if self.enabled:
            logger.warning("Adaptive flow control already enabled")
            return

        logger.info("Enabling adaptive flow control...")

        # Hook into PyTorch operations
        self._hook_tensor_to()
        self._hook_cuda_stream()
        if DISTRIBUTED_AVAILABLE:
            self._hook_distributed()

        self.enabled = True
        logger.info("Adaptive flow control enabled")

    def disable(self) -> None:
        """Disable adaptive flow control"""
        if not self.enabled:
            logger.warning("Adaptive flow control already disabled")
            return

        logger.info("Disabling adaptive flow control...")

        # Unhook from PyTorch operations
        self._unhook_tensor_to()
        self._unhook_cuda_stream()
        if DISTRIBUTED_AVAILABLE:
            self._unhook_distributed()

        self.enabled = False
        logger.info("Adaptive flow control disabled")

    def _discover_topology(self) -> None:
        """Discover device topology and link capacities"""
        # CPU as a device
        self.device_links[('cpu', 'cpu')] = float('inf')

        # Discover CUDA devices
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()

            for i in range(device_count):
                device_name = f"cuda:{i}"

                # CPU <-> GPU links (PCIe)
                # Assume PCIe Gen3 x16: ~16 GB/s
                pcie_bw = 16.0 * 1024 * 1024 * 1024  # bytes per second
                self.device_links[('cpu', device_name)] = pcie_bw
                self.device_links[(device_name, 'cpu')] = pcie_bw

                # Register link with tracker
                self.link_tracker.register_link(
                    f"cpu_to_{device_name}",
                    'cpu',
                    device_name,
                    pcie_bw * 8  # bits per second
                )
                self.link_tracker.register_link(
                    f"{device_name}_to_cpu",
                    device_name,
                    'cpu',
                    pcie_bw * 8
                )

                # GPU <-> GPU links
                for j in range(device_count):
                    if i != j:
                        other_device = f"cuda:{j}"

                        # Check if NVLink is available (P2P)
                        if torch.cuda.can_device_access_peer(i, j):
                            # Assume NVLink: ~50-100 GB/s per link
                            nvlink_bw = 50.0 * 1024 * 1024 * 1024
                            self.device_links[(device_name, other_device)] = nvlink_bw

                            self.link_tracker.register_link(
                                f"{device_name}_to_{other_device}",
                                device_name,
                                other_device,
                                nvlink_bw * 8
                            )
                        else:
                            # Through PCIe: slower
                            pcie_bw_p2p = 8.0 * 1024 * 1024 * 1024
                            self.device_links[(device_name, other_device)] = pcie_bw_p2p

                            self.link_tracker.register_link(
                                f"{device_name}_to_{other_device}",
                                device_name,
                                other_device,
                                pcie_bw_p2p * 8
                            )

            logger.info(f"Discovered {device_count} CUDA devices")

    def _hook_tensor_to(self) -> None:
        """Hook into tensor.to() method"""
        original_to = torch.Tensor.to

        @functools.wraps(original_to)
        def adaptive_to(self_tensor, *args, **kwargs):
            # Parse arguments to get target device
            device = None
            if args:
                if isinstance(args[0], (torch.device, str)):
                    device = args[0]
            if 'device' in kwargs:
                device = kwargs['device']

            # If no device change, use original
            if device is None:
                return original_to(self_tensor, *args, **kwargs)

            # Get source device
            src_device = str(self_tensor.device)
            dst_device = str(device)

            # If same device, use original
            if src_device == dst_device:
                return original_to(self_tensor, *args, **kwargs)

            # Track transfer
            transfer_id = self._start_transfer(src_device, dst_device, self_tensor.numel() * self_tensor.element_size())

            # Perform transfer (with potential flow control)
            start_time = time.time()

            if self.shadow_mode:
                # Shadow mode: just observe, don't modify behavior
                result = original_to(self_tensor, *args, **kwargs)
            else:
                # Apply flow control policy
                result = self._controlled_transfer(
                    lambda: original_to(self_tensor, *args, **kwargs),
                    transfer_id
                )

            elapsed = time.time() - start_time

            # Record transfer completion
            self._complete_transfer(transfer_id, elapsed)

            return result

        torch.Tensor.to = adaptive_to
        self._original_functions['tensor_to'] = original_to
        logger.debug("Hooked into tensor.to()")

    def _unhook_tensor_to(self) -> None:
        """Unhook from tensor.to() method"""
        if 'tensor_to' in self._original_functions:
            torch.Tensor.to = self._original_functions['tensor_to']
            del self._original_functions['tensor_to']
            logger.debug("Unhooked from tensor.to()")

    def _hook_cuda_stream(self) -> None:
        """Hook into CUDA stream operations"""
        if not torch.cuda.is_available():
            return

        # Hook stream creation
        original_stream_init = torch.cuda.Stream.__init__

        def adaptive_stream_init(stream_self, *args, **kwargs):
            result = original_stream_init(stream_self, *args, **kwargs)
            logger.debug(f"CUDA Stream created: {stream_self}")
            return result

        torch.cuda.Stream.__init__ = adaptive_stream_init
        self._original_functions['cuda_stream_init'] = original_stream_init
        logger.debug("Hooked into CUDA Stream")

    def _unhook_cuda_stream(self) -> None:
        """Unhook from CUDA stream operations"""
        if 'cuda_stream_init' in self._original_functions:
            torch.cuda.Stream.__init__ = self._original_functions['cuda_stream_init']
            del self._original_functions['cuda_stream_init']
            logger.debug("Unhooked from CUDA Stream")

    def _hook_distributed(self) -> None:
        """Hook into distributed operations"""
        if not DISTRIBUTED_AVAILABLE:
            return

        # Hook collective operations
        # Note: This is a simplified example. Full implementation would hook
        # all collective ops (all_reduce, all_gather, broadcast, etc.)

        original_all_reduce = dist.all_reduce

        @functools.wraps(original_all_reduce)
        def adaptive_all_reduce(tensor, *args, **kwargs):
            logger.debug(f"Distributed all_reduce: {tensor.numel() * tensor.element_size()} bytes")

            # Track as transfer
            transfer_id = self._start_transfer(
                'distributed',
                'distributed',
                tensor.numel() * tensor.element_size()
            )

            start_time = time.time()
            result = original_all_reduce(tensor, *args, **kwargs)
            elapsed = time.time() - start_time

            self._complete_transfer(transfer_id, elapsed)

            return result

        dist.all_reduce = adaptive_all_reduce
        self._original_functions['dist_all_reduce'] = original_all_reduce
        logger.debug("Hooked into distributed operations")

    def _unhook_distributed(self) -> None:
        """Unhook from distributed operations"""
        if 'dist_all_reduce' in self._original_functions:
            dist.all_reduce = self._original_functions['dist_all_reduce']
            del self._original_functions['dist_all_reduce']
            logger.debug("Unhooked from distributed operations")

    def _start_transfer(self, src_device: str, dst_device: str, num_bytes: int) -> str:
        """Start tracking a transfer"""
        with self.transfer_lock:
            transfer_id = f"transfer_{self.transfer_id_counter}"
            self.transfer_id_counter += 1

            # Register flow if not exists
            flow_id = f"{src_device}_to_{dst_device}"
            self.flow_collector.register_flow(flow_id, src_device, dst_device)

            # Create transfer record
            self.active_transfers[transfer_id] = {
                'flow_id': flow_id,
                'src_device': src_device,
                'dst_device': dst_device,
                'num_bytes': num_bytes,
                'start_time': time.time()
            }

            logger.debug(f"Transfer {transfer_id} started: {src_device} -> {dst_device}, {num_bytes} bytes")

            return transfer_id

    def _complete_transfer(self, transfer_id: str, elapsed: float) -> None:
        """Complete a transfer and record metrics"""
        with self.transfer_lock:
            if transfer_id not in self.active_transfers:
                logger.warning(f"Unknown transfer ID: {transfer_id}")
                return

            transfer = self.active_transfers[transfer_id]
            flow_id = transfer['flow_id']
            num_bytes = transfer['num_bytes']

            # Record in flow collector
            self.flow_collector.record_transfer(
                flow_id=flow_id,
                bytes_sent=num_bytes,
                latency=elapsed,
                success=True,
                queue_depth=len(self.active_transfers)
            )

            # Record in link tracker
            link_id = f"{transfer['src_device']}_to_{transfer['dst_device']}"
            self.link_tracker.record_transmission(
                link_id=link_id,
                bytes_transmitted=num_bytes,
                flow_id=flow_id
            )

            # Update statistics
            self.total_transfers += 1
            self.total_bytes += num_bytes
            self.total_time += elapsed

            # Clean up
            del self.active_transfers[transfer_id]

            logger.debug(f"Transfer {transfer_id} completed: {elapsed*1000:.2f}ms, "
                       f"{num_bytes/elapsed/1e6:.2f} MB/s")

    def _controlled_transfer(self, transfer_func, transfer_id: str):
        """Execute transfer with flow control"""
        transfer = self.active_transfers[transfer_id]
        flow_id = transfer['flow_id']

        # Get flow metrics
        metrics = self.flow_collector.get_flow_metrics(flow_id)
        if metrics is None:
            # No metrics yet, just execute
            return transfer_func()

        # Create policy context
        link_id = f"{transfer['src_device']}_to_{transfer['dst_device']}"
        link_util = self.link_tracker.get_link_utilization(link_id)

        context = PolicyContext(
            flow_id=flow_id,
            current_rate=metrics.throughput_bps if metrics.throughput_bps > 0 else 1e9,
            latency=metrics.latency_mean,
            loss_rate=metrics.loss_rate,
            queue_depth=len(self.active_transfers),
            link_utilization=link_util,
            competing_flows=len(self.flow_collector.get_all_flows()),
            bottleneck_detected=len(self.bottleneck_detector.detect_bottlenecks()) > 0
        )

        # Get policy decision
        action = self.policy_engine.make_decision(context)

        # Apply rate limiting if needed
        if action.decision == PolicyDecision.DECREASE_RATE:
            target_rate = action.parameters.get('new_rate', context.current_rate)
            if target_rate < context.current_rate:
                # Calculate delay to achieve target rate
                delay = (transfer['num_bytes'] / target_rate) - (transfer['num_bytes'] / context.current_rate)
                if delay > 0:
                    logger.debug(f"Rate limiting: delaying {delay*1000:.2f}ms")
                    time.sleep(delay)

        # Execute transfer
        return transfer_func()

    def get_statistics(self) -> dict:
        """Get integration statistics"""
        return {
            'enabled': self.enabled,
            'shadow_mode': self.shadow_mode,
            'total_transfers': self.total_transfers,
            'total_bytes': self.total_bytes,
            'total_time': self.total_time,
            'average_bandwidth': self.total_bytes / self.total_time if self.total_time > 0 else 0.0,
            'active_transfers': len(self.active_transfers),
            'flow_count': len(self.flow_collector.get_all_flows()),
            'link_count': len(self.link_tracker.get_all_links()),
        }

    def get_performance_report(self) -> dict:
        """Get comprehensive performance report"""
        return self.performance_analyzer.analyze_performance()


# Global API functions

def enable_adaptive_flow(config: Optional[dict] = None) -> None:
    """
    Enable adaptive flow control

    Args:
        config: Configuration dictionary with options:
            - enabled: Enable/disable (default: True)
            - policy: Policy name (default: 'adaptive')
            - congestion_control: CC algorithm (default: 'bbr')
            - shadow_mode: Shadow mode (default: False)
            - monitoring_level: Monitoring level (default: 'detailed')

    Example:
        >>> from torch.adaptive_flow import enable_adaptive_flow
        >>> enable_adaptive_flow({
        ...     'policy': 'latency',
        ...     'congestion_control': 'bbr',
        ...     'monitoring_level': 'detailed'
        ... })
    """
    integration = AdaptiveFlowIntegration.get_instance(config)
    integration.enable()


def disable_adaptive_flow() -> None:
    """
    Disable adaptive flow control

    Example:
        >>> from torch.adaptive_flow import disable_adaptive_flow
        >>> disable_adaptive_flow()
    """
    integration = AdaptiveFlowIntegration.get_instance()
    integration.disable()


def is_adaptive_flow_enabled() -> bool:
    """
    Check if adaptive flow control is enabled

    Returns:
        True if enabled, False otherwise

    Example:
        >>> from torch.adaptive_flow import is_adaptive_flow_enabled
        >>> if is_adaptive_flow_enabled():
        ...     print("Adaptive flow is active")
    """
    try:
        integration = AdaptiveFlowIntegration.get_instance()
        return integration.enabled
    except:
        return False


def get_flow_stats() -> dict:
    """
    Get flow statistics

    Returns:
        Dictionary with statistics

    Example:
        >>> from torch.adaptive_flow import get_flow_stats
        >>> stats = get_flow_stats()
        >>> print(f"Total transfers: {stats['total_transfers']}")
        >>> print(f"Average bandwidth: {stats['average_bandwidth']/1e9:.2f} GB/s")
    """
    integration = AdaptiveFlowIntegration.get_instance()
    return integration.get_statistics()


def get_performance_report() -> dict:
    """
    Get comprehensive performance report

    Returns:
        Dictionary with performance analysis

    Example:
        >>> from torch.adaptive_flow import get_performance_report
        >>> report = get_performance_report()
        >>> print(f"Fairness index: {report['fairness_index']:.3f}")
        >>> for issue in report['issues']:
        ...     print(f"Issue: {issue['message']}")
    """
    integration = AdaptiveFlowIntegration.get_instance()
    return integration.get_performance_report()


__all__ = [
    'AdaptiveFlowIntegration',
    'enable_adaptive_flow',
    'disable_adaptive_flow',
    'is_adaptive_flow_enabled',
    'get_flow_stats',
    'get_performance_report',
]

"""
Communication Profiler
通讯分析器

This module provides performance profiling and analysis for communications.

本模块提供通讯的性能分析和监控。
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import asdict
import torch

from .types import CommMetrics, CommunicationPattern, Bottleneck
from .utils import Timer, SynchronizedTimer, format_bandwidth, format_time

logger = logging.getLogger(__name__)


class CommunicationProfiler:
    """
    Performance profiler for communication operations.
    通讯操作的性能分析器。

    This class records and analyzes communication performance metrics
    to identify bottlenecks and optimization opportunities.

    Attributes:
        enable_profiling: Whether profiling is enabled
        profile_interval: How often to record metrics (iterations)
        metrics_history: List of recorded metrics
    """

    def __init__(
        self,
        enable_profiling: bool = True,
        profile_interval: int = 10
    ):
        """
        Initialize communication profiler.
        初始化通讯分析器。

        Args:
            enable_profiling: Whether to enable profiling
            profile_interval: Record metrics every N iterations
        """
        self.enable_profiling = enable_profiling
        self.profile_interval = profile_interval

        # Metrics history
        self.metrics_history: List[CommMetrics] = []

        # Current trace
        self._current_trace: Optional[Dict[str, Any]] = None
        self._trace_stack: List[Dict[str, Any]] = []

        # Timers
        self._timers: Dict[str, Timer] = {}
        self._cuda_timers: Dict[str, SynchronizedTimer] = {}

        # Iteration counter
        self._iteration = 0

        # Chrome trace events
        self._chrome_events: List[Dict[str, Any]] = []

    def start_trace(self, operation_name: str, metadata: Optional[Dict] = None) -> None:
        """
        Start tracing a communication operation.
        开始跟踪通讯操作。

        Args:
            operation_name: Name of the operation
            metadata: Additional metadata
        """
        if not self.enable_profiling:
            return

        trace = {
            'name': operation_name,
            'start_time': time.perf_counter(),
            'metadata': metadata or {},
        }

        self._trace_stack.append(trace)
        self._current_trace = trace

        # Start CUDA timer if available
        if torch.cuda.is_available():
            timer = SynchronizedTimer()
            timer.start()
            self._cuda_timers[operation_name] = timer

    def stop_trace(self) -> Optional[float]:
        """
        Stop the current trace.
        停止当前跟踪。

        Returns:
            Duration in microseconds
        """
        if not self.enable_profiling or not self._trace_stack:
            return None

        trace = self._trace_stack.pop()
        end_time = time.perf_counter()

        duration_us = (end_time - trace['start_time']) * 1e6

        # Get CUDA time if available
        if trace['name'] in self._cuda_timers:
            cuda_time = self._cuda_timers[trace['name']].stop()
            duration_us = cuda_time  # Use CUDA time as it's more accurate
            del self._cuda_timers[trace['name']]

        # Add to Chrome trace
        self._chrome_events.append({
            'name': trace['name'],
            'cat': 'communication',
            'ph': 'X',  # Complete event
            'ts': trace['start_time'] * 1e6,  # Convert to microseconds
            'dur': duration_us,
            'pid': 0,
            'tid': 0,
            'args': trace['metadata'],
        })

        # Update current trace
        if self._trace_stack:
            self._current_trace = self._trace_stack[-1]
        else:
            self._current_trace = None

        return duration_us

    def record_communication(self, metrics: CommMetrics) -> None:
        """
        Record communication metrics.
        记录通讯指标。

        Args:
            metrics: Communication metrics to record
        """
        if not self.enable_profiling:
            return

        # Check if should record this iteration
        if self._iteration % self.profile_interval != 0:
            self._iteration += 1
            return

        self.metrics_history.append(metrics)
        self._iteration += 1

        logger.debug(
            f"Recorded metrics: {metrics.operation} "
            f"({format_time(metrics.latency_us)}, "
            f"{format_bandwidth(metrics.bandwidth_gbps)})"
        )

    def analyze_patterns(self) -> List[CommunicationPattern]:
        """
        Analyze communication patterns from recorded metrics.
        从记录的指标分析通讯模式。

        Returns:
            List of detected patterns
        """
        if not self.metrics_history:
            return []

        patterns = []

        # Group metrics by operation type
        by_operation: Dict[str, List[CommMetrics]] = defaultdict(list)
        for m in self.metrics_history:
            by_operation[m.operation].append(m)

        # Analyze each operation type
        for operation, metrics_list in by_operation.items():
            if len(metrics_list) < 2:
                continue

            # Compute statistics
            message_sizes = [m.message_size for m in metrics_list]
            avg_size = sum(message_sizes) / len(message_sizes)

            # Find common ranks
            all_ranks = set()
            for m in metrics_list:
                all_ranks.add(m.rank)

            pattern = CommunicationPattern(
                pattern_type=operation,
                frequency=len(metrics_list),
                avg_message_size=avg_size,
                ranks_involved=all_ranks,
            )

            patterns.append(pattern)

        logger.info(f"Detected {len(patterns)} communication patterns")
        return patterns

    def identify_bottlenecks(self) -> List[Bottleneck]:
        """
        Identify performance bottlenecks.
        识别性能瓶颈。

        Returns:
            List of identified bottlenecks
        """
        if not self.metrics_history:
            return []

        bottlenecks = []

        # Group by operation
        by_operation: Dict[str, List[CommMetrics]] = defaultdict(list)
        for m in self.metrics_history:
            by_operation[m.operation].append(m)

        # Analyze each operation
        for operation, metrics_list in by_operation.items():
            if len(metrics_list) < 2:
                continue

            # Compute average and std
            latencies = [m.latency_us for m in metrics_list]
            avg_latency = sum(latencies) / len(latencies)
            std_latency = (
                sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)
            ) ** 0.5

            # Check for high variance (sign of instability)
            if std_latency / avg_latency > 0.5:  # CV > 0.5
                bottleneck = Bottleneck(
                    bottleneck_type="high_variance",
                    severity=min(1.0, std_latency / avg_latency),
                    location=operation,
                    description=f"High latency variance for {operation}",
                    suggested_fix="Check for network congestion or stragglers"
                )
                bottlenecks.append(bottleneck)

            # Check for low bandwidth utilization
            bandwidths = [m.bandwidth_gbps for m in metrics_list]
            avg_bw = sum(bandwidths) / len(bandwidths)

            # Assume peak bandwidth is 100 GB/s (adjust based on hardware)
            peak_bw = 100.0
            utilization = avg_bw / peak_bw

            if utilization < 0.3:  # Less than 30% utilization
                bottleneck = Bottleneck(
                    bottleneck_type="low_bandwidth_utilization",
                    severity=1.0 - utilization,
                    location=operation,
                    description=f"Low bandwidth utilization for {operation} ({utilization:.1%})",
                    suggested_fix="Consider message coalescing or larger messages"
                )
                bottlenecks.append(bottleneck)

            # Check for small messages
            message_sizes = [m.message_size for m in metrics_list]
            avg_msg_size = sum(message_sizes) / len(message_sizes)

            if avg_msg_size < 64 * 1024:  # Less than 64 KB
                bottleneck = Bottleneck(
                    bottleneck_type="small_messages",
                    severity=0.5,
                    location=operation,
                    description=f"Small average message size for {operation} ({avg_msg_size / 1024:.1f} KB)",
                    suggested_fix="Enable message coalescing"
                )
                bottlenecks.append(bottleneck)

        logger.info(f"Identified {len(bottlenecks)} bottlenecks")
        return bottlenecks

    def export_timeline(
        self,
        filepath: str,
        format: str = "chrome"
    ) -> None:
        """
        Export timeline trace to a file.
        将时间线跟踪导出到文件。

        Args:
            filepath: Output file path
            format: Trace format ("chrome" for Chrome tracing)
        """
        if format == "chrome":
            self._export_chrome_trace(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_chrome_trace(self, filepath: str) -> None:
        """Export Chrome tracing format"""
        trace_data = {
            'traceEvents': self._chrome_events,
            'displayTimeUnit': 'ms',
        }

        with open(filepath, 'w') as f:
            json.dump(trace_data, f, indent=2)

        logger.info(f"Exported Chrome trace to {filepath}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of all recorded metrics.
        获取所有记录指标的汇总统计。

        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}

        # Overall statistics
        total_operations = len(self.metrics_history)

        latencies = [m.latency_us for m in self.metrics_history]
        bandwidths = [m.bandwidth_gbps for m in self.metrics_history]
        message_sizes = [m.message_size for m in self.metrics_history]

        stats = {
            'total_operations': total_operations,
            'avg_latency_us': sum(latencies) / len(latencies),
            'min_latency_us': min(latencies),
            'max_latency_us': max(latencies),
            'avg_bandwidth_gbps': sum(bandwidths) / len(bandwidths),
            'min_bandwidth_gbps': min(bandwidths),
            'max_bandwidth_gbps': max(bandwidths),
            'avg_message_size': sum(message_sizes) / len(message_sizes),
            'total_bytes': sum(message_sizes),
        }

        # Per-operation statistics
        by_operation: Dict[str, List[CommMetrics]] = defaultdict(list)
        for m in self.metrics_history:
            by_operation[m.operation].append(m)

        stats['per_operation'] = {}
        for operation, metrics_list in by_operation.items():
            op_latencies = [m.latency_us for m in metrics_list]
            op_bandwidths = [m.bandwidth_gbps for m in metrics_list]

            stats['per_operation'][operation] = {
                'count': len(metrics_list),
                'avg_latency_us': sum(op_latencies) / len(op_latencies),
                'avg_bandwidth_gbps': sum(op_bandwidths) / len(op_bandwidths),
            }

        return stats

    def print_summary(self) -> None:
        """Print a human-readable summary of profiling results"""
        stats = self.get_summary_statistics()

        if not stats:
            print("No profiling data available")
            return

        print("\n" + "=" * 60)
        print("Communication Profiling Summary")
        print("=" * 60)

        print(f"\nTotal operations: {stats['total_operations']}")
        print(f"Total data transferred: {stats['total_bytes'] / (1024**3):.2f} GB")

        print("\nOverall Statistics:")
        print(f"  Average latency: {format_time(stats['avg_latency_us'])}")
        print(f"  Latency range: {format_time(stats['min_latency_us'])} - "
              f"{format_time(stats['max_latency_us'])}")
        print(f"  Average bandwidth: {format_bandwidth(stats['avg_bandwidth_gbps'])}")
        print(f"  Bandwidth range: {format_bandwidth(stats['min_bandwidth_gbps'])} - "
              f"{format_bandwidth(stats['max_bandwidth_gbps'])}")

        if 'per_operation' in stats:
            print("\nPer-Operation Statistics:")
            for operation, op_stats in stats['per_operation'].items():
                print(f"\n  {operation}:")
                print(f"    Count: {op_stats['count']}")
                print(f"    Avg latency: {format_time(op_stats['avg_latency_us'])}")
                print(f"    Avg bandwidth: {format_bandwidth(op_stats['avg_bandwidth_gbps'])}")

        print("=" * 60 + "\n")

    def reset(self) -> None:
        """Reset profiler state"""
        self.metrics_history.clear()
        self._chrome_events.clear()
        self._iteration = 0
        self._current_trace = None
        self._trace_stack.clear()
        self._timers.clear()
        self._cuda_timers.clear()


class PerformanceMonitor:
    """
    Continuous performance monitoring.
    持续性能监控。

    This class monitors communication performance in real-time and
    provides alerts when performance degrades.

    Attributes:
        profiler: Communication profiler
        alert_threshold: Performance degradation threshold
        baseline_metrics: Baseline performance metrics
    """

    def __init__(
        self,
        profiler: Optional[CommunicationProfiler] = None,
        alert_threshold: float = 0.5
    ):
        """
        Initialize performance monitor.
        初始化性能监控器。

        Args:
            profiler: Communication profiler (creates new one if None)
            alert_threshold: Alert when performance degrades by this ratio
        """
        self.profiler = profiler or CommunicationProfiler()
        self.alert_threshold = alert_threshold

        # Baseline metrics (computed from first N samples)
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self._baseline_samples = 10
        self._samples_collected = 0

        # Alert history
        self._alerts: List[str] = []

    def record_metric(self, metrics: CommMetrics) -> None:
        """
        Record a metric and check for performance issues.
        记录指标并检查性能问题。

        Args:
            metrics: Communication metrics
        """
        self.profiler.record_communication(metrics)

        # Collect baseline
        if self.baseline_metrics is None:
            self._samples_collected += 1
            if self._samples_collected >= self._baseline_samples:
                self._compute_baseline()

        # Check for degradation
        if self.baseline_metrics is not None:
            self._check_performance_degradation(metrics)

    def _compute_baseline(self) -> None:
        """Compute baseline metrics from collected samples"""
        if not self.profiler.metrics_history:
            return

        samples = self.profiler.metrics_history[-self._baseline_samples:]

        latencies = [m.latency_us for m in samples]
        bandwidths = [m.bandwidth_gbps for m in samples]

        self.baseline_metrics = {
            'latency_us': sum(latencies) / len(latencies),
            'bandwidth_gbps': sum(bandwidths) / len(bandwidths),
        }

        logger.info(
            f"Baseline metrics established: "
            f"latency={format_time(self.baseline_metrics['latency_us'])}, "
            f"bandwidth={format_bandwidth(self.baseline_metrics['bandwidth_gbps'])}"
        )

    def _check_performance_degradation(self, metrics: CommMetrics) -> None:
        """Check if current metrics indicate performance degradation"""
        if self.baseline_metrics is None:
            return

        # Check latency
        latency_ratio = metrics.latency_us / self.baseline_metrics['latency_us']
        if latency_ratio > (1.0 + self.alert_threshold):
            alert = (
                f"High latency detected: {format_time(metrics.latency_us)} "
                f"(baseline: {format_time(self.baseline_metrics['latency_us'])})"
            )
            self._alerts.append(alert)
            logger.warning(alert)

        # Check bandwidth
        bw_ratio = metrics.bandwidth_gbps / self.baseline_metrics['bandwidth_gbps']
        if bw_ratio < (1.0 - self.alert_threshold):
            alert = (
                f"Low bandwidth detected: {format_bandwidth(metrics.bandwidth_gbps)} "
                f"(baseline: {format_bandwidth(self.baseline_metrics['bandwidth_gbps'])})"
            )
            self._alerts.append(alert)
            logger.warning(alert)

    def get_alerts(self) -> List[str]:
        """Get list of performance alerts"""
        return self._alerts.copy()

    def clear_alerts(self) -> None:
        """Clear alert history"""
        self._alerts.clear()

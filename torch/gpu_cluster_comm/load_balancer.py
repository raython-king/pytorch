"""
Load Balancer
负载均衡器

This module provides load balancing for distributed GPU training.

本模块提供分布式GPU训练的负载均衡。
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import torch

from .types import WorkloadStats

logger = logging.getLogger(__name__)


class LoadBalancer:
    """
    Load balancer for distributed training.
    分布式训练的负载均衡器。

    This class monitors workload distribution across ranks and rebalances
    when stragglers are detected.

    Attributes:
        straggler_threshold: Efficiency threshold for straggler detection
        rebalance_interval: How often to check for rebalancing (iterations)
        workload_history: History of workload statistics per rank
    """

    def __init__(
        self,
        straggler_threshold: float = 0.8,
        rebalance_interval: int = 100
    ):
        """
        Initialize load balancer.
        初始化负载均衡器。

        Args:
            straggler_threshold: Efficiency threshold (ranks below are stragglers)
            rebalance_interval: Rebalancing check interval in iterations
        """
        self.straggler_threshold = straggler_threshold
        self.rebalance_interval = rebalance_interval

        # Workload history (per rank)
        self.workload_history: Dict[int, deque] = {}
        self._history_length = 100

        # Iteration counter
        self._iteration = 0

        # Rebalancing statistics
        self._stats = {
            'num_rebalances': 0,
            'num_stragglers_detected': 0,
        }

    def record_workload(self, rank: int, workload: WorkloadStats) -> None:
        """
        Record workload statistics for a rank.
        记录rank的工作负载统计。

        Args:
            rank: Rank identifier
            workload: Workload statistics
        """
        if rank not in self.workload_history:
            self.workload_history[rank] = deque(maxlen=self._history_length)

        self.workload_history[rank].append(workload)

    def detect_stragglers(self) -> List[int]:
        """
        Detect straggler ranks.
        检测落后的rank。

        A rank is considered a straggler if its efficiency is below
        the threshold or if it's significantly slower than others.

        Returns:
            List of straggler rank IDs
        """
        if not self.workload_history:
            return []

        stragglers = []

        # Get latest workload for each rank
        latest_workloads: Dict[int, WorkloadStats] = {}
        for rank, history in self.workload_history.items():
            if history:
                latest_workloads[rank] = history[-1]

        if not latest_workloads:
            return []

        # Compute average efficiency
        efficiencies = {
            rank: workload.efficiency
            for rank, workload in latest_workloads.items()
        }

        avg_efficiency = sum(efficiencies.values()) / len(efficiencies)

        # Detect stragglers by efficiency threshold
        for rank, efficiency in efficiencies.items():
            if efficiency < self.straggler_threshold:
                stragglers.append(rank)
                logger.debug(
                    f"Rank {rank} is a straggler "
                    f"(efficiency: {efficiency:.2%}, threshold: {self.straggler_threshold:.2%})"
                )

        # Also detect if significantly slower than average
        for rank, efficiency in efficiencies.items():
            if rank not in stragglers:
                if efficiency < avg_efficiency * 0.8:  # 20% slower
                    stragglers.append(rank)
                    logger.debug(
                        f"Rank {rank} is significantly slower than average "
                        f"(efficiency: {efficiency:.2%}, avg: {avg_efficiency:.2%})"
                    )

        if stragglers:
            self._stats['num_stragglers_detected'] += len(stragglers)
            logger.info(f"Detected {len(stragglers)} stragglers: {stragglers}")

        return stragglers

    def rebalance_workload(
        self,
        current_assignment: Dict[int, List[Any]]
    ) -> Dict[int, List[Any]]:
        """
        Rebalance workload across ranks.
        重新平衡各rank的工作负载。

        Args:
            current_assignment: Current work assignment (rank -> items)

        Returns:
            New balanced assignment
        """
        if not self.workload_history:
            return current_assignment

        # Get latest workloads
        latest_workloads: Dict[int, WorkloadStats] = {}
        for rank, history in self.workload_history.items():
            if history:
                latest_workloads[rank] = history[-1]

        if not latest_workloads:
            return current_assignment

        # Compute load for each rank (items per second)
        loads = {}
        for rank, workload in latest_workloads.items():
            if workload.throughput > 0:
                loads[rank] = workload.throughput
            else:
                loads[rank] = 1.0  # Default

        # Total work items
        all_items = []
        for items in current_assignment.values():
            all_items.extend(items)

        if not all_items:
            return current_assignment

        # Redistribute proportionally to throughput
        total_throughput = sum(loads.values())
        new_assignment: Dict[int, List[Any]] = {rank: [] for rank in loads.keys()}

        # Sort items (could use more sophisticated logic)
        # For now, just distribute evenly weighted by throughput

        item_idx = 0
        for rank in sorted(loads.keys()):
            # Number of items for this rank
            rank_ratio = loads[rank] / total_throughput
            num_items = int(len(all_items) * rank_ratio)

            # Assign items
            for _ in range(num_items):
                if item_idx < len(all_items):
                    new_assignment[rank].append(all_items[item_idx])
                    item_idx += 1

        # Assign remaining items
        ranks = sorted(loads.keys())
        rank_idx = 0
        while item_idx < len(all_items):
            new_assignment[ranks[rank_idx]].append(all_items[item_idx])
            item_idx += 1
            rank_idx = (rank_idx + 1) % len(ranks)

        self._stats['num_rebalances'] += 1

        logger.info(
            f"Rebalanced workload across {len(new_assignment)} ranks"
        )

        return new_assignment

    def predict_completion_time(
        self,
        rank: int,
        workload: int
    ) -> float:
        """
        Predict completion time for a workload on a rank.
        预测rank上工作负载的完成时间。

        Args:
            rank: Rank identifier
            workload: Number of work items

        Returns:
            Predicted completion time in microseconds
        """
        if rank not in self.workload_history or not self.workload_history[rank]:
            # No history, use default estimate
            return workload * 1000.0  # 1ms per item

        # Get recent throughput
        recent_stats = list(self.workload_history[rank])[-10:]  # Last 10 samples

        avg_throughput = sum(s.throughput for s in recent_stats) / len(recent_stats)

        if avg_throughput > 0:
            # Time = workload / throughput
            time_s = workload / avg_throughput
            return time_s * 1e6  # Convert to microseconds
        else:
            return workload * 1000.0

    def adaptive_batch_sizing(
        self,
        rank: int,
        target_time_us: float
    ) -> int:
        """
        Compute adaptive batch size for a rank.
        计算rank的自适应批次大小。

        Args:
            rank: Rank identifier
            target_time_us: Target processing time in microseconds

        Returns:
            Recommended batch size
        """
        if rank not in self.workload_history or not self.workload_history[rank]:
            # No history, use default
            return 32

        # Get recent throughput
        recent_stats = list(self.workload_history[rank])[-10:]

        avg_throughput = sum(s.throughput for s in recent_stats) / len(recent_stats)

        if avg_throughput > 0:
            # Batch size = throughput * target_time
            target_time_s = target_time_us / 1e6
            batch_size = int(avg_throughput * target_time_s)

            # Clamp to reasonable range
            batch_size = max(1, min(batch_size, 1024))
        else:
            batch_size = 32

        logger.debug(
            f"Adaptive batch size for rank {rank}: {batch_size} "
            f"(throughput: {avg_throughput:.2f} items/s)"
        )

        return batch_size

    def should_rebalance(self) -> bool:
        """
        Check if rebalancing should be performed.
        检查是否应执行重新平衡。

        Returns:
            True if rebalancing is recommended
        """
        self._iteration += 1

        # Check interval
        if self._iteration % self.rebalance_interval != 0:
            return False

        # Check if stragglers exist
        stragglers = self.detect_stragglers()

        return len(stragglers) > 0

    def get_load_imbalance_ratio(self) -> float:
        """
        Compute load imbalance ratio.
        计算负载不平衡比率。

        Returns:
            Imbalance ratio (0 = perfect balance, higher = more imbalanced)
        """
        if not self.workload_history:
            return 0.0

        # Get latest workloads
        latest_workloads: Dict[int, WorkloadStats] = {}
        for rank, history in self.workload_history.items():
            if history:
                latest_workloads[rank] = history[-1]

        if len(latest_workloads) < 2:
            return 0.0

        # Compute total times
        total_times = {
            rank: workload.total_time_us
            for rank, workload in latest_workloads.items()
        }

        # Compute imbalance as (max - min) / avg
        max_time = max(total_times.values())
        min_time = min(total_times.values())
        avg_time = sum(total_times.values()) / len(total_times)

        if avg_time > 0:
            imbalance = (max_time - min_time) / avg_time
        else:
            imbalance = 0.0

        return imbalance

    def get_rank_utilization(self, rank: int) -> Dict[str, float]:
        """
        Get utilization breakdown for a rank.
        获取rank的利用率分解。

        Args:
            rank: Rank identifier

        Returns:
            Dictionary with utilization percentages
        """
        if rank not in self.workload_history or not self.workload_history[rank]:
            return {}

        # Get latest workload
        workload = self.workload_history[rank][-1]

        total_time = workload.total_time_us

        if total_time == 0:
            return {}

        utilization = {
            'compute': (workload.compute_time_us / total_time) * 100.0,
            'communication': (workload.comm_time_us / total_time) * 100.0,
            'idle': (workload.idle_time_us / total_time) * 100.0,
        }

        return utilization

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get load balancing statistics.
        获取负载均衡统计信息。

        Returns:
            Dictionary with statistics
        """
        stats = self._stats.copy()

        # Add current state
        stats['num_ranks'] = len(self.workload_history)
        stats['load_imbalance_ratio'] = self.get_load_imbalance_ratio()
        stats['num_stragglers'] = len(self.detect_stragglers())

        return stats

    def reset_statistics(self) -> None:
        """Reset statistics counters"""
        self._stats = {
            'num_rebalances': 0,
            'num_stragglers_detected': 0,
        }
        self._iteration = 0


class DynamicLoadBalancer(LoadBalancer):
    """
    Dynamic load balancer with predictive rebalancing.
    具有预测性重新平衡的动态负载均衡器。

    Extends LoadBalancer with:
    - Predictive straggler detection
    - Proactive rebalancing
    - Historical trend analysis
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Prediction model (simple moving average for now)
        self._prediction_window = 20

    def predict_stragglers(self, lookahead: int = 10) -> List[int]:
        """
        Predict which ranks will become stragglers.
        预测哪些rank将成为落后者。

        Args:
            lookahead: Number of iterations to look ahead

        Returns:
            List of predicted straggler rank IDs
        """
        if not self.workload_history:
            return []

        predicted_stragglers = []

        for rank, history in self.workload_history.items():
            if len(history) < self._prediction_window:
                continue

            # Analyze trend in efficiency
            recent_efficiencies = [
                workload.efficiency
                for workload in list(history)[-self._prediction_window:]
            ]

            # Compute linear trend
            n = len(recent_efficiencies)
            if n < 2:
                continue

            # Simple linear regression
            x = list(range(n))
            y = recent_efficiencies

            x_mean = sum(x) / n
            y_mean = sum(y) / n

            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if denominator > 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean

                # Predict future efficiency
                future_x = n + lookahead
                predicted_efficiency = slope * future_x + intercept

                if predicted_efficiency < self.straggler_threshold:
                    predicted_stragglers.append(rank)
                    logger.debug(
                        f"Predicted straggler: rank {rank} "
                        f"(current: {recent_efficiencies[-1]:.2%}, "
                        f"predicted: {predicted_efficiency:.2%})"
                    )

        return predicted_stragglers

    def proactive_rebalance(
        self,
        current_assignment: Dict[int, List[Any]]
    ) -> Optional[Dict[int, List[Any]]]:
        """
        Proactively rebalance before stragglers occur.
        在落后者出现之前主动重新平衡。

        Args:
            current_assignment: Current work assignment

        Returns:
            New assignment if rebalancing is needed, None otherwise
        """
        # Predict stragglers
        predicted = self.predict_stragglers(lookahead=10)

        if predicted:
            logger.info(
                f"Proactive rebalancing triggered "
                f"(predicted stragglers: {predicted})"
            )
            return self.rebalance_workload(current_assignment)

        return None

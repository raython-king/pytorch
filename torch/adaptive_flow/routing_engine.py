"""
Routing Engine for Adaptive Flow Control.

Implements multiple routing strategies including shortest path, widest path,
congestion-aware routing, and ML-based adaptive routing.
"""

import time
import logging
import random
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from .topology_manager import NetworkTopology, PathFinder, LinkStatus

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Available routing strategies."""
    SHORTEST_PATH = "shortest_path"
    WIDEST_PATH = "widest_path"
    LEAST_CONGESTED = "least_congested"
    ML_ADAPTIVE = "ml_adaptive"
    MULTIPATH_ECMP = "multipath_ecmp"
    WEIGHTED_MULTIPATH = "weighted_multipath"


@dataclass
class Route:
    """
    Represents a routing decision.

    Attributes:
        path: List of device IDs forming the path
        strategy: Routing strategy used
        cost: Cost of the route
        bandwidth_estimate: Estimated bandwidth (Gbps)
        latency_estimate: Estimated latency (microseconds)
        timestamp: When route was computed
        metadata: Additional metadata
    """
    path: List[int]
    strategy: RoutingStrategy
    cost: float
    bandwidth_estimate: float
    latency_estimate: float
    timestamp: float
    metadata: Dict = None

    def __post_init__(self):
        """Initialize metadata."""
        if self.metadata is None:
            self.metadata = {}

    def is_valid(self, topology: NetworkTopology) -> bool:
        """Check if route is still valid."""
        if not self.path or len(self.path) < 2:
            return False

        # Check all links are still up
        for i in range(len(self.path) - 1):
            link = topology.get_link(self.path[i], self.path[i + 1])
            if not link or link.status == LinkStatus.DOWN:
                return False

        return True

    def get_bottleneck_bandwidth(self, topology: NetworkTopology) -> float:
        """Get minimum bandwidth along the path."""
        if len(self.path) < 2:
            return 0.0

        min_bw = float('inf')
        for i in range(len(self.path) - 1):
            link = topology.get_link(self.path[i], self.path[i + 1])
            if link:
                min_bw = min(min_bw, link.get_effective_bandwidth())

        return min_bw if min_bw != float('inf') else 0.0


class BaseRouter(ABC):
    """Base class for routing strategies."""

    def __init__(self, topology: NetworkTopology):
        """
        Initialize router.

        Args:
            topology: Network topology
        """
        self.topology = topology
        self.path_finder = PathFinder(topology)
        self._route_cache: Dict[Tuple[int, int], Route] = {}
        self._cache_ttl = 1.0  # seconds

    @abstractmethod
    def compute_route(self, src: int, dst: int, **kwargs) -> Optional[Route]:
        """
        Compute route from source to destination.

        Args:
            src: Source device ID
            dst: Destination device ID
            **kwargs: Strategy-specific parameters

        Returns:
            Route or None if no route found
        """
        pass

    def get_route(self, src: int, dst: int, use_cache: bool = True, **kwargs) -> Optional[Route]:
        """
        Get route, using cache if available and valid.

        Args:
            src: Source device
            dst: Destination device
            use_cache: Whether to use cached routes
            **kwargs: Strategy-specific parameters

        Returns:
            Route or None
        """
        cache_key = (src, dst)

        if use_cache and cache_key in self._route_cache:
            route = self._route_cache[cache_key]
            age = time.time() - route.timestamp

            if age < self._cache_ttl and route.is_valid(self.topology):
                logger.debug(f"Using cached route {src} -> {dst}")
                return route
            else:
                del self._route_cache[cache_key]

        # Compute new route
        route = self.compute_route(src, dst, **kwargs)
        if route:
            self._route_cache[cache_key] = route

        return route

    def invalidate_cache(self, src: Optional[int] = None, dst: Optional[int] = None) -> None:
        """Invalidate route cache."""
        if src is None and dst is None:
            self._route_cache.clear()
        else:
            keys_to_remove = [
                key for key in self._route_cache.keys()
                if (src is None or key[0] == src) and (dst is None or key[1] == dst)
            ]
            for key in keys_to_remove:
                del self._route_cache[key]


class ShortestPathRouter(BaseRouter):
    """Routing strategy that minimizes path length or latency."""

    def __init__(self, topology: NetworkTopology, metric: str = "latency"):
        """
        Initialize shortest path router.

        Args:
            topology: Network topology
            metric: Metric to optimize ("latency", "hops")
        """
        super().__init__(topology)
        self.metric = metric

    def compute_route(self, src: int, dst: int, **kwargs) -> Optional[Route]:
        """Compute shortest path route."""
        metric = kwargs.get("metric", self.metric)

        path = self.path_finder.find_shortest_path(src, dst, metric)
        if not path:
            return None

        # Calculate route metrics
        cost = self._calculate_cost(path, metric)
        bandwidth = self._calculate_bandwidth(path)
        latency = self._calculate_latency(path)

        return Route(
            path=path,
            strategy=RoutingStrategy.SHORTEST_PATH,
            cost=cost,
            bandwidth_estimate=bandwidth,
            latency_estimate=latency,
            timestamp=time.time(),
            metadata={"metric": metric}
        )

    def _calculate_cost(self, path: List[int], metric: str) -> float:
        """Calculate path cost."""
        total_cost = 0.0
        for i in range(len(path) - 1):
            cost = self.topology.get_link_cost(path[i], path[i + 1], metric)
            total_cost += cost
        return total_cost

    def _calculate_bandwidth(self, path: List[int]) -> float:
        """Calculate bottleneck bandwidth."""
        if len(path) < 2:
            return 0.0

        min_bw = float('inf')
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                min_bw = min(min_bw, link.get_effective_bandwidth())

        return min_bw if min_bw != float('inf') else 0.0

    def _calculate_latency(self, path: List[int]) -> float:
        """Calculate total path latency."""
        total_latency = 0.0
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                total_latency += link.latency_us
        return total_latency


class WidestPathRouter(BaseRouter):
    """Routing strategy that maximizes minimum bandwidth."""

    def compute_route(self, src: int, dst: int, **kwargs) -> Optional[Route]:
        """Compute widest path route."""
        path = self.path_finder.find_widest_path(src, dst)
        if not path:
            return None

        bandwidth = self._calculate_bottleneck_bandwidth(path)
        latency = self._calculate_latency(path)

        return Route(
            path=path,
            strategy=RoutingStrategy.WIDEST_PATH,
            cost=-bandwidth,  # Negative because we're maximizing
            bandwidth_estimate=bandwidth,
            latency_estimate=latency,
            timestamp=time.time()
        )

    def _calculate_bottleneck_bandwidth(self, path: List[int]) -> float:
        """Calculate bottleneck bandwidth."""
        if len(path) < 2:
            return 0.0

        min_bw = float('inf')
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                min_bw = min(min_bw, link.get_effective_bandwidth())

        return min_bw if min_bw != float('inf') else 0.0

    def _calculate_latency(self, path: List[int]) -> float:
        """Calculate total latency."""
        total_latency = 0.0
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                total_latency += link.latency_us
        return total_latency


class LeastCongestedRouter(BaseRouter):
    """Routing strategy that avoids congested links."""

    def compute_route(self, src: int, dst: int, **kwargs) -> Optional[Route]:
        """Compute least congested route."""
        # Use congestion as the cost metric
        path = self.path_finder.find_shortest_path(src, dst, metric="congestion")
        if not path:
            return None

        congestion = self._calculate_congestion(path)
        bandwidth = self._calculate_bandwidth(path)
        latency = self._calculate_latency(path)

        return Route(
            path=path,
            strategy=RoutingStrategy.LEAST_CONGESTED,
            cost=congestion,
            bandwidth_estimate=bandwidth,
            latency_estimate=latency,
            timestamp=time.time(),
            metadata={"avg_congestion": congestion / len(path) if path else 0.0}
        )

    def _calculate_congestion(self, path: List[int]) -> float:
        """Calculate total path congestion."""
        total_congestion = 0.0
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                total_congestion += link.congestion_level
        return total_congestion

    def _calculate_bandwidth(self, path: List[int]) -> float:
        """Calculate available bandwidth."""
        if len(path) < 2:
            return 0.0

        min_bw = float('inf')
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                min_bw = min(min_bw, link.available_bandwidth)

        return min_bw if min_bw != float('inf') else 0.0

    def _calculate_latency(self, path: List[int]) -> float:
        """Calculate path latency."""
        total_latency = 0.0
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                total_latency += link.latency_us
        return total_latency


class MLRouter(BaseRouter):
    """
    ML-based adaptive routing using learned models.

    Uses a neural network to predict optimal routes based on:
    - Network state
    - Historical performance
    - Traffic patterns
    """

    def __init__(self, topology: NetworkTopology, model: Optional[nn.Module] = None):
        """
        Initialize ML router.

        Args:
            topology: Network topology
            model: Pre-trained model (if None, uses heuristics)
        """
        super().__init__(topology)
        self.model = model
        self.fallback_router = LeastCongestedRouter(topology)

    def compute_route(self, src: int, dst: int, **kwargs) -> Optional[Route]:
        """Compute route using ML model or fallback to heuristics."""
        if self.model is None:
            # Fallback to least congested routing
            return self.fallback_router.compute_route(src, dst, **kwargs)

        try:
            # Extract features for ML model
            features = self._extract_features(src, dst)

            # Predict optimal path using model
            with torch.no_grad():
                prediction = self.model(features)

            # Convert prediction to route
            path = self._prediction_to_path(prediction, src, dst)

            if path:
                bandwidth = self._calculate_bandwidth(path)
                latency = self._calculate_latency(path)

                return Route(
                    path=path,
                    strategy=RoutingStrategy.ML_ADAPTIVE,
                    cost=0.0,  # Model-based
                    bandwidth_estimate=bandwidth,
                    latency_estimate=latency,
                    timestamp=time.time(),
                    metadata={"model_based": True}
                )
        except Exception as e:
            logger.warning(f"ML routing failed: {e}, using fallback")

        # Fallback
        return self.fallback_router.compute_route(src, dst, **kwargs)

    def _extract_features(self, src: int, dst: int) -> torch.Tensor:
        """Extract features for ML model."""
        # Placeholder: would extract network state features
        # - Link utilizations
        # - Congestion levels
        # - Historical performance
        # - Traffic patterns
        device_count = len(self.topology.get_all_devices())
        features = torch.zeros(device_count * device_count * 4)
        return features

    def _prediction_to_path(self, prediction: torch.Tensor, src: int, dst: int) -> Optional[List[int]]:
        """Convert model prediction to a path."""
        # Placeholder: would decode model output to path
        # For now, fallback to shortest path
        return self.path_finder.find_shortest_path(src, dst)

    def _calculate_bandwidth(self, path: List[int]) -> float:
        """Calculate bandwidth."""
        if len(path) < 2:
            return 0.0

        min_bw = float('inf')
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                min_bw = min(min_bw, link.get_effective_bandwidth())

        return min_bw if min_bw != float('inf') else 0.0

    def _calculate_latency(self, path: List[int]) -> float:
        """Calculate latency."""
        total_latency = 0.0
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                total_latency += link.latency_us
        return total_latency


class MultipathRouter(BaseRouter):
    """
    Multipath routing with load balancing.

    Supports:
    - ECMP (Equal-Cost Multi-Path)
    - Weighted multipath based on bandwidth
    """

    def __init__(
        self,
        topology: NetworkTopology,
        k_paths: int = 3,
        load_balance_mode: str = "weighted"
    ):
        """
        Initialize multipath router.

        Args:
            topology: Network topology
            k_paths: Number of paths to use
            load_balance_mode: "ecmp" or "weighted"
        """
        super().__init__(topology)
        self.k_paths = k_paths
        self.load_balance_mode = load_balance_mode
        self._flow_assignments: Dict[Tuple[int, int], int] = {}

    def compute_route(self, src: int, dst: int, **kwargs) -> Optional[Route]:
        """Compute multipath route."""
        flow_id = kwargs.get("flow_id", 0)

        # Find k shortest paths
        paths = self.path_finder.find_k_shortest_paths(src, dst, self.k_paths)
        if not paths:
            return None

        # Select path based on load balancing strategy
        if self.load_balance_mode == "ecmp":
            path = self._select_path_ecmp(paths, flow_id)
        else:
            path = self._select_path_weighted(paths)

        if not path:
            return None

        bandwidth = self._calculate_bandwidth(path)
        latency = self._calculate_latency(path)

        return Route(
            path=path,
            strategy=RoutingStrategy.MULTIPATH_ECMP if self.load_balance_mode == "ecmp"
                     else RoutingStrategy.WEIGHTED_MULTIPATH,
            cost=0.0,
            bandwidth_estimate=bandwidth,
            latency_estimate=latency,
            timestamp=time.time(),
            metadata={
                "available_paths": len(paths),
                "load_balance_mode": self.load_balance_mode
            }
        )

    def _select_path_ecmp(self, paths: List[List[int]], flow_id: int) -> List[int]:
        """Select path using ECMP (hash-based)."""
        # Use flow ID to select path deterministically
        path_index = flow_id % len(paths)
        return paths[path_index]

    def _select_path_weighted(self, paths: List[List[int]]) -> List[int]:
        """Select path using weighted selection based on available bandwidth."""
        # Calculate bandwidth for each path
        bandwidths = []
        for path in paths:
            bw = self._calculate_bandwidth(path)
            bandwidths.append(bw)

        total_bw = sum(bandwidths)
        if total_bw == 0:
            # All paths saturated, use random selection
            return random.choice(paths)

        # Weighted random selection
        rand = random.uniform(0, total_bw)
        cumulative = 0.0
        for i, bw in enumerate(bandwidths):
            cumulative += bw
            if rand <= cumulative:
                return paths[i]

        return paths[-1]

    def _calculate_bandwidth(self, path: List[int]) -> float:
        """Calculate bandwidth."""
        if len(path) < 2:
            return 0.0

        min_bw = float('inf')
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                min_bw = min(min_bw, link.get_effective_bandwidth())

        return min_bw if min_bw != float('inf') else 0.0

    def _calculate_latency(self, path: List[int]) -> float:
        """Calculate latency."""
        total_latency = 0.0
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                total_latency += link.latency_us
        return total_latency


class RoutingEngine:
    """
    Main routing engine that manages multiple routing strategies.

    Provides a unified interface for route computation and supports
    dynamic strategy selection.
    """

    def __init__(self, topology: NetworkTopology, default_strategy: RoutingStrategy = RoutingStrategy.SHORTEST_PATH):
        """
        Initialize routing engine.

        Args:
            topology: Network topology
            default_strategy: Default routing strategy
        """
        self.topology = topology
        self.default_strategy = default_strategy

        # Initialize routers
        self.routers: Dict[RoutingStrategy, BaseRouter] = {
            RoutingStrategy.SHORTEST_PATH: ShortestPathRouter(topology),
            RoutingStrategy.WIDEST_PATH: WidestPathRouter(topology),
            RoutingStrategy.LEAST_CONGESTED: LeastCongestedRouter(topology),
            RoutingStrategy.ML_ADAPTIVE: MLRouter(topology),
            RoutingStrategy.MULTIPATH_ECMP: MultipathRouter(topology, load_balance_mode="ecmp"),
            RoutingStrategy.WEIGHTED_MULTIPATH: MultipathRouter(topology, load_balance_mode="weighted"),
        }

        self._route_stats: Dict[RoutingStrategy, Dict[str, int]] = {
            strategy: {"requests": 0, "successes": 0, "failures": 0}
            for strategy in RoutingStrategy
        }

        logger.info(f"RoutingEngine initialized (default={default_strategy.value})")

    def compute_route(
        self,
        src: int,
        dst: int,
        strategy: Optional[RoutingStrategy] = None,
        **kwargs
    ) -> Optional[Route]:
        """
        Compute route from source to destination.

        Args:
            src: Source device
            dst: Destination device
            strategy: Routing strategy (uses default if None)
            **kwargs: Strategy-specific parameters

        Returns:
            Route or None if no route found
        """
        if strategy is None:
            strategy = self.default_strategy

        router = self.routers.get(strategy)
        if not router:
            logger.error(f"Unknown routing strategy: {strategy}")
            return None

        # Update statistics
        self._route_stats[strategy]["requests"] += 1

        try:
            route = router.get_route(src, dst, **kwargs)

            if route:
                self._route_stats[strategy]["successes"] += 1
            else:
                self._route_stats[strategy]["failures"] += 1

            return route

        except Exception as e:
            logger.error(f"Error computing route: {e}", exc_info=True)
            self._route_stats[strategy]["failures"] += 1
            return None

    def set_default_strategy(self, strategy: RoutingStrategy) -> None:
        """Set default routing strategy."""
        self.default_strategy = strategy
        logger.info(f"Default routing strategy set to {strategy.value}")

    def invalidate_cache(self, strategy: Optional[RoutingStrategy] = None) -> None:
        """
        Invalidate route cache.

        Args:
            strategy: Strategy to invalidate (all if None)
        """
        if strategy is None:
            for router in self.routers.values():
                router.invalidate_cache()
        else:
            router = self.routers.get(strategy)
            if router:
                router.invalidate_cache()

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "default_strategy": self.default_strategy.value,
            "stats": {
                strategy.value: stats.copy()
                for strategy, stats in self._route_stats.items()
            }
        }

    def select_optimal_strategy(
        self,
        src: int,
        dst: int,
        requirements: Optional[Dict[str, Any]] = None
    ) -> RoutingStrategy:
        """
        Select optimal routing strategy based on requirements.

        Args:
            src: Source device
            dst: Destination device
            requirements: Performance requirements (latency, bandwidth, etc.)

        Returns:
            Recommended routing strategy
        """
        if not requirements:
            return self.default_strategy

        # Simple heuristic-based selection
        if requirements.get("low_latency", False):
            return RoutingStrategy.SHORTEST_PATH

        if requirements.get("high_bandwidth", False):
            return RoutingStrategy.WIDEST_PATH

        if requirements.get("avoid_congestion", False):
            return RoutingStrategy.LEAST_CONGESTED

        if requirements.get("load_balance", False):
            return RoutingStrategy.WEIGHTED_MULTIPATH

        return self.default_strategy


__all__ = [
    "RoutingStrategy",
    "Route",
    "BaseRouter",
    "ShortestPathRouter",
    "WidestPathRouter",
    "LeastCongestedRouter",
    "MLRouter",
    "MultipathRouter",
    "RoutingEngine",
]

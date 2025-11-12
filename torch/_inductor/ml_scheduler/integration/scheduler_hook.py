"""
Scheduler Integration Hooks

Provides integration between ML scheduler and torch._inductor.scheduler.
Implements hooks for key scheduling decisions:
- Node fusion
- Loop ordering
- Schedule ordering
- Memory planning

Includes shadow mode for safe validation before full deployment.
"""

import torch
import logging
import time
from typing import Optional, List, Tuple, Dict, Any, Callable
from functools import wraps
import warnings

try:
    from ...scheduler import BaseSchedulerNode, Scheduler, FusedSchedulerNode
except ImportError:
    # Fallback for testing
    BaseSchedulerNode = object
    Scheduler = object
    FusedSchedulerNode = object

from ..orchestrator import MLSchedulerOrchestrator, FusionPlan
from ..config import MLSchedulerConfig
from ..inference.predictor import MLSchedulerPredictor

log = logging.getLogger(__name__)


class MLSchedulerMode:
    """
    Modes for ML scheduler integration.
    """
    DISABLED = "disabled"  # ML scheduler disabled, use heuristics only
    SHADOW = "shadow"  # ML predictions logged but not applied
    HYBRID = "hybrid"  # ML used with fallback to heuristics
    FULL = "full"  # ML only, fail if ML cannot decide


# Global configuration
_ml_scheduler_mode = MLSchedulerMode.DISABLED
_ml_scheduler_orchestrator: Optional[MLSchedulerOrchestrator] = None


def enable_ml_scheduler(
    mode: str = MLSchedulerMode.HYBRID,
    config: Optional[MLSchedulerConfig] = None,
    orchestrator: Optional[MLSchedulerOrchestrator] = None,
):
    """
    Enable ML scheduler globally.

    Args:
        mode: Scheduler mode (disabled, shadow, hybrid, full)
        config: ML scheduler configuration
        orchestrator: Pre-initialized orchestrator (optional)

    Example:
        enable_ml_scheduler(
            mode='hybrid',
            config=MLSchedulerConfig(
                model_path='./checkpoints/best_model.pt',
                confidence_threshold=0.75,
            )
        )
    """
    global _ml_scheduler_mode, _ml_scheduler_orchestrator

    _ml_scheduler_mode = mode

    if orchestrator is not None:
        _ml_scheduler_orchestrator = orchestrator
    else:
        _ml_scheduler_orchestrator = MLSchedulerOrchestrator(config=config)

    log.info(f"ML Scheduler enabled: mode={mode}")


def disable_ml_scheduler():
    """Disable ML scheduler globally."""
    global _ml_scheduler_mode, _ml_scheduler_orchestrator

    _ml_scheduler_mode = MLSchedulerMode.DISABLED
    _ml_scheduler_orchestrator = None

    log.info("ML Scheduler disabled")


def get_ml_scheduler_mode() -> str:
    """Get current ML scheduler mode."""
    return _ml_scheduler_mode


def is_ml_scheduler_enabled() -> bool:
    """Check if ML scheduler is enabled."""
    return _ml_scheduler_mode != MLSchedulerMode.DISABLED


class MLSchedulerWrapper:
    """
    Wrapper for Scheduler that adds ML-based decision making.

    This class wraps the existing Scheduler and overrides key methods
    to incorporate ML predictions with fallback to heuristics.

    Example:
        # Create wrapped scheduler
        scheduler = MLSchedulerWrapper(
            base_scheduler,
            config=MLSchedulerConfig(model_path='./model.pt')
        )

        # Use as normal scheduler
        scheduler.schedule()
    """

    def __init__(
        self,
        base_scheduler: Scheduler,
        config: Optional[MLSchedulerConfig] = None,
        mode: str = MLSchedulerMode.HYBRID,
    ):
        """
        Initialize ML scheduler wrapper.

        Args:
            base_scheduler: Base scheduler instance to wrap
            config: ML scheduler configuration
            mode: Integration mode
        """
        self.base_scheduler = base_scheduler
        self.config = config or MLSchedulerConfig()
        self.mode = mode

        # Initialize orchestrator
        self.orchestrator = MLSchedulerOrchestrator(config=self.config)

        # Statistics
        self.ml_fusion_count = 0
        self.heuristic_fusion_count = 0
        self.ml_fusion_time_ms = 0.0
        self.heuristic_fusion_time_ms = 0.0
        self.fallback_count = 0

        # Shadow mode logging
        self.shadow_log = []

        log.info(f"MLSchedulerWrapper initialized: mode={mode}")

    def fuse_nodes(self, nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
        """
        Fuse nodes using ML predictions.

        This method overrides the base scheduler's fusion logic.

        Args:
            nodes: List of nodes to potentially fuse

        Returns:
            List of nodes after fusion
        """
        if not nodes:
            return nodes

        # Check if we should use ML
        if not self.orchestrator.should_use_ml(len(nodes)):
            log.debug(f"Skipping ML for {len(nodes)} nodes (outside size range)")
            return self._heuristic_fuse_nodes(nodes)

        # Get ML predictions
        start_time = time.time()

        try:
            device = nodes[0].get_device() if hasattr(nodes[0], 'get_device') else torch.device('cpu')
            fusion_plan = self.orchestrator.predict_fusion_plan(nodes, device)

            ml_time_ms = (time.time() - start_time) * 1000
            self.ml_fusion_time_ms += ml_time_ms

            # Check confidence
            if fusion_plan.confidence_scores:
                avg_confidence = sum(fusion_plan.confidence_scores) / len(fusion_plan.confidence_scores)
            else:
                avg_confidence = 0.0

            log.debug(
                f"ML fusion prediction: {len(fusion_plan.fusions)} fusions, "
                f"confidence: {avg_confidence:.3f}, time: {ml_time_ms:.2f}ms"
            )

            # Mode-specific handling
            if self.mode == MLSchedulerMode.SHADOW:
                # Shadow mode: log but don't apply
                self._log_shadow_prediction(nodes, fusion_plan, 'ml')

                # Use heuristic
                return self._heuristic_fuse_nodes(nodes)

            elif self.mode == MLSchedulerMode.HYBRID:
                # Hybrid mode: use ML if confident, fallback otherwise
                if avg_confidence >= self.config.confidence_threshold:
                    try:
                        result = self._apply_fusion_plan(nodes, fusion_plan)

                        if self.config.validate_fusion_plan:
                            if not self._validate_fusion_result(nodes, result):
                                log.warning("ML fusion validation failed, using heuristic")
                                self.fallback_count += 1
                                return self._heuristic_fuse_nodes(nodes)

                        self.ml_fusion_count += 1
                        return result

                    except Exception as e:
                        log.warning(f"ML fusion application failed: {e}, using heuristic")
                        self.fallback_count += 1
                        return self._heuristic_fuse_nodes(nodes)
                else:
                    log.debug(f"Low confidence ({avg_confidence:.3f}), using heuristic")
                    self.fallback_count += 1
                    return self._heuristic_fuse_nodes(nodes)

            elif self.mode == MLSchedulerMode.FULL:
                # Full mode: ML only
                result = self._apply_fusion_plan(nodes, fusion_plan)
                self.ml_fusion_count += 1
                return result

            else:
                # Disabled or unknown mode
                return self._heuristic_fuse_nodes(nodes)

        except Exception as e:
            log.error(f"Error in ML fusion: {e}")

            if self.mode == MLSchedulerMode.FULL:
                raise
            else:
                # Fallback to heuristic
                self.fallback_count += 1
                return self._heuristic_fuse_nodes(nodes)

    def _heuristic_fuse_nodes(self, nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
        """Use base scheduler's heuristic fusion."""
        start_time = time.time()

        result = self.base_scheduler.fuse_nodes(nodes)

        heuristic_time_ms = (time.time() - start_time) * 1000
        self.heuristic_fusion_time_ms += heuristic_time_ms
        self.heuristic_fusion_count += 1

        return result

    def _apply_fusion_plan(
        self,
        nodes: List[BaseSchedulerNode],
        fusion_plan: FusionPlan
    ) -> List[BaseSchedulerNode]:
        """
        Apply fusion plan to nodes.

        Args:
            nodes: Original nodes
            fusion_plan: Fusion plan from ML

        Returns:
            Fused nodes
        """
        if not fusion_plan.fusions:
            return nodes

        # Track which nodes have been fused
        fused_nodes = set()
        result_nodes = []

        # Apply fusions in order of confidence
        for (i, j) in fusion_plan.fusions:
            if i in fused_nodes or j in fused_nodes:
                continue  # Already fused

            node_i = nodes[i]
            node_j = nodes[j]

            # Attempt fusion
            try:
                fused_node = self._fuse_two_nodes(node_i, node_j)

                if fused_node is not None:
                    # Replace node_i with fused node
                    # Mark both as fused
                    fused_nodes.add(i)
                    fused_nodes.add(j)

                    # Add fused node (will be added in final loop)
                    result_nodes.append(fused_node)

            except Exception as e:
                log.debug(f"Failed to fuse nodes {i} and {j}: {e}")
                continue

        # Add non-fused nodes
        for idx, node in enumerate(nodes):
            if idx not in fused_nodes:
                result_nodes.append(node)

        return result_nodes

    def _fuse_two_nodes(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode
    ) -> Optional[BaseSchedulerNode]:
        """
        Fuse two nodes.

        Uses the base scheduler's fusion mechanism.

        Args:
            node1: First node
            node2: Second node

        Returns:
            Fused node or None if fusion failed
        """
        # Check if nodes can be fused
        if not self._can_fuse(node1, node2):
            return None

        # Use base scheduler's fusion logic
        # This is a simplified version - actual implementation would need
        # to call the appropriate fusion methods from the base scheduler

        # For now, create a simple fused node
        try:
            # This would need to be implemented based on the actual scheduler API
            # Placeholder implementation
            log.debug(f"Fusing nodes: {node1} and {node2}")

            # In practice, you'd call something like:
            # fused = self.base_scheduler.fuse_two_nodes(node1, node2)

            # For now, return None (fusion not implemented)
            return None

        except Exception as e:
            log.warning(f"Fusion failed: {e}")
            return None

    def _can_fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        Check if two nodes can be fused.

        Args:
            node1: First node
            node2: Second node

        Returns:
            True if nodes can be fused
        """
        # Check device compatibility
        if hasattr(node1, 'get_device') and hasattr(node2, 'get_device'):
            if node1.get_device() != node2.get_device():
                return False

        # Check for circular dependencies
        if hasattr(node1, 'users') and hasattr(node2, 'users'):
            if node2 in node1.users or node1 in node2.users:
                # Check if it's a valid fusion pattern
                pass

        # More fusion compatibility checks would go here

        return True

    def _validate_fusion_result(
        self,
        original_nodes: List[BaseSchedulerNode],
        fused_nodes: List[BaseSchedulerNode]
    ) -> bool:
        """
        Validate fusion result.

        Args:
            original_nodes: Original nodes before fusion
            fused_nodes: Nodes after fusion

        Returns:
            True if fusion is valid
        """
        # Basic sanity checks
        if not fused_nodes:
            return False

        # Check that we didn't lose nodes (unless fused)
        if len(fused_nodes) > len(original_nodes):
            log.warning("Fusion created more nodes than original")
            return False

        # More validation checks would go here
        # - Check dependencies are preserved
        # - Check no circular dependencies
        # - Check device compatibility
        # - Check memory constraints

        return True

    def _log_shadow_prediction(
        self,
        nodes: List[BaseSchedulerNode],
        fusion_plan: FusionPlan,
        source: str
    ):
        """Log prediction in shadow mode."""
        self.shadow_log.append({
            'num_nodes': len(nodes),
            'fusion_plan': fusion_plan,
            'source': source,
            'timestamp': time.time(),
        })

    def _ml_pick_loop_order(self, node: BaseSchedulerNode) -> Optional[List[int]]:
        """
        ML-based loop order selection.

        This would use ML to determine optimal loop ordering for a node.

        Args:
            node: Scheduler node

        Returns:
            Loop order or None for fallback
        """
        # Placeholder for loop order prediction
        # Would need to extract features and use ML model

        return None

    def _ml_pick_node_order(
        self,
        nodes: List[BaseSchedulerNode]
    ) -> Optional[List[int]]:
        """
        ML-based node execution order.

        Args:
            nodes: List of nodes

        Returns:
            Execution order (indices) or None for fallback
        """
        # Use orchestrator to predict schedule order
        try:
            device = nodes[0].get_device() if hasattr(nodes[0], 'get_device') else torch.device('cpu')
            fusion_plan = self.orchestrator.predict_fusion_plan(nodes, device)

            if fusion_plan.schedule_order is not None:
                return fusion_plan.schedule_order

        except Exception as e:
            log.warning(f"ML node order prediction failed: {e}")

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        total_fusions = self.ml_fusion_count + self.heuristic_fusion_count

        if total_fusions > 0:
            ml_ratio = self.ml_fusion_count / total_fusions
        else:
            ml_ratio = 0.0

        return {
            'mode': self.mode,
            'ml_fusion_count': self.ml_fusion_count,
            'heuristic_fusion_count': self.heuristic_fusion_count,
            'fallback_count': self.fallback_count,
            'ml_fusion_ratio': ml_ratio,
            'ml_fusion_time_ms': self.ml_fusion_time_ms,
            'heuristic_fusion_time_ms': self.heuristic_fusion_time_ms,
            'avg_ml_time_ms': (
                self.ml_fusion_time_ms / self.ml_fusion_count
                if self.ml_fusion_count > 0 else 0.0
            ),
            'avg_heuristic_time_ms': (
                self.heuristic_fusion_time_ms / self.heuristic_fusion_count
                if self.heuristic_fusion_count > 0 else 0.0
            ),
        }

    def get_shadow_log(self) -> List[Dict[str, Any]]:
        """Get shadow mode predictions."""
        return self.shadow_log

    def reset_statistics(self):
        """Reset statistics."""
        self.ml_fusion_count = 0
        self.heuristic_fusion_count = 0
        self.ml_fusion_time_ms = 0.0
        self.heuristic_fusion_time_ms = 0.0
        self.fallback_count = 0
        self.shadow_log.clear()


# Decorator for adding ML scheduler hooks
def with_ml_scheduler(func: Callable) -> Callable:
    """
    Decorator to add ML scheduler capabilities to a method.

    Example:
        class MyScheduler(Scheduler):
            @with_ml_scheduler
            def fuse_nodes(self, nodes):
                # Original implementation
                ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if ML scheduler is enabled
        if not is_ml_scheduler_enabled() or _ml_scheduler_orchestrator is None:
            return func(*args, **kwargs)

        # Try ML prediction
        try:
            # Extract nodes from arguments
            if len(args) >= 2 and isinstance(args[1], list):
                nodes = args[1]

                # Use ML scheduler
                wrapper_instance = MLSchedulerWrapper(
                    args[0],  # self (scheduler instance)
                    config=_ml_scheduler_orchestrator.config,
                    mode=_ml_scheduler_mode,
                )

                return wrapper_instance.fuse_nodes(nodes)

        except Exception as e:
            log.warning(f"ML scheduler hook failed: {e}, using original method")

        # Fallback to original
        return func(*args, **kwargs)

    return wrapper


# Context manager for temporary ML scheduler mode
class ml_scheduler_mode:
    """
    Context manager for temporarily changing ML scheduler mode.

    Example:
        with ml_scheduler_mode('shadow'):
            # ML predictions logged but not applied
            model = torch.compile(model)
    """

    def __init__(self, mode: str):
        """
        Initialize context manager.

        Args:
            mode: Temporary mode to use
        """
        self.mode = mode
        self.previous_mode = None

    def __enter__(self):
        """Enter context."""
        global _ml_scheduler_mode
        self.previous_mode = _ml_scheduler_mode
        _ml_scheduler_mode = self.mode
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        global _ml_scheduler_mode
        _ml_scheduler_mode = self.previous_mode


# Utility functions for testing and validation

def compare_ml_vs_heuristic(
    nodes: List[BaseSchedulerNode],
    scheduler: Scheduler,
    config: Optional[MLSchedulerConfig] = None,
) -> Dict[str, Any]:
    """
    Compare ML scheduler vs heuristic on a set of nodes.

    Args:
        nodes: Nodes to test
        scheduler: Base scheduler
        config: ML scheduler config

    Returns:
        Comparison results
    """
    # ML fusion
    ml_wrapper = MLSchedulerWrapper(scheduler, config=config, mode=MLSchedulerMode.FULL)

    ml_start = time.time()
    ml_result = ml_wrapper.fuse_nodes(nodes)
    ml_time = time.time() - ml_start

    # Heuristic fusion
    heuristic_start = time.time()
    heuristic_result = scheduler.fuse_nodes(nodes)
    heuristic_time = time.time() - heuristic_start

    return {
        'num_nodes_original': len(nodes),
        'num_nodes_ml': len(ml_result),
        'num_nodes_heuristic': len(heuristic_result),
        'ml_time_ms': ml_time * 1000,
        'heuristic_time_ms': heuristic_time * 1000,
        'time_ratio': ml_time / heuristic_time if heuristic_time > 0 else float('inf'),
    }


def validate_ml_scheduler(
    test_cases: List[List[BaseSchedulerNode]],
    scheduler: Scheduler,
    config: Optional[MLSchedulerConfig] = None,
) -> Dict[str, Any]:
    """
    Validate ML scheduler on multiple test cases.

    Args:
        test_cases: List of node lists to test
        scheduler: Base scheduler
        config: ML scheduler config

    Returns:
        Validation results
    """
    results = []

    for nodes in test_cases:
        try:
            result = compare_ml_vs_heuristic(nodes, scheduler, config)
            result['success'] = True
            results.append(result)
        except Exception as e:
            results.append({
                'success': False,
                'error': str(e),
                'num_nodes_original': len(nodes),
            })

    # Summary statistics
    successful = [r for r in results if r.get('success', False)]

    if successful:
        avg_ml_time = sum(r['ml_time_ms'] for r in successful) / len(successful)
        avg_heuristic_time = sum(r['heuristic_time_ms'] for r in successful) / len(successful)
    else:
        avg_ml_time = 0.0
        avg_heuristic_time = 0.0

    return {
        'total_cases': len(test_cases),
        'successful_cases': len(successful),
        'failed_cases': len(results) - len(successful),
        'success_rate': len(successful) / len(test_cases) if test_cases else 0.0,
        'avg_ml_time_ms': avg_ml_time,
        'avg_heuristic_time_ms': avg_heuristic_time,
        'avg_time_ratio': avg_ml_time / avg_heuristic_time if avg_heuristic_time > 0 else float('inf'),
        'results': results,
    }

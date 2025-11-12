"""
Adaptive Policy Engine for Flow Control

This module implements adaptive policies for flow control with multi-objective
optimization, policy composition, and runtime policy switching.

Policies:
- AdaptivePolicy: Dynamically adjusts strategies based on network conditions
- FairnessPolicy: Ensures fair resource allocation
- LatencyPolicy: Minimizes end-to-end latency
- ThroughputPolicy: Maximizes aggregate throughput
- EnergyPolicy: Optimizes for energy efficiency

Features:
- Multi-objective optimization
- Policy composition and chaining
- Runtime policy switching
- Learning from feedback
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import statistics

from .flow_monitor import (
    FlowMetricsCollector,
    LinkUtilizationTracker,
    BottleneckDetector,
    PerformanceAnalyzer
)

logger = logging.getLogger(__name__)


class PolicyObjective(Enum):
    """Policy optimization objectives"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    FAIRNESS = "fairness"
    ENERGY = "energy"
    BALANCED = "balanced"


class PolicyDecision(Enum):
    """Policy decision types"""
    INCREASE_RATE = "increase_rate"
    DECREASE_RATE = "decrease_rate"
    MAINTAIN_RATE = "maintain_rate"
    REROUTE = "reroute"
    PRIORITIZE = "prioritize"
    DEPRIORITIZE = "deprioritize"
    SPLIT_FLOW = "split_flow"


@dataclass
class PolicyContext:
    """Context information for policy decisions"""
    flow_id: str
    current_rate: float
    target_rate: Optional[float] = None
    latency: float = 0.0
    loss_rate: float = 0.0
    queue_depth: int = 0
    link_utilization: float = 0.0
    competing_flows: int = 0
    bottleneck_detected: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class PolicyAction:
    """Action recommended by policy"""
    decision: PolicyDecision
    flow_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    reason: str = ""
    confidence: float = 1.0


class Policy(ABC):
    """Base class for flow control policies"""

    def __init__(self, name: str):
        """
        Initialize policy

        Args:
            name: Policy name
        """
        self.name = name
        self.enabled = True
        self.lock = threading.RLock()

        # Performance tracking
        self.decisions_made = 0
        self.successful_decisions = 0
        self.failed_decisions = 0

        logger.info(f"Policy '{name}' initialized")

    @abstractmethod
    def evaluate(self, context: PolicyContext) -> PolicyAction:
        """
        Evaluate policy and return action

        Args:
            context: Current context for decision

        Returns:
            PolicyAction to take
        """
        pass

    def learn_from_feedback(self, action: PolicyAction, outcome: dict) -> None:
        """
        Learn from action outcome

        Args:
            action: Action that was taken
            outcome: Results of the action
        """
        with self.lock:
            self.decisions_made += 1

            # Simple success/failure tracking
            if outcome.get('success', True):
                self.successful_decisions += 1
            else:
                self.failed_decisions += 1

    def get_success_rate(self) -> float:
        """Get policy success rate"""
        with self.lock:
            if self.decisions_made == 0:
                return 1.0
            return self.successful_decisions / self.decisions_made

    def reset_stats(self) -> None:
        """Reset performance statistics"""
        with self.lock:
            self.decisions_made = 0
            self.successful_decisions = 0
            self.failed_decisions = 0


class LatencyPolicy(Policy):
    """
    Policy focused on minimizing latency

    Prioritizes low latency over throughput, quickly reacts to latency increases,
    and avoids queueing delays.
    """

    def __init__(self, target_latency: float = 0.001, tolerance: float = 0.1):
        """
        Initialize latency policy

        Args:
            target_latency: Target latency in seconds
            tolerance: Acceptable latency variance (fraction)
        """
        super().__init__("LatencyPolicy")
        self.target_latency = target_latency
        self.tolerance = tolerance

        logger.info(f"LatencyPolicy: target={target_latency*1000:.1f}ms, tolerance={tolerance*100:.1f}%")

    def evaluate(self, context: PolicyContext) -> PolicyAction:
        """Evaluate latency-focused policy"""
        with self.lock:
            # Calculate latency deviation
            if context.latency > 0:
                latency_ratio = context.latency / self.target_latency
            else:
                latency_ratio = 1.0

            # Decision logic
            if latency_ratio > (1.0 + self.tolerance):
                # Latency too high, reduce rate
                reduction = min(0.5, (latency_ratio - 1.0))
                new_rate = context.current_rate * (1.0 - reduction)

                return PolicyAction(
                    decision=PolicyDecision.DECREASE_RATE,
                    flow_id=context.flow_id,
                    parameters={'new_rate': new_rate, 'reduction': reduction},
                    priority=1,
                    reason=f"Latency {context.latency*1000:.1f}ms exceeds target {self.target_latency*1000:.1f}ms",
                    confidence=min(1.0, reduction * 2)
                )

            elif latency_ratio < (1.0 - self.tolerance) and not context.bottleneck_detected:
                # Latency below target and no bottleneck, can increase
                increase = 0.1
                new_rate = context.current_rate * (1.0 + increase)

                return PolicyAction(
                    decision=PolicyDecision.INCREASE_RATE,
                    flow_id=context.flow_id,
                    parameters={'new_rate': new_rate, 'increase': increase},
                    priority=0,
                    reason=f"Latency {context.latency*1000:.1f}ms below target, room for increase",
                    confidence=0.7
                )

            else:
                # Latency in acceptable range
                return PolicyAction(
                    decision=PolicyDecision.MAINTAIN_RATE,
                    flow_id=context.flow_id,
                    parameters={'current_rate': context.current_rate},
                    priority=0,
                    reason="Latency within target range",
                    confidence=0.9
                )


class ThroughputPolicy(Policy):
    """
    Policy focused on maximizing throughput

    Aggressively increases sending rate while monitoring for congestion signals.
    """

    def __init__(self, target_utilization: float = 0.9):
        """
        Initialize throughput policy

        Args:
            target_utilization: Target link utilization (0.0 to 1.0)
        """
        super().__init__("ThroughputPolicy")
        self.target_utilization = target_utilization

        logger.info(f"ThroughputPolicy: target_utilization={target_utilization*100:.0f}%")

    def evaluate(self, context: PolicyContext) -> PolicyAction:
        """Evaluate throughput-focused policy"""
        with self.lock:
            # Check for loss
            if context.loss_rate > 0.01:  # 1% loss
                # Significant loss, reduce rate
                reduction = min(0.5, context.loss_rate * 10)
                new_rate = context.current_rate * (1.0 - reduction)

                return PolicyAction(
                    decision=PolicyDecision.DECREASE_RATE,
                    flow_id=context.flow_id,
                    parameters={'new_rate': new_rate, 'reduction': reduction},
                    priority=2,
                    reason=f"Packet loss detected: {context.loss_rate*100:.2f}%",
                    confidence=0.9
                )

            # Check link utilization
            if context.link_utilization < self.target_utilization:
                # Link underutilized, increase rate
                headroom = self.target_utilization - context.link_utilization
                increase = min(0.2, headroom * 2)
                new_rate = context.current_rate * (1.0 + increase)

                return PolicyAction(
                    decision=PolicyDecision.INCREASE_RATE,
                    flow_id=context.flow_id,
                    parameters={'new_rate': new_rate, 'increase': increase},
                    priority=0,
                    reason=f"Link utilization {context.link_utilization*100:.0f}% below target",
                    confidence=0.8
                )

            elif context.link_utilization > 0.95:
                # Link over-utilized
                reduction = 0.1
                new_rate = context.current_rate * (1.0 - reduction)

                return PolicyAction(
                    decision=PolicyDecision.DECREASE_RATE,
                    flow_id=context.flow_id,
                    parameters={'new_rate': new_rate, 'reduction': reduction},
                    priority=1,
                    reason=f"Link utilization {context.link_utilization*100:.0f}% exceeds capacity",
                    confidence=0.9
                )

            else:
                return PolicyAction(
                    decision=PolicyDecision.MAINTAIN_RATE,
                    flow_id=context.flow_id,
                    parameters={'current_rate': context.current_rate},
                    priority=0,
                    reason="Link utilization at target",
                    confidence=0.8
                )


class FairnessPolicy(Policy):
    """
    Policy focused on ensuring fairness among flows

    Implements max-min fairness and Jain's fairness index optimization.
    """

    def __init__(self, fairness_threshold: float = 0.7):
        """
        Initialize fairness policy

        Args:
            fairness_threshold: Minimum acceptable Jain's fairness index
        """
        super().__init__("FairnessPolicy")
        self.fairness_threshold = fairness_threshold
        self.flow_rates: Dict[str, float] = {}

        logger.info(f"FairnessPolicy: threshold={fairness_threshold}")

    def evaluate(self, context: PolicyContext) -> PolicyAction:
        """Evaluate fairness-focused policy"""
        with self.lock:
            # Track this flow's rate
            self.flow_rates[context.flow_id] = context.current_rate

            if len(self.flow_rates) < 2:
                # Need multiple flows for fairness
                return PolicyAction(
                    decision=PolicyDecision.MAINTAIN_RATE,
                    flow_id=context.flow_id,
                    parameters={'current_rate': context.current_rate},
                    priority=0,
                    reason="Insufficient flows for fairness evaluation",
                    confidence=0.5
                )

            # Calculate fairness index
            rates = list(self.flow_rates.values())
            n = len(rates)
            sum_rates = sum(rates)
            sum_rates_squared = sum(r * r for r in rates)

            if sum_rates_squared > 0:
                fairness = (sum_rates * sum_rates) / (n * sum_rates_squared)
            else:
                fairness = 1.0

            # Find average rate
            avg_rate = sum_rates / n

            # Determine if this flow should adjust
            rate_ratio = context.current_rate / avg_rate if avg_rate > 0 else 1.0

            if fairness < self.fairness_threshold:
                # Fairness is poor
                if rate_ratio > 1.2:
                    # This flow is using more than fair share, reduce
                    new_rate = (context.current_rate + avg_rate) / 2

                    return PolicyAction(
                        decision=PolicyDecision.DECREASE_RATE,
                        flow_id=context.flow_id,
                        parameters={'new_rate': new_rate, 'target_rate': avg_rate},
                        priority=1,
                        reason=f"Fairness {fairness:.3f} below threshold, flow using {rate_ratio:.2f}x fair share",
                        confidence=0.8
                    )

                elif rate_ratio < 0.8:
                    # This flow is using less than fair share, increase
                    new_rate = (context.current_rate + avg_rate) / 2

                    return PolicyAction(
                        decision=PolicyDecision.INCREASE_RATE,
                        flow_id=context.flow_id,
                        parameters={'new_rate': new_rate, 'target_rate': avg_rate},
                        priority=1,
                        reason=f"Fairness {fairness:.3f} below threshold, flow using {rate_ratio:.2f}x fair share",
                        confidence=0.8
                    )

            return PolicyAction(
                decision=PolicyDecision.MAINTAIN_RATE,
                flow_id=context.flow_id,
                parameters={'current_rate': context.current_rate},
                priority=0,
                reason=f"Fairness {fairness:.3f} acceptable",
                confidence=0.7
            )


class EnergyPolicy(Policy):
    """
    Policy focused on energy efficiency

    Optimizes for lower power consumption by batching transfers,
    using slower rates when possible, and consolidating traffic.
    """

    def __init__(self, power_cap: Optional[float] = None):
        """
        Initialize energy policy

        Args:
            power_cap: Maximum power budget in watts
        """
        super().__init__("EnergyPolicy")
        self.power_cap = power_cap
        self.energy_consumed = 0.0

        logger.info(f"EnergyPolicy: power_cap={power_cap}W" if power_cap else "EnergyPolicy: no cap")

    def evaluate(self, context: PolicyContext) -> PolicyAction:
        """Evaluate energy-focused policy"""
        with self.lock:
            # Energy model: Power ∝ rate^2 (simplified)
            # Lower rates are more energy efficient

            # Check if we can reduce rate without hurting performance
            if context.queue_depth == 0 and context.latency < 0.01:
                # No queuing and low latency, can reduce rate
                reduction = 0.2
                new_rate = context.current_rate * (1.0 - reduction)

                return PolicyAction(
                    decision=PolicyDecision.DECREASE_RATE,
                    flow_id=context.flow_id,
                    parameters={'new_rate': new_rate, 'reduction': reduction},
                    priority=0,
                    reason="Energy optimization: reduce rate to save power",
                    confidence=0.6
                )

            elif context.queue_depth > 10:
                # Queue building up, need to increase rate
                increase = 0.1
                new_rate = context.current_rate * (1.0 + increase)

                return PolicyAction(
                    decision=PolicyDecision.INCREASE_RATE,
                    flow_id=context.flow_id,
                    parameters={'new_rate': new_rate, 'increase': increase},
                    priority=1,
                    reason="Queue buildup requires rate increase",
                    confidence=0.7
                )

            else:
                return PolicyAction(
                    decision=PolicyDecision.MAINTAIN_RATE,
                    flow_id=context.flow_id,
                    parameters={'current_rate': context.current_rate},
                    priority=0,
                    reason="Energy-performance balance maintained",
                    confidence=0.7
                )


class AdaptivePolicy(Policy):
    """
    Adaptive policy that dynamically adjusts strategies

    Monitors network conditions and switches between different optimization
    strategies based on current state and performance.
    """

    def __init__(self, flow_collector: FlowMetricsCollector,
                 utilization_tracker: LinkUtilizationTracker,
                 bottleneck_detector: BottleneckDetector):
        """Initialize adaptive policy"""
        super().__init__("AdaptivePolicy")

        self.flow_collector = flow_collector
        self.utilization_tracker = utilization_tracker
        self.bottleneck_detector = bottleneck_detector

        # Component policies
        self.latency_policy = LatencyPolicy()
        self.throughput_policy = ThroughputPolicy()
        self.fairness_policy = FairnessPolicy()

        # Current objective
        self.current_objective = PolicyObjective.BALANCED

        # Learning parameters
        self.objective_scores: Dict[PolicyObjective, float] = {
            obj: 0.5 for obj in PolicyObjective
        }

        logger.info("AdaptivePolicy initialized with multi-objective optimization")

    def evaluate(self, context: PolicyContext) -> PolicyAction:
        """Evaluate adaptive policy"""
        with self.lock:
            # Detect current network state
            state = self._detect_network_state()

            # Choose appropriate policy based on state
            if state == 'congested':
                # Use latency policy during congestion
                action = self.latency_policy.evaluate(context)
                action.reason = f"Adaptive (Congestion): {action.reason}"

            elif state == 'underutilized':
                # Use throughput policy when underutilized
                action = self.throughput_policy.evaluate(context)
                action.reason = f"Adaptive (Underutilized): {action.reason}"

            elif state == 'unfair':
                # Use fairness policy when unfair
                action = self.fairness_policy.evaluate(context)
                action.reason = f"Adaptive (Unfair): {action.reason}"

            else:
                # Balanced state, use combination
                actions = [
                    self.latency_policy.evaluate(context),
                    self.throughput_policy.evaluate(context),
                    self.fairness_policy.evaluate(context)
                ]

                # Weight actions by priority and confidence
                action = self._combine_actions(actions, context)
                action.reason = f"Adaptive (Balanced): {action.reason}"

            return action

    def _detect_network_state(self) -> str:
        """Detect current network state"""
        # Check for bottlenecks
        bottlenecks = self.bottleneck_detector.detect_bottlenecks()
        if bottlenecks:
            return 'congested'

        # Check average utilization
        links = self.utilization_tracker.get_all_links()
        if links:
            avg_util = statistics.mean(l.utilization for l in links.values())
            if avg_util < 0.5:
                return 'underutilized'

        # Check fairness
        flows = self.flow_collector.get_all_flows()
        if len(flows) >= 2:
            rates = [m.throughput_bps for m in flows.values() if m.throughput_bps > 0]
            if rates:
                n = len(rates)
                sum_rates = sum(rates)
                sum_rates_squared = sum(r * r for r in rates)
                fairness = (sum_rates * sum_rates) / (n * sum_rates_squared) if sum_rates_squared > 0 else 1.0

                if fairness < 0.7:
                    return 'unfair'

        return 'balanced'

    def _combine_actions(self, actions: List[PolicyAction], context: PolicyContext) -> PolicyAction:
        """Combine multiple policy actions"""
        # Sort by priority and confidence
        actions.sort(key=lambda a: (a.priority, a.confidence), reverse=True)

        # Take highest priority action
        best_action = actions[0]

        # If multiple high-priority actions, average their parameters
        high_priority = [a for a in actions if a.priority == best_action.priority]
        if len(high_priority) > 1 and 'new_rate' in best_action.parameters:
            avg_rate = statistics.mean(
                a.parameters.get('new_rate', context.current_rate)
                for a in high_priority
            )
            best_action.parameters['new_rate'] = avg_rate
            best_action.reason = "Combined policy decision"

        return best_action


class PolicyEngine:
    """
    Main policy engine for flow control

    Manages multiple policies, handles policy composition and switching,
    and learns from feedback.
    """

    def __init__(self, flow_collector: FlowMetricsCollector,
                 utilization_tracker: LinkUtilizationTracker,
                 bottleneck_detector: BottleneckDetector):
        """Initialize policy engine"""
        self.flow_collector = flow_collector
        self.utilization_tracker = utilization_tracker
        self.bottleneck_detector = bottleneck_detector

        # Available policies
        self.policies: Dict[str, Policy] = {
            'latency': LatencyPolicy(),
            'throughput': ThroughputPolicy(),
            'fairness': FairnessPolicy(),
            'energy': EnergyPolicy(),
            'adaptive': AdaptivePolicy(flow_collector, utilization_tracker, bottleneck_detector)
        }

        # Current active policy
        self.active_policy: str = 'adaptive'

        # Policy chain (for composition)
        self.policy_chain: List[str] = ['adaptive']

        self.lock = threading.RLock()

        logger.info("PolicyEngine initialized with 5 policies")

    def set_active_policy(self, policy_name: str) -> None:
        """
        Set active policy

        Args:
            policy_name: Name of policy to activate
        """
        with self.lock:
            if policy_name not in self.policies:
                raise ValueError(f"Unknown policy: {policy_name}")

            self.active_policy = policy_name
            logger.info(f"Active policy changed to: {policy_name}")

    def set_policy_chain(self, policy_names: List[str]) -> None:
        """
        Set policy chain for composition

        Args:
            policy_names: List of policy names to chain
        """
        with self.lock:
            for name in policy_names:
                if name not in self.policies:
                    raise ValueError(f"Unknown policy: {name}")

            self.policy_chain = policy_names
            logger.info(f"Policy chain set to: {policy_names}")

    def make_decision(self, context: PolicyContext) -> PolicyAction:
        """
        Make policy decision for a flow

        Args:
            context: Context for decision

        Returns:
            PolicyAction to take
        """
        with self.lock:
            if len(self.policy_chain) == 1:
                # Single policy
                policy = self.policies[self.policy_chain[0]]
                return policy.evaluate(context)

            else:
                # Policy chain - combine decisions
                actions = []
                for policy_name in self.policy_chain:
                    policy = self.policies[policy_name]
                    action = policy.evaluate(context)
                    actions.append(action)

                # Combine actions
                return self._combine_chain_actions(actions, context)

    def _combine_chain_actions(self, actions: List[PolicyAction],
                               context: PolicyContext) -> PolicyAction:
        """Combine actions from policy chain"""
        # Sort by priority
        actions.sort(key=lambda a: a.priority, reverse=True)

        # Take highest priority
        best_action = actions[0]

        # Average rate adjustments from same priority
        same_priority = [a for a in actions if a.priority == best_action.priority]
        if len(same_priority) > 1 and 'new_rate' in best_action.parameters:
            avg_rate = statistics.mean(
                a.parameters.get('new_rate', context.current_rate)
                for a in same_priority
            )
            best_action.parameters['new_rate'] = avg_rate
            best_action.reason = f"Policy chain decision (priority={best_action.priority})"

        return best_action

    def provide_feedback(self, action: PolicyAction, outcome: dict) -> None:
        """
        Provide feedback on policy decision

        Args:
            action: Action that was taken
            outcome: Results of the action
        """
        with self.lock:
            # Provide feedback to all policies in chain
            for policy_name in self.policy_chain:
                policy = self.policies[policy_name]
                policy.learn_from_feedback(action, outcome)

    def get_policy_stats(self) -> Dict[str, dict]:
        """Get statistics for all policies"""
        with self.lock:
            stats = {}
            for name, policy in self.policies.items():
                stats[name] = {
                    'enabled': policy.enabled,
                    'decisions_made': policy.decisions_made,
                    'success_rate': policy.get_success_rate()
                }
            return stats


__all__ = [
    'Policy',
    'LatencyPolicy',
    'ThroughputPolicy',
    'FairnessPolicy',
    'EnergyPolicy',
    'AdaptivePolicy',
    'PolicyEngine',
    'PolicyObjective',
    'PolicyDecision',
    'PolicyContext',
    'PolicyAction',
]

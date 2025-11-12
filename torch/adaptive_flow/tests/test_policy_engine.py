"""
Tests for Policy Engine

Tests policy implementations, decision making, and policy composition.
"""

import unittest
from torch.testing._internal.common_utils import TestCase, run_tests

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from policy_engine import (
    LatencyPolicy,
    ThroughputPolicy,
    FairnessPolicy,
    EnergyPolicy,
    AdaptivePolicy,
    PolicyEngine,
    PolicyContext,
    PolicyDecision,
    PolicyObjective
)
from flow_monitor import (
    FlowMetricsCollector,
    LinkUtilizationTracker,
    BottleneckDetector
)


class TestPolicies(TestCase):
    """Test individual policy implementations"""

    def test_latency_policy_high_latency(self):
        """Test latency policy with high latency"""
        policy = LatencyPolicy(target_latency=0.001, tolerance=0.1)

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,  # 1 Gbps
            latency=0.005,  # 5ms (5x target)
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.5
        )

        action = policy.evaluate(context)
        self.assertEqual(action.decision, PolicyDecision.DECREASE_RATE)
        self.assertLess(action.parameters['new_rate'], context.current_rate)

    def test_latency_policy_low_latency(self):
        """Test latency policy with low latency"""
        policy = LatencyPolicy(target_latency=0.001, tolerance=0.1)

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.0005,  # 0.5ms (below target)
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.5,
            bottleneck_detected=False
        )

        action = policy.evaluate(context)
        self.assertEqual(action.decision, PolicyDecision.INCREASE_RATE)

    def test_throughput_policy_underutilized(self):
        """Test throughput policy with underutilized link"""
        policy = ThroughputPolicy(target_utilization=0.9)

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.5  # 50% utilization
        )

        action = policy.evaluate(context)
        self.assertEqual(action.decision, PolicyDecision.INCREASE_RATE)

    def test_throughput_policy_loss(self):
        """Test throughput policy with packet loss"""
        policy = ThroughputPolicy(target_utilization=0.9)

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.05,  # 5% loss
            queue_depth=0,
            link_utilization=0.8
        )

        action = policy.evaluate(context)
        self.assertEqual(action.decision, PolicyDecision.DECREASE_RATE)

    def test_fairness_policy_single_flow(self):
        """Test fairness policy with single flow"""
        policy = FairnessPolicy(fairness_threshold=0.7)

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.5
        )

        action = policy.evaluate(context)
        # Single flow, should maintain
        self.assertEqual(action.decision, PolicyDecision.MAINTAIN_RATE)

    def test_fairness_policy_unfair_allocation(self):
        """Test fairness policy with unfair allocation"""
        policy = FairnessPolicy(fairness_threshold=0.7)

        # Simulate multiple flows
        policy.flow_rates = {
            'flow1': 2e9,  # 2 Gbps (high)
            'flow2': 0.5e9,  # 0.5 Gbps (low)
            'flow3': 0.5e9,  # 0.5 Gbps (low)
        }

        context = PolicyContext(
            flow_id='flow1',  # The high-rate flow
            current_rate=2e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.8
        )

        action = policy.evaluate(context)
        # Should decrease high-rate flow
        self.assertEqual(action.decision, PolicyDecision.DECREASE_RATE)

    def test_energy_policy_idle(self):
        """Test energy policy when idle"""
        policy = EnergyPolicy()

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=0,  # No queue
            link_utilization=0.5
        )

        action = policy.evaluate(context)
        # Should reduce rate to save energy
        self.assertEqual(action.decision, PolicyDecision.DECREASE_RATE)

    def test_energy_policy_queuing(self):
        """Test energy policy with queuing"""
        policy = EnergyPolicy()

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=15,  # Queue building up
            link_utilization=0.5
        )

        action = policy.evaluate(context)
        # Should increase rate to drain queue
        self.assertEqual(action.decision, PolicyDecision.INCREASE_RATE)


class TestAdaptivePolicy(TestCase):
    """Test adaptive policy"""

    def setUp(self):
        self.collector = FlowMetricsCollector()
        self.tracker = LinkUtilizationTracker()
        self.detector = BottleneckDetector(self.tracker, self.collector)

    def test_adaptive_policy_initialization(self):
        """Test adaptive policy initialization"""
        policy = AdaptivePolicy(self.collector, self.tracker, self.detector)

        self.assertEqual(policy.current_objective, PolicyObjective.BALANCED)
        self.assertIsNotNone(policy.latency_policy)
        self.assertIsNotNone(policy.throughput_policy)
        self.assertIsNotNone(policy.fairness_policy)

    def test_adaptive_state_detection(self):
        """Test network state detection"""
        policy = AdaptivePolicy(self.collector, self.tracker, self.detector)

        # Setup balanced state
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')
        self.tracker.register_link('link1', 'cpu', 'cuda:0', 10e9)

        state = policy._detect_network_state()
        self.assertIsInstance(state, str)

    def test_adaptive_decision_making(self):
        """Test adaptive policy decision making"""
        policy = AdaptivePolicy(self.collector, self.tracker, self.detector)

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.5
        )

        action = policy.evaluate(context)
        self.assertIsNotNone(action)
        self.assertIn('Adaptive', action.reason)


class TestPolicyEngine(TestCase):
    """Test policy engine"""

    def setUp(self):
        self.collector = FlowMetricsCollector()
        self.tracker = LinkUtilizationTracker()
        self.detector = BottleneckDetector(self.tracker, self.collector)
        self.engine = PolicyEngine(self.collector, self.tracker, self.detector)

    def test_policy_engine_initialization(self):
        """Test policy engine initialization"""
        self.assertEqual(self.engine.active_policy, 'adaptive')
        self.assertIn('latency', self.engine.policies)
        self.assertIn('throughput', self.engine.policies)
        self.assertIn('fairness', self.engine.policies)
        self.assertIn('adaptive', self.engine.policies)

    def test_set_active_policy(self):
        """Test setting active policy"""
        self.engine.set_active_policy('latency')
        self.assertEqual(self.engine.active_policy, 'latency')

        # Invalid policy
        with self.assertRaises(ValueError):
            self.engine.set_active_policy('invalid')

    def test_policy_chain(self):
        """Test policy chain"""
        self.engine.set_policy_chain(['latency', 'fairness'])
        self.assertEqual(len(self.engine.policy_chain), 2)

        # Invalid policy in chain
        with self.assertRaises(ValueError):
            self.engine.set_policy_chain(['latency', 'invalid'])

    def test_decision_making(self):
        """Test decision making"""
        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.5
        )

        action = self.engine.make_decision(context)
        self.assertIsNotNone(action)
        self.assertIsInstance(action.decision, PolicyDecision)

    def test_feedback_learning(self):
        """Test feedback learning"""
        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.5
        )

        action = self.engine.make_decision(context)

        # Provide positive feedback
        outcome = {'success': True, 'latency_improved': True}
        self.engine.provide_feedback(action, outcome)

        # Check stats
        stats = self.engine.get_policy_stats()
        self.assertIn('adaptive', stats)
        self.assertGreater(stats['adaptive']['decisions_made'], 0)

    def test_policy_chain_combination(self):
        """Test combining decisions from policy chain"""
        self.engine.set_policy_chain(['latency', 'throughput'])

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.5
        )

        action = self.engine.make_decision(context)
        self.assertIsNotNone(action)

    def test_policy_statistics(self):
        """Test policy statistics"""
        # Make some decisions
        for i in range(5):
            context = PolicyContext(
                flow_id=f'flow{i}',
                current_rate=1e9,
                latency=0.001,
                loss_rate=0.0,
                queue_depth=0,
                link_utilization=0.5
            )
            self.engine.make_decision(context)

        stats = self.engine.get_policy_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('adaptive', stats)


if __name__ == '__main__':
    run_tests()

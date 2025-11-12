"""
Integration Tests for Adaptive Flow Control

End-to-end tests of the complete system including configuration,
monitoring, policies, and PyTorch integration.
"""

import unittest
import time
from torch.testing._internal.common_utils import TestCase, run_tests

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import AdaptiveFlowConfig, ConfigPresets
from flow_monitor import FlowMetricsCollector, LinkUtilizationTracker, BottleneckDetector, PerformanceAnalyzer
from policy_engine import PolicyEngine, PolicyContext
from advanced_congestion import create_controller


class TestConfigurationIntegration(TestCase):
    """Test configuration integration"""

    def test_config_presets(self):
        """Test configuration presets"""
        low_latency = ConfigPresets.low_latency()
        self.assertEqual(low_latency.policy, 'latency')
        self.assertEqual(low_latency.target, 'latency')

        high_throughput = ConfigPresets.high_throughput()
        self.assertEqual(high_throughput.policy, 'throughput')

        fair_sharing = ConfigPresets.fair_sharing()
        self.assertEqual(fair_sharing.policy, 'fairness')

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = AdaptiveFlowConfig(
            enabled=True,
            policy='latency',
            target='latency'
        )

        # Invalid policy
        with self.assertRaises(ValueError):
            AdaptiveFlowConfig(policy='invalid')

        # Invalid target
        with self.assertRaises(ValueError):
            AdaptiveFlowConfig(target='invalid')

    def test_config_serialization(self):
        """Test configuration serialization"""
        config = ConfigPresets.balanced()

        # To dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['policy'], 'adaptive')

        # To JSON
        config_json = config.to_json()
        self.assertIsInstance(config_json, str)
        self.assertIn('adaptive', config_json)

        # From dict
        config2 = AdaptiveFlowConfig.from_dict(config_dict)
        self.assertEqual(config2.policy, config.policy)


class TestEndToEndScenarios(TestCase):
    """Test end-to-end scenarios"""

    def setUp(self):
        """Set up monitoring and policy components"""
        self.collector = FlowMetricsCollector()
        self.tracker = LinkUtilizationTracker()
        self.detector = BottleneckDetector(self.tracker, self.collector)
        self.analyzer = PerformanceAnalyzer(
            self.collector, self.tracker, self.detector
        )
        self.engine = PolicyEngine(
            self.collector, self.tracker, self.detector
        )

    def test_single_flow_scenario(self):
        """Test single flow transfer"""
        # Register flow and link
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')
        self.tracker.register_link('link1', 'cpu', 'cuda:0', 10e9)

        # Simulate transfers
        for i in range(10):
            self.collector.record_transfer(
                flow_id='flow1',
                bytes_sent=1024*1024,
                latency=0.001,
                success=True
            )
            self.tracker.record_transmission(
                link_id='link1',
                bytes_transmitted=1024*1024,
                flow_id='flow1'
            )

        # Check metrics
        metrics = self.collector.get_flow_metrics('flow1')
        self.assertEqual(metrics.transfers_completed, 10)
        self.assertGreater(metrics.throughput_bps, 0)

        # Analyze performance
        analysis = self.analyzer.analyze_performance()
        self.assertGreater(analysis['fairness_index'], 0)

    def test_multi_flow_scenario(self):
        """Test multiple concurrent flows"""
        # Register multiple flows
        for i in range(3):
            flow_id = f'flow{i}'
            device = f'cuda:{i}'
            self.collector.register_flow(flow_id, 'cpu', device)
            self.tracker.register_link(
                f'link{i}', 'cpu', device, 10e9
            )

        # Simulate concurrent transfers
        for iteration in range(10):
            for i in range(3):
                flow_id = f'flow{i}'
                link_id = f'link{i}'

                self.collector.record_transfer(
                    flow_id=flow_id,
                    bytes_sent=1024*1024,
                    latency=0.001 + i * 0.0005,  # Varying latency
                    success=True
                )
                self.tracker.record_transmission(
                    link_id=link_id,
                    bytes_transmitted=1024*1024,
                    flow_id=flow_id
                )

        # Check fairness
        fairness = self.analyzer.compute_jains_fairness_index()
        self.assertGreater(fairness, 0)

        # All flows should have metrics
        summary = self.collector.get_summary()
        self.assertEqual(summary['total_flows'], 3)

    def test_congestion_scenario(self):
        """Test congestion detection and response"""
        # Setup link and flow
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')
        self.tracker.register_link('link1', 'cpu', 'cuda:0', 1e9)  # 1 Gbps

        # Simulate high utilization
        link = self.tracker.links['link1']
        link.utilization = 0.95
        link.active_flows.add('flow1')

        # Detect bottlenecks
        bottlenecks = self.detector.detect_bottlenecks()
        self.assertEqual(len(bottlenecks), 1)
        self.assertEqual(bottlenecks[0].severity, 'critical')

        # Policy should react to congestion
        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.010,  # High latency
            loss_rate=0.0,
            queue_depth=10,
            link_utilization=0.95,
            bottleneck_detected=True
        )

        action = self.engine.make_decision(context)
        self.assertIsNotNone(action)

    def test_loss_scenario(self):
        """Test packet loss scenario"""
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')

        # Successful transfers
        for i in range(7):
            self.collector.record_transfer(
                flow_id='flow1',
                bytes_sent=1024*1024,
                latency=0.001,
                success=True
            )

        # Failed transfers
        for i in range(3):
            self.collector.record_transfer(
                flow_id='flow1',
                bytes_sent=1024*1024,
                latency=0.0,
                success=False
            )

        metrics = self.collector.get_flow_metrics('flow1')
        self.assertGreater(metrics.loss_rate, 0)

        # Analysis should detect issue
        analysis = self.analyzer.analyze_performance()
        issues = analysis['issues']
        loss_issues = [i for i in issues if i['type'] == 'packet_loss']
        self.assertGreater(len(loss_issues), 0)

    def test_policy_switching(self):
        """Test runtime policy switching"""
        # Start with latency policy
        self.engine.set_active_policy('latency')
        self.assertEqual(self.engine.active_policy, 'latency')

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.5
        )

        action1 = self.engine.make_decision(context)

        # Switch to throughput policy
        self.engine.set_active_policy('throughput')
        action2 = self.engine.make_decision(context)

        # Both should produce valid actions
        self.assertIsNotNone(action1)
        self.assertIsNotNone(action2)

    def test_congestion_control_integration(self):
        """Test congestion control integration"""
        # Create controller
        controller = create_controller('bbr')

        # Simulate data transfer
        for i in range(10):
            controller.on_ack(bytes_acked=1500, rtt=0.001)

        # Should have positive metrics
        self.assertGreater(controller.metrics.delivered, 0)
        self.assertGreater(controller.get_pacing_rate(), 0)

        # Integrate with flow collector
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')
        for i in range(10):
            self.collector.record_transfer(
                flow_id='flow1',
                bytes_sent=1500,
                latency=controller.metrics.current_rtt,
                success=True
            )

        metrics = self.collector.get_flow_metrics('flow1')
        self.assertGreater(metrics.transfers_completed, 0)


class TestPerformanceRegression(TestCase):
    """Test for performance regressions"""

    def test_monitoring_overhead(self):
        """Test monitoring overhead is acceptable"""
        collector = FlowMetricsCollector()
        collector.register_flow('flow1', 'cpu', 'cuda:0')

        # Time many recordings
        start = time.time()
        for i in range(1000):
            collector.record_transfer(
                flow_id='flow1',
                bytes_sent=1024,
                latency=0.001,
                success=True
            )
        elapsed = time.time() - start

        # Should be fast (< 10ms for 1000 recordings)
        self.assertLess(elapsed, 0.01)

    def test_policy_decision_latency(self):
        """Test policy decision latency"""
        collector = FlowMetricsCollector()
        tracker = LinkUtilizationTracker()
        detector = BottleneckDetector(tracker, collector)
        engine = PolicyEngine(collector, tracker, detector)

        context = PolicyContext(
            flow_id='flow1',
            current_rate=1e9,
            latency=0.001,
            loss_rate=0.0,
            queue_depth=0,
            link_utilization=0.5
        )

        # Time many decisions
        start = time.time()
        for i in range(1000):
            engine.make_decision(context)
        elapsed = time.time() - start

        # Should be fast (< 50ms for 1000 decisions)
        self.assertLess(elapsed, 0.05)


if __name__ == '__main__':
    run_tests()

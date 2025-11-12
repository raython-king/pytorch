"""
Tests for Flow Monitoring and Performance Analysis

Tests FlowMetricsCollector, LinkUtilizationTracker, BottleneckDetector,
and PerformanceAnalyzer.
"""

import unittest
import time
from torch.testing._internal.common_utils import TestCase, run_tests

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flow_monitor import (
    FlowMetricsCollector,
    LinkUtilizationTracker,
    BottleneckDetector,
    PerformanceAnalyzer,
    FlowMetrics,
    LinkMetrics
)


class TestFlowMetricsCollector(TestCase):
    """Test flow metrics collection"""

    def setUp(self):
        self.collector = FlowMetricsCollector(history_size=100)

    def test_flow_registration(self):
        """Test flow registration"""
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')

        flows = self.collector.get_all_flows()
        self.assertIn('flow1', flows)
        self.assertEqual(flows['flow1'].src_device, 'cpu')
        self.assertEqual(flows['flow1'].dst_device, 'cuda:0')

    def test_transfer_recording(self):
        """Test transfer recording"""
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')

        self.collector.record_transfer(
            flow_id='flow1',
            bytes_sent=1024,
            latency=0.001,
            success=True
        )

        metrics = self.collector.get_flow_metrics('flow1')
        self.assertEqual(metrics.bytes_sent, 1024)
        self.assertEqual(metrics.transfers_completed, 1)
        self.assertEqual(metrics.transfers_failed, 0)

    def test_latency_percentiles(self):
        """Test latency percentile computation"""
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')

        # Record transfers with varying latencies
        latencies = [0.001, 0.002, 0.003, 0.004, 0.005,
                    0.006, 0.007, 0.008, 0.009, 0.010]

        for lat in latencies:
            self.collector.record_transfer(
                flow_id='flow1',
                bytes_sent=1024,
                latency=lat,
                success=True
            )

        metrics = self.collector.get_flow_metrics('flow1')
        self.assertGreater(metrics.latency_p50, 0)
        self.assertGreater(metrics.latency_p95, 0)
        self.assertGreater(metrics.latency_p99, 0)
        self.assertGreater(metrics.latency_p95, metrics.latency_p50)

    def test_loss_tracking(self):
        """Test packet loss tracking"""
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')

        # Successful transfers
        for i in range(8):
            self.collector.record_transfer(
                flow_id='flow1',
                bytes_sent=1024,
                latency=0.001,
                success=True
            )

        # Failed transfers
        for i in range(2):
            self.collector.record_transfer(
                flow_id='flow1',
                bytes_sent=1024,
                latency=0.0,
                success=False
            )

        metrics = self.collector.get_flow_metrics('flow1')
        self.assertEqual(metrics.transfers_completed, 8)
        self.assertEqual(metrics.transfers_failed, 2)
        self.assertGreater(metrics.loss_rate, 0)

    def test_throughput_computation(self):
        """Test throughput computation"""
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')

        # Record transfers
        for i in range(10):
            self.collector.record_transfer(
                flow_id='flow1',
                bytes_sent=1024*1024,  # 1 MB
                latency=0.001,
                success=True
            )
            time.sleep(0.01)  # Small delay

        metrics = self.collector.get_flow_metrics('flow1')
        self.assertGreater(metrics.throughput_bps, 0)

    def test_summary_statistics(self):
        """Test summary statistics"""
        # Register multiple flows
        for i in range(3):
            flow_id = f'flow{i}'
            self.collector.register_flow(flow_id, 'cpu', f'cuda:{i}')

            for j in range(5):
                self.collector.record_transfer(
                    flow_id=flow_id,
                    bytes_sent=1024,
                    latency=0.001,
                    success=True
                )

        summary = self.collector.get_summary()
        self.assertEqual(summary['total_flows'], 3)
        self.assertGreater(summary['total_bytes_sent'], 0)


class TestLinkUtilizationTracker(TestCase):
    """Test link utilization tracking"""

    def setUp(self):
        self.tracker = LinkUtilizationTracker(update_interval=0.1)

    def test_link_registration(self):
        """Test link registration"""
        capacity_bps = 10e9  # 10 Gbps
        self.tracker.register_link('link1', 'cpu', 'cuda:0', capacity_bps)

        links = self.tracker.get_all_links()
        self.assertIn('link1', links)
        self.assertEqual(links['link1'].capacity_bps, capacity_bps)

    def test_transmission_recording(self):
        """Test transmission recording"""
        self.tracker.register_link('link1', 'cpu', 'cuda:0', 10e9)

        self.tracker.record_transmission('link1', 1024*1024, 'flow1')

        links = self.tracker.get_all_links()
        self.assertIn('flow1', links['link1'].active_flows)

    def test_utilization_computation(self):
        """Test utilization computation"""
        capacity_bps = 10e9  # 10 Gbps
        self.tracker.register_link('link1', 'cpu', 'cuda:0', capacity_bps)

        # Send data for interval
        bytes_to_send = int(capacity_bps / 8 * 0.15)  # 15% utilization over interval
        self.tracker.record_transmission('link1', bytes_to_send, 'flow1')

        time.sleep(0.15)  # Wait for update interval

        # Trigger update
        self.tracker.record_transmission('link1', 100, 'flow1')

        utilization = self.tracker.get_link_utilization('link1')
        self.assertGreater(utilization, 0)

    def test_congested_links_detection(self):
        """Test congested links detection"""
        self.tracker.register_link('link1', 'cpu', 'cuda:0', 10e9)
        self.tracker.register_link('link2', 'cpu', 'cuda:1', 10e9)

        # Make link1 congested
        link1 = self.tracker.links['link1']
        link1.utilization = 0.85
        link1.last_update = time.time()

        # Make link2 normal
        link2 = self.tracker.links['link2']
        link2.utilization = 0.50
        link2.last_update = time.time()

        congested = self.tracker.get_congested_links(threshold=0.8)
        self.assertEqual(len(congested), 1)
        self.assertEqual(congested[0].link_id, 'link1')


class TestBottleneckDetector(TestCase):
    """Test bottleneck detection"""

    def setUp(self):
        self.collector = FlowMetricsCollector()
        self.tracker = LinkUtilizationTracker()
        self.detector = BottleneckDetector(self.tracker, self.collector)

    def test_bottleneck_detection(self):
        """Test bottleneck detection"""
        # Setup link with high utilization
        self.tracker.register_link('link1', 'cpu', 'cuda:0', 10e9)
        link = self.tracker.links['link1']
        link.utilization = 0.95  # Critical utilization
        link.active_flows.add('flow1')
        link.active_flows.add('flow2')

        # Detect bottlenecks
        bottlenecks = self.detector.detect_bottlenecks()

        self.assertEqual(len(bottlenecks), 1)
        self.assertEqual(bottlenecks[0].link_id, 'link1')
        self.assertEqual(bottlenecks[0].severity, 'critical')
        self.assertEqual(len(bottlenecks[0].affected_flows), 2)

    def test_worst_bottleneck(self):
        """Test worst bottleneck identification"""
        # Setup multiple links
        self.tracker.register_link('link1', 'cpu', 'cuda:0', 10e9)
        self.tracker.register_link('link2', 'cpu', 'cuda:1', 10e9)

        link1 = self.tracker.links['link1']
        link1.utilization = 0.85
        link1.active_flows.add('flow1')

        link2 = self.tracker.links['link2']
        link2.utilization = 0.95
        link2.active_flows.add('flow2')

        # Detect bottlenecks
        self.detector.detect_bottlenecks()

        worst = self.detector.get_worst_bottleneck()
        self.assertIsNotNone(worst)
        self.assertEqual(worst.link_id, 'link2')


class TestPerformanceAnalyzer(TestCase):
    """Test performance analysis"""

    def setUp(self):
        self.collector = FlowMetricsCollector()
        self.tracker = LinkUtilizationTracker()
        self.detector = BottleneckDetector(self.tracker, self.collector)
        self.analyzer = PerformanceAnalyzer(
            self.collector, self.tracker, self.detector
        )

    def test_fairness_computation(self):
        """Test Jain's fairness index computation"""
        # Register flows with equal throughput (perfect fairness)
        for i in range(3):
            flow_id = f'flow{i}'
            self.collector.register_flow(flow_id, 'cpu', f'cuda:{i}')
            metrics = self.collector.flows[flow_id]
            metrics.throughput_bps = 1e9  # 1 Gbps each

        fairness = self.analyzer.compute_jains_fairness_index()
        self.assertAlmostEqual(fairness, 1.0, places=2)

    def test_performance_analysis(self):
        """Test comprehensive performance analysis"""
        # Setup some flows
        for i in range(2):
            flow_id = f'flow{i}'
            self.collector.register_flow(flow_id, 'cpu', f'cuda:{i}')
            self.collector.record_transfer(
                flow_id=flow_id,
                bytes_sent=1024*1024,
                latency=0.001,
                success=True
            )

        # Setup links
        self.tracker.register_link('link1', 'cpu', 'cuda:0', 10e9)

        analysis = self.analyzer.analyze_performance()

        self.assertIn('timestamp', analysis)
        self.assertIn('flow_summary', analysis)
        self.assertIn('fairness_index', analysis)
        self.assertIn('issues', analysis)
        self.assertIn('recommendations', analysis)

    def test_export_metrics(self):
        """Test metrics export"""
        # Setup minimal data
        self.collector.register_flow('flow1', 'cpu', 'cuda:0')
        self.tracker.register_link('link1', 'cpu', 'cuda:0', 10e9)

        # Export as JSON
        json_export = self.analyzer.export_metrics(format='json')
        self.assertIsInstance(json_export, str)
        self.assertIn('fairness_index', json_export)

        # Export as CSV
        csv_export = self.analyzer.export_metrics(format='csv')
        self.assertIsInstance(csv_export, str)
        self.assertIn('metric,value', csv_export)


if __name__ == '__main__':
    run_tests()

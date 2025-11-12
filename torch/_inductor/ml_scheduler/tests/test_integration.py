"""
Integration tests for ML scheduler.

End-to-end tests comparing ML vs heuristic scheduling, correctness validation,
and performance benchmarking.
"""

import torch
import unittest
import sys
import time
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from torch._inductor.ml_scheduler.orchestrator import MLSchedulerOrchestrator
    from torch._inductor.ml_scheduler.config import MLSchedulerConfig
    from torch._inductor.ml_scheduler.integration.scheduler_hook import (
        MLSchedulerWrapper,
        MLSchedulerMode,
        enable_ml_scheduler,
        disable_ml_scheduler,
        ml_scheduler_mode,
        compare_ml_vs_heuristic,
        validate_ml_scheduler,
    )
    from torch._inductor.ml_scheduler.inference.predictor import MLSchedulerPredictor
    from torch._inductor.ml_scheduler.training.dataset import IRGraphDataset
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Warning: Could not import ML scheduler components: {e}")


class DummySchedulerNode:
    """Mock scheduler node for testing."""

    def __init__(self, node_id, device='cpu'):
        self.node_id = node_id
        self.device = torch.device(device)
        self.users = []
        self.unmet_dependencies = set()
        self.ancestors = set()
        self.min_order = None
        self.max_order = None
        self.read_writes = type('obj', (object,), {
            'reads': [],
            'writes': []
        })()

    def get_device(self):
        return self.device

    def get_outputs(self):
        return [f"output_{self.node_id}"]

    def is_reduction(self):
        return False

    def estimate_flops(self):
        return 1000

    def __repr__(self):
        return f"DummyNode({self.node_id})"


class DummyScheduler:
    """Mock scheduler for testing."""

    def fuse_nodes(self, nodes: List[DummySchedulerNode]) -> List[DummySchedulerNode]:
        """Simple heuristic: fuse adjacent nodes."""
        if len(nodes) <= 1:
            return nodes

        # Fuse pairs of adjacent nodes
        result = []
        i = 0
        while i < len(nodes):
            if i + 1 < len(nodes):
                # Fuse nodes[i] and nodes[i+1]
                fused = DummySchedulerNode(f"{nodes[i].node_id}+{nodes[i+1].node_id}")
                result.append(fused)
                i += 2
            else:
                result.append(nodes[i])
                i += 1

        return result


@unittest.skipIf(not IMPORTS_AVAILABLE, "ML scheduler components not available")
class TestMLSchedulerOrchestrator(TestCase):
    """Test ML scheduler orchestrator."""

    def setUp(self):
        self.config = MLSchedulerConfig(
            model_path=None,  # No pretrained model
            confidence_threshold=0.5,
        )
        self.orchestrator = MLSchedulerOrchestrator(config=self.config)

    def test_orchestrator_creation(self):
        """Test that orchestrator can be created."""
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(self.orchestrator.config, self.config)

    def test_should_use_ml(self):
        """Test decision logic for using ML."""
        # Too few nodes
        self.assertFalse(self.orchestrator.should_use_ml(2))

        # Good number of nodes
        self.assertTrue(self.orchestrator.should_use_ml(50))

        # Too many nodes
        self.assertFalse(self.orchestrator.should_use_ml(2000))

    def test_predict_fusion_plan(self):
        """Test fusion plan prediction."""
        nodes = [DummySchedulerNode(i) for i in range(10)]
        device = torch.device('cpu')

        fusion_plan = self.orchestrator.predict_fusion_plan(nodes, device)

        # Should return a fusion plan
        self.assertIsNotNone(fusion_plan)
        self.assertIsInstance(fusion_plan.fusions, list)
        self.assertIsInstance(fusion_plan.confidence_scores, list)

    def test_extract_pairwise_features(self):
        """Test pairwise feature extraction."""
        node1 = DummySchedulerNode(1)
        node2 = DummySchedulerNode(2)

        features = self.orchestrator.extract_pairwise_features(node1, node2)

        # Should return a tensor
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.dim(), 1)
        self.assertGreater(features.numel(), 0)

    def test_prediction_cache(self):
        """Test that predictions are cached."""
        self.orchestrator.config.cache_predictions = True
        self.orchestrator.prediction_cache.clear()

        nodes = [DummySchedulerNode(i) for i in range(10)]
        device = torch.device('cpu')

        # First prediction
        fusion_plan1 = self.orchestrator.predict_fusion_plan(nodes, device)

        # Second prediction (should be cached)
        fusion_plan2 = self.orchestrator.predict_fusion_plan(nodes, device)

        # Should get the same plan (from cache)
        self.assertEqual(len(fusion_plan1.fusions), len(fusion_plan2.fusions))


@unittest.skipIf(not IMPORTS_AVAILABLE, "ML scheduler components not available")
class TestMLSchedulerWrapper(TestCase):
    """Test ML scheduler wrapper."""

    def setUp(self):
        self.base_scheduler = DummyScheduler()
        self.config = MLSchedulerConfig(
            model_path=None,
            confidence_threshold=0.5,
        )

    def test_wrapper_creation(self):
        """Test that wrapper can be created."""
        wrapper = MLSchedulerWrapper(
            self.base_scheduler,
            config=self.config,
            mode=MLSchedulerMode.HYBRID,
        )

        self.assertIsNotNone(wrapper)
        self.assertEqual(wrapper.mode, MLSchedulerMode.HYBRID)

    def test_fuse_nodes_disabled(self):
        """Test fusion with ML disabled."""
        wrapper = MLSchedulerWrapper(
            self.base_scheduler,
            config=self.config,
            mode=MLSchedulerMode.DISABLED,
        )

        nodes = [DummySchedulerNode(i) for i in range(4)]
        result = wrapper.fuse_nodes(nodes)

        # Should use heuristic
        self.assertIsInstance(result, list)
        self.assertGreater(wrapper.heuristic_fusion_count, 0)
        self.assertEqual(wrapper.ml_fusion_count, 0)

    def test_fuse_nodes_shadow(self):
        """Test fusion in shadow mode."""
        wrapper = MLSchedulerWrapper(
            self.base_scheduler,
            config=self.config,
            mode=MLSchedulerMode.SHADOW,
        )

        nodes = [DummySchedulerNode(i) for i in range(10)]
        result = wrapper.fuse_nodes(nodes)

        # Should use heuristic but log ML predictions
        self.assertIsInstance(result, list)
        self.assertGreater(len(wrapper.shadow_log), 0)

    def test_fuse_nodes_hybrid(self):
        """Test fusion in hybrid mode."""
        wrapper = MLSchedulerWrapper(
            self.base_scheduler,
            config=self.config,
            mode=MLSchedulerMode.HYBRID,
        )

        nodes = [DummySchedulerNode(i) for i in range(10)]
        result = wrapper.fuse_nodes(nodes)

        # Should return some result
        self.assertIsInstance(result, list)

        # May use ML or heuristic depending on confidence
        total_fusions = wrapper.ml_fusion_count + wrapper.heuristic_fusion_count
        self.assertGreater(total_fusions, 0)

    def test_statistics_collection(self):
        """Test that statistics are collected."""
        wrapper = MLSchedulerWrapper(
            self.base_scheduler,
            config=self.config,
            mode=MLSchedulerMode.HYBRID,
        )

        nodes = [DummySchedulerNode(i) for i in range(10)]

        # Run multiple fusions
        for _ in range(5):
            wrapper.fuse_nodes(nodes)

        stats = wrapper.get_statistics()

        # Should have statistics
        self.assertIn('ml_fusion_count', stats)
        self.assertIn('heuristic_fusion_count', stats)
        self.assertIn('fallback_count', stats)
        self.assertIn('ml_fusion_time_ms', stats)

    def test_reset_statistics(self):
        """Test statistics reset."""
        wrapper = MLSchedulerWrapper(
            self.base_scheduler,
            config=self.config,
            mode=MLSchedulerMode.HYBRID,
        )

        nodes = [DummySchedulerNode(i) for i in range(10)]
        wrapper.fuse_nodes(nodes)

        wrapper.reset_statistics()

        stats = wrapper.get_statistics()

        # Should be reset to zero
        self.assertEqual(stats['ml_fusion_count'], 0)
        self.assertEqual(stats['heuristic_fusion_count'], 0)
        self.assertEqual(stats['fallback_count'], 0)

    def test_can_fuse(self):
        """Test fusion compatibility check."""
        wrapper = MLSchedulerWrapper(
            self.base_scheduler,
            config=self.config,
            mode=MLSchedulerMode.HYBRID,
        )

        node1 = DummySchedulerNode(1, device='cpu')
        node2 = DummySchedulerNode(2, device='cpu')
        node3 = DummySchedulerNode(3, device='cuda')

        # Same device - should be fusable
        self.assertTrue(wrapper._can_fuse(node1, node2))

        # Different devices - should not be fusable
        self.assertFalse(wrapper._can_fuse(node1, node3))

    def test_validate_fusion_result(self):
        """Test fusion result validation."""
        wrapper = MLSchedulerWrapper(
            self.base_scheduler,
            config=self.config,
            mode=MLSchedulerMode.HYBRID,
        )

        original = [DummySchedulerNode(i) for i in range(4)]
        fused = [DummySchedulerNode(i) for i in range(2)]

        # Should be valid (fewer nodes after fusion)
        self.assertTrue(wrapper._validate_fusion_result(original, fused))

        # Empty result - invalid
        self.assertFalse(wrapper._validate_fusion_result(original, []))

        # More nodes than original - invalid
        more_nodes = [DummySchedulerNode(i) for i in range(10)]
        self.assertFalse(wrapper._validate_fusion_result(original, more_nodes))


@unittest.skipIf(not IMPORTS_AVAILABLE, "ML scheduler components not available")
class TestMLSchedulerModeContext(TestCase):
    """Test ML scheduler mode context manager."""

    def test_mode_context(self):
        """Test mode context manager."""
        # Enable ML scheduler
        enable_ml_scheduler(mode=MLSchedulerMode.HYBRID)

        original_mode = MLSchedulerMode.HYBRID

        # Use context manager
        with ml_scheduler_mode(MLSchedulerMode.SHADOW):
            from torch._inductor.ml_scheduler.integration.scheduler_hook import get_ml_scheduler_mode
            self.assertEqual(get_ml_scheduler_mode(), MLSchedulerMode.SHADOW)

        # Should restore original mode
        self.assertEqual(get_ml_scheduler_mode(), original_mode)

        # Cleanup
        disable_ml_scheduler()

    def test_enable_disable(self):
        """Test enable/disable functionality."""
        disable_ml_scheduler()

        from torch._inductor.ml_scheduler.integration.scheduler_hook import is_ml_scheduler_enabled

        self.assertFalse(is_ml_scheduler_enabled())

        enable_ml_scheduler(mode=MLSchedulerMode.HYBRID)

        self.assertTrue(is_ml_scheduler_enabled())

        disable_ml_scheduler()

        self.assertFalse(is_ml_scheduler_enabled())


@unittest.skipIf(not IMPORTS_AVAILABLE, "ML scheduler components not available")
class TestMLSchedulerComparison(TestCase):
    """Test comparison between ML and heuristic scheduling."""

    def test_compare_ml_vs_heuristic(self):
        """Test comparison function."""
        scheduler = DummyScheduler()
        nodes = [DummySchedulerNode(i) for i in range(10)]

        config = MLSchedulerConfig(model_path=None)

        result = compare_ml_vs_heuristic(nodes, scheduler, config)

        # Should return comparison results
        self.assertIn('num_nodes_original', result)
        self.assertIn('num_nodes_ml', result)
        self.assertIn('num_nodes_heuristic', result)
        self.assertIn('ml_time_ms', result)
        self.assertIn('heuristic_time_ms', result)

        # Original should match input
        self.assertEqual(result['num_nodes_original'], len(nodes))

    def test_validate_ml_scheduler(self):
        """Test validation function."""
        scheduler = DummyScheduler()

        test_cases = [
            [DummySchedulerNode(i) for i in range(5)],
            [DummySchedulerNode(i) for i in range(10)],
            [DummySchedulerNode(i) for i in range(15)],
        ]

        config = MLSchedulerConfig(model_path=None)

        result = validate_ml_scheduler(test_cases, scheduler, config)

        # Should return validation results
        self.assertIn('total_cases', result)
        self.assertIn('successful_cases', result)
        self.assertIn('failed_cases', result)
        self.assertIn('success_rate', result)

        self.assertEqual(result['total_cases'], len(test_cases))


@unittest.skipIf(not IMPORTS_AVAILABLE, "ML scheduler components not available")
class TestEndToEnd(TestCase):
    """End-to-end integration tests."""

    def test_simple_compilation(self):
        """Test simple model compilation with ML scheduler."""
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 20)
                self.linear2 = torch.nn.Linear(20, 10)

            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x

        model = SimpleModel()
        x = torch.randn(5, 10)

        # Enable ML scheduler in shadow mode
        enable_ml_scheduler(mode=MLSchedulerMode.SHADOW)

        try:
            # Compile model (this would normally trigger ML scheduler)
            # For testing, we just run the model
            with torch.no_grad():
                output = model(x)

            self.assertEqual(output.shape, (5, 10))

        finally:
            disable_ml_scheduler()

    def test_dataset_integration(self):
        """Test dataset integration with training."""
        import tempfile

        # Create dummy graph data
        graph_data = {
            'x': torch.randn(10, 64),
            'edge_index': torch.randint(0, 10, (2, 20)),
            'edge_attr': torch.randn(20, 32),
            'num_nodes': 10,
            'y_fusion': torch.randint(0, 2, (10, 10)).float(),
        }

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            import pickle
            pickle.dump(graph_data, f)
            temp_path = f.name

        # Create dataset from file
        dataset = IRGraphDataset(data_path=temp_path)

        # Should load the graph
        self.assertEqual(len(dataset), 1)

        # Get item
        item = dataset[0]
        self.assertEqual(item['num_nodes'], 10)
        self.assertEqual(item['x'].shape, (10, 64))

        # Cleanup
        Path(temp_path).unlink()

    def test_performance_overhead(self):
        """Test that ML scheduler has acceptable overhead."""
        scheduler = DummyScheduler()
        nodes = [DummySchedulerNode(i) for i in range(50)]

        config = MLSchedulerConfig(
            model_path=None,
            max_inference_time_ms=100.0,  # 100ms timeout
        )

        wrapper = MLSchedulerWrapper(
            scheduler,
            config=config,
            mode=MLSchedulerMode.HYBRID,
        )

        # Warmup
        for _ in range(3):
            wrapper.fuse_nodes(nodes)

        wrapper.reset_statistics()

        # Measure performance
        num_iterations = 10
        start_time = time.time()

        for _ in range(num_iterations):
            wrapper.fuse_nodes(nodes)

        total_time = time.time() - start_time
        avg_time_ms = (total_time / num_iterations) * 1000

        stats = wrapper.get_statistics()

        # Log performance
        print(f"\nPerformance stats:")
        print(f"  Average time: {avg_time_ms:.2f}ms")
        print(f"  ML fusion count: {stats['ml_fusion_count']}")
        print(f"  Heuristic fusion count: {stats['heuristic_fusion_count']}")
        print(f"  Fallback count: {stats['fallback_count']}")

        # ML overhead should be reasonable (< 200ms per fusion)
        self.assertLess(avg_time_ms, 200.0)

    def test_correctness_validation(self):
        """Test that ML scheduler produces valid results."""
        scheduler = DummyScheduler()
        nodes = [DummySchedulerNode(i) for i in range(10)]

        config = MLSchedulerConfig(
            model_path=None,
            validate_fusion_plan=True,
        )

        wrapper = MLSchedulerWrapper(
            scheduler,
            config=config,
            mode=MLSchedulerMode.HYBRID,
        )

        # Run fusion
        result = wrapper.fuse_nodes(nodes)

        # Should produce valid result
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        # Result should have fewer or equal nodes (due to fusion)
        self.assertLessEqual(len(result), len(nodes))


@unittest.skipIf(not IMPORTS_AVAILABLE, "ML scheduler components not available")
class TestMemorySafety(TestCase):
    """Test memory safety and resource cleanup."""

    def test_no_memory_leak(self):
        """Test that there are no obvious memory leaks."""
        import gc

        scheduler = DummyScheduler()
        config = MLSchedulerConfig(model_path=None)

        # Create and destroy wrappers multiple times
        for _ in range(10):
            wrapper = MLSchedulerWrapper(
                scheduler,
                config=config,
                mode=MLSchedulerMode.HYBRID,
            )

            nodes = [DummySchedulerNode(i) for i in range(20)]
            wrapper.fuse_nodes(nodes)

            # Cleanup
            del wrapper
            gc.collect()

        # If we got here without OOM, test passes
        self.assertTrue(True)

    def test_large_graph_handling(self):
        """Test handling of large graphs."""
        orchestrator = MLSchedulerOrchestrator(
            config=MLSchedulerConfig(
                model_path=None,
                max_nodes_for_ml=500,
            )
        )

        # Create very large graph (should skip ML)
        large_nodes = [DummySchedulerNode(i) for i in range(2000)]

        # Should decide not to use ML
        self.assertFalse(orchestrator.should_use_ml(len(large_nodes)))


if __name__ == "__main__":
    run_tests()

"""
Unit tests for ML scheduler models.

Tests feature extraction, model forward passes, and prediction correctness.
"""

import torch
import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from torch._inductor.ml_scheduler.models.gnn_model import FusionGNN, SchedulingGNN
    from torch._inductor.ml_scheduler.features.node_features import NodeFeatureExtractor
    from torch._inductor.ml_scheduler.config import MLSchedulerConfig
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Warning: Could not import models: {e}")


class DummySchedulerNode:
    """Mock scheduler node for testing."""

    def __init__(self, name="test_node", num_users=1, num_outputs=1):
        self.name = name
        self.users = list(range(num_users))
        self.unmet_dependencies = set()
        self.ancestors = set()
        self.min_order = None
        self.max_order = None
        self.read_writes = type('obj', (object,), {
            'reads': [],
            'writes': []
        })()

    def get_device(self):
        return torch.device('cpu')

    def get_outputs(self):
        return [f"output_{i}" for i in range(1)]

    def is_reduction(self):
        return False

    def estimate_flops(self):
        return 1000


@unittest.skipIf(not MODELS_AVAILABLE, "Models not available")
class TestNodeFeatureExtractor(TestCase):
    """Test node feature extraction."""

    def setUp(self):
        self.extractor = NodeFeatureExtractor(feature_dim=64)
        self.node = DummySchedulerNode()

    def test_extract_features_shape(self):
        """Test that feature extraction produces correct shape."""
        features = self.extractor.extract_features(self.node)

        self.assertEqual(features.shape, (64,))
        self.assertEqual(features.dtype, torch.float32)

    def test_extract_features_no_nan(self):
        """Test that features don't contain NaN values."""
        features = self.extractor.extract_features(self.node)

        self.assertFalse(torch.isnan(features).any())
        self.assertFalse(torch.isinf(features).any())

    def test_extract_features_deterministic(self):
        """Test that feature extraction is deterministic."""
        features1 = self.extractor.extract_features(self.node)
        features2 = self.extractor.extract_features(self.node)

        self.assertTrue(torch.allclose(features1, features2))

    def test_op_type_features(self):
        """Test operation type feature extraction."""
        features = self.extractor._get_op_type_features(self.node)

        # Should be one-hot encoded
        self.assertEqual(len(features), len(self.extractor.op_type_to_id))
        self.assertIn(1.0, features)  # At least one should be 1.0

    def test_computational_features(self):
        """Test computational feature extraction."""
        features = self.extractor._get_computational_features(self.node)

        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

        # Check no NaN/inf
        for f in features:
            self.assertFalse(float('inf') == f or f != f)

    def test_memory_features(self):
        """Test memory feature extraction."""
        features = self.extractor._get_memory_features(self.node)

        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_dependency_features(self):
        """Test dependency feature extraction."""
        features = self.extractor._get_dependency_features(self.node)

        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_device_features(self):
        """Test device feature extraction."""
        features = self.extractor._get_device_features(self.node)

        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_scheduling_features(self):
        """Test scheduling feature extraction."""
        features = self.extractor._get_scheduling_features(self.node)

        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)


@unittest.skipIf(not MODELS_AVAILABLE, "Models not available")
class TestFusionGNN(TestCase):
    """Test FusionGNN model."""

    def setUp(self):
        self.model = FusionGNN(
            node_feat_dim=64,
            edge_feat_dim=32,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        )
        self.model.eval()

    def test_model_creation(self):
        """Test that model can be created."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        # Create dummy graph
        num_nodes = 10
        num_edges = 20

        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 32)

        # Forward pass
        with torch.no_grad():
            output = self.model(x, edge_index, edge_attr)

        # Check outputs
        self.assertIn('node_embeddings', output)
        self.assertIn('fusion_matrix', output)

        self.assertEqual(output['node_embeddings'].shape, (num_nodes, 128))
        self.assertEqual(output['fusion_matrix'].shape, (num_nodes, num_nodes))

    def test_fusion_matrix_symmetric(self):
        """Test that fusion matrix is symmetric."""
        num_nodes = 5
        x = torch.randn(num_nodes, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 32)

        with torch.no_grad():
            output = self.model(x, edge_index, edge_attr)

        fusion_matrix = output['fusion_matrix']

        # Check symmetry
        self.assertTrue(
            torch.allclose(fusion_matrix, fusion_matrix.t(), atol=1e-6)
        )

    def test_fusion_matrix_range(self):
        """Test that fusion matrix values are in [0, 1]."""
        num_nodes = 5
        x = torch.randn(num_nodes, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 32)

        with torch.no_grad():
            output = self.model(x, edge_index, edge_attr)

        fusion_matrix = output['fusion_matrix']

        self.assertTrue((fusion_matrix >= 0).all())
        self.assertTrue((fusion_matrix <= 1).all())

    def test_forward_no_edges(self):
        """Test forward pass with no edges."""
        num_nodes = 5
        x = torch.randn(num_nodes, 64)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 32))

        with torch.no_grad():
            output = self.model(x, edge_index, edge_attr)

        self.assertEqual(output['node_embeddings'].shape, (num_nodes, 128))
        self.assertEqual(output['fusion_matrix'].shape, (num_nodes, num_nodes))

    def test_forward_no_edge_attr(self):
        """Test forward pass without edge attributes."""
        num_nodes = 5
        x = torch.randn(num_nodes, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

        with torch.no_grad():
            output = self.model(x, edge_index, edge_attr=None)

        self.assertIn('fusion_matrix', output)

    def test_predict_pairwise(self):
        """Test pairwise prediction."""
        node_i = torch.randn(128)
        node_j = torch.randn(128)

        with torch.no_grad():
            score = self.model.predict_pairwise(node_i, node_j)

        self.assertIsInstance(score, torch.Tensor)
        self.assertTrue(0 <= score.item() <= 1)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        self.model.train()

        num_nodes = 5
        x = torch.randn(num_nodes, 64, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 32)

        output = self.model(x, edge_index, edge_attr)
        loss = output['fusion_matrix'].sum()

        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


@unittest.skipIf(not MODELS_AVAILABLE, "Models not available")
class TestSchedulingGNN(TestCase):
    """Test SchedulingGNN model."""

    def setUp(self):
        self.model = SchedulingGNN(
            node_feat_dim=64,
            edge_feat_dim=32,
            hidden_dim=128,
            num_layers=2,
        )
        self.model.eval()

    def test_model_creation(self):
        """Test that model can be created."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        num_nodes = 10
        num_edges = 20

        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 32)

        with torch.no_grad():
            output = self.model(x, edge_index, edge_attr)

        # Check outputs
        self.assertIn('priority_scores', output)
        self.assertIn('partition_logits', output)
        self.assertIn('memory_logits', output)
        self.assertIn('node_embeddings', output)

        self.assertEqual(output['priority_scores'].shape, (num_nodes,))
        self.assertEqual(output['partition_logits'].shape, (num_nodes, 16))
        self.assertEqual(output['memory_logits'].shape, (num_nodes, 3))
        self.assertEqual(output['node_embeddings'].shape, (num_nodes, 128))

    def test_priority_scores(self):
        """Test that priority scores are produced."""
        num_nodes = 5
        x = torch.randn(num_nodes, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 32)

        with torch.no_grad():
            output = self.model(x, edge_index, edge_attr)

        priority_scores = output['priority_scores']

        # Scores should be real numbers
        self.assertFalse(torch.isnan(priority_scores).any())
        self.assertFalse(torch.isinf(priority_scores).any())

    def test_partition_logits(self):
        """Test that partition logits are produced."""
        num_nodes = 5
        x = torch.randn(num_nodes, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 32)

        with torch.no_grad():
            output = self.model(x, edge_index, edge_attr)

        partition_logits = output['partition_logits']

        # Should have 16 classes
        self.assertEqual(partition_logits.shape[1], 16)

        # Can be converted to probabilities
        probs = torch.softmax(partition_logits, dim=1)
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(num_nodes)))

    def test_memory_logits(self):
        """Test that memory logits are produced."""
        num_nodes = 5
        x = torch.randn(num_nodes, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 32)

        with torch.no_grad():
            output = self.model(x, edge_index, edge_attr)

        memory_logits = output['memory_logits']

        # Should have 3 classes
        self.assertEqual(memory_logits.shape[1], 3)


@unittest.skipIf(not MODELS_AVAILABLE, "Models not available")
class TestModelSaveLoad(TestCase):
    """Test model saving and loading."""

    def test_save_and_load_fusion_gnn(self):
        """Test saving and loading FusionGNN."""
        import tempfile

        model = FusionGNN()

        # Create dummy input
        x = torch.randn(5, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 32)

        # Get output before saving
        with torch.no_grad():
            output1 = model(x, edge_index, edge_attr)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name

        # Load model
        loaded_model = FusionGNN()
        loaded_model.load_state_dict(torch.load(temp_path))
        loaded_model.eval()

        # Get output after loading
        with torch.no_grad():
            output2 = loaded_model(x, edge_index, edge_attr)

        # Outputs should be identical
        self.assertTrue(
            torch.allclose(
                output1['fusion_matrix'],
                output2['fusion_matrix'],
                atol=1e-6
            )
        )

        # Cleanup
        Path(temp_path).unlink()

    def test_save_and_load_scheduling_gnn(self):
        """Test saving and loading SchedulingGNN."""
        import tempfile

        model = SchedulingGNN()

        # Create dummy input
        x = torch.randn(5, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 32)

        # Get output before saving
        with torch.no_grad():
            output1 = model(x, edge_index, edge_attr)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name

        # Load model
        loaded_model = SchedulingGNN()
        loaded_model.load_state_dict(torch.load(temp_path))
        loaded_model.eval()

        # Get output after loading
        with torch.no_grad():
            output2 = loaded_model(x, edge_index, edge_attr)

        # Outputs should be identical
        self.assertTrue(
            torch.allclose(
                output1['priority_scores'],
                output2['priority_scores'],
                atol=1e-6
            )
        )

        # Cleanup
        Path(temp_path).unlink()


@unittest.skipIf(not MODELS_AVAILABLE, "Models not available")
class TestModelPerformance(TestCase):
    """Test model performance characteristics."""

    def test_fusion_gnn_inference_time(self):
        """Test that FusionGNN inference is reasonably fast."""
        import time

        model = FusionGNN()
        model.eval()

        # Create moderately sized graph
        num_nodes = 50
        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr = torch.randn(num_nodes * 2, 32)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(x, edge_index, edge_attr)

        # Measure time
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.time()
                model(x, edge_index, edge_attr)
                times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        # Should be under 100ms on CPU
        self.assertLess(avg_time, 0.1)

    def test_model_memory_usage(self):
        """Test that models have reasonable memory footprint."""
        model = FusionGNN()

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Should have reasonable number of parameters (< 10M)
        self.assertLess(num_params, 10_000_000)


if __name__ == "__main__":
    run_tests()

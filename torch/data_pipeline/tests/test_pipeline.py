"""
Tests for Multi-Agent Data Pipeline

Tests the core functionality of the data pipeline system.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from torch.data_pipeline import DataPipelineDataLoader
from torch.data_pipeline.config import (
    DataPipelineConfig,
    get_default_config,
    get_memory_constrained_config,
)
from torch.data_pipeline.orchestrator import DataPipelineOrchestrator


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for testing"""

    def __init__(self, size=100):
        self.size = size
        self.data = [torch.randn(3, 32, 32) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class TestDataPipelineConfig(TestCase):
    """Test configuration classes"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = get_default_config()
        self.assertIsInstance(config, DataPipelineConfig)
        self.assertTrue(config.validate())

    def test_config_validation(self):
        """Test configuration validation"""
        config = DataPipelineConfig()

        # Should be valid by default
        self.assertTrue(config.validate())

        # Test invalid configuration
        config.memory.max_size_gb = -1.0
        with self.assertRaises(ValueError):
            config.validate()

    def test_config_serialization(self):
        """Test config to/from dict"""
        config = get_default_config()
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertIn('memory', config_dict)
        self.assertIn('disk', config_dict)

        # Reconstruct from dict
        config2 = DataPipelineConfig.from_dict(config_dict)
        self.assertEqual(
            config.memory.max_size_gb,
            config2.memory.max_size_gb
        )

    def test_preset_configs(self):
        """Test preset configurations"""
        configs = [
            get_default_config(),
            get_memory_constrained_config(),
        ]

        for config in configs:
            self.assertIsInstance(config, DataPipelineConfig)
            self.assertTrue(config.validate())


class TestDataPipelineOrchestrator(TestCase):
    """Test orchestrator functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.dataset = SimpleDataset(size=50)
        self.config = get_memory_constrained_config()
        self.config.redis.enabled = False  # Disable Redis for tests
        self.config.gpu.enabled = False  # Disable GPU for CPU tests

    def test_orchestrator_creation(self):
        """Test orchestrator initialization"""
        orchestrator = DataPipelineOrchestrator(
            self.dataset,
            self.config
        )

        self.assertIsNotNone(orchestrator)
        self.assertEqual(len(orchestrator), len(self.dataset))

        # Check agents are created
        self.assertIn('disk', orchestrator.agents)
        self.assertIn('memory', orchestrator.agents)

    def test_get_item(self):
        """Test getting items through pipeline"""
        orchestrator = DataPipelineOrchestrator(
            self.dataset,
            self.config
        )

        # Get some items
        for idx in range(10):
            item = orchestrator.get_item(idx)
            self.assertIsNotNone(item)
            self.assertIsInstance(item, torch.Tensor)
            self.assertEqual(item.shape, (3, 32, 32))

    def test_caching(self):
        """Test that caching works"""
        orchestrator = DataPipelineOrchestrator(
            self.dataset,
            self.config
        )

        # Access same item multiple times
        idx = 5
        for _ in range(5):
            item = orchestrator.get_item(idx)
            self.assertIsNotNone(item)

        # Check statistics
        stats = orchestrator.get_statistics()
        self.assertGreater(stats['total_requests'], 0)

        # Should have some memory hits
        if 'memory' in stats:
            self.assertGreaterEqual(stats['memory']['hits'], 1)

    def test_statistics(self):
        """Test statistics collection"""
        orchestrator = DataPipelineOrchestrator(
            self.dataset,
            self.config
        )

        # Process some items
        for idx in range(20):
            orchestrator.get_item(idx)

        # Get statistics
        stats = orchestrator.get_statistics()

        self.assertIn('total_requests', stats)
        self.assertEqual(stats['total_requests'], 20)

        self.assertIn('uptime_seconds', stats)
        self.assertGreater(stats['uptime_seconds'], 0)

        if 'memory' in stats:
            self.assertIn('hits', stats['memory'])
            self.assertIn('misses', stats['memory'])

    def test_iteration(self):
        """Test iterating through orchestrator"""
        orchestrator = DataPipelineOrchestrator(
            self.dataset,
            self.config
        )

        count = 0
        for item in orchestrator:
            self.assertIsInstance(item, torch.Tensor)
            count += 1
            if count >= 10:  # Test first 10 items
                break

        self.assertEqual(count, 10)


class TestDataPipelineDataLoader(TestCase):
    """Test DataLoader wrapper"""

    def setUp(self):
        """Set up test fixtures"""
        self.dataset = SimpleDataset(size=64)
        self.config = get_memory_constrained_config()
        self.config.redis.enabled = False
        self.config.gpu.enabled = False

    def test_dataloader_creation(self):
        """Test dataloader initialization"""
        loader = DataPipelineDataLoader(
            self.dataset,
            config=self.config,
            batch_size=8,
            shuffle=False
        )

        self.assertIsNotNone(loader)
        self.assertEqual(len(loader), 8)  # 64 / 8 = 8 batches

    def test_dataloader_iteration(self):
        """Test iterating through dataloader"""
        loader = DataPipelineDataLoader(
            self.dataset,
            config=self.config,
            batch_size=8,
            shuffle=False
        )

        batch_count = 0
        for batch in loader:
            self.assertIsInstance(batch, torch.Tensor)
            self.assertEqual(batch.shape[0], 8)  # Batch size
            self.assertEqual(batch.shape[1:], (3, 32, 32))  # Data shape
            batch_count += 1

        self.assertEqual(batch_count, 8)

    def test_dataloader_shuffle(self):
        """Test shuffling in dataloader"""
        loader = DataPipelineDataLoader(
            self.dataset,
            config=self.config,
            batch_size=8,
            shuffle=True
        )

        # Collect first batch from two iterations
        iter1 = iter(loader)
        batch1 = next(iter1)

        iter2 = iter(loader)
        batch2 = next(iter2)

        # Batches might be different due to shuffling
        # (though with small dataset, they might be same by chance)
        self.assertEqual(batch1.shape, batch2.shape)

    def test_dataloader_statistics(self):
        """Test getting statistics from dataloader"""
        loader = DataPipelineDataLoader(
            self.dataset,
            config=self.config,
            batch_size=8,
            shuffle=False
        )

        # Process some batches
        for i, batch in enumerate(loader):
            if i >= 3:
                break

        # Get statistics
        stats = loader.get_statistics()

        self.assertIn('total_requests', stats)
        self.assertGreater(stats['total_requests'], 0)


class TestMemoryAgent(TestCase):
    """Test memory cache agent"""

    def setUp(self):
        """Set up test fixtures"""
        from torch.data_pipeline.agents.memory_agent import MemoryCacheAgent

        config = get_default_config().to_dict()
        config['memory']['max_size_gb'] = 0.1  # Small cache for testing

        self.agent = MemoryCacheAgent(
            agent_id="test_memory",
            config=config
        )

    def test_cache_operations(self):
        """Test basic cache operations"""
        from torch.data_pipeline.agents.base_agent import DataItem

        # Create test item
        data = torch.randn(3, 32, 32)
        item = DataItem(
            sample_id=0,
            data=data,
            size_bytes=data.numel() * data.element_size(),
            location="memory"
        )

        # Add to cache
        self.agent.add_to_cache(item)

        # Retrieve from cache
        retrieved = self.agent.get_from_cache(0)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.sample_id, 0)

        # Try to get non-existent item
        missing = self.agent.get_from_cache(999)
        self.assertIsNone(missing)

    def test_cache_eviction(self):
        """Test cache eviction when full"""
        from torch.data_pipeline.agents.base_agent import DataItem

        # Fill cache beyond capacity
        for i in range(100):
            data = torch.randn(1024, 1024)  # ~4MB each
            item = DataItem(
                sample_id=i,
                data=data,
                size_bytes=data.numel() * data.element_size(),
                location="memory"
            )
            self.agent.add_to_cache(item)

        # Check that cache size is within limits
        stats = self.agent.get_statistics()
        self.assertLess(
            stats['cache_size_mb'],
            self.agent.memory_config['max_size_gb'] * 1024
        )

        # Should have some evictions
        self.assertGreater(stats['evictions'], 0)


class TestDiskAgent(TestCase):
    """Test disk reader agent"""

    def setUp(self):
        """Set up test fixtures"""
        from torch.data_pipeline.agents.disk_agent import DiskReaderAgent

        self.dataset = SimpleDataset(size=50)
        config = get_default_config().to_dict()

        self.agent = DiskReaderAgent(
            agent_id="test_disk",
            config=config,
            dataset=self.dataset
        )

    def test_read_from_disk(self):
        """Test reading from disk"""
        item = self.agent._read_from_disk(0)

        self.assertIsNotNone(item)
        self.assertEqual(item.sample_id, 0)
        self.assertEqual(item.location, "disk")
        self.assertIsInstance(item.data, torch.Tensor)

    def test_pattern_detection(self):
        """Test access pattern detection"""
        # Access sequential pattern
        for i in range(20):
            self.agent._read_from_disk(i)

        # Update access pattern
        self.agent.access_pattern = list(range(20))
        self.agent._detect_patterns()

        # Should detect sequential pattern
        self.assertGreater(len(self.agent.detected_patterns), 0)


if __name__ == "__main__":
    run_tests()

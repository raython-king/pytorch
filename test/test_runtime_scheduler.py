"""
Tests for Runtime Scheduler

Tests for device management, memory scheduling, stream management,
and transfer optimization components.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

# Import runtime scheduler components
from torch.runtime_scheduler import (
    DeviceManager,
    MemoryScheduler,
    StreamManager,
    TransferOptimizer,
    OperationType,
    StreamPriority,
    TransferType,
    TransferPriority,
    get_device_manager,
    get_memory_scheduler,
    get_stream_manager,
    get_transfer_optimizer,
)
from torch.runtime_scheduler.models import (
    DevicePlacementFeatures,
    DevicePlacementModel,
    MemoryEvictionFeatures,
    MemoryEvictionModel,
    PrefetchFeatures,
    PrefetchModel,
    TransferSchedulingFeatures,
    TransferSchedulingModel,
    get_model_manager,
)


class TestDeviceManager(TestCase):
    """Tests for DeviceManager"""

    def test_singleton(self):
        """Test singleton pattern"""
        dm1 = get_device_manager()
        dm2 = get_device_manager()
        self.assertIs(dm1, dm2)

    def test_device_discovery(self):
        """Test device discovery"""
        dm = get_device_manager()
        self.assertGreater(len(dm.devices), 0)

        # Should have at least CPU
        device_types = [d.device.type for d in dm.devices]
        self.assertIn("cpu", device_types)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_devices(self):
        """Test CUDA device detection"""
        dm = get_device_manager()
        cuda_devices = [d for d in dm.devices if d.device.type == "cuda"]
        self.assertEqual(len(cuda_devices), torch.cuda.device_count())

    def test_device_selection(self):
        """Test device selection"""
        dm = get_device_manager()

        device = dm.select_device(
            op_type="matmul",
            input_sizes=[1024, 1024],
            required_memory=1024 * 1024,
        )

        self.assertIsInstance(device, torch.device)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_p2p_detection(self):
        """Test P2P detection"""
        dm = get_device_manager()

        if torch.cuda.device_count() >= 2:
            # Check if P2P is detected
            self.assertIsInstance(dm.p2p_enabled, dict)

    def test_op_tracking(self):
        """Test operation tracking"""
        dm = get_device_manager()
        device = dm.devices[0].device

        # Record operation
        op_id = 12345
        dm.record_op_start(device, op_id)

        dev_info = dm.get_device_info(device)
        self.assertIn(op_id, dev_info.active_ops)

        # Complete operation
        dm.record_op_end(device, op_id, 0.001)

        self.assertNotIn(op_id, dev_info.active_ops)
        self.assertEqual(dev_info.stats.completed_ops, 1)

    def test_stats(self):
        """Test statistics collection"""
        dm = get_device_manager()
        stats = dm.get_stats()

        self.assertIn("total_devices", stats)
        self.assertIn("devices", stats)
        self.assertGreater(stats["total_devices"], 0)


class TestMemoryScheduler(TestCase):
    """Tests for MemoryScheduler"""

    def test_singleton(self):
        """Test singleton pattern"""
        ms1 = get_memory_scheduler()
        ms2 = get_memory_scheduler()
        self.assertIs(ms1, ms2)

    def test_memory_pools(self):
        """Test memory pool initialization"""
        ms = get_memory_scheduler()
        self.assertGreater(len(ms.pools), 0)

        # Check CPU pool exists
        cpu_device = torch.device("cpu")
        self.assertIn(cpu_device, ms.pools)

    def test_allocation(self):
        """Test memory allocation"""
        ms = get_memory_scheduler()
        device = torch.device("cpu")

        # Allocate memory
        block = ms.allocate(
            size=1024,
            device=device,
            tensor_id=1,
            shape=(32, 32),
            dtype=torch.float32,
        )

        self.assertIsNotNone(block)
        self.assertEqual(block.size, 1024)
        self.assertEqual(block.device, device)
        self.assertEqual(block.tensor_id, 1)

        # Free memory
        success = ms.free(block.block_id)
        self.assertTrue(success)

    def test_eviction(self):
        """Test memory eviction"""
        ms = get_memory_scheduler()
        device = torch.device("cpu")
        pool = ms.pools[device]

        # Allocate some blocks
        blocks = []
        for i in range(5):
            block = ms.allocate(
                size=1024,
                device=device,
                tensor_id=i,
            )
            if block:
                blocks.append(block)

        self.assertGreater(len(blocks), 0)

        # Test eviction policy
        victims = ms.eviction_policy.select_victims(
            pool.blocks,
            required_size=2048,
            device=device,
        )

        self.assertIsInstance(victims, list)

    def test_access_tracking(self):
        """Test access tracking for prefetching"""
        ms = get_memory_scheduler()

        # Record access
        ms.record_access(tensor_id=1, device=torch.device("cpu"))

        # Check prefetch predictions
        predictions = ms.prefetch_scheduler.predict_next_access(tensor_id=1)
        self.assertIsInstance(predictions, list)

    def test_stats(self):
        """Test statistics collection"""
        ms = get_memory_scheduler()
        stats = ms.get_stats()

        self.assertIn("pools", stats)
        self.assertIn("total_blocks", stats)


class TestStreamManager(TestCase):
    """Tests for StreamManager"""

    def test_singleton(self):
        """Test singleton pattern"""
        sm1 = get_stream_manager()
        sm2 = get_stream_manager()
        self.assertIs(sm1, sm2)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_stream_creation(self):
        """Test stream creation"""
        sm = get_stream_manager()

        # Should have created default streams for CUDA devices
        if torch.cuda.device_count() > 0:
            device = torch.device("cuda:0")
            self.assertIn(device, sm.streams)
            self.assertGreater(len(sm.streams[device]), 0)

    def test_operation_registration(self):
        """Test operation registration"""
        sm = get_stream_manager()
        device = torch.device("cpu")

        op_id = sm.register_operation(
            op_type=OperationType.COMPUTE,
            device=device,
            priority=StreamPriority.NORMAL,
        )

        self.assertIsInstance(op_id, int)
        self.assertIn(op_id, sm.operations)

    def test_operation_scheduling(self):
        """Test operation scheduling"""
        sm = get_stream_manager()
        device = torch.device("cpu")

        # Register operation
        op_id = sm.register_operation(
            op_type=OperationType.COMPUTE,
            device=device,
        )

        # Schedule operation
        stream_info = sm.schedule_operation(op_id)

        # CPU operations may not have actual streams
        if stream_info:
            self.assertEqual(stream_info.device, device)

    def test_dependency_tracking(self):
        """Test dependency tracking"""
        sm = get_stream_manager()
        device = torch.device("cpu")

        # Register operations with dependencies
        op1 = sm.register_operation(
            op_type=OperationType.COMPUTE,
            device=device,
        )

        op2 = sm.register_operation(
            op_type=OperationType.COMPUTE,
            device=device,
            dependencies={op1},
        )

        # Check dependency
        deps = sm.synchronizer.get_dependencies(op2)
        self.assertIn(op1, deps)

        # op2 should not be ready until op1 completes
        ready = sm.synchronizer.check_ready(op2, set())
        self.assertFalse(ready)

        # Mark op1 as completed
        sm.complete_operation(op1)

        # Now op2 should be ready
        ready = sm.synchronizer.check_ready(op2, sm.completed_ops)
        self.assertTrue(ready)

    def test_stats(self):
        """Test statistics collection"""
        sm = get_stream_manager()
        stats = sm.get_stats()

        self.assertIn("total_streams", stats)
        self.assertIn("total_operations", stats)
        self.assertIn("completed_operations", stats)


class TestTransferOptimizer(TestCase):
    """Tests for TransferOptimizer"""

    def test_singleton(self):
        """Test singleton pattern"""
        to1 = get_transfer_optimizer()
        to2 = get_transfer_optimizer()
        self.assertIs(to1, to2)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_p2p_detection(self):
        """Test P2P detection"""
        to = get_transfer_optimizer()

        if torch.cuda.device_count() >= 2:
            self.assertIsInstance(to.p2p_enabled, dict)

    def test_transfer_submission(self):
        """Test transfer submission"""
        to = get_transfer_optimizer()

        src_device = torch.device("cpu")
        dst_device = torch.device("cpu")

        transfer_id = to.submit_transfer(
            src_device=src_device,
            dst_device=dst_device,
            size=1024,
            priority=TransferPriority.NORMAL,
        )

        self.assertIsInstance(transfer_id, int)
        self.assertIn(transfer_id, to.transfers)

    def test_pinned_memory(self):
        """Test pinned memory management"""
        to = get_transfer_optimizer()
        pmm = to.pinned_memory

        # Allocate pinned memory
        tensor = pmm.allocate(1024)

        if tensor is not None:
            self.assertEqual(tensor.numel(), 1024)

            # Free pinned memory
            pmm.free(tensor)

    def test_batching(self):
        """Test transfer batching"""
        to = get_transfer_optimizer()
        batcher = to.batcher

        src_device = torch.device("cpu")
        dst_device = torch.device("cpu")

        # Submit transfers for batching
        from torch.runtime_scheduler.transfer_optimizer import TransferRequest

        transfer1 = TransferRequest(
            transfer_id=1,
            transfer_type=TransferType.HOST_TO_DEVICE,
            src_device=src_device,
            dst_device=dst_device,
            size=1024,
            can_batch=True,
        )

        batch_id = batcher.add_transfer(transfer1)

        # May or may not form a batch immediately
        if batch_id is not None:
            batch = batcher.get_batch(batch_id)
            self.assertIsNotNone(batch)

    def test_stats(self):
        """Test statistics collection"""
        to = get_transfer_optimizer()
        stats = to.get_stats()

        self.assertIn("scheduler", stats)
        self.assertIn("pinned_memory", stats)
        self.assertIn("total_transfers", stats)


class TestMLModels(TestCase):
    """Tests for ML Models"""

    def test_device_placement_model(self):
        """Test device placement model"""
        model = DevicePlacementModel(num_devices=2)

        # Create features
        features = [
            DevicePlacementFeatures(
                op_type_id=0,
                input_count=2,
                total_input_size=1024,
                estimated_flops=1000.0,
                is_compute_intensive=True,
                device_id=0,
                device_utilization=0.5,
                device_memory_used=1024.0,
                device_memory_free=2048.0,
                device_queue_length=5,
                device_avg_op_time=0.001,
                op_count_on_device=10,
                avg_runtime_on_device=0.002,
                inputs_on_device=1,
                transfer_cost=0.0001,
            ),
            DevicePlacementFeatures(
                op_type_id=0,
                input_count=2,
                total_input_size=1024,
                estimated_flops=1000.0,
                is_compute_intensive=True,
                device_id=1,
                device_utilization=0.3,
                device_memory_used=512.0,
                device_memory_free=3072.0,
                device_queue_length=2,
                device_avg_op_time=0.0008,
                op_count_on_device=5,
                avg_runtime_on_device=0.0015,
                inputs_on_device=0,
                transfer_cost=0.001,
            ),
        ]

        # Predict
        device_id = model.predict(features)
        self.assertIn(device_id, [0, 1])

        # Record result
        model.record_result(features[0], actual_runtime=0.002)

    def test_memory_eviction_model(self):
        """Test memory eviction model"""
        model = MemoryEvictionModel()

        # Create features
        features = [
            MemoryEvictionFeatures(
                block_size=1024,
                allocated_time=time.time() - 10,
                last_access_time=time.time() - 5,
                access_count=3,
                ref_count=1,
                memory_pressure=0.8,
                fragmentation=0.3,
                avg_access_interval=2.0,
                access_trend=0.0,
            ),
            MemoryEvictionFeatures(
                block_size=2048,
                allocated_time=time.time() - 20,
                last_access_time=time.time() - 15,
                access_count=1,
                ref_count=1,
                memory_pressure=0.8,
                fragmentation=0.3,
                avg_access_interval=5.0,
                access_trend=-0.1,
            ),
        ]

        # Predict eviction scores
        scores = model.predict_eviction_scores(features)
        self.assertEqual(len(scores), 2)
        self.assertIsInstance(scores[0], float)

    def test_prefetch_model(self):
        """Test prefetch model"""
        model = PrefetchModel()

        # Create features
        features = [
            PrefetchFeatures(
                tensor_id=1,
                tensor_size=1024,
                current_location=0,
                access_count=5,
                last_access_time=time.time() - 1,
                avg_access_interval=0.5,
                access_regularity=0.8,
                recent_op_types=[1, 2, 1, 2, 1],
                target_device_load=0.4,
            ),
        ]

        # Predict
        priorities = model.predict_prefetch_priority(features)
        self.assertEqual(len(priorities), 1)
        self.assertEqual(priorities[0][0], 1)  # tensor_id

    def test_transfer_scheduling_model(self):
        """Test transfer scheduling model"""
        model = TransferSchedulingModel()

        # Create features
        features = [
            TransferSchedulingFeatures(
                transfer_size=1024,
                transfer_type=0,
                priority=1,
                src_device_load=0.5,
                dst_device_load=0.3,
                bandwidth_estimate=1e9,
                queue_length=3,
                queue_total_size=4096,
                has_dependencies=False,
                dependency_count=0,
            ),
        ]

        # Predict
        times = model.predict_transfer_times(features)
        self.assertEqual(len(times), 1)
        self.assertIsInstance(times[0], float)

    def test_model_manager(self):
        """Test model manager"""
        mm = get_model_manager()

        self.assertIsNotNone(mm.get_device_placement_model())
        self.assertIsNotNone(mm.get_memory_eviction_model())
        self.assertIsNotNone(mm.get_prefetch_model())
        self.assertIsNotNone(mm.get_transfer_scheduling_model())


class TestIntegration(TestCase):
    """Integration tests"""

    def test_full_workflow(self):
        """Test full workflow with all components"""
        # Get managers
        dm = get_device_manager()
        ms = get_memory_scheduler()
        sm = get_stream_manager()
        to = get_transfer_optimizer()

        # Select device
        device = dm.select_device(
            op_type="matmul",
            input_sizes=[1024, 1024],
            required_memory=1024 * 1024,
        )
        self.assertIsInstance(device, torch.device)

        # Allocate memory
        block = ms.allocate(
            size=1024,
            device=device,
            tensor_id=1,
        )
        self.assertIsNotNone(block)

        # Register and schedule operation
        op_id = sm.register_operation(
            op_type=OperationType.COMPUTE,
            device=device,
        )
        stream_info = sm.schedule_operation(op_id)

        # Start and complete operation
        sm.start_operation(op_id)
        sm.complete_operation(op_id)

        # Free memory
        ms.free(block.block_id)

        # Check stats
        dm_stats = dm.get_stats()
        ms_stats = ms.get_stats()
        sm_stats = sm.get_stats()
        to_stats = to.get_stats()

        self.assertIsInstance(dm_stats, dict)
        self.assertIsInstance(ms_stats, dict)
        self.assertIsInstance(sm_stats, dict)
        self.assertIsInstance(to_stats, dict)


if __name__ == "__main__":
    run_tests()

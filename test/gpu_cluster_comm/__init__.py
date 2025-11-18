"""
Tests for GPU Cluster Communication Optimization

This package contains comprehensive tests for all components of the
GPU cluster communication optimization system.

Test Categories:
- test_topology.py: Topology detection and management
- test_integration.py: PyTorch integration and end-to-end tests
- test_collective_optimizer.py: Collective operation optimization
- test_overlap.py: Computation-communication overlap
- test_ml_models.py: ML algorithm selection models
- test_performance.py: Performance benchmarks

Usage:
    # Run all tests
    pytest test/gpu_cluster_comm/

    # Run specific test file
    pytest test/gpu_cluster_comm/test_integration.py

    # Run specific test
    pytest test/gpu_cluster_comm/test_integration.py::TestTorchDistributedIntegration::test_initialization
"""

__all__ = []

"""
Node feature extraction from BaseSchedulerNode.
"""

import torch
import math
from typing import Optional
import logging

log = logging.getLogger(__name__)


class NodeFeatureExtractor:
    """
    Extract features from BaseSchedulerNode for ML models.
    
    Features include:
    - Operation type
    - Computational cost (FLOPs, arithmetic intensity)
    - Memory access patterns
    - Data dependencies
    - Device properties
    
    Output: 64-dimensional feature vector
    """
    
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        
        # Operation type mapping
        self.op_type_to_id = {
            'SchedulerNode': 0,
            'ExternKernelSchedulerNode': 1,
            'FusedSchedulerNode': 2,
            'NopKernelSchedulerNode': 3,
            'ForeachKernelSchedulerNode': 4,
            'GroupedSchedulerNode': 5,
        }
    
    def extract_features(self, node) -> torch.Tensor:
        """
        Extract comprehensive features from a scheduler node.
        
        Args:
            node: BaseSchedulerNode instance
            
        Returns:
            Feature tensor [feature_dim]
        """
        features = []
        
        try:
            # 1. Operation type (5 dims - one-hot)
            op_type_features = self._get_op_type_features(node)
            features.extend(op_type_features)
            
            # 2. Computational features (10 dims)
            comp_features = self._get_computational_features(node)
            features.extend(comp_features)
            
            # 3. Memory features (15 dims)
            memory_features = self._get_memory_features(node)
            features.extend(memory_features)
            
            # 4. Dependency features (10 dims)
            dep_features = self._get_dependency_features(node)
            features.extend(dep_features)
            
            # 5. Device features (9 dims)
            device_features = self._get_device_features(node)
            features.extend(device_features)
            
            # 6. Scheduling features (15 dims)
            sched_features = self._get_scheduling_features(node)
            features.extend(sched_features)
            
            # Pad or truncate to feature_dim
            features = features[:self.feature_dim]
            while len(features) < self.feature_dim:
                features.append(0.0)
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            log.warning(f"Error extracting features: {e}, returning zeros")
            return torch.zeros(self.feature_dim, dtype=torch.float32)
    
    def _get_op_type_features(self, node) -> list:
        """Extract operation type features (one-hot)."""
        op_type = type(node).__name__
        op_id = self.op_type_to_id.get(op_type, -1)
        
        # One-hot encoding
        one_hot = [0.0] * len(self.op_type_to_id)
        if op_id >= 0:
            one_hot[op_id] = 1.0
        
        return one_hot
    
    def _get_computational_features(self, node) -> list:
        """Extract computational characteristics."""
        features = []
        
        # FLOPs (log scale)
        flops = node.estimate_flops() if hasattr(node, 'estimate_flops') else None
        features.append(math.log1p(flops) if flops else 0.0)
        
        # Is reduction
        is_reduction = float(node.is_reduction()) if hasattr(node, 'is_reduction') else 0.0
        features.append(is_reduction)
        
        # Group information
        if hasattr(node, 'group') and node.group:
            device, group_info = node.group
            if isinstance(group_info, tuple):
                # Pointwise vs reduction
                features.append(1.0 if len(group_info) == 1 else 0.0)  # Is pointwise
                features.append(1.0 if len(group_info) == 2 else 0.0)  # Is reduction
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        # Number of users
        num_users = len(node.users) if hasattr(node, 'users') else 0
        features.append(math.log1p(num_users))
        
        # Number of outputs
        num_outputs = len(node.get_outputs()) if hasattr(node, 'get_outputs') else 0
        features.append(float(num_outputs))
        
        # Placeholder features for arithmetic intensity, parallelism, etc.
        features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _get_memory_features(self, node) -> list:
        """Extract memory access patterns."""
        features = []
        
        # Read/write dependencies
        if hasattr(node, 'read_writes'):
            num_reads = len(node.read_writes.reads)
            num_writes = len(node.read_writes.writes)
            features.append(math.log1p(num_reads))
            features.append(math.log1p(num_writes))
        else:
            features.extend([0.0, 0.0])
        
        # Estimated memory bytes
        # This is a placeholder - actual implementation would compute from dependencies
        features.extend([0.0, 0.0, 0.0])  # read_bytes, write_bytes, temp_bytes
        
        # Memory access patterns
        features.extend([0.0] * 10)  # Placeholder for stride, locality, etc.
        
        return features
    
    def _get_dependency_features(self, node) -> list:
        """Extract dependency information."""
        features = []
        
        # Unmet dependencies
        num_deps = len(node.unmet_dependencies) if hasattr(node, 'unmet_dependencies') else 0
        features.append(math.log1p(num_deps))
        
        # Min/max order (scheduling constraints)
        if hasattr(node, 'min_order') and node.min_order is not None:
            features.append(float(node.min_order))
        else:
            features.append(0.0)
        
        if hasattr(node, 'max_order') and node.max_order is not None:
            features.append(float(node.max_order))
        else:
            features.append(float('inf'))
        
        # Ancestors
        num_ancestors = len(node.ancestors) if hasattr(node, 'ancestors') else 0
        features.append(math.log1p(num_ancestors))
        
        # Placeholder for other dependency features
        features.extend([0.0] * 6)
        
        return features
    
    def _get_device_features(self, node) -> list:
        """Extract device-specific features."""
        features = []
        
        # Get device
        device = node.get_device() if hasattr(node, 'get_device') else None
        
        if device is not None:
            # Device type (one-hot: CPU, CUDA, XPU)
            is_cpu = float(device.type == 'cpu')
            is_cuda = float(device.type == 'cuda')
            is_xpu = float(device.type == 'xpu')
            features.extend([is_cpu, is_cuda, is_xpu])
            
            # Device index
            device_idx = device.index if device.index is not None else 0
            features.append(float(device_idx))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Placeholder for compute capability, bandwidth, etc.
        features.extend([0.0] * 5)
        
        return features
    
    def _get_scheduling_features(self, node) -> list:
        """Extract scheduling-related features."""
        features = []
        
        # Has been fused
        is_fused = float(hasattr(node, 'snodes'))
        features.append(is_fused)
        
        # Number of sub-nodes (if fused)
        if hasattr(node, 'snodes'):
            num_snodes = len(node.snodes)
            features.append(math.log1p(num_snodes))
        else:
            features.append(0.0)
        
        # Is extern kernel
        is_extern = float('ExternKernel' in type(node).__name__)
        features.append(is_extern)
        
        # Placeholder for other scheduling features
        features.extend([0.0] * 12)
        
        return features

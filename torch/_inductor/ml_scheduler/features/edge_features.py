"""
Edge feature extraction from dependency edges.

Extract features from data dependencies (edges) between scheduler nodes,
including MemoryDep, StarDep, and WeakDep types.
"""

import torch
import math
import sympy
from typing import Optional, Union
import logging

from torch._inductor.dependencies import Dep, MemoryDep, StarDep, WeakDep
from torch._inductor.virtualized import V

log = logging.getLogger(__name__)


class EdgeFeatureExtractor:
    """
    Extract features from dependency edges in the IR graph.

    Extracts 32-dimensional feature vectors representing data flow and
    dependencies between scheduler nodes.

    Features include:
    - Dependency type encoding (MemoryDep, StarDep, WeakDep)
    - Data size (bytes transferred)
    - Contiguity flags
    - Access pattern features (complexity, broadcast, indirect access)
    - Loop variable count
    - Stride information

    Output: 32-dimensional feature vector
    """

    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim

    def extract_features(self, dependency: Dep, src_node=None, dst_node=None) -> torch.Tensor:
        """
        Extract comprehensive features from a dependency edge.

        Args:
            dependency: Dep instance (MemoryDep, StarDep, or WeakDep)
            src_node: Source scheduler node (optional, for context)
            dst_node: Destination scheduler node (optional, for context)

        Returns:
            Feature tensor [feature_dim]

        Example:
            >>> extractor = EdgeFeatureExtractor()
            >>> dep = MemoryDep('buf0', sympy.Symbol('i'), (sympy.Symbol('i'),), (128,))
            >>> features = extractor.extract_features(dep)
            >>> assert features.shape == (32,)
        """
        features = []

        try:
            # 1. Dependency type features (4 dims)
            type_features = self._get_dependency_type_features(dependency)
            features.extend(type_features)

            # 2. Data transfer features (8 dims)
            data_features = self._get_data_transfer_features(dependency)
            features.extend(data_features)

            # 3. Access pattern features (10 dims)
            access_features = self._get_access_pattern_features(dependency)
            features.extend(access_features)

            # 4. Loop and stride features (10 dims)
            loop_features = self._get_loop_stride_features(dependency)
            features.extend(loop_features)

            # Ensure we have exactly feature_dim features
            features = features[:self.feature_dim]
            while len(features) < self.feature_dim:
                features.append(0.0)

            return torch.tensor(features, dtype=torch.float32)

        except Exception as e:
            log.warning(f"Error extracting edge features: {e}, returning zeros")
            return torch.zeros(self.feature_dim, dtype=torch.float32)

    def _get_dependency_type_features(self, dep: Dep) -> list:
        """
        Extract dependency type encoding (4 dims).

        Features:
        - is_memory_dep: MemoryDep type
        - is_star_dep: StarDep type (depends on entire buffer)
        - is_weak_dep: WeakDep type (ordering constraint)
        - is_fake_dep: Fake dependency flag (for WeakDep)
        """
        features = []

        # One-hot encoding of dependency types
        features.append(1.0 if isinstance(dep, MemoryDep) else 0.0)
        features.append(1.0 if isinstance(dep, StarDep) else 0.0)
        features.append(1.0 if isinstance(dep, WeakDep) else 0.0)

        # Fake dependency flag (only for WeakDep)
        if isinstance(dep, WeakDep):
            features.append(1.0 if dep.is_fake else 0.0)
        else:
            features.append(0.0)

        return features

    def _get_data_transfer_features(self, dep: Dep) -> list:
        """
        Extract data transfer characteristics (8 dims).

        Features:
        - numbytes: Data size in bytes (log scale)
        - numel: Number of elements (log scale)
        - is_contiguous: Contiguous memory access
        - is_scalar: Scalar access pattern
        - has_unbacked_symbols: Dynamic shape dependency
        - stride1_last_dim: Unit stride for last dimension
        - mode_type: Access mode encoding
        - reserved: For future use
        """
        features = []

        # Data size in bytes (log scale)
        try:
            numbytes = dep.numbytes_hint()
            features.append(math.log1p(numbytes))
        except Exception:
            features.append(0.0)

        # Number of elements (log scale)
        try:
            numel = dep.get_numel()
            if isinstance(numel, sympy.Expr):
                # Try to get a hint value
                if hasattr(V, 'graph') and hasattr(V.graph, 'sizevars'):
                    numel_hint = V.graph.sizevars.size_hint(numel)
                    features.append(math.log1p(numel_hint))
                else:
                    features.append(0.0)
            else:
                features.append(math.log1p(float(numel)))
        except Exception:
            features.append(0.0)

        # Contiguity
        try:
            features.append(1.0 if dep.is_contiguous() else 0.0)
        except Exception:
            features.append(0.0)

        # Scalar access
        if isinstance(dep, MemoryDep):
            try:
                features.append(1.0 if dep.is_scalar() else 0.0)
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)

        # Has unbacked symbols (dynamic shapes)
        try:
            features.append(1.0 if dep.has_unbacked_symbols() else 0.0)
        except Exception:
            features.append(0.0)

        # Unit stride for last dimension (important for coalescing)
        if isinstance(dep, MemoryDep):
            try:
                features.append(1.0 if dep.stride1_for_last_dim() else 0.0)
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)

        # Access mode
        if hasattr(dep, 'mode') and dep.mode is not None:
            # Encode mode as a categorical feature
            # Common modes: None, 'read', 'write', 'update'
            mode_encoding = {
                None: 0.0,
                'read': 0.25,
                'write': 0.5,
                'update': 0.75,
            }
            features.append(mode_encoding.get(dep.mode, 1.0))
        else:
            features.append(0.0)

        # Reserved for future use
        features.append(0.0)

        return features

    def _get_access_pattern_features(self, dep: Dep) -> list:
        """
        Extract memory access pattern features (10 dims).

        Features:
        - index_complexity: Complexity of index expression
        - is_indirect: Indirect memory access (via tmp/indirect buffers)
        - num_index_symbols: Number of free symbols in index
        - is_broadcast: Broadcasting pattern detected
        - index_has_mul: Index contains multiplication
        - index_has_add: Index contains addition
        - index_has_mod: Index contains modulo operation
        - index_depth: Depth of index expression tree
        - has_constant_offset: Index has constant offset
        - offset_value: Constant offset value (log scale)
        """
        features = []

        if isinstance(dep, MemoryDep):
            # Index complexity (number of operations)
            try:
                index_str = str(dep.index)
                complexity = len(index_str)
                features.append(math.log1p(complexity))
            except Exception:
                features.append(0.0)

            # Indirect access
            try:
                features.append(1.0 if dep.is_indirect() else 0.0)
            except Exception:
                features.append(0.0)

            # Number of free symbols in index
            try:
                if hasattr(dep.index, 'free_symbols'):
                    num_symbols = len(dep.index.free_symbols)
                    features.append(float(num_symbols))
                else:
                    features.append(0.0)
            except Exception:
                features.append(0.0)

            # Check for broadcast (when num_vars > num_symbols_in_index)
            try:
                num_vars = len(dep.var_names)
                if hasattr(dep.index, 'free_symbols'):
                    num_symbols = len(dep.index.free_symbols)
                    is_broadcast = num_vars > num_symbols
                    features.append(1.0 if is_broadcast else 0.0)
                else:
                    features.append(0.0)
            except Exception:
                features.append(0.0)

            # Index expression analysis
            try:
                index = dep.index
                # Has multiplication
                has_mul = isinstance(index, sympy.Mul) or any(
                    isinstance(arg, sympy.Mul) for arg in getattr(index, 'args', [])
                )
                features.append(1.0 if has_mul else 0.0)

                # Has addition
                has_add = isinstance(index, sympy.Add)
                features.append(1.0 if has_add else 0.0)

                # Has modulo
                has_mod = isinstance(index, sympy.Mod) or any(
                    isinstance(arg, sympy.Mod) for arg in getattr(index, 'args', []) if hasattr(arg, 'args')
                )
                features.append(1.0 if has_mod else 0.0)
            except Exception:
                features.extend([0.0, 0.0, 0.0])

            # Index expression depth
            try:
                depth = self._compute_expr_depth(dep.index)
                features.append(float(min(depth, 10)))  # Cap at 10
            except Exception:
                features.append(0.0)

            # Constant offset
            try:
                offset = dep.get_offset()
                if isinstance(offset, (int, sympy.Integer)) and offset != 0:
                    features.append(1.0)
                    features.append(math.log1p(abs(float(offset))))
                else:
                    features.extend([0.0, 0.0])
            except Exception:
                features.extend([0.0, 0.0])
        else:
            # Non-MemoryDep types
            features.extend([0.0] * 10)

        return features

    def _get_loop_stride_features(self, dep: Dep) -> list:
        """
        Extract loop and stride information (10 dims).

        Features:
        - num_loop_vars: Number of loop variables
        - max_loop_size: Largest loop dimension (log scale)
        - min_loop_size: Smallest loop dimension (log scale)
        - loop_size_variance: Variance in loop sizes
        - max_stride: Maximum stride value (log scale)
        - min_stride: Minimum stride value (log scale)
        - stride_ratio: Ratio of max to min stride
        - has_unit_stride: Has unit stride access
        - has_large_stride: Has stride > 1000
        - stride_alignment: Alignment of strides (power of 2)
        """
        features = []

        if isinstance(dep, MemoryDep):
            # Number of loop variables
            try:
                num_vars = len(dep.var_names)
                features.append(float(num_vars))
            except Exception:
                features.append(0.0)

            # Loop sizes
            try:
                sizes = dep.size
                if sizes and len(sizes) > 0:
                    # Get size hints
                    size_hints = []
                    for size in sizes:
                        if isinstance(size, (int, sympy.Integer)):
                            size_hints.append(float(size))
                        elif hasattr(V, 'graph') and hasattr(V.graph, 'sizevars'):
                            try:
                                hint = V.graph.sizevars.size_hint(size)
                                size_hints.append(float(hint))
                            except Exception:
                                size_hints.append(1.0)
                        else:
                            size_hints.append(1.0)

                    if size_hints:
                        max_size = max(size_hints)
                        min_size = min(size_hints)
                        features.append(math.log1p(max_size))
                        features.append(math.log1p(min_size))

                        # Variance
                        if len(size_hints) > 1:
                            mean_size = sum(size_hints) / len(size_hints)
                            variance = sum((s - mean_size) ** 2 for s in size_hints) / len(size_hints)
                            features.append(math.log1p(variance))
                        else:
                            features.append(0.0)
                    else:
                        features.extend([0.0, 0.0, 0.0])
                else:
                    features.extend([0.0, 0.0, 0.0])
            except Exception:
                features.extend([0.0, 0.0, 0.0])

            # Stride information
            try:
                if hasattr(V, 'graph') and hasattr(V.graph, 'sizevars') and dep.var_names:
                    strides = V.graph.sizevars.stride_hints(dep.index, dep.var_names)
                    if strides and len(strides) > 0:
                        stride_values = [abs(s) for s in strides]
                        max_stride = max(stride_values)
                        min_stride = min(stride_values)

                        features.append(math.log1p(max_stride))
                        features.append(math.log1p(min_stride))

                        # Stride ratio
                        if min_stride > 0:
                            ratio = max_stride / min_stride
                            features.append(math.log1p(ratio))
                        else:
                            features.append(0.0)

                        # Has unit stride
                        features.append(1.0 if 1 in stride_values else 0.0)

                        # Has large stride
                        features.append(1.0 if max_stride > 1000 else 0.0)

                        # Stride alignment (are strides powers of 2?)
                        aligned_count = sum(1 for s in stride_values if s > 0 and (s & (s - 1)) == 0)
                        alignment_ratio = aligned_count / len(stride_values) if stride_values else 0.0
                        features.append(alignment_ratio)
                    else:
                        features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            except Exception:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            # Non-MemoryDep types
            features.extend([0.0] * 10)

        return features

    def _compute_expr_depth(self, expr) -> int:
        """
        Compute depth of a sympy expression tree.

        Args:
            expr: Sympy expression

        Returns:
            Depth of expression tree
        """
        if not hasattr(expr, 'args') or not expr.args:
            return 0

        return 1 + max((self._compute_expr_depth(arg) for arg in expr.args), default=0)

    def extract_batch_features(self, dependencies: list[Dep]) -> torch.Tensor:
        """
        Extract features for a batch of dependencies.

        Args:
            dependencies: List of Dep instances

        Returns:
            Feature tensor [num_deps, feature_dim]

        Example:
            >>> extractor = EdgeFeatureExtractor()
            >>> deps = [dep1, dep2, dep3]
            >>> features = extractor.extract_batch_features(deps)
            >>> assert features.shape == (3, 32)
        """
        if not dependencies:
            return torch.zeros((0, self.feature_dim), dtype=torch.float32)

        feature_list = []
        for dep in dependencies:
            features = self.extract_features(dep)
            feature_list.append(features)

        return torch.stack(feature_list, dim=0)

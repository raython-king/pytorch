"""
Feature extraction orchestrator for ML-based scheduling.

This module provides a unified interface for extracting features from IR graphs,
including node features, edge features, and graph-level features.
"""

import torch
from typing import List, Dict, Optional, Tuple, Any
import logging
from functools import lru_cache
import hashlib
import pickle

from .node_features import NodeFeatureExtractor
from .edge_features import EdgeFeatureExtractor
from .graph_features import GraphFeatureExtractor

log = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Normalize features for stable training and inference.

    Computes and applies z-score normalization (mean=0, std=1) for each feature
    dimension independently.
    """

    def __init__(self):
        self.node_mean: Optional[torch.Tensor] = None
        self.node_std: Optional[torch.Tensor] = None
        self.edge_mean: Optional[torch.Tensor] = None
        self.edge_std: Optional[torch.Tensor] = None
        self.graph_mean: Optional[torch.Tensor] = None
        self.graph_std: Optional[torch.Tensor] = None
        self.fitted = False

    def fit(
        self,
        node_features: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
    ):
        """
        Compute normalization statistics from training data.

        Args:
            node_features: Node features [num_nodes, node_dim]
            edge_features: Edge features [num_edges, edge_dim] (optional)
            graph_features: Graph features [num_graphs, graph_dim] (optional)

        Example:
            >>> normalizer = FeatureNormalizer()
            >>> normalizer.fit(node_feats, edge_feats, graph_feats)
            >>> normalized = normalizer.normalize_node_features(node_feats)
        """
        # Compute node feature statistics
        if node_features.numel() > 0:
            self.node_mean = node_features.mean(dim=0)
            self.node_std = node_features.std(dim=0) + 1e-6  # Add epsilon for numerical stability
        else:
            self.node_mean = torch.zeros(node_features.shape[1])
            self.node_std = torch.ones(node_features.shape[1])

        # Compute edge feature statistics
        if edge_features is not None and edge_features.numel() > 0:
            self.edge_mean = edge_features.mean(dim=0)
            self.edge_std = edge_features.std(dim=0) + 1e-6
        elif edge_features is not None:
            self.edge_mean = torch.zeros(edge_features.shape[1])
            self.edge_std = torch.ones(edge_features.shape[1])

        # Compute graph feature statistics
        if graph_features is not None and graph_features.numel() > 0:
            self.graph_mean = graph_features.mean(dim=0)
            self.graph_std = graph_features.std(dim=0) + 1e-6
        elif graph_features is not None:
            self.graph_mean = torch.zeros(graph_features.shape[1])
            self.graph_std = torch.ones(graph_features.shape[1])

        self.fitted = True
        log.info("Feature normalization statistics computed successfully")

    def normalize_node_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply normalization to node features."""
        if not self.fitted or self.node_mean is None:
            log.warning("Normalizer not fitted, returning original features")
            return features

        return (features - self.node_mean) / self.node_std

    def normalize_edge_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply normalization to edge features."""
        if not self.fitted or self.edge_mean is None:
            log.warning("Normalizer not fitted for edges, returning original features")
            return features

        return (features - self.edge_mean) / self.edge_std

    def normalize_graph_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply normalization to graph features."""
        if not self.fitted or self.graph_mean is None:
            log.warning("Normalizer not fitted for graphs, returning original features")
            return features

        return (features - self.graph_mean) / self.graph_std

    def denormalize_node_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reverse normalization for node features."""
        if not self.fitted or self.node_mean is None:
            return features

        return features * self.node_std + self.node_mean

    def save_state(self) -> Dict[str, Any]:
        """Save normalization state for later use."""
        return {
            'node_mean': self.node_mean,
            'node_std': self.node_std,
            'edge_mean': self.edge_mean,
            'edge_std': self.edge_std,
            'graph_mean': self.graph_mean,
            'graph_std': self.graph_std,
            'fitted': self.fitted,
        }

    def load_state(self, state: Dict[str, Any]):
        """Load normalization state."""
        self.node_mean = state.get('node_mean')
        self.node_std = state.get('node_std')
        self.edge_mean = state.get('edge_mean')
        self.edge_std = state.get('edge_std')
        self.graph_mean = state.get('graph_mean')
        self.graph_std = state.get('graph_std')
        self.fitted = state.get('fitted', False)


class FeatureCache:
    """
    Cache for expensive feature computations.

    Uses LRU caching with configurable size to avoid recomputing features
    for identical or similar graphs.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[torch.Tensor, ...]] = {}
        self.access_order: List[str] = []

    def _compute_graph_hash(self, nodes: List) -> str:
        """
        Compute a hash representing the graph structure.

        Args:
            nodes: List of scheduler nodes

        Returns:
            Hash string
        """
        # Create a simple hash based on node types and structure
        try:
            node_types = tuple(sorted([type(node).__name__ for node in nodes]))
            num_nodes = len(nodes)
            num_edges = sum(
                len(node.unmet_dependencies) if hasattr(node, 'unmet_dependencies') else 0
                for node in nodes
            )

            # Simple hash of structure
            key_data = (node_types, num_nodes, num_edges)
            key_str = str(key_data)
            hash_obj = hashlib.md5(key_str.encode())
            return hash_obj.hexdigest()
        except Exception as e:
            log.debug(f"Failed to compute graph hash: {e}")
            return ""

    def get(self, graph_hash: str) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Retrieve cached features if available.

        Args:
            graph_hash: Hash of the graph structure

        Returns:
            Cached features tuple or None
        """
        if graph_hash in self.cache:
            # Update access order for LRU
            if graph_hash in self.access_order:
                self.access_order.remove(graph_hash)
            self.access_order.append(graph_hash)
            return self.cache[graph_hash]
        return None

    def put(self, graph_hash: str, features: Tuple[torch.Tensor, ...]):
        """
        Store features in cache.

        Args:
            graph_hash: Hash of the graph structure
            features: Tuple of feature tensors to cache
        """
        # Implement LRU eviction
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            if self.access_order:
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]

        self.cache[graph_hash] = features
        self.access_order.append(graph_hash)

    def clear(self):
        """Clear all cached features."""
        self.cache.clear()
        self.access_order.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
        }


class FeatureExtractor:
    """
    Unified feature extractor orchestrating node, edge, and graph extractors.

    This class provides a single interface for extracting all features from
    IR graphs, with support for batching, caching, and normalization.

    Example:
        >>> extractor = FeatureExtractor(
        ...     node_dim=64,
        ...     edge_dim=32,
        ...     graph_dim=32,
        ...     enable_cache=True,
        ...     normalize=True
        ... )
        >>>
        >>> # Extract features from a graph
        >>> features = extractor.extract_graph_features(nodes)
        >>> node_feats = features['node_features']  # [num_nodes, 64]
        >>> edge_feats = features['edge_features']  # [num_edges, 32]
        >>> graph_feats = features['graph_features']  # [32]
        >>>
        >>> # Batch processing
        >>> batch_features = extractor.extract_batch_features([nodes1, nodes2, nodes3])
    """

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        graph_dim: int = 32,
        enable_cache: bool = True,
        cache_size: int = 1000,
        normalize: bool = False,
    ):
        """
        Initialize feature extractor.

        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            graph_dim: Dimension of graph features
            enable_cache: Whether to cache computed features
            cache_size: Maximum number of graphs to cache
            normalize: Whether to normalize features
        """
        self.node_extractor = NodeFeatureExtractor(feature_dim=node_dim)
        self.edge_extractor = EdgeFeatureExtractor(feature_dim=edge_dim)
        self.graph_extractor = GraphFeatureExtractor(feature_dim=graph_dim)

        self.enable_cache = enable_cache
        self.cache = FeatureCache(max_size=cache_size) if enable_cache else None

        self.normalize = normalize
        self.normalizer = FeatureNormalizer() if normalize else None

        log.info(
            f"FeatureExtractor initialized: node_dim={node_dim}, "
            f"edge_dim={edge_dim}, graph_dim={graph_dim}, "
            f"cache={enable_cache}, normalize={normalize}"
        )

    def extract_graph_features(
        self,
        nodes: List,
        extract_edges: bool = True,
        use_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all features from a graph.

        Args:
            nodes: List of BaseSchedulerNode instances
            extract_edges: Whether to extract edge features
            use_cache: Whether to use cached features if available

        Returns:
            Dictionary containing:
            - 'node_features': Node feature tensor [num_nodes, node_dim]
            - 'edge_features': Edge feature tensor [num_edges, edge_dim] (if extract_edges=True)
            - 'graph_features': Graph feature tensor [graph_dim]
            - 'edge_index': Edge connectivity [2, num_edges] (if extract_edges=True)

        Example:
            >>> features = extractor.extract_graph_features(nodes)
            >>> print(features['node_features'].shape)  # torch.Size([100, 64])
            >>> print(features['graph_features'].shape)  # torch.Size([32])
        """
        # Check cache first
        if use_cache and self.enable_cache and self.cache:
            graph_hash = self.cache._compute_graph_hash(nodes)
            if graph_hash:
                cached = self.cache.get(graph_hash)
                if cached is not None:
                    log.debug("Using cached features")
                    return self._unpack_cached_features(cached, extract_edges)

        # Extract node features
        node_features_list = []
        for node in nodes:
            node_feat = self.node_extractor.extract_features(node)
            node_features_list.append(node_feat)

        if node_features_list:
            node_features = torch.stack(node_features_list, dim=0)
        else:
            node_features = torch.zeros((0, self.node_extractor.feature_dim), dtype=torch.float32)

        # Extract edge features if requested
        edge_features = None
        edge_index = None
        if extract_edges:
            edge_features_list = []
            edge_index_list = []

            # Build edge list from dependencies
            for i, dst_node in enumerate(nodes):
                if hasattr(dst_node, 'unmet_dependencies'):
                    for dep in dst_node.unmet_dependencies:
                        # Find source node index
                        src_idx = self._find_node_by_dep(nodes, dep)
                        if src_idx is not None:
                            # Extract edge features
                            src_node = nodes[src_idx]
                            edge_feat = self.edge_extractor.extract_features(dep, src_node, dst_node)
                            edge_features_list.append(edge_feat)
                            edge_index_list.append([src_idx, i])

            if edge_features_list:
                edge_features = torch.stack(edge_features_list, dim=0)
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
            else:
                edge_features = torch.zeros((0, self.edge_extractor.feature_dim), dtype=torch.float32)
                edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Extract graph-level features
        graph_features = self.graph_extractor.extract_features(nodes)

        # Apply normalization if enabled
        if self.normalize and self.normalizer and self.normalizer.fitted:
            node_features = self.normalizer.normalize_node_features(node_features)
            if edge_features is not None:
                edge_features = self.normalizer.normalize_edge_features(edge_features)
            graph_features = self.normalizer.normalize_graph_features(graph_features)

        # Prepare result
        result = {
            'node_features': node_features,
            'graph_features': graph_features,
        }

        if extract_edges:
            result['edge_features'] = edge_features
            result['edge_index'] = edge_index

        # Cache result
        if use_cache and self.enable_cache and self.cache:
            graph_hash = self.cache._compute_graph_hash(nodes)
            if graph_hash:
                cached_tuple = self._pack_features_for_cache(result, extract_edges)
                self.cache.put(graph_hash, cached_tuple)

        return result

    def extract_batch_features(
        self,
        node_lists: List[List],
        extract_edges: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract features for a batch of graphs.

        Args:
            node_lists: List of node lists (one per graph)
            extract_edges: Whether to extract edge features

        Returns:
            List of feature dictionaries

        Example:
            >>> graphs = [nodes1, nodes2, nodes3]
            >>> batch = extractor.extract_batch_features(graphs)
            >>> for features in batch:
            ...     process_graph(features['node_features'], features['graph_features'])
        """
        results = []
        for nodes in node_lists:
            features = self.extract_graph_features(nodes, extract_edges=extract_edges)
            results.append(features)
        return results

    def fit_normalizer(
        self,
        node_lists: List[List],
        extract_edges: bool = True,
    ):
        """
        Fit feature normalizer on a dataset.

        Args:
            node_lists: List of node lists (training graphs)
            extract_edges: Whether to extract edge features

        Example:
            >>> # Collect training graphs
            >>> training_graphs = [graph1_nodes, graph2_nodes, ...]
            >>> extractor.fit_normalizer(training_graphs)
            >>> # Now features will be automatically normalized
        """
        if not self.normalize or not self.normalizer:
            log.warning("Normalization not enabled, skipping fit")
            return

        log.info(f"Fitting normalizer on {len(node_lists)} graphs...")

        # Extract features without normalization
        old_normalize = self.normalize
        self.normalize = False

        all_node_features = []
        all_edge_features = []
        all_graph_features = []

        for nodes in node_lists:
            features = self.extract_graph_features(nodes, extract_edges=extract_edges, use_cache=False)
            all_node_features.append(features['node_features'])
            all_graph_features.append(features['graph_features'])
            if extract_edges and 'edge_features' in features:
                all_edge_features.append(features['edge_features'])

        # Concatenate all features
        node_features = torch.cat(all_node_features, dim=0) if all_node_features else torch.zeros((0, self.node_extractor.feature_dim))
        graph_features = torch.stack(all_graph_features, dim=0) if all_graph_features else torch.zeros((0, self.graph_extractor.feature_dim))
        edge_features = torch.cat(all_edge_features, dim=0) if all_edge_features else None

        # Fit normalizer
        self.normalizer.fit(node_features, edge_features, graph_features)

        # Restore normalization flag
        self.normalize = old_normalize

        log.info("Normalizer fitted successfully")

    def _find_node_by_dep(self, nodes: List, dep) -> Optional[int]:
        """
        Find node index by dependency name.

        Args:
            nodes: List of nodes
            dep: Dependency object

        Returns:
            Node index or None
        """
        if not hasattr(dep, 'name'):
            return None

        dep_name = dep.name

        for i, node in enumerate(nodes):
            if hasattr(node, 'get_name'):
                try:
                    if node.get_name() == dep_name:
                        return i
                except Exception:
                    pass

        return None

    def _pack_features_for_cache(
        self, features: Dict[str, torch.Tensor], has_edges: bool
    ) -> Tuple[torch.Tensor, ...]:
        """Pack features into a tuple for caching."""
        if has_edges:
            return (
                features['node_features'],
                features['edge_features'],
                features['graph_features'],
                features['edge_index'],
            )
        else:
            return (
                features['node_features'],
                features['graph_features'],
            )

    def _unpack_cached_features(
        self, cached: Tuple[torch.Tensor, ...], has_edges: bool
    ) -> Dict[str, torch.Tensor]:
        """Unpack cached features into a dictionary."""
        if has_edges and len(cached) == 4:
            return {
                'node_features': cached[0],
                'edge_features': cached[1],
                'graph_features': cached[2],
                'edge_index': cached[3],
            }
        elif len(cached) == 2:
            return {
                'node_features': cached[0],
                'graph_features': cached[1],
            }
        else:
            raise ValueError(f"Invalid cached feature tuple length: {len(cached)}")

    def clear_cache(self):
        """Clear the feature cache."""
        if self.cache:
            self.cache.clear()
            log.info("Feature cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {'size': 0, 'max_size': 0}

    def save_normalizer(self, path: str):
        """Save normalizer state to file."""
        if self.normalizer:
            import torch
            state = self.normalizer.save_state()
            torch.save(state, path)
            log.info(f"Normalizer saved to {path}")

    def load_normalizer(self, path: str):
        """Load normalizer state from file."""
        if self.normalizer:
            import torch
            state = torch.load(path)
            self.normalizer.load_state(state)
            log.info(f"Normalizer loaded from {path}")


# Export main classes
__all__ = [
    'FeatureExtractor',
    'FeatureNormalizer',
    'FeatureCache',
    'NodeFeatureExtractor',
    'EdgeFeatureExtractor',
    'GraphFeatureExtractor',
]

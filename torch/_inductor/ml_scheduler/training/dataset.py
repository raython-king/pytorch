"""
Dataset classes for ML scheduler training.

Implements PyTorch Dataset for loading IR graphs from various sources
including pickle files, JSON, and in-memory buffers.
"""

import torch
import pickle
import json
import logging
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from torch.utils.data import Dataset, IterableDataset
import numpy as np

log = logging.getLogger(__name__)


class IRGraphDataset(Dataset):
    """
    PyTorch Dataset for IR graphs collected from scheduler traces.

    Supports:
    - Loading from pickle/JSON files
    - In-memory graph lists
    - Data augmentation (graph perturbations)
    - Multiple labels: fusion decisions, schedule order, performance metrics

    Example:
        dataset = IRGraphDataset(
            data_path='./training_data/graphs',
            augment=True,
            augment_prob=0.3
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        data_list: Optional[List[Dict[str, Any]]] = None,
        augment: bool = False,
        augment_prob: float = 0.3,
        max_nodes: int = 1000,
        transform: Optional[Callable] = None,
        cache_in_memory: bool = True,
    ):
        """
        Initialize IR graph dataset.

        Args:
            data_path: Path to directory containing .pkl or .json files,
                      or path to single file
            data_list: Alternative to data_path - provide graphs directly
            augment: Enable data augmentation
            augment_prob: Probability of applying augmentation
            max_nodes: Maximum nodes per graph (filter out larger graphs)
            transform: Optional transform function
            cache_in_memory: Cache loaded graphs in memory
        """
        super().__init__()

        self.augment = augment
        self.augment_prob = augment_prob
        self.max_nodes = max_nodes
        self.transform = transform
        self.cache_in_memory = cache_in_memory

        # Storage
        self.data_list = []
        self.file_paths = []
        self.cache = {} if cache_in_memory else None

        # Load data
        if data_list is not None:
            self.data_list = data_list
            log.info(f"Loaded {len(self.data_list)} graphs from in-memory list")
        elif data_path is not None:
            self._load_from_path(data_path)
        else:
            log.warning("No data provided to IRGraphDataset")

    def _load_from_path(self, data_path: Union[str, Path]):
        """Load graphs from file path(s)."""
        data_path = Path(data_path)

        if data_path.is_file():
            # Single file
            self.file_paths = [data_path]
        elif data_path.is_dir():
            # Directory - find all .pkl and .json files
            self.file_paths = list(data_path.glob("*.pkl")) + list(data_path.glob("*.json"))
            self.file_paths.sort()
        else:
            raise ValueError(f"Invalid data_path: {data_path}")

        log.info(f"Found {len(self.file_paths)} data files in {data_path}")

        # If caching in memory, load all now
        if self.cache_in_memory:
            for idx, file_path in enumerate(self.file_paths):
                try:
                    graph_data = self._load_single_file(file_path)

                    # Filter by size
                    if graph_data['num_nodes'] <= self.max_nodes:
                        self.data_list.append(graph_data)

                except Exception as e:
                    log.warning(f"Failed to load {file_path}: {e}")

            log.info(f"Cached {len(self.data_list)} graphs in memory")

    def _load_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single graph from file."""
        if file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Convert lists to tensors
                for key in ['x', 'edge_index', 'edge_attr']:
                    if key in data and isinstance(data[key], list):
                        data[key] = torch.tensor(data[key])
                return data
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def __len__(self) -> int:
        """Return dataset size."""
        if self.data_list:
            return len(self.data_list)
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single graph sample.

        Returns:
            Dict with keys:
            - 'x': Node features [num_nodes, node_dim]
            - 'edge_index': Edge connectivity [2, num_edges]
            - 'edge_attr': Edge features [num_edges, edge_dim]
            - 'y_fusion': Fusion labels [num_nodes, num_nodes] (optional)
            - 'y_schedule': Schedule order [num_nodes] (optional)
            - 'y_performance': Performance metric (optional)
            - 'num_nodes': Number of nodes
        """
        # Load from cache or file
        if self.data_list:
            graph_data = self.data_list[idx].copy()
        elif self.cache_in_memory and idx in self.cache:
            graph_data = self.cache[idx].copy()
        else:
            # Load from file
            file_path = self.file_paths[idx]
            graph_data = self._load_single_file(file_path)

            if self.cache_in_memory:
                self.cache[idx] = graph_data.copy()

        # Data augmentation
        if self.augment and random.random() < self.augment_prob:
            graph_data = self._augment_graph(graph_data)

        # Apply transform
        if self.transform is not None:
            graph_data = self.transform(graph_data)

        return graph_data

    def _augment_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply data augmentation to graph.

        Augmentation strategies:
        1. Node feature noise
        2. Edge dropout
        3. Node permutation (preserving topology)
        """
        graph_data = graph_data.copy()

        # 1. Add noise to node features
        if 'x' in graph_data and random.random() < 0.5:
            x = graph_data['x']
            noise = torch.randn_like(x) * 0.1
            graph_data['x'] = x + noise

        # 2. Edge dropout
        if 'edge_index' in graph_data and random.random() < 0.3:
            edge_index = graph_data['edge_index']
            edge_attr = graph_data.get('edge_attr', None)

            # Keep 90% of edges
            num_edges = edge_index.size(1)
            keep_mask = torch.rand(num_edges) > 0.1

            graph_data['edge_index'] = edge_index[:, keep_mask]
            if edge_attr is not None:
                graph_data['edge_attr'] = edge_attr[keep_mask]

        # 3. Node permutation (rarely, for invariance)
        if random.random() < 0.1:
            num_nodes = graph_data['num_nodes']
            perm = torch.randperm(num_nodes)

            # Permute node features
            if 'x' in graph_data:
                graph_data['x'] = graph_data['x'][perm]

            # Update edge indices
            if 'edge_index' in graph_data:
                edge_index = graph_data['edge_index']
                # Create inverse permutation mapping
                inv_perm = torch.empty_like(perm)
                inv_perm[perm] = torch.arange(num_nodes)

                graph_data['edge_index'] = inv_perm[edge_index]

            # Permute labels if present
            if 'y_fusion' in graph_data:
                y_fusion = graph_data['y_fusion']
                graph_data['y_fusion'] = y_fusion[perm][:, perm]

            if 'y_schedule' in graph_data:
                graph_data['y_schedule'] = graph_data['y_schedule'][perm]

        return graph_data


class StreamingIRGraphDataset(IterableDataset):
    """
    Streaming dataset for very large graph collections.

    Loads graphs on-the-fly without caching everything in memory.
    Suitable for training on large-scale data that doesn't fit in RAM.

    Example:
        dataset = StreamingIRGraphDataset(
            data_path='./large_training_data',
            shuffle_buffer_size=1000
        )
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        shuffle_buffer_size: int = 1000,
        augment: bool = False,
        augment_prob: float = 0.3,
        max_nodes: int = 1000,
    ):
        """
        Initialize streaming dataset.

        Args:
            data_path: Path to directory containing graph files
            shuffle_buffer_size: Size of shuffle buffer
            augment: Enable data augmentation
            augment_prob: Probability of applying augmentation
            max_nodes: Maximum nodes per graph
        """
        super().__init__()

        self.data_path = Path(data_path)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.augment = augment
        self.augment_prob = augment_prob
        self.max_nodes = max_nodes

        # Find all data files
        self.file_paths = list(self.data_path.glob("*.pkl")) + list(self.data_path.glob("*.json"))
        self.file_paths.sort()

        log.info(f"StreamingDataset: Found {len(self.file_paths)} files")

    def _load_and_filter(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load and filter a graph."""
        try:
            if file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    graph_data = pickle.load(f)
            else:
                with open(file_path, 'r') as f:
                    graph_data = json.load(f)

            # Filter by size
            if graph_data.get('num_nodes', 0) > self.max_nodes:
                return None

            return graph_data

        except Exception as e:
            log.warning(f"Failed to load {file_path}: {e}")
            return None

    def __iter__(self):
        """Iterate over graphs with shuffling."""
        # Get worker info for multi-process data loading
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process
            file_list = self.file_paths
        else:
            # Multi-process: split files across workers
            per_worker = len(self.file_paths) // worker_info.num_workers
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker if worker_id < worker_info.num_workers - 1 else len(self.file_paths)
            file_list = self.file_paths[start_idx:end_idx]

        # Shuffle file list
        file_list = list(file_list)
        random.shuffle(file_list)

        # Streaming with shuffle buffer
        buffer = []

        for file_path in file_list:
            graph_data = self._load_and_filter(file_path)

            if graph_data is None:
                continue

            # Add to shuffle buffer
            buffer.append(graph_data)

            # When buffer is full, yield random samples
            if len(buffer) >= self.shuffle_buffer_size:
                random.shuffle(buffer)

                while len(buffer) > self.shuffle_buffer_size // 2:
                    graph = buffer.pop(0)

                    # Apply augmentation
                    if self.augment and random.random() < self.augment_prob:
                        graph = self._augment_graph(graph)

                    yield graph

        # Yield remaining samples
        random.shuffle(buffer)
        for graph in buffer:
            if self.augment and random.random() < self.augment_prob:
                graph = self._augment_graph(graph)
            yield graph

    def _augment_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data augmentation (same as IRGraphDataset)."""
        # Reuse augmentation logic
        dataset = IRGraphDataset(data_list=[graph_data], augment=False)
        return dataset._augment_graph(graph_data)


def collate_graphs(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching graphs using torch_geometric conventions.

    Creates a batch by concatenating graphs and maintaining batch indices.

    Args:
        batch: List of graph dictionaries

    Returns:
        Batched graph dictionary with 'batch' indices
    """
    # Collect node features
    x_list = [item['x'] for item in batch]

    # Collect edge indices (need to offset by cumulative node count)
    edge_index_list = []
    edge_attr_list = []
    node_offset = 0

    for item in batch:
        edge_index = item['edge_index']
        edge_index_list.append(edge_index + node_offset)

        if 'edge_attr' in item:
            edge_attr_list.append(item['edge_attr'])

        node_offset += item['num_nodes']

    # Create batch indices
    batch_indices = []
    for batch_idx, item in enumerate(batch):
        batch_indices.extend([batch_idx] * item['num_nodes'])

    # Concatenate
    batched = {
        'x': torch.cat(x_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'num_graphs': len(batch),
    }

    if edge_attr_list:
        batched['edge_attr'] = torch.cat(edge_attr_list, dim=0)

    # Batch labels (if present)
    if 'y_fusion' in batch[0]:
        # Fusion labels - keep as list (different sizes)
        batched['y_fusion'] = [item['y_fusion'] for item in batch]

    if 'y_schedule' in batch[0]:
        batched['y_schedule'] = [item['y_schedule'] for item in batch]

    if 'y_performance' in batch[0]:
        batched['y_performance'] = torch.tensor([item['y_performance'] for item in batch])

    return batched


# Helper functions for data collection

def save_graph_to_file(
    graph_data: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = 'pkl'
):
    """
    Save a graph to file.

    Args:
        graph_data: Graph dictionary
        output_path: Output file path
        format: 'pkl' or 'json'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(graph_data, f)
    elif format == 'json':
        # Convert tensors to lists for JSON
        json_data = {}
        for key, value in graph_data.items():
            if isinstance(value, torch.Tensor):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value

        with open(output_path, 'w') as f:
            json.dump(json_data, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_dataset_split(
    data_path: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Create train/val/test split from a dataset.

    Args:
        data_path: Path to data directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0

    # Load all file paths
    data_path = Path(data_path)
    file_paths = list(data_path.glob("*.pkl")) + list(data_path.glob("*.json"))
    file_paths.sort()

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(file_paths)

    # Split
    n = len(file_paths)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_files = file_paths[:train_end]
    val_files = file_paths[train_end:val_end]
    test_files = file_paths[val_end:]

    # Create datasets
    train_dataset = IRGraphDataset(data_list=[], augment=True)
    train_dataset.file_paths = train_files

    val_dataset = IRGraphDataset(data_list=[], augment=False)
    val_dataset.file_paths = val_files

    test_dataset = IRGraphDataset(data_list=[], augment=False)
    test_dataset.file_paths = test_files

    log.info(f"Created dataset split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    return train_dataset, val_dataset, test_dataset

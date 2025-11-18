"""
数据集工具函数
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


def create_synthetic_dataset(
    num_samples: int = 10000,
    feature_dim: int = 120,
    task: str = 'time_prediction',
    seed: int = 42
) -> 'CommunicationDataset':
    """
    创建合成数据集用于测试和初始训练

    Args:
        num_samples: 样本数量
        feature_dim: 特征维度
        task: 任务类型 ('time_prediction', 'algorithm_selection')
        seed: 随机种子

    Returns:
        dataset: 通讯数据集
    """
    from ..trainer import CommunicationDataset

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = CommunicationDataset()

    for _ in range(num_samples):
        # 生成随机特征
        features = torch.randn(feature_dim)

        # 生成目标
        if task == 'time_prediction':
            # 时间预测：基于特征的简单线性组合 + 噪声
            base_time = torch.abs(features[:10]).sum() * 10
            noise = torch.randn(1) * 5
            target = base_time + noise
        elif task == 'algorithm_selection':
            # 算法选择：基于特征的决策
            algo_score = features[:6].sum()
            target = torch.tensor(int(algo_score.item()) % 6, dtype=torch.long)
        else:
            target = torch.randn(1)

        dataset.add_sample(features, target)

    return dataset


def load_benchmark_dataset(
    dataset_name: str,
    data_dir: str = './datasets'
) -> 'CommunicationDataset':
    """
    加载基准数据集

    Args:
        dataset_name: 数据集名称
        data_dir: 数据目录

    Returns:
        dataset: 数据集
    """
    from ..trainer import CommunicationDataset

    dataset_path = Path(data_dir) / f"{dataset_name}.pt"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")

    dataset = CommunicationDataset()
    dataset.load_from_file(str(dataset_path))

    return dataset


def split_dataset(
    dataset: 'CommunicationDataset',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple['CommunicationDataset', 'CommunicationDataset', 'CommunicationDataset']:
    """
    划分数据集为训练集、验证集和测试集

    Args:
        dataset: 原始数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from ..trainer import CommunicationDataset
    from torch.utils.data import random_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # 设置随机种子
    generator = torch.Generator().manual_seed(seed)

    # 划分数据集
    train_data, val_data, test_data = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # 创建新的数据集对象
    train_dataset = CommunicationDataset()
    val_dataset = CommunicationDataset()
    test_dataset = CommunicationDataset()

    # 复制样本
    for idx in train_data.indices:
        train_dataset.samples.append(dataset.samples[idx])

    for idx in val_data.indices:
        val_dataset.samples.append(dataset.samples[idx])

    for idx in test_data.indices:
        test_dataset.samples.append(dataset.samples[idx])

    return train_dataset, val_dataset, test_dataset


class DatasetStatistics:
    """数据集统计工具"""

    @staticmethod
    def compute_statistics(dataset: 'CommunicationDataset') -> Dict:
        """计算数据集统计信息"""
        if len(dataset) == 0:
            return {}

        # 收集所有特征和目标
        all_features = []
        all_targets = []

        for features, target in dataset:
            all_features.append(features.numpy())
            all_targets.append(target.numpy())

        all_features = np.array(all_features)
        all_targets = np.array(all_targets)

        stats = {
            'num_samples': len(dataset),
            'feature_dim': all_features.shape[1] if len(all_features.shape) > 1 else 1,
            'target_dim': all_targets.shape[1] if len(all_targets.shape) > 1 else 1,
            'feature_mean': all_features.mean(axis=0).tolist(),
            'feature_std': all_features.std(axis=0).tolist(),
            'target_mean': float(all_targets.mean()),
            'target_std': float(all_targets.std()),
            'target_min': float(all_targets.min()),
            'target_max': float(all_targets.max()),
        }

        return stats

    @staticmethod
    def print_statistics(dataset: 'CommunicationDataset'):
        """打印数据集统计信息"""
        stats = DatasetStatistics.compute_statistics(dataset)

        print("Dataset Statistics:")
        print(f"  Number of samples: {stats.get('num_samples', 0)}")
        print(f"  Feature dimension: {stats.get('feature_dim', 0)}")
        print(f"  Target dimension: {stats.get('target_dim', 0)}")
        print(f"  Target mean: {stats.get('target_mean', 0):.4f}")
        print(f"  Target std: {stats.get('target_std', 0):.4f}")
        print(f"  Target range: [{stats.get('target_min', 0):.4f}, {stats.get('target_max', 0):.4f}]")

    @staticmethod
    def save_statistics(dataset: 'CommunicationDataset', path: str):
        """保存统计信息到文件"""
        stats = DatasetStatistics.compute_statistics(dataset)

        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Statistics saved to {path}")

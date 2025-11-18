"""
模型训练器 - ML模型的训练和评估

本模块负责：
1. 训练监督学习模型（时间预测、带宽预测、算法选择）
2. 训练强化学习智能体
3. 数据收集和预处理
4. 模型评估和验证
5. 模型保存和加载
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json
import time
from dataclasses import dataclass
from collections import defaultdict

from .performance_predictor import (
    CommunicationTimePredictor,
    BandwidthPredictor,
    CongestionPredictor,
    PerformanceMetrics
)
from .algorithm_selector import (
    AlgorithmSelectorModel,
    ParameterOptimizer,
    CostModel
)
from .rl_optimizer import CommunicationRLAgent, State, Action
from .features import CommunicationFeatureExtractor


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_dir: str = './checkpoints'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_interval: int = 10
    save_interval: int = 5


class CommunicationDataset(Dataset):
    """通讯数据集"""

    def __init__(self, data_path: Optional[str] = None):
        self.samples = []

        if data_path and Path(data_path).exists():
            self.load_from_file(data_path)

    def add_sample(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        metadata: Optional[Dict] = None
    ):
        """添加样本"""
        self.samples.append({
            'features': features,
            'target': target,
            'metadata': metadata or {}
        })

    def load_from_file(self, path: str):
        """从文件加载数据"""
        data = torch.load(path)
        self.samples = data['samples']

    def save_to_file(self, path: str):
        """保存到文件"""
        torch.save({'samples': self.samples}, path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return sample['features'], sample['target']


class MLTrainer:
    """
    ML模型训练器

    训练所有监督学习模型
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device

        # 创建checkpoint目录
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # 初始化模型
        self.time_predictor = None
        self.bandwidth_predictor = None
        self.congestion_predictor = None
        self.algorithm_selector = None
        self.parameter_optimizer = None
        self.cost_model = None

        # 训练历史
        self.training_history = defaultdict(list)

        # 特征提取器
        self.feature_extractor = CommunicationFeatureExtractor()

    def train_time_predictor(
        self,
        train_dataset: CommunicationDataset,
        val_dataset: Optional[CommunicationDataset] = None
    ) -> Dict[str, Any]:
        """
        训练时间预测模型

        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集

        Returns:
            training_info: 训练信息
        """
        print("Training CommunicationTimePredictor...")

        # 初始化模型
        self.time_predictor = CommunicationTimePredictor().to(self.device)

        # 优化器
        optimizer = torch.optim.Adam(
            self.time_predictor.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # 损失函数
        criterion = nn.MSELoss()

        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # 训练阶段
            train_loss = self._train_epoch(
                self.time_predictor,
                train_loader,
                optimizer,
                criterion
            )

            # 验证阶段
            val_loss = 0.0
            if val_loader:
                val_loss = self._validate_epoch(
                    self.time_predictor,
                    val_loader,
                    criterion
                )
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self.save_model(
                        self.time_predictor,
                        'time_predictor_best.pt'
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            # 日志
            if epoch % self.config.log_interval == 0:
                print(f"Epoch {epoch}/{self.config.num_epochs} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # 保存checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_model(
                    self.time_predictor,
                    f'time_predictor_epoch_{epoch}.pt'
                )

            # 记录历史
            self.training_history['time_predictor_train_loss'].append(train_loss)
            self.training_history['time_predictor_val_loss'].append(val_loss)

        return {
            'best_val_loss': best_val_loss,
            'final_epoch': epoch,
            'training_history': self.training_history
        }

    def train_bandwidth_predictor(
        self,
        train_dataset: CommunicationDataset,
        val_dataset: Optional[CommunicationDataset] = None
    ) -> Dict[str, Any]:
        """训练带宽预测模型"""
        print("Training BandwidthPredictor...")

        self.bandwidth_predictor = BandwidthPredictor().to(self.device)

        optimizer = torch.optim.Adam(
            self.bandwidth_predictor.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        criterion = nn.MSELoss()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch(
                self.bandwidth_predictor,
                train_loader,
                optimizer,
                criterion
            )

            val_loss = 0.0
            if val_loader:
                val_loss = self._validate_epoch(
                    self.bandwidth_predictor,
                    val_loader,
                    criterion
                )
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(
                        self.bandwidth_predictor,
                        'bandwidth_predictor_best.pt'
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break

            if epoch % self.config.log_interval == 0:
                print(f"Epoch {epoch} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

            self.training_history['bandwidth_predictor_train_loss'].append(train_loss)
            self.training_history['bandwidth_predictor_val_loss'].append(val_loss)

        return {'best_val_loss': best_val_loss}

    def train_algorithm_selector(
        self,
        train_dataset: CommunicationDataset,
        val_dataset: Optional[CommunicationDataset] = None
    ) -> Dict[str, Any]:
        """训练算法选择模型"""
        print("Training AlgorithmSelectorModel...")

        self.algorithm_selector = AlgorithmSelectorModel().to(self.device)

        optimizer = torch.optim.Adam(
            self.algorithm_selector.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 使用交叉熵损失（分类任务）
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )

        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # 训练
            train_loss, train_acc = self._train_classifier_epoch(
                self.algorithm_selector,
                train_loader,
                optimizer,
                criterion
            )

            # 验证
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self._validate_classifier_epoch(
                    self.algorithm_selector,
                    val_loader,
                    criterion
                )
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.save_model(
                        self.algorithm_selector,
                        'algorithm_selector_best.pt'
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break

            if epoch % self.config.log_interval == 0:
                print(f"Epoch {epoch} - Train Loss: {train_loss:.6f}, "
                      f"Acc: {train_acc:.4f}, Val Loss: {val_loss:.6f}, "
                      f"Val Acc: {val_acc:.4f}")

            self.training_history['algorithm_selector_train_loss'].append(train_loss)
            self.training_history['algorithm_selector_val_loss'].append(val_loss)
            self.training_history['algorithm_selector_train_acc'].append(train_acc)
            self.training_history['algorithm_selector_val_acc'].append(val_acc)

        return {
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc
        }

    def train_parameter_optimizer(
        self,
        train_dataset: CommunicationDataset,
        val_dataset: Optional[CommunicationDataset] = None
    ) -> Dict[str, Any]:
        """训练参数优化模型"""
        print("Training ParameterOptimizer...")

        self.parameter_optimizer = ParameterOptimizer().to(self.device)

        optimizer = torch.optim.Adam(
            self.parameter_optimizer.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 多任务损失
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )

        best_val_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch(
                self.parameter_optimizer,
                train_loader,
                optimizer,
                criterion
            )

            val_loss = 0.0
            if val_loader:
                val_loss = self._validate_epoch(
                    self.parameter_optimizer,
                    val_loader,
                    criterion
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(
                        self.parameter_optimizer,
                        'parameter_optimizer_best.pt'
                    )

            if epoch % self.config.log_interval == 0:
                print(f"Epoch {epoch} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        return {'best_val_loss': best_val_loss}

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                outputs = model(features)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _train_classifier_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """训练分类器一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device).long()

            optimizer.zero_grad()
            scores, confidence = model(features)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            _, predicted = scores.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / len(train_loader)

        return avg_loss, accuracy

    def _validate_classifier_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """验证分类器一个epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device).long()

                scores, confidence = model(features)
                loss = criterion(scores, targets)

                total_loss += loss.item()

                _, predicted = scores.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / len(val_loader)

        return avg_loss, accuracy

    def save_model(self, model: nn.Module, filename: str):
        """保存模型"""
        path = Path(self.config.checkpoint_dir) / filename
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'class_name': model.__class__.__name__,
            }
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, model: nn.Module, filename: str):
        """加载模型"""
        path = Path(self.config.checkpoint_dir) / filename
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

    def evaluate_models(
        self,
        test_dataset: CommunicationDataset
    ) -> Dict[str, Any]:
        """
        评估所有模型

        Returns:
            evaluation_results: 评估结果
        """
        results = {}

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )

        # 评估时间预测器
        if self.time_predictor:
            metrics = self._evaluate_regression_model(
                self.time_predictor,
                test_loader
            )
            results['time_predictor'] = metrics

        # 评估算法选择器
        if self.algorithm_selector:
            metrics = self._evaluate_classifier_model(
                self.algorithm_selector,
                test_loader
            )
            results['algorithm_selector'] = metrics

        return results

    def _evaluate_regression_model(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """评估回归模型"""
        model.eval()

        predictions = []
        actuals = []

        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                outputs = model(features)

                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(targets.numpy().flatten())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 计算指标
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / np.maximum(actuals, 1e-6))) * 100

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
        }

    def _evaluate_classifier_model(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """评估分类模型"""
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                targets = targets.to(self.device).long()

                scores, _ = model(features)
                _, predicted = scores.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / max(total, 1)

        return {'accuracy': float(accuracy)}


class RLTrainer:
    """
    强化学习训练器

    训练RL智能体
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device

        self.agent = CommunicationRLAgent(device=self.device)

        # 训练历史
        self.episode_rewards = []
        self.episode_lengths = []

    def train_rl_agent(
        self,
        simulator,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 1000
    ) -> Dict[str, Any]:
        """
        训练RL智能体

        Args:
            simulator: 通讯模拟器环境
            num_episodes: 训练episode数
            max_steps_per_episode: 每个episode的最大步数

        Returns:
            training_info: 训练信息
        """
        print(f"Training RL Agent for {num_episodes} episodes...")

        for episode in range(num_episodes):
            total_reward = self.agent.train_episode(
                simulator,
                max_steps=max_steps_per_episode
            )

            self.episode_rewards.append(total_reward)

            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}/{num_episodes} - "
                      f"Reward: {total_reward:.2f}, "
                      f"Avg Reward (100): {avg_reward:.2f}")

            # 保存checkpoint
            if episode % 100 == 0 and episode > 0:
                self.agent.save(
                    Path(self.config.checkpoint_dir) / f'rl_agent_episode_{episode}.pt'
                )

        # 保存最终模型
        self.agent.save(
            Path(self.config.checkpoint_dir) / 'rl_agent_final.pt'
        )

        return {
            'episode_rewards': self.episode_rewards,
            'average_reward': np.mean(self.episode_rewards),
            'final_reward': self.episode_rewards[-1] if self.episode_rewards else 0
        }


class DataCollector:
    """
    数据收集器

    从实际通讯中收集训练数据
    """

    def __init__(self):
        self.collected_data = []
        self.feature_extractor = CommunicationFeatureExtractor()

    def collect_communication_data(
        self,
        message: torch.Tensor,
        topology: Any,
        actual_time: float,
        algorithm_used: int,
        metadata: Optional[Dict] = None
    ):
        """
        收集单次通讯数据

        Args:
            message: 通讯消息
            topology: 拓扑结构
            actual_time: 实际执行时间
            algorithm_used: 使用的算法
            metadata: 其他元数据
        """
        # 提取特征
        msg_features = self.feature_extractor.extract_message_features(message)
        topo_features = self.feature_extractor.extract_topology_features(topology)
        sys_features = self.feature_extractor.extract_system_features()

        # 组合特征
        features = torch.cat([msg_features, topo_features, sys_features])

        # 添加到收集数据
        self.collected_data.append({
            'features': features,
            'actual_time': actual_time,
            'algorithm': algorithm_used,
            'metadata': metadata or {}
        })

    def create_dataset(
        self,
        task: str = 'time_prediction'
    ) -> CommunicationDataset:
        """
        创建数据集

        Args:
            task: 任务类型 ('time_prediction', 'algorithm_selection')

        Returns:
            dataset: 通讯数据集
        """
        dataset = CommunicationDataset()

        for data in self.collected_data:
            features = data['features']

            if task == 'time_prediction':
                target = torch.tensor([data['actual_time']], dtype=torch.float32)
            elif task == 'algorithm_selection':
                target = torch.tensor(data['algorithm'], dtype=torch.long)
            else:
                continue

            dataset.add_sample(features, target, data['metadata'])

        return dataset

    def save_collected_data(self, path: str):
        """保存收集的数据"""
        torch.save({'data': self.collected_data}, path)

    def load_collected_data(self, path: str):
        """加载收集的数据"""
        data = torch.load(path)
        self.collected_data = data['data']

"""
强化学习优化器 - 基于PPO的通讯优化RL Agent

使用Proximal Policy Optimization (PPO)算法训练智能体，
学习最优的通讯策略，包括算法选择、参数配置、压缩、overlap等决策。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MultivariateNormal
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class State:
    """状态空间"""
    topology_state: torch.Tensor  # 48维
    workload_state: torch.Tensor  # 40维
    system_state: torch.Tensor    # 24维
    message_features: torch.Tensor  # 32维
    pending_ops_count: int
    current_congestion: float
    available_bandwidth: float

    def to_tensor(self) -> torch.Tensor:
        """转换为单一张量"""
        return torch.cat([
            self.topology_state,
            self.workload_state,
            self.system_state,
            self.message_features,
            torch.tensor([float(self.pending_ops_count)]),
            torch.tensor([self.current_congestion]),
            torch.tensor([self.available_bandwidth])
        ])


@dataclass
class Action:
    """动作空间"""
    algorithm_choice: int          # 0-5: 选择哪种算法
    chunk_size_ratio: float        # 0-1: chunk大小比例
    pipeline_depth: int            # 1-8: pipeline深度
    enable_compression: bool       # 是否压缩
    compression_ratio: float       # 0-1: 压缩率
    enable_overlap: bool           # 是否overlap
    priority: int                  # 0-10: 优先级

    def to_tensor(self) -> torch.Tensor:
        """转换为张量"""
        return torch.tensor([
            float(self.algorithm_choice),
            self.chunk_size_ratio,
            float(self.pipeline_depth),
            1.0 if self.enable_compression else 0.0,
            self.compression_ratio,
            1.0 if self.enable_overlap else 0.0,
            float(self.priority)
        ], dtype=torch.float32)


class Transition(NamedTuple):
    """经验转换"""
    state: torch.Tensor
    action: torch.Tensor
    action_log_prob: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    value: float


class RolloutBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition: Transition):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """采样批次"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def get_all(self) -> List[Transition]:
        """获取所有经验"""
        return self.buffer

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """
    Actor网络 - 策略网络

    输出各种动作的概率分布
    """

    def __init__(
        self,
        state_dim: int = 147,  # 48+40+24+32+3
        hidden_dim: int = 256,
        num_algorithms: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_algorithms = num_algorithms

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Algorithm selection head (discrete)
        self.algorithm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_algorithms)
        )

        # Continuous action heads
        self.chunk_size_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 0-1
        )
        self.chunk_size_std = nn.Parameter(torch.ones(1) * 0.1)

        self.compression_ratio_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 0-1
        )
        self.compression_ratio_std = nn.Parameter(torch.ones(1) * 0.1)

        # Discrete action heads
        self.pipeline_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8)  # 1-8 levels
        )

        self.compression_enable_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # enable/disable
        )

        self.overlap_enable_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 11)  # 0-10
        )

    def forward(self, state: torch.Tensor) -> Dict[str, Any]:
        """
        前向传播

        Returns:
            distributions: 各种动作的概率分布
        """
        # Shared features
        shared_features = self.shared(state)

        # Algorithm distribution
        algo_logits = self.algorithm_head(shared_features)
        algo_dist = Categorical(logits=algo_logits)

        # Chunk size distribution (continuous)
        chunk_mean = self.chunk_size_mean(shared_features)
        chunk_std = F.softplus(self.chunk_size_std).expand_as(chunk_mean)
        chunk_dist = Normal(chunk_mean, chunk_std)

        # Compression ratio distribution
        comp_mean = self.compression_ratio_mean(shared_features)
        comp_std = F.softplus(self.compression_ratio_std).expand_as(comp_mean)
        comp_ratio_dist = Normal(comp_mean, comp_std)

        # Pipeline depth distribution
        pipeline_logits = self.pipeline_depth_head(shared_features)
        pipeline_dist = Categorical(logits=pipeline_logits)

        # Compression enable distribution
        comp_enable_logits = self.compression_enable_head(shared_features)
        comp_enable_dist = Categorical(logits=comp_enable_logits)

        # Overlap enable distribution
        overlap_logits = self.overlap_enable_head(shared_features)
        overlap_dist = Categorical(logits=overlap_logits)

        # Priority distribution
        priority_logits = self.priority_head(shared_features)
        priority_dist = Categorical(logits=priority_logits)

        return {
            'algorithm': algo_dist,
            'chunk_size': chunk_dist,
            'compression_ratio': comp_ratio_dist,
            'pipeline_depth': pipeline_dist,
            'compression_enable': comp_enable_dist,
            'overlap_enable': overlap_dist,
            'priority': priority_dist,
        }

    def sample_action(self, state: torch.Tensor) -> Tuple[Action, torch.Tensor]:
        """
        采样动作

        Returns:
            action: Action对象
            log_prob: 动作的对数概率
        """
        distributions = self.forward(state)

        # Sample from each distribution
        algo = distributions['algorithm'].sample()
        chunk_size = distributions['chunk_size'].sample().clamp(0, 1)
        comp_ratio = distributions['compression_ratio'].sample().clamp(0, 1)
        pipeline = distributions['pipeline_depth'].sample() + 1  # 1-8
        comp_enable = distributions['compression_enable'].sample()
        overlap = distributions['overlap_enable'].sample()
        priority = distributions['priority'].sample()

        # Compute log probabilities
        log_prob = (
            distributions['algorithm'].log_prob(algo) +
            distributions['chunk_size'].log_prob(chunk_size).sum() +
            distributions['compression_ratio'].log_prob(comp_ratio).sum() +
            distributions['pipeline_depth'].log_prob(pipeline - 1) +
            distributions['compression_enable'].log_prob(comp_enable) +
            distributions['overlap_enable'].log_prob(overlap) +
            distributions['priority'].log_prob(priority)
        )

        action = Action(
            algorithm_choice=algo.item(),
            chunk_size_ratio=chunk_size.item(),
            pipeline_depth=pipeline.item(),
            enable_compression=comp_enable.item() == 1,
            compression_ratio=comp_ratio.item(),
            enable_overlap=overlap.item() == 1,
            priority=priority.item()
        )

        return action, log_prob

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作的对数概率和熵

        Args:
            state: 状态张量
            action: 动作张量 [algo, chunk, pipeline, comp_enable, comp_ratio, overlap, priority]

        Returns:
            log_prob: 对数概率
            entropy: 策略熵
        """
        distributions = self.forward(state)

        # Extract action components
        algo = action[:, 0].long()
        chunk_size = action[:, 1:2]
        comp_ratio = action[:, 4:5]
        pipeline = action[:, 2].long()
        comp_enable = action[:, 3].long()
        overlap = action[:, 5].long()
        priority = action[:, 6].long()

        # Compute log probs
        algo_log_prob = distributions['algorithm'].log_prob(algo)
        chunk_log_prob = distributions['chunk_size'].log_prob(chunk_size).sum(dim=-1)
        comp_ratio_log_prob = distributions['compression_ratio'].log_prob(comp_ratio).sum(dim=-1)
        pipeline_log_prob = distributions['pipeline_depth'].log_prob(pipeline)
        comp_enable_log_prob = distributions['compression_enable'].log_prob(comp_enable)
        overlap_log_prob = distributions['overlap_enable'].log_prob(overlap)
        priority_log_prob = distributions['priority'].log_prob(priority)

        total_log_prob = (
            algo_log_prob + chunk_log_prob + comp_ratio_log_prob +
            pipeline_log_prob + comp_enable_log_prob + overlap_log_prob + priority_log_prob
        )

        # Compute entropy
        entropy = (
            distributions['algorithm'].entropy() +
            distributions['chunk_size'].entropy().sum(dim=-1) +
            distributions['compression_ratio'].entropy().sum(dim=-1) +
            distributions['pipeline_depth'].entropy() +
            distributions['compression_enable'].entropy() +
            distributions['overlap_enable'].entropy() +
            distributions['priority'].entropy()
        )

        return total_log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Critic网络 - 价值网络

    估计状态的价值函数V(s)
    """

    def __init__(
        self,
        state_dim: int = 147,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Returns:
            value: 状态价值
        """
        return self.network(state)


class CommunicationRLAgent:
    """
    基于PPO的通讯优化RL Agent

    学习最优的通讯策略
    """

    def __init__(
        self,
        state_dim: int = 147,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Networks
        self.actor = ActorNetwork(state_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Experience buffer
        self.buffer = RolloutBuffer(capacity=10000)

        # Training statistics
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_reward': [],
        }

    def select_action(
        self,
        state: State,
        deterministic: bool = False
    ) -> Tuple[Action, torch.Tensor]:
        """
        选择动作

        Args:
            state: 当前状态
            deterministic: 是否确定性选择（用于评估）

        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        state_tensor = state.to_tensor().unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            if deterministic:
                # Use mode of distributions
                distributions = self.actor(state_tensor)
                algo = distributions['algorithm'].probs.argmax()
                chunk_size = distributions['chunk_size'].mean
                comp_ratio = distributions['compression_ratio'].mean
                pipeline = distributions['pipeline_depth'].probs.argmax() + 1
                comp_enable = distributions['compression_enable'].probs.argmax()
                overlap = distributions['overlap_enable'].probs.argmax()
                priority = distributions['priority'].probs.argmax()

                action = Action(
                    algorithm_choice=algo.item(),
                    chunk_size_ratio=chunk_size.item(),
                    pipeline_depth=pipeline.item(),
                    enable_compression=comp_enable.item() == 1,
                    compression_ratio=comp_ratio.item(),
                    enable_overlap=overlap.item() == 1,
                    priority=priority.item()
                )
                log_prob = torch.tensor(0.0)
            else:
                action, log_prob = self.actor.sample_action(state_tensor)

        return action, log_prob

    def store_transition(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool
    ):
        """存储经验"""
        state_tensor = state.to_tensor()
        next_state_tensor = next_state.to_tensor()
        action_tensor = action.to_tensor()

        with torch.no_grad():
            value = self.critic(state_tensor.unsqueeze(0).to(self.device)).item()
            _, log_prob = self.actor.sample_action(state_tensor.unsqueeze(0).to(self.device))

        transition = Transition(
            state=state_tensor,
            action=action_tensor,
            action_log_prob=log_prob,
            reward=reward,
            next_state=next_state_tensor,
            done=done,
            value=value
        )

        self.buffer.push(transition)

    def update_policy(
        self,
        num_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        更新策略（PPO算法）

        Returns:
            training_metrics: 训练指标
        """
        if len(self.buffer) < batch_size:
            return {}

        # Get all transitions
        transitions = self.buffer.get_all()

        # Compute advantages
        advantages, returns = self._compute_gae(transitions)

        # Convert to tensors
        states = torch.stack([t.state for t in transitions]).to(self.device)
        actions = torch.stack([t.action for t in transitions]).to(self.device)
        old_log_probs = torch.stack([t.action_log_prob for t in transitions]).to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        actor_losses = []
        critic_losses = []
        entropies = []

        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(transitions))

            for start_idx in range(0, len(transitions), batch_size):
                end_idx = min(start_idx + batch_size, len(transitions))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, entropy = self.actor.evaluate_action(batch_states, batch_actions)
                values = self.critic(batch_states).squeeze()

                # Actor loss (PPO clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = F.mse_loss(values, batch_returns)

                # Total loss
                entropy_loss = -entropy.mean()
                total_loss = (
                    actor_loss +
                    self.value_loss_coef * critic_loss +
                    self.entropy_coef * entropy_loss
                )

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        # Clear buffer
        self.buffer.clear()

        # Update statistics
        metrics = {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
        }

        for key, value in metrics.items():
            self.training_stats[key].append(value)

        return metrics

    def _compute_gae(
        self,
        transitions: List[Transition]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算Generalized Advantage Estimation (GAE)

        Returns:
            advantages: 优势估计
            returns: 回报估计
        """
        advantages = []
        returns = []

        gae = 0
        next_value = 0

        for t in reversed(transitions):
            if t.done:
                next_value = 0
                gae = 0

            delta = t.reward + self.gamma * next_value - t.value
            gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + t.value)

            next_value = t.value

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        return advantages, returns

    def train_episode(self, env, max_steps: int = 1000) -> float:
        """
        训练一个episode

        Args:
            env: 环境
            max_steps: 最大步数

        Returns:
            total_reward: 总回报
        """
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Select action
            action, _ = self.select_action(state, deterministic=False)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store transition
            self.store_transition(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        # Update policy
        if len(self.buffer) >= 64:
            self.update_policy()

        self.training_stats['total_reward'].append(total_reward)

        return total_reward

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']

    def get_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'actor_loss_history': self.training_stats['actor_loss'],
            'critic_loss_history': self.training_stats['critic_loss'],
            'entropy_history': self.training_stats['entropy'],
            'reward_history': self.training_stats['total_reward'],
            'recent_avg_reward': np.mean(self.training_stats['total_reward'][-100:]) if self.training_stats['total_reward'] else 0,
        }


# 便捷函数

def create_rl_agent(device: str = 'cuda') -> CommunicationRLAgent:
    """创建RL智能体"""
    agent = CommunicationRLAgent(device=device)
    return agent


def compute_reward(
    action: Action,
    actual_time: float,
    predicted_time: float,
    bandwidth_utilization: float,
    congestion_penalty: float = 0.0
) -> float:
    """
    计算奖励函数

    Args:
        action: 执行的动作
        actual_time: 实际执行时间
        predicted_time: 预测时间
        bandwidth_utilization: 带宽利用率
        congestion_penalty: 拥塞惩罚

    Returns:
        reward: 奖励值
    """
    # 时间奖励（越短越好）
    time_reward = -actual_time / 1000.0  # 转换为秒

    # 预测准确性奖励
    prediction_error = abs(actual_time - predicted_time) / max(predicted_time, 1)
    accuracy_reward = -prediction_error

    # 带宽利用率奖励
    bandwidth_reward = bandwidth_utilization * 0.5

    # 拥塞惩罚
    congestion_reward = -congestion_penalty * 0.3

    # 总奖励
    reward = time_reward + accuracy_reward * 0.2 + bandwidth_reward + congestion_reward

    return reward

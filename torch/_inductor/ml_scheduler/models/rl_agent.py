"""
Reinforcement Learning agent for scheduler optimization using PPO.

This module implements a PPO-based RL agent that learns to make optimal
scheduling and fusion decisions by interacting with the compilation
environment and receiving rewards based on runtime performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import numpy as np
from dataclasses import dataclass


@dataclass
class RolloutBuffer:
    """
    Buffer for storing rollout experiences.

    Used for PPO training with GAE (Generalized Advantage Estimation).
    """
    states: List[torch.Tensor]
    actions: List[torch.Tensor]
    rewards: List[float]
    values: List[torch.Tensor]
    log_probs: List[torch.Tensor]
    dones: List[bool]

    def clear(self):
        """Clear all stored experiences."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def size(self) -> int:
        """Return number of stored experiences."""
        return len(self.states)

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Tuple of (returns, advantages)
        """
        returns = []
        advantages = []

        gae = 0
        next_value = last_value

        # Compute advantages in reverse order
        for i in reversed(range(len(self.rewards))):
            reward = self.rewards[i]
            value = self.values[i]
            done = self.dones[i]

            if done:
                next_value = 0

            # TD error
            delta = reward + gamma * next_value - value

            # GAE
            gae = delta + gamma * gae_lambda * gae * (1 - done)

            advantages.insert(0, gae)
            returns.insert(0, gae + value)

            next_value = value

        returns = torch.stack(returns)
        advantages = torch.stack(advantages)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages


class StateEncoder(nn.Module):
    """
    Encodes the current scheduling state into a fixed-size representation.

    State includes:
    - Current IR graph representation (via GNN/Transformer)
    - Partial schedule information
    - Available nodes for scheduling
    - Resource usage (memory, compute)
    """

    def __init__(
        self,
        node_feat_dim: int = 64,
        edge_feat_dim: int = 32,
        hidden_dim: int = 256,
        use_gnn: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_gnn = use_gnn

        if use_gnn:
            # Use GNN for state encoding
            try:
                from torch_geometric.nn import GATv2Conv, global_mean_pool

                self.node_encoder = nn.Sequential(
                    nn.Linear(node_feat_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                )

                self.conv_layers = nn.ModuleList([
                    GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=edge_feat_dim)
                    for _ in range(3)
                ])

                self.global_pool = global_mean_pool

            except ImportError:
                raise ImportError("torch_geometric required for GNN state encoder")
        else:
            # Use MLP for state encoding
            self.state_mlp = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Context encoder (encodes partial schedule info, resources, etc.)
        self.context_encoder = nn.Sequential(
            nn.Linear(32, 128),  # 32-dim context features
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode state.

        Args:
            node_features: Node features [num_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            context: Context features [context_dim]
            batch: Batch assignment for graph pooling

        Returns:
            State encoding [hidden_dim]
        """
        if self.use_gnn and edge_index is not None:
            # GNN encoding
            x = self.node_encoder(node_features)

            for conv in self.conv_layers:
                x = F.relu(conv(x, edge_index, edge_attr))

            # Global pooling
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            state_enc = self.global_pool(x, batch)
        else:
            # MLP encoding (average pooling)
            x = self.state_mlp(node_features)
            state_enc = x.mean(dim=0, keepdim=True)

        # Add context encoding
        if context is not None:
            context_enc = self.context_encoder(context.unsqueeze(0))
            state_enc = state_enc + context_enc

        return state_enc.squeeze(0)


class PolicyNetwork(nn.Module):
    """
    Actor network that outputs action probabilities.

    Actions can be:
    - Which node to schedule next
    - Whether to fuse two nodes
    - Memory management decisions
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 256,
        max_actions: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_actions),
        )

    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Categorical:
        """
        Compute action distribution.

        Args:
            state: State encoding [state_dim]
            action_mask: Mask for invalid actions [max_actions]

        Returns:
            Categorical distribution over actions
        """
        logits = self.network(state)

        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        return torch.distributions.Categorical(logits=logits)

    def get_action_logits(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get raw action logits."""
        logits = self.network(state)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits


class ValueNetwork(nn.Module):
    """
    Critic network that estimates state value function.

    Estimates expected cumulative reward from a given state.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value.

        Args:
            state: State encoding [state_dim]

        Returns:
            Value estimate (scalar)
        """
        return self.network(state).squeeze(-1)


class PPOAgent(nn.Module):
    """
    Proximal Policy Optimization (PPO) agent for scheduler optimization.

    PPO is a policy gradient method that:
    - Uses clipped surrogate objective for stable updates
    - Supports multiple epochs of updates per rollout
    - Balances exploration and exploitation

    Components:
    - State encoder: Encodes IR graph and scheduling state
    - Policy network (actor): Outputs action probabilities
    - Value network (critic): Estimates state values

    State: Current IR graph + partial schedule + resources
    Actions: Node selection, fusion decisions, memory planning
    Reward: -runtime - memory_penalty + correctness_bonus
    """

    def __init__(
        self,
        node_feat_dim: int = 64,
        edge_feat_dim: int = 32,
        state_dim: int = 256,
        hidden_dim: int = 256,
        max_actions: int = 128,
        use_gnn_encoder: bool = True,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """
        Initialize PPO agent.

        Args:
            node_feat_dim: Node feature dimension
            edge_feat_dim: Edge feature dimension
            state_dim: State encoding dimension
            hidden_dim: Hidden layer dimension
            max_actions: Maximum number of possible actions
            use_gnn_encoder: Use GNN for state encoding
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        super().__init__()

        self.state_dim = state_dim
        self.max_actions = max_actions
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # State encoder
        self.state_encoder = StateEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=state_dim,
            use_gnn=use_gnn_encoder,
        )

        # Policy network (actor)
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            max_actions=max_actions,
        )

        # Value network (critic)
        self.value = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            states=[],
            actions=[],
            rewards=[],
            values=[],
            log_probs=[],
            dones=[],
        )

    def encode_state(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode current state."""
        return self.state_encoder(
            node_features, edge_index, edge_attr, context, batch
        )

    def select_action(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.

        Args:
            state: State encoding [state_dim]
            action_mask: Mask for invalid actions [max_actions]
            deterministic: If True, select greedy action

        Returns:
            Tuple of (action, log_prob, value)
        """
        # Get action distribution
        action_dist = self.policy(state, action_mask)

        # Sample or select greedy action
        if deterministic:
            action = action_dist.probs.argmax()
        else:
            action = action_dist.sample()

        # Get log probability and value
        log_prob = action_dist.log_prob(action)
        value = self.value(state)

        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and values for given state-action pairs.

        Args:
            states: State encodings [batch_size, state_dim]
            actions: Actions [batch_size]
            action_masks: Action masks [batch_size, max_actions]

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        batch_size = states.size(0)

        log_probs = []
        entropies = []

        for i in range(batch_size):
            state = states[i]
            action = actions[i]
            action_mask = action_masks[i] if action_masks is not None else None

            # Get action distribution
            action_dist = self.policy(state, action_mask)

            # Log probability of taken action
            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)

            # Entropy
            entropy = action_dist.entropy()
            entropies.append(entropy)

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        # Values
        values = self.value(states)

        return log_probs, values, entropies

    def compute_ppo_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss with clipped surrogate objective.

        Args:
            states: State encodings [batch_size, state_dim]
            actions: Actions taken [batch_size]
            old_log_probs: Log probs from old policy [batch_size]
            returns: Computed returns [batch_size]
            advantages: Computed advantages [batch_size]
            action_masks: Action masks [batch_size, max_actions]

        Returns:
            Dictionary with loss components
        """
        # Evaluate actions with current policy
        log_probs, values, entropies = self.evaluate_actions(
            states, actions, action_masks
        )

        # Compute ratio (pi_new / pi_old)
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (clipped)
        value_pred_clipped = values
        value_loss = F.mse_loss(value_pred_clipped, returns)

        # Entropy bonus
        entropy_loss = -entropies.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': -entropy_loss,
            'approx_kl': ((ratio - 1) - torch.log(ratio)).mean(),
        }

    def update(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
    ) -> Dict[str, float]:
        """
        Update policy using collected rollouts.

        Args:
            optimizer: Optimizer for policy and value networks
            num_epochs: Number of epochs to train on rollout buffer
            batch_size: Batch size for updates
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            Dictionary with training metrics
        """
        if self.rollout_buffer.size() == 0:
            return {}

        # Compute returns and advantages
        with torch.no_grad():
            last_value = self.value(self.rollout_buffer.states[-1])
            returns, advantages = self.rollout_buffer.compute_returns_and_advantages(
                last_value
            )

        # Convert to tensors
        states = torch.stack(self.rollout_buffer.states)
        actions = torch.stack(self.rollout_buffer.actions)
        old_log_probs = torch.stack(self.rollout_buffer.log_probs)

        # Training metrics
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
        }

        # Update for multiple epochs
        for epoch in range(num_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(states.size(0))

            for start_idx in range(0, states.size(0), batch_size):
                end_idx = min(start_idx + batch_size, states.size(0))
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Compute loss
                loss_dict = self.compute_ppo_loss(
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_returns,
                    batch_advantages,
                )

                # Backward pass
                optimizer.zero_grad()
                loss_dict['total_loss'].backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_grad_norm
                )

                optimizer.step()

                # Accumulate metrics
                for key in metrics:
                    if key in loss_dict:
                        metrics[key] += loss_dict[key].item()

        # Average metrics
        num_updates = num_epochs * ((states.size(0) + batch_size - 1) // batch_size)
        for key in metrics:
            metrics[key] /= num_updates

        # Clear rollout buffer
        self.rollout_buffer.clear()

        return metrics

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool,
    ):
        """Store a transition in the rollout buffer."""
        self.rollout_buffer.states.append(state.detach())
        self.rollout_buffer.actions.append(action.detach())
        self.rollout_buffer.rewards.append(reward)
        self.rollout_buffer.values.append(value.detach())
        self.rollout_buffer.log_probs.append(log_prob.detach())
        self.rollout_buffer.dones.append(done)


def compute_reward(
    schedule_result: Dict,
    baseline_result: Dict,
    memory_budget: Optional[float] = None,
) -> float:
    """
    Compute reward for a scheduling decision.

    Multi-objective reward combining:
    - Runtime speedup (primary objective)
    - Compilation time overhead (penalty)
    - Memory efficiency
    - Correctness (hard constraint)

    Args:
        schedule_result: Results from ML scheduler
        baseline_result: Results from baseline scheduler
        memory_budget: Memory budget (optional)

    Returns:
        Reward value (higher is better)
    """
    # Runtime speedup (primary reward)
    speedup = baseline_result['runtime'] / max(schedule_result['runtime'], 1e-6)
    runtime_reward = (speedup - 1.0) * 10.0  # Scale to reasonable range

    # Compilation time penalty (should be small)
    compile_ratio = schedule_result['compile_time'] / max(baseline_result['compile_time'], 1e-6)
    compile_penalty = -0.5 * max(0, compile_ratio - 1.0)

    # Memory penalty
    memory_penalty = 0.0
    if memory_budget is not None:
        if schedule_result['peak_memory'] > memory_budget:
            memory_penalty = -5.0 * (schedule_result['peak_memory'] / memory_budget - 1.0)

    # Correctness penalty (large penalty for incorrect results)
    correctness_penalty = 0.0
    if not schedule_result.get('correct', True):
        correctness_penalty = -100.0

    # Total reward
    total_reward = runtime_reward + compile_penalty + memory_penalty + correctness_penalty

    return total_reward

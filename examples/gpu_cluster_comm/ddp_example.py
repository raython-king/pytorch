"""
Example: DistributedDataParallel (DDP) with GPU Cluster Communication Optimization

This example demonstrates how to use GPU cluster communication optimization
with PyTorch's DistributedDataParallel for distributed training.

Usage:
    torchrun --nproc_per_node=4 ddp_example.py
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import GPU cluster communication optimization
try:
    from torch.gpu_cluster_comm.integration import (
        TransparentOptimization,
        IntegrationMode,
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    print("Warning: GPU cluster communication optimization not available")
    OPTIMIZATION_AVAILABLE = False


class SimpleModel(nn.Module):
    """Simple model for demonstration."""

    def __init__(self, input_size=1000, hidden_size=2000, output_size=1000):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def train(rank, world_size, use_optimization=True, optimization_mode="shadow"):
    """Training function.

    Args:
        rank: Process rank
        world_size: Total number of processes
        use_optimization: Whether to use GPU cluster communication optimization
        optimization_mode: Optimization mode (disabled/shadow/enabled)
    """
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)

    # Enable GPU cluster communication optimization
    if use_optimization and OPTIMIZATION_AVAILABLE:
        mode_map = {
            'disabled': IntegrationMode.DISABLED,
            'shadow': IntegrationMode.SHADOW,
            'enabled': IntegrationMode.ENABLED,
        }
        TransparentOptimization.enable_auto_optimization(mode=mode_map[optimization_mode])
        print(f"Rank {rank}: GPU cluster optimization enabled (mode: {optimization_mode})")

    # Create model and move it to GPU with id rank
    model = SimpleModel().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    batch_size = 32

    for epoch in range(num_epochs):
        # Generate dummy data
        inputs = torch.randn(batch_size, 1000).cuda(rank)
        labels = torch.randn(batch_size, 1000).cuda(rank)

        # Forward pass
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()  # Gradient synchronization happens here

        # Optimizer step
        optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Print metrics if optimization was used
    if use_optimization and OPTIMIZATION_AVAILABLE:
        integration = TransparentOptimization.get_instance()
        if integration and rank == 0:
            integration.print_metrics()

    cleanup()

    if rank == 0:
        print("Training completed successfully!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='DDP Example with GPU Cluster Optimization')
    parser.add_argument('--no-optimization', action='store_true',
                       help='Disable GPU cluster communication optimization')
    parser.add_argument('--mode', type=str, default='shadow',
                       choices=['disabled', 'shadow', 'enabled'],
                       help='Optimization mode')

    args = parser.parse_args()

    # Get world size from environment (set by torchrun)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    train(rank, world_size,
          use_optimization=not args.no_optimization,
          optimization_mode=args.mode)


if __name__ == "__main__":
    main()

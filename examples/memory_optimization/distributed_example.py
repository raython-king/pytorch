"""
Distributed Training with Memory Optimization

Demonstrates memory optimization in a distributed training setting.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.memory_optimization import MemoryOptimizationConfig
from torch.memory_optimization.integration import DistributedIntegration


class LargeTransformer(nn.Module):
    """Large transformer model for demonstration"""
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


def setup(rank, world_size):
    """Setup distributed training"""
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='tcp://localhost:12355',
        world_size=world_size,
        rank=rank
    )


def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train(rank, world_size):
    """Training function for each process"""
    print(f"Rank {rank}: Starting training")

    # Setup distributed
    setup(rank, world_size)

    # Create model
    model = LargeTransformer()

    if torch.cuda.is_available():
        model = model.to(rank)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Setup memory optimization for distributed training
    print(f"Rank {rank}: Setting up memory optimization")

    config = MemoryOptimizationConfig()
    config.auto_detect_hardware = True
    config.zero_stage = 2  # Use ZeRO stage 2 for distributed
    config.integrate_with_gpu_cluster_comm = True

    distributed_optimizer = DistributedIntegration(
        config=config,
        world_size=world_size,
        rank=rank
    )

    # Optimize and wrap with DDP
    print(f"Rank {rank}: Optimizing model")
    model = distributed_optimizer.setup_ddp(
        model,
        optimizer=optimizer,
        device_ids=[rank] if torch.cuda.is_available() else None
    )

    # Training loop
    print(f"Rank {rank}: Starting training loop")
    model.train()

    for epoch in range(3):
        for batch_idx in range(10):
            # Create dummy batch
            if torch.cuda.is_available():
                data = torch.randint(0, 10000, (32, 128)).to(rank)
                target = torch.randint(0, 10000, (32, 128)).to(rank)
            else:
                data = torch.randint(0, 10000, (32, 128))
                target = torch.randint(0, 10000, (32, 128))

            # Forward pass
            output = model(data)

            # Reshape for loss
            output = output.view(-1, 10000)
            target = target.view(-1)

            loss = nn.functional.cross_entropy(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0 and batch_idx % 5 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Get summary (only on rank 0)
    if rank == 0:
        summary = distributed_optimizer.orchestrator.get_summary()
        print(f"\nOptimization Summary:")
        print(f"  Active strategies: {summary['active_strategies']}")
        print(f"  Memory saved: {summary['total_memory_saved_gb']:.2f} GB")

    cleanup()
    print(f"Rank {rank}: Training complete")


def main():
    """Main function"""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2

    print("=" * 80)
    print(f"Distributed Memory Optimization Example")
    print(f"World size: {world_size}")
    print("=" * 80)

    if world_size > 1:
        mp.spawn(
            train,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    else:
        print("Note: Running with single process (no distributed training)")
        train(0, 1)


if __name__ == "__main__":
    main()

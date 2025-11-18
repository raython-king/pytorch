"""
Basic Memory Optimization Example

Demonstrates how to use the Multi-Agent Memory Optimization system
to automatically optimize a PyTorch model for memory efficiency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torch.memory_optimization import MemoryOptimizer


class SimpleConvNet(nn.Module):
    """Simple CNN for demonstration"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    print("=" * 80)
    print("Multi-Agent Memory Optimization - Basic Example")
    print("=" * 80)

    # Create model
    print("\n1. Creating model...")
    model = SimpleConvNet(num_classes=10)

    if torch.cuda.is_available():
        model = model.cuda()
        print(f"   Model moved to GPU")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize Memory Optimizer with auto-detection
    print("\n2. Initializing Memory Optimizer...")
    memory_optimizer = MemoryOptimizer(auto_detect=True)

    # Optimize the model
    print("\n3. Optimizing model...")
    model = memory_optimizer.optimize_model(model, optimizer)
    print(f"   Model optimized!")

    # Show applied strategies
    summary = memory_optimizer.summary()
    print(f"\n4. Active optimization strategies:")
    for strategy in summary['active_strategies']:
        print(f"   - {strategy}")

    print(f"\n5. Memory saved: {summary['total_memory_saved_gb']:.2f} GB")

    # Create dummy data
    print("\n6. Creating dummy dataset...")
    batch_size = 32
    num_batches = 10

    dummy_data = torch.randn(batch_size * num_batches, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (batch_size * num_batches,))

    if torch.cuda.is_available():
        dummy_data = dummy_data.cuda()
        dummy_labels = dummy_labels.cuda()

    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop with memory optimization
    print("\n7. Training with memory optimization...")
    model.train()

    for epoch in range(3):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            # Use optimized training step
            with memory_optimizer.optimize_step():
                # Forward pass
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Adapt based on metrics
            if batch_idx % 5 == 0:
                memory_optimizer.adapt({
                    'iteration_time': 0.1,
                    'throughput': batch_size / 0.1,
                    'memory_usage': 0.75,
                })

        avg_loss = total_loss / len(dataloader)
        print(f"   Epoch {epoch + 1}/3 - Loss: {avg_loss:.4f}")

    # Final summary
    print("\n8. Final optimization summary:")
    final_summary = memory_optimizer.summary()

    print(f"   Total iterations: {final_summary['iterations']}")
    print(f"   Active strategies: {len(final_summary['active_strategies'])}")
    print(f"   Memory saved: {final_summary['total_memory_saved_gb']:.2f} GB")

    if 'agent_weights' in final_summary:
        print(f"\n9. Agent weights (learned):")
        for agent, weight in final_summary['agent_weights'].items():
            print(f"   {agent}: {weight:.3f}")

    if 'strategy_rankings' in final_summary:
        print(f"\n10. Strategy rankings:")
        for strategy, score in final_summary['strategy_rankings'][:5]:
            print(f"   {strategy}: {score:.3f}")

    print("\n" + "=" * 80)
    print("Optimization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

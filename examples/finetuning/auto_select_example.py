"""
Auto-Selection Fine-tuning Example

Demonstrates automatic fine-tuning method selection using Multi-Agent system.
"""

import torch
import torch.nn as nn
from torch.finetuning import FineTuner, FineTuningConfig


class LargeModel(nn.Module):
    """Large model for demonstration"""
    def __init__(self, hidden_size=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(20)
        ])
        self.head = nn.Linear(hidden_size, 1000)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)


def main():
    print("=" * 80)
    print("Multi-Agent Auto-Selection Example")
    print("=" * 80)

    # Create large model
    print("\n1. Creating large model...")
    model = LargeModel(hidden_size=1024)

    total_params = sum(p.numel() for p in model.parameters())
    model_size_gb = total_params * 4 / 1024**3  # Assume float32

    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {model_size_gb:.2f} GB")

    # Auto-detect GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU memory: {gpu_memory:.2f} GB")
        model = model.cuda()
    else:
        gpu_memory = 0
        print("   No GPU detected, using CPU")

    # Create config with preferences
    print("\n2. Creating configuration with preferences...")
    config = FineTuningConfig()

    # Set preferences
    config.prefer_memory_efficiency = 0.8  # High priority on memory
    config.prefer_training_speed = 0.5
    config.prefer_accuracy = 0.7

    config.auto_select_method = True

    print(f"   Memory efficiency preference: {config.prefer_memory_efficiency}")
    print(f"   Training speed preference: {config.prefer_training_speed}")
    print(f"   Accuracy preference: {config.prefer_accuracy}")

    # Initialize FineTuner with auto-detection
    print("\n3. Initializing Multi-Agent FineTuner...")
    finetuner = FineTuner(
        auto_detect=True,
        config=config
    )

    # Let agents select the best method
    print("\n4. Multi-Agent method selection...")
    print("   Agents analyzing hardware and model...")

    model = finetuner.prepare_model(model)

    # Show selection result
    summary = finetuner.summary()
    print(f"\n5. Selected method: {summary['applied_method']}")

    # Show statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n6. Fine-tuning statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable ratio: {trainable_params/total_params:.2%}")
    print(f"   Memory saved: ~{(1 - trainable_params/total_params) * 100:.1f}%")

    # Test forward pass
    print("\n7. Testing forward pass...")
    with torch.no_grad():
        if torch.cuda.is_available():
            x = torch.randn(2, 1024).cuda()
        else:
            x = torch.randn(2, 1024)

        y = model(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {y.shape}")
        print("   Forward pass successful!")

    print("\n" + "=" * 80)
    print("Auto-selection complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

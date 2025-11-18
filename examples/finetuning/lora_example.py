"""
LoRA Fine-tuning Example

Demonstrates how to use LoRA for parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torch.finetuning import FineTuner


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration"""
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model"""
    def __init__(self, vocab_size=1000, hidden_size=768, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        return self.head(x.mean(dim=1))


def main():
    print("=" * 80)
    print("LoRA Fine-tuning Example")
    print("=" * 80)

    # Create model
    print("\n1. Creating model...")
    model = SimpleTransformer(vocab_size=1000, hidden_size=512, num_layers=4)

    if torch.cuda.is_available():
        model = model.cuda()
        print("   Model moved to GPU")

    # Show model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Initialize FineTuner with LoRA
    print("\n2. Initializing LoRA fine-tuner...")
    finetuner = FineTuner(
        method='lora',
        auto_detect=False,  # Use manual configuration
        r=8,                # LoRA rank
        alpha=16,          # LoRA alpha
    )

    # Prepare model for fine-tuning
    print("\n3. Applying LoRA to model...")
    model = finetuner.prepare_model(model)

    # Show trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n4. Trainable parameters after LoRA: {trainable_params:,}")
    print(f"   Trainable ratio: {trainable_params/total_params:.2%}")

    # Create dummy dataset
    print("\n5. Creating dataset...")
    num_samples = 1000
    seq_len = 32

    dummy_inputs = torch.randint(0, 1000, (num_samples, seq_len))
    dummy_labels = torch.randint(0, 1000, (num_samples,))

    if torch.cuda.is_available():
        dummy_inputs = dummy_inputs.cuda()
        dummy_labels = dummy_labels.cuda()

    dataset = TensorDataset(dummy_inputs, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create optimizer (only for LoRA parameters)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    # Training loop
    print("\n6. Training with LoRA...")
    model.train()

    for epoch in range(3):
        total_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"   Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"\n   Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    # Save LoRA weights
    print("\n7. Saving LoRA weights...")
    finetuner.save("./lora_weights.pth")
    print("   LoRA weights saved to ./lora_weights.pth")

    # Summary
    print("\n8. Fine-tuning summary:")
    summary = finetuner.summary()
    print(f"   Method: {summary['applied_method']}")
    print(f"   Iterations: {summary['iterations']}")

    print("\n" + "=" * 80)
    print("Fine-tuning complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

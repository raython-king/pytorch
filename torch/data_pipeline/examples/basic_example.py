"""
Basic Example: Using Multi-Agent Data Pipeline

This example demonstrates basic usage of the data pipeline system
with a simple dataset.
"""

import torch
import torch.nn as nn
from torch.data_pipeline import DataPipelineDataLoader
from torch.data_pipeline.config import get_default_config


# Simple dataset for demonstration
class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset that returns random tensors"""

    def __init__(self, size=1000, shape=(3, 224, 224)):
        self.size = size
        self.shape = shape
        # Pre-generate data to simulate disk-based dataset
        self.data = [torch.randn(shape) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], idx % 10  # Return data and label


def main():
    print("=" * 60)
    print("Multi-Agent Data Pipeline - Basic Example")
    print("=" * 60)

    # Create dataset
    print("\n1. Creating dataset...")
    dataset = SimpleDataset(size=1000, shape=(3, 32, 32))
    print(f"   Dataset size: {len(dataset)} samples")

    # Create configuration
    print("\n2. Creating pipeline configuration...")
    config = get_default_config()

    # Adjust for demonstration
    config.memory.max_size_gb = 1.0  # 1GB memory cache
    config.gpu.prefetch_queue_size = 2
    config.prefetch.prefetch_factor = 2

    print(f"   Memory cache: {config.memory.max_size_gb} GB")
    print(f"   GPU prefetch: {config.gpu.prefetch_queue_size}")

    # Create dataloader with pipeline
    print("\n3. Creating DataPipelineDataLoader...")
    loader = DataPipelineDataLoader(
        dataset=dataset,
        config=config,
        batch_size=32,
        shuffle=True
    )
    print(f"   Batch size: 32")
    print(f"   Number of batches: {len(loader)}")

    # Simulate training loop
    print("\n4. Running training simulation...")
    print("   " + "-" * 50)

    num_epochs = 3

    for epoch in range(num_epochs):
        print(f"\n   Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (data, target) in enumerate(loader):
            # Simulate forward/backward pass
            if batch_idx == 0:
                print(f"   - Batch shape: {data.shape}")
                print(f"   - Device: {data.device}")

            # Show progress
            if batch_idx % 10 == 0:
                print(f"   - Processing batch {batch_idx}/{len(loader)}", end='\r')

        print(f"   - Completed {len(loader)} batches")

        # Print statistics after each epoch
        stats = loader.get_statistics()

        print(f"\n   Statistics:")
        print(f"   - Total requests: {stats['total_requests']}")

        if 'memory' in stats:
            mem_stats = stats['memory']
            print(f"   - Memory hit rate: {mem_stats['hit_rate']:.2%}")
            print(f"   - Memory cache size: {mem_stats['cache_size_mb']:.1f} MB")

        if 'gpu' in stats:
            gpu_stats = stats['gpu']
            print(f"   - GPU transfers: {gpu_stats['total_transfers']}")
            print(f"   - GPU prefetch hit rate: {gpu_stats['prefetch_hit_rate']:.2%}")

        if 'latency_ms' in stats:
            lat_stats = stats['latency_ms']
            print(f"   - Mean latency: {lat_stats['mean']:.2f} ms")
            print(f"   - Median latency: {lat_stats['median']:.2f} ms")

        if 'throughput_samples_per_sec' in stats:
            print(f"   - Throughput: {stats['throughput_samples_per_sec']:.1f} samples/s")

    print("\n" + "   " + "-" * 50)

    # Final statistics
    print("\n5. Final Performance Summary")
    print("   " + "=" * 50)
    final_stats = loader.get_statistics()

    print(f"\n   Overall Performance:")
    print(f"   - Total samples processed: {final_stats['total_requests']}")
    print(f"   - Total time: {final_stats['uptime_seconds']:.2f}s")
    print(f"   - Average throughput: {final_stats['throughput_samples_per_sec']:.1f} samples/s")

    if 'overall_cache_hit_rates' in final_stats:
        print(f"\n   Cache Hit Rates:")
        for layer, rate in final_stats['overall_cache_hit_rates'].items():
            print(f"   - {layer.capitalize()}: {rate:.2%}")

    # Cleanup
    print("\n6. Cleaning up...")
    loader.shutdown()
    print("   Done!")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

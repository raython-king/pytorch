"""
Simplest possible example of using GPU cluster communication optimization.

Usage:
    torchrun --nproc_per_node=2 simple_example.py
"""

import torch
import torch.distributed as dist

# Import optimization
from torch.gpu_cluster_comm import enable_optimization

def main():
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Enable optimization - one line of code!
    enable_optimization(mode='enabled')

    # Create a tensor
    tensor = torch.ones(1000, 1000).cuda(rank)

    # Perform AllReduce - automatically optimized!
    dist.all_reduce(tensor)

    if rank == 0:
        print(f"✓ AllReduce completed successfully!")
        print(f"  Result sum: {tensor[0, 0].item()}")
        print(f"  Expected: {world_size}")

    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
